use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::io::{BufRead, Lines, Write};

use failure::Error;
use itertools::Itertools;
use petgraph::prelude::{Direction, EdgeRef, NodeIndex, StableGraph};
use petgraph::visit::{VisitMap, Visitable};

use crate::io::NODE_ANNOTATION_FEATURE_KEY;
use crate::{Edge, Node, NonTerminal, Span, Terminal, Tree, WriteTree};

/// Iterator over constituency trees in a Negra export file.
///
/// `next()` moves the reader until the first `#BOS` is found, then collects lines until `#EOS` is
/// found. Returns `Some(Error)` if overlapping sentences are found, `'#BOS` -> `#EOS` is violated
/// or if the collected lines are not a well formed NEGRA sentence.
///
/// Note:   If the reader never encounters a line starting with to `#BOS`, `None` is returned.
pub struct NegraReader<R>
where
    R: BufRead,
{
    inner: Lines<R>,
}

impl<R> NegraReader<R>
where
    R: BufRead,
{
    /// Creates a new `NegraReader` over the trees in the reader.
    pub fn new(reader: R) -> NegraReader<R> {
        NegraReader {
            inner: reader.lines(),
        }
    }
}

impl<R> Iterator for NegraReader<R>
where
    R: BufRead,
{
    type Item = Result<Tree, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut cur = None;

        while let Some(line) = self.inner.next() {
            let line = match line {
                Ok(line) => line,
                Err(e) => return Some(Err(e.into())),
            };
            let line = line.trim();
            if line.starts_with("#BOS") {
                // if we encounter "#BOS" while inside a sentence, return error
                if cur.is_some() {
                    return Some(Err(format_err!("Found second #BOS before #EOS\n{}", line)));
                }

                match Builder::from_header(line) {
                    Ok(builder) => cur = Some(builder),
                    Err(e) => return Some(Err(e)),
                };
            } else if line.starts_with("#EOS") {
                if let Some(builder) = cur.take() {
                    return Some(builder.try_into_tree(line));
                } else {
                    return Some(Err(format_err!("Found #EOS without #BOS\n{}", line)));
                }
            } else if let Some(builder) = cur.as_mut() {
                if let Err(e) = builder.process_line(line) {
                    return Some(Err(e));
                }
            }
        }
        None
    }
}

/// Builder for Negra export format trees.
struct Builder {
    nt_indices: HashMap<usize, NodeIndex>,
    graph: StableGraph<Node, Edge>,
    n_terminals: usize,
    sentence_id: String,
    finished_terminals: bool,
    num_non_projective: usize,
    processed_nts: usize,
    sec_edges: Vec<(usize, String, NodeIndex)>,
}

impl Builder {
    /// Construct a new `Builder` from the `header`.
    ///
    /// * panics if `header` doesn't start with `#BOS`.
    fn from_header(header: &str) -> Result<Self, Error> {
        assert!(
            header.starts_with("#BOS"),
            "Sentence did not start with #BOS line."
        );
        let mut header_parts = header.splitn(3, |c: char| c.is_whitespace());
        header_parts.next().unwrap();

        let mut graph = StableGraph::new();
        let bos_id =
            read_required_numerical_field(header_parts.next()).map_err(|e| format_err!("{}", e))?;
        let mut root = Node::from(NonTerminal::new("VROOT", 0));
        let metadata = header_parts.next();
        root.features_mut()
            .insert("sentence_id", Some(bos_id.to_string()));
        if let Some(metadata) = metadata {
            root.features_mut()
                .insert("metadata", Some(metadata.trim()));
        }
        let mut nt_indices = HashMap::new();
        nt_indices.insert(0, graph.add_node(root));
        Ok(Builder {
            graph,
            nt_indices,
            sentence_id: bos_id.to_string(),
            finished_terminals: false,
            n_terminals: 0,
            num_non_projective: 0,
            processed_nts: 1,
            sec_edges: Vec::new(),
        })
    }

    /// Construct a `Tree` from the `Builder`.
    ///
    ///   * panics if the footer doesn't start with `#EOS`.
    ///   * returns `Err` if the sentence id doesn't match.
    ///   * returns `Err` if not all NonTerminals were processed.
    fn try_into_tree(mut self, footer: &str) -> Result<Tree, Error> {
        assert!(
            footer.starts_with("#EOS"),
            "Sentence has to end with #EOS line."
        );
        let mut parts = footer.split_whitespace();
        parts.next();
        let eos_id = read_required_string_field(parts.next()).map_err(|e| format_err!("{}", e))?;
        if eos_id != self.sentence_id {
            return Err(format_err!(
                "IDs don't match: #BOS {} and #EOS {}",
                self.sentence_id,
                eos_id
            ));
        }
        for (source, label, target) in self.sec_edges {
            self.graph.add_edge(
                *self.nt_indices.get(&source).unwrap(),
                target,
                Edge::Secondary(Some(label)),
            );
        }

        let root = *self.nt_indices.get(&0).unwrap();
        if self.processed_nts != self.nt_indices.len() {
            return Err(format_err!("Did not process all nonterminals."));
        }
        self.graph[root].set_span(Span::new(0, self.n_terminals))?;
        Ok(Tree::new_from_parts(
            self.graph,
            self.n_terminals,
            root,
            self.num_non_projective,
        ))
    }

    /// Add the given line to the builder.
    fn process_line(&mut self, line: &str) -> Result<(), Error> {
        if line.starts_with('#') {
            self.process_nonterminal_line(line)
        } else {
            self.process_terminal_line(line)
        }
    }

    /// Process line containing a NonTerminal.
    ///
    /// Returns Err if:
    ///   * No terminals were added yet.
    ///   * Parts are missing.
    fn process_nonterminal_line(&mut self, line: &str) -> Result<(), Error> {
        if self.n_terminals == 0 {
            return Err(format_err!("Tree without Terminal nodes."));
        }
        self.finished_terminals = true;
        self.processed_nts += 1;
        // Line is expected to be:
        // #SELF_ID filler label(=annotation) filler edge parent_id
        // possible secondary edges and comments are ignored
        let mut parts = line.split_whitespace();

        let self_id = read_required_numerical_field(parts.next().map(|part| &part[1..]))?;
        let self_idx = *self
            .nt_indices
            .get(&self_id)
            .ok_or_else(|| format_err!("Nonterminal without children."))?;
        read_required_string_field(parts.next())?;

        let mut label_parts = read_required_string_field(parts.next())?.split('=');
        let label = read_required_string_field(label_parts.next())?;
        self.graph[self_idx].set_label(label);

        if let Some(annotation) = label_parts.next() {
            self.graph[self_idx]
                .features_mut()
                .insert(NODE_ANNOTATION_FEATURE_KEY, Some(annotation));
        }
        read_required_string_field(parts.next())?;

        self.process_edges(self_idx, parts)?;

        let coverage = self
            .graph
            .edges_directed(self_idx, Direction::Outgoing)
            .filter(|e| e.weight().is_primary())
            .flat_map(|e| self.graph[e.target()].span().into_iter())
            .collect();
        let span = Span::from_vec(coverage)?;
        if span.skips().is_some() {
            self.num_non_projective += 1;
        }
        self.graph[self_idx].set_span(span)?;
        Ok(())
    }

    /// Process line containing a NonTerminal.
    ///
    /// Returns Err if:
    ///   * A NonTerminal was added.
    ///   * Parts are missing.
    fn process_terminal_line(&mut self, line: &str) -> Result<(), Error> {
        if self.finished_terminals {
            return Err(format_err!(
                "Order violation. First Terminals, then NonTerminals expected."
            ));
        }

        // terminal line is expected to be:
        // Form Lemma pos morph edge parent_id
        // possible secondary edges and comments are ignored.
        let mut parts = line.split_whitespace();
        let form = read_required_string_field(parts.next())?;
        let lemma = read_required_string_field(parts.next())?;
        let pos = read_required_string_field(parts.next())?;
        let morph = read_required_string_field(parts.next())?;
        let mut terminal = Terminal::new(form, pos, self.n_terminals);
        if lemma != "--" {
            terminal.set_lemma(Some(lemma));
        }
        if morph != "--" {
            terminal.features_mut().insert("morph", Some(morph));
        }
        let terminal_idx = self.graph.add_node(terminal.into());
        self.process_edges(terminal_idx, parts)?;
        self.n_terminals += 1;
        Ok(())
    }

    fn process_edges<'a>(
        &mut self,
        node_idx: NodeIndex,
        mut parts: impl Iterator<Item = &'a str>,
    ) -> Result<(), Error> {
        let edge = read_required_string_field(parts.next())?;
        let parent_id = read_required_numerical_field(parts.next())?;
        // unlabeled edges denoted as "--" in tueba
        let edge = if edge == "--" { None } else { Some(edge) };
        let parent_idx = match self.nt_indices.entry(parent_id) {
            Entry::Occupied(v) => *v.get(),
            Entry::Vacant(v) => *v.insert(
                self.graph
                    .add_node(NonTerminal::new("", self.n_terminals).into()),
            ),
        };
        self.graph
            .add_edge(parent_idx, node_idx, Edge::new_primary(edge));
        match Self::parse_optional_fields(&mut parts)? {
            OptionalLinePart::SecEdge((label, id)) => {
                self.sec_edges.push((id, label.to_string(), node_idx));
                if let OptionalLinePart::Comment(comment) = Self::parse_optional_fields(&mut parts)?
                {
                    self.graph[node_idx]
                        .features_mut()
                        .insert("comment", Some(comment));
                }
            }
            OptionalLinePart::Comment(comment) => {
                self.graph[node_idx]
                    .features_mut()
                    .insert("comment", Some(comment));
            }
            OptionalLinePart::None => {}
        }
        Ok(())
    }

    fn parse_optional_fields<'a>(
        mut parts: impl Iterator<Item = &'a str>,
    ) -> Result<OptionalLinePart<'a>, Error> {
        if let Some(part) = parts.next() {
            if !part.starts_with('%') {
                let edge = part;
                let id = read_required_numerical_field(parts.next())?;
                Ok(OptionalLinePart::SecEdge((edge, id)))
            } else {
                Ok(OptionalLinePart::Comment(format!(
                    "{} {}",
                    part,
                    parts.join(" ")
                )))
            }
        } else {
            Ok(OptionalLinePart::None)
        }
    }
}

enum OptionalLinePart<'a> {
    Comment(String),
    SecEdge((&'a str, usize)),
    None,
}

fn read_required_string_field(field: Option<&str>) -> Result<&str, Error> {
    field.ok_or_else(|| format_err!("Line missing field."))
}

fn read_required_numerical_field(field: Option<&str>) -> Result<usize, Error> {
    if let Some(parse) = field.map(str::parse::<usize>) {
        parse.map_err(|e| format_err!("Can't parse value for numerical field: {}", e))
    } else {
        Err(format_err!("Line missing field."))
    }
}

/// Writer for Negra export 4 format.
///
/// Write trees in pointer based Negra export 4 format. This format allows for discontinuous
/// phrases.
///
/// The sentence ID is taken from the root node's `"sentence_id"` feature. Other metadata is taken
/// from the root's `"metadata"` feature. This metadata is usually the annotator ID, unix timestamp
/// since last edit and a file ID as well as other comments about the sentence. The metadata is
/// written as is.
///
/// If no `sentence_id` is given, the writer numbers the sentences starting at `0`.
pub struct NegraWriter<W> {
    writer: W,
    counter: usize,
}

impl<W> NegraWriter<W> {
    /// Construct a new writer.
    pub fn new(writer: W) -> Self {
        NegraWriter { writer, counter: 0 }
    }

    /// Return a string with spaces as padding.
    fn pad(item: &str, to: usize) -> String {
        let len = item.chars().count();
        if len >= to {
            " ".to_string()
        } else if len < to {
            (len..to).map(|_| " ").collect()
        } else {
            (len..to).map(|_| " ").collect()
        }
    }

    /// Convert a terminal to its Negra String representation.
    ///
    /// The string representation does not include the edge and parent ID.
    fn terminal_to_negra_line(terminal: &Terminal) -> String {
        let mut string_rep = String::with_capacity(84);
        string_rep.push_str(terminal.form());
        string_rep.push_str(&Self::pad(terminal.form(), 24));
        let lemma = terminal.lemma().unwrap_or_else(|| "--");
        string_rep.push_str(lemma);
        string_rep.push_str(&Self::pad(lemma, 24));
        string_rep.push_str(terminal.label());
        string_rep.push_str(&Self::pad(terminal.label(), 8));
        let features = terminal.features();
        if let Some(Some(morph)) = features.and_then(|f| f.get_val("morph")) {
            string_rep.push_str(morph);
            string_rep.push_str(&Self::pad(morph, 16));
        } else {
            string_rep.push_str("--");
            string_rep.push_str(&Self::pad("--", 16));
        }
        string_rep
    }

    /// Convert a NonTerminal to its Negra String representation.
    ///
    /// The string representation does not include the edge and parent ID.
    fn nonterminal_to_negra_line(id: &str, nt: &NonTerminal) -> String {
        let mut nt_rep = String::with_capacity(84);
        nt_rep.push('#');
        nt_rep.push_str(&id);
        nt_rep.push_str(&Self::pad("#500", 24));
        nt_rep.push_str("--");
        nt_rep.push_str(&Self::pad("--", 24));
        let mut label = nt.label().to_string();
        if let Some(Some(annotation)) = nt
            .features()
            .and_then(|f| f.get_val(NODE_ANNOTATION_FEATURE_KEY))
        {
            label.push('=');
            label.push_str(annotation);
        }
        nt_rep.push_str(&label);
        nt_rep.push_str(&Self::pad(&label, 8));
        nt_rep.push_str("--");
        nt_rep.push_str(&Self::pad("--", 16));
        nt_rep
    }

    /// Get the Negra String representation for edge and parent ID.
    fn get_parent_edge(idx: NodeIndex, tree: &Tree, id_map: &HashMap<NodeIndex, String>) -> String {
        let (parent, edge) = tree.parent(idx).unwrap();
        let mut edge_rep = tree[edge].label().unwrap_or_else(|| "--").to_string();
        edge_rep.push_str(&Self::pad(&edge_rep, 8));
        let id = id_map.get(&parent).unwrap();
        edge_rep.push_str(id);
        if let Some((parent, sec_edge)) = tree.secondary_parent(idx) {
            edge_rep.push_str(&Self::pad(id, 8));
            let label = tree[sec_edge].label().unwrap_or_else(|| "--");
            edge_rep.push_str(label);
            edge_rep.push_str(&Self::pad(label, 8));
            edge_rep.push_str(id_map.get(&parent).unwrap());
        }

        edge_rep
    }
}

impl<W> WriteTree for NegraWriter<W>
where
    W: Write,
{
    fn write_tree(&mut self, tree: &Tree) -> Result<(), Error> {
        let sentence_id = if let Some(Some(sentence_id)) = tree[tree.root()]
            .features()
            .and_then(|f| f.get_val("sentence_id"))
        {
            sentence_id.to_string()
        } else {
            self.counter += 1;
            (self.counter - 1).to_string()
        };
        if let Some(Some(metadata)) = tree[tree.root()]
            .features()
            .and_then(|f| f.get_val("metadata"))
        {
            writeln!(self.writer, "#BOS {}  {}", sentence_id, metadata)?;
        } else {
            writeln!(self.writer, "#BOS {}", sentence_id)?;
        };

        let mut terminals = tree.terminals().collect::<Vec<_>>();
        tree.sort_indices(&mut terminals);
        let mut visit_map = tree.graph().visit_map();
        let mut count = 500;

        let mut nt_id_map = HashMap::new();
        nt_id_map.insert(tree.root(), "0".into());
        let mut queue = VecDeque::new();

        // find nonterminals that don't dominate other nonterminals
        for &terminal in terminals.iter() {
            visit_map.visit(terminal);
            if let Some((parent, _)) = tree.parent(terminal) {
                if tree.children(parent).all(|(n, _)| tree[n].is_terminal()) {
                    queue.push_back(parent);
                }
            }
        }

        // assign numbers from 500 onwards to nonterminals
        let mut nts = Vec::new();
        while let Some(node) = queue.pop_front() {
            if !visit_map.visit(node) || tree.root() == node {
                continue;
            }
            nt_id_map.insert(node, count.to_string());
            nts.push(node);
            count += 1;
            if let Some((parent, _)) = tree.parent(node) {
                if tree.children(parent).all(|(n, _)| visit_map.is_visited(&n)) {
                    queue.push_front(parent);
                }
            }
        }

        for terminal_idx in terminals {
            let terminal = tree[terminal_idx].terminal().unwrap();
            write!(self.writer, "{}", Self::terminal_to_negra_line(terminal))?;
            write!(
                self.writer,
                "{}",
                Self::get_parent_edge(terminal_idx, tree, &nt_id_map)
            )?;
            if let Some(Some(comment)) = terminal.features().and_then(|f| f.get_val("comment")) {
                writeln!(self.writer, "     {}", comment)?;
            } else {
                writeln!(self.writer)?;
            }
        }

        for nt_idx in nts.into_iter() {
            let nt_id = nt_id_map.get(&nt_idx).map(String::as_str).unwrap();
            let nt = tree[nt_idx].nonterminal().unwrap();
            write!(
                self.writer,
                "{}",
                Self::nonterminal_to_negra_line(nt_id, nt)
            )?;
            write!(
                self.writer,
                "{}",
                Self::get_parent_edge(nt_idx, tree, &nt_id_map)
            )?;
            if let Some(Some(comment)) = nt.features().and_then(|f| f.get_val("comment")) {
                writeln!(self.writer, " {}", comment)?;
            } else {
                writeln!(self.writer)?;
            }
        }

        writeln!(self.writer, "#EOS {}", sentence_id)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use petgraph::prelude::{NodeIndex, StableGraph};
    use std::fs::File;
    use std::io::BufReader;

    use super::{NegraReader, NegraWriter};

    use crate::io::negra::Builder;
    use crate::io::NODE_ANNOTATION_FEATURE_KEY;
    use crate::{Edge, Features, Node, NonTerminal, Span, Terminal, Tree, WriteTree};

    #[test]
    fn test_first10_ok() {
        let input = File::open("testdata/10.negra").unwrap();
        let reader = BufReader::new(input);
        let mut n = 0;
        for tree in NegraReader::new(reader) {
            let tree = tree.unwrap();
            let mut buffer = Vec::new();
            let mut negra_writer = NegraWriter::new(&mut buffer);
            negra_writer.write_tree(&tree).unwrap();
            let mut negra_iter = NegraReader::new(BufReader::new(buffer.as_slice()));
            assert_eq!(tree, negra_iter.next().unwrap().unwrap());
            n += 1;
        }
        assert_eq!(n, 10)
    }

    #[test]
    fn test_iter() {
        let f = File::open("testdata/10.negra").unwrap();
        let mut iter = NegraReader::new(BufReader::new(f));
        iter.next();
        iter.next();
        let string_rep = "#BOS 3  2 1070544990 0 %% HEADLINE\n\
                          L	L	NN	nsf		HD	500\n\
                          U			U			NE	nsf		-	501\n\
                          W		W		NE	nsf		-	501\n\
                          :			:			$.	--		--	0\n\
                          E		e			ART	nsm		-	503\n\
                          B		B		NN	nsm		HD	503\n\
                          #500			--			NX	--		APP	502\n\
                          #501			--			NX=PER	--		APP	502\n\
                          #502			--			NX	--		--	0\n\
                          #503			--			NX	--		--	0\n\
                          #EOS 3"
            .as_bytes();
        let mut reader = NegraReader::new(BufReader::new(string_rep));
        assert_eq!(
            iter.next().unwrap().unwrap(),
            reader.next().unwrap().unwrap()
        )
    }

    #[test]
    fn test_first() {
        let f = File::open("testdata/single.negra").unwrap();
        let br = BufReader::new(f);
        let tree = NegraReader::new(br).next().unwrap().unwrap();
        let mut g = StableGraph::new();
        let mut v = Terminal::new("V", "VVFIN", 0);
        v.set_lemma(Some("v"));
        v.set_features(Some("morph:3sit".into()));
        let mut d = Terminal::new("d", "ART", 1);
        d.set_lemma(Some("d"));
        d.set_features(Some("morph:nsf".into()));
        let mut a = Terminal::new("A", "NN", 2);
        a.set_lemma(Some("a"));
        a.set_features(Some("morph:nsf".into()));
        let mut s = Terminal::new("S", "NN", 3);
        s.set_lemma(Some("s"));
        s.set_features(Some("morph:asn".into()));
        let mut punct = Terminal::new("?", "$.", 4);
        punct.set_lemma(Some("?"));
        let root = g.add_node(Node::from(NonTerminal::new("VROOT", Span::new(0, 5))));
        g[root]
            .features_mut()
            .insert("metadata", Some("2 1202391857 0 %% HEADLINE"));
        g[root].features_mut().insert("sentence_id", Some("1"));
        let v = g.add_node(v.into());
        let vxfin = g.add_node(NonTerminal::new("VXFIN", Span::new(0, 1)).into());
        let d = g.add_node(d.into());
        let mut nxorg = NonTerminal::new("NX", Span::new(1, 3));
        nxorg
            .features_mut()
            .insert(NODE_ANNOTATION_FEATURE_KEY, Some("ORG"));
        let nxorg = g.add_node(nxorg.into());
        let a = g.add_node(a.into());
        let s = g.add_node(s.into());
        let nx = g.add_node(NonTerminal::new("NX", Span::new(3, 4)).into());
        let punct = g.add_node(punct.into());

        let lk = g.add_node(NonTerminal::new("LK", Span::new(0, 1)).into());
        let simpx = g.add_node(NonTerminal::new("SIMPX", Span::new(0, 4)).into());
        let mf = g.add_node(Node::NonTerminal(NonTerminal::new("MF", Span::new(1, 4))));

        g.add_edge(vxfin, v, Edge::new_primary(Some("HD")));
        g.add_edge(nxorg, d, Edge::new_primary(Some("-NE")));
        g.add_edge(nxorg, a, Edge::new_primary(Some("HD")));
        g.add_edge(nx, s, Edge::new_primary(Some("HD")));
        g.add_edge(mf, nx, Edge::new_primary(Some("OA")));
        g.add_edge(root, punct, Edge::new_primary::<String>(None));
        g.add_edge(lk, vxfin, Edge::new_primary(Some("HD")));
        g.add_edge(simpx, lk, Edge::new_primary(Some("-")));
        g.add_edge(mf, nxorg, Edge::new_primary(Some("ON")));
        g.add_edge(simpx, mf, Edge::new_primary(Some("-")));
        g.add_edge(root, simpx, Edge::new_primary::<String>(None));
        assert_eq!(tree, Tree::new_from_parts(g, 5, root, 0));
    }

    #[test]
    fn test_string() {
        let f = File::open("testdata/long_single.negra").unwrap();
        let mut reader = NegraReader::new(BufReader::new(f));
        let target_surface_form = "V , s d S e M d A , s d L U W , d s j a \" S \" g , \" w \
                                   d a w , w e s m T z \" . ";
        let tree = reader.next().unwrap().unwrap();
        let surface_form = tree
            .terminals()
            .filter_map(|id| tree[id].terminal())
            .map(|terminal| terminal.form())
            .fold(String::new(), |mut acc, form| {
                acc.push_str(form);
                acc.push(' ');
                acc
            });
        assert_eq!(surface_form, target_surface_form)
    }

    #[test]
    fn terminal() {
        let mut builder = Builder::from_header("#BOS 0").unwrap();
        let term = "was etwas   PIS *** HD  502 % some random comment that doesnt get ignored\n";
        builder.process_line(term).unwrap();
        assert_eq!(builder.graph.node_count(), 3);
        let mut term = Terminal::new("was", "PIS", 0);
        term.set_lemma(Some("etwas"));
        term.set_features(Some(Features::from(
            "morph:***|comment:% some random comment that doesnt get ignored",
        )));
        assert_eq!(builder.graph[NodeIndex::new(1)], term.into());

        assert!(builder
            .process_line("#was etwas   PIS *** HD 502\n")
            .is_err());
    }

    #[test]
    fn nonterminal() {
        let mut builder = Builder::from_header("#BOS 0").unwrap();
        let nt = "#502			--			NX	--		ON	503\n";
        assert!(builder.process_line(nt).is_err());
        assert_eq!(builder.graph.node_count(), 1);
        let term = "was etwas   PIS *** HD  502 %some random comment that gets ignored\n";
        builder.process_line(term).unwrap();
        builder.process_line(nt).unwrap();
        assert_eq!(builder.graph.node_count(), 4);
        let target = NonTerminal::new("NX", 0);
        assert_eq!(builder.graph[NodeIndex::new(2)], target.into());
    }

    #[test]
    #[should_panic]
    fn terminal_without_parent() {
        let mut builder = Builder::from_header("#BOS 0").unwrap();
        let bad_term = "was etwas   PIS *** HD\n";
        builder.process_line(bad_term).unwrap();
    }

    #[test]
    #[should_panic]
    fn terminal_missing_field() {
        let mut builder = Builder::from_header("#BOS 0").unwrap();
        let bad_term = "was    PIS *** HD 502\n";
        builder.process_line(bad_term).unwrap();
    }

    #[test]
    fn wellformed() {
        let s = "#BOS 1  2 1202391857 0 %% HEADLINE
        V		v		VVFIN	3sit		HD	500
        die			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0
        #EOS 1"
            .as_bytes();
        let mut reader = NegraReader::new(BufReader::new(s));
        reader.next().unwrap().unwrap();
    }

    #[test]
    #[should_panic]
    fn malformed_missing_bos() {
        let s = "V		v		VVFIN	3sit		HD	500
        d			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0
        #EOS 1"
            .as_bytes();
        let mut reader = NegraReader::new(BufReader::new(s));
        reader.next().unwrap().unwrap();
    }

    #[test]
    #[should_panic]
    fn malformed_invalid_start() {
        let s = "#BOS
        V		v		VVFIN	3sit		HD	500
        d			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0
        #EOS 1"
            .as_bytes();
        let mut reader = NegraReader::new(BufReader::new(s));
        reader.next().unwrap().unwrap();
    }

    #[test]
    #[should_panic]
    fn malformed_matching_ids() {
        let s = "#BOS 0
        V		v		VVFIN	3sit		HD	500
        d			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0
        #EOS 1"
            .as_bytes();
        let mut reader = NegraReader::new(BufReader::new(s));
        reader.next().unwrap().unwrap();
    }

    #[test]
    #[should_panic]
    fn malformed_missing_eos() {
        let s = "#BOS 1  2 1202391857 0 %% HEADLINE
        V		v		VVFIN	3sit		HD	500
        d			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0"
            .as_bytes();
        let mut reader = NegraReader::new(BufReader::new(s));
        reader.next().unwrap().unwrap();
    }

    #[test]
    fn wellformed_no_nts() {
        let s = "#BOS 1  2 1202391857 0 %% HEADLINE
        V		v		VVFIN	3sit		HD	0
        d			d			ART	nsf		-NE	0
        A			a			NN	nsf		HD	0
        S		s		NN	asn		HD	0
        ?			?			$.	--		--	0
        #EOS 1"
            .as_bytes();
        let mut reader = NegraReader::new(BufReader::new(s));
        reader.next().unwrap().unwrap();
    }
}
