use std::collections::{hash_map::Entry, HashMap};
use std::io::{BufRead, Lines};

use failure::Error;
use petgraph::prelude::{Direction, NodeIndex, StableGraph};

use crate::io::NODE_ANNOTATION_FEATURE_KEY;
use crate::{Edge, Node, NonTerminal, Span, Terminal, Tree};

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
                Err(e) => return Some(Err(format_err!("{}", e))),
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
        let root = self.nt_indices.remove(&0).unwrap();
        if !self.nt_indices.is_empty() {
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
        // Line is expected to be:
        // #SELF_ID filler label(=annotation) filler edge parent_id
        // possible secondary edges and comments are ignored
        let mut parts = line.split_whitespace();

        let self_id = read_required_numerical_field(parts.next().map(|part| &part[1..]))?;
        let self_idx = self
            .nt_indices
            .remove(&self_id)
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

        let edge = read_required_string_field(parts.next())?;
        let edge = if edge == "--" { None } else { Some(edge) };

        let parent_id = read_required_numerical_field(parts.next())?;
        let parent_idx = match self.nt_indices.entry(parent_id) {
            Entry::Occupied(v) => *v.get(),
            Entry::Vacant(v) => *v.insert(self.graph.add_node(NonTerminal::new("", 0).into())),
        };
        self.graph.add_edge(parent_idx, self_idx, edge.into());

        let coverage = self
            .graph
            .neighbors_directed(self_idx, Direction::Outgoing)
            .map(|c| self.graph[c].span().into_iter())
            .flatten()
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
        let edge = read_required_string_field(parts.next())?;
        let parent_id = read_required_numerical_field(parts.next())?;
        // unlabeled edges denoted as "--" in tueba
        let edge = if edge == "--" { None } else { Some(edge) };
        let mut terminal = Terminal::new(form, pos, self.n_terminals);
        terminal.set_lemma(Some(lemma));
        if morph != "--" {
            terminal.features_mut().insert("morph", Some(morph));
        }
        let terminal_idx = self.graph.add_node(terminal.into());
        let parent_idx = match self.nt_indices.entry(parent_id) {
            Entry::Occupied(v) => *v.get(),
            Entry::Vacant(v) => *v.insert(
                self.graph
                    .add_node(NonTerminal::new("", self.n_terminals).into()),
            ),
        };
        self.n_terminals += 1;
        self.graph.add_edge(parent_idx, terminal_idx, edge.into());
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use petgraph::prelude::{NodeIndex, StableGraph};

    use crate::io::{negra::Builder, NODE_ANNOTATION_FEATURE_KEY};
    use crate::{Edge, Features, NegraReader, Node, NonTerminal, Span, Terminal, Tree};

    #[test]
    fn test_first10_ok() {
        let input = File::open("testdata/10.negra").unwrap();
        let reader = BufReader::new(input);
        let mut n = 0;
        for tree in NegraReader::new(reader) {
            tree.unwrap();
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
        let vxfin = g.add_node(NonTerminal::new(
            "VXFIN",
            Span::new(0, 1),
        ).into());
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
        let simpx = g.add_node(NonTerminal::new(
            "SIMPX",
            Span::new(0, 4),
        ).into());
        let mf = g.add_node(Node::NonTerminal(NonTerminal::new("MF", Span::new(1, 4))));


        g.add_edge(vxfin, v, Edge::from(Some("HD")));
        g.add_edge(nxorg, d, Edge::from(Some("-NE")));
        g.add_edge(nxorg, a, Edge::from(Some("HD")));
        g.add_edge(nx, s, Edge::from(Some("HD")));
        g.add_edge(mf, nx, Edge::from(Some("OA")));
        g.add_edge(root, punct, Edge::default());
        g.add_edge(lk, vxfin, Edge::from(Some("HD")));
        g.add_edge(simpx, lk, Edge::from(Some("-")));
        g.add_edge(mf, nxorg, Edge::from(Some("ON")));
        g.add_edge(simpx, mf, Edge::from(Some("-")));
        g.add_edge(root, simpx, Edge::default());
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
        let term = "was etwas   PIS *** HD  502 %some random comment that gets ignored\n";
        builder.process_line(term).unwrap();
        assert_eq!(builder.graph.node_count(), 3);
        let mut term = Terminal::new("was", "PIS", 0);
        term.set_lemma(Some("etwas"));
        term.set_features(Some(Features::from("morph:***")));
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
