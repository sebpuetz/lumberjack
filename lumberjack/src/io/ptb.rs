use std::io::{BufRead, Lines};

use failure::Error;
use pest::iterators::Pair;
use pest::Parser;
use petgraph::prelude::{Direction, EdgeRef, NodeIndex, StableGraph};

use crate::io::{ReadTree, WriteTree, NODE_ANNOTATION_FEATURE_KEY};
use crate::{Edge, Node, NonTerminal, Projectivity, Span, Terminal, Tree};

/// `PTBFormat`
pub enum PTBFormat {
    /// PTB V2 Format.
    ///
    /// Trees don't include edge labels, some nodes contain additional tags. Node labels are split
    /// on the first `"-"`, additional tags are put together into the same `annotation` field.
    ///
    /// E.g. `"(TAG-annotation1-annotation2 (POS terminal))"` results in a non-terminal with:
    /// * `label == "TAG"`
    /// * `annotation == Some("annotation1-annotation2")`
    /// * `edge == None`
    PTB,
    /// Simple Format.
    ///
    /// Assumes trees don't include edge labels or any other annotations.
    ///
    /// E.g. `"(TAG (POS terminal))"` results in a non-terminal with:
    /// * `label == "TAG"`
    /// * `annotation == None`
    /// * `edge == None`
    Simple,
    /// TueBa PTB V2 Format.
    ///
    /// Trees include edge labels, some nodes contain semantic annotations. Node labels are first
    /// split on `":"`, to seperate content into node tag and edge, then the tag is split again on
    /// `"="` to seperate the tag and grammatical annotation.
    ///
    /// E.g., `"(TAG=annotation:edge_label form (POS terminal))"` results in a terminal with:
    /// * `label == "TAG"`
    /// * `annotation == Some("annotation")`
    /// * `edge == Some("edge_label")`
    TueBa,
}

impl WriteTree for PTBFormat {
    fn tree_to_string(&self, tree: &Tree) -> Result<String, Error> {
        if tree.projective() {
            Ok(self.format_sub_tree(tree, tree.root(), None))
        } else {
            Err(format_err!("Can't linearize nonprojective tree"))
        }
    }
}

// dummy struct required by pest
#[derive(Parser)]
#[grammar = "io/ptb.pest"]
struct PTBParser;

impl ReadTree for PTBFormat {
    fn string_to_tree(&self, string: &str) -> Result<Tree, Error> {
        let mut graph = StableGraph::new();
        let mut n_terminals = 0;
        let mut parsed_line = PTBParser::parse(Rule::tree, string)?;
        let (_, root, _) =
            self.parse_value(parsed_line.next().unwrap(), &mut graph, &mut n_terminals)?;
        Ok(Tree::new(
            graph,
            n_terminals,
            root,
            Projectivity::Projective,
        ))
    }
}

impl PTBFormat {
    pub fn try_from_str(s: &str) -> Result<PTBFormat, Error> {
        let s = s.to_lowercase();
        match s.as_str() {
            "tueba" => Ok(PTBFormat::TueBa),
            "ptb" => Ok(PTBFormat::PTB),
            "simple" => Ok(PTBFormat::Simple),
            _ => Err(format_err!("Unknown format: {}", s)),
        }
    }

    fn format_sub_tree(&self, sentence: &Tree, position: NodeIndex, edge: Option<&str>) -> String {
        let sent = sentence.graph();

        match &sent[position] {
            Node::Terminal(terminal) => self.fmt_term(terminal.form(), terminal.label(), edge),
            Node::NonTerminal(nt) => {
                let mut nodes: Vec<_> = sent
                    .edges_directed(position, Direction::Outgoing)
                    .collect::<Vec<_>>();
                // sort child nodes by covered span
                nodes.sort_by(|edge_ref_1, edge_ref_2| {
                    let span_1 = sentence.graph()[edge_ref_1.target()].span();
                    let span_2 = sentence.graph()[edge_ref_2.target()].span();
                    span_1.cmp(&span_2)
                });
                let mut sub_tree_rep = Vec::with_capacity(nodes.len());
                sub_tree_rep.push(self.fmt_inner(nt, edge));
                sub_tree_rep.extend(nodes.into_iter().map(|edge_ref| {
                    self.format_sub_tree(sentence, edge_ref.target(), edge_ref.weight().label())
                }));
                let node_sep = if let PTBFormat::TueBa = self { "" } else { " " };
                format!("({})", sub_tree_rep.join(node_sep))
            }
        }
    }

    fn fmt_inner(&self, nt: &NonTerminal, edge: Option<&str>) -> String {
        let mut representation = nt.label().to_string();
        let annotation = if let Some(annotation) = nt
            .features()
            .map(|f| f.get_val(NODE_ANNOTATION_FEATURE_KEY))
        {
            annotation
        } else {
            None
        };
        match self {
            PTBFormat::PTB => {
                if let Some(annotation) = annotation {
                    representation.push('-');
                    representation.push_str(annotation);
                }
            }
            PTBFormat::Simple => (),
            PTBFormat::TueBa => {
                if let Some(annotation) = annotation {
                    representation.push('=');
                    representation.push_str(annotation);
                }
                representation.push(':');
                representation.push_str(edge.unwrap_or("--"));
            }
        }
        representation
    }

    fn fmt_term(&self, form: &str, pos: &str, edge: Option<&str>) -> String {
        let pos = pos.replace("(", "LBR").replace(")", "RBR");
        let form = form.replace("(", "LBR").replace(")", "RBR");
        if let PTBFormat::TueBa = self {
            let edge = edge.unwrap_or("--");
            format!("({}:{} {})", pos, edge, form)
        } else {
            format!("({} {})", pos, form)
        }
    }

    // this method traverses the linearized tree and builds a StableGraph
    fn parse_value(
        &self,
        pair: Pair<Rule>,
        g: &mut StableGraph<Node, Edge>,
        terminals: &mut usize,
    ) -> Result<(Span, NodeIndex, Edge), Error> {
        match pair.as_rule() {
            Rule::nonterminal => {
                let mut pairs = pair.into_inner();
                // first rule after matching nonterminal will always be the label of the inner node
                let (label, edge, annotation) = self.process_label(pairs.next().unwrap())?;
                let mut nt = NonTerminal::new(label, 0);
                if annotation.is_some() {
                    nt.features_mut().insert(
                        NODE_ANNOTATION_FEATURE_KEY,
                        annotation.map(ToOwned::to_owned),
                    );
                };
                let nt_idx = g.add_node(Node::NonTerminal(nt));
                // collect children
                let mut lower = 0;
                let mut upper = 0;
                for (idx, inner_pair) in pairs.enumerate() {
                    let (span, child_idx, edge) = self.parse_value(inner_pair, g, terminals)?;
                    if idx == 0 {
                        lower = span.lower();
                    }
                    upper = span.upper();
                    g.add_edge(nt_idx, child_idx, edge);
                }
                let span = Span::new_continuous(lower, upper);
                g[nt_idx].nonterminal_mut().unwrap().set_span(span.clone());

                Ok((span, nt_idx, edge.into()))
            }
            Rule::preterminal => {
                let (edge, pos, form) = self.process_preterminal(pair)?;
                let term_idx = g.add_node(Node::Terminal(Terminal::new(form, pos, *terminals)));
                let span = Span::from(*terminals);
                *terminals += 1;
                Ok((span, term_idx, edge.into()))
            }
            _ => {
                eprintln!("{:?}", pair);
                unreachable!()
            }
        }
    }

    // Preterminal consists of POS, optional Edge label and Terminal
    fn process_preterminal<'a>(
        &self,
        pair: Pair<'a, Rule>,
    ) -> Result<(Option<&'a str>, &'a str, &'a str), Error> {
        let mut pairs = pair.into_inner();
        let pos = pairs.next().unwrap();
        let (tag, edge, _) = self.process_label(pos)?;

        let form = pairs.next().unwrap();
        if let Rule::terminal = form.as_rule() {
            Ok((edge, tag, form.as_str()))
        } else {
            Err(format_err!(
                "Preterminal not starting with form: {}",
                form.as_str()
            ))
        }
    }

    // All nodes in the tree start with a label corresponding either to the parse tag or to the POS of
    // a given token. The label is optionally followed by an edge label.
    fn process_label<'a>(
        &self,
        pair: Pair<'a, Rule>,
    ) -> Result<(&'a str, Option<&'a str>, Option<&'a str>), Error> {
        if let Rule::node_label = pair.as_rule() {
            let label = pair.as_str();
            // split label and edge label
            match self {
                PTBFormat::PTB => {
                    if let Some(idx) = label.find('-') {
                        Ok((&label[..idx], None, Some(&label[idx..])))
                    } else {
                        Ok((label, None, None))
                    }
                }
                PTBFormat::TueBa => {
                    let mut parts = label.split(':');
                    let tag = parts.next().unwrap();
                    let edge = parts.next();
                    let mut label_parts = tag.split('=');
                    let label = label_parts.next().unwrap();
                    let annotation = label_parts.next();
                    Ok((label, edge, annotation))
                }
                PTBFormat::Simple => Ok((label, None, None)),
            }
        } else {
            Err(format_err!(
                "Node did not start with label {}",
                pair.as_str()
            ))
        }
    }
}

/// `PTBLineFormat`.
///
/// This enum specifies whether the trees are encoded in single-line or multi-line format.
pub enum PTBLineFormat {
    SingleLine,
    MultiLine,
}

/// Iterator over trees in PTB format file.
pub struct PTBTreeIter<R> {
    inner: Lines<R>,
    line_format: PTBLineFormat,
    format: PTBFormat,
}

impl<R> Iterator for PTBTreeIter<R>
where
    R: BufRead,
{
    type Item = Result<Tree, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if let PTBLineFormat::SingleLine = self.line_format {
            while let Some(line) = self.inner.next() {
                let line = match line {
                    Ok(line) => line,
                    Err(err) => return Some(Err(err.into())),
                };
                if line.starts_with('%') {
                    continue;
                }
                return Some(self.format.string_to_tree(&line));
            }
            return None;
        } else {
            let mut buffer = String::new();
            let mut open = 0;
            while let Some(line) = self.inner.next() {
                let line = match line {
                    Ok(line) => line,
                    Err(err) => return Some(Err(err.into())),
                };
                if (line.starts_with('%') && buffer.is_empty()) || line.is_empty() {
                    continue;
                }
                let (line_open, line_closed) = count_pars(&line);
                open += line_open;
                open -= line_closed;
                buffer.push_str(line.as_str());;
                if open == 0 {
                    return Some(self.format.string_to_tree(&buffer));
                }
            }
        }
        None
    }
}

impl<R> PTBTreeIter<R>
where
    R: BufRead,
{
    /// Constructs a new tree iterator.
    pub fn new(read: R, format: PTBFormat, line_format: PTBLineFormat) -> Self {
        PTBTreeIter {
            inner: read.lines(),
            format,
            line_format,
        }
    }
}

fn count_pars(line: &str) -> (usize, usize) {
    let mut open = 0;
    let mut closed = 0;
    for c in line.chars() {
        if c == '(' {
            open += 1
        }
        if c == ')' {
            closed += 1
        }
    }
    (open, closed)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use petgraph::prelude::NodeIndex;
    use petgraph::stable_graph::StableGraph;

    use crate::io::{
        ptb::{PTBFormat, PTBLineFormat, PTBTreeIter},
        ReadTree, WriteTree,
    };
    use crate::{Edge, Node, NonTerminal, Projectivity, Span, Terminal, Tree};

    #[test]
    pub fn test_multiline() {
        let input = File::open("testdata/single_multiline.ptb").unwrap();
        let mut reader = PTBTreeIter::new(
            BufReader::new(input),
            PTBFormat::TueBa,
            PTBLineFormat::MultiLine,
        );
        let tree = reader.next().unwrap().unwrap();
        assert_eq!(tree.n_terminals(), 4);
    }

    #[test]
    pub fn test_with_edge_labels() {
        let input = "(NX:edge (NN Nounphrase) (PX:edge (PP on) (NX:another (DET a ) \
                     (ADJ single) (NX line))))";
        PTBFormat::TueBa.string_to_tree(input).unwrap();
    }

    #[test]
    fn test_single_terminal() {
        let input = "(T t)";
        let t = PTBFormat::Simple.string_to_tree(input).unwrap();
        assert_eq!(input, PTBFormat::Simple.tree_to_string(&t).unwrap())
    }

    #[test]
    pub fn test_without_edge_labels() {
        let input = "(NX (NN Nounphrase) (PX (PP on) (NX (DET a ) (ADJ single) (NX line))))";
        PTBFormat::TueBa.string_to_tree(input).unwrap();
    }

    #[test]
    pub fn readable_test() {
        let l = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (SEC:label (TERM1 t1)) (TERM t))";
        let mut cmp_graph = StableGraph::new();

        let term1 = Terminal::new("t1", "TERM1", 0);
        let term1 = cmp_graph.add_node(Node::Terminal(term1));

        let term2 = Terminal::new("t2", "TERM2", 1);
        let term2 = cmp_graph.add_node(Node::Terminal(term2));

        let nt = NonTerminal::new("FIRST", Span::new_continuous(0, 2));
        let first = cmp_graph.add_node(Node::NonTerminal(nt));
        cmp_graph.add_edge(first, term1, Edge::default());
        cmp_graph.add_edge(first, term2, Edge::default());

        let term3 = Terminal::new("t1", "TERM1", 2);
        let term3 = cmp_graph.add_node(Node::Terminal(term3));

        let nt2 = NonTerminal::new("SEC", Span::new_continuous(2, 3));
        let sec = cmp_graph.add_node(Node::NonTerminal(nt2));
        cmp_graph.add_edge(sec, term3, Edge::default());

        let term4 = Terminal::new("t", "TERM", 3);
        let term4 = cmp_graph.add_node(Node::Terminal(term4));

        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 4));
        let root = cmp_graph.add_node(Node::NonTerminal(root));

        cmp_graph.add_edge(root, first, Edge::default());
        cmp_graph.add_edge(root, sec, Edge::from(Some("label")));
        cmp_graph.add_edge(root, term4, Edge::default());
        let tree2 = Tree::new(cmp_graph, 4, NodeIndex::new(6), Projectivity::Projective);
        let tree = PTBFormat::TueBa.string_to_tree(l).unwrap();
        assert_eq!(tree, tree2);

        assert_eq!(4, tree.n_terminals());
    }

    #[test]
    #[should_panic]
    pub fn empty_line() {
        let l = "";
        PTBFormat::TueBa.string_to_tree(l).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn closed_to_early() {
        // Tree is closed after Term1 t1))))
        let l = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (SEC:label (TERM1 t1)))) (TERM t))";
        PTBFormat::TueBa.string_to_tree(l).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn missing_par() {
        // missing parenthesis at end of string
        let l = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (SEC:label (TERM1 t1)) (TERM t)";
        PTBFormat::TueBa.string_to_tree(l).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn second_tree() {
        // missing parenthesis at end of string
        let l = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (SEC:label (TERM1 t1)) (TERM t)) (ROOT (Second tree))";
        PTBFormat::TueBa.string_to_tree(l).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn illegal_char() {
        // parenthesis as terminal in TERM1 node
        let l = "(ROOT (FIRST (TERM1 () (TERM2 t2)) (SEC:label (TERM1 t1)) (TERM t))";
        PTBFormat::TueBa.string_to_tree(l).unwrap();
    }

    #[test]
    pub fn write_test() {
        let tree = PTBFormat::TueBa
            .string_to_tree("(VROOT:- (NP=NE:- (N:- n)) (VP:HD (V:HD v)))")
            .unwrap();

        assert_eq!(
            PTBFormat::TueBa.tree_to_string(&tree).unwrap(),
            "(VROOT:--(NP=NE:-(N:- n))(VP:HD(V:HD v)))"
        );

        assert_eq!(
            PTBFormat::PTB.tree_to_string(&tree).unwrap(),
            "(VROOT (NP-NE (N n)) (VP (V v)))"
        );

        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree).unwrap(),
            "(VROOT (NP (N n)) (VP (V v)))"
        )
    }
}
