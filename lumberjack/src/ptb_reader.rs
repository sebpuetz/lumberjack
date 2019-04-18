use std::io::BufRead;
use std::io::Lines;

use failure::Error;
use pest::iterators::Pair;
use pest::Parser;
use petgraph::stable_graph::{NodeIndex, StableGraph};

use crate::edge::Edge;
use crate::node::{NTBuilder, Node, TerminalBuilder};
use crate::span::Span;
use crate::tree::{Projectivity, Tree};

// dummy struct required by pest
#[derive(Parser)]
#[grammar = "ptb.pest"]
struct PTBParser;

/// `PTBBuilder`
///
/// This struct is used to construct a `Tree` from stringly representations.
pub struct PTBBuilder {
    edge_sep: String,
    annotation_sep: String,
}

impl Default for PTBBuilder {
    fn default() -> Self {
        PTBBuilder {
            edge_sep: ":".into(),
            annotation_sep: "=".into(),
        }
    }
}

impl PTBBuilder {
    pub fn new(edge_sep: impl Into<String>, annotation_sep: impl Into<String>) -> PTBBuilder {
        PTBBuilder {
            edge_sep: edge_sep.into(),
            annotation_sep: annotation_sep.into(),
        }
    }

    /// Constructs new tree from a `&str` containing wellformed PTB.
    pub fn ptb_to_tree(&self, line: &str) -> Result<Tree, Error> {
        let mut graph = StableGraph::new();
        let mut terminals = Vec::new();
        let mut parsed_line = PTBParser::parse(Rule::tree, line)?;
        let (_, root, _) =
            self.parse_value(parsed_line.next().unwrap(), &mut graph, &mut terminals)?;
        Ok(Tree::new(
            graph,
            terminals.len(),
            root,
            Projectivity::Projective,
        ))
    }

    // this method traverses the linearized tree and builds a StableGraph
    fn parse_value(
        &self,
        pair: Pair<Rule>,
        g: &mut StableGraph<Node, Edge>,
        terminals: &mut Vec<NodeIndex>,
    ) -> Result<(Span, NodeIndex, Edge), Error> {
        match pair.as_rule() {
            Rule::nonterminal => {
                let mut pairs = pair.into_inner();
                // first rule after matching nonterminal will always be the label of the inner node
                let (label, edge, annotation) = self.process_label(pairs.next().unwrap())?;

                // collect children
                let mut spans = Vec::new();
                let mut children = Vec::new();
                for inner_pair in pairs {
                    let (span, child_idx, edge) = self.parse_value(inner_pair, g, terminals)?;
                    spans.push(span);
                    children.push((child_idx, edge));
                }

                // safe to unwrap as the grammar rejects strings with Nonterminals without child nodes
                // first.0 is lowest idx covered, last.1 highest idx
                let span = Span::new_continuous(
                    spans.first().unwrap().lower(),
                    spans.last().unwrap().upper(),
                );
                let nt = NTBuilder::new(label)
                    .span(span.clone())
                    .annotation(annotation);
                let node = Node::NonTerminal(nt.try_into_nt()?);
                let idx = g.add_node(node);
                for (child_idx, edge) in children {
                    g.add_edge(idx, child_idx, edge);
                }
                Ok((span, idx, edge.into()))
            }
            Rule::preterminal => {
                let n_terminals = terminals.len();
                let (edge, pos, form) = self.process_preterminal(pair)?;
                let span = Span::new_continuous(n_terminals, n_terminals + 1);

                let terminal = Node::Terminal(
                    TerminalBuilder::new(form, pos, span.clone()).try_into_terminal()?,
                );
                let idx = g.add_node(terminal);
                terminals.push(idx);
                Ok((span, idx, edge.into()))
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
            let mut parts = label.split(&self.edge_sep);
            let tag = parts.next().unwrap();
            // splits label at first index of given annotation seperator.
            let (tag, annotation) = if let Some(idx) = tag.find(&self.annotation_sep) {
                let (tag, annotation) = tag.split_at(idx);
                (tag, annotation.into())
            } else {
                (tag, None)
            };
            let edge = parts.next();
            Ok((tag, edge, annotation))
        } else {
            Err(format_err!(
                "Node did not start with label {}",
                pair.as_str()
            ))
        }
    }
}

/// `PTBFormat`.
///
/// This enum specifies whether the trees are encoded in single-line or multi-line format.
pub enum PTBFormat {
    SingleLine,
    MultiLine,
}

/// Iterator over trees in PTB format file.
pub struct PTBTreeIter<R> {
    inner: Lines<R>,
    multiline: PTBFormat,
    builder: PTBBuilder,
}

impl<R> Iterator for PTBTreeIter<R>
    where
        R: BufRead,
{
    type Item = Result<Tree, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if let PTBFormat::SingleLine = self.multiline {
            if let Some(line) = self.inner.next() {
                let line = match line {
                    Ok(line) => line,
                    Err(err) => return Some(Err(err.into())),
                };
                return Some(self.builder.ptb_to_tree(&line));
            } else {
                return None;
            }
        } else {
            let mut buffer = String::new();
            let mut open = 0;
            while let Some(line) = self.inner.next() {
                let line = match line {
                    Ok(line) => line,
                    Err(err) => return Some(Err(err.into())),
                };
                if line.is_empty() {
                    continue;
                }
                let (line_open, line_closed) = count_pars(&line);
                open += line_open;
                open -= line_closed;
                buffer.push_str(line.as_str());;
                if open == 0 {
                    return Some(self.builder.ptb_to_tree(&buffer));
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
    pub fn new(read: R, builder: PTBBuilder, format: PTBFormat) -> Self {
        PTBTreeIter {
            inner: read.lines(),
            builder,
            multiline: format,
        }
    }

    pub fn new_with_defaults(read: R) -> Self {
        PTBTreeIter {
            inner: read.lines(),
            builder: PTBBuilder::default(),
            multiline: PTBFormat::SingleLine,
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

    use failure::Error;
    use petgraph::prelude::NodeIndex;
    use petgraph::stable_graph::StableGraph;

    use super::PTBBuilder;
    use crate::edge::Edge;
    use crate::node::{Node, NonTerminal, TerminalBuilder};
    use crate::ptb_reader::{PTBFormat, PTBTreeIter};
    use crate::span::Span;
    use crate::tree::{Projectivity, Tree};

    #[test]
    pub fn test_multiline() {
        let input = File::open("testdata/single_multiline.ptb").unwrap();
        let builder = PTBBuilder::default();
        let mut reader = PTBTreeIter::new(BufReader::new(input), builder, PTBFormat::MultiLine);
        let tree = reader.next().unwrap().unwrap();
        assert_eq!(tree.n_terminals(), 4);
    }

    #[test]
    pub fn test_with_edge_labels() {
        let input = "(NX:edge (NN Nounphrase) (PX:edge (PP on) (NX:another (DET a ) \
        (ADJ single) (NX line))))";
        let builder = PTBBuilder::default();
        builder.ptb_to_tree(input).unwrap();
    }

    #[test]
    pub fn test_without_edge_labels() {
        let input = "(NX (NN Nounphrase) (PX (PP on) (NX (DET a ) (ADJ single) (NX line))))";
        let builder = PTBBuilder::default();
        builder.ptb_to_tree(input).unwrap();
    }

    #[test]
    pub fn readable_test() -> Result<(), Error> {
        let l = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (SEC:label (TERM1 t1)) (TERM t))";
        let mut cmp_graph = StableGraph::new();

        let span1 = Span::new_continuous(0, 1);
        let term1 = TerminalBuilder::new("t1", "TERM1", span1);
        let term1 = cmp_graph.add_node(Node::Terminal(term1.try_into_terminal()?));

        let span2 = Span::new_continuous(1, 2);
        let term2 = TerminalBuilder::new("t2", "TERM2", span2);
        let term2 = cmp_graph.add_node(Node::Terminal(term2.try_into_terminal()?));

        let nt = NonTerminal::new("FIRST", Span::new_continuous(0, 2));
        let first = cmp_graph.add_node(Node::NonTerminal(nt));
        cmp_graph.add_edge(first, term1, Edge::default());
        cmp_graph.add_edge(first, term2, Edge::default());

        let span3 = Span::new_continuous(2, 3);
        let term3 = TerminalBuilder::new("t1", "TERM1", span3);
        let term3 = cmp_graph.add_node(Node::Terminal(term3.try_into_terminal()?));

        let nt2 = NonTerminal::new("SEC", Span::new_continuous(2, 3));
        let sec = cmp_graph.add_node(Node::NonTerminal(nt2));
        cmp_graph.add_edge(sec, term3, Edge::default());

        let span4 = Span::new_continuous(3, 4);
        let term4 = TerminalBuilder::new("t", "TERM", span4);
        let term4 = cmp_graph.add_node(Node::Terminal(term4.try_into_terminal()?));

        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 4));
        let root = cmp_graph.add_node(Node::NonTerminal(root));

        cmp_graph.add_edge(root, first, Edge::default());
        cmp_graph.add_edge(root, sec, Edge::from(Some("label")));
        cmp_graph.add_edge(root, term4, Edge::default());
        let tree2 = Tree::new(cmp_graph, 4, NodeIndex::new(6), Projectivity::Projective);
        let tree = PTBBuilder::default().ptb_to_tree(l)?;
        assert_eq!(tree, tree2);

        assert_eq!(4, tree.n_terminals());
        Ok(())
    }

    #[test]
    #[should_panic]
    pub fn empty_line() {
        let l = "";
        PTBBuilder::default().ptb_to_tree(l).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn closed_to_early() {
        // Tree is closed after Term1 t1))))
        let l = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (SEC:label (TERM1 t1)))) (TERM t))";
        PTBBuilder::default().ptb_to_tree(l).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn missing_par() {
        // missing parenthesis at end of string
        let l = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (SEC:label (TERM1 t1)) (TERM t)";
        PTBBuilder::default().ptb_to_tree(l).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn second_tree() {
        // missing parenthesis at end of string
        let l = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (SEC:label (TERM1 t1)) (TERM t)) (ROOT (Second tree))";
        PTBBuilder::default().ptb_to_tree(l).unwrap();
    }

    #[test]
    #[should_panic]
    pub fn illegal_char() {
        // parenthesis as terminal in TERM1 node
        let l = "(ROOT (FIRST (TERM1 () (TERM2 t2)) (SEC:label (TERM1 t1)) (TERM t))";
        PTBBuilder::default().ptb_to_tree(l).unwrap();
    }
}
