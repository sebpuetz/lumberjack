use crate::node::Node;
use crate::tree::Tree;
use failure::Error;
use petgraph::prelude::Direction::*;
use petgraph::prelude::NodeIndex;
use petgraph::visit::EdgeRef;

/// Formatter specifying how a `Tree` is turned into a `String`.
///
/// The struct stores information on whether edges and annotations are included, how those
/// are seperated from the node label and how nodes are seperated.
///
/// The `Default` implementation returns a formatter that seperates nodes by a single whitespace
/// and does not include edges or annotation.
///
/// At this point, multi-line output is not supported.
///
/// Parentheses are replaced by `"LBR"` and `"RBR"` respectively.
#[derive(Clone)]
pub struct PTBFormatter {
    edge_sep: Option<String>,
    annotation_sep: Option<String>,
    node_sep: String,
}

impl Default for PTBFormatter {
    fn default() -> PTBFormatter {
        PTBFormatter::new(" ", None, None)
    }
}

impl PTBFormatter {
    /// Construct a new `PTBFormatter`.
    ///
    /// * `node_sep` specifies how nodes are seperated
    /// * `edge_sep` specifies whether edges are included and how label and edge are seperated.
    /// * `annotation_sep` specifies whether annotations are included and how label and edge are
    ///                    seperated.
    ///
    /// Edges and annotation will not be included if the respective parameter is `None`.
    pub fn new(
        node_sep: impl Into<String>,
        edge_sep: Option<String>,
        annotation_sep: Option<String>,
    ) -> Self {
        PTBFormatter {
            edge_sep,
            annotation_sep,
            node_sep: node_sep.into(),
        }
    }

    /// Returns linearized (bracketed) representation of a `Tree`.
    ///
    /// Parentheses `"("` and `")"` are replaced by `"LBR"` and `"RBR"` respectively.
    ///
    /// Returns `Error` if the `Tree` is nonprojective.
    pub fn format(&self, sentence: &Tree) -> Result<String, Error> {
        if sentence.projective() {
            Ok(self.format_sub_tree(sentence, sentence.root(), None))
        } else {
            Err(format_err!("Can not format non-projective tree."))
        }
    }

    fn format_sub_tree(&self, sentence: &Tree, position: NodeIndex, edge: Option<&str>) -> String {
        let sent = sentence.graph();

        match &sent[position] {
            Node::Terminal(terminal) => self.fmt_term(terminal.form(), terminal.pos(), edge),
            Node::NonTerminal(nt) => {
                let mut nodes: Vec<_> = sent.edges_directed(position, Outgoing).collect::<Vec<_>>();
                // sort child nodes by covered span
                nodes.sort_by(|edge_ref_1, edge_ref_2| {
                    let span_1 = sentence.graph()[edge_ref_1.target()].span();
                    let span_2 = sentence.graph()[edge_ref_2.target()].span();
                    span_1.cmp(&span_2)
                });
                let mut sub_tree_rep = Vec::with_capacity(nodes.len());
                sub_tree_rep.push(self.fmt_inner(nt.label(), edge, nt.annotation()));
                sub_tree_rep.extend(nodes.into_iter().map(|edge_ref| {
                    self.format_sub_tree(sentence, edge_ref.target(), edge_ref.weight().label())
                }));
                format!("({})", sub_tree_rep.join(&self.node_sep))
            }
        }
    }

    fn fmt_inner(&self, label: &str, edge: Option<&str>, annotation: Option<&str>) -> String {
        let mut representation = label.to_string();
        if let Some(ref annotation_sep) = self.annotation_sep {
            if let Some(annotation) = annotation {
                representation.push_str(annotation_sep);
                representation.push_str(annotation);
            }
        }
        let edge = edge.unwrap_or("--");

        if let Some(ref edge_sep) = self.edge_sep {
            representation.push_str(edge_sep);
            representation.push_str(edge);
        }
        representation
    }

    fn fmt_term(&self, form: &str, pos: &str, edge: Option<&str>) -> String {
        let edge = edge.unwrap_or("--");
        let pos = pos.replace("(", "LBR").replace(")", "RBR");
        let form = form.replace("(", "LBR").replace(")", "RBR");
        if let Some(ref sep) = self.edge_sep {
            format!("({}{}{} {})", pos, sep, edge, form)
        } else {
            format!("({} {})", pos, form)
        }
    }

    /// Tries to construct a `PTBFormatter` from `&str`.
    ///
    /// Supported types are `"tuebav2"` and `"simple"`.
    ///
    /// * `"simple"` will not include annotations or edges and seperates nodes by a space.
    /// * `"tuebav2"` will include annotations and edges and doesn't seperate nodes.
    ///               Annotation and label will be seperated by `"="`, edge and label by `":"`.
    pub fn try_from_str(s: &str) -> Result<Self, Error> {
        let s_lower = s.to_lowercase();
        match s_lower.as_str() {
            "tuebav2" => Ok(PTBFormatter::new("", Some(":".into()), Some("=".into()))),
            "simple" => Ok(PTBFormatter::new(" ", None, None)),
            _ => Err(format_err!("Unknown format: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ptb_reader::PTBBuilder;
    use crate::ptb_writer::PTBFormatter;

    #[test]
    pub fn test() {
        let tree = PTBBuilder::default()
            .ptb_to_tree("(VROOT:- (NP:- (N:- n)) (VP:HD (V:HD v)))")
            .unwrap();

        let mut fmt = PTBFormatter::try_from_str("tuebav2").unwrap();

        assert_eq!(
            fmt.format(&tree).unwrap(),
            "(VROOT:--(NP:-(N:- n))(VP:HD(V:HD v)))"
        );

        fmt.node_sep = " ".to_string();
        assert_eq!(
            fmt.format(&tree).unwrap(),
            "(VROOT:-- (NP:- (N:- n)) (VP:HD (V:HD v)))"
        );

        fmt.edge_sep = None;
        assert_eq!(fmt.format(&tree).unwrap(), "(VROOT (NP (N n)) (VP (V v)))")
    }
}
