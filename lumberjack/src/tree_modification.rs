use failure::Error;
use petgraph::prelude::DfsPostOrder;

use crate::util::{Climber, LabelSet};
use crate::{Node, NonTerminal, Projectivity, Span, Tree};

/// Trait to annotate Part of Speech tags.
///
/// Returns:
/// * `Error` if length of `self` and `pos_iter` don't match.
/// * `Ok` otherwise.
pub trait AnnotatePOS {
    fn annotate_pos<S>(&mut self, pos_iter: impl IntoIterator<Item = S>) -> Result<(), Error>
    where
        S: Into<String>;
}

impl AnnotatePOS for Tree {
    fn annotate_pos<S>(&mut self, pos_iter: impl IntoIterator<Item = S>) -> Result<(), Error>
    where
        S: Into<String>,
    {
        let terminals = self.terminals().collect::<Vec<_>>();
        let mut pos_iter = pos_iter.into_iter();
        for terminal in terminals {
            if let Some(pos) = pos_iter.next() {
                self[terminal].terminal_mut().unwrap().set_label(pos);
            } else {
                return Err(format_err!("Not enough POS tags were provided"));
            }
        }
        if pos_iter.next().is_some() {
            return Err(format_err!(
                "Number of POS tags is greater than number of terminals."
            ));
        }
        Ok(())
    }
}

/// Trait specifying methods to modify tree structure.
pub trait TreeOps {
    /// Insert an intermediate node above terminals.
    ///
    /// If a terminal is not dominated by a node with label matched by `tag_set` a new non-terminal
    /// node is inserted above with a specified label. Runs of terminals whose parent node is not
    /// matched by `tag_set` are collected under a single new node.
    fn insert_intermediate(
        &mut self,
        tag_set: &LabelSet,
        insertion_label: &str,
    ) -> Result<(), Error>;

    /// Remove non-terminals not matched by `tag_set`.
    ///
    /// The root node will never be removed. Root node is determined by the `tree::is_root()`
    /// method. Detached material is re-attached above the removed node.
    fn filter_nonterminals(&mut self, tag_set: &LabelSet) -> Result<(), Error>;
}

impl TreeOps for Tree {
    fn insert_intermediate(
        &mut self,
        tag_set: &LabelSet,
        insertion_label: &str,
    ) -> Result<(), Error> {
        let terminals = self.terminals().collect::<Vec<_>>();
        let mut prev_attachment = None;
        for (position, terminal) in terminals.into_iter().enumerate() {
            let (parent, edge_id) = self
                .parent(terminal)
                .ok_or_else(|| format_err!("Terminal without parent:\n{}", self[terminal]))?;

            if tag_set.matches(self[parent].label()) {
                continue;
            }

            let weight = self.graph_mut().remove_edge(edge_id).unwrap();
            if let Some((prev_position, prev_insert)) = prev_attachment {
                if prev_position == position - 1 && self.parent(prev_insert).unwrap().0 == parent {
                    self.graph_mut().add_edge(prev_insert, terminal, weight);
                    self[prev_insert].extend_span()?;
                    prev_attachment = Some((position, prev_insert));
                    continue;
                }
            }

            let span = self.graph()[terminal].span().clone();
            let nt = Node::NonTerminal(NonTerminal::new(insertion_label, span));
            let inserted_idx = self.graph_mut().add_node(nt);
            self.graph_mut()
                .add_edge(parent, inserted_idx, weight.clone());
            self.graph_mut().add_edge(inserted_idx, terminal, weight);
            prev_attachment = Some((position, inserted_idx))
        }
        Ok(())
    }

    fn filter_nonterminals(&mut self, tag_set: &LabelSet) -> Result<(), Error> {
        // divide indices into keep- and delete-list, root is excluded as we don't want to break the
        // tree. Collecting is necessary because .node_indices() borrows from the graph
        let (keep, delete) = self
            .graph()
            .node_indices()
            .filter(|node| *node != self.root())
            .fold((Vec::new(), Vec::new()), |(mut keep, mut delete), node| {
                if let Node::NonTerminal(ref nt) = self[node] {
                    if tag_set.matches(nt.label()) {
                        keep.push(node)
                    } else {
                        delete.push(node)
                    }
                } else {
                    keep.push(node)
                }

                (keep, delete)
            });

        for node in keep {
            // get id of the incoming edge of the node currently looking for re-attachment, since
            // EdgeReference borrows from graph only clone edge index
            let (_, id) = self.parent(node).ok_or_else(|| {
                format_err!("Non-root node without incoming edge: {}", self[node])
            })?;

            // climb up until field specified in tag_set or root is found
            let mut climber = Climber::new(node);
            while let Some(parent_idx) = climber.next(&self) {
                let parent = self[parent_idx]
                    .nonterminal()
                    .ok_or_else(|| format_err!("Terminal as parent: {}", self[parent_idx]))?;
                if tag_set.matches(parent.label()) || parent_idx == self.root() {
                    // safe to unwrap, id is guaranteed to be valid (line 138)
                    let weight = self.graph_mut().remove_edge(id).unwrap();
                    self.graph_mut().update_edge(parent_idx, node, weight);
                    break;
                }
            }
        }
        for node in delete {
            self.graph_mut().remove_node(node);
        }
        Ok(())
    }
}

/// Projectivization Trait.
///
/// Projectivization is done by re-attaching the non-projective content at the highest point
/// allowing non-crossing edges while maintaining the linear order of the sentence.
pub trait Projectivize {
    fn projectivize(&mut self);
}

impl Projectivize for Tree {
    fn projectivize(&mut self) {
        if !self.projective() {
            let terminals = self.terminals().collect::<Vec<_>>();;
            let mut dfs = DfsPostOrder::new(self.graph(), self.root());
            let mut log = vec![None; terminals.len()];

            while let Some(attachment_point_candidate) = dfs.next(self.graph()) {
                let span = if let Node::NonTerminal(nt) = &self[attachment_point_candidate] {
                    if let Span::Discontinuous(span) = nt.span() {
                        span.to_owned()
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };

                let mut skips = span.skips().to_owned();
                while let Some(&skipped) = skips.iter().next() {
                    // check if terminal at idx skipped has already been reattached. We're
                    // doing a postorder traversal, generally if something has been
                    // reattached it ends up in the correct place, unless there are
                    // multiple non-terminals covering the span. In that case, the correct
                    // attachment is that non-terminal starting at the higher index.
                    if let Some(claimed) = log[skipped] {
                        if claimed >= span.lower() {
                            // remove skipped idx so the loop can terminate
                            skips.remove(&skipped);
                            continue;
                        }
                    }

                    let mut climber = Climber::new(terminals[skipped]);

                    // cheap clone since terminal span is continuous (actually copy)
                    let mut reattach_span = self[terminals[skipped]].span().clone();
                    // keep track of which node is used to reattach non-projective material
                    let mut attachment_handle = terminals[skipped];

                    'a: while let Some(attachment_handle_candidate) = climber.next(&self) {
                        // spans being eq implies unary chain, keep higher node as handle
                        // for reattachment
                        if self[attachment_handle_candidate].span() != &reattach_span {
                            for covered in self[attachment_handle_candidate].span() {
                                if !span.skips().contains(&covered) {
                                    for covered in self[attachment_handle].span() {
                                        skips.remove(&covered);
                                        log[covered] = Some(span.lower());
                                    }
                                    let rm_edge = self.parent(attachment_handle).unwrap().1;
                                    let edge = self.graph_mut().remove_edge(rm_edge).unwrap();
                                    self.graph_mut().update_edge(
                                        attachment_point_candidate,
                                        attachment_handle,
                                        edge,
                                    );
                                    break 'a;
                                }
                            }
                            reattach_span = self[attachment_handle_candidate].span().clone();
                        }
                        attachment_handle = attachment_handle_candidate;
                    }
                }
                self[attachment_point_candidate]
                    .nonterminal_mut()
                    .unwrap()
                    .set_span(Span::new_continuous(span.lower(), span.upper()));
            }
            self.set_projectivity(Projectivity::Projective);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::prelude::StableGraph;

    use super::{AnnotatePOS, TreeOps};
    use crate::io::{PTBFormat, ReadTree, WriteTree};
    use crate::tree_modification::LabelSet;
    use crate::{Edge, Node, NonTerminal, Projectivity, Span, Terminal, Tree};

    #[test]
    pub fn annotate_pos() {
        let input = "(NX (NN Nounphrase) (PX (PP on) (NX (DET a) (ADJ single) (NX line))))";
        let mut tree = PTBFormat::TueBa.string_to_tree(input).unwrap();
        let pos = vec!["A", "B", "C", "D", "E"];
        tree.annotate_pos(pos).unwrap();
        let target = "(NX (A Nounphrase) (PX (B on) (NX (C a) (D single) (E line))))";
        assert_eq!(target, PTBFormat::Simple.tree_to_string(&tree).unwrap());

        let pos_too_short = vec!["A"];
        assert!(tree.annotate_pos(pos_too_short).is_err());
        let pos_too_long = vec!["A", "B", "C", "D", "E", "F"];
        assert!(tree.annotate_pos(pos_too_long).is_err());
    }

    #[test]
    fn filter_nonproj() {
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let first = NonTerminal::new("L", Span::from_vec(vec![0, 2]).unwrap());
        let first_idx = g.add_node(Node::NonTerminal(first));
        g.add_edge(root_idx, first_idx, Edge::default());
        let second = NonTerminal::new("L1", Span::new_continuous(1, 2));
        let second_idx = g.add_node(Node::NonTerminal(second));
        g.add_edge(root_idx, second_idx, Edge::default());
        let third = NonTerminal::new("L", Span::new_continuous(3, 4));
        let third_idx = g.add_node(Node::NonTerminal(third));
        g.add_edge(root_idx, third_idx, Edge::default());
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(first_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(second_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", 2);
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", 3);
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(third_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", 4);
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());

        let tree = Tree::new(g, 5, root_idx, Projectivity::Nonprojective);
        let mut tags = HashSet::new();
        tags.insert("L".into());
        let mut filtered_tree = tree.clone();
        filtered_tree
            .filter_nonterminals(&LabelSet::Positive(tags))
            .unwrap();

        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let first = NonTerminal::new("L", Span::from_vec(vec![0, 2]).unwrap());
        let first_idx = g.add_node(Node::NonTerminal(first));
        g.add_edge(root_idx, first_idx, Edge::default());
        let third = NonTerminal::new("L", Span::new_continuous(3, 4));
        let third_idx = g.add_node(Node::NonTerminal(third));
        g.add_edge(root_idx, third_idx, Edge::default());
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(first_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(root_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", 2);
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", 3);
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(third_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", 4);
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());
        let target = Tree::new(g, 5, root_idx, Projectivity::Nonprojective);
        assert_eq!(target, filtered_tree);

        let mut tags = HashSet::new();
        tags.insert("L1".into());
        let mut filtered_tree = tree.clone();
        filtered_tree
            .filter_nonterminals(&LabelSet::Positive(tags))
            .unwrap();
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let second = NonTerminal::new("L1", Span::new_continuous(1, 2));
        let second_idx = g.add_node(Node::NonTerminal(second));
        g.add_edge(root_idx, second_idx, Edge::default());
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(root_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(second_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", 2);
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(root_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", 3);
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(root_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", 4);
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());
        let target = Tree::new(g, 5, root_idx, Projectivity::Projective);
        assert_eq!(target, filtered_tree);
    }

    #[test]
    fn insert_unks_nonproj() {
        // non projective tree, where one inserted node collects two nodes.
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let first = NonTerminal::new("L", Span::from_vec(vec![0, 2]).unwrap());
        let first_idx = g.add_node(Node::NonTerminal(first));
        g.add_edge(root_idx, first_idx, Edge::default());
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(first_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(root_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", 2);
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", 3);
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(root_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", 4);
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());
        let mut set = HashSet::new();
        set.insert("L".into());
        let mut unk_tree = Tree::new(g, 5, root_idx, Projectivity::Nonprojective);
        unk_tree
            .insert_intermediate(&LabelSet::Positive(set), "UNK")
            .unwrap();

        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let first = NonTerminal::new("L", Span::from_vec(vec![0, 2]).unwrap());
        let first_idx = g.add_node(Node::NonTerminal(first));
        g.add_edge(root_idx, first_idx, Edge::default());
        let first_unk = NonTerminal::new("UNK", Span::new_continuous(1, 2));
        let first_unk_idx = g.add_node(Node::NonTerminal(first_unk));
        g.add_edge(root_idx, first_unk_idx, Edge::default());
        let second_unk = NonTerminal::new("UNK", Span::new_continuous(3, 5));
        let second_unk_idx = g.add_node(Node::NonTerminal(second_unk));
        g.add_edge(root_idx, second_unk_idx, Edge::default());
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(first_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(first_unk_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", 2);
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", 3);
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(second_unk_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", 4);
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(second_unk_idx, term5_idx, Edge::default());
        let target = Tree::new(g, 5, root_idx, Projectivity::Nonprojective);
        assert_eq!(target, unk_tree);
    }

    #[test]
    fn project_node_indices() {
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("FIRST", Span::new_continuous(0, 2));
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term3 = Terminal::new("t3", "TERM3", 2);
        let second = NonTerminal::new("SECOND", 3);
        let term4 = Terminal::new("t4", "TERM4", 3);
        let term5 = Terminal::new("t5", "TERM5", 4);
        let root_idx = g.add_node(Node::NonTerminal(root));
        let first_idx = g.add_node(Node::NonTerminal(first));
        let term1_idx = g.add_node(Node::Terminal(term1));
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(root_idx, first_idx, Edge::default());
        g.add_edge(first_idx, term1_idx, Edge::default());
        g.add_edge(first_idx, term2_idx, Edge::default());
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(root_idx, term3_idx, Edge::default());
        let second_idx = g.add_node(Node::NonTerminal(second));
        g.add_edge(root_idx, second_idx, Edge::default());
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(second_idx, term4_idx, Edge::default());
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());

        let tree = Tree::new(g, 5, root_idx, Projectivity::Projective);
        let mut tags = HashSet::new();
        tags.insert("FIRST".into());
        let indices = tree.project_nt_indices(&LabelSet::Positive(tags));
        let target = vec![first_idx, first_idx, root_idx, root_idx, root_idx];
        assert_eq!(indices, target)
    }

    #[test]
    fn project_node_indices_nonprojective() {
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("FIRST", Span::from_vec(vec![0, 2]).unwrap());
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term3 = Terminal::new("t3", "TERM3", 2);
        let second = NonTerminal::new("SECOND", Span::new_continuous(3, 4));
        let term4 = Terminal::new("t4", "TERM4", 3);
        let term5 = Terminal::new("t5", "TERM5", 4);
        let root_idx = g.add_node(Node::NonTerminal(root));
        let first_idx = g.add_node(Node::NonTerminal(first));
        let term1_idx = g.add_node(Node::Terminal(term1));
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(root_idx, first_idx, Edge::default());
        g.add_edge(first_idx, term1_idx, Edge::default());
        g.add_edge(root_idx, term2_idx, Edge::default());
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let second_idx = g.add_node(Node::NonTerminal(second));
        g.add_edge(root_idx, second_idx, Edge::default());
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(second_idx, term4_idx, Edge::default());
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());

        let tree = Tree::new(g, 5, root_idx, Projectivity::Nonprojective);
        let mut tags = HashSet::new();
        tags.insert("FIRST".into());
        let indices = tree.project_nt_indices(&LabelSet::Positive(tags));
        let target = vec![first_idx, root_idx, first_idx, root_idx, root_idx];
        assert_eq!(indices, target)
    }
}
