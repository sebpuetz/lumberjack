use failure::Error;
use petgraph::prelude::{DfsPostOrder, NodeIndex};
use petgraph::visit::{VisitMap, Visitable};

use crate::util::Climber;
use crate::Tree;

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

/// Trait specifying methods to modify trees.
pub trait TreeOps {
    /// Annotate the parent tag as a feature.
    ///
    /// Annotates the tag of each terminal's parent as a feature.
    ///
    /// Returns `Error` if the tree contains `Terminal`s without a parent node.
    fn annotate_parent_tag(&mut self, feature_name: &str) -> Result<(), Error>;

    /// Insert an intermediate node above terminals.
    ///
    /// If a terminal is not dominated by a node with label matched by `tag_set` a new non-terminal
    /// node is inserted above with a specified label. Runs of terminals whose parent node is not
    /// matched by `tag_set` are collected under a single new node.
    ///
    /// Returns `Error` if the tree contains `Terminal`s without a parent node.
    ///
    /// E.g.:
    /// ```
    /// use lumberjack::{Tree, TreeOps};
    ///
    /// /// Inserts NonTerminal with label `"INSERT"` above Terminals.
    /// ///
    /// /// All Terminals that are dominated by a node matched by match_fn will have a NonTerminal
    /// /// with label `"INSERT"` inserted above.
    /// fn do_insertion(tree: &mut Tree) {
    ///     tree.insert_intermediate(|tree, parent_idx| {
    ///         if tree[parent_idx].label() == "label" {
    ///             Some("INSERT".to_string())
    ///         } else {
    ///             None
    ///         }
    ///     }).unwrap();
    /// }
    /// ```
    fn insert_intermediate<F>(&mut self, match_fn: F) -> Result<(), Error>
    where
        F: Fn(&Tree, NodeIndex) -> Option<String>;

    /// Remove non-terminals not matched by the match function.
    ///
    /// The root node will never be removed. Root node is determined by the `tree::is_root()`
    /// method. Detached material is re-attached above the removed node.
    ///
    /// E.g.:
    /// ```
    /// use lumberjack::{Tree, TreeOps};
    ///
    /// /// Remove all nonterminals from the tree that don't have a feature `"key"`.
    /// fn do_filtering(tree: &mut Tree) {
    ///     tree.filter_nonterminals(|tree, nonterminal_idx| {
    ///         let nt = tree[nonterminal_idx].nonterminal().unwrap();
    ///         if let Some(features) = nt.features() {
    ///              features.get_val("key").is_none()
    ///         } else {
    ///             false
    ///         }
    ///     }).unwrap();
    /// }
    /// ```
    fn filter_nonterminals<F>(&mut self, match_fn: F) -> Result<(), Error>
    where
        F: Fn(&Tree, NodeIndex) -> bool;

    /// Reattach terminals matched by the match function.
    ///
    /// The method iterates over the terminals in no particular order and reattaches those
    /// terminals for which `match_fn` returns true to the attachment point given in `attachment`.
    ///
    /// This method will remove NonTerminal nodes if all Terminals below it are reattached. This
    /// includes but is not limited to unary chains.
    ///
    /// Panics if the attachment point is the index of a Terminal.
    ///
    /// E.g.:
    /// ```
    /// use lumberjack::{Tree, TreeOps};
    ///
    /// /// Reattach all terminals.
    /// ///
    /// /// Reattach terminals to the root that are attached to a parent without feature `"key"`.
    /// fn do_reattachment(tree: &mut Tree) {
    ///     let root = tree.root();
    ///     tree.reattach_terminals(root, |tree, terminal_idx| {
    ///         if let Some((parent, _)) = tree.parent(terminal_idx) {
    ///             let parent_nt = tree[parent].nonterminal().unwrap();
    ///             if let Some(features) = parent_nt.features() {
    ///                  return features.get_val("key").is_none()
    ///             }
    ///         }
    ///         false
    ///     });
    /// }
    /// ```
    fn reattach_terminals<F>(&mut self, attachment: NodeIndex, match_fn: F)
    where
        F: Fn(&Tree, NodeIndex) -> bool;
}

impl TreeOps for Tree {
    fn annotate_parent_tag(&mut self, feature_name: &str) -> Result<(), Error> {
        let terminals = self.terminals().collect::<Vec<_>>();
        for terminal in terminals.into_iter() {
            let (parent, _) = self
                .parent(terminal)
                .ok_or_else(|| format_err!("Terminal without parent:\n{}", self[terminal]))?;

            let label = self[parent].label().to_owned();
            self[terminal]
                .features_mut()
                .insert(feature_name, Some(label));
        }
        Ok(())
    }

    fn insert_intermediate<F>(&mut self, match_fn: F) -> Result<(), Error>
    where
        F: Fn(&Tree, NodeIndex) -> Option<String>,
    {
        let terminals = self.terminals().collect::<Vec<_>>();
        let mut visit_map = self.graph().visit_map();

        for terminal in terminals.into_iter() {
            if !visit_map.visit(terminal) {
                continue;
            }
            let (parent, _) = self
                .parent(terminal)
                .ok_or_else(|| format_err!("Terminal without parent:\n{}", self[terminal]))?;
            if let Some(label) = match_fn(self, parent) {
                let mut children = self
                    .children(parent)
                    .filter(|(c, _)| self[*c].is_terminal())
                    .collect::<Vec<_>>();
                children.sort_by(|c1, c2| self[c1.0].span().cmp(self[c2.0].span()));
                let mut children = children.into_iter();
                let (terminal, _) = children.next().unwrap();
                let mut insert = self.insert_unary_above(terminal, label.clone());

                let mut prev_idx = self[terminal].span().start;
                for (child, edge) in children {
                    visit_map.visit(child);
                    let sibling_idx = self[child].span().start;
                    if sibling_idx == prev_idx + 1 {
                        prev_idx += 1;
                        self.reattach_node(insert, edge)?;
                    } else {
                        insert = self.insert_unary_above(child, label.clone());
                        prev_idx = sibling_idx;
                    }
                }
            };
        }
        Ok(())
    }

    fn filter_nonterminals<F>(&mut self, match_fn: F) -> Result<(), Error>
    where
        F: Fn(&Tree, NodeIndex) -> bool,
    {
        let nts = self
            .nonterminals()
            .filter(|nt| *nt != self.root())
            .collect::<Vec<_>>();

        for nt in nts {
            if !match_fn(self, nt) {
                self.remove_node(nt)?;
            }
        }
        Ok(())
    }

    fn reattach_terminals<F>(&mut self, attachment: NodeIndex, match_fn: F)
    where
        F: Fn(&Tree, NodeIndex) -> bool,
    {
        let terminals = self.terminals().collect::<Vec<_>>();

        for terminal in terminals {
            if match_fn(&self, terminal) {
                let mut climber = Climber::new(terminal, self);
                while self.siblings(terminal).count() == 0 {
                    if let Some(parent) = climber.next(self) {
                        if parent != self.root() && parent != attachment {
                            self.remove_node(parent).unwrap();
                        }
                    }
                }
                let (_, edge) = self.parent(terminal).unwrap();
                self.reattach_node(attachment, edge).unwrap();
            }
        }
        self.reset_nt_spans();
    }
}

pub trait UnaryChains {
    /// Collapse unary chains.
    ///
    /// Collapses unary chains into the node label of the lowest node in the chain, delimiting each
    /// node with `delim`.
    ///
    /// E.g. assuming `delim == "_"`, `(S (UC1 (UC2 (T t))))` is collapsed into `(UC2_UC1_S_T t)`.
    ///
    /// Collapsing is not lossless, Edge labels and annotations associated with the collapsed
    /// nonterminals are lost.
    fn collapse_unary_chains(&mut self, delim: &str) -> Result<(), Error>;

    /// Restore unary chains.
    ///
    /// Inverse of `collapse_unary_chains`. Expands the unary chains collapsed into node labels.
    ///
    /// E.g. assuming `delim == "_"`, `(UC2_UC1_S_T t)` is expanded into (S (UC1 (UC2 (T t)))).
    fn restore_unary_chains(&mut self, delim: &str) -> Result<(), Error>;
}

impl UnaryChains for Tree {
    fn collapse_unary_chains(&mut self, delim: &str) -> Result<(), Error> {
        let terminals = self.terminals().collect::<Vec<_>>();
        for terminal in terminals {
            let mut cur = terminal;
            let mut climber = Climber::new(terminal, self);

            while let Some(node) = climber.next(self) {
                if self[node].span() == self[cur].span() {
                    // spans are equal in unary branches.
                    let label = self.remove_node(node).unwrap().set_label(String::new());
                    let features = self[cur].features_mut();
                    let chain = features
                        .get_val("unary_chain")
                        .map(|chain| format!("{}{}{}", chain, delim, label))
                        .unwrap_or_else(|| label);
                    features.insert("unary_chain", Some(chain));
                } else {
                    cur = node;
                }
            }
        }
        Ok(())
    }

    fn restore_unary_chains(&mut self, delim: &str) -> Result<(), Error> {
        let nodes = self.graph().node_indices().collect::<Vec<_>>();
        for mut cur in nodes {
            if let Some(chain) = self[cur].features_mut().remove("unary_chain") {
                for label in chain.split(delim) {
                    cur = self.insert_unary_above(cur, label);
                }
            } else {
                continue;
            };
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
        if !self.is_projective() {
            let mut terminals = self.terminals().collect::<Vec<_>>();
            // terminals need to be sorted, otherwise indexing through spans can be incorrect
            self.sort_indices(&mut terminals);

            let mut dfs = DfsPostOrder::new(self.graph(), self.root());
            while let Some(attachment_point) = dfs.next(self.graph()) {
                // as long as the node at attachment_point is discontinuous, skipped indices will
                // be returned.
                while let Some(&skipped) = self[attachment_point]
                    .span()
                    .skips()
                    .and_then(|s| s.iter().next())
                {
                    // start climbing at skipped index
                    let mut climber = Climber::new(terminals[skipped], self);
                    while let Some((handle_parent, handle_edge)) = climber.next_with_edge(&self) {
                        // if the parent node of the handle covers the span of the attachment point,
                        // we are higher in the tree than the target node. This means, the edge
                        // below is introducing the nonprojective edge.
                        if self[handle_parent]
                            .span()
                            .covers_span(self[attachment_point].span())
                        {
                            let (new_edge, edge) =
                                self.reattach_node(attachment_point, handle_edge).unwrap();
                            self[new_edge] = edge;
                            break;
                        }
                    }
                }
                // return if all spans have been fixed
                if self.is_projective() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AnnotatePOS, TreeOps, UnaryChains};
    use crate::io::PTBFormat;
    use crate::{Features, Terminal, Tree};

    #[test]
    fn un_collapse_unary() {
        let input = "(ROOT (UNARY (T t)))";
        let mut t = PTBFormat::Simple.string_to_tree(input).unwrap();
        t.collapse_unary_chains("_").unwrap();
        assert_eq!(
            Some(&Features::from("unary_chain:UNARY_ROOT")),
            t[t.root()].features()
        );
        assert_eq!("(T t)", PTBFormat::Simple.tree_to_string(&t).unwrap());
        t.restore_unary_chains("_").unwrap();
        assert_eq!(input, PTBFormat::Simple.tree_to_string(&t).unwrap());

        let input = "(ROOT (UNARY (T t)) (ANOTHER (T2 t2)))";
        let mut t = PTBFormat::Simple.string_to_tree(input).unwrap();
        t.collapse_unary_chains("_").unwrap();
        assert_eq!(
            "(ROOT (T t) (T2 t2))",
            PTBFormat::Simple.tree_to_string(&t).unwrap()
        );
        t.restore_unary_chains("_").unwrap();
        assert_eq!(input, PTBFormat::Simple.tree_to_string(&t).unwrap());

        let input = "(ROOT (UNARY (INTERMEDIATE (T t) (T2 t2))) (ANOTHER (T3 t3)))";
        let mut t = PTBFormat::Simple.string_to_tree(input).unwrap();
        t.collapse_unary_chains("_").unwrap();
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&t).unwrap(),
            "(ROOT (INTERMEDIATE (T t) (T2 t2)) (T3 t3))"
        );
        t.restore_unary_chains("_").unwrap();
        assert_eq!(input, PTBFormat::Simple.tree_to_string(&t).unwrap());

        let input = "(ROOT (BRANCHING (T1 t1) (T2 t2)) (ANOTHER-BRANCH (T3 t3) (T4 t4)))";
        let mut t = PTBFormat::Simple.string_to_tree(input).unwrap();
        t.collapse_unary_chains("_").unwrap();
        assert_eq!(input, PTBFormat::Simple.tree_to_string(&t).unwrap());
        t.restore_unary_chains("_").unwrap();
        assert_eq!(input, PTBFormat::Simple.tree_to_string(&t).unwrap());
    }

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
        let mut tree = Tree::new("t1", "TERM1");
        let t1 = tree.root();
        tree.insert_unary_above(t1, "ROOT");
        let l = tree.insert_unary_above(t1, "L");
        let t2 = tree.push_terminal("t2", "TERM2").unwrap();
        let l_1 = tree.insert_unary_above(t2, "L1");
        tree.insert_terminal(l, Terminal::new("t3", "TERM3", 2));
        let t4 = tree.push_terminal("t4", "TERM4").unwrap();
        let ll = tree.insert_unary_above(t4, "L");
        tree.insert_terminal(ll, Terminal::new("t5", "TERM5", 4));

        let mut filtered_tree = tree.clone();
        filtered_tree
            .filter_nonterminals(|tree, nt| tree[nt].label() == "L")
            .unwrap();

        tree.remove_node(l_1).unwrap();
        assert_eq!(tree, filtered_tree);

        tree.insert_unary_above(t2, "L1");
        let mut filtered_tree = tree.clone();
        filtered_tree
            .filter_nonterminals(|tree, nt| tree[nt].label() == "L1")
            .unwrap();

        tree.remove_node(l).unwrap();
        tree.remove_node(ll).unwrap();
        assert_eq!(tree, filtered_tree);
    }

    #[test]
    fn insert_unks_nonproj() {
        // non projective tree, where one inserted node collects two nodes.
        let mut tree = Tree::new("t1", "TERM1");
        let t1 = tree.root();
        tree.insert_unary_above(t1, "ROOT");
        let l = tree.insert_unary_above(t1, "L");
        let t2 = tree.push_terminal("t2", "TERM2").unwrap();
        tree.insert_terminal(l, Terminal::new("t3", "TERM3", 2));
        let t4 = tree.push_terminal("t4", "TERM4").unwrap();
        let t5 = tree.push_terminal("t5", "TERM5").unwrap();

        let mut insert_tree = tree.clone();
        insert_tree
            .insert_intermediate(|tree, nt| {
                if tree[nt].label() == "L" {
                    None
                } else {
                    Some("UNK".to_string())
                }
            })
            .unwrap();

        tree.insert_unary_above(t2, "UNK");
        let unk = tree.insert_unary_above(t4, "UNK");
        let (_, edge) = tree.parent(t5).unwrap();
        tree.reattach_node(unk, edge).unwrap();
        assert_eq!(tree, insert_tree);
    }

    #[test]
    fn reattach() {
        let input = "(ROOT (BRANCHING (T1 t1) (T2 t2)) (ANOTHER-BRANCH (T3 t3) (T4 t4)))";
        let mut reattach_tree = PTBFormat::Simple.string_to_tree(input).unwrap();
        let root = reattach_tree.root();
        reattach_tree.reattach_terminals(root, |t, node| t[node].label().starts_with("T"));

        let mut target = Tree::new("t1", "T1");
        let t1 = target.root();
        target.insert_unary_above(t1, "ROOT");
        target.push_terminal("t2", "T2").unwrap();
        target.push_terminal("t3", "T3").unwrap();
        target.push_terminal("t4", "T4").unwrap();
        assert_eq!(target, reattach_tree);
    }

    #[test]
    fn reattach_punct() {
        let input = "(ROOT (BRANCHING (T1 t1) ($, ,) (T2 t2)) (ANOTHER-BRANCH (T3 t3) ($, ($. .)) (T4 t4)) ($. .))";
        let mut reattach_tree = PTBFormat::Simple.string_to_tree(input).unwrap();
        let root = reattach_tree.root();
        reattach_tree.reattach_terminals(root, |t, node| t[node].label().starts_with("$"));

        let mut target = Tree::new("t1", "T1");
        let t1 = target.root();
        target.insert_unary_above(t1, "ROOT");
        let branching = target.insert_unary_above(t1, "BRANCHING");
        target.push_terminal(",", "$,").unwrap();
        target.insert_terminal(branching, Terminal::new("t2", "T2", 2));
        let t3 = target.push_terminal("t3", "T3").unwrap();
        let another_branch = target.insert_unary_above(t3, "ANOTHER-BRANCH");
        target.push_terminal(".", "$.").unwrap();
        target.insert_terminal(another_branch, Terminal::new("t4", "T4", 5));
        target.push_terminal(".", "$.").unwrap();

        assert_eq!(target, reattach_tree)
    }
}
