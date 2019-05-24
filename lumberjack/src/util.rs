use std::borrow::Borrow;
use std::collections::HashSet;

use petgraph::prelude::{EdgeIndex, NodeIndex};

use crate::Tree;

/// Struct to climb up a `Tree`.
///
/// This struct does not borrow from the tree in order to allow mutation during climbing.
///
/// `Climber` advances to the next node in the tree before returning the previous node. This means
/// that removing a node that was just returned through the `next` method will not invalidate the
/// climber. Through this mechanism, reattachment of the node returned from `next` will not be
/// reflected in the path since the `Climber` already advanced past it. If the updated path is
/// important, the climber should be re-initialized with the previously returned node.
pub struct Climber {
    cur: Option<(NodeIndex, EdgeIndex)>,
}

impl Climber {
    /// Constructs a new `Climber`.
    pub fn new(node: NodeIndex, tree: &Tree) -> Self {
        let cur = tree.parent(node);
        Climber { cur }
    }

    /// Moves up the tree by following the first incoming edge.
    ///
    /// This method behaves like an iterator, returning `Some(NodeIndex)` before reaching the final
    /// state. Calling this method again in the final state will return `None`.
    pub fn next(&mut self, tree: &Tree) -> Option<NodeIndex> {
        if let Some((node, _)) = self.cur {
            self.cur = tree.parent(node);
            Some(node)
        } else {
            None
        }
    }

    /// Moves up the tree by following the first incoming edge.
    ///
    /// This method behaves like an iterator, returning `Some(NodeIndex)` before reaching the final
    /// state. Calling this method again in the final state will return `None`.
    ///
    /// Returns a tuple of `(NodeIndex, EdgeIndex)` where `NodeIndex` is the parent node's index
    /// and `EdgeIndex` is the incoming edge`s index.
    pub fn next_with_edge(&mut self, tree: &Tree) -> Option<(NodeIndex, EdgeIndex)> {
        if let Some((node, edge)) = self.cur {
            self.cur = tree.parent(node);
            Some((node, edge))
        } else {
            None
        }
    }
}

/// LabelSet.
#[derive(Clone)]
pub enum LabelSet {
    /// Variant used for positive matching.
    Positive(HashSet<String>),
    /// Variant used for negative matching.
    Negative(HashSet<String>),
}

impl LabelSet {
    /// Returns whether the query matched the `LabelSet`.
    ///
    /// If `self` is `LabelSet::Positive`, `true` is returned if the query was found, `false`
    /// otherwise. If `self` is `LabelSet::Negative`, `true` is returned if the query was not foud.
    pub fn matches(&self, q: impl Borrow<str>) -> bool {
        match self {
            LabelSet::Positive(ref set) => set.contains(q.borrow()),
            LabelSet::Negative(ref set) => !set.contains(q.borrow()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::io::PTBFormat;
    use crate::util::{Climber, LabelSet};

    #[test]
    fn label_set_test() {
        let set = vec!["a".to_string(), "b".to_string(), "c".to_string()]
            .into_iter()
            .collect::<HashSet<_>>();
        let positive_label_set = LabelSet::Positive(set.clone());
        assert!(positive_label_set.matches("a"));
        assert!(positive_label_set.matches("b"));
        assert!(positive_label_set.matches("c"));
        assert!(!positive_label_set.matches("d"));
        assert!(!positive_label_set.matches("e"));
        let positive_label_set = LabelSet::Negative(set);
        assert!(!positive_label_set.matches("a"));
        assert!(!positive_label_set.matches("b"));
        assert!(!positive_label_set.matches("c"));
        assert!(positive_label_set.matches("d"));
        assert!(positive_label_set.matches("e"));
    }

    #[test]
    fn climber_test() {
        let input = "(NX (NN Nounphrase) (PX (PP on) (NX (DET a) (ADJ single) (NX line))))";
        let tree = PTBFormat::TueBa.string_to_tree(input).unwrap();
        let mut climber = Climber::new(tree.root(), &tree);
        assert!(climber.next(&tree).is_none());
        let mut terminals = tree.terminals();
        terminals.next();
        let on_idx = terminals.next().unwrap();
        let mut climber = Climber::new(on_idx, &tree);
        let px_idx = climber.next(&tree).unwrap();
        assert!(tree[px_idx].nonterminal().is_some());
        assert_eq!(tree[px_idx].nonterminal().unwrap().label(), "PX");
        let nx_idx = climber.next(&tree).unwrap();
        assert!(tree[nx_idx].nonterminal().is_some());
        assert_eq!(tree[nx_idx].nonterminal().unwrap().label(), "NX");

        assert!(climber.next(&tree).is_none());
    }
}
