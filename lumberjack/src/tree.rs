use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use petgraph::prelude::{Bfs, Direction, EdgeIndex, EdgeRef, NodeIndex, StableGraph};

use crate::util::LabelSet;
use crate::{Edge, Node};

/// Enum describing whether a tree is projective.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Projectivity {
    Projective,
    Nonprojective,
}

/// `Tree`
///
/// `Tree`s represent constituency trees and consist of `Node`s. The nodes are either
/// `Terminal`s or `NonTerminal`s. Relations between nodes are expressed as `Edge`s.
#[derive(Debug, Clone)]
pub struct Tree {
    graph: StableGraph<Node, Edge>,
    n_terminals: usize,
    root: NodeIndex,
    projectivity: Projectivity,
}

impl Tree {
    pub(crate) fn new(
        graph: StableGraph<Node, Edge>,
        n_terminals: usize,
        root: NodeIndex,
        projectivity: Projectivity,
    ) -> Self {
        Tree {
            graph,
            n_terminals,
            root,
            projectivity,
        }
    }

    /// Get the number of terminals in the tree.
    pub fn n_terminals(&self) -> usize {
        self.n_terminals
    }

    /// Get the index of the root of the tree.
    pub fn root(&self) -> NodeIndex {
        self.root
    }

    /// Get an iterator over the terminals in the constituency tree.
    pub fn terminals<'a>(&'a self) -> impl Iterator<Item = NodeIndex> + 'a {
        self.graph
            .node_indices()
            .filter(move |idx| self.graph[*idx].is_terminal())
    }

    /// Get an iterator over the terminal indices in the constituency tree.
    pub fn nonterminals<'a>(&'a self) -> impl Iterator<Item = NodeIndex> + 'a {
        self.graph
            .node_indices()
            .filter(move |idx| !self.graph[*idx].is_terminal())
    }

    /// Get the parent and corresponding edge of a tree node.
    ///
    /// * Returns `NodeIndex` of immediately dominating node and corresponding `EdgeIndex`.
    /// * Returns `None` if `node` doesn't exist or doesn't have incoming edges.
    pub fn parent(&self, node: NodeIndex) -> Option<(NodeIndex, EdgeIndex)> {
        self.graph
            .edges_directed(node, Direction::Incoming)
            .next()
            .map(|edge_ref| (edge_ref.source(), edge_ref.id()))
    }

    /// Get an iterator over `node`'s children.
    pub fn children<'a>(&'a self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + 'a {
        self.graph
            .edges_directed(node, Direction::Outgoing)
            .map(|edge_ref| edge_ref.target())
    }

    /// Get an iterator over `node`'s siblings.
    pub fn siblings<'a>(&'a self, node: NodeIndex) -> Box<Iterator<Item = NodeIndex> + 'a> {
        if let Some((parent, _)) = self.parent(node) {
            Box::new(self.children(parent).filter(move |&target| target != node))
        } else {
            Box::new(std::iter::empty::<NodeIndex<u32>>())
        }
    }

    /// Get an iterator over `node`'s descendents.
    pub fn descendent_terminals<'a>(
        &'a self,
        node: NodeIndex,
    ) -> Box<Iterator<Item = NodeIndex> + 'a> {
        if let Node::NonTerminal(nt) = &self[node] {
            let terminals = self.terminals().collect::<Vec<_>>();
            Box::new(nt.span().into_iter().map(move |idx| terminals[idx]))
        } else {
            Box::new(std::iter::empty::<NodeIndex<u32>>())
        }
    }

    /// Get sibling-relation of two tree nodes.
    ///
    /// Returns whether two nodes are immediately dominated by the same node.
    pub fn are_siblings(&self, node_1: NodeIndex, node_2: NodeIndex) -> bool {
        match (self.parent(node_1), self.parent(node_2)) {
            (Some(parent_1), Some(parent_2)) => parent_1 == parent_2,
            _ => false,
        }
    }

    /// Get an immutable reference to the underlying `StableGraph`.
    pub fn graph(&self) -> &StableGraph<Node, Edge> {
        &self.graph
    }

    /// Get a mutable reference to the underlying `StableGraph`.
    pub(crate) fn graph_mut(&mut self) -> &mut StableGraph<Node, Edge> {
        &mut self.graph
    }

    /// Returns whether the tree is projective.
    pub fn projective(&self) -> bool {
        self.projectivity == Projectivity::Projective
    }

    /// Set the tree's projectivity.
    pub(crate) fn set_projectivity(&mut self, projectivity: Projectivity) {
        self.projectivity = projectivity
    }

    /// Project indices of `NonTerminal`s onto `Terminal`s.
    ///
    /// This method projects the `NodeIndex` of `NonTerminal`s with a label in
    /// `tag_set` onto the terminal nodes. The hierarchically closest node's `NodeIndex`
    /// will be assigned to each terminal.
    ///
    /// Returns a `Vec<NodeIndex>`.
    pub fn project_nt_indices(&self, tag_set: &LabelSet) -> Vec<NodeIndex> {
        let mut bfs = Bfs::new(&self.graph, self.root);
        let mut indices = vec![self.root; self.n_terminals];
        while let Some(node) = bfs.next(&self.graph) {
            if let Some(inner) = self.graph[node].nonterminal() {
                if tag_set.matches(inner.label()) {
                    for id in self.graph[node].span() {
                        indices[id] = node;
                    }
                }
            }
        }
        indices
    }

    /// Project labels of `NonTerminal`s onto `Terminal`s.
    ///
    /// This method projects the label for each node in `tag_set` onto the terminal nodes.
    /// The hierarchically closest node's label will be assigned to each terminal.
    pub fn project_tag_set(&self, tag_set: &LabelSet) -> Vec<&str> {
        let indices = self.project_nt_indices(tag_set);
        indices
            .into_iter()
            .map(|nt_idx| {
                // safe to unwrap, we get the IDs from project_ids()
                // which checks if nt_idx is Node::Inner
                self.graph[nt_idx].nonterminal().unwrap().label()
            })
            .collect::<Vec<_>>()
    }

    /// Project unique IDs for nodes with label in `tag_set` onto terminals.
    pub fn project_ids(&self, tag_set: &LabelSet) -> Vec<usize> {
        let indices = self.project_nt_indices(tag_set);
        let mut ids = vec![0; self.n_terminals];
        let mut mapping = HashMap::new();
        for (term_id, idx) in indices.into_iter().enumerate() {
            let n = mapping.len();
            let id = *mapping.entry(idx).or_insert(n);
            ids[term_id] = id;
        }
        ids
    }

    // helper method to sort a vec of node indices
    // order is determined by:
    // 1. lower bound of span (starting point of span)
    // 2. upper bound of span (end point of span)
    // 3. number of covered indices by span
    // 4. Inner nodes before terminal nodes
    // 5. alphabetical order
    fn sort_indices(&self, indices: &mut Vec<NodeIndex>) {
        indices.sort_by(
            |node1, node2| match self[*node1].span().cmp(&self[*node2].span()) {
                Ordering::Equal => match (&self[*node1], &self[*node2]) {
                    (Node::NonTerminal(_), Node::Terminal(_)) => Ordering::Greater,
                    (Node::Terminal(_), Node::NonTerminal(_)) => Ordering::Less,
                    (Node::NonTerminal(nt1), Node::NonTerminal(nt2)) => {
                        nt1.label().cmp(nt2.label())
                    }
                    (Node::Terminal(t1), Node::Terminal(t2)) => t1.form().cmp(t2.form()),
                },
                ordering => ordering,
            },
        );
    }
}

impl PartialEq for Tree {
    fn eq(&self, other: &Tree) -> bool {
        // cheap checks first, node count and number of terminals
        if self.n_terminals != other.n_terminals {
            return false;
        };
        if self.graph.node_count() != other.graph.node_count() {
            return false;
        };

        // sort indices by criteria defined above
        let mut nodes1 = self.graph.node_indices().collect::<Vec<_>>();
        self.sort_indices(&mut nodes1);
        let mut nodes2 = other.graph.node_indices().collect::<Vec<_>>();
        other.sort_indices(&mut nodes2);

        // two trees are equal iff after sorting for all node pairs (node1, node2) it holds that
        // node1 == node2, parent(node1) == parent(node2) and
        // incoming_edge(node1) == incoming_edge(node2)
        for (node1, node2) in nodes1.into_iter().zip(nodes2) {
            if self[node1] != other[node2] {
                return false;
            }
            let p1 = self
                .parent(node1)
                .map(|(parent_id, edge_id)| (&self[parent_id], &self[edge_id]));
            let p2 = other
                .parent(node2)
                .map(|(parent_id, edge_id)| (&other[parent_id], &other[edge_id]));
            if p1 != p2 {
                return false;
            }
        }
        true
    }
}

impl Index<NodeIndex> for Tree {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &<Self as Index<NodeIndex>>::Output {
        &self.graph[index]
    }
}

impl Index<EdgeIndex> for Tree {
    type Output = Edge;

    fn index(&self, index: EdgeIndex) -> &<Self as Index<EdgeIndex>>::Output {
        &self.graph[index]
    }
}

impl IndexMut<NodeIndex> for Tree {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Node {
        &mut self.graph[index]
    }
}

impl IndexMut<EdgeIndex> for Tree {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Edge {
        &mut self.graph[index]
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::prelude::{NodeIndex, StableGraph};

    use crate::util::LabelSet;
    use crate::{Edge, Node, NonTerminal, Projectivity, Span, Terminal, Tree};

    #[test]
    fn project_node_labels() {
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("FIRST", Span::from_vec(vec![0, 2]).unwrap());
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term3 = Terminal::new("t3", "TERM3", 2);
        let second = NonTerminal::new("SECOND", 3);
        let term4 = Terminal::new("t4", "TERM4", 4);
        let term5 = Terminal::new("t5", "TERM5", 5);
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
        let indices = tree.project_tag_set(&LabelSet::Positive(tags));
        let target = vec!["FIRST", "ROOT", "FIRST", "ROOT", "ROOT"];
        assert_eq!(indices, target)
    }

    #[test]
    fn project_node_ids() {
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 5));
        let first = NonTerminal::new("L", Span::from_vec(vec![0, 2]).unwrap());
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term2 = Terminal::new("t2", "TERM1", 1);
        let term3 = Terminal::new("t3", "TERM3", 2);
        let second = NonTerminal::new("L", Span::new_continuous(3, 4));
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
        tags.insert("L".into());
        let indices = tree.project_ids(&LabelSet::Positive(tags));
        // 0 is first "L", 1 is ROOT and 2 is second "L"
        let target = vec![0, 1, 0, 2, 1];
        assert_eq!(indices, target)
    }

    #[test]
    fn terminals() {
        let tree = some_tree();
        let terminals = tree
            .terminals()
            .map(|terminal| tree.graph()[terminal].terminal().unwrap().form())
            .collect::<Vec<_>>();
        assert_eq!(
            vec![
                "t1".to_string(),
                "t2".to_string(),
                "t3".to_string(),
                "t4".to_string(),
                "t5".to_string()
            ],
            terminals
        );
    }

    #[test]
    fn siblings() {
        let tree = some_tree();
        assert!(tree.siblings(tree.root()).next().is_none());
        // NodeIndex(1) is Node::Inner("FIRST" ..)
        let siblings = tree
            .siblings(NodeIndex::new(1))
            .into_iter()
            .map(|sibling| match tree.graph()[sibling] {
                Node::NonTerminal(ref nt) => nt.label().to_string(),
                Node::Terminal(ref t) => t.form().to_string(),
            })
            .collect::<Vec<_>>();
        // reverse order of addition
        assert_eq!(siblings, vec!["t5", "SECOND", "t3"]);
    }

    #[test]
    fn equality() {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut g = StableGraph::new();
        let term1 = Terminal::new("t1", "TERM1", 0);
        let term2 = Terminal::new("t2", "TERM1", 1);
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("FIRST", Span::new_continuous(0, 2));
        let second = NonTerminal::new("SECOND", Span::new_continuous(3, 4));
        let term4 = Terminal::new("t4", "TERM4", 3);
        let term5 = Terminal::new("t5", "TERM5", 4);
        let term3 = Terminal::new("t3", "TERM3", 2);
        let term1_idx = g.add_node(Node::Terminal(term1));
        let term3_idx = g.add_node(Node::Terminal(term3));
        let term2_idx = g.add_node(Node::Terminal(term2.clone()));
        let first_idx = g.add_node(Node::NonTerminal(first));
        let term4_idx = g.add_node(Node::Terminal(term4));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let second_idx = g.add_node(Node::NonTerminal(second));
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, second_idx, Edge::default());
        g.add_edge(first_idx, term2_idx, Edge::default());
        g.add_edge(root_idx, term3_idx, Edge::default());
        g.add_edge(first_idx, term1_idx, Edge::default());
        g.add_edge(root_idx, term5_idx, Edge::default());
        g.add_edge(second_idx, term4_idx, Edge::default());
        g.add_edge(root_idx, first_idx, Edge::default());
        let some_tree = some_tree();
        let mut other_tree = Tree::new(g.clone(), 5, root_idx, Projectivity::Projective);
        assert_eq!(some_tree, other_tree);
        other_tree[term2_idx]
            .terminal_mut()
            .unwrap()
            .set_lemma(Some("some_lemma"));
        assert_ne!(some_tree, other_tree);
        g.remove_node(term2_idx);
        let other_tree = Tree::new(g.clone(), 4, root_idx, Projectivity::Projective);
        assert_ne!(some_tree, other_tree);
        let new_t2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(first_idx, new_t2_idx, Edge::default());
        let other_tree = Tree::new(g.clone(), 5, root_idx, Projectivity::Projective);
        assert_eq!(some_tree, other_tree);
    }

    fn some_tree() -> Tree {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("FIRST", Span::new_continuous(0, 2));
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
        g.add_edge(first_idx, term2_idx, Edge::default());
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(root_idx, term3_idx, Edge::default());
        let second_idx = g.add_node(Node::NonTerminal(second));
        g.add_edge(root_idx, second_idx, Edge::default());
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(second_idx, term4_idx, Edge::default());
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());

        Tree::new(g, 5, root_idx, Projectivity::Projective)
    }
}
