use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::ops::{Index, IndexMut};

use failure::{format_err, Error};
use petgraph::prelude::*;

use crate::{Edge, Node, NonTerminal, Span};

/// Enum describing whether a tree is projective.
#[derive(Debug, Clone, Copy)]
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
            .edges_directed(node, Incoming)
            .next()
            .map(|edge_ref| (edge_ref.source(), edge_ref.id()))
    }

    /// Get an iterator over `node`'s children.
    pub fn children<'a>(&'a self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + 'a {
        self.graph
            .edges_directed(node, Outgoing)
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

    /// Returns whether the tree is projective.
    pub fn projective(&self) -> bool {
        match self.projectivity {
            Projectivity::Projective => true,
            Projectivity::Nonprojective => false,
        }
    }

    /// Projectivize the `Tree`.
    ///
    /// Projectivization is done by re-attaching the non-projective content at the highest point
    /// allowing non-crossing edges while maintaining the linear order of the sentence.
    pub fn projectivize(&mut self) {
        if !self.projective() {
            let terminals = self.terminals().collect::<Vec<_>>();;
            let mut dfs = DfsPostOrder::new(&self.graph, self.root);
            let mut log = vec![None; terminals.len()];

            while let Some(attachment_point_candidate) = dfs.next(&self.graph) {
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
                                    let edge = self.graph.remove_edge(rm_edge).unwrap();
                                    self.graph.update_edge(
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
                    .set_span(Span::new_continuous(span.lower(), span.upper()))
            }
            self.projectivity = Projectivity::Projective;
        }
    }

    /// Project indices of `NonTerminal`s onto `Terminal`s.
    ///
    /// This method projects the `NodeIndex` of `NonTerminal`s with a label in
    /// `tag_set` onto the terminal nodes. The hierarchically closest node's `NodeIndex`
    /// will be assigned to each terminal.
    ///
    /// Returns a `Vec<NodeIndex>`.
    pub fn project_nt_indices(&self, tag_set: &HashSet<String>) -> Vec<NodeIndex> {
        let mut bfs = Bfs::new(&self.graph, self.root);
        let mut indices = vec![self.root; self.n_terminals];
        while let Some(node) = bfs.next(&self.graph) {
            if let Some(inner) = self.graph[node].nonterminal() {
                if tag_set.contains(inner.label()) {
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
    pub fn project_tag_set(&self, tag_set: &HashSet<String>) -> Vec<&str> {
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
    pub fn project_ids(&self, tag_set: &HashSet<String>) -> Vec<usize> {
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

    /// Removes all nodes that are not specified in `tag_set`.
    ///
    /// The root node will never be removed. Root node is determined by the `tree::is_root()`
    /// method. Nodes without incoming edges are treated as the root.
    pub fn filter_nodes(&mut self, tag_set: &HashSet<String>) -> Result<(), Error> {
        // divide indices into keep- and delete-list, root is excluded as we don't want to break the
        // tree. Collecting is necessary because .node_indices() borrows from the graph
        let (keep, delete) = self
            .graph()
            .node_indices()
            .filter(|node| *node != self.root)
            .fold((Vec::new(), Vec::new()), |(mut keep, mut delete), node| {
                if let Node::NonTerminal(ref nt) = self[node] {
                    if tag_set.contains(nt.label()) {
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
                format_err!("Non-root node without incoming edge: {}", self.graph[node])
            })?;

            // climb up until field specified in tag_set or root is found
            let mut climber = Climber::new(node);
            while let Some(parent_idx) = climber.next(&self) {
                let parent = self[parent_idx]
                    .nonterminal()
                    .ok_or_else(|| format_err!("Terminal as parent: {}", self[parent_idx]))?;
                if tag_set.contains(parent.label()) || parent_idx == self.root {
                    // safe to unwrap, id is guaranteed to be valid (line 138)
                    let weight = self.graph.remove_edge(id).unwrap();
                    self.graph.update_edge(parent_idx, node, weight);
                    break;
                }
            }
        }
        for node in delete {
            self.graph.remove_node(node);
        }
        Ok(())
    }

    /// Insert an intermediate node above terminals.
    ///
    /// If a terminal is not dominated by a node with label in `tag_set`
    /// a new non-terminal node is inserted above with label `insertion`.
    pub fn insert_intermediate(
        &mut self,
        tag_set: &HashSet<String>,
        insertion: &str,
    ) -> Result<(), Error> {
        let terminals = self.terminals().collect::<Vec<_>>();
        let mut prev_attachment = None;
        for (position, terminal) in terminals.into_iter().enumerate() {
            let (parent, edge_id) = self
                .parent(terminal)
                .ok_or_else(|| format_err!("Terminal without parent:\n{}", self[terminal]))?;

            if tag_set.contains(self.graph[parent].nonterminal().unwrap().label()) {
                continue;
            }

            let weight = self.graph.remove_edge(edge_id).unwrap();
            if let Some((prev_position, prev_insert)) = prev_attachment {
                if prev_position == position - 1 && self.parent(prev_insert).unwrap().0 == parent {
                    self.graph.add_edge(prev_insert, terminal, weight);
                    self.graph[prev_insert].extend_span()?;
                    prev_attachment = Some((position, prev_insert));
                    continue;
                }
            }

            let span = self.graph[terminal].span().clone();
            let nt = Node::NonTerminal(NonTerminal::new(insertion, span));
            let inserted_idx = self.graph.add_node(nt);
            self.graph.add_edge(parent, inserted_idx, weight.clone());
            self.graph.add_edge(inserted_idx, terminal, weight);
            prev_attachment = Some((position, inserted_idx))
        }
        Ok(())
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

/// Struct to climb up a `Tree`.
///
/// This struct does not borrow from the tree in order to allow mutation during climbing.
pub struct Climber {
    cur: NodeIndex,
}

impl Climber {
    /// Constructs a new `Climber`.
    pub fn new(node: NodeIndex) -> Self {
        Climber { cur: node }
    }

    /// Moves up the tree by following the first incoming edge.
    ///
    /// This method behaves like an iterator, returning `Some(NodeIndex)` before reaching the final
    /// state. Calling this method again in the final state will return `None`.
    pub fn next(&mut self, tree: &Tree) -> Option<NodeIndex> {
        if let Some(parent) = tree
            .graph()
            .neighbors_directed(self.cur, Direction::Incoming)
            .next()
        {
            self.cur = parent;
            Some(parent)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::prelude::NodeIndex;
    use petgraph::prelude::StableGraph;

    use crate::{Edge, Node, NonTerminal, Projectivity, Span, Terminal, Tree};

    #[test]
    fn insert_unks_nonproj() {
        // non projective tree, where one inserted node collects two nodes.
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let first = NonTerminal::new("L", Span::from_vec(vec![0, 2]).unwrap());
        let first_idx = g.add_node(Node::NonTerminal(first));
        g.add_edge(root_idx, first_idx, Edge::default());
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(first_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(root_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(root_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());
        let mut set = HashSet::new();
        set.insert("L".into());
        let mut unk_tree = Tree::new(g, 5, root_idx, Projectivity::Nonprojective);
        unk_tree.insert_intermediate(&set, "UNK").unwrap();

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
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(first_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(first_unk_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(second_unk_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
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
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let second = NonTerminal::new("SECOND", Span::new_continuous(3, 4));
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
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
        let indices = tree.project_nt_indices(&tags);
        let target = vec![first_idx, first_idx, root_idx, root_idx, root_idx];
        assert_eq!(indices, target)
    }

    #[test]
    fn project_node_indices_nonprojective() {
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("FIRST", Span::from_vec(vec![0, 2]).unwrap());
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let second = NonTerminal::new("SECOND", Span::new_continuous(3, 4));
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
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
        let indices = tree.project_nt_indices(&tags);
        let target = vec![first_idx, root_idx, first_idx, root_idx, root_idx];
        assert_eq!(indices, target)
    }

    #[test]
    fn project_node_labels() {
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("FIRST", Span::from_vec(vec![0, 2]).unwrap());
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let second = NonTerminal::new("SECOND", Span::new_continuous(3, 4));
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
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
        let indices = tree.project_tag_set(&tags);
        let target = vec!["FIRST", "ROOT", "FIRST", "ROOT", "ROOT"];
        assert_eq!(indices, target)
    }

    #[test]
    fn project_node_ids() {
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("L", Span::from_vec(vec![0, 2]).unwrap());
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let second = NonTerminal::new("L", Span::new_continuous(3, 4));
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
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
        let indices = tree.project_ids(&tags);
        // 0 is first "L", 1 is ROOT and 2 is second "L"
        let target = vec![0, 1, 0, 2, 1];
        assert_eq!(indices, target)
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
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(first_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(second_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(third_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());

        let tree = Tree::new(g, 5, root_idx, Projectivity::Nonprojective);
        let mut tags = HashSet::new();
        tags.insert("L".into());
        let mut filtered_tree = tree.clone();
        filtered_tree.filter_nodes(&tags).unwrap();

        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let first = NonTerminal::new("L", Span::from_vec(vec![0, 2]).unwrap());
        let first_idx = g.add_node(Node::NonTerminal(first));
        g.add_edge(root_idx, first_idx, Edge::default());
        let third = NonTerminal::new("L", Span::new_continuous(3, 4));
        let third_idx = g.add_node(Node::NonTerminal(third));
        g.add_edge(root_idx, third_idx, Edge::default());
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(first_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(root_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(first_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(third_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());
        let target = Tree::new(g, 5, root_idx, Projectivity::Nonprojective);
        assert_eq!(target, filtered_tree);

        let mut tags = HashSet::new();
        tags.insert("L1".into());
        let mut filtered_tree = tree.clone();
        filtered_tree.filter_nodes(&tags).unwrap();
        let mut g = StableGraph::new();
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let root_idx = g.add_node(Node::NonTerminal(root));
        let second = NonTerminal::new("L1", Span::new_continuous(1, 2));
        let second_idx = g.add_node(Node::NonTerminal(second));
        g.add_edge(root_idx, second_idx, Edge::default());
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term1_idx = g.add_node(Node::Terminal(term1));
        g.add_edge(root_idx, term1_idx, Edge::default());
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(second_idx, term2_idx, Edge::default());
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let term3_idx = g.add_node(Node::Terminal(term3));
        g.add_edge(root_idx, term3_idx, Edge::default());
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term4_idx = g.add_node(Node::Terminal(term4));
        g.add_edge(root_idx, term4_idx, Edge::default());
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
        let term5_idx = g.add_node(Node::Terminal(term5));
        g.add_edge(root_idx, term5_idx, Edge::default());
        let target = Tree::new(g, 5, root_idx, Projectivity::Projective);
        assert_eq!(target, filtered_tree);
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
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let root = NonTerminal::new("ROOT", Span::new_continuous(0, 6));
        let first = NonTerminal::new("FIRST", Span::new_continuous(0, 2));
        let second = NonTerminal::new("SECOND", Span::new_continuous(3, 4));
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
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
        let term1 = Terminal::new("t1", "TERM1", Span::new_continuous(0, 1));
        let term2 = Terminal::new("t2", "TERM1", Span::new_continuous(1, 2));
        let term3 = Terminal::new("t3", "TERM3", Span::new_continuous(2, 3));
        let second = NonTerminal::new("SECOND", Span::new_continuous(3, 4));
        let term4 = Terminal::new("t4", "TERM4", Span::new_continuous(3, 4));
        let term5 = Terminal::new("t5", "TERM5", Span::new_continuous(4, 5));
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
