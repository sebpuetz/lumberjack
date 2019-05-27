use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use failure::Error;
use fixedbitset::FixedBitSet;
use petgraph::prelude::{Direction, EdgeIndex, EdgeRef, NodeIndex, StableGraph};
use petgraph::visit::{Dfs, DfsPostOrder, VisitMap};

use crate::util::Climber;
use crate::{Continuity, Edge, Node, NonTerminal, Span, Terminal};

/// `Tree`
///
/// `Tree`s represent constituency trees and consist of `Node`s. The nodes are either
/// `Terminal`s or `NonTerminal`s. Relations between nodes are expressed as `Edge`s.
#[derive(Debug, Clone)]
pub struct Tree {
    graph: StableGraph<Node, Edge>,
    n_terminals: usize,
    root: NodeIndex,
    num_discontinuous: usize,
}

impl Tree {
    /// Construct a new Tree with a single Terminal.
    ///
    /// ```
    /// use lumberjack::Tree;
    /// use lumberjack::io::PTBFormat;
    ///
    /// let tree = Tree::new("a", "A");
    /// assert_eq!("(A a)", PTBFormat::Simple.tree_to_string(&tree).unwrap());
    /// ```
    pub fn new(form: impl Into<String>, pos: impl Into<String>) -> Self {
        let terminal = Terminal::new(form, pos, 0);
        let mut graph = StableGraph::new();
        let root = graph.add_node(Node::from(terminal));
        Tree {
            graph,
            n_terminals: 1,
            root,
            num_discontinuous: 0,
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
    pub fn children<'a>(
        &'a self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, EdgeIndex)> + 'a {
        self.graph
            .edges_directed(node, Direction::Outgoing)
            .map(|edge_ref| (edge_ref.target(), edge_ref.id()))
    }

    /// Move a Terminal to a different position in the sentence.
    ///
    /// This method does not reattach the Terminal and can change the Tree's projectivity.
    ///
    /// Panics if index is out of bounds or the index of a NonTerminal is passed as argument.
    pub fn move_terminal(&mut self, terminal: NodeIndex, index: usize) {
        if self[terminal].span().start == index {
            return;
        }
        assert!(self[terminal].is_terminal(), "Can't move NonTerminals.");
        assert!(index < self.n_terminals, "Index out of bounds.");
        let old_pos = self[terminal].span().start;
        let terminals = self
            .terminals()
            .filter(|t| *t != terminal)
            .collect::<Vec<_>>();
        for terminal in terminals {
            let terminal = self[terminal]
                .terminal_mut()
                .expect("Unexpected NonTerminal");
            let pos = terminal.idx();
            if old_pos <= pos && pos <= index {
                // shift to the right
                terminal.set_idx(pos - 1);
            } else if index <= pos && pos <= old_pos {
                // shift to the left
                terminal.set_idx(pos + 1);
            }
        }
        self[terminal].set_span(index).unwrap();
        self.reset_nt_spans();
    }

    /// Insert a new terminal node.
    ///
    /// Inserts a new terminal node with the given parent at the index specified in the Terminal's
    /// Span.
    ///
    /// Panics if the specified parent node is a Terminal or if the index is out of bounds.
    pub fn insert_terminal(&mut self, parent: NodeIndex, terminal: Terminal) -> NodeIndex {
        assert!(
            !self[parent].is_terminal(),
            "Can't insert node below terminal."
        );
        let index = terminal.span().start;
        assert!(index <= self.n_terminals, "Index out of bounds.");
        let terminals = self.terminals().collect::<Vec<_>>();
        for terminal in terminals {
            let terminal = self[terminal]
                .terminal_mut()
                .expect("Unexpected NonTerminal");
            let pos = terminal.idx();
            if pos >= index {
                terminal.set_idx(pos + 1);
            }
        }
        let terminal = self.graph.add_node(terminal.into());
        self.graph.add_edge(parent, terminal, Edge::default());
        self.reset_nt_spans();
        self.n_terminals += 1;
        terminal
    }

    /// Adds a Terminal to the end of the sentence.
    ///
    /// This method adds a terminal to the end of the sentence and attaches it to the
    /// root node.
    ///
    /// Returns `Err` if the root node is a Terminal.
    ///
    /// ```
    /// use lumberjack::Tree;
    /// use lumberjack::io::PTBFormat;
    ///
    /// let mut tree = Tree::new("a", "A");
    /// let root = tree.root();
    /// let root = tree.insert_unary_above(root, "ROOT");
    /// tree.push_terminal("b", "B").unwrap();
    /// assert_eq!("(ROOT (A a) (B b))", PTBFormat::Simple.tree_to_string(&tree).unwrap());
    /// ```
    pub fn push_terminal(
        &mut self,
        form: impl Into<String>,
        pos: impl Into<String>,
    ) -> Result<NodeIndex, Error> {
        if self[self.root].is_terminal() {
            return Err(format_err!(
                "Can't append terminal node if root is a terminal."
            ));
        }
        let terminal = Terminal::new(form, pos, self.n_terminals);
        self.n_terminals += 1;
        let terminal = self.graph.add_node(terminal.into());
        self.graph.add_edge(self.root, terminal, Edge::default());
        let root = self.root;
        self[root].extend_span().unwrap();
        Ok(terminal)
    }

    /// Insert a new unary node above a node.
    ///
    /// Inserts a new unary node above `child` and returns the index of the inserted node.
    pub fn insert_unary_above<S>(&mut self, child: NodeIndex, node_label: S) -> NodeIndex
    where
        S: Into<String>,
    {
        let span = self[child].span().to_owned();
        let node = NonTerminal::new(node_label, span).into();
        let insert = self.graph.add_node(node);
        if let Some((parent, old_edge_idx)) = self.parent(child) {
            let old_edge = self.graph.remove_edge(old_edge_idx).unwrap();
            self.graph.add_edge(parent, insert, Edge::default());
            self.graph.add_edge(insert, child, old_edge);
        } else {
            self.graph.add_edge(insert, child, Edge::default());
            if child == self.root {
                self.root = insert;
            }
        }
        insert
    }

    /// Insert a new unary node below a node.
    ///
    /// Insert a new node that is dominated by `node` and dominates all children of `node`.
    ///
    /// Returns:
    ///  * `NodeIndex` of the new node
    ///  * `Err` if `node` is a terminal. The tree structure is unchanged when `Err` is returned.
    pub fn insert_unary_below<S>(
        &mut self,
        node: NodeIndex,
        node_label: S,
    ) -> Result<NodeIndex, Error>
    where
        S: Into<String>,
    {
        if self[node].is_terminal() {
            return Err(format_err!("Can't attach nodes to terminals."));
        }
        // collect children before attaching new node
        let children = self.children(node).collect::<Vec<_>>();
        let span = self[node].span().to_owned();
        let new_node = NonTerminal::new(node_label, span).into();
        let insert = self.graph.add_node(new_node);
        self.graph.add_edge(node, insert, Edge::default());
        for (child, edge) in children {
            self.graph.remove_edge(edge).unwrap();
            self.graph.add_edge(insert, child, Edge::default());
        }
        Ok(insert)
    }

    /// Remove a node.
    ///
    /// This method will remove a node and attach all its children to the node above.
    ///
    /// Returns `Err` without mutating the tree if the structure is broken by the removal:
    ///   * The last node can't be removed.
    ///   * The root can't be removed if it has more than one outgoing edge.
    ///   * `Terminal`s can only be removed if they are not the last node in the branch.
    ///
    /// Otherwise return `Ok(node)`.
    ///
    /// Panics if the node is not in the tree.
    ///
    /// Removing a `Terminal` is fairly expensive since the spans for all nodes in the `Tree` will
    /// be recalculated. Removing a `NonTerminal` is cheaper, the outgoing edges of the removed
    /// node get attached to its parent, not changing the spans at all.
    pub fn remove_node(&mut self, node: NodeIndex) -> Result<Node, Error> {
        assert!(
            self.graph.contains_node(node),
            "Can't remove node that's not in the tree."
        );
        if self[node].is_terminal() {
            self.remove_terminal(node)
        } else {
            self.remove_nonterminal(node)
        }
    }

    /// Reattach a node.
    ///
    /// Remove `edge` and reattach the node to `new_parent` with an empty edge weight. This method
    /// does not change the position of the reattached node wrt. linear order in the sentence.
    ///
    /// Returns `Err` if:
    ///   * `edge` is the last outgoing edge of a node.
    ///   * `new_parent` is a `Terminal` node.
    ///
    /// Returns `Ok(old_edge)` otherwise.
    ///
    /// Panics if any of the indices is not present in the tree.
    pub fn reattach_node(
        &mut self,
        new_parent: NodeIndex,
        edge: EdgeIndex,
    ) -> Result<(EdgeIndex, Edge), Error> {
        assert!(
            self.graph.contains_node(new_parent),
            "Reattachment point has to be in the tree."
        );
        assert!(
            self.graph.edge_weight(edge).is_some(),
            "Edge to be removed has to be in the tree."
        );

        // ensure we're not removing the last child of a node and that the attachment point is NT
        let (parent, child) = self.graph.edge_endpoints(edge).unwrap();
        if self.siblings(child).count() == 0 && parent != new_parent {
            return Err(format_err!("Last child of a node."));
        } else if self[new_parent].is_terminal() {
            return Err(format_err!("Terminal node as new parent."));
        } else if child == new_parent {
            return Err(format_err!("New parent is node itself."));
        }

        let mut climber = Climber::new(parent, self);
        // climb tree to check if the old parent is dominated by the new parent
        while let Some(node) = climber.next(self) {
            // if new parent is higher in the tree, we need to remove the indices from the old
            // parent's span
            if new_parent == node {
                let coverage = self[child].span().into_iter().collect::<Vec<_>>();
                let before = self[parent].continuity();
                let after = self[parent]
                    .nonterminal_mut()
                    .unwrap()
                    .remove_indices(coverage);
                self.projectivity_change(before, after);
                break;
            }
        }
        let edge = self.graph.remove_edge(edge).unwrap();
        let edge_idx = self.graph.add_edge(new_parent, child, Edge::default());
        let child_span = self[child].span().to_owned();

        let mut climber = Climber::new(child, self);
        while let Some(parent) = climber.next(self) {
            // if new parent doesn't cover the child's span, extend new parent's span.
            if self.graph[parent].span().covers_span(&child_span) {
                break;
            } else {
                let before = self[parent].nonterminal().unwrap().continuity();
                // extending the new parent's span can change projectivity
                let after = self[parent]
                    .nonterminal_mut()
                    .unwrap()
                    .merge_spans(&child_span);
                self.projectivity_change(before, after);
            };
        }

        Ok((edge_idx, edge))
    }

    /// Get an iterator over `node`'s siblings.
    pub fn siblings<'a>(
        &'a self,
        node: NodeIndex,
    ) -> Box<Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        if let Some((parent, _)) = self.parent(node) {
            Box::new(
                self.children(parent)
                    .filter(move |(target, _)| *target != node),
            )
        } else {
            Box::new(std::iter::empty::<(NodeIndex, EdgeIndex)>())
        }
    }

    /// Get an iterator over `node`'s descendent.
    pub fn descendent_terminals(&self, node: NodeIndex) -> TerminalDescendents<FixedBitSet> {
        TerminalDescendents::new(self, node)
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
    pub fn is_projective(&self) -> bool {
        self.num_discontinuous == 0
    }

    /// Project indices of NonTerminals onto Terminals.
    ///
    /// Projects the `NodeIndex` of `NonTerminal`s matched by `match_fn` into a vector of length
    /// `n_terminals`. The method climbs the tree starting at each terminal and applies the closure
    /// at each step. If the closure returns `true`, the index of the currently considered
    /// `NonTerminal` is assigned to the corresponding slot in the vector.
    ///
    /// Contrary to the other projection method, this method defaults to assigning the root node's
    /// index if no other `NonTerminal` was matched.
    ///
    /// Returns a `Vec<NodeIndex>`.
    pub fn project_nt_indices<F>(&self, match_fn: F) -> Vec<NodeIndex>
    where
        F: Fn(&Tree, NodeIndex) -> bool,
    {
        let mut terminals = self.terminals().collect::<Vec<_>>();
        self.sort_indices(&mut terminals);
        let mut indices = vec![self.root; self.n_terminals];
        for terminal in terminals {
            let mut climber = Climber::new(terminal, self);
            while let Some(parent) = climber.next(self) {
                if match_fn(self, parent) {
                    indices[self[terminal].span().start] = parent;
                    break;
                }
            }
        }
        indices
    }

    /// Apply closure at each NonTerminal while climbing up the tree.
    ///
    /// Climbs up the tree starting at each terminal and applies the closure at each step.
    /// If the closure returns `true`, climbing is stopped and the method advances to the
    /// next terminal.
    ///
    /// Order of terminals does not necessarily correspond to linear order in the sentence.
    ///
    /// The closure takes the Tree, the current NonTerminal's index and the current Terminal's
    /// index as arguments. The Tree structure can also be mutated in this method but it won't
    /// be reflected in the path if the mutation happens just above or below the current node.
    ///
    /// E.g.
    /// ```
    /// use lumberjack::Tree;
    ///
    /// /// Remove all terminals' ancestors starting with `"A"`.
    /// ///
    /// /// Terminates upon finding the first ancestor starting with `"B"`.
    /// fn climb_map(tree: &mut Tree) {
    ///     tree.map_climber_path(|tree, ancestor_idx, terminal_idx| {
    ///         if tree[ancestor_idx].label().starts_with("A") {
    ///             tree.remove_node(ancestor_idx).unwrap();
    ///         } else if tree[ancestor_idx].label().starts_with("B") {
    ///             let label = tree[ancestor_idx].label().to_owned();
    ///             tree[terminal_idx]
    ///                 .features_mut()
    ///                 .insert(
    ///                     "b_ancestor",
    ///                     Some(label),
    ///                 );
    ///             return true
    ///         }
    ///         false
    ///     })
    /// }
    /// ```
    pub fn map_climber_path<F>(&mut self, mut match_fn: F)
    where
        F: FnMut(&mut Tree, NodeIndex, NodeIndex) -> bool,
    {
        let terminals = self.terminals().collect::<Vec<_>>();
        for terminal in terminals {
            let mut climber = Climber::new(terminal, self);
            while let Some(parent) = climber.next(self) {
                if match_fn(self, parent, terminal) {
                    break;
                }
            }
        }
    }

    /// Project labels of NonTerminals onto Terminals.
    ///
    /// Labels will be annotated under `feature_name` in each terminal's `Features`.
    ///
    /// The method starts climbing the tree from each terminal and will stop climbing as soon as
    /// the `match_fn` returns `true`. If it never returns `true` *no* feature will be annotated.
    ///
    /// E.g.
    /// ```
    /// use lumberjack::Tree;
    ///
    /// /// Annotate ancestor tags of the closest ancestor starting with `"A"` on terminals.
    /// fn project_tags(tree: &mut Tree) {
    ///     tree.project_tag_set("tag", |tree, ancestor_nt| {
    ///         tree[ancestor_nt].label().starts_with("A")
    ///     })
    /// }
    /// ```
    pub fn project_tag_set<F>(&mut self, feature_name: &str, match_fn: F)
    where
        F: Fn(&Tree, NodeIndex) -> bool,
    {
        self.map_climber_path(|tree, nt, t| {
            if match_fn(tree, nt) {
                let label = tree[nt].label().to_owned();
                tree[t].features_mut().insert(feature_name, Some(label));
                true
            } else {
                false
            }
        });
    }

    /// Project unique IDs for nodes with label in `tag_set` onto terminals.
    ///
    /// IDs will be annotated under `feature_name` in each terminal's `Features`.
    ///
    /// The method starts climbing the tree from each terminal and will stop climbing as soon as
    /// the `match_fn` returns `true`. If it never returns `true` *no* feature will be annotated.
    ///
    /// E.g.
    /// ```
    /// use lumberjack::Tree;
    ///
    /// /// Add feature `"id"` with a unique identifiers on terminals.
    /// ///
    /// /// Adds features with unique ID for the hierarchically closest NonTerminal starting with
    /// /// `"A"` on the Terminal nodes.
    /// fn project_unique_ids(tree: &mut Tree) {
    ///     tree.project_ids("id", |tree, nonterminal| {
    ///         tree[nonterminal].label().starts_with("A")
    ///     });
    /// }
    /// ```
    pub fn project_ids<F>(&mut self, feature_name: &str, match_fn: F)
    where
        F: Fn(&Tree, NodeIndex) -> bool,
    {
        let mut mapping = HashMap::new();
        self.map_climber_path(|tree, nt, t| {
            if match_fn(tree, nt) {
                let n = mapping.len();
                let id = *mapping.entry(nt).or_insert(n);
                tree[t]
                    .features_mut()
                    .insert(feature_name, Some(id.to_string()));
                true
            } else {
                false
            }
        });
    }
}

// (crate) private methods
impl Tree {
    pub(crate) fn new_from_parts(
        graph: StableGraph<Node, Edge>,
        n_terminals: usize,
        root: NodeIndex,
        num_non_projective: usize,
    ) -> Self {
        Tree {
            graph,
            n_terminals,
            root,
            num_discontinuous: num_non_projective,
        }
    }

    pub(crate) fn projectivity_change(&mut self, before: Continuity, after: Continuity) -> usize {
        match (before, after) {
            (Continuity::Continuous, Continuity::Discontinuous) => self.num_discontinuous += 1,
            (Continuity::Discontinuous, Continuity::Continuous) => self.num_discontinuous -= 1,
            _ => (),
        };
        self.num_discontinuous
    }

    /// Remove nonterminal node from the graph.
    ///
    /// Panics if a terminal node is given as argument.
    fn remove_nonterminal(&mut self, nonterminal: NodeIndex) -> Result<Node, Error> {
        assert!(
            !self[nonterminal].is_terminal(),
            "Remove Terminals with Tree::remove_terminal()"
        );
        if let Some((parent, _)) = self.parent(nonterminal) {
            for (child, _) in self.children(nonterminal).collect::<Vec<_>>() {
                self.graph.add_edge(parent, child, Edge::default());
            }
        } else if self.children(self.root).count() == 1 {
            let (child, _) = self.children(self.root).next().unwrap();
            self.root = child;
        } else {
            return Err(format_err!("Root has multiple outgoing edges."));
        }

        let node = self.graph.remove_node(nonterminal).unwrap();
        if node.span().skips().is_some() {
            self.num_discontinuous -= 1;
        }
        Ok(node)
    }

    /// Remove a terminal from the graph.
    ///
    /// Panics if a nonterminal node is given as argument.
    fn remove_terminal(&mut self, terminal: NodeIndex) -> Result<Node, Error> {
        assert!(
            self[terminal].is_terminal(),
            "Remove NonTerminals with Tree::remove_nonterminal()"
        );
        if self.siblings(terminal).count() == 0 {
            return Err(format_err!("Last terminal of a branch."));
        }
        self.compact_terminal_spans()?;
        self.reset_nt_spans();
        self.n_terminals -= 1;
        Ok(self.graph.remove_node(terminal).unwrap())
    }

    /// Reset terminal spans to increasing by 1 at each step.
    ///
    /// If a span was skipped because of e.g. removed terminals, restore the correct order.
    /// If two terminals with same span are present, alphabetical order of forms is used as the
    /// tie-breaker.
    pub(crate) fn compact_terminal_spans(&mut self) -> Result<(), Error> {
        let mut terminals = self.terminals().collect::<Vec<_>>();
        self.sort_indices(&mut terminals);
        for (idx, term) in terminals.into_iter().enumerate() {
            self[term].set_span(idx)?;
        }
        Ok(())
    }

    /// Resets nonterminal spans based on terminal spans.
    pub(crate) fn reset_nt_spans(&mut self) {
        let mut dfs = DfsPostOrder::new(&self.graph, self.root);
        self.num_discontinuous = 0;
        while let Some(node) = dfs.next(&self.graph) {
            if !self[node].is_terminal() {
                let coverage = self
                    .children(node)
                    .flat_map(|(child, _)| self[child].span().into_iter())
                    .collect::<Vec<_>>();
                let span = Span::from_vec(coverage).unwrap();
                if span.skips().is_some() {
                    self.num_discontinuous += 1;
                }
                self[node].nonterminal_mut().unwrap().set_span(span);
            }
        }
    }

    // helper method to sort a vec of node indices
    // order is determined by:
    // 1. lower bound of span (starting point of span)
    // 2. upper bound of span (end point of span)
    // 3. number of covered indices by span
    // 4. Inner nodes before terminal nodes
    // 5. alphabetical order
    pub(crate) fn sort_indices(&self, indices: &mut Vec<NodeIndex>) {
        indices.sort_by(
            |node1, node2| match self[*node1].span().cmp(&self[*node2].span()) {
                Ordering::Equal => match (&self[*node1], &self[*node2]) {
                    (Node::NonTerminal(_), Node::Terminal(_)) => Ordering::Greater,
                    (Node::Terminal(_), Node::NonTerminal(_)) => Ordering::Less,
                    (Node::NonTerminal(nt1), Node::NonTerminal(nt2)) => {
                        let order = self
                            .children(*node1)
                            .count()
                            .cmp(&self.children(*node2).count());
                        if order != Ordering::Equal {
                            order
                        } else {
                            nt1.label().cmp(nt2.label())
                        }
                    }
                    (Node::Terminal(t1), Node::Terminal(t2)) => t1.form().cmp(t2.form()),
                },
                ordering => ordering,
            },
        );
    }
}

// recursively check if at each level the trees are identical.
fn subtree_eq(tree_1_idx: NodeIndex, tree_1: &Tree, tree_2_idx: NodeIndex, tree_2: &Tree) -> bool {
    let mut nodes1 = tree_1
        .children(tree_1_idx)
        .map(|(n, _)| n)
        .collect::<Vec<_>>();
    tree_1.sort_indices(&mut nodes1);
    let mut nodes2 = tree_2
        .children(tree_2_idx)
        .map(|(n, _)| n)
        .collect::<Vec<_>>();
    tree_2.sort_indices(&mut nodes2);

    for (node1, node2) in nodes1.into_iter().zip(nodes2) {
        if tree_1[node1] != tree_2[node2] {
            return false;
        }
        let p1 = tree_1
            .parent(node1)
            .map(|(parent_id, edge_id)| (&tree_1[parent_id], &tree_1[edge_id]));
        let p2 = tree_2
            .parent(node2)
            .map(|(parent_id, edge_id)| (&tree_2[parent_id], &tree_2[edge_id]));
        if p1 != p2 {
            return false;
        }

        if !subtree_eq(node1, tree_1, node2, tree_2) {
            return false;
        }
    }
    true
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

        subtree_eq(self.root(), self, other.root(), other)
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

/// Iterator over terminal descendents of a node.
///
/// The terminals are returned in no particular order.
pub struct TerminalDescendents<'a, VM> {
    tree: &'a Tree,
    dfs: Dfs<NodeIndex, VM>,
}

impl<'a> TerminalDescendents<'a, FixedBitSet> {
    fn new(tree: &'a Tree, node: NodeIndex) -> Self {
        TerminalDescendents {
            tree,
            dfs: Dfs::new(tree.graph(), node),
        }
    }
}

impl<'a, VM> Iterator for TerminalDescendents<'a, VM>
where
    VM: VisitMap<NodeIndex>,
{
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.dfs.next(self.tree.graph()) {
            if self.tree[node].is_terminal() {
                return Some(node);
            } else {
                continue;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use petgraph::prelude::{NodeIndex, StableGraph};

    use crate::io::PTBFormat;
    use crate::{Edge, Node, NonTerminal, Span, Terminal, Tree};

    #[test]
    fn construct_tree() {
        let ptb = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let from_ptb = PTBFormat::Simple.string_to_tree(ptb).unwrap();

        let mut tree = Tree::new("t1", "TERM1");
        let root = tree.root();
        tree.insert_unary_above(root, "ROOT");
        tree.push_terminal("t2", "TERM2").unwrap();
        let root = tree.root();
        tree.insert_unary_below(root, "FIRST").unwrap();
        tree.push_terminal("t3", "TERM3").unwrap();
        let t4 = tree.push_terminal("t4", "TERM4").unwrap();
        tree.insert_unary_above(t4, "SECOND");
        tree.push_terminal("t5", "TERM5").unwrap();
        assert_eq!(from_ptb, tree);
    }

    #[test]
    fn reset_spans() {
        let mut tree = some_tree();
        let nt_indices = tree.nonterminals().collect::<Vec<_>>();
        for nt in nt_indices {
            tree[nt]
                .nonterminal_mut()
                .unwrap()
                .set_span(Span::from_vec(vec![0, 2]).unwrap());
        }
        tree.reset_nt_spans();
        assert_eq!(tree, some_tree());

        let mut tree = some_tree();
        let term_indices = tree.terminals().collect::<Vec<_>>();
        for (idx, term) in term_indices.into_iter().enumerate() {
            tree[term].set_span(idx + 1).unwrap();
        }
        tree.compact_terminal_spans().unwrap();
        assert_eq!(tree, some_tree());
    }

    #[test]
    fn project_node_labels() {
        let mut tree = Tree::new("t1", "TERM1");
        let t1 = tree.root();
        tree.insert_unary_above(t1, "ROOT");
        let first = tree.insert_unary_above(t1, "FIRST");
        tree.push_terminal("t2", "TERM2").unwrap();
        tree.insert_terminal(first, Terminal::new("t3", "TERM", 2));
        let t4 = tree.push_terminal("t4", "TERM4").unwrap();
        let second = tree.insert_unary_above(t4, "SECOND");
        tree.insert_terminal(second, Terminal::new("t4", "TERM4", 4));

        tree.project_tag_set("proj", |tree, nt| {
            tree[nt].label() == "FIRST" || nt == tree.root()
        });
        let features = tree
            .terminals()
            .map(|terminal| tree[terminal].features().unwrap().to_string())
            .collect::<Vec<_>>();
        let target = vec![
            "proj:FIRST",
            "proj:ROOT",
            "proj:FIRST",
            "proj:ROOT",
            "proj:ROOT",
        ];
        assert_eq!(features, target)
    }

    #[test]
    fn project_node_indices_nonprojective() {
        let mut tree = Tree::new("t1", "TERM1");
        let t1 = tree.root();
        let root_idx = tree.insert_unary_above(t1, "ROOT");
        let first_idx = tree.insert_unary_above(t1, "FIRST");
        tree.push_terminal("t2", "TERM2").unwrap();
        tree.insert_terminal(first_idx, Terminal::new("t3", "TERM3", 2));
        let t4 = tree.push_terminal("t4", "TERM4").unwrap();
        tree.insert_unary_above(t4, "SECOND");
        tree.push_terminal("t5", "TERM5").unwrap();
        let indices = tree.project_nt_indices(|tree, nt| tree[nt].label() == "FIRST");
        let target = vec![first_idx, root_idx, first_idx, root_idx, root_idx];
        assert_eq!(indices, target)
    }

    #[test]
    fn project_node_indices() {
        let mut tree = Tree::new("t1", "TERM1");
        let t1 = tree.root();
        let root_idx = tree.insert_unary_above(t1, "ROOT");
        tree.push_terminal("t2", "TERM2").unwrap();
        let root = tree.root();
        let first_idx = tree.insert_unary_below(root, "FIRST").unwrap();
        tree.push_terminal("t3", "TERM3").unwrap();
        let t4 = tree.push_terminal("t4", "TERM4").unwrap();
        tree.insert_unary_above(t4, "SECOND");
        tree.push_terminal("t5", "TERM5").unwrap();

        let indices = tree.project_nt_indices(|tree, nt| tree[nt].label() == "FIRST");
        let target = vec![first_idx, first_idx, root_idx, root_idx, root_idx];
        assert_eq!(indices, target)
    }

    #[test]
    fn project_node_ids() {
        let mut tree = Tree::new("t1", "TERM1");
        let t1 = tree.root();
        tree.insert_unary_above(t1, "ROOT");
        let first = tree.insert_unary_above(t1, "L");
        tree.push_terminal("t2", "TERM2").unwrap();
        tree.insert_terminal(first, Terminal::new("t3", "TERM", 2));
        let t4 = tree.push_terminal("t4", "TERM4").unwrap();
        tree.insert_unary_above(t4, "L");
        tree.push_terminal("t5", "TERM5").unwrap();

        tree.project_ids("id", |tree, nt| {
            tree[nt].label() == "L" || tree.root() == nt
        });
        let features = tree
            .terminals()
            .map(|terminal| tree[terminal].features().unwrap().to_string())
            .collect::<Vec<_>>();
        // 0 is first "L", 1 is ROOT and 2 is second "L"
        let target = vec!["id:0", "id:1", "id:0", "id:2", "id:1"];
        assert_eq!(features, target)
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
    fn insert_unary_above() {
        let ptb = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let root = tree.root();
        let new_root = tree.insert_unary_above(root, "NewRoot");
        assert_eq!(tree.root(), new_root);
        assert_eq!(
            tree[tree.root()],
            NonTerminal::new("NewRoot", Span::new(0, 5)).into()
        );
        assert_eq!("(NewRoot (ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5)))",
                   PTBFormat::Simple.tree_to_string(&tree).unwrap());
    }

    #[test]
    fn insert_terminal() {
        let ptb = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        tree.insert_terminal(NodeIndex::new(1), Terminal::new("t13", "TERM13", 2));
        assert_eq!("(ROOT (FIRST (TERM1 t1) (TERM2 t2) (TERM13 t13)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))",
                   PTBFormat::Simple.tree_to_string(&tree).unwrap());
        assert!(tree.is_projective());
        tree.insert_terminal(NodeIndex::new(0), Terminal::new("non_proj", "NONPROJ", 1));
        assert!(!tree.is_projective());
    }

    #[test]
    fn push_terminal() {
        let ptb = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let terminal = tree.push_terminal("pushed", "PUSH").unwrap();
        assert_eq!(
            tree[terminal],
            Node::from(Terminal::new("pushed", "PUSH", 5))
        );
        assert_eq!("(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5) (PUSH pushed))",
                   PTBFormat::Simple.tree_to_string(&tree).unwrap());
    }

    #[test]
    fn move_terminal() {
        let ptb = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        tree.move_terminal(NodeIndex::new(6), 4);
        assert_eq!(
            "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (TERM5 t5) (SECOND (TERM4 t4)))",
            PTBFormat::Simple.tree_to_string(&tree).unwrap()
        );
        tree.move_terminal(NodeIndex::new(6), 3);
        assert_eq!(ptb, PTBFormat::Simple.tree_to_string(&tree).unwrap());
        tree.move_terminal(NodeIndex::new(6), 0);
        assert_eq!(
            "(ROOT (SECOND (TERM4 t4)) (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (TERM5 t5))",
            PTBFormat::Simple.tree_to_string(&tree).unwrap()
        );
        let root = tree.root();
        let (_, edge) = tree.parent(NodeIndex::new(2)).unwrap();
        tree.reattach_node(root, edge).unwrap();
        tree.move_terminal(NodeIndex::new(2), 0);
        assert_eq!(
            "(ROOT (TERM1 t1) (SECOND (TERM4 t4)) (FIRST (TERM2 t2)) (TERM3 t3) (TERM5 t5))",
            PTBFormat::Simple.tree_to_string(&tree).unwrap()
        );
    }

    #[test]
    fn insert_unary_below() {
        let ptb = "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let root = tree.root();
        let below_root = tree.insert_unary_below(root, "BelowRoot").unwrap();
        let (root_child, _) = tree.children(tree.root()).next().unwrap();
        assert_eq!(below_root, root_child);
        assert_eq!(
            tree[root_child],
            NonTerminal::new("BelowRoot", Span::new(0, 5)).into()
        );
        assert_eq!("(ROOT (BelowRoot (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5)))",
                   PTBFormat::Simple.tree_to_string(&tree).unwrap());
    }

    #[test]
    fn rm_terminal() {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = some_tree();
        let terminals = tree.terminals().collect::<Vec<_>>();
        let t3 = tree.remove_node(terminals[2]).unwrap();
        assert_eq!(t3, Node::Terminal(Terminal::new("t3", "TERM3", 2)));
        assert!(tree.remove_node(terminals[3]).is_err());
    }

    #[test]
    fn rm_nonterminal() {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = some_tree();
        let nt = tree
            .nonterminals()
            .filter(|nt| tree[*nt].label() == "SECOND")
            .next()
            .unwrap();

        let nt = tree.remove_node(nt).unwrap();
        assert_eq!(nt, Node::NonTerminal(NonTerminal::new("SECOND", 3)));
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree).unwrap(),
            "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (TERM4 t4) (TERM5 t5))"
        );
    }

    #[test]
    fn rm_minimal_tree() {
        let mut tree = PTBFormat::Simple.string_to_tree("(ROOT (T t))").unwrap();
        let root = tree.root();
        let root = tree.remove_node(root).unwrap();
        assert_eq!(root, Node::NonTerminal(NonTerminal::new("ROOT", 0)));
        assert_eq!(
            &tree[tree.root()],
            &Node::Terminal(Terminal::new("t", "T", 0))
        );
        assert_eq!(PTBFormat::Simple.tree_to_string(&tree).unwrap(), "(T t)");
    }

    #[test]
    fn rm_fail_last_node() {
        let mut tree = PTBFormat::Simple.string_to_tree("(T t)").unwrap();
        let root = tree.root();
        assert!(tree.remove_node(root).is_err());
    }

    #[test]
    fn rm_root_multiple_attached() {
        let mut tree = PTBFormat::Simple
            .string_to_tree("(ROOT (T t) (T2 t2))")
            .unwrap();
        let root = tree.root();
        assert!(tree.remove_node(root).is_err());
    }

    #[test]
    fn reattach_terminal_to_same_parent() {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = some_tree();
        let terminals = tree.terminals().collect::<Vec<_>>();
        let (parent, edge) = tree.parent(terminals[0]).unwrap();
        assert_eq!(Edge::default(), tree.reattach_node(parent, edge).unwrap().1);
        assert_eq!(tree, some_tree());
    }

    #[test]
    fn reattach_terminal() {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = some_tree();
        let terminals = tree.terminals().collect::<Vec<_>>();
        let (_, edge) = tree.parent(terminals[0]).unwrap();
        let root = tree.root();
        assert_eq!(Edge::default(), tree.reattach_node(root, edge).unwrap().1);
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree).unwrap(),
            "(ROOT (TERM1 t1) (FIRST (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))"
        );
    }

    #[test]
    fn reattach_terminal_to_terminal() {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = some_tree();
        let terminals = tree.terminals().collect::<Vec<_>>();
        let (_, edge) = tree.parent(terminals[0]).unwrap();
        assert!(tree.reattach_node(terminals[1], edge).is_err());
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree).unwrap(),
            "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))"
        );
    }

    #[test]
    fn reattach_last_terminal() {
        let mut tree = some_tree();
        let terminals = tree.terminals().collect::<Vec<_>>();
        let (_, edge) = tree.parent(terminals[3]).unwrap();
        let root = tree.root();
        assert!(tree.reattach_node(root, edge).is_err());
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree).unwrap(),
            "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))"
        );
    }

    #[test]
    fn reattach_nt_to_self() {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = some_tree();
        let nt = tree
            .nonterminals()
            .filter(|nt| tree[*nt].label() == "FIRST")
            .next()
            .unwrap();
        let (_, edge) = tree.parent(nt).unwrap();
        assert!(tree.reattach_node(nt, edge).is_err());
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree).unwrap(),
            "(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))"
        )
    }

    #[test]
    fn reattach_nt_projective() {
        let mut tree = PTBFormat::Simple.string_to_tree("(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (MOVE (TERM3 t3)) (SECOND (TERM4 t4)) (TERM5 t5))").unwrap();
        let nt = tree
            .nonterminals()
            .filter(|nt| tree[*nt].label() == "MOVE")
            .next()
            .unwrap();
        let target = tree
            .nonterminals()
            .filter(|nt| tree[*nt].label() == "FIRST")
            .next()
            .unwrap();
        let (_, edge) = tree.parent(nt).unwrap();
        assert_eq!(Edge::default(), tree.reattach_node(target, edge).unwrap().1);
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree).unwrap(),
            "(ROOT (FIRST (TERM1 t1) (TERM2 t2) (MOVE (TERM3 t3))) (SECOND (TERM4 t4)) (TERM5 t5))"
        )
    }

    #[test]
    fn reattach_nonprojective() {
        let mut graph = StableGraph::new();
        let root = graph.add_node(NonTerminal::new("ROOT", Span::new(0, 3)).into());
        let a = graph.add_node(NonTerminal::new("A", Span::from_vec(vec![0, 2]).unwrap()).into());
        let a_term = graph.add_node(Terminal::new("a", "a_term", 0).into());
        let b_term = graph.add_node(Terminal::new("b", "b_term", 2).into());
        let c_term = graph.add_node(Terminal::new("c", "c_term", 1).into());
        let a_nt_edge = graph.add_edge(root, a, Edge::default());
        let c_edge = graph.add_edge(root, c_term, Edge::default());
        graph.add_edge(a, a_term, Edge::default());
        let b_edge = graph.add_edge(a, b_term, Edge::default());
        let non_proj = Tree::new_from_parts(graph, 3, root, 1);
        let mut tree = non_proj.clone();

        tree.reattach_node(a, c_edge).unwrap();
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree).unwrap(),
            "(ROOT (A (a_term a) (c_term c) (b_term b)))"
        );
        let (_, edge) = tree.parent(c_term).unwrap();
        tree.reattach_node(root, edge).unwrap();
        assert_eq!(non_proj, tree);

        let mut tree_2 = non_proj.clone();
        tree_2.reattach_node(root, b_edge).unwrap();
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree_2).unwrap(),
            "(ROOT (A (a_term a)) (c_term c) (b_term b))"
        );

        let mut tree_2 = non_proj.clone();
        tree_2.reattach_node(a, c_edge).unwrap();
        tree_2.reattach_node(root, a_nt_edge).unwrap();
        assert_eq!(
            PTBFormat::Simple.tree_to_string(&tree_2).unwrap(),
            "(ROOT (A (a_term a) (c_term c) (b_term b)))"
        );
    }

    #[test]
    fn siblings() {
        let tree = some_tree();
        assert!(tree.siblings(tree.root()).next().is_none());
        // NodeIndex(1) is Node::Inner("FIRST" ..)
        let siblings = tree
            .siblings(NodeIndex::new(3))
            .into_iter()
            .map(|(sibling, _)| match tree.graph()[sibling] {
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
        let term2 = Terminal::new("t2", "TERM2", 1);
        let root = NonTerminal::new("ROOT", Span::new(0, 5));
        let first = NonTerminal::new("FIRST", Span::new(0, 2));
        let second = NonTerminal::new("SECOND", Span::new(3, 4));
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
        let mut other_tree = Tree::new_from_parts(g.clone(), 5, root_idx, 0);
        assert_eq!(some_tree, other_tree);
        other_tree[term2_idx]
            .terminal_mut()
            .unwrap()
            .set_lemma(Some("some_lemma"));
        assert_ne!(some_tree, other_tree);
        g.remove_node(term2_idx);
        let other_tree = Tree::new_from_parts(g.clone(), 4, root_idx, 0);
        assert_ne!(some_tree, other_tree);
        let new_t2_idx = g.add_node(Node::Terminal(term2));
        g.add_edge(first_idx, new_t2_idx, Edge::default());
        let other_tree = Tree::new_from_parts(g.clone(), 5, root_idx, 0);
        assert_eq!(some_tree, other_tree);
    }

    fn some_tree() -> Tree {
        //(ROOT (FIRST (TERM1 t1) (TERM2 t2)) (TERM3 t3) (SECOND (TERM4 t4)) (TERM5 t5))";
        let mut tree = Tree::new("t1", "TERM1");
        let root = tree.root();
        tree.insert_unary_above(root, "ROOT");
        tree.push_terminal("t2", "TERM2").unwrap();
        let root = tree.root();
        tree.insert_unary_below(root, "FIRST").unwrap();
        tree.push_terminal("t3", "TERM3").unwrap();
        let t4 = tree.push_terminal("t4", "TERM4").unwrap();
        tree.insert_unary_above(t4, "SECOND");
        tree.push_terminal("t5", "TERM5").unwrap();
        tree
    }
}
