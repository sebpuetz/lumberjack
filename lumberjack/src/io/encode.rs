///! Encoding Module
///
/// This module contains structs, traits and methods required to:
///
/// * convert trees to a sequence of labels
/// * construct trees from such sequences.
///
/// The trees can be encoded on an absolute and relative scale.
///
/// See Gómez-Rodríguez and Vilares (2018).
use std::convert::TryFrom;
use std::ops::Deref;
use std::vec::IntoIter;

use failure::Error;

use crate::tree::Tree;
use crate::tree_modification::TreeOps;
use crate::util::Climber;
use crate::{Edge, Node, NonTerminal, Terminal};
use petgraph::prelude::{NodeIndex, StableGraph};
use petgraph::Direction;

static EMPTY_NODE: &str = "DUMMY";

/// Encode trait.
///
/// Defines methods that encode a tree through a sequence of labels.
///
/// All methods expect trees without unary chains. Unary chains can get collapsed through
/// `TreeOps::collapse_unary_chains`.
///
/// All methods return `Error` on non-projective trees.
pub trait Encode {
    /// Absolute scale encoding.
    ///
    /// Encode a tree as a sequence of labels where each label is a tuple consisting of the lowest
    /// common ancestor of the current terminal with the next terminal in the tree and
    /// an optional leaf-unary-chain. Common ancestors consist of a `(label, num_common)` tuple,
    /// where `num_common` describes the number of common ancestors.
    fn encode_absolute(&self) -> Result<AbsoluteEncoding, Error>;

    /// Annotate absolute scale tags as features.
    ///
    /// The feature key is `"abs_ancestor"`.
    fn annotate_absolute(&mut self) -> Result<(), Error>;

    /// Relative scale encoding.
    ///
    /// Similar to the absolute scale encoding, each terminal also gets assigned a tuple consisting
    /// of the lowest common ancestor with the following terminal and an optional leaf-unary-chain.
    fn encode_relative(&self) -> Result<RelativeEncoding, Error>;

    /// Annotate relative scale tags as features.
    ///
    /// The feature key is `"rel_ancestor"`.
    fn annotate_relative(&mut self) -> Result<(), Error>;
}

impl Encode for Tree {
    fn encode_absolute(&self) -> Result<AbsoluteEncoding, Error> {
        if !self.is_projective() {
            return Err(format_err!("Can't encode nonprojective tree."));
        }
        let terminals = self.terminals().collect::<Vec<_>>();
        let mut encoding = Vec::with_capacity(terminals.len());
        for terminal in terminals.into_iter() {
            let common_nt = get_common(self, terminal)?
                .map(|(common, n_common)| AbsoluteAncestor::new(n_common, common));

            let chain = self[terminal]
                .features()
                .and_then(|f| f.get_val("unary_chain").map(ToOwned::to_owned));
            encoding.push((common_nt, chain));
        }
        Ok(AbsoluteEncoding(encoding))
    }

    fn annotate_absolute(&mut self) -> Result<(), Error> {
        if !self.is_projective() {
            return Err(format_err!("Can't encode nonprojective tree."));
        }
        let terminals = self.terminals().collect::<Vec<_>>();
        for terminal in terminals.into_iter() {
            let mut string_rep = get_common(self, terminal)?
                .map(|(common, n_common)| AbsoluteAncestor::new(n_common, common).to_string())
                .unwrap_or_else(|| "NONE".to_string());
            if let Some(chain) = self[terminal]
                .features()
                .and_then(|f| f.get_val("unary_chain"))
            {
                string_rep = format!("{}~{}", string_rep, chain);
            };

            self[terminal]
                .features_mut()
                .insert("abs_ancestor", Some(string_rep));
        }
        Ok(())
    }

    fn encode_relative(&self) -> Result<RelativeEncoding, Error> {
        if !self.is_projective() {
            return Err(format_err!("Can't encode nonprojective tree."));
        }
        let mut prev_n = 0;
        let terminals = self.terminals().collect::<Vec<_>>();
        let mut encoding = Vec::with_capacity(terminals.len());
        for terminal in terminals.into_iter() {
            let common_nt = match get_common(self, terminal)? {
                Some((common, n_common)) => {
                    let ancestor = if n_common == 1 {
                        Some(RelativeAncestor::Root(common))
                    } else {
                        Some(RelativeAncestor::new(n_common as isize - prev_n, common))
                    };
                    prev_n = n_common as isize;
                    ancestor
                }
                None => None,
            };

            let chain = self[terminal]
                .features()
                .and_then(|f| f.get_val("unary_chain").map(ToOwned::to_owned));
            encoding.push((common_nt, chain));
        }
        Ok(RelativeEncoding(encoding))
    }

    fn annotate_relative(&mut self) -> Result<(), Error> {
        if !self.is_projective() {
            return Err(format_err!("Can't encode nonprojective tree."));
        }
        let terminals = self.terminals().collect::<Vec<_>>();
        let mut prev_n = 0;
        for terminal in terminals.into_iter() {
            let mut string_rep = get_common(self, terminal)?
                .map(|(common, n_common)| {
                    let ancestor = if n_common == 1 {
                        RelativeAncestor::Root(common)
                    } else {
                        RelativeAncestor::new(n_common as isize - prev_n, common)
                    };
                    prev_n = n_common as isize;
                    ancestor.to_string()
                })
                .unwrap_or_else(|| "NONE".to_string());

            if let Some(chain) = self[terminal]
                .features()
                .and_then(|f| f.get_val("unary_chain"))
            {
                string_rep = format!("{}~{}", string_rep, chain);
            };
            self[terminal]
                .features_mut()
                .insert("rel_ancestor", Some(string_rep));
        }
        Ok(())
    }
}

/// Helper method to get the lowest common ancestor.
fn get_common(tree: &Tree, terminal: NodeIndex) -> Result<Option<(String, usize)>, Error> {
    let idx = tree[terminal].span().start;
    let mut common = None;
    let mut n_common = 0usize;
    let mut climber = Climber::new(terminal, tree);
    while let Some(parent) = climber.next(tree) {
        let span = tree[parent].span();
        if span.skips().is_some() {
            return Err(format_err!("Can't get lowest common ancestor"));
        }
        if span.contains(idx + 1) && common.is_none() {
            let common_nt = tree[parent]
                .nonterminal()
                .ok_or_else(|| format_err!("Terminal without parent:\n{}", tree[parent]))?;
            let common_label =
                if let Some(chain) = common_nt.features().and_then(|f| f.get_val("unary_chain")) {
                    format!("{}_{}", chain, common_nt.label())
                } else {
                    common_nt.label().to_string()
                };

            common = Some(common_label);
        }
        if common.is_some() {
            n_common += 1
        }
    }
    Ok(common.map(|common| (common, n_common)))
}

/// Decode trait.
///
/// Defines methods how to construct trees from encodings.
pub trait Decode: Sized {
    /// Construct a tree from an absolute scale encoding.
    ///
    /// Decoding from a `RelativeEncoding` requires conversion into an `AbsoluteEncoding`:
    ///
    /// ```
    /// use lumberjack::Tree;
    /// use lumberjack::io::{AbsoluteEncoding, Encode, Decode, PTBFormat};
    ///
    /// let tree = PTBFormat::Simple.string_to_tree("(Some (Test tree))").unwrap();
    /// let terminals = tree.terminals().map(|t| tree[t].terminal().unwrap().clone()).collect::<Vec<_>>();
    /// let relative_encoding = tree.encode_relative().unwrap();
    /// let absolute = AbsoluteEncoding::try_from_relative(relative_encoding).unwrap_or_fix();
    /// let tree = Tree::decode(absolute, terminals);
    /// ```
    fn decode(encoding: AbsoluteEncoding, terminals: Vec<Terminal>) -> Self;

    fn remove_dummy_nodes(&mut self) -> Result<(), Error>;
}

impl Decode for Tree {
    fn decode(encoding: AbsoluteEncoding, terminals: Vec<Terminal>) -> Self {
        let encoding = encoding.0;
        let mut graph = StableGraph::new();
        let n_terminals = terminals.len();
        let root = Node::NonTerminal(NonTerminal::new(EMPTY_NODE, 0));
        let mut root_idx = graph.add_node(root);
        let mut prev = None;
        let mut prev_n = 0;
        for (idx, ((ancestor, unary_chain), mut terminal)) in
            encoding.into_iter().zip(terminals).enumerate()
        {
            let mut cur = root_idx;
            // leaf unary chains are encoded as part of the POS label of terminals.
            if let Some(unary_chain) = unary_chain {
                terminal
                    .features_mut()
                    .insert("unary_chain", Some(unary_chain));
            }
            let term_idx = graph.add_node(Node::Terminal(terminal));
            match ancestor {
                // None means either final token or direct attachment to the root
                // In case of direct attachment to the root, n_common for the previous token was
                // 1, thus prev_n is 0 and the loop below doesn't iterate, attachment is made
                // directly to the root.
                None => {
                    if n_terminals == 1 {
                        graph.remove_node(root_idx);
                        root_idx = term_idx;
                        break;
                    }
                    for _ in 0..prev_n {
                        if let Some(node) = graph
                            .neighbors_directed(cur, Direction::Outgoing)
                            .find(|n| !graph[*n].is_terminal())
                        {
                            cur = node;
                        } else {
                            break;
                        }
                    }
                    graph.add_edge(cur, term_idx, Edge::default());
                }
                Some(ancestor) => {
                    let (n_common, common_nt) = ancestor.into_parts();
                    for i in 1..n_common {
                        let child_idx = graph
                            .neighbors_directed(cur, Direction::Outgoing)
                            .next()
                            .filter(|n| !graph[*n].is_terminal());
                        // child_idx being none indicates no attachment at the current label
                        // if the current depth is deeper than n_common of the previous step,
                        // In both cases, insert new node.
                        if child_idx.is_none() || i > prev_n {
                            let nt = Node::NonTerminal(NonTerminal::new(EMPTY_NODE, idx));
                            let idx = graph.add_node(nt);
                            graph.add_edge(cur, idx, Edge::default());
                            cur = idx;
                        } else if let Some(node) = child_idx {
                            cur = node;
                        }
                    }

                    // '_' is used to delimit unary chains from actual label
                    if let Some(idx) = common_nt.rfind('_') {
                        let (chain, label) = common_nt.split_at(idx);
                        // Nodes in previous for loop are inserted with dummy label, cur indexes the
                        // nonterminal receiving the current label.
                        graph[cur].set_label(&label[1..]);
                        graph[cur].features_mut().insert("unary_chain", Some(chain));
                    } else {
                        graph[cur].set_label(common_nt);
                    }

                    if prev.is_none() || n_common > prev_n {
                        graph.add_edge(cur, term_idx, Edge::default());
                    } else {
                        graph.add_edge(prev.unwrap(), term_idx, Edge::default());
                    }

                    prev = Some(cur);
                    prev_n = n_common - 1;
                }
            }
        }

        let mut tree = Tree::new_from_parts(graph, n_terminals, root_idx, 0);
        tree.reset_nt_spans();
        tree
    }

    fn remove_dummy_nodes(&mut self) -> Result<(), Error> {
        self.filter_nonterminals(|tree, nt| tree[nt].label() != EMPTY_NODE)?;
        let root = self.root();
        if self[root]
            .nonterminal()
            .map(|nt| nt.label() == EMPTY_NODE)
            .unwrap_or(false)
            && self.children(root).count() == 1
        {
            self.remove_node(root)?;
        }
        Ok(())
    }
}

/// Common ancestor with absolute offset.
#[derive(Clone, Debug, PartialEq)]
pub struct AbsoluteAncestor {
    common: usize,
    label: String,
}
impl ToString for AbsoluteAncestor {
    fn to_string(&self) -> String {
        format!("{}+{}", self.label, self.common)
    }
}

impl AbsoluteAncestor {
    /// Construct a new `AbsoluteAncestor`.
    fn new<S>(n_common: usize, label: S) -> Self
    where
        S: Into<String>,
    {
        AbsoluteAncestor {
            common: n_common,
            label: label.into(),
        }
    }

    /// Get this ancestor's parts.
    pub fn into_parts(self) -> (usize, String) {
        (self.common, self.label)
    }

    /// Get the common label.
    pub fn label(&self) -> &str {
        self.label.as_str()
    }

    /// Get the number of common ancestors.
    pub fn n_common(&self) -> usize {
        self.common
    }
}

impl<'a> TryFrom<&'a str> for AbsoluteAncestor {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut parts = value.split('+');
        let label = parts
            .next()
            .ok_or_else(|| format_err!("Missing common label."))?;
        let offset = parts
            .next()
            .ok_or_else(|| format_err!("Missing offset."))?
            .parse::<usize>()?;
        Ok(AbsoluteAncestor::new(offset, label))
    }
}

/// Common ancestor with relative offset.
#[derive(Clone, Debug, PartialEq)]
pub enum RelativeAncestor {
    Regular { offset: isize, label: String },
    Root(String),
}

impl RelativeAncestor {
    fn new<S>(offset: isize, label: S) -> Self
    where
        S: Into<String>,
    {
        RelativeAncestor::Regular {
            offset,
            label: label.into(),
        }
    }
}

impl ToString for RelativeAncestor {
    fn to_string(&self) -> String {
        match self {
            RelativeAncestor::Regular { label, offset } => format!("{}+{}", label, offset),
            RelativeAncestor::Root(label) => label.to_string(),
        }
    }
}

impl<'a> TryFrom<&'a str> for RelativeAncestor {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut parts = value.split('+');
        let label = parts
            .next()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format_err!("Missing common label."))?;
        if let Some(offset) = parts.next() {
            let offset = offset.parse::<isize>()?;
            Ok(RelativeAncestor::new(offset, label))
        } else {
            Ok(RelativeAncestor::Root(label.to_owned()))
        }
    }
}

/// Absolute Encoding.
///
/// Trees are encoded as a sequence of tuples:
///  * For each token a the common nonterminal label with the following token and the number
///     of common nonterminal nodes.
///  * Optional leaf unary chains.
#[derive(Clone, Debug, PartialEq)]
pub struct AbsoluteEncoding(Vec<(Option<AbsoluteAncestor>, Option<String>)>);

impl AbsoluteEncoding {
    pub fn new(inner: Vec<(Option<AbsoluteAncestor>, Option<String>)>) -> Self {
        AbsoluteEncoding(inner)
    }

    /// Fallible conversion from `RelativeEncoding`.
    ///
    /// Returns:
    ///     * `ConversionResult::Success` if conversion was succesful or
    ///     * `ConversionResult::Error` if an error was encountered.
    ///
    /// Errors are recoverable, and can be fixed by e.g.:
    ///
    /// ```
    /// use lumberjack::io::{AbsoluteEncoding, Encode, PTBFormat};
    ///
    /// let tree = PTBFormat::Simple.string_to_tree("(Some (Test tree))").unwrap();
    /// let rel_encoding = tree.encode_relative().unwrap();
    /// let abs_encoding = AbsoluteEncoding::try_from_relative(rel_encoding).unwrap_or_fix();;
    /// ```
    pub fn try_from_relative(encoding: RelativeEncoding) -> ConversionResult {
        let mut prev_n = 0;
        let mut abs_encoding = Vec::with_capacity(encoding.0.len());
        let mut iter = encoding.0.into_iter();
        while let Some((ancestor, leaf_unary)) = iter.next() {
            if let Some(ancestor) = ancestor {
                let (n_common, label) = match ancestor {
                    RelativeAncestor::Regular { label, offset } => {
                        let n_common = offset + prev_n;
                        if n_common < 1 {
                            return ConversionResult::Error(RelativeToAbsoluteError {
                                iter,
                                prev_n,
                                label,
                                leaf_unary,
                                abs_encoding,
                            });
                        }
                        (n_common, label)
                    }
                    RelativeAncestor::Root(label) => (1, label),
                };
                let abs_ancestor = AbsoluteAncestor::new(n_common as usize, label);
                abs_encoding.push((Some(abs_ancestor), leaf_unary));
                prev_n = n_common;
            } else {
                abs_encoding.push((None, leaf_unary));
            }
        }
        ConversionResult::Success(AbsoluteEncoding(abs_encoding))
    }
}

/// Conversion Result.
///
/// This enum is used intentionally to replace `Result` since the `Error` variant is fixable and
/// typical usage of `Result` does not include matching the result.
///
/// E.g.:
///
/// ```
/// use lumberjack::io::{AbsoluteEncoding, Encode, PTBFormat};
/// let tree = PTBFormat::Simple.string_to_tree("(Some (Test tree))").unwrap();
/// let relative_encoding = tree.encode_relative().unwrap();
/// let absolute = AbsoluteEncoding::try_from_relative(relative_encoding).unwrap_or_fix();
/// ```
#[derive(Clone, Debug)]
pub enum ConversionResult {
    /// Succesful conversion result.
    Success(AbsoluteEncoding),
    /// Failed conversion result.
    Error(RelativeToAbsoluteError),
}

impl ConversionResult {
    /// Unwrap or fix the conversion result.
    ///
    /// Returns the converted `AbsoluteEncoding`.
    pub fn unwrap_or_fix(self) -> AbsoluteEncoding {
        match self {
            ConversionResult::Success(enc) => enc,
            ConversionResult::Error(e) => e.fix(),
        }
    }

    /// Convert into `Result`.
    pub fn into_result(self) -> Result<AbsoluteEncoding, RelativeToAbsoluteError> {
        match self {
            ConversionResult::Success(enc) => Ok(enc),
            ConversionResult::Error(e) => Err(e),
        }
    }
}

/// RelativeToAbsoluteError
///
/// Recoverable error during conversion from relative scale labels to absolute scale.
#[derive(Clone, Debug)]
pub struct RelativeToAbsoluteError {
    iter: IntoIter<(Option<RelativeAncestor>, Option<String>)>,
    prev_n: isize,
    leaf_unary: Option<String>,
    label: String,
    abs_encoding: Vec<(Option<AbsoluteAncestor>, Option<String>)>,
}

impl RelativeToAbsoluteError {
    /// Fix conversion in case of error.
    ///
    /// The conversion from relative scale to absolute scale can fail if the sequence of offsets
    /// leads to a negative absolute number of common ancestors. In that case, this method
    /// replaces all negative values with 2, which leads to attachment with a node with the given
    /// label below the root node.
    pub fn fix(mut self) -> AbsoluteEncoding {
        self.abs_encoding
            .push((Some(AbsoluteAncestor::new(2, self.label)), self.leaf_unary));
        self.prev_n = 2;
        for (ancestor, leaf_unary) in self.iter {
            if let Some(ancestor) = ancestor {
                let (n_common, label) = match ancestor {
                    RelativeAncestor::Regular { label, offset } => {
                        let mut n_common = offset + self.prev_n;
                        if n_common < 1 {
                            n_common = 2;
                        }
                        (n_common, label)
                    }
                    RelativeAncestor::Root(label) => (1, label),
                };
                let abs_ancestor = AbsoluteAncestor::new(n_common as usize, label);
                self.abs_encoding.push((Some(abs_ancestor), leaf_unary));
                self.prev_n = n_common;
            } else {
                self.abs_encoding.push((None, leaf_unary));
            }
        }

        AbsoluteEncoding(self.abs_encoding)
    }
}

impl From<AbsoluteEncoding> for RelativeEncoding {
    fn from(encoding: AbsoluteEncoding) -> Self {
        let mut prev_n = 0;
        let mut rel_encoding = Vec::with_capacity(encoding.len());
        for (ancestor, leaf_unary) in encoding.0 {
            if let Some(ancestor) = ancestor {
                let (n_common, label) = ancestor.into_parts();
                let offset = n_common as isize - prev_n;
                prev_n = n_common as isize;
                if n_common == 1 {
                    rel_encoding.push((Some(RelativeAncestor::Root(label)), leaf_unary));
                } else {
                    rel_encoding.push((Some(RelativeAncestor::new(offset, label)), leaf_unary))
                }
            } else {
                rel_encoding.push((None, leaf_unary))
            }
        }
        RelativeEncoding(rel_encoding)
    }
}

impl Deref for AbsoluteEncoding {
    type Target = Vec<(Option<AbsoluteAncestor>, Option<String>)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl IntoIterator for AbsoluteEncoding {
    type Item = (Option<AbsoluteAncestor>, Option<String>);
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Relative Encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct RelativeEncoding(Vec<(Option<RelativeAncestor>, Option<String>)>);

impl RelativeEncoding {
    pub(crate) fn new(inner: Vec<(Option<RelativeAncestor>, Option<String>)>) -> Self {
        RelativeEncoding(inner)
    }
}

impl Deref for RelativeEncoding {
    type Target = Vec<(Option<RelativeAncestor>, Option<String>)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod test {
    use std::convert::TryFrom;

    use super::{AbsoluteEncoding, Decode, RelativeEncoding};

    use crate::io::encode::{AbsoluteAncestor, ConversionResult, Encode, RelativeAncestor};
    use crate::io::PTBFormat;
    use crate::tree_modification::UnaryChains;
    use crate::Tree;

    #[test]
    fn test_try_from_str() {
        let ancestor = RelativeAncestor::try_from("Label+-1").unwrap();
        assert_eq!(ancestor, RelativeAncestor::new(-1, "Label"));
        assert_eq!(ancestor.to_string(), "Label+-1");
        let ancestor = RelativeAncestor::try_from("Label+1").unwrap();
        assert_eq!(ancestor, RelativeAncestor::new(1, "Label"));
        assert_eq!(ancestor.to_string(), "Label+1");
        let ancestor = RelativeAncestor::try_from("");
        assert!(ancestor.is_err());
        let ancestor = RelativeAncestor::try_from("Label+no_number");
        assert!(ancestor.is_err());
        let ancestor = RelativeAncestor::try_from("Root").unwrap();
        assert_eq!(ancestor, RelativeAncestor::Root("Root".into()));
        assert_eq!(ancestor.to_string(), "Root");

        let ancestor = AbsoluteAncestor::try_from("Label+-1");
        assert!(ancestor.is_err());
        let ancestor = AbsoluteAncestor::try_from("Label+1").unwrap();
        assert_eq!(ancestor.to_string(), "Label+1");
        assert_eq!(ancestor, AbsoluteAncestor::new(1, "Label"));
        let ancestor = AbsoluteAncestor::try_from("");
        assert!(ancestor.is_err());
        let ancestor = AbsoluteAncestor::try_from("Label+no_number");
        assert!(ancestor.is_err());
        let ancestor = AbsoluteAncestor::try_from("Root");
        assert!(ancestor.is_err());
    }

    #[test]
    fn encode_absolute() {
        // S and U1 get collapsed into U2 label, both terminals are attached in same manner.
        let ptb = "(S (U1 (U2 (T1 t1) (T2 t2))))";
        let mut tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let terminals = tree
            .terminals()
            .filter_map(|t| tree[t].terminal().cloned())
            .collect::<Vec<_>>();
        tree.collapse_unary_chains("_").unwrap();
        let encoding = tree.encode_absolute().unwrap();
        let target = AbsoluteEncoding(vec![
            (Some(AbsoluteAncestor::new(1, "U1_S_U2".to_string())), None),
            (None, None),
        ]);
        assert_eq!(target, encoding);

        let original_tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let mut reconstructed = Tree::decode(encoding, terminals);
        reconstructed.restore_unary_chains("_").unwrap();
        assert_eq!(original_tree, reconstructed);

        // leaf unary chains (NT1 (U1 (U2 ..))) and (R (U3 ..)) get collapsed
        let ptb = "(R (NT1 (U1 (U2 (T t))) (NT2 (NT3 (T t) (T t)) (U3 (T t)))) (T t))";
        let mut tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let terminals = tree
            .terminals()
            .filter_map(|t| tree[t].terminal().cloned())
            .collect::<Vec<_>>();
        tree.collapse_unary_chains("_").unwrap();
        let target = AbsoluteEncoding(vec![
            (
                Some(AbsoluteAncestor::new(2, "NT1")),
                Some("U2_U1".to_string()),
            ),
            (Some(AbsoluteAncestor::new(4, "NT3")), None),
            (Some(AbsoluteAncestor::new(3, "NT2")), None),
            (Some(AbsoluteAncestor::new(1, "R")), Some("U3".into())),
            (None, None),
        ]);
        let encoding = tree.encode_absolute().unwrap();
        assert_eq!(target, encoding);

        let original_tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let mut reconstructed = Tree::decode(encoding, terminals);
        reconstructed.restore_unary_chains("_").unwrap();
        assert_eq!(original_tree, reconstructed);

        // Tree as single unary chain
        let ptb = "(S (U (T t)))";
        let mut tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let terminals = tree
            .terminals()
            .filter_map(|t| tree[t].terminal().cloned())
            .collect::<Vec<_>>();
        tree.collapse_unary_chains("_").unwrap();
        let ancestor = (None, Some("U_S".into()));
        let encoding = tree.encode_absolute().unwrap();
        for enc in encoding.iter() {
            assert_eq!(enc, &ancestor);
        }
        let original_tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let mut reconstructed = Tree::decode(encoding, terminals);
        reconstructed.restore_unary_chains("_").unwrap();
        assert_eq!(original_tree, reconstructed);

        let ptb = "(S (T t) (NT (T2 t) (T3 t)))";
        let tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let terminals = tree
            .terminals()
            .filter_map(|t| tree[t].terminal().cloned())
            .collect::<Vec<_>>();

        let encoding = tree.encode_absolute().unwrap();
        let target = AbsoluteEncoding(vec![
            (Some(AbsoluteAncestor::new(1, "S")), None),
            (Some(AbsoluteAncestor::new(2, "NT")), None),
            (None, None),
        ]);
        assert_eq!(encoding, target);
        let reconstructed = Tree::decode(encoding, terminals);
        assert_eq!(reconstructed, tree);
    }

    #[test]
    fn encode() {
        let ptb = "(S (NP (PRP My) (NN daughter)) (VP (VBD broke) (NP (NP (DET the) (JJ red) \
                   (NN toy)) (PP (IN with) (NP (DET a) (NN hammer))))) (. .))";
        let tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let terminals = tree
            .terminals()
            .map(|t| tree[t].terminal().unwrap().clone())
            .collect::<Vec<_>>();
        let mut enc = tree.encode_relative().unwrap();
        enc.0[2] = (
            Some(RelativeAncestor::Regular {
                label: "VP".to_string(),
                offset: -5,
            }),
            None,
        );

        let enc = AbsoluteEncoding::try_from_relative(enc).unwrap_or_fix();
        let dec_tree = Tree::decode(enc, terminals.clone());
        assert_eq!(ptb, PTBFormat::Simple.tree_to_string(&dec_tree).unwrap());
    }

    #[test]
    fn rel_encode() {
        let ptb = "(S (NP (PRP My) (NN daughter)) (VP (VBD broke) (NP (NP (DET the) (JJ red) \
                   (NN toy)) (PP (IN with) (NP (DET a) (NN hammer))))) (. .))";
        let tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let terminals = tree
            .terminals()
            .map(|t| tree[t].terminal().unwrap().clone())
            .collect::<Vec<_>>();
        let rel_enc = tree.encode_relative().unwrap();
        let target = RelativeEncoding(vec![
            (Some(RelativeAncestor::new(2, "NP")), None),
            (Some(RelativeAncestor::Root("S".to_string())), None),
            (Some(RelativeAncestor::new(1, "VP")), None),
            (Some(RelativeAncestor::new(2, "NP")), None),
            (Some(RelativeAncestor::new(0, "NP")), None),
            (Some(RelativeAncestor::new(-1, "NP")), None),
            (Some(RelativeAncestor::new(1, "PP")), None),
            (Some(RelativeAncestor::new(1, "NP")), None),
            (Some(RelativeAncestor::Root("S".to_string())), None),
            (None, None),
        ]);
        assert_eq!(target, rel_enc);
        let enc = match AbsoluteEncoding::try_from_relative(rel_enc) {
            ConversionResult::Success(enc) => enc,
            ConversionResult::Error(_) => panic!(),
        };
        assert_eq!(tree, Tree::decode(enc, terminals))
    }

    #[test]
    fn conversion() {
        let ptb = "(S (NP (PRP My) (NN daughter)) (VP (VBD broke) (NP (NP (DET the) (JJ red) \
                   (NN toy)) (PP (IN with) (NP (DET a) (NN hammer))))) (. .))";
        let tree = PTBFormat::Simple.string_to_tree(ptb).unwrap();
        let terminals = tree
            .terminals()
            .map(|t| tree[t].terminal().unwrap().clone())
            .collect::<Vec<_>>();
        assert_eq!(
            tree.encode_relative().unwrap(),
            tree.encode_absolute().unwrap().into()
        );
        let abs_encoding = tree.encode_absolute().unwrap();
        let conv_encoding: AbsoluteEncoding =
            match AbsoluteEncoding::try_from_relative(tree.encode_relative().unwrap()) {
                ConversionResult::Success(enc) => enc,
                ConversionResult::Error(_) => panic!(),
            };
        assert_eq!(abs_encoding, conv_encoding);

        assert_eq!(tree, Tree::decode(abs_encoding, terminals.clone()));
        assert_eq!(tree, Tree::decode(conv_encoding, terminals.clone()));
    }
}
