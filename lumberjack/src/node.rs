use std::fmt;
use std::mem;

use failure::Error;

use crate::{Continuity, Features, Span};

/// Enum representing Nodes in a constituency tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Node {
    /// Nonterminal Node.
    NonTerminal(NonTerminal),
    /// Terminal Node.
    Terminal(Terminal),
}

impl From<NonTerminal> for Node {
    fn from(nt: NonTerminal) -> Self {
        Node::NonTerminal(nt)
    }
}

impl From<Terminal> for Node {
    fn from(t: Terminal) -> Self {
        Node::Terminal(t)
    }
}

impl Node {
    /// Returns whether a `self` is `Terminal`.
    pub fn is_terminal(&self) -> bool {
        match &self {
            Node::Terminal(_) => true,
            Node::NonTerminal { .. } => false,
        }
    }

    /// Get a `Option<&Terminal>`.
    ///
    /// Returns `None` if `self` is a `Node::NonTerminal`.
    pub fn terminal(&self) -> Option<&Terminal> {
        match &self {
            Node::Terminal(ref terminal) => Some(terminal),
            Node::NonTerminal { .. } => None,
        }
    }

    /// Get a `Option<&NonTerminal>`.
    ///
    /// Returns `None` if `self` is a `Node::Terminal`.
    pub fn nonterminal(&self) -> Option<&NonTerminal> {
        match self {
            Node::Terminal(_) => None,
            Node::NonTerminal(ref inner) => Some(inner),
        }
    }

    /// Get a `Option<&mut NonTerminal>`.
    ///
    /// Returns `None` if `self` is a `Node::Terminal`.
    pub fn nonterminal_mut(&mut self) -> Option<&mut NonTerminal> {
        match self {
            Node::Terminal(_) => None,
            Node::NonTerminal(ref mut inner) => Some(inner),
        }
    }

    /// Returns whether this node's span is continuous.
    ///
    /// This does **not** return whether this node introduces a edge cutting another node's span.
    /// It does return whether this node's span is continuous.
    pub fn continuity(&self) -> Continuity {
        if self.span().skips().is_some() {
            Continuity::Discontinuous
        } else {
            Continuity::Continuous
        }
    }

    /// Get a `Option<&mut Terminal>`.
    ///
    /// Returns `None` if `self` is a `Node::NonTerminal`.
    pub fn terminal_mut(&mut self) -> Option<&mut Terminal> {
        match self {
            Node::NonTerminal(_) => None,
            Node::Terminal(ref mut terminal) => Some(terminal),
        }
    }

    /// Get the node's label.
    ///
    /// Returns the part-of-speech for `Terminal`s and the node label for `NonTerminal`s.
    pub fn label(&self) -> &str {
        match self {
            Node::NonTerminal(nt) => nt.label(),
            Node::Terminal(t) => t.label(),
        }
    }

    /// Set the node's label.
    ///
    /// Returns the replaced label.
    pub fn set_label(&mut self, s: impl Into<String>) -> String {
        match self {
            Node::NonTerminal(nt) => nt.set_label(s),
            Node::Terminal(t) => t.set_label(s),
        }
    }

    /// Get this `Node`'s `Features`.
    pub fn features(&self) -> Option<&Features> {
        match self {
            Node::NonTerminal(nt) => nt.features(),
            Node::Terminal(t) => t.features(),
        }
    }

    /// Get this `Node`'s `Features` mutably.
    ///
    /// This method initializes the features if they are `None`.
    pub fn features_mut(&mut self) -> &mut Features {
        match self {
            Node::NonTerminal(nt) => nt.features_mut(),
            Node::Terminal(t) => t.features_mut(),
        }
    }

    /// Set this `NonTerminal`'s `Features`.
    ///
    /// Returns the replaced value.
    pub fn set_features(&mut self, features: Option<Features>) -> Option<Features> {
        match self {
            Node::NonTerminal(nt) => nt.set_features(features),
            Node::Terminal(t) => t.set_features(features),
        }
    }

    /// Get a `Node`'s span.
    pub fn span(&self) -> &Span {
        match self {
            Node::Terminal(ref terminal) => &terminal.span,
            Node::NonTerminal(ref inner) => &inner.span,
        }
    }

    /// Extend the upper bounds of a span.
    pub(crate) fn set_span(&mut self, span: impl Into<Span>) -> Result<Span, Error> {
        let span = span.into();
        match self {
            Node::Terminal(t) => {
                if span.skips().is_some() {
                    return Err(format_err!("Can't assign discontinuous span to terminal."));
                } else if span.n_indices() != 1 {
                    return Err(format_err!("Terminals cover only single indices."));
                }
                Ok(mem::replace(&mut t.span, span))
            }
            Node::NonTerminal(nt) => Ok(mem::replace(&mut nt.span, span)),
        }
    }

    /// Extend the upper bounds of a span.
    pub(crate) fn extend_span(&mut self) -> Result<(), Error> {
        match self {
            Node::Terminal(_) => Err(format_err!("Can't extend terminal's span.")),
            Node::NonTerminal(nt) => {
                nt.span.extend();
                Ok(())
            }
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Node::Terminal(terminal) => write!(f, "{} {}", terminal.pos, terminal.form),
            Node::NonTerminal(nt) => write!(f, "{}", nt.label),
        }
    }
}

/// Struct representing a non terminal tree node.
///
/// `NonTerminal`s are defined by their `label`, optional `annotation` and their covered `span`.
#[derive(Clone, Debug, Eq)]
pub struct NonTerminal {
    label: String,
    features: Option<Features>,
    span: Span,
}

impl PartialEq<NonTerminal> for NonTerminal {
    fn eq(&self, other: &NonTerminal) -> bool {
        if self.label != other.label() || self.span != other.span {
            return false;
        }
        match (&self.features, &other.features) {
            (Some(f), Some(f_other)) => f == f_other,
            (None, None) => true,
            (Some(f), None) => f.inner().is_empty(),
            (None, Some(f)) => f.inner().is_empty(),
        }
    }
}

impl NonTerminal {
    pub(crate) fn new(label: impl Into<String>, span: impl Into<Span>) -> Self {
        NonTerminal {
            label: label.into(),
            features: None,
            span: span.into(),
        }
    }

    /// Returns whether this NonTerminal's Span is continuous.
    ///
    /// This does **not** return whether this nonterminal introduces a edge cutting another node's
    /// span. It does return whether this node's span is continuous.
    pub fn continuity(&self) -> Continuity {
        if self.span().skips().is_some() {
            Continuity::Discontinuous
        } else {
            Continuity::Continuous
        }
    }

    pub(crate) fn set_span(&mut self, span: impl Into<Span>) -> Span {
        mem::replace(&mut self.span, span.into())
    }

    /// Merge the coverage of the NonTerminal with a span.
    ///
    /// After merging, the NonTerminal will also cover those indices in `span`.
    ///
    /// Returns the NonTerminal's continuity after merging the Spans.
    #[allow(dead_code)]
    pub(crate) fn merge_spans(&mut self, span: &Span) -> Continuity {
        let span = self.span.merge_spans(span);
        self.span = span;
        if self.span.skips().is_some() {
            Continuity::Discontinuous
        } else {
            Continuity::Continuous
        }
    }

    /// Remove indices from the NonTerminal's span.
    ///
    /// Returns the NonTerminal's continuity after removing the indices.
    #[allow(dead_code)]
    pub(crate) fn remove_indices(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
    ) -> Continuity {
        self.span.remove_indices(indices)
    }

    /// Get this `NonTerminal`'s `Features`.
    pub fn features(&self) -> Option<&Features> {
        self.features.as_ref()
    }

    /// Get this `NonTerminal`'s `Features` mutably.
    ///
    /// This method initializes the features if they are `None`.
    pub fn features_mut(&mut self) -> &mut Features {
        if let Some(ref mut features) = self.features {
            features
        } else {
            self.features = Some(Features::default());
            self.features.as_mut().unwrap()
        }
    }

    /// Set this `NonTerminal`'s `Features`.
    ///
    /// Returns the replaced value.
    pub fn set_features(&mut self, features: Option<Features>) -> Option<Features> {
        mem::replace(&mut self.features, features)
    }

    /// Get the `NonTerminal`'s span.
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Return the label of the `NonTerminal`.
    pub fn label(&self) -> &str {
        self.label.as_str()
    }

    /// Return old label and replace with `label`.
    pub fn set_label(&mut self, label: impl Into<String>) -> String {
        mem::replace(&mut self.label, label.into())
    }
}

impl fmt::Display for NonTerminal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// Struct representing a Terminal.
///
/// `Terminal`s are represented by:
/// * `form` - word form
/// * `pos` - part of speech tag
/// * `lemma` - (optional) lemma
/// * `morph` - (optional) morphological features
/// * `span` - position in the sentence
#[derive(Clone, Debug, Eq)]
pub struct Terminal {
    form: String,
    pos: String,
    lemma: Option<String>,
    features: Option<Features>,
    span: Span,
}

impl PartialEq<Terminal> for Terminal {
    fn eq(&self, other: &Terminal) -> bool {
        if self.pos != other.pos || self.span != other.span || self.lemma != other.lemma {
            return false;
        }
        match (&self.features, &other.features) {
            (Some(f), Some(f_other)) => f == f_other,
            (None, None) => true,
            (Some(f), None) => f.inner().is_empty(),
            (None, Some(f)) => f.inner().is_empty(),
        }
    }
}

impl Terminal {
    pub(crate) fn new(form: impl Into<String>, pos: impl Into<String>, idx: usize) -> Self {
        Terminal {
            form: form.into(),
            pos: pos.into(),
            lemma: None,
            features: None,
            span: idx.into(),
        }
    }

    /// Returns the `Terminal`'s span.
    ///
    /// A `Terminal`'s span is defined as a tuple `(n, n+1)` where `n` is the 0-based position of
    /// the `Terminal` in the sentence.
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Return the `Terminal`s form.
    pub fn form(&self) -> &str {
        self.form.as_str()
    }

    /// Replace form with `new_form`. Return old value.
    pub fn set_form(&mut self, new_form: impl Into<String>) -> String {
        mem::replace(&mut self.form, new_form.into())
    }

    /// Return part of speech.
    pub fn label(&self) -> &str {
        self.pos.as_str()
    }

    /// Replace part of speech with `new_pos`. Return old value.
    pub fn set_label(&mut self, new_pos: impl Into<String>) -> String {
        mem::replace(&mut self.pos, new_pos.into())
    }

    /// Get this `Terminal`'s `Features`.
    pub fn features(&self) -> Option<&Features> {
        self.features.as_ref()
    }

    /// Get this `Terminal`'s `Features` mutably.
    ///
    /// This method initializes the features if they are `None`.
    pub fn features_mut(&mut self) -> &mut Features {
        if let Some(ref mut features) = self.features {
            features
        } else {
            self.features = Some(Features::default());
            self.features.as_mut().unwrap()
        }
    }

    /// Set this `Terminal`'s `Features`.
    ///
    /// Returns the replaced value.
    pub fn set_features(&mut self, features: Option<Features>) -> Option<Features> {
        mem::replace(&mut self.features, features)
    }

    /// Return lemma if present, else `None`.
    pub fn lemma(&self) -> Option<&str> {
        self.lemma.as_ref().map(String::as_str)
    }

    /// Replace lemma with `new_lemma`. Return old value.
    pub fn set_lemma<S>(&mut self, new_lemma: Option<S>) -> Option<String>
    where
        S: Into<String>,
    {
        mem::replace(&mut self.lemma, new_lemma.map(Into::into))
    }
}

impl fmt::Display for Terminal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.pos, self.form)
    }
}

#[cfg(test)]
mod test {
    use crate::{Node, NonTerminal, Span, Terminal};

    #[test]
    fn node_terminal() {
        let mut terminal = Node::Terminal(Terminal::new("form", "pos", 0));
        assert!(terminal.is_terminal());
        assert!(terminal.terminal().is_some());
        assert!(terminal.nonterminal().is_none());
        assert!(terminal.extend_span().is_err());
        assert_eq!(terminal.set_label("other_pos"), "pos");
        assert_eq!(terminal.label(), "other_pos");
        assert_eq!(
            terminal
                .terminal_mut()
                .unwrap()
                .set_features(Some("morph".into())),
            None
        );
        assert_eq!(
            terminal.terminal().unwrap().features(),
            Some(&"morph".into())
        );
        assert_eq!(
            terminal.terminal_mut().unwrap().set_lemma(Some("lemma")),
            None
        );
        assert_eq!(terminal.terminal().unwrap().lemma(), Some("lemma"));
        assert_eq!(
            terminal.terminal_mut().unwrap().set_form("other_form"),
            "form"
        );
        assert_eq!(terminal.terminal().unwrap().form(), "other_form");
        assert_eq!(format!("{}", terminal), "other_pos other_form")
    }

    #[test]
    fn node_nonterminal() {
        let mut nonterminal = Node::NonTerminal(NonTerminal::new("label", 0));
        assert!(!nonterminal.is_terminal());
        assert_eq!(nonterminal.terminal(), None);
        assert!(nonterminal.nonterminal().is_some());
        assert_eq!(nonterminal.set_label("other_label"), "label");
        assert_eq!(nonterminal.label(), "other_label");
        assert_eq!(nonterminal.nonterminal_mut().unwrap().set_span(3), 0.into());
        assert_eq!(nonterminal.span(), &3.into());
        nonterminal.extend_span().unwrap();
        assert_eq!(nonterminal.span(), &Span::new(3, 5));
        assert_eq!(
            nonterminal
                .nonterminal_mut()
                .unwrap()
                .features_mut()
                .insert("some", Some("feature")),
            None
        );
        assert_eq!(
            nonterminal
                .nonterminal_mut()
                .unwrap()
                .features()
                .unwrap()
                .get_val("some"),
            Some("feature")
        );
        assert_eq!(format!("{}", nonterminal), "other_label")
    }
}
