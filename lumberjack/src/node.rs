use std::fmt;
use std::mem;

use failure::Error;

use crate::Span;

/// Enum representing Nodes in a constituency tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Node {
    /// Nonterminal Node.
    NonTerminal(NonTerminal),
    /// Terminal Node.
    Terminal(Terminal),
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

    /// Get a `Node`'s span.
    pub fn span(&self) -> &Span {
        match self {
            Node::Terminal(ref terminal) => &terminal.span,
            Node::NonTerminal(ref inner) => &inner.span,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NonTerminal {
    label: String,
    annotation: Option<String>,
    span: Span,
}

impl NonTerminal {
    pub(crate) fn new(label: impl Into<String>, span: impl Into<Span>) -> Self {
        NonTerminal {
            label: label.into(),
            annotation: None,
            span: span.into(),
        }
    }

    pub(crate) fn new_with_annotation(
        label: impl Into<String>,
        annotation: Option<impl Into<String>>,
        span: impl Into<Span>,
    ) -> Self {
        NonTerminal {
            label: label.into(),
            annotation: annotation.map(Into::into),
            span: span.into(),
        }
    }

    pub(crate) fn set_span(&mut self, span: impl Into<Span>) -> Span {
        mem::replace(&mut self.span, span.into())
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

    /// Return annotation if present.
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_ref().map(String::as_str)
    }

    /// Return old annotation and replace with `annotation`.
    pub fn set_annotation(&mut self, annotation: Option<impl Into<String>>) -> Option<String> {
        mem::replace(&mut self.annotation, annotation.map(Into::into))
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Terminal {
    form: String,
    pos: String,
    lemma: Option<String>,
    morph: Option<String>,
    span: Span,
}

impl Terminal {
    pub(crate) fn new(form: impl Into<String>, pos: impl Into<String>, idx: usize) -> Self {
        Terminal {
            form: form.into(),
            pos: pos.into(),
            lemma: None,
            morph: None,
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

    /// Return lemma if present, else `None`.
    pub fn lemma(&self) -> Option<&str> {
        self.lemma.as_ref().map(String::as_str)
    }

    /// Replace lemma with `new_lemma`. Return old value.
    pub fn set_lemma(&mut self, new_lemma: Option<impl Into<String>>) -> Option<String> {
        mem::replace(&mut self.lemma, new_lemma.map(Into::into))
    }

    /// Return morphological features if present, else `None`.
    pub fn morph(&self) -> Option<&str> {
        self.morph.as_ref().map(String::as_str)
    }

    /// Replace morphological features with `new_morph`. Return old value.
    pub fn set_morph(&mut self, new_morph: Option<impl Into<String>>) -> Option<String> {
        mem::replace(&mut self.morph, new_morph.map(Into::into))
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
            terminal.terminal_mut().unwrap().set_morph(Some("morph")),
            None
        );
        assert_eq!(terminal.terminal().unwrap().morph(), Some("morph"));
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
        assert_eq!(nonterminal.span(), &Span::new_continuous(3, 5));
        assert_eq!(
            nonterminal
                .nonterminal_mut()
                .unwrap()
                .set_annotation(Some("annotation")),
            None
        );
        assert_eq!(
            nonterminal.nonterminal_mut().unwrap().annotation(),
            Some("annotation")
        );
        assert_eq!(format!("{}", nonterminal), "other_label")
    }
}
