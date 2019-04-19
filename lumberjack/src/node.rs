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
    pub(crate) fn new<S>(label: S, span: Span) -> Self
    where
        S: Into<String>,
    {
        NonTerminal {
            label: label.into(),
            annotation: None,
            span,
        }
    }

    pub(crate) fn set_span(&mut self, span: Span) {
        mem::replace(&mut self.span, span);
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
    pub fn set_label<S>(&mut self, label: S) -> String
    where
        S: Into<String>,
    {
        mem::replace(&mut self.label, label.into())
    }

    /// Return annotation if present.
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_ref().map(String::as_str)
    }

    /// Return old annotation and replace with `annotation`.
    pub fn set_annotation<S>(&mut self, annotation: Option<S>) -> Option<String>
    where
        S: Into<String>,
    {
        mem::replace(&mut self.annotation, annotation.map(Into::into))
    }
}

#[derive(Debug)]
pub(crate) struct NTBuilder {
    label: String,
    annotation: Option<String>,
    span: Option<Span>,
}

impl NTBuilder {
    pub(crate) fn new<S>(label: S) -> Self
    where
        S: Into<String>,
    {
        NTBuilder {
            label: label.into(),
            annotation: None,
            span: None,
        }
    }

    pub(crate) fn span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub(crate) fn annotation<S>(mut self, annotation: Option<S>) -> Self
    where
        S: Into<String>,
    {
        self.annotation = annotation.map(Into::into);
        self
    }

    pub(crate) fn try_into_nt(self) -> Result<NonTerminal, Error> {
        if let Some(span) = self.span {
            Ok(NonTerminal {
                label: self.label,
                span,
                annotation: self.annotation,
            })
        } else {
            Err(format_err!(
                "Could not convert into NonTerminal, missing span"
            ))
        }
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
    #[allow(dead_code)]
    pub(crate) fn new<S>(form: S, pos: S, span: Span) -> Self
    where
        S: Into<String>,
    {
        Terminal {
            form: form.into(),
            pos: pos.into(),
            lemma: None,
            morph: None,
            span,
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
    pub fn set_form<S>(&mut self, new_form: S) -> String
    where
        S: Into<String>,
    {
        mem::replace(&mut self.form, new_form.into())
    }

    /// Return part of speech.
    pub fn pos(&self) -> &str {
        self.pos.as_str()
    }

    /// Replace part of speech with `new_pos`. Return old value.
    pub fn set_pos<S>(&mut self, new_pos: S) -> String
    where
        S: Into<String>,
    {
        mem::replace(&mut self.pos, new_pos.into())
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

    /// Return morphological features if present, else `None`.
    pub fn morph(&self) -> Option<&str> {
        self.morph.as_ref().map(String::as_str)
    }

    /// Replace morphological features with `new_morph`. Return old value.
    pub fn set_morph<S>(&mut self, new_morph: Option<S>) -> Option<String>
    where
        S: Into<String>,
    {
        mem::replace(&mut self.morph, new_morph.map(Into::into))
    }
}

#[derive(Debug)]
pub(crate) struct TerminalBuilder {
    form: String,
    pos: String,
    span: Span,
    lemma: Option<String>,
    morph: Option<String>,
}

#[allow(dead_code)]
impl TerminalBuilder {
    pub fn new<S>(form: S, pos: S, span: Span) -> Self
    where
        S: Into<String>,
    {
        TerminalBuilder {
            form: form.into(),
            pos: pos.into(),
            span,
            lemma: None,
            morph: None,
        }
    }

    pub fn try_into_terminal(self) -> Result<Terminal, Error> {
        if self.span.lower() != self.span.upper() - 1 {
            return Err(format_err!(
                "Span of terminal has to be of length 1: ({},{})",
                self.span.lower(),
                self.span.upper()
            ));
        }
        Ok(Terminal {
            form: self.form,
            pos: self.pos,
            span: self.span,
            lemma: self.lemma,
            morph: self.morph,
        })
    }

    pub fn lemma<S>(mut self, lemma: S) -> TerminalBuilder
    where
        S: Into<String>,
    {
        self.lemma = Some(lemma.into());
        self
    }

    pub fn morph<S>(mut self, morph: S) -> TerminalBuilder
    where
        S: Into<String>,
    {
        self.morph = Some(morph.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use failure::Error;

    use crate::node::{NTBuilder, TerminalBuilder};
    use crate::{NonTerminal, Span, Terminal};

    #[test]
    fn terminal_builder() -> Result<(), Error> {
        let form = "test_form";
        let pos = "test_pos";
        let lemma = "lemma";
        let span = Span::new_continuous(0, 1);
        let builder = TerminalBuilder::new(form, pos, span.clone()).lemma(lemma);
        let terminal = builder.try_into_terminal()?;

        assert_eq!(form, terminal.form());
        assert_eq!(pos, terminal.pos());
        assert_eq!(Some(lemma), terminal.lemma());
        assert_eq!(None, terminal.morph());

        let terminal2 = Terminal::new(form, pos, span);
        assert_eq!(form, terminal2.form());
        assert_eq!(pos, terminal2.pos());
        Ok(())
    }

    #[test]
    fn nt_builder() -> Result<(), Error> {
        let label = "test_label";
        let span = Span::new_continuous(0, 2);
        let nt = NTBuilder::new(label).span(span.clone()).try_into_nt()?;
        assert_eq!(label, nt.label());
        assert_eq!(*nt.span(), span);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn nt_builder_fail() {
        let label = "test_label";
        let _nt: NonTerminal = NTBuilder::new(label).try_into_nt().unwrap();
    }

    #[test]
    #[should_panic]
    fn terminal_invalid_span_too_big() {
        let form = "test_form";
        let pos = "test_pos";
        let lemma = "lemma";
        let span = Span::new_continuous(0, 2);
        let builder = TerminalBuilder::new(form, pos, span).lemma(lemma);
        let _terminal = builder.try_into_terminal().unwrap();
    }
}
