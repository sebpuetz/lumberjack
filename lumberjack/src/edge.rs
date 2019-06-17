use std::fmt;
use std::mem;

/// Enum representing Edges in Constituency Trees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Edge {
    Primary(Option<String>),
    Secondary(Option<String>),
}

// implementing display comes in handy for debugging using Dot Graphs
impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.label() {
            Some(label) => write!(f, "{}", label),
            None => write!(f, "--"),
        }
    }
}

impl Edge {
    /// Create a new primary Edge.
    ///
    /// Creates a new primary Edge with the given label.
    pub fn new_primary<S>(label: Option<S>) -> Self
    where
        S: Into<String>,
    {
        Edge::Primary(label.map(Into::into))
    }

    /// Create a new secondary Edge.
    ///
    /// Creates a new secondary Edge with the given label.    
    pub fn new_secondary<S>(label: Option<S>) -> Self
    where
        S: Into<String>,
    {
        Edge::Secondary(label.map(Into::into))
    }

    /// Return whether the Edge is primary.
    pub fn is_primary(&self) -> bool {
        if let Edge::Primary(_) = self {
            true
        } else {
            false
        }
    }

    /// Get the Edge label.
    pub fn label(&self) -> Option<&str> {
        match self {
            Edge::Primary(e) => e.as_ref().map(String::as_str),
            Edge::Secondary(e) => e.as_ref().map(String::as_str),
        }
    }

    /// Set the Edge label.
    pub fn set_label<S>(&mut self, new_label: Option<S>) -> Option<String>
    where
        S: Into<String>,
    {
        match self {
            Edge::Primary(e) => mem::replace(e, new_label.map(Into::into)),
            Edge::Secondary(e) => mem::replace(e, new_label.map(Into::into)),
        }
    }
}
