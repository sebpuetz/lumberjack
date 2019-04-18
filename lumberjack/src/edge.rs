use std::fmt;
use std::mem;

/// Struct representing an edge in a constituency Tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge(Option<String>);

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
    pub fn label(&self) -> Option<&str> {
        self.0.as_ref().map(String::as_ref)
    }

    pub fn set_label<S>(&mut self, new_label: Option<S>) -> Option<String>
    where
        S: Into<String>,
    {
        let new_label = new_label.map(Into::into);
        mem::replace(&mut self.0, new_label)
    }
}

impl<S> From<Option<S>> for Edge
where
    S: Into<String>,
{
    fn from(label: Option<S>) -> Edge {
        Edge(label.map(Into::into))
    }
}

impl Default for Edge {
    fn default() -> Edge {
        Edge(None)
    }
}
