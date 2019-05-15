mod conllx;
pub use crate::io::conllx::ToConllx;
mod negra;
pub use crate::io::negra::{negra_to_tree, NegraReader};
mod ptb;
pub use crate::io::ptb::{PTBFormat, PTBLineFormat, PTBReader, PTBWriter};

use crate::tree::Tree;

use failure::Error;

pub(crate) static NODE_ANNOTATION_FEATURE_KEY: &str = "node_annotation";

/// Trait to write a `Tree`.
pub trait WriteTree {
    fn write_tree(&mut self, tree: &Tree) -> Result<(), Error>;
}
