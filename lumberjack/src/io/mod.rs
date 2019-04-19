mod negra;
pub use negra::{NegraTreeIter, negra_to_tree};
mod ptb;
pub use ptb::{PTBFormat, PTBLineFormat, PTBTreeIter};

use failure::Error;
use crate::tree::Tree;

pub trait WriteTree {
    fn tree_to_string(&self, tree: &Tree) -> Result<String, Error>;
}

pub trait ReadTree {
    fn string_to_tree(&self, string: &str) -> Result<Tree, Error>;
}

#[derive(Clone, Copy)]
pub enum Format {
    NEGRA,
    PTB,
    Simple,
    TueBa,
}

impl WriteTree for Format {
    fn tree_to_string(&self, tree: &Tree) -> Result<String, Error> {
        match self {
            Format::NEGRA => Err(format_err!("NEGRA to String is currently not supported.")),
            Format::PTB => PTBFormat::PTB.tree_to_string(tree),
            Format::Simple => PTBFormat::Simple.tree_to_string(tree),
            Format::TueBa => PTBFormat::TueBa.tree_to_string(tree),
        }
    }
}

impl ReadTree for Format {
    fn string_to_tree(&self, tree: &str) -> Result<Tree, Error> {
        match self {
            Format::NEGRA => negra_to_tree(tree),
            Format::PTB => PTBFormat::PTB.string_to_tree(tree),
            Format::Simple => PTBFormat::Simple.string_to_tree(tree),
            Format::TueBa => PTBFormat::TueBa.string_to_tree(tree),
        }
    }
}