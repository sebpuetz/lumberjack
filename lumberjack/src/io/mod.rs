mod negra;
pub use negra::{negra_to_tree, NegraTreeIter};
mod ptb;
pub use ptb::{PTBFormat, PTBLineFormat, PTBTreeIter};

use crate::tree::Tree;
use failure::Error;

pub(crate) static NODE_ANNOTATION_FEATURE_KEY: &str = "node_annotation";

pub trait WriteTree {
    fn tree_to_string(&self, tree: &Tree) -> Result<String, Error>;
}

pub trait ReadTree {
    fn string_to_tree(&self, string: &str) -> Result<Tree, Error>;
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Format {
    NEGRA,
    PTB,
    Simple,
    TueBa,
}

impl Format {
    pub fn try_from_str(s: &str) -> Result<Format, Error> {
        match s.to_lowercase().as_str() {
            "negra" => Ok(Format::NEGRA),
            "ptb" => Ok(Format::PTB),
            "simple" => Ok(Format::Simple),
            "tueba" => Ok(Format::TueBa),
            _ => Err(format_err!("Unknown format: {}", s)),
        }
    }
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
