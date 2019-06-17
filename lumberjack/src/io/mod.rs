mod conllx;
pub use crate::io::conllx::{ToConllx, TryFromConllx};
mod encode;
pub use crate::io::encode::{AbsoluteEncoding, Decode, Encode, RelativeEncoding};
mod negra;
pub use crate::io::negra::{NegraReader, NegraWriter};
mod ptb;
pub use crate::io::ptb::{PTBFormat, PTBLineFormat, PTBReader, PTBWriter};

use crate::tree::Tree;

use failure::Error;

pub(crate) static NODE_ANNOTATION_FEATURE_KEY: &str = "node_annotation";

/// Trait to write a `Tree`.
pub trait WriteTree {
    fn write_tree(&mut self, tree: &Tree) -> Result<(), Error>;
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    use conllx::graph::Sentence;

    use crate::io::{
        Decode, Encode, NegraWriter, PTBFormat, PTBLineFormat, TryFromConllx, WriteTree,
    };
    use crate::{NegraReader, PTBReader, Projectivize, Tree, UnaryChains};

    #[test]
    pub fn roundtrip() {
        let input = File::open("testdata/test.ptb").unwrap();
        let reader = BufReader::new(input).lines();
        for line in reader {
            let line = line.unwrap();
            let tree = PTBFormat::TueBa.string_to_tree(&line).unwrap();
            assert_eq!(line, PTBFormat::TueBa.tree_to_string(&tree).unwrap());
        }
    }

    #[test]
    pub fn negra_to_ptb() {
        let ptb_input = File::open("testdata/test.ptb").unwrap();
        let mut ptb_reader = BufReader::new(ptb_input).lines();
        let negra_input = File::open("testdata/test.negra").unwrap();
        let negra_reader = NegraReader::new(BufReader::new(negra_input));
        for tree in negra_reader {
            let ptb_tree = ptb_reader.next().unwrap().unwrap();
            let mut negra_tree = tree.unwrap();
            negra_tree.projectivize();
            let negra_to_ptb = PTBFormat::TueBa.tree_to_string(&negra_tree).unwrap();
            assert_eq!(ptb_tree, negra_to_ptb);
        }
    }

    #[test]
    pub fn ptb_to_negra() {
        let ptb_input = File::open("testdata/test.ptb").unwrap();
        let ptb_reader = PTBReader::new(
            BufReader::new(ptb_input),
            PTBFormat::TueBa,
            PTBLineFormat::SingleLine,
        );
        for (idx, tree) in ptb_reader.enumerate() {
            let mut tree = tree.unwrap();
            let root = tree.root();
            tree[root]
                .features_mut()
                .insert("sentence_id", Some(idx.to_string()));
            let mut buffer = Vec::new();
            let mut negra_writer = NegraWriter::new(&mut buffer);
            negra_writer.write_tree(&tree).unwrap();
            let mut negra_iter = NegraReader::new(BufReader::new(buffer.as_slice()));
            let new_tree = negra_iter.next().unwrap().unwrap();
            assert_eq!(tree, new_tree)
        }
    }

    #[test]
    fn negra_roundtrip() {
        let negra_input = File::open("testdata/test.negra").unwrap();
        let negra_reader = NegraReader::new(BufReader::new(negra_input));
        for tree in negra_reader {
            let tree = tree.unwrap();
            let mut buffer = Vec::new();
            let mut negra_writer = NegraWriter::new(&mut buffer);
            negra_writer.write_tree(&tree).unwrap();
            let mut negra_iter = NegraReader::new(BufReader::new(buffer.as_slice()));
            assert_eq!(tree, negra_iter.next().unwrap().unwrap())
        }
    }

    #[test]
    pub fn encoding_roundtrip() {
        let input = File::open("testdata/test.ptb").unwrap();
        let reader = BufReader::new(input).lines();
        for line in reader {
            let line = line.unwrap();
            // The trees are intentionally read as simple format because unary chains and features
            // are not preserved when collapsing chains and decoding.
            let mut tree = PTBFormat::Simple.string_to_tree(&line).unwrap();
            let terminals = tree
                .terminals()
                .collect::<Vec<_>>()
                .into_iter()
                .map(|node| tree[node].terminal().unwrap().clone())
                .collect();
            tree.collapse_unary_chains("_").unwrap();
            let encoding = tree.encode_absolute().unwrap();
            let mut decoded_tree = Tree::decode(encoding, terminals);
            decoded_tree.restore_unary_chains("_").unwrap();
            tree.restore_unary_chains("_").unwrap();

            assert_eq!(
                PTBFormat::Simple.tree_to_string(&tree).unwrap(),
                PTBFormat::Simple.tree_to_string(&decoded_tree).unwrap()
            );
            assert_eq!(tree, decoded_tree);
        }
    }

    #[test]
    pub fn conllx_roundtrip() {
        let input = File::open("testdata/test.ptb").unwrap();
        let reader = BufReader::new(input).lines();
        for line in reader {
            let line = line.unwrap();
            let mut tree = PTBFormat::TueBa.string_to_tree(&line).unwrap();
            tree.collapse_unary_chains("_").unwrap();

            tree.annotate_relative().unwrap();
            let relative_conllx = Sentence::from(&tree);
            let mut rel_decoded_tree =
                Tree::try_from_conllx_with_relative_encoding(&relative_conllx).unwrap();
            rel_decoded_tree.restore_unary_chains("_").unwrap();

            tree.annotate_absolute().unwrap();
            let absolute_conllx = Sentence::from(&tree);
            let mut abs_decoded_tree =
                Tree::try_from_conllx_with_absolute_encoding(&absolute_conllx).unwrap();
            abs_decoded_tree.restore_unary_chains("_").unwrap();

            tree.restore_unary_chains("_").unwrap();
            assert_eq!(
                PTBFormat::Simple.tree_to_string(&tree).unwrap(),
                PTBFormat::Simple.tree_to_string(&rel_decoded_tree).unwrap()
            );
            assert_eq!(
                PTBFormat::Simple.tree_to_string(&tree).unwrap(),
                PTBFormat::Simple.tree_to_string(&abs_decoded_tree).unwrap()
            );
            assert_eq!(abs_decoded_tree, rel_decoded_tree)
        }
    }
}
