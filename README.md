[![Crate](https://img.shields.io/crates/v/lumberjack.svg)](https://crates.io/crates/lumberjack)
[![Build Status](https://travis-ci.org/sebpuetz/lumberjack.svg?branch=master)](https://travis-ci.org/sebpuetz/lumberjack)

# lumberjack
Read and process constituency trees in various formats.

## Install:
* From crates.io:
```bash
cargo install lumberjack-utils
```
* From GitHub:
```bash
cargo install --git https://github.com/sebpuetz/lumberjack
```

## Usage as standalone:

* Convert treebank in NEGRA export 4 format to bracketed TueBa V2 format
```bash
lumberjack-conversion --input_file treebank.negra --input_format negra \
    --output_format tueba --output_file treebank.tueba --projectivize
``` 
* Retain only root node, `NP`s and `PP`s and print to simple bracketed format:
```bash
echo "NP PP" > filter_set.txt
lumberjack-conversion --input_file treebank.simple --input_format simple \
    --output_format tueba --output_file treebank.filtered \
    --filter filter_set.txt
```
* Convert from treebank in simple bracketed to CONLLX format and annotate
parent tags of terminals as features.
```bash
lumberjack-conversion --input_file treebank.simple --input_format  simple\
    --output_format conllx --output_file treebank.conll --parent 
```
* Modifications in the following order:

1. Reattach all terminals with part-of-speech starting with `$` to the
root node
2. Remove all nonterminals except the root, `S`s, `NP`s, `PP`s and `VP`s
3. Assign unique identifiers based on the closest `S` to terminals
4. Insert nodes with label `label` above terminals that aren't dominated by `NP` or `PP`
5. Annotate label of parent node on terminals.
6. Print to CONLLX format with annotations.

```bash
echo "S VP NP PP" > filter_set.txt
echo "NP PP" > insert_set.txt
echo "S" > id_set.txt
lumberjack-conversion --input_file treebank.simple --input_format simple\
    --output_format conllx --insertion_set insert_set.txt \
    --insertion_label label --id_set id_set.txt --reattach $\
    --parent parent --output_file treebank.conllx
```

## Usage as rust library:
* read and projectivize trees from NEGRA format and print to simple
 bracketed format
```rust
use std::io::{BufReader, File};

use lumberjack::io::{NegraReader, PTBFormat};
use lumberjack::Projectivize;

fn print_negra(path: &str) {
    let file = File::open(path).unwrap();
    let reader = NegraReader::new(BufReader::new(file));
    for tree in reader {
        let mut tree = tree.unwrap();
        tree.projectivize();
        println!("{}", PTBFormat::Simple.tree_to_string(&tree).unwrap());
    }
}
```
* filter non-terminal nodes from trees in a treebank and print to
 simple bracketed format:
```rust
use lumberjack::{io::PTBFormat, Tree, TreeOps, util::LabelSet};

fn filter_nodes(iter: impl Iterator<Item=Tree>, set: LabelSet) {
    for mut tree in iter {
        tree.filter_nonterminals(|tree, nt| set.matches(tree[nt].label())).unwrap();
        println!("{}", PTBFormat::Simple.tree_to_string(&tree).unwrap());
    }
}
```
* convert treebank in simple bracketed format to CONLLX with constituency structure
encoded in the features field
```rust
use conllx::graph::Sentence;
use lumberjack::io::Encode;
use lumberjack::{Tree, TreeOps, UnaryChains};

fn to_conllx(iter: impl Iterator<Item=Tree>) {
    for mut tree in iter {
        tree.collaps_unary_chains().unwrap();
        tree.annotate_absolute().unwrap();
        println!("{}", Sentence::from(&tree));    
    }
}
```
