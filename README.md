[![Build Status](https://travis-ci.org/sebpuetz/lumberjack.svg?branch=master)](https://travis-ci.org/sebpuetz/lumberjack)

# lumberjack
Read and process constituency trees in various formats.

## Install:
* From GitHub:
```bash
git clone https://github.com/sebpuetz/lumberjack
cd lumberjack
cargo install --path . 
```

## Usage as standalone:

* Convert treebank in NEGRA export 4 format to bracketed TueBa V2 format
```bash
lumberjack-conversion --input_file treebank.negra --input_format negra \
    --output_format tueba --output_format treebank.tueba
``` 
* Retain only root node, NPs and VPs and print to simple bracketed format:
```bash
echo "NP PP" > filter_set.txt
lumberjack-conversion --input_file treebank.simple --input_format simple \
    --output_format tueba --output_format treebank.simple \
    --filter filter_set.txt
```
* Convert from treebank in simple bracketed to CONLLX format and annotate
parent tags of terminals as features.
```bash
lumberjack-conversion --input_file treebank.simple --input_format  simple\
    --output_format conllx --output_file treebank.conll --parent 
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
use lumberjack::{io::PTBFormat, TreeOps, util::LabelSet};

fn filter_nodes(iter: impl Iterator<Item=Tree>, set: LabelSet) {
    for mut tree in iter {
        tree.filter_nonterminals(&set).unwrap();
        println!("{}", PTBFormat::Simple.tree_to_string(&tree).unwrap());
    }
}
```
* convert treebank in simple bracketed format to CONLLX with constituency structure
encoded in the features field
```rust
use conllx::graph::Sentence;
use lumberjack::io::Encode;
use lumberjack::TreeOps;

fn to_conllx(iter: impl Iterator<Item=Tree>) {
    for mut tree in iter {
        tree.collaps_unary_chains().unwrap();
        tree.annotate_absolute().unwrap();
        println!("{}", Sentence::from(&tree));    
    }
}
```