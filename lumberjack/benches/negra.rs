#![feature(test)]
extern crate test;

use test::{black_box, Bencher};

use lumberjack::io::{NegraReader, PTBFormat, PTBLineFormat};
use lumberjack::PTBReader;
use std::fs::File;
use std::io::BufReader;

#[bench]
pub fn bench(b: &mut Bencher) {
    b.iter(|| {
        black_box(
            NegraReader::new(BufReader::new(File::open("testdata/test.negra").unwrap()))
                .into_iter()
                .map(|t| t.unwrap())
                .collect::<Vec<_>>(),
        );
    });
}

#[bench]
pub fn bench_ptb(b: &mut Bencher) {
    b.iter(|| {
        black_box(
            PTBReader::new(
                BufReader::new(File::open("testdata/test.ptb").unwrap()),
                PTBFormat::Simple,
                PTBLineFormat::SingleLine,
            )
            .into_iter()
            .map(|t| t.unwrap())
            .collect::<Vec<_>>(),
        );
    });
}
