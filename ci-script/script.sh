cargo fmt --all -- --check
cargo build --verbose
cargo test --verbose
cargo clippy -- -D warnings
cargo run --bin lumberjack-conversion -- --input_file \
    lumberjack/testdata/test.negra --input_format negra --output_format tueba \
    --projectivize | diff lumberjack/testdata/test.ptb -
