use std::io::{BufReader, Read};

use clap::{App, AppSettings, Arg};
use failure::Error;
use stdinout::{Input, OrExit, Output};

use lumberjack::io::{Format, PTBFormat, PTBLineFormat, WriteTree};
use lumberjack::{NegraTreeIter, PTBTreeIter, Tree};

fn main() {
    let app = build();
    let mut help = Vec::new();
    app.write_help(&mut help).unwrap();
    let matches = app.get_matches();
    let in_path = matches.value_of(INPUT).map(ToOwned::to_owned);
    let input = Input::from(in_path);
    let reader = BufReader::new(input.buf_read().or_exit("Can't open input reader.", 1));
    let out_path = matches.value_of(OUTPUT).map(ToOwned::to_owned);
    let output = Output::from(out_path);
    let mut writer = output.write().or_exit("Can't open output writer.", 1);

    let in_format = matches.value_of(IN_FORMAT).unwrap();
    let in_format = Format::try_from_str(in_format).or_exit("Can't read input format.", 1);
    let multiline = matches.is_present(MULTILINE);

    if Format::NEGRA == in_format && multiline {
        eprintln!("NEGRA input format does not include multiline option.");
        eprintln!("{}", String::from_utf8(help).unwrap());
        std::process::exit(1)
    }

    let out_format = matches.value_of(OUT_FORMAT).unwrap();
    let out_formatter = Format::try_from_str(out_format).or_exit("Can't read output format.", 1);

    for tree in get_reader(in_format, reader, multiline) {
        let mut tree = tree.or_exit("Could not read tree.", 1);
        tree.projectivize();
        let tree_string = out_formatter
            .tree_to_string(&tree)
            .or_exit("Can't linearize tree.", 1);
        writeln!(writer, "{}", tree_string).or_exit("Can't write to output.", 1);
    }
}

fn get_reader<'a, R>(
    in_format: Format,
    input: BufReader<R>,
    multiline: bool,
) -> Box<Iterator<Item = Result<Tree, Error>> + 'a>
where
    R: Read + 'a,
{
    let multiline = if multiline {
        PTBLineFormat::MultiLine
    } else {
        PTBLineFormat::SingleLine
    };

    match in_format {
        Format::NEGRA => Box::new(NegraTreeIter::new(input).map(|tree| {
            tree.map(|mut tree| {
                tree.projectivize();
                tree
            })
        })),
        Format::Simple => Box::new(PTBTreeIter::new(input, PTBFormat::Simple, multiline)),
        Format::PTB => Box::new(PTBTreeIter::new(input, PTBFormat::PTB, multiline)),
        Format::TueBa => Box::new(PTBTreeIter::new(input, PTBFormat::TueBa, multiline)),
    }
}

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";
static IN_FORMAT: &str = "IN_FORMAT";
static OUT_FORMAT: &str = "OUT_FORMAT";
static MULTILINE: &str = "MULTILINE";

fn build<'a, 'b>() -> App<'a, 'b> {
    App::new("lumberjack-convert")
        .settings(DEFAULT_CLAP_SETTINGS)
        .version("0.1")
        .arg(
            Arg::with_name(INPUT)
                .long("input_file")
                .takes_value(true)
                .help("Input file"),
        )
        .arg(
            Arg::with_name(IN_FORMAT)
                .long("input_format")
                .takes_value(true)
                .possible_values(&["negra", "ptb", "simple", "tueba"])
                .default_value("tueba")
                .help("Input format:"),
        )
        .arg(
            Arg::with_name(MULTILINE)
                .long("multiline_brackets")
                .help("Specify whether each bracketed tree is on its own line."),
        )
        .arg(
            Arg::with_name(OUTPUT)
                .long("output_file")
                .takes_value(true)
                .help("Output file"),
        )
        .arg(
            Arg::with_name(OUT_FORMAT)
                .long("output_format")
                .takes_value(true)
                .possible_values(&["ptb", "simple", "tueba"])
                .default_value("simple")
                .help("Output format:"),
        )
}
