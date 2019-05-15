#[macro_use]
extern crate failure;

use std::convert::TryFrom;
use std::io::{BufReader, Read, Write};

use clap::{App, AppSettings, Arg};
use conllx::io::Writer;
use failure::Error;
use stdinout::{Input, OrExit, Output};

use lumberjack::io::{PTBFormat, PTBLineFormat, PTBWriter, WriteTree};
use lumberjack::tree_modification::{Projectivize, TreeOps};
use lumberjack::util::LabelSet;
use lumberjack::{NegraReader, PTBReader, Tree};

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
    let writer = output.write().or_exit("Can't open output writer.", 1);

    let in_format = matches.value_of(IN_FORMAT).unwrap();
    let in_format = InFormat::try_from(in_format).or_exit("Can't read input format.", 1);
    let multiline = matches.is_present(MULTILINE);

    let out_format = matches.value_of(OUT_FORMAT).unwrap();
    let out_formatter = OutFormat::try_from(out_format).or_exit("Can't read output format.", 1);
    let parent = matches.is_present(PARENT) && out_formatter == OutFormat::Conllx;
    let projectivize = matches.is_present(PROJECTIVIZE);

    let filter_set = matches.value_of(FILTER_SET).map(get_set_from_file);
    let insertion_set = matches.value_of(INSERTION_SET).map(get_set_from_file);
    // has default value, safe to unwrap.
    let insertion_label = matches.value_of(INSERTION_LABEL).unwrap();

    let mut writer = get_writer(out_formatter, writer);

    for tree in get_reader(in_format, reader, multiline) {
        let mut tree = tree.or_exit("Could not read tree.", 1);
        if let Some(filter_set) = filter_set.as_ref() {
            tree.filter_nonterminals(filter_set).unwrap();
        }

        if let Some(insertion_set) = insertion_set.as_ref() {
            tree.insert_intermediate(insertion_set, insertion_label)
                .or_exit("Can't insert nodes.", 1);
        }

        if parent {
            tree.annotate_parent_tag()
                .or_exit("Can't annotate parent tags.", 1);
        }

        if projectivize {
            tree.projectivize();
        }

        writer
            .write_tree(&tree)
            .or_exit("Can't write to output.", 1)
    }
}

fn get_set_from_file(path: &str) -> LabelSet {
    LabelSet::Positive(
        std::fs::read_to_string(path)
            .unwrap()
            .split_whitespace()
            .map(ToOwned::to_owned)
            .collect(),
    )
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum InFormat {
    NEGRA,
    PTB,
    Simple,
    TueBa,
}

impl<'a> TryFrom<&'a str> for InFormat {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "negra" => Ok(InFormat::NEGRA),
            "ptb" => Ok(InFormat::PTB),
            "simple" => Ok(InFormat::Simple),
            "tueba" => Ok(InFormat::TueBa),
            _ => Err(format_err!("Unknown input format: {}", value)),
        }
    }
}
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum OutFormat {
    Conllx,
    PTB,
    Simple,
    TueBa,
}

impl<'a> TryFrom<&'a str> for OutFormat {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "conllx" => Ok(OutFormat::Conllx),
            "ptb" => Ok(OutFormat::PTB),
            "simple" => Ok(OutFormat::Simple),
            "tueba" => Ok(OutFormat::TueBa),
            _ => Err(format_err!("Unknown output format: {}", value)),
        }
    }
}

pub fn get_reader<'a, R>(
    in_format: InFormat,
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
        InFormat::NEGRA => Box::new(NegraReader::new(input)),
        InFormat::Simple => Box::new(PTBReader::new(input, PTBFormat::Simple, multiline)),
        InFormat::PTB => Box::new(PTBReader::new(input, PTBFormat::PTB, multiline)),
        InFormat::TueBa => Box::new(PTBReader::new(input, PTBFormat::TueBa, multiline)),
    }
}

pub fn get_writer<'a, W>(out_format: OutFormat, writer: W) -> Box<WriteTree + 'a>
where
    W: Write + 'a,
{
    match out_format {
        OutFormat::Conllx => Box::new(Writer::new(writer)),
        OutFormat::PTB => Box::new(PTBWriter::new(writer, PTBFormat::PTB)),
        OutFormat::Simple => Box::new(PTBWriter::new(writer, PTBFormat::Simple)),
        OutFormat::TueBa => Box::new(PTBWriter::new(writer, PTBFormat::TueBa)),
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
static INSERTION_LABEL: &str = "INSERTION_LABEL";
static INSERTION_SET: &str = "INSERTION_SET";
static FILTER_SET: &str = "FILTER_SET";
static PARENT: &str = "PARENT";
static PROJECTIVIZE: &str = "PROJECTIVIZE";

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
        .arg(
            Arg::with_name(PARENT)
                .long("parent")
                .help("Annotate parent tags of terminals as feature."),
        )
        .arg(
            Arg::with_name(INSERTION_SET)
                .long("insertion_set")
                .takes_value(true)
                .help("Path to file with insertion set."),
        )
        .arg(
            Arg::with_name(INSERTION_LABEL)
                .long("insertion_set")
                .default_value("UNK")
                .takes_value(true)
                .help("Label to insert."),
        )
        .arg(
            Arg::with_name(FILTER_SET)
                .long("filter")
                .takes_value(true)
                .help("Path to file with insertion set."),
        )
        .arg(Arg::with_name(PROJECTIVIZE).long("projectivize").help(
            "Projectivize trees before writing. Required for conversions from NEGRA \
             to PTB and CONLLX with encoding.",
        ))
}
