#[macro_use]
extern crate failure;

use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufReader, Read, Write};

use clap::{App, AppSettings, Arg};
use conllx::graph::Node;
use conllx::io::{ReadSentence, Reader, Writer};
use failure::Error;
use stdinout::{Input, OrExit, Output};

use lumberjack::io::{
    Decode, Encode, NegraWriter, PTBFormat, PTBLineFormat, PTBWriter, TryFromConllx, WriteTree,
};
use lumberjack::util::LabelSet;
use lumberjack::{AnnotatePOS, NegraReader, PTBReader, Projectivize, Tree, TreeOps, UnaryChains};

fn main() {
    let app = build();
    let mut help = Vec::new();
    app.write_help(&mut help).unwrap();
    let matches = app.get_matches();
    let multiline = matches.is_present(MULTILINE);
    let in_format = matches.value_of(IN_FORMAT).unwrap();
    let in_format = InFormat::try_from(in_format).or_exit("Can't read input format.", 1);
    let in_path = matches.value_of(INPUT).map(ToOwned::to_owned);
    let input = Input::from(in_path);
    let reader = BufReader::new(input.buf_read().or_exit("Can't open input reader.", 1));

    let out_format = matches.value_of(OUT_FORMAT).unwrap();
    let out_formatter = OutFormat::try_from(out_format).or_exit("Can't read output format.", 1);
    let out_path = matches.value_of(OUTPUT).map(ToOwned::to_owned);
    let output = Output::from(out_path);
    let writer = output.write().or_exit("Can't open output writer.", 1);

    let parent_feature = matches.value_of(PARENT);
    let reattach_label = matches.value_of(REATTACH);

    let remove_dummies = matches.is_present(REMOVE_DUMMIES);
    let projectivize = matches.is_present(PROJECTIVIZE);
    let sec_edges = matches.is_present(SEC_EDGES);
    let filter_set = matches.value_of(FILTER_SET).map(get_set_from_file);
    let id_set = matches.value_of(ID_SET).map(get_set_from_file);
    let id_feature_name = matches.value_of(ID_FEATURE_NAME).unwrap();
    let insertion_set = matches.value_of(INSERTION_SET).map(get_set_from_file);
    let insertion_label = matches.value_of(INSERTION_LABEL).unwrap();

    let mut pos_sentences = matches.value_of(ANNOTATE_POS).map(|path| {
        let f = File::open(path).or_exit("Can't read POS file.", 1);
        Reader::new(BufReader::new(f))
    });

    let mut writer = get_writer(out_formatter, writer);

    for tree in get_reader(in_format, reader, multiline) {
        let mut tree = tree.or_exit("Could not read tree.", 1);

        if remove_dummies {
            tree.remove_dummy_nodes()
                .or_exit("Can't remove dummy nopdes.", 1);
        }

        if projectivize {
            if let Some(edges) = tree.projectivize() {
                if sec_edges {
                    for (parent, edge, child) in edges {
                        let label = if let Some(label) = edge.label() {
                            format!("np_{}", label)
                        } else {
                            "np".to_string()
                        };
                        tree.add_secondary_edge(parent, child, Some(label));
                    }
                }
            }
        }

        if let Some(pos_sentences) = pos_sentences.as_mut() {
            let sentence = pos_sentences
                .read_sentence()
                .or_exit(
                    "Number of POS sentences doesn't match number of input sentences.",
                    1,
                )
                .or_exit("Can't read POS sentence", 1);
            tree.annotate_pos(
                sentence
                    .iter()
                    .filter_map(Node::token)
                    .map(|token| token.pos().or_exit("Token missing POS", 1)),
            )
            .or_exit("Failed to annotate POS.", 1);
        }

        if let Some(label) = reattach_label {
            let root = tree.root();
            tree.reattach_terminals(root, |tree, nt| tree[nt].label().starts_with(label))
        }

        if let Some(id_set) = id_set.as_ref() {
            tree.project_ids(id_feature_name, |tree, nt| {
                tree.root() == nt || id_set.matches(tree[nt].label())
            })
        }

        if let Some(filter_set) = filter_set.as_ref() {
            tree.filter_nonterminals(|tree, nt| filter_set.matches(tree[nt].label()))
                .or_exit("Can't filter nodes.", 1);
        }

        if let Some(insertion_set) = insertion_set.as_ref() {
            tree.insert_intermediate(|tree, nt| {
                if !insertion_set.matches(tree[nt].label()) {
                    Some(insertion_label.into())
                } else {
                    None
                }
            })
            .or_exit("Can't insert nodes.", 1);
        }

        if let Some(name) = parent_feature.as_ref() {
            tree.annotate_parent_tag(name)
                .or_exit("Can't annotate parent tags.", 1);
        }

        if out_formatter == OutFormat::Absolute {
            tree.collapse_unary_chains("_")
                .or_exit("Could not collapse unary chains.", 1);
            tree.annotate_absolute()
                .or_exit("Could not encode tree.", 1)
        } else if out_formatter == OutFormat::Relative {
            tree.collapse_unary_chains("_")
                .or_exit("Could not collapse unary chains.", 1);
            tree.annotate_relative()
                .or_exit("Could not encode tree.", 1)
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
enum InFormat {
    Absolute,
    Relative,
    NEGRA,
    PTB,
    Simple,
    TueBa,
}

impl<'a> TryFrom<&'a str> for InFormat {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        use InFormat::*;
        match value.to_lowercase().as_str() {
            "absolute" => Ok(Absolute),
            "relative" => Ok(Relative),
            "negra" => Ok(NEGRA),
            "ptb" => Ok(PTB),
            "simple" => Ok(Simple),
            "tueba" => Ok(TueBa),
            _ => Err(format_err!("Unknown input format: {}", value)),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum OutFormat {
    Absolute,
    Conllx,
    Negra,
    PTB,
    Relative,
    Simple,
    TueBa,
}

impl<'a> TryFrom<&'a str> for OutFormat {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        use OutFormat::*;
        match value.to_lowercase().as_str() {
            "absolute" => Ok(Absolute),
            "conllx" => Ok(Conllx),
            "negra" => Ok(Negra),
            "ptb" => Ok(PTB),
            "relative" => Ok(Relative),
            "simple" => Ok(Simple),
            "tueba" => Ok(TueBa),
            _ => Err(format_err!("Unknown output format: {}", value)),
        }
    }
}

fn get_reader<'a, R>(
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

    use InFormat::*;
    match in_format {
        Absolute => Box::new(
            Reader::new(input)
                .sentences()
                .map(|s| s.or_exit("Can't read sentence", 1))
                .map(|s| TryFromConllx::try_from_conllx_with_absolute_encoding(&s)),
        ),
        Relative => Box::new(
            Reader::new(input)
                .sentences()
                .map(|s| s.or_exit("Can't read sentence", 1))
                .map(|s| TryFromConllx::try_from_conllx_with_relative_encoding(&s)),
        ),
        NEGRA => Box::new(NegraReader::new(input)),
        Simple => Box::new(PTBReader::new(input, PTBFormat::Simple, multiline)),
        PTB => Box::new(PTBReader::new(input, PTBFormat::PTB, multiline)),
        TueBa => Box::new(PTBReader::new(input, PTBFormat::TueBa, multiline)),
    }
}

fn get_writer<'a, W>(out_format: OutFormat, writer: W) -> Box<WriteTree + 'a>
where
    W: Write + 'a,
{
    use OutFormat::*;
    match out_format {
        Absolute | Conllx | Relative => Box::new(Writer::new(writer)),
        Negra => Box::new(NegraWriter::new(writer)),
        PTB => Box::new(PTBWriter::new(writer, PTBFormat::PTB)),
        Simple => Box::new(PTBWriter::new(writer, PTBFormat::Simple)),
        TueBa => Box::new(PTBWriter::new(writer, PTBFormat::TueBa)),
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
static ID_SET: &str = "ID_SET";
static ID_FEATURE_NAME: &str = "ID_FEATURE_NAME";
static FILTER_SET: &str = "FILTER_SET";
static PARENT: &str = "PARENT";
static PROJECTIVIZE: &str = "PROJECTIVIZE";
static SEC_EDGES: &str = "SEC_EDGES";
static REMOVE_DUMMIES: &str = "REMOVE_DUMMIES";
static REATTACH: &str = "REATTACH";
static ANNOTATE_POS: &str = "ANNOTATE_POS";

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
                .possible_values(&["absolute", "negra", "ptb", "relative", "simple", "tueba"])
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
                .possible_values(&[
                    "absolute", "conllx", "negra", "ptb", "relative", "simple", "tueba",
                ])
                .default_value("simple")
                .help("Output format:"),
        )
        .arg(
            Arg::with_name(PARENT)
                .long("parent")
                .takes_value(true)
                .value_name("FEATURE_NAME")
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
                .long("insertion_label")
                .default_value("UNK")
                .value_name("LABEL")
                .takes_value(true)
                .help("Label to insert."),
        )
        .arg(
            Arg::with_name(ID_SET)
                .long("id_set")
                .value_name("ID_SET_FILE")
                .takes_value(true)
                .help(
                    "Path to file with ID set. Annotates unique IDs for the nodes with tags \
                     given in the file onto terminals.",
                ),
        )
        .arg(
            Arg::with_name(ID_FEATURE_NAME)
                .long("id_feature")
                .value_name("FEATURE_NAME")
                .default_value("id")
                .takes_value(true)
                .help("Name of the feature that the IDs should be annotated under."),
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
        .arg(
            Arg::with_name(SEC_EDGES)
                .long("sec_edges")
                .help(
                    "Annotate projectivized edges as \
                     secondary edges. This option only has effect for negra as output format.",
                )
                .requires(PROJECTIVIZE),
        )
        .arg(
            Arg::with_name(REATTACH)
                .long("reattach")
                .takes_value(true)
                .value_name("PREFIX")
                .help("Reattach terminals with the given prefix to the root ndoe."),
        )
        .arg(
            Arg::with_name(REMOVE_DUMMIES)
                .long("remove_dummies")
                .help("Remove nodes with DUMMY label as introduced by incorrect tag sequences."),
        )
        .arg(
            Arg::with_name(ANNOTATE_POS)
                .long("pos")
                .value_name("FILE")
                .help("CONLLX format file with Part-of-speech tags.")
                .takes_value(true),
        )
}
