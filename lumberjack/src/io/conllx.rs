use std::convert::TryFrom;

use conllx::graph::{Node, Sentence};
use conllx::io::{WriteSentence, Writer};
use conllx::token::{Features, Token};
use failure::Error;

use crate::io::encode::{AbsoluteAncestor, ConversionResult, RelativeAncestor};
use crate::io::{AbsoluteEncoding, Decode, RelativeEncoding};
use crate::tree_modification::TreeOps;
use crate::{Terminal, Tree, WriteTree};
use std::io::Write;

impl<W> WriteTree for Writer<W>
where
    W: Write,
{
    fn write_tree(&mut self, tree: &Tree) -> Result<(), Error> {
        self.write_sentence(&tree.into())
    }
}

/// Conversion Trait to CONLLX.
///
/// Creates a `Token` for each `Terminal` in the tree.
/// Form, POS and Features are carried over.
pub trait ToConllx {
    /// Nonconsuming conversion to CONLLX.
    fn to_conllx(&self) -> Sentence;
}

/// Conversion Trait from CONLLX to `Tree`.
pub trait TryFromConllx: Sized {
    /// Attempt to construct a tree from labels annotated on a `Sentence`.
    ///
    /// Assumes that the encoding is found by key `"abs_ancestor"`.
    fn try_from_conllx_with_absolute_encoding(sentence: &Sentence) -> Result<Self, Error>;

    /// Attempt to construct a tree from labels annotated on a `Sentence`.
    ///
    /// Assumes that the encoding is found by key `"rel_ancestor"`.
    ///
    /// This method converts the relative encoding to an absolute scale one before constructing
    /// the tree. If that conversion fails, `ConversionResult::fix` is called to finish the
    /// conversion.
    fn try_from_conllx_with_relative_encoding(sentence: &Sentence) -> Result<Self, Error>;
}

impl TryFromConllx for Tree {
    fn try_from_conllx_with_absolute_encoding(sentence: &Sentence) -> Result<Self, Error> {
        let mut encoding = Vec::with_capacity(sentence.len() - 1);
        let mut terminals = Vec::with_capacity(sentence.len() - 1);
        for (idx, token) in sentence.iter().filter_map(Node::token).enumerate() {
            let ancestor = token
                .features()
                .and_then(|f| f.as_map().get("abs_ancestor"))
                .ok_or_else(|| format_err!("Missing ancestor feature."))?
                .as_ref()
                .map(String::as_str)
                .ok_or_else(|| format_err!("Ancestor feature missing value."))?;
            let terminal = Terminal::new(token.form(), token.pos().unwrap_or("_"), idx);
            terminals.push(terminal);
            let mut parts = ancestor.split('~');
            let ancestor = parts
                .next()
                .ok_or_else(|| format_err!("Invalid ancestor part."))?;
            let unary_chain = parts
                .next()
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned);

            let ancestor = match ancestor {
                "NONE" => None,
                s => {
                    let rel_ancestor = AbsoluteAncestor::try_from(s)?;
                    Some(rel_ancestor)
                }
            };
            encoding.push((ancestor, unary_chain))
        }
        let mut tree = Tree::decode(AbsoluteEncoding::new(encoding), terminals);
        tree.restore_unary_chains("_")?;
        Ok(tree)
    }

    fn try_from_conllx_with_relative_encoding(sentence: &Sentence) -> Result<Self, Error> {
        let mut encoding = Vec::with_capacity(sentence.len() - 1);
        let mut terminals = Vec::with_capacity(sentence.len() - 1);
        for (idx, token) in sentence.iter().filter_map(Node::token).enumerate() {
            let ancestor = token
                .features()
                .and_then(|f| f.as_map().get("rel_ancestor"))
                .ok_or_else(|| format_err!("Missing ancestor feature."))?
                .as_ref()
                .map(String::as_str)
                .ok_or_else(|| format_err!("Ancestor feature missing value."))?;
            let terminal = Terminal::new(token.form(), token.pos().unwrap_or("_"), idx);
            terminals.push(terminal);
            let mut parts = ancestor.split('~');
            let ancestor = parts
                .next()
                .ok_or_else(|| format_err!("Invalid ancestor part."))?;
            let unary_chain = parts
                .next()
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned);

            let ancestor = match ancestor {
                "NONE" => None,
                s => {
                    let rel_ancestor = RelativeAncestor::try_from(s)?;
                    Some(rel_ancestor)
                }
            };
            encoding.push((ancestor, unary_chain))
        }
        let encoding = match AbsoluteEncoding::try_from_relative(RelativeEncoding::new(encoding)) {
            ConversionResult::Success(encoding) => encoding,
            ConversionResult::Error(err) => err.fix(),
        };
        let mut tree = Tree::decode(encoding, terminals);
        tree.restore_unary_chains("_")?;
        Ok(tree)
    }
}

impl ToConllx for Tree {
    fn to_conllx(&self) -> Sentence {
        self.into()
    }
}

impl<'a> From<&'a Terminal> for Token {
    fn from(terminal: &Terminal) -> Self {
        let mut token = Token::new(terminal.form());
        token.set_lemma(terminal.lemma());
        token.set_pos(Some(terminal.label()));
        if let Some(features) = terminal.features() {
            token.set_features(Some(Features::from_string(features.to_string())));
        }
        token
    }
}

impl From<Tree> for Sentence {
    fn from(mut tree: Tree) -> Self {
        let mut tokens = Vec::with_capacity(tree.n_terminals());

        let terminals = tree.terminals().collect::<Vec<_>>();
        for terminal in terminals {
            let terminal = tree[terminal].terminal_mut().unwrap();
            let mut token = Token::new(terminal.set_form(String::new()));
            let lemma = terminal.set_lemma::<String>(None);
            token.set_lemma(lemma);
            token.set_pos(Some(terminal.set_label(String::new())));
            if let Some(morph) = terminal.set_features(None) {
                token.set_features(Some(Features::from_string(morph.to_string())));
            }
            tokens.push((token, terminal.span().start));
        }
        tokens.sort_by(|t0, t1| t0.1.cmp(&t1.1));
        let mut sentence = Sentence::new();
        for (token, _) in tokens {
            sentence.push(token);
        }
        sentence
    }
}
impl<'a> From<&'a Tree> for Sentence {
    fn from(tree: &'a Tree) -> Self {
        let mut tokens = tree
            .terminals()
            .filter_map(|t| tree[t].terminal().map(|t| (t.into(), t.span().start)))
            .collect::<Vec<_>>();

        tokens.sort_by(|t0, t1| t0.1.cmp(&t1.1));
        let mut sentence = Sentence::new();
        for (token, _) in tokens {
            sentence.push(token);
        }
        sentence
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use conllx::graph::Sentence;
    use conllx::token::{Features, Token, TokenBuilder};

    use crate::io::conllx::ToConllx;
    use crate::io::negra::negra_to_tree;
    use crate::io::ptb::PTBFormat;

    #[test]
    fn to_conllx() {
        let input = "(NX (NN Nounphrase) (PX (PP on) (NX (DET a) (ADJ single) (NX line))))";
        let mut tree = PTBFormat::TueBa.string_to_tree(input).unwrap();
        let nounphrase = tree.terminals().next().unwrap();
        tree[nounphrase]
            .features_mut()
            .insert::<_, String>("feature", None);
        tree[nounphrase].features_mut().insert("key", Some("val"));
        let conll_sentence = tree.to_conllx();
        let mut target = Sentence::new();
        target.push(
            TokenBuilder::new("Nounphrase")
                .pos("NN")
                .features(Features::from_string("feature|key:val"))
                .into(),
        );
        target.push(TokenBuilder::new("on").pos("PP").into());
        target.push(TokenBuilder::new("a").pos("DET").into());
        target.push(TokenBuilder::new("single").pos("ADJ").into());
        target.push(TokenBuilder::new("line").pos("NX").into());
        assert_eq!(conll_sentence, target);

        let input = fs::read_to_string("testdata/long_single.negra").unwrap();
        let tree = negra_to_tree(&input).unwrap();
        let conll_sentence = tree.to_conllx();
        assert_eq!(
            &Token::from(TokenBuilder::new("V").lemma("v").pos("ADJD")),
            conll_sentence[1].token().unwrap()
        );
        assert_eq!(
            &Token::from(TokenBuilder::new(",").lemma(",").pos("$,")),
            conll_sentence[2].token().unwrap()
        );
        assert_eq!(
            &Token::from(
                TokenBuilder::new("e")
                    .lemma("e")
                    .pos("ART")
                    .features(Features::from_string("gsf"))
            ),
            conll_sentence[6].token().unwrap()
        );
        let conll_sentence = Sentence::from(tree);
        assert_eq!(
            &Token::from(TokenBuilder::new("V").lemma("v").pos("ADJD")),
            conll_sentence[1].token().unwrap()
        );
        assert_eq!(
            &Token::from(TokenBuilder::new(",").lemma(",").pos("$,")),
            conll_sentence[2].token().unwrap()
        );
        assert_eq!(
            &Token::from(
                TokenBuilder::new("e")
                    .lemma("e")
                    .pos("ART")
                    .features(Features::from_string("gsf"))
            ),
            conll_sentence[6].token().unwrap()
        );
    }

    #[test]
    fn into_conllx() {
        let input = "(NX (NN Nounphrase) (PX (PP on) (NX (DET a) (ADJ single) (NX line))))";
        let mut tree = PTBFormat::TueBa.string_to_tree(input).unwrap();
        let nounphrase = tree.terminals().next().unwrap();
        tree[nounphrase]
            .features_mut()
            .insert::<_, String>("feature", None);
        tree[nounphrase].features_mut().insert("key", Some("val"));
        let conll_sentence = Sentence::from(tree);
        let mut target = Sentence::new();
        target.push(
            TokenBuilder::new("Nounphrase")
                .pos("NN")
                .features(Features::from_string("feature|key:val"))
                .into(),
        );
        target.push(TokenBuilder::new("on").pos("PP").into());
        target.push(TokenBuilder::new("a").pos("DET").into());
        target.push(TokenBuilder::new("single").pos("ADJ").into());
        target.push(TokenBuilder::new("line").pos("NX").into());
        assert_eq!(conll_sentence, target);
    }
}
