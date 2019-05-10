use crate::{Terminal, Tree};
use conllx::graph::Sentence;
use conllx::token::{Features, Token};

/// Conversion Trait to CONLLX.
///
/// Creates a `Token` for each `Terminal` in the tree.
/// Form, POS and Features are carried over.
pub trait ToConllx {
    /// Nonconsuming conversion to CONLLX.
    fn to_conllx(&self) -> Sentence;
    /// Consuming conversion to CONLLX.
    fn into_conllx(self) -> Sentence;
}

impl ToConllx for Tree {
    fn to_conllx(&self) -> Sentence {
        self.into()
    }

    fn into_conllx(self) -> Sentence {
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
            tokens.push((token, terminal.span().lower()));
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
            .filter_map(|t| tree[t].terminal().map(|t| (t.into(), t.span().lower())))
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
    use crate::io::ReadTree;

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
        let conll_sentence = tree.into_conllx();
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
