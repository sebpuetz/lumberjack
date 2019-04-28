use std::collections::HashMap;
use std::io::{BufRead, Lines};

use failure::Error;
use pest::iterators::Pair;
use pest::Parser;
use petgraph::stable_graph::StableGraph;

use crate::node::{NTBuilder, TerminalBuilder};
use crate::{Edge, Node, NonTerminal, Projectivity, Span, Tree};

/// Iterator over constituency trees in a NEGRA export file.
///
/// `next()` moves the reader until the first `#BOS` is found, then collects lines until `#EOS` is
/// found. Returns `Some(Error)` if overlapping sentences are found, `'#BOS` -> `#EOS` is violated
/// or if the collected lines are not a well formed NEGRA sentence.
///
/// Note:   If the reader never encounters a line according to `Rule::bos`, `None` is returned.
///         `Rule::bos` expects a line starting with `#BOS SENT_ID` followed by optional comments.
pub struct NegraTreeIter<R>
where
    R: BufRead,
{
    inner: Lines<R>,
}

impl<R> NegraTreeIter<R>
where
    R: BufRead,
{
    /// Creates a new `NegraTreeIter` over the NEGRA trees in the reader.
    pub fn new(reader: R) -> NegraTreeIter<R> {
        NegraTreeIter {
            inner: reader.lines(),
        }
    }
}
impl<R> Iterator for NegraTreeIter<R>
where
    R: BufRead,
{
    type Item = Result<Tree, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut inside = false;
        let mut buffer = String::new();

        while let Some(line) = self.inner.next() {
            let line = match line {
                Ok(line) => line,
                Err(e) => return Some(Err(format_err!("{}", e))),
            };
            if line.starts_with("#BOS") {
                // if we encounter "#BOS" while inside a sentence, return error
                if inside {
                    return Some(Err(format_err!("Found second #BOS before #EOS\n{}", line)));
                }
                inside = true;
            }

            if inside {
                buffer.push_str(&line);
                buffer.push('\n');
            }

            if line.starts_with("#EOS") {
                if inside {
                    return Some(negra_to_tree(&buffer));
                } else {
                    return Some(Err(format_err!("Found #EOS without #BOS\n{}", line)));
                }
            }
        }
        None
    }
}

/// Builds `Tree<NonProjectiveEdge>` from `&str`.
///
/// Assumes following structure:
/// #BOS
/// TERMINAL
/// NONTERMINAL
/// #EOS
///
/// Note: This method does not skip comments or newlines
pub fn negra_to_tree(negra_string: &str) -> Result<Tree, Error> {
    let mut parsed_line = NEGRAParser::parse(Rule::sent, negra_string)?;
    build_tree(parsed_line.next().unwrap())
}

// dummy struct required by pest
#[derive(Parser)]
#[grammar = "io/negra.pest"]
struct NEGRAParser;

fn build_tree(pair: Pair<Rule>) -> Result<Tree, Error> {
    let mut graph = StableGraph::new();
    match pair.as_rule() {
        Rule::sent => (),
        _ => return Err(format_err!("Expected sent, got {:?}", pair)),
    }
    let mut pairs = pair.into_inner();
    let bos = pairs.next().unwrap();

    // safe to do since we only enter this method after matching on Rule::sent
    let start_id = bos.into_inner().next().unwrap().as_str().parse::<usize>()?;

    // map parent_id -> [(edge, child_id),[..]]
    let mut edges = HashMap::new();
    let mut n_terminals = 0;
    let mut projectivity = Projectivity::Projective;
    for pair in pairs {
        match pair.as_rule() {
            Rule::terminal => {
                let (parent, edge, terminal) = process_terminal(pair, n_terminals)?;
                let idx = graph.add_node(terminal);

                edges
                    .entry(parent)
                    .or_insert_with(Vec::new)
                    .push((edge, idx));
                n_terminals += 1;
            }
            Rule::nonterminal => {
                let (parent, self_id, edge, nonterminal) = process_nonterminal(pair)?;
                let mut coverage = Vec::new();
                // following nodes are higher in the tree, nonterminal is finished, just add all edges
                let edge_list = edges.remove(&self_id).ok_or_else(|| {
                    format_err!("Nonterminal without children:\n{:?}", nonterminal)
                })?;

                // iterate over children to collect indices covered by this nonterminal
                for (_, node) in edge_list.iter() {
                    coverage.extend(graph[*node].span());
                }

                let span = Span::from_vec(coverage)?;
                if span.discontinuous().is_some() {
                    projectivity = Projectivity::Nonprojective;
                }
                let nt = nonterminal.span(span).try_into_nt()?;

                // add nonterminal and outgoing edges to graph
                let idx = graph.add_node(Node::NonTerminal(nt));
                for (edge, node) in edge_list {
                    graph.add_edge(idx, node, edge);
                }
                // every nonterminal has a parent, either root or other nonterminal
                edges
                    .entry(parent)
                    .or_insert_with(Vec::new)
                    .push((edge, idx));
            }
            Rule::sent_end => {
                let end_id = pair
                    .into_inner()
                    .next()
                    .unwrap()
                    .as_str()
                    .parse::<usize>()?;
                // guard to make sure we stayed in the same sentence
                if end_id != start_id {
                    return Err(format_err!("Mismatch in sentence ID"));
                }
                // return error if no node had 0 (root) as parent
                let edge_list = edges
                    .remove(&0)
                    .ok_or_else(|| format_err!("Sentence without root"))?;

                // root is guaranteed to cover sentence, thus span is 0..n_terminals
                let span = Span::new_continuous(0, n_terminals);
                let root = graph.add_node(Node::NonTerminal(NonTerminal::new("VROOT", span)));
                for (edge, node) in edge_list {
                    graph.add_edge(root, node, edge);
                }
                return Ok(Tree::new(graph, n_terminals, root, projectivity));
            }
            _ => unreachable!(),
        }
    }
    Err(format_err!("Tree without content"))
}

// returns a tuple of parent_id, own_id, parent_edge, ntbuilder
// NTBuilder is returned rather than NonTerminal because span depends on other nodes.
fn process_nonterminal(pair: Pair<Rule>) -> Result<(usize, usize, Edge, NTBuilder), Error> {
    // nonterminal rule is defined as:
    // ID ~ other ~ label ~ other ~ edge_label ~ ID ~ consume_line? ~ NEWLINE
    // thus safe to unwrap up to second ID as the rule would not match otherwise
    match pair.as_rule() {
        Rule::nonterminal => (),
        _ => return Err(format_err!("Expected terminal, got {:?}", pair)),
    }
    let mut parts = pair.into_inner();
    let self_id = parts.next().unwrap().as_str().parse::<usize>()?;
    //other
    parts.next();
    let mut label_parts = parts.next().unwrap().as_str().split('=');
    let label = label_parts.next().unwrap();
    let annotation = label_parts.next().map(|s| s.to_owned());
    //other
    parts.next();
    let edge = parts.next().unwrap().as_str();
    let edge = if edge == "--" { None } else { Some(edge) };
    let parent_id = parts.next().unwrap().as_str().parse::<usize>()?;
    let ntbuilder = NTBuilder::new(label).annotation(annotation);
    Ok((parent_id, self_id, edge.into(), ntbuilder))
}

// returns parent_id, parent_edge, terminal
fn process_terminal(pair: Pair<Rule>, idx: usize) -> Result<(usize, Edge, Node), Error> {
    // terminal rule is defined as:
    // form ~ lemma ~ pos ~ morph ~ edge_label ~ ID ~ consume_line? ~ NEWLINE
    // thus safe to unwrap up to ID
    match pair.as_rule() {
        Rule::terminal => (),
        _ => return Err(format_err!("Expected terminal, got {:?}", pair)),
    }
    let mut parts = pair.into_inner();

    let form = parts.next().unwrap().as_str();
    let lemma = parts.next().unwrap().as_str();
    let pos = parts.next().unwrap().as_str();
    let morph = parts.next().unwrap().as_str();
    let edge = parts.next().unwrap().as_str();
    // unlabeled edges denoted as "--" in tueba
    let edge = if edge == "--" { None } else { Some(edge) };
    let parent_id = parts.next().unwrap().as_str().parse::<usize>()?;
    let span = Span::new_continuous(idx, idx + 1);
    let mut builder = TerminalBuilder::new(form, pos, span).lemma(lemma);
    if morph != "--" {
        builder = builder.morph(morph);
    }

    Ok((
        parent_id,
        edge.into(),
        Node::Terminal(builder.try_into_terminal()?),
    ))
}

#[cfg(test)]
mod tests {
    use pest::Parser;
    use petgraph::prelude::StableGraph;
    use std::fs;
    use std::fs::File;
    use std::io::BufReader;

    use super::{
        negra_to_tree, process_nonterminal, process_terminal, NEGRAParser, NegraTreeIter, Rule,
    };

    use crate::{Edge, Node, NonTerminal, Projectivity, Span, Terminal, Tree};

    #[test]
    fn test_first10_ok() {
        let input = File::open("testdata/10.negra").unwrap();
        let reader = BufReader::new(input);
        let mut n = 0;
        for tree in NegraTreeIter::new(reader) {
            tree.unwrap();
            n += 1;
        }
        assert_eq!(n, 10)
    }

    #[test]
    fn test_iter() {
        let f = File::open("testdata/10.negra").unwrap();
        let mut iter = NegraTreeIter::new(BufReader::new(f));
        iter.next();
        iter.next();
        let string_rep = "#BOS 3  2 1070544990 0 %% HEADLINE\n\
                          L	L	NN	nsf		HD	500\n\
                          U			U			NE	nsf		-	501\n\
                          W		W		NE	nsf		-	501\n\
                          :			:			$.	--		--	0\n\
                          E		e			ART	nsm		-	503\n\
                          B		B		NN	nsm		HD	503\n\
                          #500			--			NX	--		APP	502\n\
                          #501			--			NX=PER	--		APP	502\n\
                          #502			--			NX	--		--	0\n\
                          #503			--			NX	--		--	0\n\
                          #EOS 3";
        let target = negra_to_tree(string_rep).unwrap();
        assert_eq!(iter.next().unwrap().unwrap(), target)
    }

    #[test]
    fn test_first() {
        let f = File::open("testdata/single.negra").unwrap();
        let br = BufReader::new(f);
        let tree = NegraTreeIter::new(br).next().unwrap().unwrap();
        let mut g = StableGraph::new();
        let mut v = Terminal::new("V", "VVFIN", Span::new_continuous(0, 1));
        v.set_lemma(Some("v"));
        v.set_morph(Some("3sit"));
        let mut d = Terminal::new("d", "ART", Span::new_continuous(1, 2));
        d.set_lemma(Some("d"));
        d.set_morph(Some("nsf"));
        let mut a = Terminal::new("A", "NN", Span::new_continuous(2, 3));
        a.set_lemma(Some("a"));
        a.set_morph(Some("nsf"));
        let mut s = Terminal::new("S", "NN", Span::new_continuous(3, 4));
        s.set_lemma(Some("s"));
        s.set_morph(Some("asn"));
        let mut punct = Terminal::new("?", "$.", Span::new_continuous(4, 5));
        punct.set_lemma(Some("?"));

        let mf = g.add_node(Node::NonTerminal(NonTerminal::new(
            "MF",
            Span::new_continuous(1, 4),
        )));
        let a = g.add_node(Node::Terminal(a));
        let v = g.add_node(Node::Terminal(v));
        let lk = g.add_node(Node::NonTerminal(NonTerminal::new(
            "LK",
            Span::new_continuous(0, 1),
        )));
        let d = g.add_node(Node::Terminal(d));
        let mut nxorg = NonTerminal::new("NX", Span::new_continuous(1, 3));
        nxorg.set_annotation(Some("ORG"));
        let s = g.add_node(Node::Terminal(s));
        let nxorg = g.add_node(Node::NonTerminal(nxorg));
        let root = g.add_node(Node::NonTerminal(NonTerminal::new(
            "VROOT",
            Span::new_continuous(0, 5),
        )));
        let punct = g.add_node(Node::Terminal(punct));
        let simpx = g.add_node(Node::NonTerminal(NonTerminal::new(
            "SIMPX",
            Span::new_continuous(0, 4),
        )));
        let nx = g.add_node(Node::NonTerminal(NonTerminal::new(
            "NX",
            Span::new_continuous(3, 4),
        )));
        let vxfin = g.add_node(Node::NonTerminal(NonTerminal::new(
            "VXFIN",
            Span::new_continuous(0, 1),
        )));
        g.add_edge(mf, nx, Edge::from(Some("OA")));
        g.add_edge(vxfin, v, Edge::from(Some("HD")));
        g.add_edge(nxorg, d, Edge::from(Some("-NE")));
        g.add_edge(lk, vxfin, Edge::from(Some("HD")));
        g.add_edge(mf, nxorg, Edge::from(Some("ON")));
        g.add_edge(simpx, lk, Edge::from(Some("-")));
        g.add_edge(root, simpx, Edge::default());
        g.add_edge(simpx, mf, Edge::from(Some("-")));
        g.add_edge(root, punct, Edge::default());
        g.add_edge(nxorg, a, Edge::from(Some("HD")));
        g.add_edge(nx, s, Edge::from(Some("HD")));
        assert_eq!(tree, Tree::new(g, 5, root, Projectivity::Projective));
    }

    #[test]
    fn test_string() {
        let t = fs::read_to_string("testdata/long_single.negra").unwrap();
        let target_surface_form = "V , s d S e M d A , s d L U W , d s j a \" S \" g , \" w \
                                   d a w , w e s m T z \" . ";
        let tree = negra_to_tree(&t).unwrap();
        let surface_form = tree
            .terminals()
            .filter_map(|id| tree[id].terminal())
            .map(|terminal| terminal.form())
            .fold(String::new(), |mut acc, form| {
                acc.push_str(form);
                acc.push(' ');
                acc
            });
        assert_eq!(surface_form, target_surface_form)
    }

    #[test]
    fn terminal() {
        let term = "was etwas   PIS *** HD  502 %some random comment that gets ignored\n";
        let mut v = NEGRAParser::parse(Rule::terminal, term).unwrap();
        let (parent_id, edge, terminal) = process_terminal(v.next().unwrap(), 0).unwrap();
        assert_eq!(parent_id, 502);
        assert_eq!(edge, Edge::from(Some("HD")));
        let mut term = Terminal::new("was", "PIS", Span::new_continuous(0, 1));
        term.set_lemma(Some("etwas"));
        term.set_morph(Some("***"));
        assert_eq!(terminal, Node::Terminal(term));

        let term = "#was etwas   PIS *** HD 502\n";
        NEGRAParser::parse(Rule::terminal, term).unwrap();
        assert!(NEGRAParser::parse(Rule::nonterminal, term).is_err());
    }

    #[test]
    fn nonterminal() {
        let nt = "#502			--			NX	--		ON	503\n";
        let mut v = NEGRAParser::parse(Rule::nonterminal, nt).unwrap();
        let (parent_id, own_id, edge, ntbuilder) = process_nonterminal(v.next().unwrap()).unwrap();
        let span = Span::new_continuous(0, 1);
        let ntbuilder = ntbuilder.span(span.clone());
        assert_eq!(parent_id, 503);
        assert_eq!(own_id, 502);
        assert_eq!(edge, Edge::from(Some("ON")));
        let nt = NonTerminal::new("NX", span);
        assert_eq!(ntbuilder.try_into_nt().unwrap(), nt);
    }

    #[test]
    #[should_panic]
    fn terminal_without_parent() {
        let bad_term = "was etwas   PIS *** HD\n";
        NEGRAParser::parse(Rule::terminal, bad_term).unwrap();
    }

    #[test]
    #[should_panic]
    fn terminal_missing_field() {
        let bad_term = "was    PIS *** HD 502\n";
        NEGRAParser::parse(Rule::terminal, bad_term).unwrap();
    }

    #[test]
    fn wellformed() {
        let s = "#BOS 1  2 1202391857 0 %% HEADLINE
        V		v		VVFIN	3sit		HD	500
        die			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0
        #EOS 1";
        NEGRAParser::parse(Rule::sent, s).unwrap();
    }

    #[test]
    #[should_panic]
    fn malformed_missing_bos() {
        let s = "V		v		VVFIN	3sit		HD	500
        d			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0
        #EOS 1";
        NEGRAParser::parse(Rule::sent, s).unwrap();
    }

    #[test]
    #[should_panic]
    fn malformed_invalid_start() {
        let s = "
        #BOS 1  2 1202391857 0 %% HEADLINE
        V		v		VVFIN	3sit		HD	500
        d			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0
        #EOS 1";
        NEGRAParser::parse(Rule::sent, s).unwrap();
    }

    #[test]
    #[should_panic]
    fn malformed_missing_eos() {
        let s = "#BOS 1  2 1202391857 0 %% HEADLINE
        V		v		VVFIN	3sit		HD	500
        d			d			ART	nsf		-NE	502
        A			a			NN	nsf		HD	502
        S		s		NN	asn		HD	503
        ?			?			$.	--		--	0
        #500			--			VXFIN	--		HD	501
        #501			--			LK	--		-	505
        #502			--			NX=ORG	--		ON	504
        #503			--			NX	--		OA	504
        #504			--			MF	--		-	505
        #505			--			SIMPX	--		--	0";
        NEGRAParser::parse(Rule::sent, s).unwrap();
    }

    #[test]
    fn wellformed_no_nts() {
        let s = "#BOS 1  2 1202391857 0 %% HEADLINE
        V		v		VVFIN	3sit		HD	0
        d			d			ART	nsf		-NE	0
        A			a			NN	nsf		HD	0
        S		s		NN	asn		HD	0
        ?			?			$.	--		--	0
        #EOS 1";
        NEGRAParser::parse(Rule::sent, s).unwrap();
    }
}
