use std::cmp::Ordering;
use std::collections::HashSet;
use std::ops::Range;

use failure::Error;

/// Span of a node.
///
/// Spans are non-empty ranges that optionally skip indices.
///
/// Spans are non-inclusive and do not cover the `end`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// Lower bounds of the span.
    pub start: usize,
    /// Upper bounds of the span.
    pub end: usize,
    skips: Option<HashSet<usize>>,
}

impl From<usize> for Span {
    fn from(idx: usize) -> Self {
        Span {
            start: idx,
            end: idx + 1,
            skips: None,
        }
    }
}

impl Span {
    /// Create new span with skipped indices.
    ///
    /// `skips` has to be non-empty, to construct a span without skips, use `Span::new`.
    ///
    /// Skipped indices outside of lower and upper are ignored.
    pub(crate) fn new_with_skips(lower: usize, upper: usize, skips: HashSet<usize>) -> Self {
        assert!(lower < upper, "Span start has to be smaller then end.");
        assert!(
            !skips.is_empty(),
            "Skips have to be non-empty if using this constructor."
        );
        assert!(upper - lower - skips.len() > 0, "Can't skip all indices.");
        Span {
            start: lower,
            end: upper,
            skips: Some(skips),
        }
    }

    /// Create new continuous span.
    pub(crate) fn new(lower: usize, upper: usize) -> Self {
        assert!(lower < upper, "Span start has to be smaller then end.");
        Span {
            start: lower,
            end: upper,
            skips: None,
        }
    }

    /// Return whether then span covers the index.
    pub fn contains(&self, index: usize) -> bool {
        if self.start <= index && self.end > index {
            self.skips
                .as_ref()
                .map(|skips| !skips.contains(&index))
                .unwrap_or(true)
        } else {
            false
        }
    }

    /// Get this spans bounds as a tuple.
    pub fn bounds(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    /// Get the number of indices covered.
    pub fn n_indices(&self) -> usize {
        if let Some(ref skips) = self.skips {
            self.end - self.start - skips.len()
        } else {
            self.end - self.start
        }
    }

    /// Get the skipped indices of this span.
    ///
    /// Returns `None` if the span is continuous.
    pub fn skips(&self) -> Option<&HashSet<usize>> {
        self.skips.as_ref()
    }

    // Internally used constructor to build a span from a vec.
    pub(crate) fn from_vec(mut coverage: Vec<usize>) -> Result<Self, Error> {
        coverage.sort();
        let (lower, upper) = match (coverage.first(), coverage.last()) {
            (Some(first), Some(last)) => (*first, *last + 1),
            _ => return Err(format_err!("Can't build range from empty vec")),
        };
        let mut skips = HashSet::new();

        let mut prev = upper;
        for id in coverage.into_iter().rev() {
            if prev == id + 1 {
                prev = id;
            } else {
                // duplicate entries end up in this branch but don't get added since the range
                // (id + 1..prev) is empty
                skips.extend(id + 1..prev);

                prev = id;
            }
        }
        if skips.is_empty() {
            Ok(Span::new(lower, upper))
        } else {
            Ok(Span::new_with_skips(lower, upper, skips))
        }
    }

    // Method used internally to increment the upper bounds of a span.
    pub(crate) fn extend(&mut self) {
        self.end += 1;
    }
}

impl Ord for Span {
    /// Order of spans is determined by:
    ///   1. start index
    ///   2. end index
    ///   3. number of covered indices.
    fn cmp(&self, other: &Span) -> Ordering {
        if self.start != other.start {
            self.start.cmp(&other.start)
        } else if self.end != other.end {
            self.end.cmp(&other.end)
        } else {
            self.n_indices().cmp(&other.n_indices())
        }
    }
}

impl PartialOrd for Span {
    fn partial_cmp(&self, other: &Span) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> IntoIterator for &'a Span {
    type Item = usize;
    type IntoIter = SpanIter<'a>;

    fn into_iter(self) -> SpanIter<'a> {
        SpanIter {
            range: self.start..self.end,
            skip: self.skips.as_ref(),
        }
    }
}

/// Iterator over range excluding indices in `skip`.
#[derive(Debug, Eq, PartialEq)]
pub struct SpanIter<'a> {
    range: Range<usize>,
    skip: Option<&'a HashSet<usize>>,
}

impl<'a> Iterator for SpanIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(next) = self.range.next() {
            if let Some(skip) = self.skip {
                if skip.contains(&next) {
                    continue;
                }
            }
            return Some(next);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::Span;

    #[test]
    #[should_panic]
    fn invalid_cont_span_2_1() {
        Span::new(2, 1);
    }

    #[test]
    #[should_panic]
    fn invalid_skip_span_2_1() {
        Span::new(2, 1);
    }

    #[test]
    #[should_panic]
    fn invalid_skip_full_span() {
        let mut skip = HashSet::new();
        skip.insert(0);
        skip.insert(1);
        Span::new_with_skips(0, 2, skip);
    }

    #[test]
    fn simple_test() {
        let mut skip = HashSet::new();
        skip.insert(1);
        skip.insert(2);
        let span = Span::new_with_skips(0, 4, skip);
        assert_eq!(span.start, 0);
        assert_eq!(span.end, 4);
        assert_eq!(span.bounds(), (0, 4));
        assert_eq!(span.into_iter().collect::<Vec<_>>(), vec![0, 3]);
    }

    #[test]
    fn contains_skipspan() {
        let skip = vec![3, 5].into_iter().collect::<HashSet<usize>>();
        let span = Span::new_with_skips(0, 10, skip);
        assert!(span.contains(0));
        assert!(span.contains(1));
        assert!(span.contains(2));
        assert!(!span.contains(3));
        assert!(span.contains(4));
        assert!(!span.contains(5));
        assert!(span.contains(6));
        assert!(!span.contains(10));
    }

    #[test]
    fn contains_no_skips() {
        let span = Span::new(0, 10);
        assert!(span.contains(0));
        assert!(span.contains(1));
        assert!(span.contains(2));
        assert!(span.contains(4));
        assert!(span.contains(6));
        assert!(!span.contains(10))
    }

    #[test]
    fn contains_span() {
        let span = Span::new(0, 10);
        assert!(span.contains(0));
        assert!(span.contains(1));
        assert!(span.contains(2));
        assert!(span.contains(4));
        assert!(span.contains(6));
        assert!(!span.contains(10))
    }

    #[test]
    fn test_from_ordered_vec() {
        let v = vec![1, 2, 4, 6, 8, 10];
        let span = Span::from_vec(v.clone()).unwrap();
        for (target, test) in span.into_iter().zip(v) {
            assert_eq!(target, test)
        }
    }

    #[test]
    fn test_from_vec_duplicates() {
        let v = vec![1, 2, 4, 6, 8, 10];
        let v_t = vec![1, 4, 2, 4, 6, 8, 8, 10];
        let span = Span::from_vec(v_t).unwrap();
        for (target, test) in span.into_iter().zip(v) {
            assert_eq!(target, test)
        }
    }

    #[test]
    fn test_from_vec_single() {
        let v = vec![1];
        let span = Span::from_vec(v.clone()).unwrap();

        for (target, test) in span.into_iter().zip(v) {
            assert_eq!(target, test)
        }
    }

    #[test]
    #[should_panic]
    fn test_empty_vec() {
        let v = vec![];
        let span = Span::from_vec(v.clone()).unwrap();

        for (target, test) in span.into_iter().zip(v) {
            assert_eq!(target, test)
        }
    }

    #[test]
    fn test_from_vec() {
        let v = vec![1, 3];
        let span = Span::from_vec(v.clone()).unwrap();
        if let Some(_) = span.skips() {
            for (target, test) in span.into_iter().zip(v) {
                assert_eq!(target, test);
            }
        } else {
            assert_eq!(0, 1);
        }
    }
}
