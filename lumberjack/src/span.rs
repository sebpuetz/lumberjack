use std::cmp::Ordering;
use std::collections::HashSet;
use std::ops::Range;

use failure::Error;

/// Span enum.
///
/// Enum to represent the range of indices covered by a node.
#[derive(Debug, PartialEq, Eq)]
pub enum Span {
    /// Variant covering continuous indices.
    Continuous(ContinuousSpan),
    /// Variant covering discontinuous indices.
    Discontinuous(SkipSpan),
}

impl Clone for Span {
    fn clone(&self) -> Self {
        match self {
            Span::Continuous(span) => Span::Continuous(*span),
            Span::Discontinuous(span) => Span::Discontinuous(span.clone()),
        }
    }
}

impl Span {
    /// Return whether `index` is inside the `Span`.
    pub fn contains(&self, index: usize) -> bool {
        match self {
            Span::Continuous(span) => span.contains(index),
            Span::Discontinuous(span) => span.contains(index),
        }
    }

    /// Get this spans lower bounds.
    pub fn lower(&self) -> usize {
        match self {
            Span::Continuous(span) => span.lower,
            Span::Discontinuous(span) => span.lower,
        }
    }

    /// Get this spans upper bounds.
    pub fn upper(&self) -> usize {
        match self {
            Span::Continuous(span) => span.upper,
            Span::Discontinuous(span) => span.upper,
        }
    }

    /// Get this spans bounds as a tuple.
    pub fn bounds(&self) -> (usize, usize) {
        (self.lower(), self.upper())
    }

    /// Get the number of indices covered.
    pub fn n_indices(&self) -> usize {
        match self {
            Span::Discontinuous(span) => span.upper - span.lower - span.skip.len(),
            Span::Continuous(span) => span.upper - span.lower,
        }
    }

    pub (crate) fn discontinuous(&self) -> Option<&SkipSpan> {
        if let Span::Discontinuous(span) = self {
            Some(span)
        } else {
            None
        }
    }

    // Internally used constructor to build a span from a vec.
    pub(crate) fn from_vec(mut coverage: Vec<usize>) -> Result<Self, Error> {
        coverage.sort();
        let (lower, upper) = match (coverage.first(), coverage.last()) {
            (Some(first), Some(last)) => (*first, *last + 1),
            _ => return Err(format_err!("Can't build range from empty vec")),
        };
        let mut skip = HashSet::new();

        let mut prev = upper;
        for id in coverage.into_iter().rev() {
            if prev == id + 1 {
                prev = id;
            } else {
                // duplicate entries end up in this branch but don't get added since the range
                // (id + 1..prev) is empty
                skip.extend(id + 1..prev);

                prev = id;
            }
        }
        if !skip.is_empty() {
            Ok(Span::Discontinuous(SkipSpan::new(lower, upper, skip)))
        } else {
            Ok(Span::Continuous(ContinuousSpan::new(lower, upper)))
        }
    }

    // Method used internally to increment the upper bounds of a span.
    pub (crate) fn extend(&mut self) {
        match self {
            Span::Continuous(span) => span.upper += 1,
            Span::Discontinuous(span) => span.upper += 1,
        }
    }

    // Internally used constructor for convenience.
    pub (crate) fn new_continuous(lower: usize, upper: usize) -> Self {
        Span::Continuous(ContinuousSpan::new(lower, upper))
    }
}

/// Struct representing the span covered by node attached to this edge
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct ContinuousSpan {
    pub lower: usize,
    pub upper: usize,
}

impl ContinuousSpan {
    pub(crate) fn new(lower: usize, upper: usize) -> Self {
        assert!(lower < upper);
        ContinuousSpan { lower, upper }
    }

    /// Return whether `index` is inside the `Span`.
    pub fn contains(&self, index: usize) -> bool {
        index < self.upper && index >= self.lower
    }

    /// Get this spans lower bounds.
    pub fn lower(&self) -> usize {
        self.lower
    }

    /// Get this spans upper bounds.
    pub fn upper(&self) -> usize {
        self.upper
    }

    /// Get this spans bounds as a tuple.
    pub fn bounds(&self) -> (usize, usize) {
        (self.lower, self.upper)
    }

    /// Number of covered indices.
    pub fn n_indices(&self) -> usize {
        self.upper - self.lower
    }
}

/// Struct representing discontinuous Coverage
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkipSpan {
    lower: usize,
    upper: usize,
    skip: HashSet<usize>,
}

impl SkipSpan {
    // Internally used constructor for SkipSpan.
    pub(crate) fn new(lower: usize, upper: usize, skip: HashSet<usize>) -> Self {
        assert!(lower < upper);
        assert!(!skip.is_empty(), "SkipSpan hast to contain skipped indices.");
        assert_ne!(skip.len(), upper - lower, "Can't skip all indices.");

        SkipSpan { lower, upper, skip }
    }

    /// Return whether `index` is inside the `Span`.
    pub fn contains(&self, index: usize) -> bool {
        index < self.upper && index >= self.lower && !self.skip.contains(&index)
    }

    /// Get this span's skipped indices.
    pub fn skips(&self) -> &HashSet<usize> {
        &self.skip
    }

    /// Get this spans lower bounds.
    pub fn lower(&self) -> usize {
        self.lower
    }

    /// Get this spans upper bounds.
    pub fn upper(&self) -> usize {
        self.upper
    }

    /// Get this spans bounds as a tuple.
    pub fn bounds(&self) -> (usize, usize) {
        (self.lower, self.upper)
    }

    /// Number of covered indices.
    pub fn n_indices(&self) -> usize {
        self.upper - self.lower - self.skip.len()
    }
}

impl Ord for Span {
    fn cmp(&self, other: &Span) -> Ordering {
        if self.lower() != other.lower() {
            self.lower().cmp(&other.lower())
        } else if self.upper() != other.upper() {
            self.upper().cmp(&other.upper())
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
        match self {
            Span::Discontinuous(span) => span.into_iter(),
            Span::Continuous(span) => span.into_iter(),
        }
    }
}

impl<'a> IntoIterator for &'a SkipSpan {
    type Item = usize;
    type IntoIter = SpanIter<'a>;

    fn into_iter(self) -> SpanIter<'a> {
        SpanIter{range: self.lower..self.upper, skip: Some(&self.skip)}
    }
}

impl<'a> IntoIterator for &'a ContinuousSpan {
    type Item = usize;
    type IntoIter = SpanIter<'a>;

    fn into_iter(self) -> SpanIter<'a> {
        SpanIter{range: self.lower..self.upper, skip: None}
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
    use crate::span::{Span, ContinuousSpan};
    use crate::span::SkipSpan;
    use std::collections::HashSet;

    #[test]
    #[should_panic]
    fn terminal_invalid_cont_span_2_1() {
        Span::new_continuous(2, 1);
    }

    #[test]
    #[should_panic]
    fn terminal_invalid_skip_span_2_1() {
        Span::new_continuous(2, 1);
    }

    #[test]
    #[should_panic]
    fn terminal_invalid_skip_full_span() {
        let mut skip = HashSet::new();
        skip.insert(0);
        skip.insert(1);
        SkipSpan::new(0, 2, skip);
    }

    #[test]
    fn simple_test() {
        let mut skip = HashSet::new();
        skip.insert(1);
        skip.insert(2);
        let span = SkipSpan::new(0, 4, skip);
        assert_eq!(span.lower, 0);
        assert_eq!(span.upper, 4);
        assert_eq!(span.into_iter().collect::<Vec<_>>(), vec![0, 3]);
    }

    #[test]
    fn contains_skipspan() {
        let skip = vec![3, 5].into_iter().collect::<HashSet<usize>>();
        let span = SkipSpan::new(0, 10, skip);
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
    fn contains_contspan() {
        let span = ContinuousSpan::new(0, 10);
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
    fn test_enum_fromvec() {
        let v = vec![1, 3];
        let span = Span::from_vec(v.clone()).unwrap();
        if let Span::Discontinuous(span) = span {
            for (target, test) in span.into_iter().zip(v) {
                assert_eq!(target, test);
            }
        } else {
            assert_eq!(0, 1);
        }
    }
}
