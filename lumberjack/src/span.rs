use std::cmp::Ordering;
use std::collections::HashSet;
use std::ops::Range;

use crate::Projectivity;
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
        // there is currently no way to ensure that skips are within this span.
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

    /// Return whether the span covers the index.
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

    /// Return whether the span covers a superset of indices of another span.
    pub fn covers_span(&self, other: &Span) -> bool {
        if self.start <= other.start && self.end >= other.end {
            if let Some(skips) = self.skips.as_ref() {
                for skip in skips {
                    if other.contains(*skip) {
                        return false;
                    }
                }
            }
            true
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

    /// Merge coverage of spans.
    pub(crate) fn merge_spans(&mut self, span: &Span) -> Span {
        // find bounds of merged span
        let start = if self.start <= span.start {
            self.start
        } else {
            span.start
        };
        let end = if self.end >= span.end {
            self.end
        } else {
            span.end
        };

        // determine skips by what is not contained in either span.
        let mut skips = HashSet::new();
        for i in start..end {
            if !self.contains(i) && !span.contains(i) {
                skips.insert(i);
            }
        }

        // determine whether span has skipped indices
        if skips.is_empty() {
            Span::new(start, end)
        } else {
            Span::new_with_skips(start, end, skips)
        }
    }

    /// Remove indices from span.
    ///
    /// Panics if `indices` provides as many or more indices than the span covers or if a provided
    /// index is out-of-bounds.
    pub(crate) fn remove_indices(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
    ) -> Projectivity {
        // insert all indices to be removed into skips
        let start = self.start;
        let end = self.end;
        let indices = indices.into_iter().map(|idx| {
            assert!(idx >= start && idx < end);
            idx
        });
        let skips = if let Some(skips) = self.skips.as_mut() {
            skips.extend(indices);
            skips
        } else {
            self.skips = Some(indices.into_iter().collect::<HashSet<_>>());
            self.skips.as_mut().unwrap()
        };

        // make sure not the entire span is removed, false positives can come from indices outside
        // of the span's bounds.
        assert!(
            skips.len() < (self.end - self.start),
            "Can't remove all indices."
        );

        // if bounds are part of the skipped indices, adjust bounds and remove from skips
        while skips.remove(&self.start) {
            self.start += 1;
        }
        // upper bound is exclusive, thus check whether end - 1 is skipped.
        while skips.remove(&(self.end - 1)) {
            self.end -= 1;
        }

        if self.skips.as_ref().map(HashSet::is_empty).unwrap_or(false) {
            self.skips = None;
            Projectivity::Projective
        } else {
            Projectivity::Nonprojective
        }
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

    use crate::Projectivity;
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
    fn remove_indices() {
        let skip = vec![3, 5].into_iter().collect::<HashSet<usize>>();
        let span = Span::new_with_skips(0, 10, skip);
        let mut clone = span.clone();
        assert_eq!(Projectivity::Projective, clone.remove_indices(0..5));
        assert_eq!(
            clone,
            Span::from_vec((6..10).into_iter().collect()).unwrap()
        );
        assert_eq!(Projectivity::Nonprojective, clone.remove_indices(vec![7]));
        assert_eq!(clone, Span::from_vec(vec![6, 8, 9]).unwrap());
    }
    #[test]
    #[should_panic]
    fn remove_all_indices() {
        let skip = vec![3, 5].into_iter().collect::<HashSet<usize>>();
        let mut span = Span::new_with_skips(0, 10, skip);
        span.remove_indices(0..10);
    }

    #[test]
    fn merge_spans() {
        let skip = vec![3, 5].into_iter().collect::<HashSet<usize>>();
        let span = Span::new_with_skips(0, 10, skip);
        let mut other = span.clone();
        assert_eq!(Projectivity::Nonprojective, other.remove_indices(vec![6]));
        other = other.merge_spans(&span);
        assert_eq!(other, span);

        let new_span = Span::from_vec(vec![3, 5]).unwrap();
        other = other.merge_spans(&new_span);
        assert!(other.skips().is_none());
        assert_eq!(
            other,
            Span::from_vec((0..10).into_iter().collect()).unwrap()
        );

        let skip = Span::from_vec((15..20).into_iter().collect()).unwrap();
        other = other.merge_spans(&skip);
        assert!(other.skips().is_some());
        assert_eq!(
            other,
            Span::from_vec(
                (0..20)
                    .into_iter()
                    .filter(|&idx| idx < 10 || idx >= 15)
                    .collect()
            )
            .unwrap()
        )
    }

    #[test]
    fn covers_span() {
        let skip = vec![3, 5].into_iter().collect::<HashSet<usize>>();
        let span = Span::new_with_skips(0, 10, skip);
        assert!(span.covers_span(&0.into()), "start span excluded");
        assert!(span.covers_span(&1.into()), "inside span excluded");
        assert!(!span.covers_span(&3.into()), "skipped idx included");
        assert!(!span.covers_span(&10.into()), "end span included");

        let skip = vec![3, 5].into_iter().collect::<HashSet<usize>>();
        let other = Span::new_with_skips(0, 10, skip);
        assert!(span.covers_span(&other));
        assert!(other.covers_span(&span));

        let skip = vec![7, 8].into_iter().collect::<HashSet<usize>>();
        let other = Span::new_with_skips(6, 10, skip);
        assert!(span.covers_span(&other));
        assert!(!other.covers_span(&span));

        let skip = vec![7, 8].into_iter().collect::<HashSet<usize>>();
        let other = Span::new_with_skips(6, 10, skip);
        assert!(span.covers_span(&other));
        assert!(!other.covers_span(&span));

        let skip = vec![3, 4, 5].into_iter().collect::<HashSet<usize>>();
        let other = Span::new_with_skips(0, 10, skip);
        assert!(span.covers_span(&other));
        assert!(!other.covers_span(&span));

        let skip = vec![3, 4].into_iter().collect::<HashSet<usize>>();
        let other = Span::new_with_skips(0, 10, skip);
        assert!(!span.covers_span(&other));
        assert!(!other.covers_span(&span));

        let skip = vec![3].into_iter().collect::<HashSet<usize>>();
        let other = Span::new_with_skips(0, 10, skip);
        assert!(!span.covers_span(&other));
        assert!(other.covers_span(&span));
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
    fn test_empty_vec() {
        assert!(Span::from_vec(vec![]).is_err());
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
