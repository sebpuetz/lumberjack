use std::mem;

use itertools::Itertools;

/// Features.
///
/// These can be e.g. morphological features on `Terminal`
/// or syntactic-semantic labels on `NonTerminal` nodes.
#[derive(Clone, Default, Debug, Eq, PartialEq)]
pub struct Features {
    vec: Vec<(String, Option<String>)>,
}

impl<S> From<S> for Features
where
    S: AsRef<str>,
{
    fn from(s: S) -> Self {
        let vec = s
            .as_ref()
            .split('|')
            .map(|f| {
                if let Some(idx) = f.find(':') {
                    let (k, v) = f.split_at(idx);
                    (k.into(), Some(v[1..].into()))
                } else {
                    (f.into(), None)
                }
            })
            .collect();
        Features { vec }
    }
}

impl Features {
    /// Construct empty `Features`.
    pub fn new() -> Self {
        Features::default()
    }

    /// Construct `Features` from `vec`.
    pub fn from_vec(vec: Vec<(String, Option<String>)>) -> Self {
        Features { vec }
    }

    /// Get a slice of the backing `Vec`.
    pub fn inner(&self) -> &[(String, Option<String>)] {
        &self.vec
    }

    /// Get the backing `Vec` mutably.
    pub fn inner_mut(&mut self) -> &mut Vec<(String, Option<String>)> {
        &mut self.vec
    }

    /// Insert `key` with `val`.
    ///
    /// If `key` was present, the replaced value is returned, otherwise `None`.
    pub fn insert<K, V>(&mut self, key: K, val: Option<V>) -> Option<String>
    where
        V: Into<String>,
        K: AsRef<str>,
    {
        let key = key.as_ref();
        let val = val.map(Into::into);
        for i in 0..self.vec.len() {
            if self.vec[i].0 == key {
                return mem::replace(&mut self.vec[i].1, val);
            }
        }
        self.vec.push((key.into(), val));
        None
    }

    /// Get the value associated with `key`.
    pub fn get_val(&self, key: &str) -> Option<&str> {
        self.vec.iter().find_map(|(k, v)| {
            if key == k.as_str() {
                v.as_ref().map(String::as_str)
            } else {
                None
            }
        })
    }

    /// Remove the tuple associated with `key`.
    ///
    /// Returns `None` if `key` was not found.
    pub fn remove(&mut self, key: &str) -> Option<(String, Option<String>)> {
        for i in 0..self.vec.len() {
            if self.vec[i].0 == key {
                return Some(self.vec.remove(i));
            }
        }
        None
    }
}

impl ToString for Features {
    fn to_string(&self) -> String {
        self.vec
            .iter()
            .map(|(k, v)| {
                if let Some(v) = v {
                    format!("{}:{}", k, v)
                } else {
                    k.to_owned()
                }
            })
            .join("|")
    }
}

#[cfg(test)]
mod test {
    use super::Features;

    #[test]
    fn features_test() {
        let mut features = Features::from("key:value|some_feature|another:one");
        assert_eq!(
            features,
            Features::from_vec(vec![
                ("key".into(), Some("value".into())),
                ("some_feature".into(), None),
                ("another".into(), Some("one".into()))
            ])
        );
        assert_eq!(features.to_string(), "key:value|some_feature|another:one");
        assert_eq!(features.get_val("some_feature"), None);
        assert_eq!(features.get_val("key"), Some("value"));
        assert_eq!(
            features.remove("some_feature"),
            Some(("some_feature".into(), None))
        );
        assert_eq!(features.remove("nonsense"), None);
        assert_eq!(features.get_val("nonsense"), None);
        assert_eq!(
            features,
            Features::from_vec(vec![
                ("key".into(), Some("value".into())),
                ("another".into(), Some("one".into()))
            ])
        );
        let replace: Option<String> = None;
        assert_eq!(features.insert("key", replace), Some("value".into()));
    }
}
