use std::collections::btree_map::BTreeMap;
use std::iter::FromIterator;

use itertools::Itertools;

/// Features.
///
/// These can be e.g. morphological features on `Terminal`
/// or syntactic-semantic labels on `NonTerminal` nodes.
#[derive(Clone, Default, Debug, Eq, PartialEq)]
pub struct Features {
    map: BTreeMap<String, Option<String>>,
}

impl<S> From<S> for Features
where
    S: AsRef<str>,
{
    fn from(s: S) -> Self {
        s.as_ref()
            .split('|')
            .filter(|s| !s.is_empty())
            .map(|f| {
                let mut parts = f.split(':');
                let k = parts.next().unwrap();
                let v = parts.next();
                (k, v)
            })
            .collect()
    }
}

impl<K, V> FromIterator<(K, Option<V>)> for Features
where
    K: Into<String>,
    V: Into<String>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (K, Option<V>)>,
    {
        let map = iter
            .into_iter()
            .map(|(k, v)| (k.into(), v.map(Into::into)))
            .collect();
        Features { map }
    }
}

impl Features {
    /// Construct empty `Features`.
    pub fn new() -> Self {
        Features::default()
    }

    /// Get a slice of the backing `BTreeMap`.
    pub fn inner(&self) -> &BTreeMap<String, Option<String>> {
        &self.map
    }

    /// Get the backing `BTreeMap` mutably.
    pub fn inner_mut(&mut self) -> &mut BTreeMap<String, Option<String>> {
        &mut self.map
    }

    /// Insert `key` with `val`.
    ///
    /// If `key` was present, the replaced value is returned, otherwise `None`.
    pub fn insert<K, V>(&mut self, key: K, val: Option<V>) -> Option<String>
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.map
            .insert(key.into(), val.map(Into::into))
            .and_then(|v| v)
    }

    /// Get the value associated with `key`.
    pub fn get_val(&self, key: &str) -> Option<&str> {
        self.map
            .get(key)
            .and_then(|v| v.as_ref().map(String::as_str))
    }

    /// Remove the tuple associated with `key`.
    ///
    /// Returns `None` if `key` was not found.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        if let Some(rm) = self.map.remove(key) {
            rm
        } else {
            None
        }
    }
}

impl ToString for Features {
    fn to_string(&self) -> String {
        self.map
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
            vec![
                ("key", Some("value")),
                ("some_feature", None),
                ("another", Some("one"))
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(features.to_string(), "another:one|key:value|some_feature");
        assert_eq!(features.get_val("some_feature"), None);
        assert_eq!(features.get_val("key"), (Some("value")));
        assert_eq!(features.remove("some_feature"), None);
        assert_eq!(features.remove("nonsense"), None);
        assert_eq!(features.get_val("nonsense"), None);
        assert_eq!(
            features,
            vec![("key", Some("value")), ("another", Some("one"))]
                .into_iter()
                .collect()
        );
        let replace: Option<String> = None;
        assert_eq!(features.insert("key", replace), Some("value".into()));
    }
}
