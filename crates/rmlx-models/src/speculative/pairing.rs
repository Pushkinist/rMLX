//! What a drafter's config says about the verifier it was built for.
//!
//! A drafter reads the verifier's embeddings, hidden states or both, so it is
//! only useful on the base it was trained against. Under greedy verification a
//! drafter built for another base does not fail: the output stays the
//! verifier's own and only the accept rate falls, with nothing to say why. The
//! drafter's config restates some of the base's trunk, and this module is the
//! one comparison of those restatements with the verifier.

use std::path::Path;

use rmlx_core::error::{Error, Result};

use crate::arch::Architecture;

/// The trunk facts of a verifier that a drafter's config can restate.
#[allow(
    clippy::exhaustive_structs,
    reason = "plain data: the three facts a drafter config restates; callers without a loaded verifier build one by field"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifierTrunk {
    /// Width of the decoder-trunk hidden state.
    pub hidden_size: usize,
    /// Decoder layer count.
    pub num_hidden_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
}

/// The keys under which a drafter's config restates the verifier's trunk.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrunkKeys<'a> {
    /// Key of the verifier's hidden size.
    pub hidden_size: &'a str,
    /// Key of the verifier's layer count, or `None` when the config does not
    /// restate it. The drafter's own `num_hidden_layers` is its own depth for
    /// some families, so this is not `num_hidden_layers` for all of them.
    pub num_hidden_layers: Option<&'a str>,
    /// Key of the verifier's vocabulary size.
    pub vocab_size: &'a str,
}

impl<'a> TrunkKeys<'a> {
    /// Keys that spell the hidden size and the vocabulary as the verifier's
    /// config does, with the layer count under `num_hidden_layers`, if any.
    pub(crate) const fn named(num_hidden_layers: Option<&'a str>) -> Self {
        Self {
            hidden_size: "hidden_size",
            num_hidden_layers,
            vocab_size: "vocab_size",
        }
    }
}

impl VerifierTrunk {
    /// The trunk facts of a loaded verifier.
    pub fn of(verifier: &Architecture) -> Self {
        Self {
            hidden_size: verifier.hidden_size(),
            num_hidden_layers: verifier.num_hidden_layers(),
            vocab_size: verifier.vocab_size(),
        }
    }

    /// Refuse a drafter whose config declares a trunk fact the verifier does not
    /// have. A key that is absent or not an integer declares nothing; a config
    /// that declares none is allowed through with a warning naming `draft_dir`,
    /// since there is nothing to compare.
    ///
    /// `source` names where `declared` came from, for the message. `declared` are
    /// the maps of the config that hold the keys, searched in order. `keys` says
    /// under which key the config restates each fact.
    ///
    /// # Errors
    /// [`Error::SpeculativePairing`] naming the key, the drafter's value and the
    /// verifier's value.
    pub(crate) fn refuse_another_base(
        self,
        draft_dir: &Path,
        source: &str,
        declared: &[&serde_json::Map<String, serde_json::Value>],
        keys: TrunkKeys<'_>,
    ) -> Result<()> {
        let facts = [
            (Some(keys.hidden_size), "hidden_size", self.hidden_size),
            (
                keys.num_hidden_layers,
                "num_hidden_layers",
                self.num_hidden_layers,
            ),
            (Some(keys.vocab_size), "vocab_size", self.vocab_size),
        ];
        let mut declared_any = false;
        for (key, fact, verifier_value) in facts {
            let Some(key) = key else { continue };
            let Some(value) = declared
                .iter()
                .find_map(|map| map.get(key).and_then(serde_json::Value::as_u64))
            else {
                continue;
            };
            declared_any = true;
            if value != verifier_value as u64 {
                return Err(Error::SpeculativePairing {
                    reason: format!(
                        "{source} declares {key} = {value} and the verifier has {fact} = \
                         {verifier_value}; the drafter was built for another base model"
                    ),
                });
            }
        }
        if !declared_any {
            tracing::warn!(
                draft = %draft_dir.display(),
                source,
                "the drafter's config declares none of the keys that restate the verifier's \
                 hidden size, layer count and vocabulary; it cannot be checked against the \
                 verifier's base model"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "pairing_tests.rs"]
mod pairing_tests;
