//! `DraftKind` — which drafter a `--draft-model` snapshot is, and how that is
//! decided.

use std::str::FromStr;

/// Speculative drafter kind.
///
/// Decides which loader builds the drafter and which round loop drives the
/// request. Three kinds are sidecar heads that hook into the verifier's forward
/// pass; [`DraftKind::TwoModel`] is the classic form — a separate, smaller full
/// model of the same family, loaded as its own `Architecture` and run against
/// the verifier by `SpeculativeDispatcher::spec_generate_greedy`.
///
/// `rmlx-models` carries the plain enum (no clap dep). The `clap::ValueEnum`
/// adapter lives in `rmlx-cli::main` (see `DraftKindArg`).
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — the compiler forces a new kind through as_str(), index() and every dispatch match; ALL then puts it in front of the tests that reach the string surfaces (from_str, the clap value) the compiler cannot"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftKind {
    /// Multi-Token Prediction sidecar (Qwen3.5-family head, Gemma4 assistant).
    Mtp,
    /// Draft-Flash block drafter (attention-based draft head).
    DFlash,
    /// EAGLE-3 speculative drafter.
    Eagle3,
    /// A full draft model: any registered architecture sharing the verifier's
    /// vocabulary.
    TwoModel,
}

impl DraftKind {
    /// Every kind, once — the population the string-surface tests sweep.
    ///
    /// Paired with [`Self::index`], whose match is exhaustive: a new kind does
    /// not compile until it has an index, and `every_kind_is_in_all_once` does
    /// not pass until it is in this list at that index. The parser's `other`
    /// arm and the CLI value enum are separate string tables the compiler
    /// cannot tie to this enum; sweeping them from here is what makes a kind
    /// that is unreachable from the flag fail a test rather than go quiet.
    pub const ALL: &'static [Self] = &[Self::Mtp, Self::DFlash, Self::Eagle3, Self::TwoModel];

    /// This kind's position in [`Self::ALL`].
    pub const fn index(self) -> usize {
        match self {
            Self::Mtp => 0,
            Self::DFlash => 1,
            Self::Eagle3 => 2,
            Self::TwoModel => 3,
        }
    }

    /// Canonical name used in log fields, `decode_config` and the CLI value.
    pub fn as_str(self) -> &'static str {
        match self {
            DraftKind::Mtp => "mtp",
            DraftKind::DFlash => "dflash",
            DraftKind::Eagle3 => "eagle3",
            DraftKind::TwoModel => "two_model",
        }
    }
}

/// What a draft snapshot's `config.json` says about it.
///
/// Read by [`Declared::from_snapshot`] from `architectures[0]` and
/// `model_type` — both, because export tools set one or the other: the Gemma4
/// assistant carries `Gemma4AssistantForCausalLM` / `gemma4_assistant`, a
/// Qwen3.5-family MTP sidecar carries `model_type = qwen3_5_mtp` and no
/// `architectures` at all, DFlash carries `DFlash*DraftModel` over a plain
/// `qwen3` model type, and EAGLE-3 carries `*Eagle3` over `llama`. The
/// architecture is read first for that reason.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed: the three answers a declaration can give, consumed by one match in the serve layer"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Declared {
    /// A drafter-specific marker: this snapshot is this kind and nothing else.
    Sidecar(DraftKind),
    /// A registered generative architecture, which a full draft model is.
    ///
    /// An inference from the registry, not a marker the snapshot carries, so
    /// an explicit `--draft-kind` outranks it: the registry is edited whenever
    /// a model is supported, for reasons that have nothing to do with drafting.
    FullModel,
    /// Nothing this crate can name. `--draft-kind` exists for this snapshot.
    Unknown,
}

impl Declared {
    /// The declaration behind an `architectures[0]` / `model_type` pair.
    pub fn from_snapshot(arch: &str, model_type: &str) -> Self {
        if arch.contains("Eagle3") {
            Declared::Sidecar(DraftKind::Eagle3)
        } else if arch.contains("DFlash") {
            Declared::Sidecar(DraftKind::DFlash)
        } else if model_type == "gemma4_assistant"
            || arch.contains("Gemma4Assistant")
            || model_type.contains("qwen3_5_mtp")
            || arch.contains("qwen3_5_mtp")
        {
            Declared::Sidecar(DraftKind::Mtp)
        } else if crate::arch::registry::is_generative_arch(arch) {
            Declared::FullModel
        } else {
            Declared::Unknown
        }
    }

    /// The kind this declaration selects on its own, if it selects one.
    pub fn kind(self) -> Option<DraftKind> {
        match self {
            Declared::Sidecar(kind) => Some(kind),
            Declared::FullModel => Some(DraftKind::TwoModel),
            Declared::Unknown => None,
        }
    }
}

impl std::fmt::Display for DraftKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DraftKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mtp" => Ok(DraftKind::Mtp),
            "dflash" => Ok(DraftKind::DFlash),
            "eagle3" => Ok(DraftKind::Eagle3),
            "two_model" => Ok(DraftKind::TwoModel),
            other => Err(format!(
                "unknown draft-kind '{other}'; valid values: mtp, dflash, eagle3, two_model"
            )),
        }
    }
}

#[cfg(test)]
#[path = "draft_kind_tests.rs"]
mod draft_kind_tests;
