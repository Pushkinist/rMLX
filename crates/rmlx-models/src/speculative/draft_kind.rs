//! `DraftKind` — which drafter a `--draft-model` snapshot is, and how that is
//! decided.

use std::str::FromStr;

use rmlx_loader::ModelConfig;

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
    reason = "closed dispatch enum — adding a kind requires a loader arm in SpeculativeGenerator, a spelling in as_str()/from_str(), and a declaration rule in from_declaration()"
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
    /// Canonical name used in log fields, `decode_config` and the CLI value.
    pub fn as_str(self) -> &'static str {
        match self {
            DraftKind::Mtp => "mtp",
            DraftKind::DFlash => "dflash",
            DraftKind::Eagle3 => "eagle3",
            DraftKind::TwoModel => "two_model",
        }
    }

    /// The kind a drafter snapshot declares in its `config.json`, or `None`
    /// when the declaration identifies no drafter.
    ///
    /// See [`Self::from_declaration`] for the rule.
    pub fn from_config(cfg: &ModelConfig) -> Option<Self> {
        let arch = cfg.architectures.first().map_or("", String::as_str);
        let model_type = cfg
            .extras
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        Self::from_declaration(arch, model_type)
    }

    /// The kind a snapshot's `architectures[0]` / `model_type` pair declares.
    ///
    /// Both fields are read because export tools set one or the other: the
    /// Gemma4 assistant carries `Gemma4AssistantForCausalLM` / `gemma4_assistant`,
    /// a Qwen3.5-family MTP sidecar carries `model_type = qwen3_5_mtp` and no
    /// `architectures` at all, DFlash carries `DFlash*DraftModel` over a plain
    /// `qwen3` model type, and EAGLE-3 carries `*Eagle3` over `llama`. A full
    /// model declares a registered architecture. A declaration matching none
    /// of those is `None`: the `--draft-kind` flag exists for that snapshot.
    pub fn from_declaration(arch: &str, model_type: &str) -> Option<Self> {
        if arch.contains("Eagle3") {
            Some(DraftKind::Eagle3)
        } else if arch.contains("DFlash") {
            Some(DraftKind::DFlash)
        } else if model_type == "gemma4_assistant"
            || arch.contains("Gemma4Assistant")
            || model_type.contains("qwen3_5_mtp")
            || arch.contains("qwen3_5_mtp")
        {
            Some(DraftKind::Mtp)
        } else if crate::arch::registry::is_arch_supported(arch) {
            Some(DraftKind::TwoModel)
        } else {
            None
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
