//! jina-embeddings-v4 multi-task LoRA: adapter parse + per-task injection.
//!
//! jina-v4 ships **one** PEFT adapter file carrying **three** task adapters
//! (`retrieval`, `text-matching`, `code`) in a single `safetensors`. Selecting
//! a task swaps the active set of [`LoraDelta`]s into every adapted decoder
//! [`Linear`] (the seam already lives in `model.rs`).
//!
//! # Verified key schema (enumerated from the real adapter header, NOT docs)
//!
//! File: `adapters/adapter_model.safetensors` (343.2 MB, 1518 BF16 tensors).
//!
//! - **Decoder LoRA** — 36 layers x 7 projections x 3 tasks x {A,B} = 1512:
//! ```text
//! base_model.model.model.language_model.layers.{0..35}.<proj>.lora_{A,B}.<task>.weight
//! <proj> in { self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj }
//! <task> in { retrieval, text-matching, code }
//! ```
//! Note the **double** `model.model.` + `.language_model.` segment — the
//! recon doc originally mis-stated this; always trust the header.
//! - **Projector LoRA** — 6 tensors, *different* prefix (no `.language_model.`,
//!   no `.layers.`, single `model.`):
//! ```text
//! base_model.model.multi_vector_projector.lora_{A,B}.<task>.weight
//! ```
//! The projector module itself is a later subtask — these tensors are
//! parsed and **retained** (`projector`) for that future wiring, NOT
//! stubbed or applied here.
//! - **NO** `visual.*` keys (config `exclude_modules=".*visual.*"`) and **NO**
//!   `single_vector_projector` keys (mean-pool, no learned head). Asserted.
//!
//! `r = 32`, `lora_alpha = 32` => `scaling = alpha / r = 1.0` for every task.
//! Every tensor is BF16; `LoraDelta` keeps the native bf16 dtype (plain-math
//! `Linear` only, no dequant — jina-v4 is unquantized).
//!
//! A-factor shape is `[r, in]`, B-factor `[out, r]` — exactly the layout
//! [`LoraDelta::apply`] expects (`out = base + scaling*(x@A^T)@B^T`). No
//! transpose at load time; the math path transposes on use.

use std::collections::HashMap;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::ShardHandle;
use rmlx_mlx::Array;
use tracing::{debug, info};

use super::model::{JinaV4Text, LoraDelta, MultiVectorProjector, ProjId};

/// The three jina-v4 runtime LoRA tasks. `retrieval` is jina's default.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — three jina-v4 LoRA tasks (Retrieval/TextMatching/Code); adding a task requires updating JinaV4Task::ALL, name(), from_name(), and adapter loading"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JinaV4Task {
    /// Asymmetric query/passage retrieval (jina default).
    Retrieval,
    /// Symmetric semantic text similarity.
    TextMatching,
    /// Code <-> natural-language retrieval.
    Code,
}

impl JinaV4Task {
    /// jina's default task when none is requested.
    pub const DEFAULT: JinaV4Task = JinaV4Task::Retrieval;

    /// Every task, in `config.json` `task_names` order.
    pub const ALL: [JinaV4Task; 3] = [
        JinaV4Task::Retrieval,
        JinaV4Task::TextMatching,
        JinaV4Task::Code,
    ];

    /// The adapter-key task segment (matches `config.json` `task_names`).
    pub fn name(self) -> &'static str {
        match self {
            JinaV4Task::Retrieval => "retrieval",
            JinaV4Task::TextMatching => "text-matching",
            JinaV4Task::Code => "code",
        }
    }

    /// Parse a task from its canonical name (case-sensitive, jina convention).
    pub fn from_name(s: &str) -> Result<JinaV4Task> {
        match s {
            "retrieval" => Ok(JinaV4Task::Retrieval),
            "text-matching" => Ok(JinaV4Task::TextMatching),
            "code" => Ok(JinaV4Task::Code),
            other => Err(Error::Config(format!(
                "jina-v4: unknown LoRA task {other:?} (expected one of \
                 retrieval | text-matching | code)"
            ))),
        }
    }
}

/// Parsed `adapters/adapter_config.json` (only the fields rMLX needs).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — three fields are the complete LoRA adapter config contract; adding a field requires updating from_json and all AdapterConfig callers"
)]
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// LoRA rank `r` (32 for jina-v4).
    pub r: usize,
    /// `lora_alpha` (32 for jina-v4).
    pub lora_alpha: usize,
    /// `exclude_modules` regex (jina: `.*visual.*` -> no vision LoRA).
    pub exclude_modules: Option<String>,
}

impl AdapterConfig {
    /// `lora_alpha / r` — 1.0 for jina-v4 (r = alpha = 32).
    pub fn scaling(&self) -> f32 {
        self.lora_alpha as f32 / self.r as f32
    }

    fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| Error::Config(format!("jina-v4: cannot read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "jina-v4: malformed adapter_config.json at {}: {e}",
                path.display()
            ))
        })?;
        let r = v.get("r").and_then(serde_json::Value::as_u64).unwrap_or(32) as usize;
        let lora_alpha = v
            .get("lora_alpha")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(32) as usize;
        let exclude_modules = v
            .get("exclude_modules")
            .and_then(|x| x.as_str())
            .map(str::to_owned);
        if r == 0 {
            return Err(Error::Config(
                "jina-v4: adapter_config.json has r = 0".to_owned(),
            ));
        }
        Ok(AdapterConfig {
            r,
            lora_alpha,
            exclude_modules,
        })
    }
}

// ---------------------------------------------------------------------------
// Adapter weight bundle (all tasks, parsed once)
// ---------------------------------------------------------------------------

/// One (A, B) LoRA factor pair, native bf16, as stored on disk.
///
/// `a`: `[r, in]`, `b`: `[out, r]` — the [`LoraDelta::apply`] convention. No
/// transpose performed at parse time (the forward path transposes on use).
struct FactorPair {
    a: Array,
    b: Array,
}

/// All three task adapters, parsed from the single adapter safetensors.
///
/// Decoder factors are keyed `(layer, proj, task)`. Projector factors are
/// retained separately, keyed by task, for the later projector subtask — they
/// are loaded but **not** applied here (no projector module yet).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed adapter store — private HashMap fields; public API is scaling(), apply_decoder(), etc.; adding a field requires updating load_from_dir"
)]
#[allow(missing_debug_implementations)]
pub struct JinaV4Adapters {
    cfg: AdapterConfig,
    /// `(layer, proj, task) -> (A, B)` for the 36x7x3 decoder cells.
    decoder: HashMap<(usize, ProjId, JinaV4Task), FactorPair>,
    /// `task -> (A, B)` for `multi_vector_projector` (future subtask only).
    projector: HashMap<JinaV4Task, FactorPair>,
}

impl JinaV4Adapters {
    /// `scaling = lora_alpha / r` (1.0 for jina-v4).
    pub fn scaling(&self) -> f32 {
        self.cfg.scaling()
    }

    /// Parsed adapter config.
    pub fn config(&self) -> &AdapterConfig {
        &self.cfg
    }

    /// Number of decoder `(layer, proj, task)` factor pairs parsed.
    pub fn decoder_pair_count(&self) -> usize {
        self.decoder.len()
    }

    /// Number of `multi_vector_projector` task factor pairs retained.
    pub fn projector_pair_count(&self) -> usize {
        self.projector.len()
    }

    /// Parse every task adapter from `<model_dir>/adapters/`.
    ///
    /// `num_layers` is the text tower's decoder depth (36); used to enumerate
    /// the exact expected key set and to **assert full coverage** — a missing
    /// (layer, proj, task) tensor is a hard error, not a silent skip.
    pub fn load(model_dir: &Path, num_layers: usize) -> Result<Self> {
        let adapters_dir = model_dir.join("adapters");
        let cfg = AdapterConfig::from_file(&adapters_dir.join("adapter_config.json"))?;

        // Single standalone adapter file — NOT in the model shard index, so
        // open it directly (mmap, zero-copy header parse).
        let handle = ShardHandle::open(&adapters_dir, "adapter_model.safetensors")?;
        let st = handle.safetensors()?;

        // Defensive: the config says vision is excluded — prove the file
        // actually honors it before we trust the rest of the schema.
        let visual_keys = st
            .names()
            .into_iter()
            .filter(|n| n.contains("visual"))
            .count();
        if visual_keys != 0 {
            return Err(Error::Loader(format!(
                "jina-v4: adapter unexpectedly contains {visual_keys} visual.* \
                 LoRA tensors (exclude_modules={:?} says there should be none)",
                cfg.exclude_modules
            )));
        }

        let load = |name: &str| -> Result<Array> {
            let t = st.tensor(name).map_err(|e| {
                Error::Loader(format!("jina-v4: adapter tensor '{name}' not found: {e}"))
            })?;
            let tv = rmlx_loader::TensorView {
                name,
                dtype: t.dtype(),
                shape: t.shape().to_vec(),
                bytes: t.data(),
            };
            Array::from_safetensor_view(&tv)
        };

        // ---- decoder factors: 36 x 7 x 3 x {A,B} -----------------------
        let mut decoder = HashMap::with_capacity(num_layers * ProjId::ALL.len() * 3);
        for layer in 0..num_layers {
            for proj in ProjId::ALL {
                let seg = proj.key_segment();
                for task in JinaV4Task::ALL {
                    let tname = task.name();
                    let base =
                        format!("base_model.model.model.language_model.layers.{layer}.{seg}");
                    let a = load(&format!("{base}.lora_A.{tname}.weight"))?;
                    let b = load(&format!("{base}.lora_B.{tname}.weight"))?;
                    decoder.insert((layer, proj, task), FactorPair { a, b });
                }
            }
        }

        // ---- projector factors: 3 tasks (retained for later subtask) ---
        let mut projector = HashMap::with_capacity(3);
        for task in JinaV4Task::ALL {
            let tname = task.name();
            let base = "base_model.model.multi_vector_projector";
            let a = load(&format!("{base}.lora_A.{tname}.weight"))?;
            let b = load(&format!("{base}.lora_B.{tname}.weight"))?;
            projector.insert(task, FactorPair { a, b });
        }

        info!(
            decoder_pairs = decoder.len(),
            projector_pairs = projector.len(),
            r = cfg.r,
            lora_alpha = cfg.lora_alpha,
            scaling = cfg.scaling(),
            "jina-v4: parsed multi-task LoRA adapters (3 tasks)"
        );
        Ok(JinaV4Adapters {
            cfg,
            decoder,
            projector,
        })
    }

    /// Install `task`'s full decoder LoRA set onto `text`'s [`Linear`]s.
    ///
    /// A clean replace of any previously-active task (each seam overwritten).
    /// The projector LoRA is intentionally NOT applied (no projector module
    /// until a later subtask). Factors are bf16 throughout; `scaling` is
    /// `lora_alpha / r` (1.0).
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn apply_task(&self, text: &mut JinaV4Text, task: JinaV4Task) -> Result<()> {
        let scaling = self.scaling();
        // Pre-validate: every cell this task needs must exist before we
        // mutate any seam (avoids a half-applied state on a missing key).
        for layer in 0..text.num_layers() {
            for proj in ProjId::ALL {
                if !self.decoder.contains_key(&(layer, proj, task)) {
                    return Err(Error::Loader(format!(
                        "jina-v4: missing LoRA delta for task {} (layer {layer}, {})",
                        task.name(),
                        proj.key_segment()
                    )));
                }
            }
        }
        text.install_task_loras(|layer, proj| {
            let fp = &self.decoder[&(layer, proj, task)];
            LoraDelta {
                a: fp
                    .a
                    .try_clone()
                    .expect("jina-v4: clone LoRA A (ref-counted, infallible)"),
                b: fp
                    .b
                    .try_clone()
                    .expect("jina-v4: clone LoRA B (ref-counted, infallible)"),
                scaling,
            }
        });
        debug!(task = task.name(), scaling, "jina-v4: task LoRA applied");
        Ok(())
    }

    /// Install `task`'s `multi_vector_projector` LoRA delta onto `projector`.
    ///
    /// Same clean-replace semantics as [`apply_task`]; called alongside it so
    /// the projector's adapter stays consistent with the active decoder task
    /// (jina applies the same `task_label` to projector + backbone — ref
    /// `modeling_jina_embeddings_v4.py:262`). The projector factors were parsed
    /// once at load (single `model.` prefix, no `.layers.`).
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    pub fn apply_projector(
        &self,
        projector: &mut MultiVectorProjector,
        task: JinaV4Task,
    ) -> Result<()> {
        let fp = self.projector.get(&task).ok_or_else(|| {
            Error::Loader(format!(
                "jina-v4: missing projector LoRA delta for task {}",
                task.name()
            ))
        })?;
        projector.set_lora(LoraDelta {
            a: fp
                .a
                .try_clone()
                .expect("jina-v4: clone projector LoRA A (ref-counted, infallible)"),
            b: fp
                .b
                .try_clone()
                .expect("jina-v4: clone projector LoRA B (ref-counted, infallible)"),
            scaling: self.scaling(),
        });
        debug!(task = task.name(), "jina-v4: projector LoRA applied");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests (gated on the on-disk adapter; skip-with-msg if absent)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "lora_tests.rs"]
mod tests;
