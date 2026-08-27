// LOC-exempt: closed-format KV block reader/writer with one codec arm per persisted layout.
// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Serialize / deserialize an N-layer KV-cache block to a **safetensors** file
//!.
//!
//! A "block" is the full KV state of a model at a point in a decode session:
//! one [`KvStorage`] per attention layer, plus (for hybrid linear-attention
//! archs such as Qwen3.5-MoE GDN) an optional [`LinearAttnCache`] per
//! linear-attention layer. This module writes that state to a single
//! safetensors file and reads it back, reconstructing the storages.
//!
//! # Why safetensors
//!
//! safetensors is already a workspace dependency (no new dep). Named tensors +
//! a JSON `__metadata__` header give us a self-describing, debuggable on-disk
//! format: every tensor is `l{idx}.<component>` and the header records the
//! `model_id`, `kv_quant`, `n_layers`, and per-layer geometry needed to rebuild
//! the storage. A reader verifies the header against the loaded model
//! (`model_id` + `kv_quant`) **before** hydrating — a mismatch is an `Err`, never
//! a silently-wrong cache.
//!
//! # Serialization strategy (general across all variants)
//!
//! Every [`KvStorage`] variant is reduced to a uniform set of **codes / scales
//! ( / rotations / biases)** tensors, which are exactly the buffers MLX stores:
//!
//! | Variant | Tensors written |
//! |-----------|--------------------------------------------------------------------|
//! | `K8V4` | `k.codes` `k.scales` `v.codes` `v.scales` |
//! | `K8V8` | `k.codes` `k.scales` `v.codes` `v.scales` (V is also q8_0) |
//! | `Planar` | + `v.rotations` on the V side |
//! | `None` | geometry only — bf16 K/V live on the parent `KvCache`, not storage |
//! | `Mixed` | `k.codes/scales/biases` `v.codes/scales/biases` (mx.quantize tuples)|
//! | `Paged` | gather to contiguous codes/scales(/rotations), + page geometry meta |
//! | LinearAttn| `conv_state` + `delta_state` recurrent tensors (whole, untruncated) |
//!
//! For the CPU scalar quant paths the per-step `Vec<TurboBlocks>` /
//! `Vec<PlanarBlocks>` are flattened then rebuilt on read — the round trip is
//! byte-exact on the codes. GDN state has **no sequence axis**: the whole
//! recurrent state is serialized, never truncated.
//!
//! This module is purely additive — it reads the existing `KvStorage` buffers
//! and never changes cache semantics.
//!
//! The writer is wired into the SSD-spill path via [`write_caches`]; the
//! reader/hydrate path is exercised by tests today and its prompt-cache
//! consumer lands in a follow-up. `#[allow(dead_code)]` still marks the
//! reader API that has no non-test caller yet (same idiom as `paged.rs`
//! page-recycling) — the writer side is now live.

#![allow(dead_code)]
#![allow(
    clippy::assigning_clones,
    clippy::items_after_statements,
    clippy::match_same_arms,
    clippy::needless_pass_by_value
)]
use std::collections::HashMap;
use std::mem::size_of;
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::planarquant::PlanarBlocks;
use rmlx_kv_quant::rotating::{RotatingStateSnapshot, RotatingTensorSnapshot};
use rmlx_kv_quant::turboquant::TurboBlocks;
use rmlx_mlx::{Array, Device, Dtype};
use safetensors::tensor::{Dtype as StDtype, Metadata, View};
use safetensors::SafeTensors;

use crate::traits::ExactReplayMetadata;
use rmlx_kv_quant::kvcache::KvCache;
use rmlx_kv_quant::linear_attn::LinearAttnCache;
use rmlx_kv_quant::mixed_quant::{MixedKvState, MixedTuple};
use rmlx_kv_quant::paged::{PagedKStorage, PagedPlanarVStorage, PagedVStorage};
use rmlx_kv_quant::storage::{
    IsoBlocks, KvStorage, QuantIsoK3, QuantIsoK4, QuantIsoV3, QuantIsoV4, QuantK, QuantKTurbo3,
    QuantKTurbo4, QuantPlanarK, QuantPlanarV, QuantRotorK3, QuantRotorK4, QuantRotorV3,
    QuantRotorV4, QuantV, RotorBlocks, RotorKBlocks, ISOV3_LAYOUT_TAG, ISOV4_LAYOUT_TAG,
    ISO_K_ONLY_3_LAYOUT_TAG, ISO_K_ONLY_4_LAYOUT_TAG, ISO_SYM_3_LAYOUT_TAG, ISO_SYM_4_LAYOUT_TAG,
    K8VTURBO2_TCQ_LAYOUT_TAG, K8VTURBO3_TCQ_LAYOUT_TAG, PLANARK4_LAYOUT_TAG, ROTORV3_LAYOUT_TAG,
    ROTORV4_LAYOUT_TAG, ROTOR_K_ASYM_3_LAYOUT_PREFIX, ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX,
    ROTOR_K_ASYM_4_LAYOUT_PREFIX, ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX, ROTOR_K_ONLY_3_LAYOUT_TAG,
    ROTOR_K_ONLY_3_QJL_LAYOUT_TAG, ROTOR_K_ONLY_4_LAYOUT_TAG, ROTOR_K_ONLY_4_QJL_LAYOUT_TAG,
    ROTOR_SYM_3_LAYOUT_TAG, ROTOR_SYM_3_QJL_LAYOUT_TAG, ROTOR_SYM_4_LAYOUT_TAG,
    ROTOR_SYM_4_QJL_LAYOUT_TAG, TURBOSYM3_LAYOUT_TAG, TURBOSYM4_LAYOUT_TAG,
};
use rmlx_kv_quant::KvQuant;

/// Per-layer off-storage bf16 K/V seed restored on hydrate. `Some` only for a
/// `KvQuant::None` layer whose live K/V lived on the parent `KvCache`
/// (`decode_fp16_{k,v}`); `None` for every quantised variant (their K/V is
/// inside the reconstructed `KvStorage`).
type NoneBf16Seed = Option<(Array, Array)>;

/// Result of [`KvBlockReader::hydrate`]: per-layer storages, per-layer
/// off-storage bf16 seeds, and the linear-attn recurrent caches.
type HydratedLayers = (Vec<KvStorage>, Vec<NoneBf16Seed>, Vec<LinearAttnCache>);

// ── Metadata keys ───────────────────────────────────────────────────────────

const META_MODEL_ID: &str = "model_id";
const META_KV_QUANT: &str = "kv_quant";
const META_N_LAYERS: &str = "n_layers";
const META_SEQ_LEN: &str = "seq_len";
const META_N_LINEAR: &str = "n_linear";
const META_FORMAT_VERSION: &str = "format_version";
const META_PROMPT_IDS_TENSOR: &str = "prompt_ids_tensor";
const META_EXACT_REPLAY_ID: &str = "exact_replay_id";
const META_EXACT_REPLAY_PIECE: &str = "exact_replay_piece";
const FORMAT_VERSION: &str = "2";

/// Full prompt identity persisted with a format-v2 block.
///
/// The block hash only identifies the last complete block. This value keeps
/// the non-block-aligned tail (and, for exact replay, the first decode token)
/// alongside the KV state so hydrate can verify the request before promotion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlockIdentity {
    pub prompt_ids: Vec<u32>,
    pub exact_replay: Option<ExactReplayMetadata>,
}

/// Per-layer geometry JSON key, `l{idx}.geom`.
fn geom_key(idx: usize) -> String {
    format!("l{idx}.geom")
}

fn offset_key(idx: usize) -> String {
    format!("l{idx}.offset")
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors raised while reading a KV block back from disk.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum BlockIoError {
    /// The file's `model_id` does not match the model being hydrated.
    #[error("KV block model_id mismatch: file has '{found}', expected '{expected}'")]
    ModelIdMismatch {
        /// model_id in the block file.
        found: String,
        /// model_id the current model expects.
        expected: String,
    },
    /// The file's `kv_quant` does not match the model's configured quant.
    #[error("KV block kv_quant mismatch: file has '{found}', expected '{expected}'")]
    KvQuantMismatch {
        /// KvQuant string in the block file.
        found: String,
        /// KvQuant string the current model expects.
        expected: String,
    },
    /// A required header metadata key is missing or malformed.
    #[error("KV block header: {0}")]
    Header(String),
    /// A referenced tensor was absent from the file.
    #[error("KV block: missing tensor '{0}'")]
    MissingTensor(String),
    /// A store was handed to serialization with its CPU blocks shorter than its
    /// accumulated `shape[2]` — persisting it would truncate the store.
    #[error("KV block: refusing to persist a truncated store: {0}")]
    TruncatedStore(String),
}

impl From<BlockIoError> for Error {
    fn from(e: BlockIoError) -> Self {
        Error::Mlx(e.to_string())
    }
}

// ── Owned tensor view for safetensors serialization ───────────────────────────

/// An owned (bytes + shape + dtype) tensor that implements safetensors `View`.
struct OwnedTensor {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: StDtype,
}

impl OwnedTensor {
    fn from_array(a: &Array) -> Result<Self> {
        // GPU arrays are pre-materialized on the inference thread before the
        // spill job is queued (see `KvCache::eval_for_spill`), so this eval is a
        // no-op for them — crucially it does NOT touch a Metal stream, which the
        // drain thread lacks. For CPU arrays (the block_io unit tests, which call
        // the writer directly without pre-eval) it forces materialization so
        // `to_bytes` reads a valid host pointer rather than segfaulting.
        a.eval()?;
        let bytes = a.to_bytes()?;
        let shape = a.shape().iter().map(|&d| d as usize).collect();
        let dtype = st_dtype(a.dtype());
        Ok(Self {
            bytes,
            shape,
            dtype,
        })
    }

    fn from_u8(v: &[u8]) -> Self {
        Self {
            bytes: v.to_vec(),
            shape: vec![v.len()],
            dtype: StDtype::U8,
        }
    }

    fn from_f32(v: &[f32]) -> Self {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Self {
            bytes,
            shape: vec![v.len()],
            dtype: StDtype::F32,
        }
    }

    fn from_u32(v: &[u32]) -> Self {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Self {
            bytes,
            shape: vec![v.len()],
            dtype: StDtype::U32,
        }
    }
}

impl View for &OwnedTensor {
    fn dtype(&self) -> StDtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

fn st_dtype(d: Dtype) -> StDtype {
    match d {
        Dtype::Bf16 => StDtype::BF16,
        Dtype::F16 => StDtype::F16,
        Dtype::F32 => StDtype::F32,
        Dtype::U8 => StDtype::U8,
        Dtype::U32 => StDtype::U32,
        Dtype::I32 => StDtype::I32,
    }
}

#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn rmlx_dtype(d: StDtype) -> Result<Dtype> {
    Ok(match d {
        StDtype::BF16 => Dtype::Bf16,
        StDtype::F16 => Dtype::F16,
        StDtype::F32 => Dtype::F32,
        StDtype::U8 => Dtype::U8,
        StDtype::U32 => Dtype::U32,
        StDtype::I32 => Dtype::I32,
        other => return Err(Error::Mlx(format!("KV block: unsupported dtype {other:?}"))),
    })
}

// ── Writer ─────────────────────────────────────────────────────────────────────

/// Writes an N-layer KV block to a safetensors file.
///
/// `model_id` is `<arch>/<snapshot_dir_name>` (taken as a constructor arg; the
/// caller derives it from `ModelConfig`). `kv_quant` is the configured cache
/// quant — both are written to the header and verified by [`KvBlockReader`].
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed writer — fields are private; public API is write_to_path, not struct literal construction"
)]
#[allow(missing_debug_implementations)]
pub struct KvBlockWriter<'a> {
    model_id: String,
    kv_quant: KvQuant,
    layers: &'a [KvStorage],
    /// Optional linear-attention recurrent state, one per linear-attn layer.
    /// Empty for non-hybrid archs.
    linear: &'a [LinearAttnCache],
}

impl<'a> KvBlockWriter<'a> {
    pub(super) fn new(
        model_id: impl Into<String>,
        kv_quant: KvQuant,
        layers: &'a [KvStorage],
        linear: &'a [LinearAttnCache],
    ) -> Self {
        Self {
            model_id: model_id.into(),
            kv_quant,
            layers,
            linear,
        }
    }

    /// Serialize the block to `path`.
    pub fn write(&self, path: &Path, device: Device) -> Result<()> {
        serialize_block(
            path,
            device,
            &self.model_id,
            self.kv_quant,
            self.layers,
            self.linear,
        )
    }
}

/// Serialize a KV block held as `&[KvCache]` (+ optional GDN linear state) to
/// `path`.
///
/// Prompt-cache entries (`Gemma4Entry`, `Qwen35MoeEntry`) hold `Vec<KvCache>`,
/// whereas [`KvBlockWriter`] is `KvStorage`-oriented. This bridge borrows each
/// cache's internal [`KvStorage`] (no tensor copy) and reuses the exact same
/// serialization path as [`KvBlockWriter::write`]. The Array→host
/// `eval()`/`to_bytes()` happens here, so the caller must invoke this off the
/// hot path (the spill drain thread).
///
/// # Contract: `kv_caches` MUST be layer-ordered
///
/// Element `i` of `kv_caches` MUST correspond to model layer `i` (0-based).
/// The on-disk `.kvb` format does not persist `layer_idx`; the hydrate path
/// ([`read_caches_inner`]) reconstructs it by position via `.enumerate()`.
/// Spilling out-of-order layers would silently scramble rotor3/rotor4 seeds
/// at hydrate. All current callers pass `Vec<KvCache>` indexed by layer, so
/// this is implicit but load-bearing.
pub fn write_caches(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    kv_caches: &[KvCache],
    lin_caches: &[LinearAttnCache],
) -> Result<()> {
    let layers: Vec<&KvStorage> = kv_caches.iter().map(KvCache::storage).collect();
    let none_bf16 = none_bf16_payloads(kv_caches)?;
    let rotating: Vec<_> = kv_caches
        .iter()
        .map(KvCache::rotating_snapshot)
        .collect::<Result<_>>()?;
    serialize_block_refs(
        path, device, model_id, kv_quant, &layers, &none_bf16, &rotating, lin_caches,
    )
}

/// Collect the live bf16 K/V mirror of every layer that spills geometry-only,
/// so the writer can persist the unquantised prefix that lives off the storage
/// buffer (on `KvCache::decode_fp16_{k,v}`).
///
/// Element `i` is `Some` only for a layer whose storage holds no packed payload
/// ([`KvStorage::geometry_only_max_seq`]) and that actually holds a filled bf16
/// pair. That covers `KvQuant::None`, whose K/V has always lived there, and the
/// bf16-mirror codecs, whose `exit_prefill` builds no packed store at all —
/// without this the tier would spill those layers as empty geometry and the
/// hydrate would come back holding nothing. Layers with a real packed payload,
/// and rotating (SWA) windows with no mirror to give, are `None` here and spill
/// through their own path.
///
/// # Why this does no tensor work
///
/// It runs on the spill **drain** thread, which has no Metal stream: every
/// array it hands on must already be materialised, in row-major order, at
/// exactly the length it should be persisted at. `exit_prefill` guarantees the
/// first two (it stores the seed through `Array::contiguous`). The third is
/// checked rather than fixed, and the check **fails the whole block** rather
/// than dropping the layer: the writer reads each layer's `seq_len` off its own
/// buffer, but `META_SEQ_LEN` is a single max across layers that hydrate then
/// applies to every reconstructed layer. Dropping one layer's payload would
/// therefore hand that layer back claiming the block's full offset while
/// holding nothing — a frozen cache with zero content and no error. Refusing
/// the block costs one re-prefill; refusing the layer costs correctness.
/// Compacting the mirror instead would need a device this thread does not have.
///
/// The comparison is `!=`, not `<`: a cache whose `offset` runs *past* its
/// mirror is equally inconsistent, and spilling the short buffer under the
/// longer claim is the same defect in the other direction.
fn none_bf16_payloads(kv_caches: &[KvCache]) -> Result<Vec<NoneBf16Seed>> {
    kv_caches
        .iter()
        .map(|c| {
            if c.storage().geometry_only_max_seq().is_none() {
                return Ok(None);
            }
            let Some((k, v)) = c.decode_fp16_kv() else {
                return Ok(None);
            };
            let buf_seq = k.shape().get(2).copied().unwrap_or(0);
            if c.offset() != buf_seq {
                return Err(Error::Mlx(format!(
                    "kv-spill: layer {} bf16 mirror is {buf_seq} tokens against an \
                     offset of {}; the block records one seq_len for every layer, so \
                     spilling this one would hydrate a layer holding nothing at the \
                     block's length. Refusing the block.",
                    c.layer_idx(),
                    c.offset(),
                )));
            }
            // Ref-count clone only (mlx-c is COW); host materialisation happens
            // later in `OwnedTensor::from_array`.
            Ok(Some((k.try_clone()?, v.try_clone()?)))
        })
        .collect()
}

/// Timed variant of [`write_caches`] for SSD-tier spill observability.
///
/// Returns `(bytes_written, dur_serialize_us, dur_write_us)`:
/// - `bytes_written` — final `.kvb` file size in bytes (via `fs::metadata`
///   after the write, 0 if the metadata read fails).
/// - `dur_serialize_us` — time spent building the in-memory tensor buffers
///   and safetensors layout (CPU eval / `to_bytes`), before the FS write.
/// - `dur_write_us` — time spent in `safetensors::serialize_to_file` (FS
///   write + implicit fsync on most platforms).
///
/// The split happens inside [`serialize_block_refs_timed`].
pub(crate) fn write_caches_timed(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    kv_caches: &[KvCache],
    lin_caches: &[LinearAttnCache],
) -> Result<(u64, u64, u64)> {
    let rotating: Vec<_> = kv_caches
        .iter()
        .map(KvCache::rotating_snapshot)
        .collect::<Result<_>>()?;
    write_caches_timed_with_snapshots(
        path, device, model_id, kv_quant, kv_caches, lin_caches, &rotating,
    )
}

/// Production spill entry point. Rotating snapshots must be captured on the
/// inference thread; the drain thread only serializes host-owned bytes.
pub(crate) fn write_caches_timed_with_snapshots(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    kv_caches: &[KvCache],
    lin_caches: &[LinearAttnCache],
    rotating: &[Option<RotatingStateSnapshot>],
) -> Result<(u64, u64, u64)> {
    let layers: Vec<&KvStorage> = kv_caches.iter().map(KvCache::storage).collect();
    let none_bf16 = none_bf16_payloads(kv_caches)?;
    serialize_block_refs_timed(
        path, device, model_id, kv_quant, &layers, &none_bf16, rotating, lin_caches, None,
    )
}

/// Timed production spill entry point carrying the full prompt identity.
///
/// `rotating` must have been captured on the inference owner thread. The
/// drain thread only serializes the supplied snapshots and identity data.
pub(crate) fn write_caches_timed_with_identity(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    kv_caches: &[KvCache],
    lin_caches: &[LinearAttnCache],
    rotating: &[Option<RotatingStateSnapshot>],
    prompt_ids: &[u32],
    exact_replay: Option<&ExactReplayMetadata>,
) -> Result<(u64, u64, u64)> {
    let layers: Vec<&KvStorage> = kv_caches.iter().map(KvCache::storage).collect();
    let none_bf16 = none_bf16_payloads(kv_caches)?;
    let identity = BlockIdentity {
        prompt_ids: prompt_ids.to_vec(),
        exact_replay: exact_replay.cloned(),
    };
    serialize_block_refs_timed(
        path,
        device,
        model_id,
        kv_quant,
        &layers,
        &none_bf16,
        rotating,
        lin_caches,
        Some(&identity),
    )
}

/// Reconstruct a KV block as `Vec<KvCache>` (+ optional GDN linear state) from
/// a `.kvb` file (SSD hydrate bridge — inverse of [`write_caches`]).
///
/// Opens `path`, verifies the header (`model_id` + `kv_quant`) against the
/// model being hydrated, reconstructs each layer's [`KvStorage`] via
/// [`KvBlockReader::hydrate`], and wraps each one as a decode-ready
/// [`KvCache`] with `offset` set to the recorded `seq_len`. Returns
/// `Err(BlockIoError::ModelIdMismatch | KvQuantMismatch)` on a metadata
/// mismatch and any deserialize error otherwise — the caller treats every
/// `Err` as a corrupt block (delete file + index row, fall through to
/// prefill). Host-materialization (`to_bytes`) happens here, so call this off
/// the hot path.
pub(crate) fn read_caches(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    policy: DispatchPolicy,
) -> Result<(Vec<KvCache>, Vec<LinearAttnCache>)> {
    let (kv_caches, lin_caches, _, _, _, _, _) =
        read_caches_inner(path, device, model_id, kv_quant, policy)?
            .ok_or_else(|| Error::Mlx(format!("KV block read: {} not found", path.display())))?;
    Ok((kv_caches, lin_caches))
}

/// Reconstructed caches plus the hydrate phase timings:
/// `(kv_caches, lin_caches, bytes_read, dur_read_us, dur_dequant_us, dur_finalize_us)`.
type TimedCaches = (Vec<KvCache>, Vec<LinearAttnCache>, u64, u64, u64, u64);

/// Timed hydrate result including the v2 prompt identity carried by the
/// record. [`read_caches_timed`] intentionally drops the identity for callers
/// that only need reconstructed caches.
pub(crate) type TimedCachesWithIdentity = (
    Vec<KvCache>,
    Vec<LinearAttnCache>,
    Option<BlockIdentity>,
    u64,
    u64,
    u64,
    u64,
);

/// Timed variant of [`read_caches`] for SSD-tier hydrate observability.
///
/// Returns `(kv_caches, lin_caches, bytes_read, dur_read_us, dur_dequant_us, dur_finalize_us)`:
/// - `bytes_read` — file size in bytes (via `fs::metadata` before the mmap
///   open; 0 if the metadata call fails).
/// - `dur_read_us` — time for `KvBlockReader::open` (mmap + safetensors header
///   parse).
/// - `dur_dequant_us` — time for `KvBlockReader::hydrate` (CPU-side storage
///   reconstruction / dequant).
/// - `dur_finalize_us` — time to wrap each reconstructed [`KvStorage`] into a
///   decode-ready [`KvCache`]; CPU-only struct construction, not a GPU upload.
///
/// `Ok(None)` means the file is no longer on disk — the routine outcome of an
/// LRU eviction landing between the index lookup and the read. The caller
/// distinguishes it from `Err` (a genuinely bad block) on the type.
pub(crate) fn read_caches_timed(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    policy: DispatchPolicy,
) -> Result<Option<TimedCaches>> {
    Ok(
        read_caches_inner(path, device, model_id, kv_quant, policy)?.map(
            |(kv_caches, lin_caches, _identity, bytes, read, dequant, finalize)| {
                (kv_caches, lin_caches, bytes, read, dequant, finalize)
            },
        ),
    )
}

/// Timed hydrate variant that also returns the persisted v2 prompt identity.
pub(crate) fn read_caches_timed_with_identity(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    policy: DispatchPolicy,
) -> Result<Option<TimedCachesWithIdentity>> {
    read_caches_inner(path, device, model_id, kv_quant, policy)
}

/// Shared core for [`read_caches`] and [`read_caches_timed`].
///
/// Returns `(kv_caches, lin_caches, bytes_read, dur_read_us, dur_dequant_us, dur_finalize_us)`,
/// or `Ok(None)` when the block file does not exist.
/// Mirrors the write-side `serialize_block_refs` pattern.
fn read_caches_inner(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    policy: DispatchPolicy,
) -> Result<Option<TimedCachesWithIdentity>> {
    use std::time::Instant;

    let bytes_read = std::fs::metadata(path).map_or(0, |m| m.len());

    let t_read = Instant::now();
    let Some(reader) = KvBlockReader::open_existing(path)? else {
        return Ok(None);
    };
    let dur_read_us = t_read.elapsed().as_micros() as u64;

    let t_dequant = Instant::now();
    let (storages, none_bf16, lin_caches) = reader.hydrate(model_id, kv_quant, device)?;
    let rotating = reader.rotating_snapshots()?;
    let identity = reader.block_identity()?;
    if rotating.iter().any(Option::is_some) && identity.is_none() {
        return Err(BlockIoError::Header(
            "format-v2 rotating payload is missing full prompt identity".into(),
        )
        .into());
    }
    let layer_offsets = reader.layer_offsets(storages.len())?;
    let dur_dequant_us = t_dequant.elapsed().as_micros() as u64;

    let t_finalize = Instant::now();
    // `layer_idx` is reconstructed positionally: assumes `kv_caches` was
    // layer-ordered at spill — see `write_caches` contract. A `None`-storage
    // layer that carried an off-storage bf16 prefix re-seeds the decode buffers
    // so an exact-hit replay reads the real K/V instead of zeros.
    if rotating.len() != storages.len() {
        return Err(BlockIoError::Header("rotating layer count mismatch".into()).into());
    }
    let kv_caches: Vec<KvCache> = storages
        .into_iter()
        .zip(none_bf16)
        .zip(rotating)
        .zip(layer_offsets)
        .enumerate()
        .map(
            |(layer_idx, (((s, bf16), rotating), layer_offset))| -> Result<KvCache> {
                let mut cache = match rotating {
                    Some(ref snapshot) => KvCache::with_quant_max_seq_window(
                        kv_quant,
                        snapshot.max_size,
                        Some(snapshot.max_size),
                    )
                    .with_layer_idx(layer_idx)
                    .with_dispatch_policy(policy),
                    None => KvCache::from_storage(s, kv_quant, layer_offset, layer_idx, policy),
                };
                if let Some(snapshot) = rotating.as_ref() {
                    cache.restore_rotating_snapshot(snapshot, device)?;
                }
                match bf16 {
                    Some((k, v)) => Ok(cache.with_decode_fp16_seed(k, v)),
                    None => Ok(cache),
                }
            },
        )
        .collect::<Result<Vec<_>>>()?;
    let dur_finalize_us = t_finalize.elapsed().as_micros() as u64;

    Ok(Some((
        kv_caches,
        lin_caches,
        identity,
        bytes_read,
        dur_read_us,
        dur_dequant_us,
        dur_finalize_us,
    )))
}

/// Shared serialization core over an owned-slice of storages.
fn serialize_block(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    layers: &[KvStorage],
    linear: &[LinearAttnCache],
) -> Result<()> {
    let refs: Vec<&KvStorage> = layers.iter().collect();
    // Storage-only callers (the `KvBlockWriter` struct path used by tests)
    // carry no off-storage bf16 — None layers serialize geometry-only.
    serialize_block_refs(path, device, model_id, kv_quant, &refs, &[], &[], linear)
}

/// Shared serialization core over a slice of storage references.
///
/// `none_bf16[i]`, when `Some`, holds the off-storage bf16 `(K, V)` of a
/// [`KvStorage::None`] layer (from `KvCache::decode_fp16_{k,v}`); an empty
/// slice means "no off-storage payload for any layer" (the struct-writer path).
fn serialize_block_refs(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    layers: &[&KvStorage],
    none_bf16: &[NoneBf16Seed],
    rotating: &[Option<RotatingStateSnapshot>],
    linear: &[LinearAttnCache],
) -> Result<()> {
    serialize_block_refs_timed(
        path, device, model_id, kv_quant, layers, none_bf16, rotating, linear, None,
    )
    .map(|_| ())
}

/// Timed serialization core — splits the work into a tensor-build phase
/// (CPU eval / `to_bytes`) and an FS-write phase (`serialize_to_file`).
///
/// Returns `(bytes_written, dur_serialize_us, dur_write_us)`.
fn serialize_block_refs_timed(
    path: &Path,
    device: Device,
    model_id: &str,
    kv_quant: KvQuant,
    layers: &[&KvStorage],
    none_bf16: &[NoneBf16Seed],
    rotating: &[Option<RotatingStateSnapshot>],
    linear: &[LinearAttnCache],
    identity: Option<&BlockIdentity>,
) -> Result<(u64, u64, u64)> {
    use std::time::Instant;

    // ── Phase 1: build in-memory tensor buffers + metadata (CPU eval) ────────
    let t_ser = Instant::now();
    let mut tensors: Vec<(String, OwnedTensor)> = Vec::new();
    let mut meta: HashMap<String, String> = HashMap::new();

    meta.insert(META_MODEL_ID.into(), model_id.to_string());
    meta.insert(META_KV_QUANT.into(), kv_quant.to_string());
    meta.insert(META_N_LAYERS.into(), layers.len().to_string());
    meta.insert(META_FORMAT_VERSION.into(), FORMAT_VERSION.into());

    if let Some(identity) = identity {
        if rotating.iter().any(Option::is_some) && identity.prompt_ids.is_empty() {
            return Err(BlockIoError::Header(
                "rotating format-v2 payload requires non-empty prompt identity".into(),
            )
            .into());
        }
        meta.insert(META_PROMPT_IDS_TENSOR.into(), "__prompt_ids".into());
        tensors.push((
            "__prompt_ids".into(),
            OwnedTensor::from_u32(&identity.prompt_ids),
        ));
        if let Some(replay) = &identity.exact_replay {
            meta.insert(META_EXACT_REPLAY_ID.into(), replay.id.to_string());
            meta.insert(META_EXACT_REPLAY_PIECE.into(), replay.piece.clone());
        }
    }

    let mut max_seq_len = 0i32;
    let mut rotating_seq_len: Option<i32> = None;
    for (idx, storage) in layers.iter().enumerate() {
        let bf16 = none_bf16.get(idx).and_then(Option::as_ref);
        let ring = rotating.get(idx).and_then(Option::as_ref);
        if let Some(ring) = ring.filter(|ring| ring.offset > 0) {
            if let Some(previous) = rotating_seq_len {
                if previous != ring.offset {
                    return Err(BlockIoError::Header(
                        "rotating layers have inconsistent absolute offsets".into(),
                    )
                    .into());
                }
            } else {
                rotating_seq_len = Some(ring.offset);
            }
        }
        let (geom, seq) = write_layer(idx, storage, bf16, ring, device, &mut tensors)?;
        meta.insert(geom_key(idx), geom);
        meta.insert(offset_key(idx), seq.to_string());
        max_seq_len = max_seq_len.max(seq);
    }
    if let Some(ring_offset) = rotating_seq_len {
        if max_seq_len != ring_offset {
            return Err(BlockIoError::Header(
                "rotating snapshot offset does not match block seq_len".into(),
            )
            .into());
        }
    }
    meta.insert(META_SEQ_LEN.into(), max_seq_len.to_string());

    // Linear-attn recurrent state (whole, untruncated — GDN has no seq axis).
    meta.insert(META_N_LINEAR.into(), linear.len().to_string());
    for (idx, lac) in linear.iter().enumerate() {
        if let Some(conv) = &lac.conv_state {
            tensors.push((
                format!("lin{idx}.conv_state"),
                OwnedTensor::from_array(conv)?,
            ));
        }
        if let Some(delta) = &lac.delta_state {
            tensors.push((
                format!("lin{idx}.delta_state"),
                OwnedTensor::from_array(delta)?,
            ));
        }
    }
    let dur_serialize_us = t_ser.elapsed().as_micros() as u64;

    // ── Phase 2: write to disk ────────────────────────────────────────────────
    let t_write = Instant::now();
    let refs: Vec<(String, &OwnedTensor)> = tensors.iter().map(|(n, t)| (n.clone(), t)).collect();
    safetensors::serialize_to_file(refs, Some(meta), path)
        .map_err(|e| Error::Mlx(format!("KV block serialize: {e}")))?;
    let dur_write_us = t_write.elapsed().as_micros() as u64;

    let bytes_written = std::fs::metadata(path).map_or(0, |m| m.len());
    Ok((bytes_written, dur_serialize_us, dur_write_us))
}

/// Write one layer's tensors. Returns `(geometry-json, seq_len)`.
#[allow(
    clippy::too_many_lines,
    reason = "single match over the closed KvStorage enum with one writer arm per codec; splitting per-variant arms into helpers would add indirection without reducing local complexity"
)]
fn write_layer(
    idx: usize,
    storage: &KvStorage,
    none_bf16: Option<&(Array, Array)>,
    rotating: Option<&RotatingStateSnapshot>,
    device: Device,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<(String, i32)> {
    if let Some(ring) = rotating {
        if ring.offset < 0 || ring.max_size <= 0 || ring.valid_len != ring.offset.min(ring.max_size)
        {
            return Err(
                BlockIoError::Header(format!("invalid rotating snapshot for layer {idx}")).into(),
            );
        }
        let Some(keys) = ring.keys.as_ref() else {
            if ring.values.is_some() {
                return Err(
                    BlockIoError::Header(format!("rotating layer {idx} has V without K")).into(),
                );
            }
            return Ok((
                format!(
                    "{{\"tag\":\"rotating\",\"max_seq\":{},\"offset\":{},\"keep\":{},\"idx\":{},\"valid_len\":{},\"dtype\":\"bf16\"}}",
                    ring.max_size, ring.offset, ring.keep, ring.idx, ring.valid_len
                ),
                ring.offset,
            ));
        };
        let Some(values) = ring.values.as_ref() else {
            return Err(
                BlockIoError::Header(format!("rotating layer {idx} has K without V")).into(),
            );
        };
        out.push((
            format!("l{idx}.rotating.k"),
            OwnedTensor::from_u8(&keys.bytes),
        ));
        out.push((
            format!("l{idx}.rotating.v"),
            OwnedTensor::from_u8(&values.bytes),
        ));
        let dtype = |d: Dtype| match d {
            Dtype::Bf16 => "bf16",
            Dtype::F16 => "f16",
            Dtype::F32 => "f32",
            Dtype::U8 => "u8",
            Dtype::U32 => "u32",
            Dtype::I32 => "i32",
        };
        return Ok((
            format!(
                "{{\"tag\":\"rotating\",\"max_seq\":{},\"offset\":{},\"keep\":{},\"idx\":{},\"valid_len\":{},\"k_shape\":[{}],\"v_shape\":[{}],\"dtype\":\"{}\"}}",
                ring.max_size,
                ring.offset,
                ring.keep,
                ring.idx,
                ring.valid_len,
                csv(&keys.shape),
                csv(&values.shape),
                dtype(keys.dtype)
            ),
            ring.offset,
        ));
    }
    // A layer with no packed payload has only its geometry to persist. Three
    // cases arrive here and the treatment is the same for all of them: a
    // rotating SWA layer (its KV lives in the bf16 ring, which is not
    // serialisable), a codec whose decode reads only the bf16 mirror so
    // `exit_prefill` built no store, and `KvStorage::None`, which never had
    // one. Stamping a codec geometry with no codes/scales tensors behind it
    // would make the reader fail on `missing tensor 'lN.k.codes'`.
    //
    // When the cache handed us its live bf16 mirror, that mirror IS the layer's
    // KV, so persist it under `l{idx}.{k,v}.bf16` and tag the layer
    // "none_bf16"; hydrate re-seeds the decode buffers from it and the
    // reconstructed cache decodes off exactly the bytes the spilling one did.
    // With no mirror (a rotating window, an unfilled cache) the layer falls
    // back to geometry-only "none" and the window is re-established on reuse.
    if let Some(max_seq) = storage.geometry_only_max_seq() {
        let Some((k, v)) = none_bf16 else {
            return Ok((format!("{{\"tag\":\"none\",\"max_seq\":{max_seq}}}"), 0));
        };
        // `seq` is the buffer's own length, and the block records one `seq_len`
        // for every layer — so the mirror handed over must already equal the
        // cache's `offset`. `none_bf16_payloads` enforces that by REFUSING the
        // block when it does not (it cannot compact: no device on this thread).
        // A decode-expanded buffer grown to the `max_seq` ceiling therefore
        // never reaches here; if it did, its zeroed tail would be persisted as
        // live KV.
        let seq = k.shape().get(2).copied().unwrap_or(0);
        out.push((format!("l{idx}.k.bf16"), OwnedTensor::from_array(k)?));
        out.push((format!("l{idx}.v.bf16"), OwnedTensor::from_array(v)?));
        return Ok((
            format!("{{\"tag\":\"none_bf16\",\"max_seq\":{max_seq}}}"),
            seq,
        ));
    }
    match storage {
        KvStorage::K8V4 { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            Ok((geom_kv("k8v4", *max_seq, k_shape(k.as_ref())), seq))
        }
        KvStorage::K8V8 { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_k(idx, "v", v.as_ref(), device, out)?;
            Ok((geom_kv("k8v8", *max_seq, k_shape(k.as_ref())), seq))
        }
        KvStorage::Planar {
            k,
            v,
            max_seq,
            bits,
        } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_planar_v(idx, v.as_ref(), device, out)?;
            // Tag encodes bit-width so the read side knows which codebook to
            // use (3-bit vs 4-bit).
            let tag = if *bits == 3 { "planar3" } else { "planar" };
            Ok((geom_kv(tag, *max_seq, k_shape(k.as_ref())), seq))
        }
        // `KvStorage::None` never reaches here — it has no packed payload and
        // is served by the geometry-only branch above.
        KvStorage::None { max_seq } => {
            Ok((format!("{{\"tag\":\"none\",\"max_seq\":{max_seq}}}"), 0))
        }
        KvStorage::Mixed { state, max_seq } => write_mixed(idx, state, *max_seq, out),
        KvStorage::Paged {
            quant,
            k,
            v_k8,
            v_planar,
            max_seq,
        } => write_paged(
            idx,
            *quant,
            k.as_ref(),
            v_k8.as_deref(),
            v_planar.as_deref(),
            *max_seq,
            device,
            out,
        ),
        // K8VTurbo3 — same layout as K8V4 but tagged "k8vturbo3".
        // V uses QuantV bits=3; codes/scales packing is the same format.
        KvStorage::K8VTurbo3 { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            Ok((geom_kv("k8vturbo3", *max_seq, k_shape(k.as_ref())), seq))
        }
        // K8VTurbo3Tcq — byte-for-byte identical pack as K8VTurbo3.
        // The Viterbi assignment is encode-side only; the codes stream and the
        // decoder are shared. Tagged separately so a hydrate cannot silently
        // demote a TCQ payload to plain turbo3 on the next decode-step encode.
        KvStorage::K8VTurbo3Tcq { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            Ok((
                geom_kv(K8VTURBO3_TCQ_LAYOUT_TAG, *max_seq, k_shape(k.as_ref())),
                seq,
            ))
        }
        // K8VTurbo2Tcq — byte-for-byte identical pack as K8VTurbo2
        // (2-bit LSB-first, 16 values per u32). Tagged separately via
        // K8VTURBO2_TCQ_LAYOUT_TAG to prevent silent demotion to nearest-centroid
        // on cross-restart hydrate.
        KvStorage::K8VTurbo2Tcq { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            Ok((
                geom_kv(K8VTURBO2_TCQ_LAYOUT_TAG, *max_seq, k_shape(k.as_ref())),
                seq,
            ))
        }
        // TurboSym3 — symmetric WHT-3 K + turbo3 V.
        // K is `QuantKTurbo3` (3-bit codes, same GPU pack as V-side turbo3).
        // Layout tag: TURBOSYM3_LAYOUT_TAG = "tsym3_wht_3_3".
        KvStorage::TurboSym3 { k, v, max_seq } => {
            let seq = write_quant_k_turbo3(idx, k.as_ref(), device, out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            Ok((
                geom_kv(TURBOSYM3_LAYOUT_TAG, *max_seq, k_turbo3_shape(k.as_ref())),
                seq,
            ))
        }
        // TurboSym4 — symmetric WHT-4 K + tq4 V.
        // Both axes use TurboQuant 4-bit; K is `QuantKTurbo4` (not `QuantK`).
        // Geometry tag is `TURBOSYM4_LAYOUT_TAG` = "tsym4_wht_4_4".
        KvStorage::TurboSym4 { k, v, max_seq } => {
            let seq = write_quant_k_turbo4(idx, k.as_ref(), device, out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            Ok((
                geom_kv(TURBOSYM4_LAYOUT_TAG, *max_seq, k_turbo4_shape(k.as_ref())),
                seq,
            ))
        }
        // PlanarK — K-only payload (codes/scales/rotations); V is bf16 and
        // lives on the parent KvCache, NOT in KvStorage. Tag = PLANARK4_LAYOUT_TAG.
        KvStorage::PlanarK { k, max_seq } => {
            let seq = write_quant_planar_k_side(idx, k.as_ref(), device, out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            Ok((geom_kv(PLANARK4_LAYOUT_TAG, *max_seq, shape), seq))
        }
        // K8VTurbo2 — same layout as K8V4 but tagged "k8vturbo2".
        // V uses QuantV bits=2; codes/scales packing is the same format.
        KvStorage::K8VTurbo2 { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            Ok((geom_kv("k8vturbo2", *max_seq, k_shape(k.as_ref())), seq))
        }
        // IsoV3 SSD spill — K side uses QuantK (q8_0) writer; V side serializes
        // the four IsoBlocks buffers (codes_packed, scales, quaternions, norms)
        // flat into safetensors tensors.
        KvStorage::IsoV3 { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_iso_v3(idx, v.as_ref(), out)?;
            Ok((
                geom_kv(ISOV3_LAYOUT_TAG, *max_seq, k_shape(k.as_ref())),
                seq,
            ))
        }
        // IsoV4 SSD spill — K side identical to IsoV3 (q8_0); V side uses the
        // same flat IsoBlocks layout. Differentiated only by the geometry tag
        // (`ISOV4_LAYOUT_TAG`, currently "iso_v_4_v2" — bumped when the GPU
        // append's byte orientation was fixed) so the reader picks the 4-bit codec /
        // pack on hydrate.
        KvStorage::IsoV4 { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_iso_v4(idx, v.as_ref(), out)?;
            Ok((
                geom_kv(ISOV4_LAYOUT_TAG, *max_seq, k_shape(k.as_ref())),
                seq,
            ))
        }
        // RotorV3 SSD spill — K is q8_0; V uses four flat buffers
        // (codes_packed, scales, norms, rotors). The static rotor table is
        // persisted so cross-restart identity is preserved regardless of
        // any seed-source drift.
        KvStorage::RotorV3 { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_rotor_v3(idx, v.as_ref(), out)?;
            Ok((
                geom_kv(ROTORV3_LAYOUT_TAG, *max_seq, k_shape(k.as_ref())),
                seq,
            ))
        }
        // RotorV4 SSD spill — K is q8_0; V uses four flat buffers identical in
        // structure to RotorV3 (codes_packed, scales, norms, rotors) but with
        // 4-bit codes (1 u32 per group of 8 multivector components).
        KvStorage::RotorV4 { k, v, max_seq } => {
            let seq = write_quant_k(idx, "k", k.as_ref(), device, out)?;
            write_quant_rotor_v4(idx, v.as_ref(), out)?;
            Ok((
                geom_kv(ROTORV4_LAYOUT_TAG, *max_seq, k_shape(k.as_ref())),
                seq,
            ))
        }
        // IsoSym3 — both K and V are IsoBlocks (codes_packed / scales /
        // quaternions / norms each); K under `l{idx}.k.*` and V under
        // `l{idx}.v.*`. Layout tag: ISO_SYM_3_LAYOUT_TAG.
        KvStorage::IsoSym3 { k, v, max_seq } => {
            write_quant_iso_k3(idx, k.as_ref(), out)?;
            write_quant_iso_v3(idx, v.as_ref(), out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            Ok((geom_kv(ISO_SYM_3_LAYOUT_TAG, *max_seq, shape), seq))
        }
        // IsoSym4 — same payload layout as IsoSym3 with 4-bit codes.
        KvStorage::IsoSym4 { k, v, max_seq } => {
            write_quant_iso_k4(idx, k.as_ref(), out)?;
            write_quant_iso_v4(idx, v.as_ref(), out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            Ok((geom_kv(ISO_SYM_4_LAYOUT_TAG, *max_seq, shape), seq))
        }
        // IsoKOnly3 — K-only payload (codes_packed/scales/quaternions/norms
        // under `l{idx}.k.*`); V is bf16 and lives on the parent KvCache.
        // Layout tag: ISO_K_ONLY_3_LAYOUT_TAG.
        KvStorage::IsoKOnly3 { k, max_seq } => {
            write_quant_iso_k3(idx, k.as_ref(), out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            Ok((geom_kv(ISO_K_ONLY_3_LAYOUT_TAG, *max_seq, shape), seq))
        }
        // IsoKOnly4 — same shape as IsoKOnly3 with 4-bit codes.
        KvStorage::IsoKOnly4 { k, max_seq } => {
            write_quant_iso_k4(idx, k.as_ref(), out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            Ok((geom_kv(ISO_K_ONLY_4_LAYOUT_TAG, *max_seq, shape), seq))
        }
        // RotorSym3 — K is rotor3 K (codes_packed/scales/norms + optional
        // qjl_codes/qjl_norms/qjl_s under l{idx}.k.*); V is rotor3 V
        // (codes_packed/scales/norms/rotors under l{idx}.v.*). The static K
        // rotor table also lives at l{idx}.k.rotors so hydrate is independent
        // of the global seed.
        KvStorage::RotorSym3 { k, v, max_seq } => {
            write_quant_rotor_k3(idx, k.as_ref(), out)?;
            write_quant_rotor_v3(idx, v.as_ref(), out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            let tag = if k.as_ref().is_some_and(QuantRotorK3::use_qjl) {
                ROTOR_SYM_3_QJL_LAYOUT_TAG
            } else {
                ROTOR_SYM_3_LAYOUT_TAG
            };
            Ok((geom_kv(tag, *max_seq, shape), seq))
        }
        // RotorSym4 — same shape as RotorSym3 with 4-bit codes.
        KvStorage::RotorSym4 { k, v, max_seq } => {
            write_quant_rotor_k4(idx, k.as_ref(), out)?;
            write_quant_rotor_v4(idx, v.as_ref(), out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            let tag = if k.as_ref().is_some_and(QuantRotorK4::use_qjl) {
                ROTOR_SYM_4_QJL_LAYOUT_TAG
            } else {
                ROTOR_SYM_4_LAYOUT_TAG
            };
            Ok((geom_kv(tag, *max_seq, shape), seq))
        }
        // RotorKOnly3 — K-only payload; V is bf16 off-storage.
        KvStorage::RotorKOnly3 { k, max_seq } => {
            write_quant_rotor_k3(idx, k.as_ref(), out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            let tag = if k.as_ref().is_some_and(QuantRotorK3::use_qjl) {
                ROTOR_K_ONLY_3_QJL_LAYOUT_TAG
            } else {
                ROTOR_K_ONLY_3_LAYOUT_TAG
            };
            Ok((geom_kv(tag, *max_seq, shape), seq))
        }
        // RotorKOnly4 — same shape as RotorKOnly3 with 4-bit codes.
        KvStorage::RotorKOnly4 { k, max_seq } => {
            write_quant_rotor_k4(idx, k.as_ref(), out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            let tag = if k.as_ref().is_some_and(QuantRotorK4::use_qjl) {
                ROTOR_K_ONLY_4_QJL_LAYOUT_TAG
            } else {
                ROTOR_K_ONLY_4_LAYOUT_TAG
            };
            Ok((geom_kv(tag, *max_seq, shape), seq))
        }
        // RotorKAsym3 — K is rotor3 K (codes_packed/scales/norms + optional
        // qjl_codes/qjl_norms/qjl_s under l{idx}.k.*); V is affine QuantV
        // (`l{idx}.v.codes`, `l{idx}.v.scales`, `l{idx}.v.biases`) matching
        // K8V4 V-side layout.
        KvStorage::RotorKAsym3 {
            k,
            v,
            max_seq,
            v_bits,
            v_group_size,
        } => {
            write_quant_rotor_k3(idx, k.as_ref(), out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            let prefix = if k.as_ref().is_some_and(QuantRotorK3::use_qjl) {
                ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX
            } else {
                ROTOR_K_ASYM_3_LAYOUT_PREFIX
            };
            let tag = format!("{prefix}_v{v_bits}_g{v_group_size}");
            Ok((geom_kv(&tag, *max_seq, shape), seq))
        }
        // RotorKAsym4 — same shape with rotor4 K + affine V.
        KvStorage::RotorKAsym4 {
            k,
            v,
            max_seq,
            v_bits,
            v_group_size,
        } => {
            write_quant_rotor_k4(idx, k.as_ref(), out)?;
            write_quant_v(idx, v.as_ref(), device, out)?;
            let shape = k.as_ref().map(|q| q.shape.clone()).unwrap_or_default();
            let seq = shape.get(2).copied().unwrap_or(0);
            let prefix = if k.as_ref().is_some_and(QuantRotorK4::use_qjl) {
                ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX
            } else {
                ROTOR_K_ASYM_4_LAYOUT_PREFIX
            };
            let tag = format!("{prefix}_v{v_bits}_g{v_group_size}");
            Ok((geom_kv(&tag, *max_seq, shape), seq))
        }
    }
}

/// Capture shape of a `QuantKTurbo3` for geometry serialization.
fn k_turbo3_shape(k: Option<&QuantKTurbo3>) -> Vec<i32> {
    k.map(|q| q.shape.clone()).unwrap_or_default()
}

/// Write a `QuantKTurbo3` (TurboQuant 3-bit) on the K side. Returns seq length.
///
/// Mirrors `write_quant_k_turbo4` exactly — identical codes/scales layout;
/// only the bit-width differs. Trims GPU buffers to the filled prefix (C3 trim).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn write_quant_k_turbo3(
    idx: usize,
    k: Option<&QuantKTurbo3>,
    device: Device,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<i32> {
    let Some(qk) = k else { return Ok(0) };
    if let (Some(codes), Some(scales)) = (&qk.gpu_codes_buf, &qk.gpu_scales_buf) {
        let prev_seq = qk.shape[2];
        let filled_codes = prev_seq * qk.gpu_words_per_step;
        let filled_scales = prev_seq * qk.gpu_scales_per_step;
        let codes_trimmed = if filled_codes < codes.shape()[0] {
            codes.slice(&[0], &[filled_codes], &[1], device)?
        } else {
            codes.try_clone()?
        };
        let scales_trimmed = if filled_scales < scales.shape()[0] {
            scales.slice(&[0], &[filled_scales], &[1], device)?
        } else {
            scales.try_clone()?
        };
        out.push((
            format!("l{idx}.k.codes"),
            OwnedTensor::from_array(&codes_trimmed)?,
        ));
        out.push((
            format!("l{idx}.k.scales"),
            OwnedTensor::from_array(&scales_trimmed)?,
        ));
    } else {
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for b in &qk.blocks {
            codes.extend_from_slice(&b.codes);
            scales.extend_from_slice(&b.scales);
        }
        out.push((format!("l{idx}.k.codes"), OwnedTensor::from_u8(&codes)));
        out.push((format!("l{idx}.k.scales"), OwnedTensor::from_f32(&scales)));
    }
    Ok(qk.shape[2])
}

/// Capture shape of a `QuantKTurbo4` for geometry serialization.
fn k_turbo4_shape(k: Option<&QuantKTurbo4>) -> Vec<i32> {
    k.map(|q| q.shape.clone()).unwrap_or_default()
}

/// Write a `QuantKTurbo4` (TurboQuant 4-bit) on the K side. Returns seq length.
///
/// Mirrors `write_quant_v` exactly — identical layout (`gpu_codes_buf` u32 +
/// `gpu_scales_buf` f32 + CPU `blocks: Vec<TurboBlocks>`). Trims GPU buffers
/// to the filled prefix before serialising (C3 trim).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn write_quant_k_turbo4(
    idx: usize,
    k: Option<&QuantKTurbo4>,
    device: Device,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<i32> {
    let Some(qk) = k else { return Ok(0) };
    if let (Some(codes), Some(scales)) = (&qk.gpu_codes_buf, &qk.gpu_scales_buf) {
        let prev_seq = qk.shape[2];
        let filled_codes = prev_seq * qk.gpu_words_per_step;
        let filled_scales = prev_seq * qk.gpu_scales_per_step;
        let codes_trimmed = if filled_codes < codes.shape()[0] {
            codes.slice(&[0], &[filled_codes], &[1], device)?
        } else {
            codes.try_clone()?
        };
        let scales_trimmed = if filled_scales < scales.shape()[0] {
            scales.slice(&[0], &[filled_scales], &[1], device)?
        } else {
            scales.try_clone()?
        };
        out.push((
            format!("l{idx}.k.codes"),
            OwnedTensor::from_array(&codes_trimmed)?,
        ));
        out.push((
            format!("l{idx}.k.scales"),
            OwnedTensor::from_array(&scales_trimmed)?,
        ));
    } else {
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for b in &qk.blocks {
            codes.extend_from_slice(&b.codes);
            scales.extend_from_slice(&b.scales);
        }
        out.push((format!("l{idx}.k.codes"), OwnedTensor::from_u8(&codes)));
        out.push((format!("l{idx}.k.scales"), OwnedTensor::from_f32(&scales)));
    }
    Ok(qk.shape[2])
}

// ── QuantK / QuantV / QuantPlanarV (CPU-Vec or GPU-Array) ─────────────────────

fn k_shape(k: Option<&QuantK>) -> Vec<i32> {
    k.map(|q| q.shape.clone()).unwrap_or_default()
}

/// Geometry JSON for the affine K8* / Planar variants.
fn geom_kv(tag: &str, max_seq: i32, shape: Vec<i32>) -> String {
    format!(
        "{{\"tag\":\"{tag}\",\"max_seq\":{max_seq},\"shape\":[{}]}}",
        csv(&shape)
    )
}

fn csv(shape: &[i32]) -> String {
    shape
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// Write a `QuantK` (q8_0) on the `side` ("k" or "v"). Returns the seq length.
///
/// C3 fix: GPU buffers are allocated in paged increments (KV_PAGE_SIZE
/// multiples) that may exceed the filled prefix. Trim to
/// `prev_seq * words_per_step` before serialising so the on-disk payload
/// matches the logical sequence length, preventing OOB slice_update on hydrate.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn write_quant_k(
    idx: usize,
    side: &str,
    k: Option<&QuantK>,
    device: Device,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<i32> {
    let Some(qk) = k else { return Ok(0) };
    if let (Some(codes), Some(scales)) = (&qk.gpu_codes_buf, &qk.gpu_scales_buf) {
        let prev_seq = qk.shape[2];
        let filled_codes = prev_seq * qk.gpu_words_per_step;
        let filled_scales = prev_seq * qk.gpu_scales_per_step;
        // GPU buffer is 1D; shape()[0] == total allocated length.
        let codes_trimmed = if filled_codes < codes.shape()[0] {
            codes.slice(&[0], &[filled_codes], &[1], device)?
        } else {
            codes.try_clone()?
        };
        let scales_trimmed = if filled_scales < scales.shape()[0] {
            scales.slice(&[0], &[filled_scales], &[1], device)?
        } else {
            scales.try_clone()?
        };
        out.push((
            format!("l{idx}.{side}.codes"),
            OwnedTensor::from_array(&codes_trimmed)?,
        ));
        out.push((
            format!("l{idx}.{side}.scales"),
            OwnedTensor::from_array(&scales_trimmed)?,
        ));
    } else {
        out.push((
            format!("l{idx}.{side}.codes"),
            OwnedTensor::from_u8(&qk.codes),
        ));
        out.push((
            format!("l{idx}.{side}.scales"),
            OwnedTensor::from_f32(&qk.scales),
        ));
    }
    Ok(qk.shape[2])
}

/// C3 fix: trim GPU buffers to filled prefix before serialising.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn write_quant_v(
    idx: usize,
    v: Option<&QuantV>,
    device: Device,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qv) = v else { return Ok(()) };
    if let (Some(codes), Some(scales)) = (&qv.gpu_codes_buf, &qv.gpu_scales_buf) {
        let prev_seq = qv.shape[2];
        let filled_codes = prev_seq * qv.gpu_words_per_step;
        let filled_scales = prev_seq * qv.gpu_scales_per_step;
        let codes_trimmed = if filled_codes < codes.shape()[0] {
            codes.slice(&[0], &[filled_codes], &[1], device)?
        } else {
            codes.try_clone()?
        };
        let scales_trimmed = if filled_scales < scales.shape()[0] {
            scales.slice(&[0], &[filled_scales], &[1], device)?
        } else {
            scales.try_clone()?
        };
        out.push((
            format!("l{idx}.v.codes"),
            OwnedTensor::from_array(&codes_trimmed)?,
        ));
        out.push((
            format!("l{idx}.v.scales"),
            OwnedTensor::from_array(&scales_trimmed)?,
        ));
    } else {
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for b in &qv.blocks {
            codes.extend_from_slice(&b.codes);
            scales.extend_from_slice(&b.scales);
        }
        out.push((format!("l{idx}.v.codes"), OwnedTensor::from_u8(&codes)));
        out.push((format!("l{idx}.v.scales"), OwnedTensor::from_f32(&scales)));
    }
    Ok(())
}

/// C3 fix: trim GPU buffers (codes, scales, rotations) to filled prefix.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn write_quant_planar_v(
    idx: usize,
    v: Option<&QuantPlanarV>,
    device: Device,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qv) = v else { return Ok(()) };
    if let (Some(codes), Some(scales), Some(rot)) =
        (&qv.gpu_codes_buf, &qv.gpu_scales_buf, &qv.gpu_rotations_buf)
    {
        let prev_seq = qv.shape[2];
        let filled_codes = prev_seq * qv.gpu_codes_words_per_step;
        let filled_scales = prev_seq * qv.gpu_scales_per_step;
        let filled_rot = prev_seq * qv.gpu_rotations_words_per_step;
        let codes_trimmed = if filled_codes < codes.shape()[0] {
            codes.slice(&[0], &[filled_codes], &[1], device)?
        } else {
            codes.try_clone()?
        };
        let scales_trimmed = if filled_scales < scales.shape()[0] {
            scales.slice(&[0], &[filled_scales], &[1], device)?
        } else {
            scales.try_clone()?
        };
        let rot_trimmed = if filled_rot < rot.shape()[0] {
            rot.slice(&[0], &[filled_rot], &[1], device)?
        } else {
            rot.try_clone()?
        };
        out.push((
            format!("l{idx}.v.codes"),
            OwnedTensor::from_array(&codes_trimmed)?,
        ));
        out.push((
            format!("l{idx}.v.scales"),
            OwnedTensor::from_array(&scales_trimmed)?,
        ));
        out.push((
            format!("l{idx}.v.rotations"),
            OwnedTensor::from_array(&rot_trimmed)?,
        ));
    } else {
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        let mut rot = Vec::new();
        for b in &qv.blocks {
            codes.extend_from_slice(&b.codes);
            scales.extend_from_slice(&b.scales);
            rot.extend_from_slice(&b.rotations);
        }
        out.push((format!("l{idx}.v.codes"), OwnedTensor::from_u8(&codes)));
        out.push((format!("l{idx}.v.scales"), OwnedTensor::from_f32(&scales)));
        out.push((format!("l{idx}.v.rotations"), OwnedTensor::from_u8(&rot)));
    }
    Ok(())
}

// ── PlanarK (K-axis PlanarQuant 4-bit) ───────────────────────────────────────
//
// K-side counterpart of `write_quant_planar_v`. Buffer layout is identical
// (codes/scales/rotations); only the tensor name prefix differs
// (`l{idx}.k.*` vs `l{idx}.v.*`). Returns the seq length.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn write_quant_planar_k_side(
    idx: usize,
    k: Option<&QuantPlanarK>,
    device: Device,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<i32> {
    let Some(qk) = k else { return Ok(0) };
    if let (Some(codes), Some(scales), Some(rot)) =
        (&qk.gpu_codes_buf, &qk.gpu_scales_buf, &qk.gpu_rotations_buf)
    {
        let prev_seq = qk.shape[2];
        let filled_codes = prev_seq * qk.gpu_codes_words_per_step;
        let filled_scales = prev_seq * qk.gpu_scales_per_step;
        let filled_rot = prev_seq * qk.gpu_rotations_words_per_step;
        let codes_trimmed = if filled_codes < codes.shape()[0] {
            codes.slice(&[0], &[filled_codes], &[1], device)?
        } else {
            codes.try_clone()?
        };
        let scales_trimmed = if filled_scales < scales.shape()[0] {
            scales.slice(&[0], &[filled_scales], &[1], device)?
        } else {
            scales.try_clone()?
        };
        let rot_trimmed = if filled_rot < rot.shape()[0] {
            rot.slice(&[0], &[filled_rot], &[1], device)?
        } else {
            rot.try_clone()?
        };
        out.push((
            format!("l{idx}.k.codes"),
            OwnedTensor::from_array(&codes_trimmed)?,
        ));
        out.push((
            format!("l{idx}.k.scales"),
            OwnedTensor::from_array(&scales_trimmed)?,
        ));
        out.push((
            format!("l{idx}.k.rotations"),
            OwnedTensor::from_array(&rot_trimmed)?,
        ));
    } else {
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        let mut rot = Vec::new();
        for b in &qk.blocks {
            codes.extend_from_slice(&b.codes);
            scales.extend_from_slice(&b.scales);
            rot.extend_from_slice(&b.rotations);
        }
        out.push((format!("l{idx}.k.codes"), OwnedTensor::from_u8(&codes)));
        out.push((format!("l{idx}.k.scales"), OwnedTensor::from_f32(&scales)));
        out.push((format!("l{idx}.k.rotations"), OwnedTensor::from_u8(&rot)));
    }
    Ok(qk.shape[2])
}

// ── IsoV3 ────────────────────────────────────────────────────────────────────

/// Serialize a `QuantIsoV3` V payload (IsoQuant 3-bit, quaternion SO(4)).
///
/// Writes four tensors per layer:
/// - `l{idx}.v.codes_packed` — packed 3-bit codes (`Vec<u32>` flattened as U8)
/// - `l{idx}.v.scales` — per-group f32 scales
/// - `l{idx}.v.quaternions` — per-group quaternion `[w,x,y,z]` f32s
/// - `l{idx}.v.norms` — per-token L2 norm f32s
///
/// All four buffers are flattened across all accumulated blocks in append order
/// so the reader can reconstruct a single `IsoBlocks` from the concatenation.
#[allow(
    clippy::unnecessary_wraps,
    reason = "returns Result<()> for consistency with all other write_quant_* helpers; callers use ? on this"
)]
fn write_quant_iso_v3(
    idx: usize,
    v: Option<&QuantIsoV3>,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qv) = v else { return Ok(()) };
    ensure_iso_blocks_cover_shape(
        qv.blocks.iter().map(|b| b.n_tokens).sum(),
        &qv.shape,
        idx,
        "v",
    )?;
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut quaternions: Vec<f32> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    for blk in &qv.blocks {
        // codes is Vec<u32>; flatten to raw bytes (little-endian u32 words).
        for &w in &blk.codes {
            codes.extend_from_slice(&w.to_le_bytes());
        }
        scales.extend_from_slice(&blk.scales);
        quaternions.extend_from_slice(&blk.quaternions);
        norms.extend_from_slice(&blk.norms);
    }
    out.push((
        format!("l{idx}.v.codes_packed"),
        OwnedTensor::from_u8(&codes),
    ));
    out.push((format!("l{idx}.v.scales"), OwnedTensor::from_f32(&scales)));
    out.push((
        format!("l{idx}.v.quaternions"),
        OwnedTensor::from_f32(&quaternions),
    ));
    out.push((format!("l{idx}.v.norms"), OwnedTensor::from_f32(&norms)));
    Ok(())
}

/// Serialize a `QuantIsoV4` V payload (IsoQuant 4-bit).
///
/// Identical wire layout to `write_quant_iso_v3` — the four buffers
/// (`codes_packed`, `scales`, `quaternions`, `norms`) are flattened over all
/// accumulated blocks. The bit-width is encoded in the layer geometry tag
/// (`ISOV4_LAYOUT_TAG`), not in the tensor names; the reader uses the tag to
/// dispatch to `QuantIsoV4`.
#[allow(
    clippy::unnecessary_wraps,
    reason = "returns Result<()> for consistency with all other write_quant_* helpers; callers use ? on this"
)]
fn write_quant_iso_v4(
    idx: usize,
    v: Option<&QuantIsoV4>,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qv) = v else { return Ok(()) };
    ensure_iso_blocks_cover_shape(
        qv.blocks.iter().map(|b| b.n_tokens).sum(),
        &qv.shape,
        idx,
        "v",
    )?;
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut quaternions: Vec<f32> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    for blk in &qv.blocks {
        for &w in &blk.codes {
            codes.extend_from_slice(&w.to_le_bytes());
        }
        scales.extend_from_slice(&blk.scales);
        quaternions.extend_from_slice(&blk.quaternions);
        norms.extend_from_slice(&blk.norms);
    }
    out.push((
        format!("l{idx}.v.codes_packed"),
        OwnedTensor::from_u8(&codes),
    ));
    out.push((format!("l{idx}.v.scales"), OwnedTensor::from_f32(&scales)));
    out.push((
        format!("l{idx}.v.quaternions"),
        OwnedTensor::from_f32(&quaternions),
    ));
    out.push((format!("l{idx}.v.norms"), OwnedTensor::from_f32(&norms)));
    Ok(())
}

/// Serialize a `QuantIsoK3` K payload (IsoQuant 3-bit K).
///
/// Wire layout identical to [`write_quant_iso_v3`] (codes_packed / scales /
/// quaternions / norms) but under the `l{idx}.k.*` tensor names so the K-side
/// reader can dispatch separately.
#[allow(
    clippy::unnecessary_wraps,
    reason = "returns Result<()> for consistency with all other write_quant_* helpers; callers use ? on this"
)]
fn write_quant_iso_k3(
    idx: usize,
    k: Option<&QuantIsoK3>,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qk) = k else { return Ok(()) };
    ensure_iso_blocks_cover_shape(
        qk.blocks.iter().map(|b| b.n_tokens).sum(),
        &qk.shape,
        idx,
        "k",
    )?;
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut quaternions: Vec<f32> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    for blk in &qk.blocks {
        for &w in &blk.codes {
            codes.extend_from_slice(&w.to_le_bytes());
        }
        scales.extend_from_slice(&blk.scales);
        quaternions.extend_from_slice(&blk.quaternions);
        norms.extend_from_slice(&blk.norms);
    }
    out.push((
        format!("l{idx}.k.codes_packed"),
        OwnedTensor::from_u8(&codes),
    ));
    out.push((format!("l{idx}.k.scales"), OwnedTensor::from_f32(&scales)));
    out.push((
        format!("l{idx}.k.quaternions"),
        OwnedTensor::from_f32(&quaternions),
    ));
    out.push((format!("l{idx}.k.norms"), OwnedTensor::from_f32(&norms)));
    Ok(())
}

/// Serialize a `QuantIsoK4` K payload (IsoQuant 4-bit K).
///
/// Identical wire layout to [`write_quant_iso_k3`]; bit-width encoded in the
/// layer geometry tag (`ISO_SYM_4_LAYOUT_TAG` / `ISO_K_ONLY_4_LAYOUT_TAG`),
/// not in the tensor names.
#[allow(
    clippy::unnecessary_wraps,
    reason = "returns Result<()> for consistency with all other write_quant_* helpers; callers use ? on this"
)]
fn write_quant_iso_k4(
    idx: usize,
    k: Option<&QuantIsoK4>,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qk) = k else { return Ok(()) };
    ensure_iso_blocks_cover_shape(
        qk.blocks.iter().map(|b| b.n_tokens).sum(),
        &qk.shape,
        idx,
        "k",
    )?;
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut quaternions: Vec<f32> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    for blk in &qk.blocks {
        for &w in &blk.codes {
            codes.extend_from_slice(&w.to_le_bytes());
        }
        scales.extend_from_slice(&blk.scales);
        quaternions.extend_from_slice(&blk.quaternions);
        norms.extend_from_slice(&blk.norms);
    }
    out.push((
        format!("l{idx}.k.codes_packed"),
        OwnedTensor::from_u8(&codes),
    ));
    out.push((format!("l{idx}.k.scales"), OwnedTensor::from_f32(&scales)));
    out.push((
        format!("l{idx}.k.quaternions"),
        OwnedTensor::from_f32(&quaternions),
    ));
    out.push((format!("l{idx}.k.norms"), OwnedTensor::from_f32(&norms)));
    Ok(())
}

/// Serialize a `QuantRotorK3` K payload (rotor3 K codec).
///
/// Wire layout under `l{idx}.k.*`:
///
/// | Tensor name           | dtype | semantics |
/// |-----------------------|-------|-----------|
/// | `l{idx}.k.codes_packed` | u8 (LE u32 underneath) | packed 3-bit codes |
/// | `l{idx}.k.scales`     | f32   | per-group scale |
/// | `l{idx}.k.norms`      | f32   | per-token L2 norm |
/// | `l{idx}.k.rotors`     | f32   | static rotor table `[n_groups, 4]` |
/// | `l{idx}.k.qjl_codes`  | u8    | packed 1-bit QJL signs (when QJL active) |
/// | `l{idx}.k.qjl_norms`  | f32   | per-token residual L2 norm (QJL active) |
/// | `l{idx}.k.qjl_s`      | f32   | static QJL projection matrix (QJL active) |
///
/// The QJL fields are omitted when the cache has no `qjl_s_matrix`. The
/// geometry tag (`ROTOR_SYM_3_QJL_LAYOUT_TAG` vs. `ROTOR_SYM_3_LAYOUT_TAG`)
/// is the load-bearing signal to the reader.
/// Loud guard: a rotor K store must reach serialization with its CPU `blocks`
/// covering the full accumulated `shape[2]`.
///
/// On the fused decode path the store keeps a **ring-only tail** — `blocks`
/// trail `shape[2]`, with the GPU ring holding the decode tail. The spill clone
/// (`KvCache::try_deep_clone`) materialises that tail into complete blocks
/// before the store reaches here; if it did not, serializing `qk.blocks`
/// directly would persist a **truncated** store. Rather than silently write the
/// short prefix, reject it — the invariant is enforced at the persistence
/// boundary too, not only at the codec.
fn ensure_rotor_k_blocks_cover_shape(
    blocks_tokens: usize,
    shape: &[i32],
    idx: usize,
) -> Result<()> {
    if shape.len() != 4 {
        return Ok(());
    }
    let full_tokens: usize = shape.iter().take(3).map(|&d| d.max(0) as usize).product();
    if blocks_tokens != full_tokens {
        return Err(BlockIoError::TruncatedStore(format!(
            "l{idx}.k rotor store: CPU blocks hold {blocks_tokens} tokens but shape {shape:?} \
             implies {full_tokens} — the ring-only decode tail was not materialised before spill"
        ))
        .into());
    }
    Ok(())
}

/// V-side mirror of [`ensure_rotor_k_blocks_cover_shape`]: a rotor V store must
/// reach serialization with its CPU `blocks` covering the full accumulated
/// `shape[2]`. On the fused symmetric decode path the store keeps a ring-only
/// tail; the spill clone (`KvCache::try_deep_clone`) materialises it into
/// complete blocks first. Reject a short store rather than persist a truncated V.
fn ensure_rotor_v_blocks_cover_shape(
    blocks_tokens: usize,
    shape: &[i32],
    idx: usize,
) -> Result<()> {
    if shape.len() != 4 {
        return Ok(());
    }
    let full_tokens: usize = shape.iter().take(3).map(|&d| d.max(0) as usize).product();
    if blocks_tokens != full_tokens {
        return Err(BlockIoError::TruncatedStore(format!(
            "l{idx}.v rotor store: CPU blocks hold {blocks_tokens} tokens but shape {shape:?} \
             implies {full_tokens} — the ring-only decode tail was not materialised before spill"
        ))
        .into());
    }
    Ok(())
}

/// Iso mirror of [`ensure_rotor_v_blocks_cover_shape`], for either axis.
///
/// An iso K store (K-only or symmetric) and an iso V store (symmetric) both keep
/// a ring-only decode tail on the fused path; the spill clone
/// (`KvCache::try_deep_clone`) materialises it into complete blocks first. Reject
/// a short store rather than persist a truncated iso payload. `axis` is `"k"` or
/// `"v"` for the diagnostic.
fn ensure_iso_blocks_cover_shape(
    blocks_tokens: usize,
    shape: &[i32],
    idx: usize,
    axis: &str,
) -> Result<()> {
    if shape.len() != 4 {
        return Ok(());
    }
    let full_tokens: usize = shape.iter().take(3).map(|&d| d.max(0) as usize).product();
    if blocks_tokens != full_tokens {
        return Err(BlockIoError::TruncatedStore(format!(
            "l{idx}.{axis} iso store: CPU blocks hold {blocks_tokens} tokens but shape {shape:?} \
             implies {full_tokens} — the ring-only decode tail was not materialised before spill"
        ))
        .into());
    }
    Ok(())
}

fn write_quant_rotor_k3(
    idx: usize,
    k: Option<&QuantRotorK3>,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qk) = k else { return Ok(()) };
    ensure_rotor_k_blocks_cover_shape(qk.blocks.iter().map(|b| b.n_tokens).sum(), &qk.shape, idx)?;
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    let mut qjl_codes: Vec<u8> = Vec::new();
    let mut qjl_norms: Vec<f32> = Vec::new();
    for blk in &qk.blocks {
        for &w in &blk.codes {
            codes.extend_from_slice(&w.to_le_bytes());
        }
        scales.extend_from_slice(&blk.scales);
        norms.extend_from_slice(&blk.norms);
        qjl_codes.extend_from_slice(&blk.qjl_codes);
        qjl_norms.extend_from_slice(&blk.qjl_norms);
    }
    out.push((
        format!("l{idx}.k.codes_packed"),
        OwnedTensor::from_u8(&codes),
    ));
    out.push((format!("l{idx}.k.scales"), OwnedTensor::from_f32(&scales)));
    out.push((format!("l{idx}.k.norms"), OwnedTensor::from_f32(&norms)));
    out.push((
        format!("l{idx}.k.rotors"),
        OwnedTensor::from_f32(&qk.rotors),
    ));
    if let Some(s_matrix) = &qk.qjl_s_matrix {
        out.push((
            format!("l{idx}.k.qjl_codes"),
            OwnedTensor::from_u8(&qjl_codes),
        ));
        out.push((
            format!("l{idx}.k.qjl_norms"),
            OwnedTensor::from_f32(&qjl_norms),
        ));
        out.push((format!("l{idx}.k.qjl_s"), OwnedTensor::from_f32(s_matrix)));
    }
    Ok(())
}

/// Serialize a `QuantRotorK4` K payload (rotor4 K codec).
/// Identical wire layout to [`write_quant_rotor_k3`]; bit-width encoded in
/// the geometry tag.
fn write_quant_rotor_k4(
    idx: usize,
    k: Option<&QuantRotorK4>,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qk) = k else { return Ok(()) };
    ensure_rotor_k_blocks_cover_shape(qk.blocks.iter().map(|b| b.n_tokens).sum(), &qk.shape, idx)?;
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    let mut qjl_codes: Vec<u8> = Vec::new();
    let mut qjl_norms: Vec<f32> = Vec::new();
    for blk in &qk.blocks {
        for &w in &blk.codes {
            codes.extend_from_slice(&w.to_le_bytes());
        }
        scales.extend_from_slice(&blk.scales);
        norms.extend_from_slice(&blk.norms);
        qjl_codes.extend_from_slice(&blk.qjl_codes);
        qjl_norms.extend_from_slice(&blk.qjl_norms);
    }
    out.push((
        format!("l{idx}.k.codes_packed"),
        OwnedTensor::from_u8(&codes),
    ));
    out.push((format!("l{idx}.k.scales"), OwnedTensor::from_f32(&scales)));
    out.push((format!("l{idx}.k.norms"), OwnedTensor::from_f32(&norms)));
    out.push((
        format!("l{idx}.k.rotors"),
        OwnedTensor::from_f32(&qk.rotors),
    ));
    if let Some(s_matrix) = &qk.qjl_s_matrix {
        out.push((
            format!("l{idx}.k.qjl_codes"),
            OwnedTensor::from_u8(&qjl_codes),
        ));
        out.push((
            format!("l{idx}.k.qjl_norms"),
            OwnedTensor::from_f32(&qjl_norms),
        ));
        out.push((format!("l{idx}.k.qjl_s"), OwnedTensor::from_f32(s_matrix)));
    }
    Ok(())
}

/// Serialize a `QuantRotorV3` V payload (rotor3 codec).
///
/// Layout on disk:
///
/// | Tensor name           | dtype | semantics |
/// |-----------------------|-------|-----------|
/// | `l{idx}.v.codes_packed` | u8 (LE u32 underneath) | packed 3-bit codes |
/// | `l{idx}.v.scales`     | f32   | per-group scale |
/// | `l{idx}.v.norms`      | f32   | per-token L2 norm |
/// | `l{idx}.v.rotors`     | f32   | static rotor table `[n_groups, 4]` |
///
/// The rotor table is persisted **once** per layer — it is shared across all
/// accumulated blocks. The reader reconstructs `QuantRotorV3::from_cpu_blocks`
/// with the rotor table and a single concatenated block.
#[allow(
    clippy::unnecessary_wraps,
    reason = "returns Result<()> for consistency with all other write_quant_* helpers; callers use ? on this"
)]
fn write_quant_rotor_v3(
    idx: usize,
    v: Option<&QuantRotorV3>,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qv) = v else { return Ok(()) };
    ensure_rotor_v_blocks_cover_shape(qv.blocks.iter().map(|b| b.n_tokens).sum(), &qv.shape, idx)?;
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    for blk in &qv.blocks {
        for &w in &blk.codes {
            codes.extend_from_slice(&w.to_le_bytes());
        }
        scales.extend_from_slice(&blk.scales);
        norms.extend_from_slice(&blk.norms);
    }
    out.push((
        format!("l{idx}.v.codes_packed"),
        OwnedTensor::from_u8(&codes),
    ));
    out.push((format!("l{idx}.v.scales"), OwnedTensor::from_f32(&scales)));
    out.push((format!("l{idx}.v.norms"), OwnedTensor::from_f32(&norms)));
    out.push((
        format!("l{idx}.v.rotors"),
        OwnedTensor::from_f32(&qv.rotors),
    ));
    Ok(())
}

/// Serialize a `QuantRotorV4` V payload (rotor4 codec).
///
/// On-disk layout is identical to rotor3 (`codes_packed`, `scales`, `norms`,
/// `rotors`) — the only difference is the bit packing: 4-bit codes pack 8
/// multivector components into exactly 1 u32 per group (vs. rotor3's 3-bit
/// 10-vals-per-u32). The tensor names and byte representation are the same,
/// so the reader can reconstruct `QuantRotorV4::from_cpu_blocks` symmetrically.
#[allow(
    clippy::unnecessary_wraps,
    reason = "returns Result<()> for consistency with all other write_quant_* helpers; callers use ? on this"
)]
fn write_quant_rotor_v4(
    idx: usize,
    v: Option<&QuantRotorV4>,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<()> {
    let Some(qv) = v else { return Ok(()) };
    ensure_rotor_v_blocks_cover_shape(qv.blocks.iter().map(|b| b.n_tokens).sum(), &qv.shape, idx)?;
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    for blk in &qv.blocks {
        for &w in &blk.codes {
            codes.extend_from_slice(&w.to_le_bytes());
        }
        scales.extend_from_slice(&blk.scales);
        norms.extend_from_slice(&blk.norms);
    }
    out.push((
        format!("l{idx}.v.codes_packed"),
        OwnedTensor::from_u8(&codes),
    ));
    out.push((format!("l{idx}.v.scales"), OwnedTensor::from_f32(&scales)));
    out.push((format!("l{idx}.v.norms"), OwnedTensor::from_f32(&norms)));
    out.push((
        format!("l{idx}.v.rotors"),
        OwnedTensor::from_f32(&qv.rotors),
    ));
    Ok(())
}

// ── Mixed ──────────────────────────────────────────────────────────────────────

fn write_mixed(
    idx: usize,
    state: &MixedKvState,
    max_seq: i32,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<(String, i32)> {
    if let Some(t) = &state.keys {
        out.push((
            format!("l{idx}.k.codes"),
            OwnedTensor::from_array(&t.codes)?,
        ));
        out.push((
            format!("l{idx}.k.scales"),
            OwnedTensor::from_array(&t.scales)?,
        ));
        out.push((
            format!("l{idx}.k.biases"),
            OwnedTensor::from_array(&t.biases)?,
        ));
    }
    if let Some(t) = &state.values {
        out.push((
            format!("l{idx}.v.codes"),
            OwnedTensor::from_array(&t.codes)?,
        ));
        out.push((
            format!("l{idx}.v.scales"),
            OwnedTensor::from_array(&t.scales)?,
        ));
        out.push((
            format!("l{idx}.v.biases"),
            OwnedTensor::from_array(&t.biases)?,
        ));
    }
    let geom = format!(
        "{{\"tag\":\"mixed\",\"max_seq\":{},\"k_bits\":{},\"v_bits\":{},\
         \"k_group_size\":{},\"v_group_size\":{},\"offset\":{},\"rotate_k\":{}}}",
        max_seq,
        state.k_bits,
        state.v_bits,
        state.k_group_size,
        state.v_group_size,
        state.offset,
        state.rotate_k,
    );
    Ok((geom, state.offset))
}

// ── Paged ──────────────────────────────────────────────────────────────────────

/// Serialize a Paged layer. The page table + per-page slabs are flattened by
/// `gather()` into one contiguous codes/scales(/rotations) Array per component;
/// the page geometry is recorded so the reader can re-append into a fresh table.
#[allow(clippy::too_many_arguments)]
fn write_paged(
    idx: usize,
    quant: KvQuant,
    k: Option<&PagedKStorage>,
    v_k8: Option<&PagedVStorage>,
    v_planar: Option<&PagedPlanarVStorage>,
    max_seq: i32,
    device: Device,
    out: &mut Vec<(String, OwnedTensor)>,
) -> Result<(String, i32)> {
    let mut total_tokens = 0i32;
    let mut page_tokens = 0i32;
    let mut shape: Vec<i32> = Vec::new();

    if let Some(pk) = k {
        let (codes, scales) = pk.gather(device)?;
        out.push((format!("l{idx}.k.codes"), OwnedTensor::from_array(&codes)?));
        out.push((
            format!("l{idx}.k.scales"),
            OwnedTensor::from_array(&scales)?,
        ));
        total_tokens = pk.total_tokens;
        page_tokens = pk.page_tokens;
        shape = pk.shape.clone();
    }
    if let Some(pv) = v_k8 {
        let (codes, scales) = pv.gather(device)?;
        out.push((format!("l{idx}.v.codes"), OwnedTensor::from_array(&codes)?));
        out.push((
            format!("l{idx}.v.scales"),
            OwnedTensor::from_array(&scales)?,
        ));
        if shape.is_empty() {
            shape = pv.shape.clone();
            total_tokens = pv.total_tokens;
            page_tokens = pv.page_tokens;
        }
    }
    if let Some(pv) = v_planar {
        let (codes, scales, rot) = pv.gather(device)?;
        out.push((format!("l{idx}.v.codes"), OwnedTensor::from_array(&codes)?));
        out.push((
            format!("l{idx}.v.scales"),
            OwnedTensor::from_array(&scales)?,
        ));
        out.push((
            format!("l{idx}.v.rotations"),
            OwnedTensor::from_array(&rot)?,
        ));
        if shape.is_empty() {
            shape = pv.shape.clone();
            total_tokens = pv.total_tokens;
            page_tokens = pv.page_tokens;
        }
    }

    let geom = format!(
        "{{\"tag\":\"paged\",\"max_seq\":{max_seq},\"quant\":\"{quant}\",\
         \"page_tokens\":{page_tokens},\"total_tokens\":{total_tokens},\"shape\":[{}]}}",
        csv(&shape)
    );
    Ok((geom, total_tokens))
}

// ── Reader ─────────────────────────────────────────────────────────────────────

/// Reads an N-layer KV block back from a safetensors file, verifying the header
/// against the model being hydrated.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed reader — field is private; public API is hydrate, not struct literal construction"
)]
#[allow(missing_debug_implementations)]
pub struct KvBlockReader {
    bytes: Vec<u8>,
}

impl KvBlockReader {
    /// Load the file into memory. Header verification happens in [`Self::hydrate`].
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_existing(path)?
            .ok_or_else(|| Error::Mlx(format!("KV block read: {} not found", path.display())))
    }

    /// Like [`Self::open`], but reports a file that is not there as `Ok(None)`
    /// rather than an error.
    ///
    /// LRU eviction unlinks blocks whose rows it has already deleted, so a
    /// hydrate that read the row a moment earlier finds the file gone as a
    /// matter of routine. That is a miss, not corruption, and the single
    /// `fs::read` classifies it without a second `stat` (and without the
    /// window one would open).
    pub fn open_existing(path: &Path) -> Result<Option<Self>> {
        match std::fs::read(path) {
            Ok(bytes) => Ok(Some(Self { bytes })),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Mlx(format!("KV block read: {e}"))),
        }
    }

    /// Read the `model_id` header without hydrating.
    pub fn model_id(&self) -> Result<String> {
        read_meta(&self.header()?, META_MODEL_ID)
    }

    /// Read the `kv_quant` header string without hydrating.
    pub fn kv_quant_str(&self) -> Result<String> {
        read_meta(&self.header()?, META_KV_QUANT)
    }

    /// Return the supported on-disk format version. Missing, older, and future
    /// versions are rejected rather than guessed at: a cache must miss and be
    /// recomputed, never hydrated under an incompatible wire contract.
    fn format_version(header: &Metadata) -> Result<u32> {
        let Some(raw) = header
            .metadata()
            .as_ref()
            .and_then(|m| m.get(META_FORMAT_VERSION))
        else {
            return Err(BlockIoError::Header("missing format_version".into()).into());
        };
        let version = raw
            .parse::<u32>()
            .map_err(|e| BlockIoError::Header(format!("bad format_version: {e}")))?;
        match version {
            2 => Ok(version),
            _ => Err(BlockIoError::Header(format!("unsupported format_version {version}")).into()),
        }
    }

    /// Read the full prompt identity embedded in a v2 record.
    ///
    /// A v2 record may omit identity for storage-only/test callers, but the
    /// production hydrate bridge requires it before promotion.
    fn block_identity(&self) -> Result<Option<BlockIdentity>> {
        let header = self.header()?;
        Self::format_version(&header)?;
        let Some(tensor_name) = header
            .metadata()
            .as_ref()
            .and_then(|m| m.get(META_PROMPT_IDS_TENSOR))
        else {
            if header.metadata().as_ref().is_some_and(|m| {
                m.contains_key(META_EXACT_REPLAY_ID) || m.contains_key(META_EXACT_REPLAY_PIECE)
            }) {
                return Err(BlockIoError::Header(
                    "exact replay metadata without prompt identity tensor".into(),
                )
                .into());
            }
            return Ok(None);
        };
        if tensor_name != "__prompt_ids" {
            return Err(
                BlockIoError::Header("invalid prompt identity tensor marker".into()).into(),
            );
        }
        let st = self.parse()?;
        let tensor = tensor_req(&st, tensor_name)?;
        let view = st
            .tensor(tensor_name)
            .map_err(|e| Error::Mlx(format!("KV block identity tensor: {e}")))?;
        if view.dtype() != StDtype::U32 || view.shape().len() != 1 {
            return Err(
                BlockIoError::Header("prompt identity tensor must be rank-1 U32".into()).into(),
            );
        }
        let bytes = view.data();
        if bytes.len() % size_of::<u32>() != 0 {
            return Err(
                BlockIoError::Header("prompt identity tensor has a partial U32".into()).into(),
            );
        }
        let prompt_ids = bytes
            .chunks_exact(size_of::<u32>())
            .map(|chunk| {
                let bytes: [u8; size_of::<u32>()] = chunk.try_into().map_err(|_| {
                    BlockIoError::Header("prompt identity tensor has a partial U32".into())
                })?;
                Ok(u32::from_le_bytes(bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        let shape_len =
            view.shape().first().copied().ok_or_else(|| {
                BlockIoError::Header("prompt identity tensor has no shape".into())
            })?;
        if prompt_ids.len() != shape_len {
            return Err(BlockIoError::Header(
                "prompt identity tensor shape does not match payload".into(),
            )
            .into());
        }
        // Keep `tensor` above as a required-tensor validation. The view is
        // borrowed from the same parsed file and the conversion intentionally
        // remains host-only.
        let _ = tensor;
        let metadata = header.metadata().as_ref();
        let replay_id = metadata.and_then(|m| m.get(META_EXACT_REPLAY_ID));
        let replay_piece = metadata.and_then(|m| m.get(META_EXACT_REPLAY_PIECE));
        let exact_replay = match (replay_id, replay_piece) {
            (None, None) => None,
            (Some(id), Some(piece)) => Some(ExactReplayMetadata {
                id: id
                    .parse()
                    .map_err(|e| BlockIoError::Header(format!("bad exact replay id: {e}")))?,
                piece: piece.clone(),
            }),
            _ => {
                return Err(BlockIoError::Header(
                    "exact replay id and piece must be present together".into(),
                )
                .into())
            }
        };
        Ok(Some(BlockIdentity {
            prompt_ids,
            exact_replay,
        }))
    }

    /// Read restart-safe rotating-ring snapshots, if present.
    fn rotating_snapshots(&self) -> Result<Vec<Option<RotatingStateSnapshot>>> {
        let header = self.header()?;
        let st = self.parse()?;
        let n: usize = read_meta(&header, META_N_LAYERS)?
            .parse()
            .map_err(|e| BlockIoError::Header(format!("bad n_layers: {e}")))?;
        Self::format_version(&header)?;
        let mut out = Vec::with_capacity(n);
        for idx in 0..n {
            let geom = read_meta(&header, &geom_key(idx))?;
            if geom_tag(&geom) != "rotating" {
                out.push(None);
                continue;
            }
            let shape = |side: &str| -> Result<Vec<i32>> {
                let key = format!("{side}_shape");
                let prefix = format!("\"{key}\":[");
                let start = geom
                    .find(&prefix)
                    .ok_or_else(|| BlockIoError::Header(format!("rotating geom missing {key}")))?
                    + prefix.len();
                let rest = &geom[start..];
                let end = rest
                    .find(']')
                    .ok_or_else(|| BlockIoError::Header("rotating shape unterminated".into()))?;
                rest[..end]
                    .split(',')
                    .map(|s| {
                        s.parse().map_err(|e| {
                            Error::from(BlockIoError::Header(format!("bad rotating shape: {e}")))
                        })
                    })
                    .collect::<Result<Vec<i32>>>()
            };
            let dtype = match geom_field(&geom, "dtype") {
                Some("bf16") => Dtype::Bf16,
                Some("f16") => Dtype::F16,
                Some("f32") => Dtype::F32,
                Some("u8") => Dtype::U8,
                Some("u32") => Dtype::U32,
                Some("i32") => Dtype::I32,
                _ => return Err(BlockIoError::Header("bad rotating dtype".into()).into()),
            };
            let offset = geom_i32(&geom, "offset")?;
            let key_name = format!("l{idx}.rotating.k");
            let value_name = format!("l{idx}.rotating.v");
            let has_keys = st.tensor(&key_name).is_ok();
            let has_values = st.tensor(&value_name).is_ok();
            if !has_keys && !has_values {
                if offset != 0 {
                    return Err(BlockIoError::Header(
                        "non-empty rotating payload is missing K/V tensors".into(),
                    )
                    .into());
                }
                out.push(Some(RotatingStateSnapshot {
                    keys: None,
                    values: None,
                    offset,
                    max_size: geom_i32(&geom, "max_seq")?,
                    keep: geom_i32(&geom, "keep")?,
                    valid_len: geom_i32(&geom, "valid_len")?,
                    idx: geom_i32(&geom, "idx")?,
                }));
                continue;
            }
            if has_keys != has_values {
                return Err(BlockIoError::Header(
                    "rotating payload must contain both K and V tensors".into(),
                )
                .into());
            }
            let tensor = |name: String, shape: Vec<i32>| -> Result<RotatingTensorSnapshot> {
                let t = tensor_req(&st, &name)?;
                let bytes = t.to_bytes()?;
                let expected: usize =
                    shape.iter().map(|&x| x.max(0) as usize).product::<usize>() * dtype.itemsize();
                if bytes.len() != expected {
                    return Err(BlockIoError::Header(format!(
                        "rotating tensor {name} byte length mismatch"
                    ))
                    .into());
                }
                Ok(RotatingTensorSnapshot {
                    bytes,
                    shape,
                    dtype,
                })
            };
            let keys = tensor(format!("l{idx}.rotating.k"), shape("k")?)?;
            let values = tensor(format!("l{idx}.rotating.v"), shape("v")?)?;
            out.push(Some(RotatingStateSnapshot {
                keys: Some(keys),
                values: Some(values),
                offset,
                max_size: geom_i32(&geom, "max_seq")?,
                keep: geom_i32(&geom, "keep")?,
                valid_len: geom_i32(&geom, "valid_len")?,
                idx: geom_i32(&geom, "idx")?,
            }));
        }
        Ok(out)
    }

    /// Read the recorded filled sequence length (`seq_len` header) — the number
    /// of prompt tokens this block was spilled at. Used by the hydrate
    /// path to set each reconstructed `KvCache`'s `offset`.
    pub fn seq_len(&self) -> Result<i32> {
        read_meta(&self.header()?, META_SEQ_LEN)?
            .parse()
            .map_err(|e| BlockIoError::Header(format!("bad seq_len: {e}")).into())
    }

    /// Read exact per-layer offsets for v2 records. Shared-KV consumer layers
    /// are intentionally empty (`0`) while their producer carries the full
    /// prompt offset; flattening both to the block maximum changes the cache
    /// topology.
    fn layer_offsets(&self, n_layers: usize) -> Result<Vec<i32>> {
        let header = self.header()?;
        Self::format_version(&header)?;
        let metadata = header
            .metadata()
            .as_ref()
            .ok_or_else(|| BlockIoError::Header("missing metadata".into()))?;
        (0..n_layers)
            .map(|idx| {
                metadata
                    .get(&offset_key(idx))
                    .ok_or_else(|| {
                        Error::from(BlockIoError::Header(format!(
                            "missing per-layer offset for layer {idx}"
                        )))
                    })?
                    .parse::<i32>()
                    .map_err(|e| {
                        BlockIoError::Header(format!("bad per-layer offset for layer {idx}: {e}"))
                            .into()
                    })
            })
            .collect()
    }

    fn parse(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.bytes)
            .map_err(|e| Error::Mlx(format!("KV block deserialize: {e}")))
    }

    /// Parse just the header (`__metadata__` + tensor index) without building views.
    fn header(&self) -> Result<Metadata> {
        SafeTensors::read_metadata(&self.bytes)
            .map(|(_, m)| m)
            .map_err(|e| Error::Mlx(format!("KV block header parse: {e}")))
    }

    /// Reconstruct the per-layer [`KvStorage`] vector (plus linear-attn caches)
    /// after verifying the file's `model_id` and `kv_quant` match the model.
    ///
    /// Returns `Err(BlockIoError::ModelIdMismatch | KvQuantMismatch)` on a
    /// mismatch — the cache is never silently hydrated from a wrong file.
    pub(super) fn hydrate(
        &self,
        expected_model_id: &str,
        expected_kv_quant: KvQuant,
        device: Device,
    ) -> Result<HydratedLayers> {
        let header = self.header()?;
        let st = self.parse()?;
        Self::format_version(&header)?;

        let found_model = read_meta(&header, META_MODEL_ID)?;
        if found_model != expected_model_id {
            return Err(BlockIoError::ModelIdMismatch {
                found: found_model,
                expected: expected_model_id.to_string(),
            }
            .into());
        }
        let found_quant = read_meta(&header, META_KV_QUANT)?;
        let expected_quant = expected_kv_quant.to_string();
        if found_quant != expected_quant {
            return Err(BlockIoError::KvQuantMismatch {
                found: found_quant,
                expected: expected_quant,
            }
            .into());
        }

        let n_layers: usize = read_meta(&header, META_N_LAYERS)?
            .parse()
            .map_err(|e| BlockIoError::Header(format!("bad n_layers: {e}")))?;

        let mut layers = Vec::with_capacity(n_layers);
        let mut none_bf16: Vec<NoneBf16Seed> = Vec::with_capacity(n_layers);
        for idx in 0..n_layers {
            let geom = read_meta(&header, &geom_key(idx))?;
            layers.push(read_layer(&st, idx, &geom, device)?);
            // For a "none_bf16" layer (KvQuant::None spill that carried the
            // off-storage bf16 prefix), restore the K/V pair so the caller can
            // re-seed the parent KvCache's decode buffers. All other tags hold
            // their K/V inside the reconstructed KvStorage and have no bf16 seed.
            if geom_tag(&geom) == "none_bf16" {
                let k = tensor_req(&st, &format!("l{idx}.k.bf16"))?;
                let v = tensor_req(&st, &format!("l{idx}.v.bf16"))?;
                none_bf16.push(Some((k, v)));
            } else {
                none_bf16.push(None);
            }
        }

        let n_linear: usize = header
            .metadata()
            .as_ref()
            .and_then(|m| m.get(META_N_LINEAR))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let mut linear = Vec::with_capacity(n_linear);
        for idx in 0..n_linear {
            let mut lac = LinearAttnCache::new();
            lac.conv_state = tensor_opt(&st, &format!("lin{idx}.conv_state"))?;
            lac.delta_state = tensor_opt(&st, &format!("lin{idx}.delta_state"))?;
            linear.push(lac);
        }

        Ok((layers, none_bf16, linear))
    }
}

fn read_meta(header: &Metadata, key: &str) -> Result<String> {
    header
        .metadata()
        .as_ref()
        .and_then(|m| m.get(key))
        .cloned()
        .ok_or_else(|| BlockIoError::Header(format!("missing key '{key}'")).into())
}

/// Load a tensor by name into an `Array`, or `None` if absent.
fn tensor_opt(st: &SafeTensors<'_>, name: &str) -> Result<Option<Array>> {
    match st.tensor(name) {
        Ok(view) => {
            let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
            let dtype = rmlx_dtype(view.dtype())?;
            Ok(Some(Array::from_bytes(view.data(), &shape, dtype)?))
        }
        Err(_) => Ok(None),
    }
}

/// Load a required tensor by name.
fn tensor_req(st: &SafeTensors<'_>, name: &str) -> Result<Array> {
    tensor_opt(st, name)?.ok_or_else(|| BlockIoError::MissingTensor(name.to_string()).into())
}

/// Pull the `"tag"` field out of a geometry JSON string (no serde dep — the
/// JSON is produced by this module so a small scan is enough).
fn geom_tag(geom: &str) -> &str {
    geom_field(geom, "tag").unwrap_or("")
}

/// Extract a string or numeric/bool field value from the flat geometry JSON.
fn geom_field<'a>(geom: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":");
    let start = geom.find(&needle)? + needle.len();
    let rest = geom[start..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        Some(&stripped[..end])
    } else {
        let end = rest.find([',', '}'])?;
        Some(rest[..end].trim())
    }
}

fn geom_i32(geom: &str, key: &str) -> Result<i32> {
    geom_field(geom, key)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| BlockIoError::Header(format!("geom missing/bad '{key}'")).into())
}

/// Parse the `"shape":[a,b,c,d]` field.
fn geom_shape(geom: &str) -> Result<Vec<i32>> {
    let start = geom
        .find("\"shape\":[")
        .ok_or_else(|| Error::from(BlockIoError::Header("geom missing shape".into())))?
        + "\"shape\":[".len();
    let rest = &geom[start..];
    let end = rest
        .find(']')
        .ok_or_else(|| Error::from(BlockIoError::Header("geom unterminated shape".into())))?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<i32>()
                .map_err(|e| Error::from(BlockIoError::Header(format!("bad shape elem: {e}"))))
        })
        .collect()
}

#[allow(
    clippy::too_many_lines,
    reason = "single match over the geometry-tag dispatch; one arm per layout tag, splitting per-tag arms into helpers would scatter the registry"
)]
fn read_layer(st: &SafeTensors<'_>, idx: usize, geom: &str, device: Device) -> Result<KvStorage> {
    match geom_tag(geom) {
        // The physical K/V for this layer is restored separately through
        // `rotating_snapshots`; the storage slot remains geometry-only.
        "rotating" => Ok(KvStorage::None {
            max_seq: geom_i32(geom, "max_seq")?,
        }),
        "k8v4" => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::K8V4 {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(read_quant_v(st, idx, &shape)?),
                max_seq,
            })
        }
        "k8v8" => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::K8V8 {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(read_quant_k(st, idx, "v", &shape)?),
                max_seq,
            })
        }
        "planar" => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::Planar {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(read_quant_planar_v(st, idx, &shape, 4)?),
                max_seq,
                bits: 4,
            })
        }
        // Planar3 — same layout as "planar" but 3-bit V codebook.
        "planar3" => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::Planar {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(read_quant_planar_v(st, idx, &shape, 3)?),
                max_seq,
                bits: 3,
            })
        }
        // Both tags reconstruct geometry-only None storage. The "none_bf16"
        // variant additionally carries an off-storage bf16 K/V prefix, read in
        // `hydrate` and re-seeded onto the parent KvCache by the caller.
        "none" | "none_bf16" => Ok(KvStorage::None {
            max_seq: geom_i32(geom, "max_seq")?,
        }),
        "mixed" => read_mixed(st, idx, geom),
        "paged" => read_paged(st, idx, geom, device),
        // K8VTurbo3 — same structure as K8V4 but bits=3 on V.
        "k8vturbo3" => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            let v = read_quant_v_bits(st, idx, &shape, 3)?;
            Ok(KvStorage::K8VTurbo3 {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(v),
                max_seq,
            })
        }
        // K8VTurbo3Tcq — byte-for-byte compatible with k8vturbo3 on the V
        // codes/scales side. Hydrated `QuantV` is tagged `use_tcq=true` so
        // subsequent decode-step encodes re-enter the Viterbi path.
        tag if tag == K8VTURBO3_TCQ_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            let mut v = read_quant_v_bits(st, idx, &shape, 3)?;
            v.use_tcq = true;
            Ok(KvStorage::K8VTurbo3Tcq {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(v),
                max_seq,
            })
        }
        // K8VTurbo2Tcq — byte-for-byte compatible with k8vturbo2 on the V
        // codes/scales side (2-bit pack). Hydrated `QuantV` is tagged
        // `use_tcq=true` so subsequent decode-step encodes re-enter the Viterbi
        // path instead of falling back to nearest-centroid.
        tag if tag == K8VTURBO2_TCQ_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            let mut v = read_quant_v_bits(st, idx, &shape, 2)?;
            v.use_tcq = true;
            Ok(KvStorage::K8VTurbo2Tcq {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(v),
                max_seq,
            })
        }
        // TurboSym3 — symmetric WHT-3 K + turbo3 V. Match against the canonical
        // layout tag constant.
        tag if tag == TURBOSYM3_LAYOUT_TAG => read_tsym3(st, idx, geom),
        // TurboSym4 — symmetric WHT-4 K + tq4 V. Match against the canonical
        // layout tag constant.
        tag if tag == TURBOSYM4_LAYOUT_TAG => read_tsym4(st, idx, geom),
        // PlanarK — K-only payload (codes/scales/rotations); V is bf16
        // off-storage. Match against the canonical layout tag constant.
        tag if tag == PLANARK4_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::PlanarK {
                k: Some(read_quant_planar_k(st, idx, &shape)?),
                max_seq,
            })
        }
        // K8VTurbo2 — same structure as K8V4 but bits=2 on V.
        "k8vturbo2" => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            let v = read_quant_v_bits(st, idx, &shape, 2)?;
            Ok(KvStorage::K8VTurbo2 {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(v),
                max_seq,
            })
        }
        // IsoV3 — K is QuantK (q8_0); V is QuantIsoV3. Geometry bits must be
        // 3 and group_size must be 4; mismatch is a hard error.
        tag if tag == ISOV3_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::IsoV3 {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(read_quant_iso_v3(st, idx, &shape)?),
                max_seq,
            })
        }
        // IsoV4 — K is QuantK (q8_0); V is QuantIsoV4.
        tag if tag == ISOV4_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::IsoV4 {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(read_quant_iso_v4(st, idx, &shape)?),
                max_seq,
            })
        }
        // RotorV3 — K is QuantK (q8_0); V is QuantRotorV3.
        tag if tag == ROTORV3_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::RotorV3 {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(read_quant_rotor_v3(st, idx, &shape)?),
                max_seq,
            })
        }
        // RotorV4 — K is QuantK (q8_0); V is QuantRotorV4.
        tag if tag == ROTORV4_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::RotorV4 {
                k: Some(read_quant_k(st, idx, "k", &shape)?),
                v: Some(read_quant_rotor_v4(st, idx, &shape)?),
                max_seq,
            })
        }
        // IsoSym3 — K is QuantIsoK3; V is QuantIsoV3.
        tag if tag == ISO_SYM_3_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::IsoSym3 {
                k: Some(read_quant_iso_k3(st, idx, &shape, max_seq)?),
                v: Some(read_quant_iso_v3(st, idx, &shape)?),
                max_seq,
            })
        }
        // IsoSym4 — K is QuantIsoK4; V is QuantIsoV4.
        tag if tag == ISO_SYM_4_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::IsoSym4 {
                k: Some(read_quant_iso_k4(st, idx, &shape, max_seq)?),
                v: Some(read_quant_iso_v4(st, idx, &shape)?),
                max_seq,
            })
        }
        // IsoKOnly3 — K-only payload; V is bf16 off-storage.
        tag if tag == ISO_K_ONLY_3_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::IsoKOnly3 {
                k: Some(read_quant_iso_k3(st, idx, &shape, max_seq)?),
                max_seq,
            })
        }
        // IsoKOnly4 — K-only payload; V is bf16 off-storage.
        tag if tag == ISO_K_ONLY_4_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::IsoKOnly4 {
                k: Some(read_quant_iso_k4(st, idx, &shape, max_seq)?),
                max_seq,
            })
        }
        // RotorSym3 — K is QuantRotorK3; V is QuantRotorV3.
        // QJL fields hydrated when layout tag carries `_qjl` suffix.
        tag if tag == ROTOR_SYM_3_LAYOUT_TAG || tag == ROTOR_SYM_3_QJL_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            let use_qjl = tag == ROTOR_SYM_3_QJL_LAYOUT_TAG;
            Ok(KvStorage::RotorSym3 {
                k: Some(read_quant_rotor_k3(st, idx, &shape, use_qjl)?),
                v: Some(read_quant_rotor_v3(st, idx, &shape)?),
                max_seq,
            })
        }
        // RotorSym4 — K is QuantRotorK4; V is QuantRotorV4.
        tag if tag == ROTOR_SYM_4_LAYOUT_TAG || tag == ROTOR_SYM_4_QJL_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            let use_qjl = tag == ROTOR_SYM_4_QJL_LAYOUT_TAG;
            Ok(KvStorage::RotorSym4 {
                k: Some(read_quant_rotor_k4(st, idx, &shape, use_qjl)?),
                v: Some(read_quant_rotor_v4(st, idx, &shape)?),
                max_seq,
            })
        }
        // RotorKOnly3 — K-only payload; V is bf16 off-storage.
        tag if tag == ROTOR_K_ONLY_3_LAYOUT_TAG || tag == ROTOR_K_ONLY_3_QJL_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            let use_qjl = tag == ROTOR_K_ONLY_3_QJL_LAYOUT_TAG;
            Ok(KvStorage::RotorKOnly3 {
                k: Some(read_quant_rotor_k3(st, idx, &shape, use_qjl)?),
                max_seq,
            })
        }
        // RotorKOnly4 — same shape as RotorKOnly3 with 4-bit codes.
        tag if tag == ROTOR_K_ONLY_4_LAYOUT_TAG || tag == ROTOR_K_ONLY_4_QJL_LAYOUT_TAG => {
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            let use_qjl = tag == ROTOR_K_ONLY_4_QJL_LAYOUT_TAG;
            Ok(KvStorage::RotorKOnly4 {
                k: Some(read_quant_rotor_k4(st, idx, &shape, use_qjl)?),
                max_seq,
            })
        }
        // RotorKAsym3 — K is QuantRotorK3 (optional QJL); V is affine QuantV
        // at `v_bits` / `v_group_size` parsed from the layout tag suffix
        // `_v{v_bits}g{v_group_size}`.
        tag if rotor_k_asym_3_prefix_match(tag).is_some() => {
            let (use_qjl, v_bits, v_group_size) = rotor_k_asym_3_prefix_match(tag)
                .ok_or_else(|| BlockIoError::Header(format!("bad rotor_k_asym_3 tag '{tag}'")))?;
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::RotorKAsym3 {
                k: Some(read_quant_rotor_k3(st, idx, &shape, use_qjl)?),
                v: Some(read_quant_v_bits(st, idx, &shape, v_bits)?),
                max_seq,
                v_bits,
                v_group_size,
            })
        }
        // RotorKAsym4 — mirror with rotor4 K.
        tag if rotor_k_asym_4_prefix_match(tag).is_some() => {
            let (use_qjl, v_bits, v_group_size) = rotor_k_asym_4_prefix_match(tag)
                .ok_or_else(|| BlockIoError::Header(format!("bad rotor_k_asym_4 tag '{tag}'")))?;
            let max_seq = geom_i32(geom, "max_seq")?;
            let shape = geom_shape(geom)?;
            Ok(KvStorage::RotorKAsym4 {
                k: Some(read_quant_rotor_k4(st, idx, &shape, use_qjl)?),
                v: Some(read_quant_v_bits(st, idx, &shape, v_bits)?),
                max_seq,
                v_bits,
                v_group_size,
            })
        }
        other => Err(BlockIoError::Header(format!("unknown layer tag '{other}'")).into()),
    }
}

/// Parse the suffix `_v{vb}g{vg}` of a rotor_k_asym_3 layout tag.
///
/// Returns `Some((use_qjl, v_bits, v_group_size))` when the tag matches the
/// `ROTOR_K_ASYM_3_LAYOUT_PREFIX` or `ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX` shape;
/// `None` otherwise. Mirrors the tag format emitted by [`write_layer`] above.
fn rotor_k_asym_3_prefix_match(tag: &str) -> Option<(bool, u8, u16)> {
    // ROTOR_K_ASYM_3_LAYOUT_PREFIX ("rotor_k_asym_3") is a strict prefix of
    // ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX ("rotor_k_asym_3_qjl"), so try the QJL
    // form first to avoid a false-positive match.
    if let Some(rest) = tag.strip_prefix(ROTOR_K_ASYM_3_QJL_LAYOUT_PREFIX) {
        return parse_v_suffix(rest).map(|(vb, vg)| (true, vb, vg));
    }
    if let Some(rest) = tag.strip_prefix(ROTOR_K_ASYM_3_LAYOUT_PREFIX) {
        return parse_v_suffix(rest).map(|(vb, vg)| (false, vb, vg));
    }
    None
}

/// Same parser as `rotor_k_asym_3_prefix_match` for rotor_k_asym_4.
fn rotor_k_asym_4_prefix_match(tag: &str) -> Option<(bool, u8, u16)> {
    // ROTOR_K_ASYM_4_LAYOUT_PREFIX ("rotor_k_asym_4") is a strict prefix of
    // ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX ("rotor_k_asym_4_qjl"), so try the QJL
    // form first to avoid a false-positive match.
    if let Some(rest) = tag.strip_prefix(ROTOR_K_ASYM_4_QJL_LAYOUT_PREFIX) {
        return parse_v_suffix(rest).map(|(vb, vg)| (true, vb, vg));
    }
    if let Some(rest) = tag.strip_prefix(ROTOR_K_ASYM_4_LAYOUT_PREFIX) {
        return parse_v_suffix(rest).map(|(vb, vg)| (false, vb, vg));
    }
    None
}

/// Parse `_v{v_bits}_g{v_group_size}` into `(v_bits, v_group_size)`. Mirrors
/// the KvQuant Display form for asymmetric rotor-K variants.
fn parse_v_suffix(rest: &str) -> Option<(u8, u16)> {
    let rest = rest.strip_prefix("_v")?;
    let (bits_s, group_s) = rest.split_once("_g")?;
    let bits: u8 = bits_s.parse().ok()?;
    let group: u16 = group_s.parse().ok()?;
    Some((bits, group))
}

/// Hydrate a `KvStorage::TurboSym3` from a `"tsym3_wht_3_3"`-tagged geometry.
/// Mirrors `read_tsym4` exactly but for 3-bit codes.
fn read_tsym3(st: &SafeTensors<'_>, idx: usize, geom: &str) -> Result<KvStorage> {
    let max_seq = geom_i32(geom, "max_seq")?;
    let shape = geom_shape(geom)?;
    Ok(KvStorage::TurboSym3 {
        k: Some(read_quant_k_turbo3(st, idx, &shape, max_seq)?),
        v: Some(read_quant_v(st, idx, &shape)?),
        max_seq,
    })
}

/// Hydrate a CPU-path `QuantKTurbo3` from serialized codes + scales tensors.
/// Mirrors `read_quant_k_turbo4` exactly: same on-disk layout (`u8` codes +
/// `f32` scales packed via `TurboBlocks`), 3-bit width.
///
/// Note: tensors loaded from safetensors are pre-materialized byte-buffers;
/// `.to_bytes()` is sufficient — no separate graph-eval step needed.
fn read_quant_k_turbo3(
    st: &SafeTensors<'_>,
    idx: usize,
    shape: &[i32],
    max_seq: i32,
) -> Result<QuantKTurbo3> {
    let codes_t = tensor_req(st, &format!("l{idx}.k.codes"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.k.scales"))?;
    let block = TurboBlocks {
        codes: codes_t.to_bytes()?,
        scales: bytes_to_f32(&scales_t.to_bytes()?),
        original_shape: shape4(shape),
        bits: 3,
    };
    Ok(QuantKTurbo3::from_cpu_blocks(
        vec![block],
        shape.to_vec(),
        3,
        max_seq,
    ))
}

/// Hydrate a `KvStorage::TurboSym4` from a `"tsym4_wht_4_4"`-tagged geometry.
/// Mirrors `read_quant_v` for the V side, and uses [`read_quant_k_turbo4`]
/// for the K side (identical TurboQuant codes/scales layout, different Rust
/// type).
fn read_tsym4(st: &SafeTensors<'_>, idx: usize, geom: &str) -> Result<KvStorage> {
    let max_seq = geom_i32(geom, "max_seq")?;
    let shape = geom_shape(geom)?;
    Ok(KvStorage::TurboSym4 {
        k: Some(read_quant_k_turbo4(st, idx, &shape)?),
        v: Some(read_quant_v(st, idx, &shape)?),
        max_seq,
    })
}

/// Hydrate a CPU-path `QuantKTurbo4` from serialized codes + scales tensors.
/// Mirrors `read_quant_v` exactly: same on-disk layout (`u8` codes + `f32`
/// scales packed via `TurboBlocks`).
fn read_quant_k_turbo4(st: &SafeTensors<'_>, idx: usize, shape: &[i32]) -> Result<QuantKTurbo4> {
    let codes_t = tensor_req(st, &format!("l{idx}.k.codes"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.k.scales"))?;
    codes_t.eval()?;
    scales_t.eval()?;
    let block = TurboBlocks {
        codes: codes_t.to_bytes()?,
        scales: bytes_to_f32(&scales_t.to_bytes()?),
        original_shape: shape4(shape),
        bits: 4,
    };
    Ok(QuantKTurbo4::from_cpu_blocks(
        vec![block],
        shape.to_vec(),
        4,
    ))
}

// ── Reconstruct QuantK / QuantV / QuantPlanarV (CPU-Vec form) ─────────────────

fn read_quant_k(st: &SafeTensors<'_>, idx: usize, side: &str, shape: &[i32]) -> Result<QuantK> {
    let codes = tensor_req(st, &format!("l{idx}.{side}.codes"))?;
    let scales = tensor_req(st, &format!("l{idx}.{side}.scales"))?;
    codes.eval()?;
    scales.eval()?;
    Ok(QuantK::from_cpu_parts(
        codes.to_bytes()?,
        bytes_to_f32(&scales.to_bytes()?),
        shape.to_vec(),
    ))
}

fn read_quant_v(st: &SafeTensors<'_>, idx: usize, shape: &[i32]) -> Result<QuantV> {
    read_quant_v_bits(st, idx, shape, 4)
}

fn read_quant_v_bits(st: &SafeTensors<'_>, idx: usize, shape: &[i32], bits: u8) -> Result<QuantV> {
    let codes = tensor_req(st, &format!("l{idx}.v.codes"))?;
    let scales = tensor_req(st, &format!("l{idx}.v.scales"))?;
    codes.eval()?;
    scales.eval()?;
    let block = TurboBlocks {
        codes: codes.to_bytes()?,
        scales: bytes_to_f32(&scales.to_bytes()?),
        original_shape: shape4(shape),
        bits,
    };
    Ok(QuantV::from_cpu_blocks(vec![block], shape.to_vec(), bits))
}

fn read_quant_planar_v(
    st: &SafeTensors<'_>,
    idx: usize,
    shape: &[i32],
    bits: u8,
) -> Result<QuantPlanarV> {
    let codes = tensor_req(st, &format!("l{idx}.v.codes"))?;
    let scales = tensor_req(st, &format!("l{idx}.v.scales"))?;
    let rotations = tensor_req(st, &format!("l{idx}.v.rotations"))?;
    codes.eval()?;
    scales.eval()?;
    rotations.eval()?;
    let block = PlanarBlocks {
        codes: codes.to_bytes()?,
        scales: bytes_to_f32(&scales.to_bytes()?),
        rotations: rotations.to_bytes()?,
        original_shape: shape4(shape),
        bits,
    };
    Ok(QuantPlanarV::from_cpu_blocks(
        vec![block],
        shape.to_vec(),
        bits,
    ))
}

/// Reconstruct a `QuantPlanarK` from serialized codes/scales/rotations tensors
/// under `l{idx}.k.*`. Mirrors `read_quant_planar_v` with only the
/// tensor-name prefix differing (`k` vs `v`).
fn read_quant_planar_k(st: &SafeTensors<'_>, idx: usize, shape: &[i32]) -> Result<QuantPlanarK> {
    let codes = tensor_req(st, &format!("l{idx}.k.codes"))?;
    let scales = tensor_req(st, &format!("l{idx}.k.scales"))?;
    let rotations = tensor_req(st, &format!("l{idx}.k.rotations"))?;
    codes.eval()?;
    scales.eval()?;
    rotations.eval()?;
    let block = PlanarBlocks {
        codes: codes.to_bytes()?,
        scales: bytes_to_f32(&scales.to_bytes()?),
        rotations: rotations.to_bytes()?,
        original_shape: shape4(shape),
        bits: 4,
    };
    Ok(QuantPlanarK::from_cpu_blocks(vec![block], shape.to_vec()))
}

/// Reconstruct a `QuantIsoV3` from the four serialized V-side tensors
/// (`codes_packed`, `scales`, `quaternions`, `norms`).
///
/// All four flat buffers were written in block-append order by
/// `write_quant_iso_v3` — we reconstruct a single `IsoBlocks` from the
/// concatenation and wrap it in a `QuantIsoV3` via `from_cpu_blocks`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape is a 4-element vec from geom_shape, caller verified length via the writer path"
)]
fn read_quant_iso_v3(st: &SafeTensors<'_>, idx: usize, shape: &[i32]) -> Result<QuantIsoV3> {
    let codes_t = tensor_req(st, &format!("l{idx}.v.codes_packed"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.v.scales"))?;
    let quats_t = tensor_req(st, &format!("l{idx}.v.quaternions"))?;
    let norms_t = tensor_req(st, &format!("l{idx}.v.norms"))?;
    // Force materialization before reading bytes.
    codes_t.eval()?;
    scales_t.eval()?;
    quats_t.eval()?;
    norms_t.eval()?;
    // Reinterpret the raw bytes of codes_packed as Vec<u32> (LE u32 words).
    let codes_bytes = codes_t.to_bytes()?;
    #[allow(
        clippy::unwrap_used,
        reason = "chunks_exact(4) guarantees each chunk has length 4; try_into from &[u8] of length 4 to [u8; 4] is infallible"
    )]
    let codes: Vec<u32> = codes_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scales = bytes_to_f32(&scales_t.to_bytes()?);
    let quaternions = bytes_to_f32(&quats_t.to_bytes()?);
    let norms = bytes_to_f32(&norms_t.to_bytes()?);
    // n_tokens: shape is [B, kv_h, S, D]; S = shape[2].
    let n_tokens = (shape[0] as usize) * (shape[1] as usize) * (shape[2].max(0) as usize);
    let block = IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens,
    };
    Ok(QuantIsoV3::from_cpu_blocks(vec![block], shape.to_vec()))
}

/// Reconstruct a `QuantIsoV4` from the four serialized V-side tensors.
/// Wire-format identical to `read_quant_iso_v3`; the codec differentiation
/// lives in the geometry tag.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape is a 4-element vec from geom_shape, caller verified length via the writer path"
)]
fn read_quant_iso_v4(st: &SafeTensors<'_>, idx: usize, shape: &[i32]) -> Result<QuantIsoV4> {
    let codes_t = tensor_req(st, &format!("l{idx}.v.codes_packed"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.v.scales"))?;
    let quats_t = tensor_req(st, &format!("l{idx}.v.quaternions"))?;
    let norms_t = tensor_req(st, &format!("l{idx}.v.norms"))?;
    // Force materialization before reading bytes (mlx Array::eval).
    codes_t.eval()?;
    scales_t.eval()?;
    quats_t.eval()?;
    norms_t.eval()?;
    let codes_bytes = codes_t.to_bytes()?;
    #[allow(
        clippy::unwrap_used,
        reason = "chunks_exact(4) guarantees each chunk has length 4; try_into from &[u8] of length 4 to [u8; 4] is infallible"
    )]
    let codes: Vec<u32> = codes_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scales = bytes_to_f32(&scales_t.to_bytes()?);
    let quaternions = bytes_to_f32(&quats_t.to_bytes()?);
    let norms = bytes_to_f32(&norms_t.to_bytes()?);
    let n_tokens = (shape[0] as usize) * (shape[1] as usize) * (shape[2].max(0) as usize);
    let block = IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens,
    };
    Ok(QuantIsoV4::from_cpu_blocks(vec![block], shape.to_vec()))
}

// K-side iso readers — see `read_quant_iso_k3` / `read_quant_iso_k4` below
// the rotor module for the K-axis IsoQuant reconstruction (mirror of
// `read_quant_iso_v3` / `read_quant_iso_v4`).

/// Reconstruct a `QuantRotorV3` from the four serialized V-side tensors
/// (`codes_packed`, `scales`, `norms`, `rotors`).
///
/// The static `rotors` table is part of the on-disk payload and is loaded as-is
/// (no re-derivation from the seed); this guarantees cross-restart identity even
/// if the global rotor seed constant ever changes.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape is a 4-element vec from geom_shape, caller verified length via the writer path"
)]
fn read_quant_rotor_v3(st: &SafeTensors<'_>, idx: usize, shape: &[i32]) -> Result<QuantRotorV3> {
    let codes_t = tensor_req(st, &format!("l{idx}.v.codes_packed"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.v.scales"))?;
    let norms_t = tensor_req(st, &format!("l{idx}.v.norms"))?;
    let rotors_t = tensor_req(st, &format!("l{idx}.v.rotors"))?;
    codes_t.eval()?;
    scales_t.eval()?;
    norms_t.eval()?;
    rotors_t.eval()?;
    let codes_bytes = codes_t.to_bytes()?;
    #[allow(
        clippy::unwrap_used,
        reason = "chunks_exact(4) guarantees each chunk has length 4; try_into from &[u8] of length 4 to [u8; 4] is infallible"
    )]
    let codes: Vec<u32> = codes_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scales = bytes_to_f32(&scales_t.to_bytes()?);
    let norms = bytes_to_f32(&norms_t.to_bytes()?);
    let rotors = bytes_to_f32(&rotors_t.to_bytes()?);
    let n_tokens = (shape[0] as usize) * (shape[1] as usize) * (shape[2].max(0) as usize);
    let block = RotorBlocks {
        codes,
        scales,
        norms,
        n_tokens,
    };
    // layer_idx is not persisted (the rotor table itself is; it is the
    // only deterministic dependency on layer_idx). Use 0 as a placeholder.
    Ok(QuantRotorV3::from_cpu_blocks(
        rotors,
        vec![block],
        shape.to_vec(),
        0,
    ))
}

/// Reconstruct a `QuantRotorV4` from the four serialized V-side tensors
/// (`codes_packed`, `scales`, `norms`, `rotors`).
///
/// Structurally identical to `read_quant_rotor_v3` — same tensor names and
/// LE-u32 byte packing. The only semantic difference is the codebook (16
/// centroids for rotor4 vs. 8 for rotor3); that difference lives in the
/// encode/decode functions, not in the on-disk binary layout, so the
/// deserializer is a direct mirror.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape is a 4-element vec from geom_shape, caller verified length via the writer path"
)]
fn read_quant_rotor_v4(st: &SafeTensors<'_>, idx: usize, shape: &[i32]) -> Result<QuantRotorV4> {
    let codes_t = tensor_req(st, &format!("l{idx}.v.codes_packed"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.v.scales"))?;
    let norms_t = tensor_req(st, &format!("l{idx}.v.norms"))?;
    let rotors_t = tensor_req(st, &format!("l{idx}.v.rotors"))?;
    codes_t.eval()?;
    scales_t.eval()?;
    norms_t.eval()?;
    rotors_t.eval()?;
    let codes_bytes = codes_t.to_bytes()?;
    #[allow(
        clippy::unwrap_used,
        reason = "chunks_exact(4) guarantees each chunk has length 4; try_into from &[u8] of length 4 to [u8; 4] is infallible"
    )]
    let codes: Vec<u32> = codes_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scales = bytes_to_f32(&scales_t.to_bytes()?);
    let norms = bytes_to_f32(&norms_t.to_bytes()?);
    let rotors = bytes_to_f32(&rotors_t.to_bytes()?);
    let n_tokens = (shape[0] as usize) * (shape[1] as usize) * (shape[2].max(0) as usize);
    let block = RotorBlocks {
        codes,
        scales,
        norms,
        n_tokens,
    };
    // layer_idx is not persisted (the rotor table itself is; it is the
    // only deterministic dependency on layer_idx). Use 0 as a placeholder.
    Ok(QuantRotorV4::from_cpu_blocks(
        rotors,
        vec![block],
        shape.to_vec(),
        0,
    ))
}

fn read_mixed(st: &SafeTensors<'_>, idx: usize, geom: &str) -> Result<KvStorage> {
    let max_seq = geom_i32(geom, "max_seq")?;
    let k_bits = geom_i32(geom, "k_bits")?;
    let v_bits = geom_i32(geom, "v_bits")?;
    let k_group_size = geom_i32(geom, "k_group_size")?;
    let v_group_size = geom_i32(geom, "v_group_size")?;
    let offset = geom_i32(geom, "offset")?;
    let rotate_k = geom_field(geom, "rotate_k") == Some("true");

    let keys = read_mixed_tuple(st, idx, "k")?;
    let values = read_mixed_tuple(st, idx, "v")?;

    let state = MixedKvState::from_parts(
        k_bits,
        v_bits,
        k_group_size,
        v_group_size,
        offset,
        keys,
        values,
        rotate_k,
    );
    Ok(KvStorage::Mixed { state, max_seq })
}

fn read_mixed_tuple(st: &SafeTensors<'_>, idx: usize, side: &str) -> Result<Option<MixedTuple>> {
    let Some(codes) = tensor_opt(st, &format!("l{idx}.{side}.codes"))? else {
        return Ok(None);
    };
    let scales = tensor_req(st, &format!("l{idx}.{side}.scales"))?;
    let biases = tensor_req(st, &format!("l{idx}.{side}.biases"))?;
    Ok(Some(MixedTuple {
        codes,
        scales,
        biases,
    }))
}

fn read_paged(st: &SafeTensors<'_>, idx: usize, geom: &str, device: Device) -> Result<KvStorage> {
    let max_seq = geom_i32(geom, "max_seq")?;
    let page_tokens = geom_i32(geom, "page_tokens")?;
    let total_tokens = geom_i32(geom, "total_tokens")?;
    let shape = geom_shape(geom)?;
    let quant_str = geom_field(geom, "quant").unwrap_or("k8v8");
    let quant: KvQuant = quant_str.parse().map_err(|_| {
        Error::from(BlockIoError::Header(format!(
            "bad paged quant '{quant_str}'"
        )))
    })?;

    let n_pages = if page_tokens > 0 {
        (((total_tokens + page_tokens - 1) / page_tokens).max(1)) as usize
    } else {
        1
    };

    let mut k = PagedKStorage::new(max_seq, page_tokens, n_pages);
    let k_codes = tensor_req(st, &format!("l{idx}.k.codes"))?;
    let k_scales = tensor_req(st, &format!("l{idx}.k.scales"))?;
    if total_tokens > 0 {
        k.append(
            &shape_with_seq(&shape, total_tokens),
            k_codes,
            k_scales,
            device,
        )?;
    }

    let (v_k8, v_planar) = if quant == KvQuant::Planar {
        let mut pv = PagedPlanarVStorage::new(max_seq, page_tokens, n_pages);
        let codes = tensor_req(st, &format!("l{idx}.v.codes"))?;
        let scales = tensor_req(st, &format!("l{idx}.v.scales"))?;
        let rot = tensor_req(st, &format!("l{idx}.v.rotations"))?;
        if total_tokens > 0 {
            pv.append(
                &shape_with_seq(&shape, total_tokens),
                codes,
                scales,
                rot,
                device,
            )?;
        }
        (None, Some(Box::new(pv)))
    } else {
        let bits = if matches!(quant, KvQuant::K8V8) { 8 } else { 4 };
        let mut pv = PagedVStorage::new(max_seq, page_tokens, n_pages, bits);
        let codes = tensor_req(st, &format!("l{idx}.v.codes"))?;
        let scales = tensor_req(st, &format!("l{idx}.v.scales"))?;
        if total_tokens > 0 {
            pv.append(&shape_with_seq(&shape, total_tokens), codes, scales, device)?;
        }
        (Some(Box::new(pv)), None)
    };

    Ok(KvStorage::Paged {
        quant,
        k: Some(k),
        v_k8,
        v_planar,
        max_seq,
    })
}

// ── Small helpers ───────────────────────────────────────────────────────────

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn shape4(shape: &[i32]) -> [i32; 4] {
    let mut s = [1i32; 4];
    for (i, &d) in shape.iter().take(4).enumerate() {
        s[i] = d;
    }
    s
}

/// Build a `[B, kv_h, seq, D]` shape from a recorded logical shape with `seq`
/// substituted at axis=2.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn shape_with_seq(shape: &[i32], seq: i32) -> Vec<i32> {
    let mut s = shape.to_vec();
    if s.len() == 4 {
        s[2] = seq;
    }
    s
}

// ── K-side IsoQuant readers ───────────────────────────────────────────────────

/// Reconstruct a `QuantIsoK3` from the four serialized K-side tensors
/// (`codes_packed`, `scales`, `quaternions`, `norms`). Mirror of
/// [`read_quant_iso_v3`] reading from `l{idx}.k.*` instead of `l{idx}.v.*`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape is a 4-element vec from geom_shape"
)]
fn read_quant_iso_k3(
    st: &SafeTensors<'_>,
    idx: usize,
    shape: &[i32],
    max_seq: i32,
) -> Result<QuantIsoK3> {
    let codes_t = tensor_req(st, &format!("l{idx}.k.codes_packed"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.k.scales"))?;
    let quats_t = tensor_req(st, &format!("l{idx}.k.quaternions"))?;
    let norms_t = tensor_req(st, &format!("l{idx}.k.norms"))?;
    materialize_iso_k_tensors(&codes_t, &scales_t, &quats_t, &norms_t)?;
    let codes_bytes = codes_t.to_bytes()?;
    #[allow(
        clippy::unwrap_used,
        reason = "chunks_exact(4) guarantees each chunk has length 4; try_into infallible"
    )]
    let codes: Vec<u32> = codes_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scales = bytes_to_f32(&scales_t.to_bytes()?);
    let quaternions = bytes_to_f32(&quats_t.to_bytes()?);
    let norms = bytes_to_f32(&norms_t.to_bytes()?);
    let n_tokens = (shape[0] as usize) * (shape[1] as usize) * (shape[2].max(0) as usize);
    let block = IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens,
    };
    Ok(QuantIsoK3::from_cpu_blocks(
        vec![block],
        shape.to_vec(),
        max_seq,
    ))
}

/// Reconstruct a `QuantIsoK4` (mirror of `read_quant_iso_k3`).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape is a 4-element vec from geom_shape"
)]
fn read_quant_iso_k4(
    st: &SafeTensors<'_>,
    idx: usize,
    shape: &[i32],
    max_seq: i32,
) -> Result<QuantIsoK4> {
    let codes_t = tensor_req(st, &format!("l{idx}.k.codes_packed"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.k.scales"))?;
    let quats_t = tensor_req(st, &format!("l{idx}.k.quaternions"))?;
    let norms_t = tensor_req(st, &format!("l{idx}.k.norms"))?;
    materialize_iso_k_tensors(&codes_t, &scales_t, &quats_t, &norms_t)?;
    let codes_bytes = codes_t.to_bytes()?;
    #[allow(
        clippy::unwrap_used,
        reason = "chunks_exact(4) guarantees each chunk has length 4; try_into infallible"
    )]
    let codes: Vec<u32> = codes_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scales = bytes_to_f32(&scales_t.to_bytes()?);
    let quaternions = bytes_to_f32(&quats_t.to_bytes()?);
    let norms = bytes_to_f32(&norms_t.to_bytes()?);
    let n_tokens = (shape[0] as usize) * (shape[1] as usize) * (shape[2].max(0) as usize);
    let block = IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens,
    };
    Ok(QuantIsoK4::from_cpu_blocks(
        vec![block],
        shape.to_vec(),
        max_seq,
    ))
}

/// Force materialisation of the four K-side iso tensors before byte-extract.
fn materialize_iso_k_tensors(
    codes_t: &Array,
    scales_t: &Array,
    quats_t: &Array,
    norms_t: &Array,
) -> Result<()> {
    codes_t.eval()?;
    scales_t.eval()?;
    quats_t.eval()?;
    norms_t.eval()?;
    Ok(())
}

// ── K-side rotor readers ──────────────────────────────────────────────────────

/// Force materialisation of the four base K-side rotor tensors.
fn materialize_rotor_k_tensors(
    codes_t: &Array,
    scales_t: &Array,
    norms_t: &Array,
    rotors_t: &Array,
) -> Result<()> {
    codes_t.eval()?;
    scales_t.eval()?;
    norms_t.eval()?;
    rotors_t.eval()?;
    Ok(())
}

/// Force materialisation of the three K-side QJL sideband tensors.
fn materialize_rotor_k_qjl_tensors(
    qjl_codes_t: &Array,
    qjl_norms_t: &Array,
    qjl_s_t: &Array,
) -> Result<()> {
    qjl_codes_t.eval()?;
    qjl_norms_t.eval()?;
    qjl_s_t.eval()?;
    Ok(())
}

/// Reconstruct a `QuantRotorK3` from the K-side serialized tensors. Mirror of
/// [`read_quant_rotor_v3`] reading from `l{idx}.k.*` plus optional QJL fields
/// (`qjl_codes` / `qjl_norms` / `qjl_s`) when `use_qjl`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape is a 4-element vec from geom_shape"
)]
fn read_quant_rotor_k3(
    st: &SafeTensors<'_>,
    idx: usize,
    shape: &[i32],
    use_qjl: bool,
) -> Result<QuantRotorK3> {
    let codes_t = tensor_req(st, &format!("l{idx}.k.codes_packed"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.k.scales"))?;
    let norms_t = tensor_req(st, &format!("l{idx}.k.norms"))?;
    let rotors_t = tensor_req(st, &format!("l{idx}.k.rotors"))?;
    materialize_rotor_k_tensors(&codes_t, &scales_t, &norms_t, &rotors_t)?;
    let codes_bytes = codes_t.to_bytes()?;
    #[allow(
        clippy::unwrap_used,
        reason = "chunks_exact(4) guarantees length 4; try_into infallible"
    )]
    let codes: Vec<u32> = codes_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scales = bytes_to_f32(&scales_t.to_bytes()?);
    let norms = bytes_to_f32(&norms_t.to_bytes()?);
    let rotors = bytes_to_f32(&rotors_t.to_bytes()?);
    let n_tokens = (shape[0] as usize) * (shape[1] as usize) * (shape[2].max(0) as usize);

    let (qjl_codes, qjl_norms, qjl_s_matrix) = if use_qjl {
        let qjl_codes_t = tensor_req(st, &format!("l{idx}.k.qjl_codes"))?;
        let qjl_norms_t = tensor_req(st, &format!("l{idx}.k.qjl_norms"))?;
        let qjl_s_t = tensor_req(st, &format!("l{idx}.k.qjl_s"))?;
        materialize_rotor_k_qjl_tensors(&qjl_codes_t, &qjl_norms_t, &qjl_s_t)?;
        (
            qjl_codes_t.to_bytes()?,
            bytes_to_f32(&qjl_norms_t.to_bytes()?),
            Some(bytes_to_f32(&qjl_s_t.to_bytes()?)),
        )
    } else {
        (Vec::new(), Vec::new(), None)
    };

    let block = RotorKBlocks {
        codes,
        scales,
        norms,
        qjl_codes,
        qjl_norms,
        n_tokens,
    };
    Ok(QuantRotorK3::from_cpu_blocks(
        rotors,
        qjl_s_matrix,
        vec![block],
        shape.to_vec(),
        0,
    ))
}

/// Reconstruct a `QuantRotorK4` (mirror of `read_quant_rotor_k3`).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape is a 4-element vec from geom_shape"
)]
fn read_quant_rotor_k4(
    st: &SafeTensors<'_>,
    idx: usize,
    shape: &[i32],
    use_qjl: bool,
) -> Result<QuantRotorK4> {
    let codes_t = tensor_req(st, &format!("l{idx}.k.codes_packed"))?;
    let scales_t = tensor_req(st, &format!("l{idx}.k.scales"))?;
    let norms_t = tensor_req(st, &format!("l{idx}.k.norms"))?;
    let rotors_t = tensor_req(st, &format!("l{idx}.k.rotors"))?;
    materialize_rotor_k_tensors(&codes_t, &scales_t, &norms_t, &rotors_t)?;
    let codes_bytes = codes_t.to_bytes()?;
    #[allow(
        clippy::unwrap_used,
        reason = "chunks_exact(4) guarantees length 4; try_into infallible"
    )]
    let codes: Vec<u32> = codes_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let scales = bytes_to_f32(&scales_t.to_bytes()?);
    let norms = bytes_to_f32(&norms_t.to_bytes()?);
    let rotors = bytes_to_f32(&rotors_t.to_bytes()?);
    let n_tokens = (shape[0] as usize) * (shape[1] as usize) * (shape[2].max(0) as usize);

    let (qjl_codes, qjl_norms, qjl_s_matrix) = if use_qjl {
        let qjl_codes_t = tensor_req(st, &format!("l{idx}.k.qjl_codes"))?;
        let qjl_norms_t = tensor_req(st, &format!("l{idx}.k.qjl_norms"))?;
        let qjl_s_t = tensor_req(st, &format!("l{idx}.k.qjl_s"))?;
        materialize_rotor_k_qjl_tensors(&qjl_codes_t, &qjl_norms_t, &qjl_s_t)?;
        (
            qjl_codes_t.to_bytes()?,
            bytes_to_f32(&qjl_norms_t.to_bytes()?),
            Some(bytes_to_f32(&qjl_s_t.to_bytes()?)),
        )
    } else {
        (Vec::new(), Vec::new(), None)
    };

    let block = RotorKBlocks {
        codes,
        scales,
        norms,
        qjl_codes,
        qjl_norms,
        n_tokens,
    };
    Ok(QuantRotorK4::from_cpu_blocks(
        rotors,
        qjl_s_matrix,
        vec![block],
        shape.to_vec(),
        0,
    ))
}

#[cfg(test)]
#[path = "block_io_identity_codec_tests.rs"]
mod identity_codec_tests;

#[cfg(test)]
#[path = "block_io_tests.rs"]
mod block_io_tests;
