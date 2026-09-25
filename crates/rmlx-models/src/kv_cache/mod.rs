//! KV cache primitive for incremental decoding.
//!
//! `KvCache` holds the accumulated K and V tensors for one attention layer.
//! Layout: `[B, kv_heads, S, head_dim]` (B=1 in all current use-cases).
//!
//! Usage:
//! - Pass `cache: Option<&mut KvCache>` into `Attention::forward`.
//! - When `None`: Attention recomputes K/V from scratch (existing behaviour).
//! - When `Some`: K/V are written into a pre-allocated buffer on every call and
//!   the filled slice is returned for SDPA. `cache.offset()` tells callers what
//!   position offset to pass for RoPE.
//!
//! The generator allocates one `Vec<KvCache>` (one entry per layer) and resets
//! all on every new request via `KvCache::reset`.
//!
//! # Quantization modes
//!
//! `auto` resolves to [`DEFAULT_KV_QUANT`], unquantised bf16, on every arch.
//! Every quantised codec is opt-in. `KvQuant::None` (`--kv-quant none`, alias
//! `bf16`) holds a `[B, kv_h, S, D]` bf16 K and V buffer per layer. On the
//! arches that call `context::resolve_context` and `with_max_seq_ceiling` —
//! gemma4, qwen3, qwen3_5_moe, qwen3_vl_moe — that buffer starts small and
//! grows toward the `--max-ctx` ceiling. laguna, gemma3, qwen2 and bitnet
//! construct at the resolved `--max-ctx`
//! (`max_ctx_override.unwrap_or(KV_MAX_SEQ_DEFAULT)`, no ceiling set), so a
//! large `--max-ctx` is an eager allocation there.
//!
//! `none` is a pure-bf16 control on every arch: [`kv_quant_for_layer`]'s
//! boundary promotion applies only to a base mode that quantizes, so no layer
//! of a `none` run holds a packed store.
//!
//! `KvQuant::K8V4` stores K as affine q8_0 (symmetric 8-bit,
//! `group_size=128`) and V with the TurboQuant 4-bit Lloyd-Max N(0,1)
//! codebook. The split is per axis (K vs V), not per layer index.
//!
//! `KvQuant::K8V8` stores both K and V with affine q8_0.
//!
//! `KvQuant::Planar` is K = q8_0, V = PlanarQuant 4-bit.
//!
//! # Qwen MoE
//!
//! Qwen MoE rejects every codec that stores K below 8 bits
//! ([`cache_type::validate_resolved`]; `docs/KV_LAYER_POLICY.md` § "Qwen MoE
//! rejects low-bit K codecs"). K8V8 and K8V4 keep K at 8 bits and pass.

#![allow(clippy::match_same_arms, clippy::trivially_copy_pass_by_ref)]
pub mod attention_dispatch;
pub mod cache_type;

use rmlx_kv_quant::KvQuant;

// Callers import the codec layer
// (`KvCache`, `LinearAttnCache`, `KvQuant`, `storage`, `paged`, `mixed_quant`,
// `rot_k`, `rotating`, `q8`, etc.) directly from `rmlx_kv_quant::*`, and the
// SSD-tier layer (`KvBlockReader`, `KvBlockWriter`, `SsdKvIndex`, `SsdSpiller`,
// `SsdHydrator`, `SpillJob`, `HydratedBlock`, `write_caches`, `set_ssd_*_hook`,
// `call_ssd_*_hook`) directly from `rmlx_kv_ssd::*`.

#[cfg(test)]
mod tests;

// ── Public constant ───────────────────────────────────────────────────────────

// `KV_MAX_SEQ_DEFAULT` lives in `rmlx_kv_quant::quant`; import it from
// `rmlx_kv_quant::KV_MAX_SEQ_DEFAULT`.

/// Default number of tail layers forced to the boundary floor
/// ([`boundary_floor`]) by [`kv_quant_for_layer`]. A setting, not a measured
/// optimum; see `docs/KV_LAYER_POLICY.md` § "Layer-adaptive overrides".
///
/// The value itself lives in `rmlx_core::kv_boundary` so `rmlx-metrics` can
/// recognise a `decode_config` that spells it out without a second copy; the
/// policy that reads it stays here.
pub const LAYER_ADAPTIVE_TAIL_N: usize = rmlx_core::kv_boundary::DEFAULT_BOUNDARY_TAIL_N;

/// Default number of head layers forced to the boundary floor
/// ([`boundary_floor`]) by [`kv_quant_for_layer`], at every context length. A
/// setting, not a measured optimum, like [`LAYER_ADAPTIVE_TAIL_N`].
///
/// The value lives in `rmlx_core::kv_boundary`, for the reason given on
/// [`LAYER_ADAPTIVE_TAIL_N`].
pub const LAYER_ADAPTIVE_HEAD_N: usize = rmlx_core::kv_boundary::DEFAULT_BOUNDARY_HEAD_N;

/// Layer-adaptive KV quantization.
///
/// Returns [`boundary_floor`] of a **quantizing** `base_quant` on:
/// - the **first** `head_n` layers (by absolute layer index); and
/// - the **last** `tail_n` layers (by absolute layer index).
///
/// Returns `base_quant` for all other layers, and for every layer when
/// `base_quant` quantizes neither side.
///
/// The counts are a setting, not a measured optimum. The promotion is
/// unconditional: it does not read the context length.
///
/// # Usage
///
/// Call during cache-vector construction inside each arch's `generate_greedy`:
///
/// ```ignore
/// let caches: Vec<KvCache> = (0..n_layers)
/// .map(|i| {
/// let q = kv_quant_for_layer(
/// i, n_layers, kv_quant,
/// LAYER_ADAPTIVE_TAIL_N, LAYER_ADAPTIVE_HEAD_N,
/// );
/// KvCache::with_quant_max_seq(q, max_seq)
/// })
/// .collect();
/// ```
///
/// General-purpose: works for any arch with any `KvQuant` base mode. The
/// override is by layer index, not by model name or KV mode — no hardcoded
/// model branches.
///
/// The promotion is a **quality floor for a codec that quantizes**: it buys
/// back loss the base codec introduced on the layers where that loss costs
/// most. A base mode that quantizes *neither* axis has no such loss, so the
/// promotion is skipped for it — see [`base_is_unquantized`]. That leaves it a
/// no-op for `KvQuant::K8V8` (already the target) and for `KvQuant::None`
/// (bf16 both sides), and in force for every quantizing base mode, including
/// the K-only families whose V side is bf16 but whose K side is below the
/// floor.
///
/// **The floor is 8 bits, not a codec switch** — which target delivers it is
/// [`boundary_floor`]'s decision, and for a base whose widths are parameters it
/// is that base's own 8-bit form.
///
/// `shares_kv` is the calling stack's cross-layer-KV topology — the same value
/// the arch passes to `KvCache::with_shares_kv`, and for every stack but Gemma4
/// that is the constructor default `false`. It is an input to the policy, not
/// decoration: on a stack that shares, `Mixed` / `RotK` keep their bf16 K/V
/// mirror for the consumer layers to read, and promoting in-family would buy a
/// packed store on top of a mirror that already decodes at model dtype. See
/// [`boundary_floor`].
///
/// Two per-arch filters can cancel the promotion independently of this
/// function: a windowed layer runs the bf16 rotating ring regardless of the
/// flag, and a shared-KV *consumer* layer (Gemma4 `num_kv_shared_layers`) owns
/// no cache to promote. Note that the second cancels nothing on the Gemma4
/// checkpoints whose `num_kv_shared_layers` is 0 — every layer owns a cache
/// there while `shares_kv` is still true for the stack, which is exactly the
/// case the argument above exists for. See `docs/KV_LAYER_POLICY.md`
/// § "Which codec the floor is".
///
/// When `head_n == 0` and `tail_n == 0`, `base_quant` is always returned.
pub fn kv_quant_for_layer(
    layer_idx: usize,
    n_layers: usize,
    base_quant: KvQuant,
    tail_n: usize,
    head_n: usize,
    shares_kv: bool,
) -> KvQuant {
    let is_tail = tail_n > 0 && layer_idx >= n_layers.saturating_sub(tail_n);
    let is_head = head_n > 0 && layer_idx < head_n;
    if (is_tail || is_head) && !base_is_unquantized(base_quant) {
        boundary_floor(base_quant, shares_kv)
    } else {
        base_quant
    }
}

/// The codec a boundary layer is promoted to, for a base that quantizes.
///
/// The promotion is a **quality floor at 8 bits**, not a codec switch. A base
/// whose widths are parameters carries its own 8-bit form, so the floor is
/// applied inside its family: same store, same group geometry, same K rotation,
/// both axes raised to 8 bits. A base whose width is baked into its variant has
/// no such form to raise to and falls back to [`KvQuant::K8V8`].
///
/// # The mirror is a floor of its own
///
/// `shares_kv` decides whether the in-family target is reachable at all. On a
/// cross-layer-KV stack the `Mixed` / `RotK` bf16 K/V mirror is what the
/// consumer layers read, so it survives the promotion — and a layer then holds
/// the packed store *plus* two full bf16 buffers. That is 24.50 bits per value
/// at group 64 against `K8V8`'s 16.00, while dropping the layer's own decode
/// from bf16 to 8-bit affine: more bytes and less precision, the inverse of
/// the promotion on both axes. The final arm therefore diverts any target that
/// still mirrors both axes back to `K8V8`, whose two mirrors *are* the layer's
/// numerics and are above any 8-bit floor.
///
/// The divert reads the target's own `feeds_bf16_*` predicates rather than
/// naming variants, so it is decided by the codec's decode disposition and
/// stays correct for a codec added later. For every non-parametric base the
/// target is already `K8V8` and the arm is an identity.
///
/// # Why the family has to be kept
///
/// `KvQuant::K8V8` does not materialise a packed store: its decode reads the
/// bf16 mirror on both axes, so `KvQuant::materialises_packed_store` is false
/// for it and a layer holding it holds two full bf16 buffers and nothing else —
/// **16 bits per value**, the same bytes as `KvQuant::None`. Sending a
/// parametric base there did not apply a floor, it exempted the layer from
/// quantization: `mixed_k8g64_v4g64` stores 8 + 32/64 bits on K and 4 + 32/64
/// on V, 6.50 bits per value, so the "8-bit" promotion *raised* those layers by
/// 2.46x. Over the standard 2 head + 8 tail layers that is the dominant term in
/// the codec's whole-cache rate and it does not shrink with context.
///
/// The parametric families read their own store at decode
/// (`KvQuant::decode_reads_packed_store` is true for `Mixed` and `RotK`), so
/// raising them in-family costs 8.50 bits per value at group 64 against the
/// 16.00 of the fallback — **on a stack that does not share K/V**; see the
/// mirror section above for the one that does.
///
/// # The floor is never free — it is cheaper in-family
///
/// The in-family target is a wider store than the base, and the extra width is
/// paid for: at `mixed_k4g64_v4g64` a boundary layer holds 8.50 bits per value
/// where the base holds 4.50. What in-family buys is the comparison against
/// the fallback, which would hold 16.00 bits per value on the same layer.
///
/// # What the fallback costs
///
/// Ten codecs materialise a packed store and eight of them take the `K8V8`
/// fallback, so for those eight the floor is paid at the full bf16 rate — the
/// promoted layer materialises no store at all and holds two bf16 buffers. The
/// `SideStore::IsoRing` four (`Iso{3,4}Sym`, `IsoKOnly{3,4}`) store 7.125 /
/// 8.125 bits per value on the ring at `head_dim = 128`, and the
/// `SideStore::Rotor` four (`Rotor{3,4}Sym`, `RotorKOnly{3,4}`) 8.75 / 9.75, so
/// `K8V8`'s 16.00 roughly doubles the bytes of their boundary layers. The cost
/// is bought deliberately: an SO(4)-rotated or rotor 3-/4-bit ring has no 8-bit
/// form, and the alternative to paying for the floor is not having one.
/// Neither group is diverted by the arm below — they do not mirror both axes.
///
/// How many layers this reaches is a property of the model, not of the policy:
/// a windowed layer runs the bf16 rotating ring whatever it is handed and a
/// shared-KV consumer layer owns no cache, so on `gemma-4-e2b` the promotion
/// reaches zero layers and no measurement taken there can say anything about
/// it. `--kv-boundary-layers` moves the counts for a run that wants to price
/// them; see [`KvBoundary`].
///
/// # Totality
///
/// Both rewritten variants are always valid. `validate_mixed_side` accepts 8
/// bits at every group size it accepts at all (32, 64, 128) and `RotK`'s V slot
/// is validated by that same function, so an 8-bit rewrite of an
/// already-validated base cannot produce a codec the store will not build.
/// Raising to 8 never lowers a side either — 8 is the widest width either
/// validator accepts.
///
/// The match is **exhaustive on purpose** (no wildcard), same reasoning as the
/// decode predicates on [`KvQuant`]: a new variant has to be *listed* here
/// before the crate compiles. Listing is not deciding, though — a new
/// parametric variant added to the fallback arm compiles cleanly and inherits
/// `K8V8`. What catches that is
/// `store_bearing_boundary_promotion_never_costs_more`, which sweeps every base in
/// `ALL_KV_QUANTS` that materialises a packed store, names the eight that take
/// the fallback deliberately, and fails on any other store-bearing base that
/// lands there.
fn boundary_floor(base_quant: KvQuant, shares_kv: bool) -> KvQuant {
    /// The width the boundary promotion floors both axes to.
    const FLOOR_BITS: u8 = 8;
    let target = match base_quant {
        KvQuant::Mixed {
            k_bits,
            v_bits,
            k_group_size,
            v_group_size,
        } => KvQuant::Mixed {
            k_bits: k_bits.max(FLOOR_BITS),
            v_bits: v_bits.max(FLOOR_BITS),
            k_group_size,
            v_group_size,
        },
        // `RotK`'s K is fixed at 8-bit/group-64 by `MixedKvState::new_rotated`
        // and is already at the floor; only its V carries a width.
        KvQuant::RotK {
            v_bits,
            v_group_size,
        } => KvQuant::RotK {
            v_bits: v_bits.max(FLOOR_BITS),
            v_group_size,
        },
        // Widths baked into the variant: no 8-bit form of their own family to
        // raise to, so the floor is `K8V8`.
        KvQuant::None
        | KvQuant::K8V4
        | KvQuant::K8V8
        | KvQuant::Planar
        | KvQuant::Planar3
        | KvQuant::PlanarK
        | KvQuant::K8VTurbo3
        | KvQuant::K8VTurbo3Tcq
        | KvQuant::K8VTurbo2
        | KvQuant::K8VTurbo2Tcq
        | KvQuant::TurboSym3
        | KvQuant::TurboSym4
        | KvQuant::Iso3
        | KvQuant::Iso4
        | KvQuant::Iso3Sym
        | KvQuant::Iso4Sym
        | KvQuant::IsoKOnly3
        | KvQuant::IsoKOnly4
        | KvQuant::Rotor3
        | KvQuant::Rotor4
        | KvQuant::Rotor3Sym
        | KvQuant::Rotor4Sym
        | KvQuant::RotorKOnly3
        | KvQuant::RotorKOnly4
        // The `RotorK*Asym` V width is a parameter, but its K is a 3-/4-bit
        // rotor that has no 8-bit form, so raising V alone would leave the
        // layer below the floor on K. It takes the fallback.
        | KvQuant::RotorK3Asym { .. }
        | KvQuant::RotorK4Asym { .. } => KvQuant::K8V8,
    };
    // A target that still reads a bf16 mirror on *both* axes decodes at model
    // dtype whatever its packed store holds, so the store buys no floor and is
    // charged on top of the mirrors. `K8V8` is those same two mirrors without
    // the store: fewer bytes and, being bf16, above any 8-bit floor. Read off
    // the target's own decode predicates rather than a variant list, so a codec
    // whose mirror survives promotion is diverted here without being named.
    if target.feeds_bf16_k_at_decode(shares_kv) && target.feeds_bf16_v_at_decode(shares_kv) {
        KvQuant::K8V8
    } else {
        target
    }
}

/// The **nominal** per-layer codec vector for a model of `n_layers` layers at
/// base codec `base` — one entry per decoder layer, `kv_quant_for_layer` at the
/// standard [`LAYER_ADAPTIVE_TAIL_N`] / [`LAYER_ADAPTIVE_HEAD_N`] constants.
///
/// This is the **one** producer of that vector. Every consumer calls it rather
/// than re-running the loop: each arch's cache-construction loop
/// (`caches[i]` is built at `quants[i]`), the SSD attach that folds the vector
/// into the layout key, and the per-request prompt-cache seed. Two of those
/// three describe what the third builds, so a second copy of the loop is not a
/// duplication of style — it is a way for the description to stop matching the
/// thing described, silently, the next time the constants or the rule move.
/// `scripts/check_kv_layer_quants.sh` (in `make ci`) keeps it that way by
/// failing on a direct [`kv_quant_for_layer`] call outside this module.
///
/// `shares_kv` is the stack's cross-layer-KV topology and must be the value the
/// arch passes to `KvCache::with_shares_kv` — Gemma4's
/// `SHARES_KV_ACROSS_LAYERS`, the constructor default `false` everywhere else.
/// It changes the vector's contents (see [`boundary_floor`]), so a caller that
/// seeds the prompt cache or salts an SSD layout key must pass the same value
/// the caller that builds the caches passed, or the key describes a mixture
/// nothing runs.
///
/// **Nominal, not effective.** Two per-arch filters can make an entry a no-op
/// on the built cache and are deliberately *not* folded in here, because they
/// are properties of one layer's geometry rather than of the codec policy: a
/// windowed layer runs the bf16 rotating ring whatever codec it is handed, and
/// a shared-KV *consumer* layer (Gemma4 `num_kv_shared_layers`) owns no cache
/// at all. `shares_kv` is not one of those — it is a whole-stack property that
/// changes which codec the policy picks, not whether the picked one is used. A
/// consumer that needs the effective codec of a *built* cache must read the
/// cache, not this vector.
///
/// The head/tail counts come from [`active_kv_boundary`], not from the
/// constants directly, so `--kv-boundary-layers` moves every consumer at once.
#[must_use]
pub fn kv_layer_quants(n_layers: usize, base: KvQuant, shares_kv: bool) -> Vec<KvQuant> {
    kv_layer_quants_at(n_layers, base, shares_kv, active_kv_boundary())
}

/// [`kv_layer_quants`] at an explicit boundary rather than the installed one.
///
/// Private on purpose: the public producer is the one that reads the process
/// configuration, and a second public entry point would let a caller build a
/// cache stack at one boundary while the SSD layout key and the prompt-cache
/// seed describe another. Tests use it to compare boundaries inside one
/// process, which the install-once accessor cannot do.
fn kv_layer_quants_at(
    n_layers: usize,
    base: KvQuant,
    shares_kv: bool,
    boundary: KvBoundary,
) -> Vec<KvQuant> {
    (0..n_layers)
        .map(|i| {
            kv_quant_for_layer(
                i,
                n_layers,
                base,
                boundary.tail_n,
                boundary.head_n,
                shares_kv,
            )
        })
        .collect()
}

/// How many head and tail layers [`kv_layer_quants`] holds at the boundary
/// floor.
///
/// `head_n = 0, tail_n = 0` turns the promotion off entirely: every layer runs
/// the base codec.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed two-field pair — a head count and a tail count are the whole contract, and a caller that names both has named the boundary"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBoundary {
    /// Number of leading layers held at [`boundary_floor`].
    pub head_n: usize,
    /// Number of trailing layers held at [`boundary_floor`].
    pub tail_n: usize,
}

impl Default for KvBoundary {
    fn default() -> Self {
        Self {
            head_n: LAYER_ADAPTIVE_HEAD_N,
            tail_n: LAYER_ADAPTIVE_TAIL_N,
        }
    }
}

/// Largest head or tail count [`KvBoundary::parse`] accepts.
///
/// No shipped decoder stack is anywhere near this deep; the bound exists so a
/// mistyped count is refused at the CLI rather than silently flooring every
/// layer of every model.
const MAX_BOUNDARY_LAYERS: usize = 512;

/// Why a `--kv-boundary-layers` value was refused.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum KvBoundaryParseError {
    /// The value was not two comma-separated fields.
    #[error("expected '<head>,<tail>', got '{0}'")]
    Shape(String),
    /// One of the two fields was not a non-negative integer.
    #[error("'{field}' is not a non-negative integer (in '{value}')")]
    NotANumber {
        /// The offending field as written.
        field: String,
        /// The whole value the operator passed.
        value: String,
    },
    /// One of the two counts was above [`MAX_BOUNDARY_LAYERS`].
    #[error("{field}={count} exceeds the {max}-layer maximum")]
    TooLarge {
        /// Which side was too large (`head` or `tail`).
        field: &'static str,
        /// The count that was refused.
        count: usize,
        /// The accepted maximum.
        max: usize,
    },
}

impl KvBoundary {
    /// Parse a `<head>,<tail>` pair.
    ///
    /// Both counts are required and both may be `0`; `0,0` is the valid
    /// spelling of "no boundary promotion at all".
    pub fn parse(value: &str) -> Result<Self, KvBoundaryParseError> {
        let mut fields = value.split(',');
        let (Some(head), Some(tail), None) = (fields.next(), fields.next(), fields.next()) else {
            return Err(KvBoundaryParseError::Shape(value.to_string()));
        };
        let parse_one = |s: &str| -> Result<usize, KvBoundaryParseError> {
            s.trim()
                .parse::<usize>()
                .map_err(|_| KvBoundaryParseError::NotANumber {
                    field: s.to_string(),
                    value: value.to_string(),
                })
        };
        let head_n = parse_one(head)?;
        let tail_n = parse_one(tail)?;
        for (field, count) in [("head", head_n), ("tail", tail_n)] {
            if count > MAX_BOUNDARY_LAYERS {
                return Err(KvBoundaryParseError::TooLarge {
                    field,
                    count,
                    max: MAX_BOUNDARY_LAYERS,
                });
            }
        }
        Ok(Self { head_n, tail_n })
    }

    /// The `docs/METRICS_SCHEMA.md` §3.2 `decode_config` terms for this boundary,
    /// or `None` when it is the shipped default.
    ///
    /// `None` is what keeps a default run's cell identical to every row
    /// recorded before the flag existed; only a run that moved the setting off
    /// its default gets a cell of its own.
    #[must_use]
    pub fn decode_config(self) -> Option<String> {
        (self != Self::default()).then(|| {
            use rmlx_core::kv_boundary::{BOUNDARY_HEAD_KEY, BOUNDARY_TAIL_KEY};
            format!(
                "{BOUNDARY_HEAD_KEY}={},{BOUNDARY_TAIL_KEY}={}",
                self.head_n, self.tail_n
            )
        })
    }
}

static KV_BOUNDARY: std::sync::OnceLock<KvBoundary> = std::sync::OnceLock::new();

/// Set the first time [`active_kv_boundary`] answers. See
/// [`install_kv_boundary`] for what it protects.
static KV_BOUNDARY_READ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Why an [`install_kv_boundary`] call was refused.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum KvBoundaryInstallError {
    /// Something already built a per-layer vector at the default.
    #[error(
        "the boundary was already read (something built a per-layer vector, an SSD layout key \
         or a prompt-cache seed at {read:?}) before this install of {requested:?}; those \
         describe a mixture this process would no longer build"
    )]
    AlreadyRead {
        /// The value the earlier reader was answered with.
        read: KvBoundary,
        /// The value this call tried to install.
        requested: KvBoundary,
    },
    /// A second install disagrees with the first.
    #[error("the boundary is already installed as {installed:?}; refusing to change it to {requested:?}")]
    AlreadyInstalled {
        /// The value in force.
        installed: KvBoundary,
        /// The value this call tried to install.
        requested: KvBoundary,
    },
}

/// Install the process-global [`KvBoundary`] read by [`kv_layer_quants`].
///
/// Called once at command startup, before any model loads; `None` means the
/// flag was not passed and the default applies. Installing the same value
/// twice is a no-op.
///
/// The boundary is process-global rather than a parameter because three places
/// must agree on it — the arch loops that build the caches, the SSD layout key
/// and the per-request prompt-cache seed — and a per-call parameter is exactly
/// how two of them come to describe a mixture the third does not build.
///
/// # Why a read latches it
///
/// [`active_kv_boundary`] answers with the default when nothing is installed,
/// so a read before the install is **indistinguishable from a default run**.
/// Move an eager preload above the install and it builds its caches at the
/// default while every key written afterwards describes the requested
/// boundary: no error, no warning, and a stored block handed to a request whose
/// layers were built differently. That is the same shape as the cache-seed
/// drift this vector's single-producer rule exists to prevent, so a late
/// install is an error rather than a warning — the caller cannot repair it
/// afterwards, and continuing means running a configuration nobody asked for.
///
/// # Errors
///
/// [`KvBoundaryInstallError::AlreadyRead`] when the value has already been
/// handed out, and [`KvBoundaryInstallError::AlreadyInstalled`] when a
/// different value is already in force.
pub fn install_kv_boundary(boundary: Option<KvBoundary>) -> Result<(), KvBoundaryInstallError> {
    let requested = boundary.unwrap_or_default();
    if let Some(&installed) = KV_BOUNDARY.get() {
        if installed == requested {
            return Ok(());
        }
        return Err(KvBoundaryInstallError::AlreadyInstalled {
            installed,
            requested,
        });
    }
    // Checked BEFORE storing: a refused install that had already written the
    // value would govern every later read, which is the failure this refusal
    // exists to prevent rather than a milder version of it. Nothing is
    // installed at this point, so the earlier reader was answered with the
    // default.
    if KV_BOUNDARY_READ.load(std::sync::atomic::Ordering::Acquire) {
        return Err(KvBoundaryInstallError::AlreadyRead {
            read: KvBoundary::default(),
            requested,
        });
    }
    if KV_BOUNDARY.set(requested).is_err() {
        let installed = KV_BOUNDARY.get().copied().unwrap_or_default();
        if installed == requested {
            return Ok(());
        }
        return Err(KvBoundaryInstallError::AlreadyInstalled {
            installed,
            requested,
        });
    }
    tracing::info!(
        head_n = requested.head_n,
        tail_n = requested.tail_n,
        source = if requested == KvBoundary::default() {
            "default"
        } else {
            "cli"
        },
        "kv boundary-layer counts installed"
    );
    Ok(())
}

/// The active [`KvBoundary`] — the installed one, or the default when
/// [`install_kv_boundary`] has not been called (tests / unit paths).
///
/// Latches: after this answers, a later install is refused rather than
/// silently arriving too late to govern what was already built.
#[must_use]
pub fn active_kv_boundary() -> KvBoundary {
    KV_BOUNDARY_READ.store(true, std::sync::atomic::Ordering::Release);
    KV_BOUNDARY.get().copied().unwrap_or_default()
}

/// Code width [`KvQuant::approx_code_bits`] reports for a side that is kept at
/// model dtype (bf16) instead of quantized.
const MODEL_DTYPE_CODE_BITS: u32 = 16;

/// Does `base_quant` keep **both** K and V at model dtype?
///
/// Keyed off the codec's own code widths, never off a codec name or an arch:
/// any mode that quantizes nothing reports `MODEL_DTYPE_CODE_BITS` on both
/// sides ([`KvQuant::None`] today) and any mode that quantizes at least one
/// side reports that side below it — including the K-only families
/// (`PlanarK`, `IsoKOnly*`, `RotorKOnly*`), which keep a bf16 V but a 3-/4-bit
/// K and therefore still have K-side loss for the boundary promotion to
/// recover.
///
/// Used by [`kv_quant_for_layer`] to decide whether the boundary promotion has
/// anything to buy. Promoting an unquantized base allocates a packed q8_0 K+V
/// store on top of the model-dtype buffers the layer already holds and can
/// only *lower* its precision — the inverse of what the override exists for.
///
/// The test is deliberately "quantizes nothing", not "at least as wide as the
/// K8V8 floor on both axes": the second would also divert equal-width bases of
/// a different family (`mixed_k8g64_v8g64`, `rot_k_v8g64`) out of the
/// promotion, which is a separate question about codecs that genuinely read
/// their packed store and is not decided here.
fn base_is_unquantized(base_quant: KvQuant) -> bool {
    let (k_bits, v_bits) = base_quant.approx_code_bits();
    k_bits >= MODEL_DTYPE_CODE_BITS && v_bits >= MODEL_DTYPE_CODE_BITS
}

/// Per-layer attributes for the resolve-time net-benefit check.
///
/// Model-agnostic: every field is a layer geometry attribute the arch parser
/// already knows. No arch name is carried — the check keys off geometry +
/// codec only, so any architecture interleaving windowed + global attention is
/// covered identically.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct KvLayerShape {
    /// Per-KV-head dimension (e.g. 256 for Gemma4, 128 for Bonsai/Qwen).
    pub head_dim: u64,
    /// Number of KV heads (GQA group count).
    pub kv_heads: u64,
    /// Sliding-window size in tokens for windowed layers, or `None` for a
    /// global (full-attention) layer.
    pub window: Option<u64>,
}

/// Emit one structured `warn!` when the resolved KV codec is estimated to
/// **increase** resident KV versus plain bf16 on the active layer mix.
///
/// Why this can happen, generally: a quantized codec can keep a warm-TTFT bf16
/// decode seed (`decode_fp16_k/v`) alongside its packed codes and per-group
/// scales on every **global** layer (see
/// [`rmlx_kv_quant::KvQuant::feeds_bf16_k_at_decode`]). Where it does, those
/// codes and scales are pure overhead on top of a buffer the same size as bf16
/// and the codec is net-negative. A codec whose decode reads only the mirror
/// builds no store at all
/// ([`rmlx_kv_quant::KvQuant::materialises_packed_store`]), so it holds exactly
/// the bf16 bytes and never reaches this warn.
/// **Windowed layers always run the bf16 rotating ring regardless of the flag**
/// (`RotatingKVCache.to_quantized` raises `NotImplementedError` in mlx-lm;
/// rMLX matches it), so they are a no-op for the codec and contribute zero to
/// the delta — the net-negative is a property of the global layers only.
///
/// This is advisory: the codec is **not** changed (keeping it is the operator's
/// explicit choice, and forcing bf16 globally would change numerics). The warn
/// gives the operator the byte math so they can pick `--kv-quant none` when the
/// codec buys nothing at their context size.
///
/// Keyed on `(KvLayerShape, KvQuant, eff_seq, shares_kv)` — no arch *name*, but
/// `shares_kv` is the caller's own cross-layer-KV topology, the same flag its
/// caches carry (see [`rmlx_kv_quant::KvCache::shares_kv`]). It is what decides
/// whether `Mixed` / `RotK` are estimated with their bf16 mirror; assuming one
/// on a stack that keeps none would over-report those codecs by two full bf16
/// buffers per layer and warn about bytes nothing allocates. Call once per
/// request from the arch `generate` path after the codec is resolved and the
/// layer mix is known.
///
/// `layers` is the per-layer shape vector (one entry per decoder layer);
/// `eff_seq` is the effective prompt+generate length the global layers will
/// hold (the resolved `--max-ctx` ceiling or the prompt length — either is a
/// fine estimate for the sign of the saving).
///
/// The emitted byte count is an estimate, so read the sign and not the
/// magnitude. `KvQuant::estimated_resident_bytes_per_layer` sizes each side
/// byte-for-byte against its store, but it does not model page rounding, the
/// static rotation tables, or GPU/CPU residual coexistence — and it
/// under-reports an iso codec whenever the CPU blocks `exit_prefill` built are
/// what a layer holds. That happens two ways, and only one of them ends: it is
/// a **window** on a layer the fused decode path serves, closing at the first
/// fused decode step that drops the blocks; it is **permanent** on a layer
/// whose shape that path's gate rejects (batch > 1, or a `head_dim` that is not
/// a power of two at most 512 — `head_dim = 80` qualifies), because the ring is
/// then never allocated and there is nothing to drop. The blocks are 6.07x the
/// ring at `head_dim = 128` — 43.25 bits per value against the ring's 7.125 —
/// because the blocks are a host `Vec` form that carries an `f32` scale, a
/// replicated `f32` quaternion per group and an `f32` norm, none of which the
/// ring stores. The sign is what the warning is for; a reader must not size a
/// buffer from the number.
pub fn warn_if_kv_codec_net_negative(
    quant: KvQuant,
    layers: &[KvLayerShape],
    eff_seq: u64,
    shares_kv: bool,
) {
    let (total_saving, n_global, n_windowed) =
        kv_codec_net_saving_total(quant, layers, eff_seq, shares_kv);
    if total_saving < 0 {
        tracing::warn!(
            kv_quant = %quant,
            eff_seq,
            n_global,
            n_windowed,
            est_extra_bytes = -total_saving,
            "KV codec increases resident KV vs bf16 on this layer mix — the per-global-layer warm-TTFT bf16 seed plus codec scales exceed the bytes saved at this context; windowed layers already run bf16 and are unaffected. Read the sign, not the magnitude: the estimator under-reports an iso codec while a layer holds the CPU blocks the prefill encode built — until the first fused decode step drops them, or for the whole request on a layer the fused path's shape gate rejects (batch > 1, or a head_dim that is not a power of two at most 512), where they are never dropped. Consider --kv-quant none if memory is the goal."
        );
    }
}

/// Pure decision behind [`warn_if_kv_codec_net_negative`]: total estimated
/// net byte saving across the layer mix (negative = codec costs more than
/// bf16), plus the global / windowed layer counts.
///
/// Split out so the sign decision is unit-testable without a tracing
/// subscriber. `KvQuant::None` (or an empty layer list) returns `(0, _, _)` —
/// bf16 is never net-negative against itself.
///
/// Keyed entirely on `(KvLayerShape, KvQuant, eff_seq)`; model-agnostic.
#[must_use]
pub fn kv_codec_net_saving_total(
    quant: KvQuant,
    layers: &[KvLayerShape],
    eff_seq: u64,
    shares_kv: bool,
) -> (i64, usize, usize) {
    if matches!(quant, KvQuant::None) || layers.is_empty() {
        return (0, 0, 0);
    }
    let mut total_saving: i64 = 0;
    let mut n_global: usize = 0;
    let mut n_windowed: usize = 0;
    for l in layers {
        let is_windowed = l.window.is_some();
        if is_windowed {
            n_windowed += 1;
        } else {
            n_global += 1;
        }
        let seq = match l.window {
            Some(w) => eff_seq.min(w),
            None => eff_seq,
        };
        total_saving = total_saving.saturating_add(quant.estimated_net_saving_per_layer(
            seq,
            l.head_dim,
            l.kv_heads,
            is_windowed,
            shares_kv,
        ));
    }
    (total_saving, n_global, n_windowed)
}

// The `--max-ctx` virtual ceiling (issue #25) is resolved by
// [`crate::context::resolve_context`], the one producer of every context
// bound in the tree. `initial_max_seq` is the small lazy start
// ([`rmlx_kv_quant::KV_MAX_SEQ_DEFAULT`], capped by the ceiling); the codec's
// power-of-two grow path ([`rmlx_kv_quant::KvCache::ensure_prefill_capacity`])
// takes it up to the ceiling as the prompt fills. Wire it in an arch
// `generate` as:
//
// ```ignore
// let ctx = resolve_context(&limits, max_ctx_override)?;
// KvCache::with_quant_max_seq(q, ctx.initial_max_seq).with_max_seq_ceiling(ctx.ceiling)
// ```

// ── The auto default ──────────────────────────────────────────────────────────

/// The KV codec `--kv-quant auto` resolves to — for every architecture, every
/// checkpoint and every prompt length.
///
/// **Unquantised bf16.** This is the one producer of the auto default; a caller
/// that needs "whatever auto picks" reads this constant and nothing else. There
/// is no per-arch table and no per-context re-selection behind it, so an
/// operator who passes no flag gets the same cache the CLI, the server, the
/// image branch and every speculative drafter build.
///
/// # Why bf16 and not a quantised codec
///
/// * **The bf16-mirror codecs cost bytes they do not save.** `K8V8`, `K8V4`,
///   `Planar`, `Planar3`, `PlanarK`, the turbo and iso/rotor asymmetric
///   families all decode off the bf16 mirror `exit_prefill` materialises and
///   never read their packed store, so that store is not built
///   ([`rmlx_kv_quant::KvQuant::materialises_packed_store`]). Their resident KV
///   is therefore equal to bf16's, and so is their output at temp=0.
/// * **A store-reading codec does not beat bf16 at decode.** At
///   `heads_per_kv ≥ 4` no fused decode over a current store can beat the bf16
///   `sdpa_vector` path; see `docs/KV_QUANT.md` § "Fused flash-decode over a
///   quant store — the break-even condition". A codec that holds less
///   resident KV is an opt-in memory setting.
///
/// Every codec stays selectable with an explicit `--kv-quant` / `--cache-type-*`
/// / `--kv-preset`; only what `auto` resolves to is fixed here.
///
/// See `docs/KV_QUANT.md` § "The auto default".
pub const DEFAULT_KV_QUANT: KvQuant = KvQuant::None;

/// Layer-name fuzzy-match helper.
///
/// **No in-tree caller yet. Consumed by future codec encode paths.**
/// Returns `None` if no calibration is attached to the builder.
///
/// Looks up `layer_key` in a `KvCalibration` layers map using case-insensitive
/// comparison on the first three dot-separated components (the "dotted prefix").
///
/// # Examples
///
/// ```text
/// query: "model.layers.0.self_attn"  →  matches "MODEL.LAYERS.0.SELF_ATTN"
/// query: "model.layers.0"            →  matches key "model.layers.0.self_attn"
///                                        via 3-prefix (component count = 3, passes guard)
/// query: "model.layers.0.self_attn.k_proj" → 3-prefix matches
///        "model.layers.0.self_attn" key (returns that entry)
/// ```
///
/// Returns `None` when no fuzzy match is found.
pub fn lookup_layer_calibration<'c>(
    calib: &'c rmlx_loader::KvCalibration,
    layer_key: &str,
) -> Option<&'c rmlx_loader::LayerCalib> {
    // Fast path: exact match first.
    if let Some(entry) = calib.layers.get(layer_key) {
        return Some(entry);
    }

    // Fuzzy path: case-insensitive 3-dotted-component prefix comparison.
    // Take the first 3 dot-components of the query key (e.g. "model.layers.0")
    // and compare them case-insensitively against the first 3 components of
    // each map key.
    let query_prefix: Vec<&str> = layer_key.splitn(4, '.').take(3).collect();
    if query_prefix.len() < 3 {
        return None;
    }
    for (k, v) in &calib.layers {
        let key_prefix: Vec<&str> = k.splitn(4, '.').take(3).collect();
        if key_prefix.len() >= 3
            && key_prefix
                .iter()
                .zip(query_prefix.iter())
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
        {
            return Some(v);
        }
    }
    None
}

// ── KvCacheBuilder ────────────────────────────────────────────────────────────

/// Builder for KV-cache construction with optional calibration attachment.
///
/// Gains a `calibration` field so the per-arch generate path can pass
/// per-layer high-precision indices to codec storage during cache construction.
/// `with_calibration()` stores the calibration.
///
/// **No in-tree caller yet.** The surface is wired end-to-end
/// (loader → `ModelLoadConfig` → `KvCacheBuilder`) but per-arch construction
/// code that calls `with_calibration` is deferred.
/// Codec behavior is unchanged — indices are stored but not yet consumed.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed builder — field set is the complete calibration-attach contract; adding a field requires a review of all call sites"
)]
#[derive(Debug, Default)]
pub struct KvCacheBuilder {
    /// Optional KV calibration to forward to codec storage.
    ///
    /// `None` (the default) = no calibration; codec behavior unchanged.
    /// `Some(calib)` = calibration present; per-layer lookup via
    /// [`lookup_layer_calibration`] during cache construction.
    pub calibration: Option<rmlx_loader::KvCalibration>,
}

impl KvCacheBuilder {
    /// Attach an optional [`KvCalibration`] to this builder.
    ///
    /// Stores the calibration for forwarding to codec storage
    /// during per-layer cache construction. Returns `self` for chaining.
    ///
    /// Codec behavior is not changed — the indices are stored for future
    /// consumption by encode/decode paths.
    ///
    /// [`KvCalibration`]: rmlx_loader::KvCalibration
    #[must_use]
    pub fn with_calibration(mut self, calib: Option<rmlx_loader::KvCalibration>) -> Self {
        self.calibration = calib;
        self
    }
}

// The SSD-tier process-global event recorder + 5 Prometheus observation hook
// globals (`set_ssd_event_recorder`,
// `set_ssd_{spill_prom,hydrate_prom,bytes_used,evict_total}_hook`, plus the
// internal `call_*` accessors) live in `rmlx_kv_ssd::hooks`. Cross-crate
// callers in `rmlx-server` / `rmlx-cli` import them from
// `rmlx_kv_ssd::set_ssd_*_hook`.

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use cache_type::{
    parse as parse_cache_type_str, resolve as resolve_cache_type,
    validate_resolved as validate_resolved_kv_quant, CacheType, CacheTypeSpec,
    ParseError as CacheTypeParseError, ResolveError, ResolverContext,
};
// `KvCache` and `LinearAttnCache` re-exports were dropped — every
// caller imports them directly from `rmlx_kv_quant::*`.
// `DEFAULT_KV_QUANT`, `kv_quant_for_layer` and `LAYER_ADAPTIVE_TAIL_N` are
// defined inline above.
// No re-export needed — already top-level items in this module.
