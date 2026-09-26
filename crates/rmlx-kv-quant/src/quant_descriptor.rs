//! The per-codec facts of [`KvQuant`], stated once per variant.
//!
//! [`KvQuant::descriptor`] is the one place that classifies a codec. The
//! public predicates on `KvQuant` read their answer from it. Each arm names
//! one variant and states every fact as a literal, so a new variant does not
//! compile until each of its facts is written down. The census
//! (`scripts/kv_update_census.py`) refuses a struct-update base (`..X`) in the
//! fn body: a row copied from a neighbour would inherit facts nobody stated.
//!
//! `codec_facts_tests.rs` holds the same facts in a table written apart from
//! this file, so a wrong fact here for an existing codec turns a cell red.

use super::{KvQuant, SideStore};

/// When decode reads the bf16 mirror of one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MirrorRule {
    Never,
    Always,
    /// Only on a cache that shares its K/V with other layers
    /// ([`crate::KvCache::shares_kv`]).
    WhenSharesKv,
}

impl MirrorRule {
    pub(super) fn applies(self, shares_kv: bool) -> bool {
        match self {
            Self::Never => false,
            Self::Always => true,
            Self::WhenSharesKv => shares_kv,
        }
    }
}

/// When the codec's encode and dequant run on the CPU on the default hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HotPathClass {
    Metal,
    Cpu(&'static str),
    /// CPU only while the rotor QJL switch is on. The switch is read at call
    /// time by [`KvQuant::cpu_hot_path_reason`], never here.
    CpuWhenQjl(&'static str),
}

/// Every fact one codec states. There is no `Default`: each arm of
/// [`KvQuant::descriptor`] writes each field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool is one independent codec fact that a predicate on KvQuant returns"
)]
pub(super) struct CodecDescriptor {
    /// Dense index from zero. It proves that [`super::ALL_KV_QUANTS`] names
    /// every variant.
    pub(super) index: usize,
    pub(super) k_mirror: MirrorRule,
    pub(super) v_mirror: MirrorRule,
    pub(super) reads_packed_store: bool,
    pub(super) carries_msl: bool,
    pub(super) hot_path: HotPathClass,
    /// Codebook width per side as `(k, v)`; 16 for a side kept at model dtype.
    pub(super) code_bits: (u32, u32),
    /// Packed-store layout per side as `(k, v)`; `None` for a side kept at
    /// model dtype.
    pub(super) side_stores: (Option<SideStore>, Option<SideStore>),
    pub(super) k_below_8bit: bool,
    pub(super) k_only_iso_rotor: bool,
    /// `(k_bits, v_bits, k_group_size, v_group_size)` of the Mixed state, for
    /// the codecs on the Mixed machinery.
    pub(super) mixed_params: Option<(i32, i32, i32, i32)>,
}

const ISO_V_ON_CPU: &str = "IsoQuant (quaternion SO(4)) V-only: a GPU iso encode/dequant branch \
     exists but is shadowed by the bf16 decode seed; prefill V-encode runs \
     on CPU";

const ROTOR_ON_CPU: &str = "RotorQuant (Clifford Cl(3,0)) encode + dequant run on CPU on the \
     default hot path (the bf16 decode seed shadows the GPU branch); the \
     GPU fused-QK encoder is opt-in (--fused-qk)";

const ROTOR_SYM_QJL_ON_CPU: &str = "RotorQuant (Clifford Cl(3,0)) symmetric with QJL enabled \
     (rotor_qjl_enabled): the QJL residual forces K and V onto the \
     CPU encode + dequant path every decode step; disable QJL \
     (--rotor-qjl off) to route both axes through the Metal \
     flash-decode kernel";

const ROTOR_K_ONLY_QJL_ON_CPU: &str = "RotorQuant (Clifford Cl(3,0)) K-only with QJL enabled \
     (rotor_qjl_enabled): the QJL residual forces the K append onto \
     CPU every decode step; disable QJL (--rotor-qjl off) to route \
     the rotor K encode through the Metal kernel";

impl KvQuant {
    /// The facts of this codec. One arm per variant, one literal per arm.
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "one literal per codec is the point: each fact of each codec is read in one place"
    )]
    pub(super) fn descriptor(self) -> CodecDescriptor {
        use HotPathClass::{Cpu, CpuWhenQjl, Metal};
        use MirrorRule::{Always, Never, WhenSharesKv};
        use SideStore::{IsoBlocks, IsoRing, Planar, Rotor, Turbo, Q8};
        match self {
            // Plain bf16: no store, no kernel.
            KvQuant::None => CodecDescriptor {
                index: 0,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: false,
                hot_path: Metal,
                code_bits: (16, 16),
                side_stores: (None, None),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            // The bf16-mirror family: decode reads both mirrors and never the
            // packed store, so `exit_prefill` builds no store for them.
            KvQuant::K8V4 => CodecDescriptor {
                index: 1,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, 4),
                side_stores: (Some(Q8), Some(Turbo)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::K8V8 => CodecDescriptor {
                index: 2,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, 8),
                side_stores: (Some(Q8), Some(Q8)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::Planar => CodecDescriptor {
                index: 3,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, 4),
                side_stores: (Some(Q8), Some(Planar)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::Planar3 => CodecDescriptor {
                index: 4,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, 3),
                side_stores: (Some(Q8), Some(Planar)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            // Sub-8-bit K, but `validate_resolved` guards it with its own error,
            // so it is not in the `k_below_8bit` class.
            KvQuant::PlanarK => CodecDescriptor {
                index: 5,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (4, 16),
                side_stores: (Some(Planar), None),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            // Mixed machinery: decode reads the affine 3-tuples every step; the
            // mirrors exist only for a cross-layer-KV producer. Not in the
            // `k_below_8bit` class: `validate_resolved` checks Mixed K bits
            // from the field.
            KvQuant::Mixed {
                k_bits,
                v_bits,
                k_group_size,
                v_group_size,
            } => CodecDescriptor {
                index: 6,
                k_mirror: WhenSharesKv,
                v_mirror: WhenSharesKv,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (u32::from(k_bits), u32::from(v_bits)),
                side_stores: (
                    Some(SideStore::Affine {
                        group: u32::from(k_group_size),
                    }),
                    Some(SideStore::Affine {
                        group: u32::from(v_group_size),
                    }),
                ),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: Some((
                    i32::from(k_bits),
                    i32::from(v_bits),
                    i32::from(k_group_size),
                    i32::from(v_group_size),
                )),
            },
            // K is fixed at 8-bit, group 64, by `MixedKvState::new_rotated`.
            KvQuant::RotK {
                v_bits,
                v_group_size,
            } => CodecDescriptor {
                index: 7,
                k_mirror: WhenSharesKv,
                v_mirror: WhenSharesKv,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, u32::from(v_bits)),
                side_stores: (
                    Some(SideStore::Affine { group: 64 }),
                    Some(SideStore::Affine {
                        group: u32::from(v_group_size),
                    }),
                ),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: Some((8, i32::from(v_bits), 64, i32::from(v_group_size))),
            },
            KvQuant::K8VTurbo3 => CodecDescriptor {
                index: 8,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, 3),
                side_stores: (Some(Q8), Some(Turbo)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::K8VTurbo3Tcq => CodecDescriptor {
                index: 9,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, 3),
                side_stores: (Some(Q8), Some(Turbo)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::K8VTurbo2 => CodecDescriptor {
                index: 10,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, 2),
                side_stores: (Some(Q8), Some(Turbo)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::K8VTurbo2Tcq => CodecDescriptor {
                index: 11,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (8, 2),
                side_stores: (Some(Q8), Some(Turbo)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::TurboSym3 => CodecDescriptor {
                index: 12,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (3, 3),
                side_stores: (Some(Turbo), Some(Turbo)),
                k_below_8bit: true,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::TurboSym4 => CodecDescriptor {
                index: 13,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (4, 4),
                side_stores: (Some(Turbo), Some(Turbo)),
                k_below_8bit: true,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            // V-only iso: the bf16 decode seed shadows the GPU iso branch, and
            // the V encode that runs (at prefill) is CPU. No ring path, so the
            // store it would hold is the CPU-block form.
            KvQuant::Iso3 => CodecDescriptor {
                index: 14,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Cpu(ISO_V_ON_CPU),
                code_bits: (8, 3),
                side_stores: (Some(Q8), Some(IsoBlocks)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::Iso4 => CodecDescriptor {
                index: 15,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Cpu(ISO_V_ON_CPU),
                code_bits: (8, 4),
                side_stores: (Some(Q8), Some(IsoBlocks)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            // Fused symmetric: a flash kernel reads both packed rings, so no
            // mirror exists. Iso has no QJL sideband, so the hot path is Metal.
            KvQuant::Iso3Sym => CodecDescriptor {
                index: 16,
                k_mirror: Never,
                v_mirror: Never,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (3, 3),
                side_stores: (Some(IsoRing), Some(IsoRing)),
                k_below_8bit: true,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::Iso4Sym => CodecDescriptor {
                index: 17,
                k_mirror: Never,
                v_mirror: Never,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (4, 4),
                side_stores: (Some(IsoRing), Some(IsoRing)),
                k_below_8bit: true,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            // K-only: K is re-quantised into the packed ring every decode step
            // by the iso MSL kernel; V is the bf16 mirror.
            KvQuant::IsoKOnly3 => CodecDescriptor {
                index: 18,
                k_mirror: Never,
                v_mirror: Always,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (3, 16),
                side_stores: (Some(IsoRing), None),
                k_below_8bit: true,
                k_only_iso_rotor: true,
                mixed_params: None,
            },
            KvQuant::IsoKOnly4 => CodecDescriptor {
                index: 19,
                k_mirror: Never,
                v_mirror: Always,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: Metal,
                code_bits: (4, 16),
                side_stores: (Some(IsoRing), None),
                k_below_8bit: true,
                k_only_iso_rotor: true,
                mixed_params: None,
            },
            // V-only rotor: the bf16 decode seed shadows the GPU branch, so the
            // rotor codec fires only at prefill, on the CPU.
            KvQuant::Rotor3 => CodecDescriptor {
                index: 20,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Cpu(ROTOR_ON_CPU),
                code_bits: (8, 3),
                side_stores: (Some(Q8), Some(Rotor)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::Rotor4 => CodecDescriptor {
                index: 21,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Cpu(ROTOR_ON_CPU),
                code_bits: (8, 4),
                side_stores: (Some(Q8), Some(Rotor)),
                k_below_8bit: false,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            // Fused symmetric rotor. The QJL residual cannot be reproduced in
            // the flash inner loop, so a QJL store keeps both axes on the CPU.
            KvQuant::Rotor3Sym => CodecDescriptor {
                index: 22,
                k_mirror: Never,
                v_mirror: Never,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: CpuWhenQjl(ROTOR_SYM_QJL_ON_CPU),
                code_bits: (3, 3),
                side_stores: (Some(Rotor), Some(Rotor)),
                k_below_8bit: true,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::Rotor4Sym => CodecDescriptor {
                index: 23,
                k_mirror: Never,
                v_mirror: Never,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: CpuWhenQjl(ROTOR_SYM_QJL_ON_CPU),
                code_bits: (4, 4),
                side_stores: (Some(Rotor), Some(Rotor)),
                k_below_8bit: true,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            // K-only rotor: `update_rotor_k_only` gates the GPU K encode on the
            // store's sticky QJL flag.
            KvQuant::RotorKOnly3 => CodecDescriptor {
                index: 24,
                k_mirror: Never,
                v_mirror: Always,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: CpuWhenQjl(ROTOR_K_ONLY_QJL_ON_CPU),
                code_bits: (3, 16),
                side_stores: (Some(Rotor), None),
                k_below_8bit: true,
                k_only_iso_rotor: true,
                mixed_params: None,
            },
            KvQuant::RotorKOnly4 => CodecDescriptor {
                index: 25,
                k_mirror: Never,
                v_mirror: Always,
                reads_packed_store: true,
                carries_msl: true,
                hot_path: CpuWhenQjl(ROTOR_K_ONLY_QJL_ON_CPU),
                code_bits: (4, 16),
                side_stores: (Some(Rotor), None),
                k_below_8bit: true,
                k_only_iso_rotor: true,
                mixed_params: None,
            },
            // Rotor K with a V that is TurboQuant at a fixed group of 32
            // (`validate_rotor_k_asym_v`); `v_group_size` is a layout-key tag
            // only, so it does not reach a store parameter.
            KvQuant::RotorK3Asym {
                v_bits,
                v_group_size: _,
            } => CodecDescriptor {
                index: 26,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Cpu(ROTOR_ON_CPU),
                code_bits: (3, u32::from(v_bits)),
                side_stores: (Some(Rotor), Some(Turbo)),
                k_below_8bit: true,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
            KvQuant::RotorK4Asym {
                v_bits,
                v_group_size: _,
            } => CodecDescriptor {
                index: 27,
                k_mirror: Always,
                v_mirror: Always,
                reads_packed_store: false,
                carries_msl: true,
                hot_path: Cpu(ROTOR_ON_CPU),
                code_bits: (4, u32::from(v_bits)),
                side_stores: (Some(Rotor), Some(Turbo)),
                k_below_8bit: true,
                k_only_iso_rotor: false,
                mixed_params: None,
            },
        }
    }
}
