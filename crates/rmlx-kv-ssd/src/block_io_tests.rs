//! Rotor K-side tests use `..` wildcard in match arms deliberately
//! (KvStorage exhaustive coverage is checked in the non-test `block_io.rs`).
#![allow(clippy::wildcard_enum_match_arm)]

use std::sync::Mutex;

use super::*;
use rmlx_core::DispatchPolicy;
use rmlx_mlx::Device;

const MODEL_ID: &str = "Qwen3ForCausalLM/test-snapshot";

/// Process-global lock for tests that mutate `RMLX_ROTOR_QJL` in the env.
/// Held during the entire test body (set → build → write → read → assert → clear).
/// Same pattern as `serve_tests.rs::ENV_LOCK`.
static ROTOR_QJL_ENV_LOCK: Mutex<()> = Mutex::new(());

fn tmp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "rmlx_blockio_{name}_{}.safetensors",
        std::process::id()
    ));
    p
}

// Deterministic LCG f32 data in [-1, 1].
fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / u32::MAX as f32).mul_add(2.0, -1.0)
        })
        .collect()
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: data is a &[f32] with known length; byte reinterpret valid for f32
    // (alignment ≥ 1, size = 4); total byte count = data.len() * 4 fits in isize.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn to_vec(a: &Array) -> Vec<f32> {
    a.eval().unwrap();
    bytes_to_f32(&a.to_bytes().unwrap())
}

// Build a single-layer storage of `quant` populated with `[1,kv_h,S,D]` K/V
// by quantizing directly through the CPU quant primitives, then dequantize K
// back to f32 for tolerance comparison.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
#[allow(
    clippy::too_many_lines,
    reason = "single match over the closed KvQuant enum used by the SSD round-trip suite; one arm per variant, each is small and self-contained"
)]
fn build_storage(
    quant: KvQuant,
    shape: &[i32],
    seed: u64,
    device: Device,
) -> (KvStorage, Vec<f32>) {
    use rmlx_kv_quant::planarquant::planar_quantize;
    use rmlx_kv_quant::q8::q8_quantize;
    use rmlx_kv_quant::turboquant::{turbo_quantize_v, GROUP_SIZE};

    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k_data = lcg(n, seed);
    let v_data = lcg(n, seed ^ 0xABCD);

    let storage = match quant {
        KvQuant::K8V4 => {
            let (kc, ks) = q8_quantize(&k_data);
            let vblk = turbo_quantize_v(&v_data, 4, shape).unwrap();
            KvStorage::K8V4 {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(QuantV::from_cpu_blocks(vec![vblk], shape.to_vec(), 4)),
                max_seq: 4096,
            }
        }
        KvQuant::K8V8 => {
            let (kc, ks) = q8_quantize(&k_data);
            let (vc, vs) = q8_quantize(&v_data);
            KvStorage::K8V8 {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(QuantK::from_cpu_parts(vc, vs, shape.to_vec())),
                max_seq: 4096,
            }
        }
        KvQuant::Planar => {
            let (kc, ks) = q8_quantize(&k_data);
            let vblk = planar_quantize(&v_data, GROUP_SIZE, 4, shape).unwrap();
            KvStorage::Planar {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(QuantPlanarV::from_cpu_blocks(vec![vblk], shape.to_vec(), 4)),
                max_seq: 4096,
                bits: 4,
            }
        }
        // Planar3 — same layout as Planar but 3-bit V codebook.
        KvQuant::Planar3 => {
            let (kc, ks) = q8_quantize(&k_data);
            let vblk = planar_quantize(&v_data, GROUP_SIZE, 3, shape).unwrap();
            KvStorage::Planar {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(QuantPlanarV::from_cpu_blocks(vec![vblk], shape.to_vec(), 3)),
                max_seq: 4096,
                bits: 3,
            }
        }
        KvQuant::Mixed {
            k_bits,
            v_bits,
            k_group_size,
            v_group_size,
        } => {
            let mut state = MixedKvState::new(
                i32::from(k_bits),
                i32::from(v_bits),
                i32::from(k_group_size),
                i32::from(v_group_size),
            );
            let k = arr(&k_data, shape);
            let v = arr(&v_data, shape);
            state
                .bulk_init_from_fp16(&k, &v, device, DispatchPolicy::default())
                .unwrap();
            KvStorage::Mixed {
                state,
                max_seq: 4096,
            }
        }
        KvQuant::RotKTq4V => {
            // K: build via MixedKvState K-only bulk init (rotate + affine-quantize).
            // V: build via QuantV CPU path (TurboQuant scalar).
            use rmlx_kv_quant::turboquant::turbo_quantize_v;
            let mut k_state = MixedKvState::new_k_only_rotated();
            let k_arr = arr(&k_data, shape);
            let (k_codes, k_scales, k_biases) = k_state
                .bulk_init_k_from_fp16(&k_arr, device, DispatchPolicy::default())
                .unwrap();
            k_state.keys = Some(MixedTuple {
                codes: k_codes,
                scales: k_scales,
                biases: k_biases,
            });
            k_state.offset = shape[2];

            let vblk = turbo_quantize_v(&v_data, 4, shape).unwrap();
            let qv = QuantV::from_cpu_blocks(vec![vblk], shape.to_vec(), 4);
            KvStorage::RotKTq4V {
                k_state,
                v: Some(qv),
                max_seq: 4096,
            }
        }
        // TurboSym3 — symmetric WHT-3 K + turbo3 V (CPU-only build).
        KvQuant::TurboSym3 => {
            use rmlx_kv_quant::storage::QuantKTurbo3;
            let kblk = turbo_quantize_v(&k_data, 3, shape).unwrap();
            let vblk = turbo_quantize_v(&v_data, 3, shape).unwrap();
            KvStorage::TurboSym3 {
                k: Some(QuantKTurbo3::from_cpu_blocks(
                    vec![kblk],
                    shape.to_vec(),
                    3,
                    4096,
                )),
                v: Some(QuantV::from_cpu_blocks(vec![vblk], shape.to_vec(), 3)),
                max_seq: 4096,
            }
        }
        // TurboSym4 — symmetric WHT-4 K + tq4 V (CPU-only build).
        KvQuant::TurboSym4 => {
            use rmlx_kv_quant::turboquant::turbo_quantize_v;
            let kblk = turbo_quantize_v(&k_data, 4, shape).unwrap();
            let vblk = turbo_quantize_v(&v_data, 4, shape).unwrap();
            KvStorage::TurboSym4 {
                k: Some(QuantKTurbo4::from_cpu_blocks(vec![kblk], shape.to_vec(), 4)),
                v: Some(QuantV::from_cpu_blocks(vec![vblk], shape.to_vec(), 4)),
                max_seq: 4096,
            }
        }
        // PlanarK — K-axis PlanarQuant 4-bit. V is bf16 off-storage.
        KvQuant::PlanarK => {
            use rmlx_kv_quant::storage::QuantPlanarK;
            let kblk = planar_quantize(&k_data, GROUP_SIZE, 4, shape).unwrap();
            KvStorage::PlanarK {
                k: Some(QuantPlanarK::from_cpu_blocks(vec![kblk], shape.to_vec())),
                max_seq: 4096,
            }
        }
        // K8VTurbo2 — K is QuantK (q8_0), V is QuantV bits=2.
        KvQuant::K8VTurbo2 => {
            let (kc, ks) = q8_quantize(&k_data);
            let vblk = turbo_quantize_v(&v_data, 2, shape).unwrap();
            KvStorage::K8VTurbo2 {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(QuantV::from_cpu_blocks(vec![vblk], shape.to_vec(), 2)),
                max_seq: 4096,
            }
        }
        // Iso3 — K is QuantK (q8_0); V is QuantIsoV3 (CPU path).
        KvQuant::Iso3 => {
            use rmlx_kv_quant::isoquant::iso_encode_fast;
            use rmlx_kv_quant::storage::{IsoBlocks, QuantIsoV3};
            let (kc, ks) = q8_quantize(&k_data);
            let head_dim = shape[3] as usize;
            let n_tokens_total = (shape[0] as usize) * (shape[1] as usize) * (shape[2] as usize);
            let (codes_u32, scales, quaternions, norms) = iso_encode_fast(
                &v_data,
                head_dim,
                rmlx_kv_quant::storage::ISO3_GROUP_SIZE,
                rmlx_kv_quant::storage::ISO3_BITS,
            )
            .unwrap();
            let blk = IsoBlocks {
                codes: codes_u32,
                scales,
                quaternions,
                norms,
                n_tokens: n_tokens_total,
            };
            let qv = QuantIsoV3::from_cpu_blocks(vec![blk], shape.to_vec());
            KvStorage::IsoV3 {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(qv),
                max_seq: 4096,
            }
        }
        // Iso4 — K is QuantK (q8_0); V is QuantIsoV4 (CPU path,
        // 4-bit codebook + dense 8-vals-per-u32 pack).
        KvQuant::Iso4 => {
            use rmlx_kv_quant::isoquant::iso_encode_fast;
            use rmlx_kv_quant::storage::{IsoBlocks, QuantIsoV4};
            let (kc, ks) = q8_quantize(&k_data);
            let head_dim = shape[3] as usize;
            let n_tokens_total = (shape[0] as usize) * (shape[1] as usize) * (shape[2] as usize);
            let (codes_u32, scales, quaternions, norms) = iso_encode_fast(
                &v_data,
                head_dim,
                rmlx_kv_quant::storage::ISO4_GROUP_SIZE,
                rmlx_kv_quant::storage::ISO4_BITS,
            )
            .unwrap();
            let blk = IsoBlocks {
                codes: codes_u32,
                scales,
                quaternions,
                norms,
                n_tokens: n_tokens_total,
            };
            let qv = QuantIsoV4::from_cpu_blocks(vec![blk], shape.to_vec());
            KvStorage::IsoV4 {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(qv),
                max_seq: 4096,
            }
        }
        // Rotor3 — K is QuantK (q8_0); V is QuantRotorV3 (CPU path).
        KvQuant::Rotor3 => {
            use rmlx_kv_quant::clifford::make_rotor_table;
            use rmlx_kv_quant::rotorquant::{n_groups_for, rotor3_encode};
            use rmlx_kv_quant::storage::{QuantRotorV3, RotorBlocks};
            let (kc, ks) = q8_quantize(&k_data);
            let head_dim = shape[3] as usize;
            let n_tokens_total = (shape[0] as usize) * (shape[1] as usize) * (shape[2] as usize);
            let n_groups = n_groups_for(head_dim);
            // Use layer_idx=0 / head_idx=0 — the SSD writer persists the
            // rotor table so cross-restart identity does not depend on the
            // seed inputs at hydrate time.
            let rotors = make_rotor_table(0, 0, n_groups);
            let (codes, scales, norms) = rotor3_encode(&v_data, &rotors, head_dim).unwrap();
            let blk = RotorBlocks {
                codes,
                scales,
                norms,
                n_tokens: n_tokens_total,
            };
            let qv = QuantRotorV3::from_cpu_blocks(rotors, vec![blk], shape.to_vec(), 0);
            KvStorage::RotorV3 {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(qv),
                max_seq: 4096,
            }
        }
        // Rotor4 — K is QuantK (q8_0); V is QuantRotorV4 (CPU path).
        KvQuant::Rotor4 => {
            use rmlx_kv_quant::clifford::make_rotor_table;
            use rmlx_kv_quant::rotorquant::{n_groups_for, rotor4_encode};
            use rmlx_kv_quant::storage::{QuantRotorV4, RotorBlocks};
            let (kc, ks) = q8_quantize(&k_data);
            let head_dim = shape[3] as usize;
            let n_tokens_total = (shape[0] as usize) * (shape[1] as usize) * (shape[2] as usize);
            let n_groups = n_groups_for(head_dim);
            // Use layer_idx=0 / head_idx=0 — the SSD writer persists the
            // rotor table so cross-restart identity does not depend on the
            // seed inputs at hydrate time.
            let rotors = make_rotor_table(0, 0, n_groups);
            let (codes, scales, norms) = rotor4_encode(&v_data, &rotors, head_dim).unwrap();
            let blk = RotorBlocks {
                codes,
                scales,
                norms,
                n_tokens: n_tokens_total,
            };
            let qv = QuantRotorV4::from_cpu_blocks(rotors, vec![blk], shape.to_vec(), 0);
            KvStorage::RotorV4 {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(qv),
                max_seq: 4096,
            }
        }
        // K8VTurbo3Tcq — K is QuantK (q8_0), V is QuantV bits=3. Build with
        // the Viterbi encoder so the test round-trips the same codes the
        // production path would write. `use_tcq=true` is set via
        // `from_cpu_blocks_tcq` so any post-hydrate decode-step encode would
        // continue using Viterbi.
        KvQuant::K8VTurbo3Tcq => {
            use rmlx_kv_quant::tcq::tcq_quantize_v3;
            let (kc, ks) = q8_quantize(&k_data);
            let vblk = tcq_quantize_v3(&v_data, shape).unwrap();
            KvStorage::K8VTurbo3Tcq {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(QuantV::from_cpu_blocks_tcq(vec![vblk], shape.to_vec(), 3)),
                max_seq: 4096,
            }
        }
        // K8VTurbo2Tcq — K is QuantK (q8_0), V is QuantV bits=2. Build with
        // the Viterbi encoder so the test round-trips the same codes the
        // production path would write. `use_tcq=true` is set via
        // `from_cpu_blocks_tcq` so any post-hydrate decode-step encode would
        // continue using Viterbi.
        KvQuant::K8VTurbo2Tcq => {
            use rmlx_kv_quant::tcq::tcq_quantize_v2;
            let (kc, ks) = q8_quantize(&k_data);
            let vblk = tcq_quantize_v2(&v_data, shape).unwrap();
            KvStorage::K8VTurbo2Tcq {
                k: Some(QuantK::from_cpu_parts(kc, ks, shape.to_vec())),
                v: Some(QuantV::from_cpu_blocks_tcq(vec![vblk], shape.to_vec(), 2)),
                max_seq: 4096,
            }
        }
        // Iso3Sym / Iso4Sym / IsoKOnly3 / IsoKOnly4 — K-side iso codec built
        // via the axis-agnostic isoquant CPU encoder.
        KvQuant::Iso3Sym => {
            use rmlx_kv_quant::isoquant::iso_encode_fast;
            use rmlx_kv_quant::storage::{
                IsoBlocks, QuantIsoK3, QuantIsoV3, ISO3_BITS, ISO3_GROUP_SIZE,
            };
            let head_dim = shape[3] as usize;
            let n_tokens_total = (shape[0] as usize) * (shape[1] as usize) * (shape[2] as usize);
            // K-side iso3 encode.
            let (k_codes, k_scales, k_quats, k_norms) =
                iso_encode_fast(&k_data, head_dim, ISO3_GROUP_SIZE, ISO3_BITS).unwrap();
            let k_blk = IsoBlocks {
                codes: k_codes,
                scales: k_scales,
                quaternions: k_quats,
                norms: k_norms,
                n_tokens: n_tokens_total,
            };
            let qk = QuantIsoK3::from_cpu_blocks(vec![k_blk], shape.to_vec(), 4096);
            // V-side iso3 encode.
            let (v_codes, v_scales, v_quats, v_norms) =
                iso_encode_fast(&v_data, head_dim, ISO3_GROUP_SIZE, ISO3_BITS).unwrap();
            let v_blk = IsoBlocks {
                codes: v_codes,
                scales: v_scales,
                quaternions: v_quats,
                norms: v_norms,
                n_tokens: n_tokens_total,
            };
            let qv = QuantIsoV3::from_cpu_blocks(vec![v_blk], shape.to_vec());
            KvStorage::IsoSym3 {
                k: Some(qk),
                v: Some(qv),
                max_seq: 4096,
            }
        }
        KvQuant::Iso4Sym => {
            use rmlx_kv_quant::isoquant::iso_encode_fast;
            use rmlx_kv_quant::storage::{
                IsoBlocks, QuantIsoK4, QuantIsoV4, ISO4_BITS, ISO4_GROUP_SIZE,
            };
            let head_dim = shape[3] as usize;
            let n_tokens_total = (shape[0] as usize) * (shape[1] as usize) * (shape[2] as usize);
            let (k_codes, k_scales, k_quats, k_norms) =
                iso_encode_fast(&k_data, head_dim, ISO4_GROUP_SIZE, ISO4_BITS).unwrap();
            let k_blk = IsoBlocks {
                codes: k_codes,
                scales: k_scales,
                quaternions: k_quats,
                norms: k_norms,
                n_tokens: n_tokens_total,
            };
            let qk = QuantIsoK4::from_cpu_blocks(vec![k_blk], shape.to_vec(), 4096);
            let (v_codes, v_scales, v_quats, v_norms) =
                iso_encode_fast(&v_data, head_dim, ISO4_GROUP_SIZE, ISO4_BITS).unwrap();
            let v_blk = IsoBlocks {
                codes: v_codes,
                scales: v_scales,
                quaternions: v_quats,
                norms: v_norms,
                n_tokens: n_tokens_total,
            };
            let qv = QuantIsoV4::from_cpu_blocks(vec![v_blk], shape.to_vec());
            KvStorage::IsoSym4 {
                k: Some(qk),
                v: Some(qv),
                max_seq: 4096,
            }
        }
        KvQuant::IsoKOnly3 => {
            use rmlx_kv_quant::isoquant::iso_encode_fast;
            use rmlx_kv_quant::storage::{IsoBlocks, QuantIsoK3, ISO3_BITS, ISO3_GROUP_SIZE};
            let head_dim = shape[3] as usize;
            let n_tokens_total = (shape[0] as usize) * (shape[1] as usize) * (shape[2] as usize);
            let (k_codes, k_scales, k_quats, k_norms) =
                iso_encode_fast(&k_data, head_dim, ISO3_GROUP_SIZE, ISO3_BITS).unwrap();
            let k_blk = IsoBlocks {
                codes: k_codes,
                scales: k_scales,
                quaternions: k_quats,
                norms: k_norms,
                n_tokens: n_tokens_total,
            };
            let qk = QuantIsoK3::from_cpu_blocks(vec![k_blk], shape.to_vec(), 4096);
            KvStorage::IsoKOnly3 {
                k: Some(qk),
                max_seq: 4096,
            }
        }
        KvQuant::IsoKOnly4 => {
            use rmlx_kv_quant::isoquant::iso_encode_fast;
            use rmlx_kv_quant::storage::{IsoBlocks, QuantIsoK4, ISO4_BITS, ISO4_GROUP_SIZE};
            let head_dim = shape[3] as usize;
            let n_tokens_total = (shape[0] as usize) * (shape[1] as usize) * (shape[2] as usize);
            let (k_codes, k_scales, k_quats, k_norms) =
                iso_encode_fast(&k_data, head_dim, ISO4_GROUP_SIZE, ISO4_BITS).unwrap();
            let k_blk = IsoBlocks {
                codes: k_codes,
                scales: k_scales,
                quaternions: k_quats,
                norms: k_norms,
                n_tokens: n_tokens_total,
            };
            let qk = QuantIsoK4::from_cpu_blocks(vec![k_blk], shape.to_vec(), 4096);
            KvStorage::IsoKOnly4 {
                k: Some(qk),
                max_seq: 4096,
            }
        }
        // Rotor K-side variants. K-side uses QuantRotorK3/K4 via `append` so
        // the QJL sideband (when active per env) is captured.
        KvQuant::Rotor3Sym => {
            use rmlx_kv_quant::storage::{QuantRotorK3, QuantRotorV3};
            let mut qk = QuantRotorK3::new(vec![shape[0], shape[1], 0, shape[3]], 0);
            qk.append(&k_data, shape).unwrap();
            let mut qv = QuantRotorV3::new(vec![shape[0], shape[1], 0, shape[3]], 4096, 0);
            qv.append(&v_data, shape).unwrap();
            KvStorage::RotorSym3 {
                k: Some(qk),
                v: Some(qv),
                max_seq: 4096,
            }
        }
        KvQuant::Rotor4Sym => {
            use rmlx_kv_quant::storage::{QuantRotorK4, QuantRotorV4};
            let mut qk = QuantRotorK4::new(vec![shape[0], shape[1], 0, shape[3]], 0);
            qk.append(&k_data, shape).unwrap();
            let mut qv = QuantRotorV4::new(vec![shape[0], shape[1], 0, shape[3]], 4096, 0);
            qv.append(&v_data, shape).unwrap();
            KvStorage::RotorSym4 {
                k: Some(qk),
                v: Some(qv),
                max_seq: 4096,
            }
        }
        KvQuant::RotorKOnly3 => {
            use rmlx_kv_quant::storage::QuantRotorK3;
            let mut qk = QuantRotorK3::new(vec![shape[0], shape[1], 0, shape[3]], 0);
            qk.append(&k_data, shape).unwrap();
            KvStorage::RotorKOnly3 {
                k: Some(qk),
                max_seq: 4096,
            }
        }
        KvQuant::RotorKOnly4 => {
            use rmlx_kv_quant::storage::QuantRotorK4;
            let mut qk = QuantRotorK4::new(vec![shape[0], shape[1], 0, shape[3]], 0);
            qk.append(&k_data, shape).unwrap();
            KvStorage::RotorKOnly4 {
                k: Some(qk),
                max_seq: 4096,
            }
        }
        // RotorK3Asym — rotor3 K + affine V at (v_bits, v_group_size).
        KvQuant::RotorK3Asym {
            v_bits,
            v_group_size,
        } => {
            use rmlx_kv_quant::storage::QuantRotorK3;
            let mut qk = QuantRotorK3::new(vec![shape[0], shape[1], 0, shape[3]], 0);
            qk.append(&k_data, shape).unwrap();
            // V: affine via TurboQuant V kernel (bits=v_bits). The SSD path uses
            // the QuantV codec — for tests we build with the same kernel the
            // production exit_prefill uses.
            let vblk = turbo_quantize_v(&v_data, v_bits, shape).unwrap();
            KvStorage::RotorKAsym3 {
                k: Some(qk),
                v: Some(QuantV::from_cpu_blocks(vec![vblk], shape.to_vec(), v_bits)),
                max_seq: 4096,
                v_bits,
                v_group_size,
            }
        }
        // RotorK4Asym — mirror with rotor4 K.
        KvQuant::RotorK4Asym {
            v_bits,
            v_group_size,
        } => {
            use rmlx_kv_quant::storage::QuantRotorK4;
            let mut qk = QuantRotorK4::new(vec![shape[0], shape[1], 0, shape[3]], 0);
            qk.append(&k_data, shape).unwrap();
            let vblk = turbo_quantize_v(&v_data, v_bits, shape).unwrap();
            KvStorage::RotorKAsym4 {
                k: Some(qk),
                v: Some(QuantV::from_cpu_blocks(vec![vblk], shape.to_vec(), v_bits)),
                max_seq: 4096,
                v_bits,
                v_group_size,
            }
        }
        other => panic!("build_storage: unsupported quant {other:?}"),
    };

    let k_recon = dequant_k(&storage, device);
    (storage, k_recon)
}

/// Round-trip one storage variant: build → write → read → dequant K, assert
/// the dequant matches the pre-serialization dequant byte-for-byte (codes
/// are stored exactly, so reconstruction is lossless relative to the
/// already-quantized state).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn roundtrip_variant(name: &str, quant: KvQuant) {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(quant, shape, 0x1234_5678, device);

    let layers = vec![storage];
    let path = tmp_path(name);
    KvBlockWriter::new(MODEL_ID, quant, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, quant, device).unwrap();
    assert_eq!(rebuilt.len(), 1, "{name}: layer count");

    // Dequant K from the rebuilt storage and compare to the pre-write dequant.
    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(
        k_recon_before.len(),
        k_recon_after.len(),
        "{name}: K length mismatch"
    );
    let max_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Codes round-trip exactly → dequant is bit-identical. Allow a hair of
    // slack for any bf16 cast on the GPU path (CPU path is exact f32).
    assert!(
        max_err < 1e-3,
        "{name}: K dequant round-trip error {max_err} too large"
    );
    let _ = std::fs::remove_file(&path);
}

// Dequant the K side of a storage to flat f32 (CPU paths only — tests run on CPU).
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn dequant_k(storage: &KvStorage, device: Device) -> Vec<f32> {
    match storage {
        KvStorage::K8V4 { k, .. } | KvStorage::K8V8 { k, .. } | KvStorage::Planar { k, .. } => {
            let (flat, _) = k
                .as_ref()
                .unwrap()
                .dequantize_choice(device, Dtype::F32)
                .unwrap();
            flat
        }
        KvStorage::Mixed { state, .. } => {
            // Dequant the K tuple via mx.dequantize.
            let t = state.keys.as_ref().unwrap();
            let out = rmlx_mlx::dequantize(
                &t.codes,
                &t.scales,
                Some(&t.biases),
                state.k_group_size,
                state.k_bits,
                "affine",
                device,
            )
            .unwrap();
            to_vec(&out)
        }
        KvStorage::Paged { k, .. } => {
            let pk = k.as_ref().unwrap();
            let (codes, scales) = pk.gather(device).unwrap();
            codes.eval().unwrap();
            scales.eval().unwrap();
            let codes_v = codes.to_bytes().unwrap();
            let scales_v = bytes_to_f32(&scales.to_bytes().unwrap());
            rmlx_kv_quant::q8::q8_dequantize(&codes_v, &scales_v)
        }
        KvStorage::None { .. } => Vec::new(),
        // RotKTq4V — K is stored in the MixedKvState (same as Mixed/RotK).
        KvStorage::RotKTq4V { k_state, .. } => {
            let t = k_state.keys.as_ref().unwrap();
            let out = rmlx_mlx::dequantize(
                &t.codes,
                &t.scales,
                Some(&t.biases),
                k_state.k_group_size,
                k_state.k_bits,
                "affine",
                device,
            )
            .unwrap();
            to_vec(&out)
        }
        // K8VTurbo3 — K is QuantK (same as K8V4).
        KvStorage::K8VTurbo3 { k, .. } => {
            let (flat, _) = k
                .as_ref()
                .unwrap()
                .dequantize_choice(device, Dtype::F32)
                .unwrap();
            flat
        }
        // TurboSym3 — K is `QuantKTurbo3` (TurboQuant 3-bit).
        KvStorage::TurboSym3 { k, .. } => k.as_ref().unwrap().dequant().unwrap(),
        // TurboSym4 — K is `QuantKTurbo4` (TurboQuant 4-bit).
        KvStorage::TurboSym4 { k, .. } => {
            let (flat, _) = k
                .as_ref()
                .unwrap()
                .dequantize_choice(device, Dtype::F32)
                .unwrap();
            flat
        }
        // PlanarK — K is `QuantPlanarK` (same dequant API as V row).
        KvStorage::PlanarK { k, .. } => {
            let (flat, _) = k
                .as_ref()
                .unwrap()
                .dequantize_choice(device, Dtype::F32)
                .unwrap();
            flat
        }
        // K8VTurbo2 — K is QuantK (same as K8V4).
        KvStorage::K8VTurbo2 { k, .. } => {
            let (flat, _) = k
                .as_ref()
                .unwrap()
                .dequantize_choice(device, Dtype::F32)
                .unwrap();
            flat
        }
        // IsoV3 — K is QuantK; the test helper dequants K so the exhaustive
        // match stays happy.
        // IsoV4 mirrors IsoV3 on K (q8_0 affine).
        // RotorV3 — K is also QuantK (q8_0 affine).
        // RotorV4 — K is also QuantK (q8_0 affine).
        KvStorage::IsoV3 { k, .. }
        | KvStorage::IsoV4 { k, .. }
        | KvStorage::RotorV3 { k, .. }
        | KvStorage::RotorV4 { k, .. } => {
            let (flat, _) = k
                .as_ref()
                .unwrap()
                .dequantize_choice(device, Dtype::F32)
                .unwrap();
            flat
        }
        // K8VTurbo3Tcq — K is QuantK (same as K8V4 / K8VTurbo3).
        // K8VTurbo2Tcq — K is QuantK (same pattern; V-side bits=2).
        KvStorage::K8VTurbo3Tcq { k, .. } | KvStorage::K8VTurbo2Tcq { k, .. } => {
            let (flat, _) = k
                .as_ref()
                .unwrap()
                .dequantize_choice(device, Dtype::F32)
                .unwrap();
            flat
        }
        // IsoSym3 / IsoKOnly3 — K is `QuantIsoK3` (CPU-only).
        KvStorage::IsoSym3 { k, .. } | KvStorage::IsoKOnly3 { k, .. } => {
            k.as_ref().unwrap().dequant().unwrap()
        }
        KvStorage::IsoSym4 { k, .. } | KvStorage::IsoKOnly4 { k, .. } => {
            k.as_ref().unwrap().dequant().unwrap()
        }
        // RotorSym3 / RotorKOnly3 — K is `QuantRotorK3` (CPU-only).
        KvStorage::RotorSym3 { k, .. } | KvStorage::RotorKOnly3 { k, .. } => {
            k.as_ref().unwrap().dequant().unwrap()
        }
        // RotorSym4 / RotorKOnly4 — K is `QuantRotorK4` (CPU-only).
        KvStorage::RotorSym4 { k, .. } | KvStorage::RotorKOnly4 { k, .. } => {
            k.as_ref().unwrap().dequant().unwrap()
        }
        // RotorKAsym3 / RotorKAsym4 — K is QuantRotorK3 / QuantRotorK4
        // (CPU-only); V is affine but only K is dequant-probed here.
        KvStorage::RotorKAsym3 { k, .. } => k.as_ref().unwrap().dequant().unwrap(),
        KvStorage::RotorKAsym4 { k, .. } => k.as_ref().unwrap().dequant().unwrap(),
    }
}

#[test]
fn roundtrip_k8v4() {
    roundtrip_variant("k8v4", KvQuant::K8V4);
}

#[test]
fn roundtrip_k8v8() {
    roundtrip_variant("k8v8", KvQuant::K8V8);
}

#[test]
fn roundtrip_planar() {
    roundtrip_variant("planar", KvQuant::Planar);
}

/// TurboSym3 round-trip: build (CPU TurboQuant 3-bit K+V) → write → read →
/// dequant K must match pre-write dequant exactly (codes round-trip
/// bit-identically through the safetensors serialization).
#[test]
fn roundtrip_tsym3() {
    roundtrip_variant("tsym3", KvQuant::TurboSym3);
}

/// TurboSym4 round-trip: build (CPU TurboQuant 4-bit K+V) → write → read →
/// dequant K must match pre-write dequant exactly (codes round-trip
/// bit-identically through the safetensors serialization).
#[test]
fn roundtrip_tsym4() {
    roundtrip_variant("tsym4", KvQuant::TurboSym4);
}

/// Planar3 SSD round-trip: K codes/scales + V 3-bit codes survive the
/// spill/hydrate cycle (codes stored exactly → dequant is bit-identical).
/// Also verifies that `bits=3` is preserved in the reconstructed storage.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn roundtrip_planar3() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::Planar3, shape, 0xA127_B310, device);

    // Capture V codes and bits before serialization.
    let (v_codes_before, v_bits_before) = match &storage {
        KvStorage::Planar {
            v: Some(qpv), bits, ..
        } => {
            assert_eq!(*bits, 3, "build_storage must produce bits=3 for Planar3");
            (qpv.blocks[0].codes.clone(), *bits)
        }
        _ => panic!("expected KvStorage::Planar for Planar3"),
    };
    assert_eq!(v_bits_before, 3, "Planar3 storage must have bits=3");

    let layers = vec![storage];
    let path = tmp_path("planar3");
    KvBlockWriter::new(MODEL_ID, KvQuant::Planar3, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, KvQuant::Planar3, device).unwrap();
    assert_eq!(rebuilt.len(), 1, "planar3: layer count");

    // Assert bits=3 is preserved through the round-trip.
    let (v_codes_after, v_bits_after) = match &rebuilt[0] {
        KvStorage::Planar {
            v: Some(qpv), bits, ..
        } => (qpv.blocks[0].codes.clone(), *bits),
        _ => panic!("expected KvStorage::Planar after Planar3 hydrate"),
    };
    assert_eq!(
        v_bits_after, 3,
        "Planar3: bits must remain 3 after SSD round-trip"
    );
    assert_eq!(
        v_codes_before, v_codes_after,
        "Planar3 V codes are not byte-identical after SSD round-trip"
    );

    // K dequant must match pre-write within tolerance (codes round-trip exactly).
    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(
        k_recon_before.len(),
        k_recon_after.len(),
        "planar3: K length mismatch"
    );
    let max_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-3,
        "planar3: K dequant round-trip error {max_err} too large"
    );
    let _ = std::fs::remove_file(&path);
}

/// PlanarK SSD round-trip: K codes/scales/rotations survive the spill/hydrate
/// cycle (codes stored exactly → dequant is bit-identical).
#[test]
fn roundtrip_planar_k() {
    roundtrip_variant("planar_k", KvQuant::PlanarK);
}

/// K8VTurbo2 write → read → dequant K round-trip.
/// Mirrors the K8V4 / K8V8 / Planar SSD round-trip pattern. V codes are
/// stored exactly (bits=2 pack format is the same Vec<u8> layout as bits=4).
#[test]
fn roundtrip_k8vturbo2() {
    roundtrip_variant("k8vturbo2", KvQuant::K8VTurbo2);
}

/// K8VTurbo3Tcq SSD round-trip — V-side codes must survive a write → read
/// cycle byte-for-byte, and the hydrated `QuantV` must carry `use_tcq = true`
/// so any subsequent decode-step encode would re-enter the Viterbi path (not
/// silently fall back to nearest-centroid).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test scaffolding: shape established by build_storage"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by build_storage / writer / reader"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the K8VTurbo3Tcq variant only; wildcard panics on shape drift"
)]
fn roundtrip_k8vturbo3_tcq() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::K8VTurbo3Tcq, shape, 0x7A57_F00D, device);

    let v_codes_before = match &storage {
        KvStorage::K8VTurbo3Tcq { v: Some(qv), .. } => qv.blocks[0].codes.clone(),
        _ => panic!("expected K8VTurbo3Tcq storage"),
    };
    let v_bits_before = match &storage {
        KvStorage::K8VTurbo3Tcq { v: Some(qv), .. } => qv.blocks[0].bits,
        _ => unreachable!(),
    };
    assert_eq!(v_bits_before, 3, "build_storage should produce bits=3 V");

    let layers = vec![storage];
    let path = tmp_path("k8vturbo3tcq");
    KvBlockWriter::new(MODEL_ID, KvQuant::K8VTurbo3Tcq, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::K8VTurbo3Tcq, device)
        .unwrap();
    assert_eq!(rebuilt.len(), 1, "layer count");

    let (v_codes_after, v_bits_after, use_tcq_after) = match &rebuilt[0] {
        KvStorage::K8VTurbo3Tcq { v: Some(qv), .. } => {
            (qv.blocks[0].codes.clone(), qv.blocks[0].bits, qv.use_tcq)
        }
        _ => panic!("expected K8VTurbo3Tcq after hydrate"),
    };
    assert_eq!(
        v_bits_after, 3,
        "K8VTurbo3Tcq V must reconstruct with bits=3"
    );
    assert!(
        use_tcq_after,
        "K8VTurbo3Tcq hydrated QuantV must carry use_tcq=true so post-hydrate \
         decode-step encodes stay on the Viterbi path"
    );
    assert_eq!(
        v_codes_before, v_codes_after,
        "K8VTurbo3Tcq V codes are not byte-identical after SSD round-trip"
    );

    let _ = std::fs::remove_file(&path);
}

/// K8VTurbo2 V-side round-trip — verify the QuantV bits=2 codes survive
/// serialise → deserialise byte-for-byte. The K-side is covered by
/// `roundtrip_k8vturbo2`; this asserts the 2-bit V layout decoder in
/// `read_quant_v_bits(..., 2)` matches what `write_quant_v` writes.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test scaffolding: shape established by build_storage"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by build_storage / writer / reader"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the K8VTurbo2 variant only; wildcard panics on shape drift"
)]
fn roundtrip_k8vturbo2_v_codes_byte_identical() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::K8VTurbo2, shape, 0xCAFE_F00D, device);

    let v_codes_before = match &storage {
        KvStorage::K8VTurbo2 { v: Some(qv), .. } => qv.blocks[0].codes.clone(),
        _ => panic!("expected K8VTurbo2 storage"),
    };
    let v_bits_before = match &storage {
        KvStorage::K8VTurbo2 { v: Some(qv), .. } => qv.blocks[0].bits,
        _ => unreachable!(),
    };
    assert_eq!(v_bits_before, 2, "build_storage should produce bits=2 V");

    let layers = vec![storage];
    let path = tmp_path("k8vturbo2_v_codes");
    KvBlockWriter::new(MODEL_ID, KvQuant::K8VTurbo2, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::K8VTurbo2, device)
        .unwrap();
    assert_eq!(rebuilt.len(), 1, "layer count");

    let (v_codes_after, v_bits_after) = match &rebuilt[0] {
        KvStorage::K8VTurbo2 { v: Some(qv), .. } => (qv.blocks[0].codes.clone(), qv.blocks[0].bits),
        _ => panic!("expected K8VTurbo2 after hydrate"),
    };
    assert_eq!(v_bits_after, 2, "K8VTurbo2 V must reconstruct with bits=2");
    assert_eq!(
        v_codes_before, v_codes_after,
        "K8VTurbo2 V codes are not byte-identical after SSD round-trip"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn roundtrip_mixed() {
    roundtrip_variant(
        "mixed",
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
    );
}

/// RotKTq4V write → read → dequant K must be bit-identical (codes stored
/// exactly). Also verifies the V-side block count round-trips (the CPU QuantV
/// has one TurboBlock after build).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn roundtrip_rot_k_tq4v() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::RotKTq4V, shape, 0xDEAD_BEEF, device);

    // Capture the V codes before serialization for comparison.
    let v_codes_before = match &storage {
        KvStorage::RotKTq4V { v: Some(qv), .. } => qv.blocks[0].codes.clone(),
        _ => panic!("expected RotKTq4V storage"),
    };

    let layers = vec![storage];
    let path = tmp_path("rot_k_tq4v");
    KvBlockWriter::new(MODEL_ID, KvQuant::RotKTq4V, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, KvQuant::RotKTq4V, device).unwrap();
    assert_eq!(rebuilt.len(), 1, "layer count");

    // K codes round-trip: dequant must be bit-identical.
    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(k_recon_before.len(), k_recon_after.len(), "K length");
    let max_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-3,
        "rot_k_tq4v: K dequant round-trip error {max_err}"
    );

    // V codes round-trip: CPU blocks must be byte-identical.
    let v_codes_after = match &rebuilt[0] {
        KvStorage::RotKTq4V { v: Some(qv), .. } => qv.blocks[0].codes.clone(),
        _ => panic!("expected RotKTq4V after hydrate"),
    };
    assert_eq!(v_codes_before, v_codes_after, "V codes round-trip");

    let _ = std::fs::remove_file(&path);
}

/// IsoV3 SSD round-trip: all four V-side buffers (codes_packed, scales,
/// quaternions, norms) survive the spill/hydrate cycle bit-identically.
/// K dequant matches within tolerance (codes round-trip exactly).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: shape set by build_storage"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by build_storage / writer / reader"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the IsoV3 variant only; wildcard panics on shape drift"
)]
fn roundtrip_iso3() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::Iso3, shape, 0xA128_B11C, device);

    // Capture the four V-side buffers before serialization.
    let (v_codes_before, v_scales_before, v_quats_before, v_norms_before) = match &storage {
        KvStorage::IsoV3 { v: Some(qv), .. } => {
            let blk = &qv.blocks[0];
            (
                blk.codes.clone(),
                blk.scales.clone(),
                blk.quaternions.clone(),
                blk.norms.clone(),
            )
        }
        _ => panic!("expected KvStorage::IsoV3 for Iso3"),
    };

    let layers = vec![storage];
    let path = tmp_path("iso3");
    KvBlockWriter::new(MODEL_ID, KvQuant::Iso3, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, KvQuant::Iso3, device).unwrap();
    assert_eq!(rebuilt.len(), 1, "iso3: layer count");

    // All four V-side buffers must be byte/value-identical after round-trip.
    let (v_codes_after, v_scales_after, v_quats_after, v_norms_after) = match &rebuilt[0] {
        KvStorage::IsoV3 { v: Some(qv), .. } => {
            let blk = &qv.blocks[0];
            (
                blk.codes.clone(),
                blk.scales.clone(),
                blk.quaternions.clone(),
                blk.norms.clone(),
            )
        }
        _ => panic!("expected KvStorage::IsoV3 after iso3 hydrate"),
    };

    assert_eq!(
        v_codes_before, v_codes_after,
        "iso3 V codes_packed are not bit-identical after SSD round-trip"
    );
    // scales / quaternions / norms are f32 and stored via to_le_bytes — should
    // be bit-identical. Allow a hair of tolerance to guard against any
    // future platform-specific f32 serialisation difference.
    let max_scale_err = v_scales_before
        .iter()
        .zip(&v_scales_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_scale_err < 1e-6,
        "iso3 V scales round-trip err {max_scale_err}"
    );

    let max_quat_err = v_quats_before
        .iter()
        .zip(&v_quats_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_quat_err < 1e-6,
        "iso3 V quaternions round-trip err {max_quat_err}"
    );

    let max_norm_err = v_norms_before
        .iter()
        .zip(&v_norms_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_norm_err < 1e-6,
        "iso3 V norms round-trip err {max_norm_err}"
    );

    // K dequant must also match within tolerance.
    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(
        k_recon_before.len(),
        k_recon_after.len(),
        "iso3: K length mismatch"
    );
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_k_err < 1e-3,
        "iso3: K dequant round-trip error {max_k_err} too large"
    );

    let _ = std::fs::remove_file(&path);
}

/// Iso4 SSD round-trip: build → write → read → assert all four V-side buffers
/// (codes_packed, scales, quaternions, norms) are bit/value-identical, and K
/// dequant matches within tolerance.
///
/// Mirrors `roundtrip_iso3` exactly with `KvQuant::Iso4` / `KvStorage::IsoV4`.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the IsoV4 variant only; wildcard panics on shape drift"
)]
fn roundtrip_iso4() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::Iso4, shape, 0xA129_F12C, device);

    let (v_codes_before, v_scales_before, v_quats_before, v_norms_before) = match &storage {
        KvStorage::IsoV4 { v: Some(qv), .. } => {
            let blk = &qv.blocks[0];
            (
                blk.codes.clone(),
                blk.scales.clone(),
                blk.quaternions.clone(),
                blk.norms.clone(),
            )
        }
        _ => panic!("expected KvStorage::IsoV4 for Iso4"),
    };

    let layers = vec![storage];
    let path = tmp_path("iso4");
    KvBlockWriter::new(MODEL_ID, KvQuant::Iso4, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, KvQuant::Iso4, device).unwrap();
    assert_eq!(rebuilt.len(), 1, "iso4: layer count");

    let (v_codes_after, v_scales_after, v_quats_after, v_norms_after) = match &rebuilt[0] {
        KvStorage::IsoV4 { v: Some(qv), .. } => {
            let blk = &qv.blocks[0];
            (
                blk.codes.clone(),
                blk.scales.clone(),
                blk.quaternions.clone(),
                blk.norms.clone(),
            )
        }
        _ => panic!("expected KvStorage::IsoV4 after iso4 hydrate"),
    };

    assert_eq!(
        v_codes_before, v_codes_after,
        "iso4 V codes_packed are not bit-identical after SSD round-trip"
    );

    let max_scale_err = v_scales_before
        .iter()
        .zip(&v_scales_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_scale_err < 1e-6,
        "iso4 V scales round-trip err {max_scale_err}"
    );

    let max_quat_err = v_quats_before
        .iter()
        .zip(&v_quats_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_quat_err < 1e-6,
        "iso4 V quaternions round-trip err {max_quat_err}"
    );

    let max_norm_err = v_norms_before
        .iter()
        .zip(&v_norms_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_norm_err < 1e-6,
        "iso4 V norms round-trip err {max_norm_err}"
    );

    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(
        k_recon_before.len(),
        k_recon_after.len(),
        "iso4: K length mismatch"
    );
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_k_err < 1e-3,
        "iso4: K dequant round-trip error {max_k_err} too large"
    );

    let _ = std::fs::remove_file(&path);
}

/// Rotor3 SSD round-trip: build → write → read → assert all four V-side
/// buffers (codes_packed, scales, norms, rotors) are bit/value-identical, and
/// K dequant matches within tolerance.
///
/// Mirrors `roundtrip_iso4` with `KvQuant::Rotor3` / `KvStorage::RotorV3`.
/// The fourth V buffer here is `rotors` (the static rotor table) rather than
/// `quaternions` — same wire layout idea, different semantics.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the RotorV3 variant only; wildcard panics on shape drift"
)]
fn roundtrip_rotor3() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 96];
    let (storage, k_recon_before) = build_storage(KvQuant::Rotor3, shape, 0xA130_F13C, device);

    let (v_codes_before, v_scales_before, v_norms_before, v_rotors_before) = match &storage {
        KvStorage::RotorV3 { v: Some(qv), .. } => {
            let blk = &qv.blocks[0];
            (
                blk.codes.clone(),
                blk.scales.clone(),
                blk.norms.clone(),
                qv.rotors.clone(),
            )
        }
        _ => panic!("expected KvStorage::RotorV3 for Rotor3"),
    };

    let layers = vec![storage];
    let path = tmp_path("rotor3");
    KvBlockWriter::new(MODEL_ID, KvQuant::Rotor3, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, KvQuant::Rotor3, device).unwrap();
    assert_eq!(rebuilt.len(), 1, "rotor3: layer count");

    let (v_codes_after, v_scales_after, v_norms_after, v_rotors_after) = match &rebuilt[0] {
        KvStorage::RotorV3 { v: Some(qv), .. } => {
            let blk = &qv.blocks[0];
            (
                blk.codes.clone(),
                blk.scales.clone(),
                blk.norms.clone(),
                qv.rotors.clone(),
            )
        }
        _ => panic!("expected KvStorage::RotorV3 after rotor3 hydrate"),
    };

    assert_eq!(
        v_codes_before, v_codes_after,
        "rotor3 V codes_packed are not bit-identical after SSD round-trip"
    );

    let max_scale_err = v_scales_before
        .iter()
        .zip(&v_scales_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_scale_err < 1e-6,
        "rotor3 V scales round-trip err {max_scale_err}"
    );

    let max_norm_err = v_norms_before
        .iter()
        .zip(&v_norms_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_norm_err < 1e-6,
        "rotor3 V norms round-trip err {max_norm_err}"
    );

    let max_rotors_err = v_rotors_before
        .iter()
        .zip(&v_rotors_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_rotors_err < 1e-6,
        "rotor3 V rotor table round-trip err {max_rotors_err}"
    );

    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(
        k_recon_before.len(),
        k_recon_after.len(),
        "rotor3: K length mismatch"
    );
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_k_err < 1e-3,
        "rotor3: K dequant round-trip error {max_k_err} too large"
    );

    let _ = std::fs::remove_file(&path);
}

/// RotorV4 round-trip: build → write → read → verify codes/scales/norms/rotors
/// are bit-exact and K dequant is within q8_0 tolerance.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the RotorV4 variant only; wildcard panics on shape drift"
)]
fn roundtrip_rotor4() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 96];
    let (storage, k_recon_before) = build_storage(KvQuant::Rotor4, shape, 0xA131_F14C, device);

    let (v_codes_before, v_scales_before, v_norms_before, v_rotors_before) = match &storage {
        KvStorage::RotorV4 { v: Some(qv), .. } => {
            let blk = &qv.blocks[0];
            (
                blk.codes.clone(),
                blk.scales.clone(),
                blk.norms.clone(),
                qv.rotors.clone(),
            )
        }
        _ => panic!("expected KvStorage::RotorV4 for Rotor4"),
    };

    let layers = vec![storage];
    let path = tmp_path("rotor4");
    KvBlockWriter::new(MODEL_ID, KvQuant::Rotor4, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, KvQuant::Rotor4, device).unwrap();
    assert_eq!(rebuilt.len(), 1, "rotor4: layer count");

    let (v_codes_after, v_scales_after, v_norms_after, v_rotors_after) = match &rebuilt[0] {
        KvStorage::RotorV4 { v: Some(qv), .. } => {
            let blk = &qv.blocks[0];
            (
                blk.codes.clone(),
                blk.scales.clone(),
                blk.norms.clone(),
                qv.rotors.clone(),
            )
        }
        _ => panic!("expected KvStorage::RotorV4 after rotor4 hydrate"),
    };

    assert_eq!(
        v_codes_before, v_codes_after,
        "rotor4 V codes_packed are not bit-identical after SSD round-trip"
    );

    let max_scale_err = v_scales_before
        .iter()
        .zip(&v_scales_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_scale_err < 1e-6,
        "rotor4 V scales round-trip err {max_scale_err}"
    );

    let max_norm_err = v_norms_before
        .iter()
        .zip(&v_norms_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_norm_err < 1e-6,
        "rotor4 V norms round-trip err {max_norm_err}"
    );

    let max_rotors_err = v_rotors_before
        .iter()
        .zip(&v_rotors_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_rotors_err < 1e-6,
        "rotor4 V rotor table round-trip err {max_rotors_err}"
    );

    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(
        k_recon_before.len(),
        k_recon_after.len(),
        "rotor4: K length mismatch"
    );
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_k_err < 1e-3,
        "rotor4: K dequant round-trip error {max_k_err} too large"
    );

    let _ = std::fs::remove_file(&path);
}

/// `None` (bf16) keeps K/V on the parent KvCache, not in storage. Verify the
/// geometry round-trips and the variant reconstructs as `None`.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn roundtrip_none() {
    let device = Device::Cpu;
    let layers = vec![KvStorage::None { max_seq: 4096 }];
    let path = tmp_path("none");
    KvBlockWriter::new(MODEL_ID, KvQuant::None, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader.hydrate(MODEL_ID, KvQuant::None, device).unwrap();
    assert!(matches!(rebuilt[0], KvStorage::None { max_seq: 4096 }));
    let _ = std::fs::remove_file(&path);
}

/// `KvQuant::None` keeps its live K/V as bf16 on the parent `KvCache`
/// (`decode_fp16_{k,v}`), off the geometry-only `KvStorage::None` buffer. The
/// spill path must persist that bf16 prefix and hydrate must re-seed it, or an
/// exact-hit SSD replay reads zeros and decodes garbage.
///
/// This drives the REAL `write_caches` (spill bridge) + `read_caches` (hydrate
/// bridge) on disk — not the storage-only `KvBlockWriter` struct path — so the
/// `KvCache::decode_fp16_{k,v}` capture/restore seam is exercised end-to-end.
/// Uses `kv_h > 1` and two layers so a head-axis or per-layer scramble would
/// surface. bf16 round-trips bit-exact, so the assert is exact equality.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a missing bf16 seed is exactly the failure this test must surface, so a panic with the diagnostic message is the intended outcome"
)]
fn roundtrip_none_bf16_payload_via_spill_hydrate() {
    let device = Device::Cpu;
    // [B, kv_h, S, D] with kv_h > 1 so a head-axis transpose would be caught.
    let shape = [1i32, 8, 32, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();

    // Two None-storage layers, each with a distinct bf16 K/V pair. bf16 cast up
    // front so the round-trip is bit-exact (the cache stores exactly these
    // buffers; no quantisation is applied on the None path).
    let mut kv_caches = Vec::new();
    let mut want: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for layer in 0..2u64 {
        let k_bf16 = arr(&lcg(n, 0x511 ^ layer), &shape)
            .astype(Dtype::Bf16, device)
            .unwrap();
        let v_bf16 = arr(&lcg(n, 0x9A7 ^ layer), &shape)
            .astype(Dtype::Bf16, device)
            .unwrap();
        // Reference values read back through bf16 so equality is exact.
        let k_ref = to_vec(&k_bf16.astype(Dtype::F32, device).unwrap());
        let v_ref = to_vec(&v_bf16.astype(Dtype::F32, device).unwrap());
        want.push((k_ref, v_ref));

        let cache = KvCache::with_quant_max_seq(KvQuant::None, 4096)
            .with_layer_idx(layer as usize)
            .with_decode_fp16_seed(k_bf16, v_bf16);
        kv_caches.push(cache);
    }

    let path = tmp_path("none_bf16");
    write_caches(&path, device, MODEL_ID, KvQuant::None, &kv_caches, &[]).unwrap();

    // Sanity: the spilled file must carry real K/V tensors, not geometry-only.
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, bf16_seeds, _lin) = reader.hydrate(MODEL_ID, KvQuant::None, device).unwrap();
    assert_eq!(rebuilt.len(), 2, "layer count");
    for layer in 0..2 {
        assert!(
            matches!(rebuilt[layer], KvStorage::None { .. }),
            "layer {layer} reconstructs as None storage"
        );
        let (k_hyd, v_hyd) = bf16_seeds[layer]
            .as_ref()
            .expect("None layer must carry a restored bf16 K/V seed");
        let k_got = to_vec(&k_hyd.astype(Dtype::F32, device).unwrap());
        let v_got = to_vec(&v_hyd.astype(Dtype::F32, device).unwrap());
        assert_eq!(
            k_got, want[layer].0,
            "layer {layer} K bf16 must round-trip exactly"
        );
        assert_eq!(
            v_got, want[layer].1,
            "layer {layer} V bf16 must round-trip exactly"
        );
    }

    // And the full bridge re-seeds the parent KvCache: a hydrated None cache
    // must expose the restored bf16 K/V via `decode_fp16_kv`, with `offset`
    // set to the spilled seq length (32 tokens) so decode resumes at the right
    // position.
    let (hydrated, _lin) = read_caches(&path, device, MODEL_ID, KvQuant::None).unwrap();
    assert_eq!(hydrated.len(), 2, "bridge layer count");
    for layer in 0..2 {
        assert_eq!(
            hydrated[layer].offset(),
            shape[2],
            "layer {layer} offset must equal spilled seq_len"
        );
        let (k_hyd, v_hyd) = hydrated[layer]
            .decode_fp16_kv()
            .expect("hydrated None cache must carry bf16 decode K/V");
        let k_got = to_vec(&k_hyd.astype(Dtype::F32, device).unwrap());
        let v_got = to_vec(&v_hyd.astype(Dtype::F32, device).unwrap());
        assert_eq!(k_got, want[layer].0, "layer {layer} bridge K mismatch");
        assert_eq!(v_got, want[layer].1, "layer {layer} bridge V mismatch");
    }

    let _ = std::fs::remove_file(&path);
}

/// Paged: build a paged K8V4 storage directly, write → read → dequant K.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn roundtrip_paged() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k_data = lcg(n, 0xFEED);
    let v_data = lcg(n, 0xBEEF);

    // Build a Paged K8V4 storage by quantizing K (q8_0) and V (turbo4) and
    // appending into the paged structures directly.
    use rmlx_kv_quant::q8::q8_quantize;
    use rmlx_kv_quant::turboquant::turbo_quantize_v;

    let (k_codes, k_scales) = q8_quantize(&k_data);
    let mut pk = PagedKStorage::new(4096, 32, 4);
    // Build codes/scales arrays as the paged append expects (flat u32/f32).
    let k_codes_arr = u8_codes_to_u32_array(&k_codes, device);
    let k_scales_arr = arr(&k_scales, &[k_scales.len() as i32]);
    pk.append(shape, k_codes_arr, k_scales_arr, device).unwrap();

    let vblk = turbo_quantize_v(&v_data, 4, shape).unwrap();
    let mut pv = PagedVStorage::new(4096, 32, 4, 4);
    let v_codes_arr = u8_codes_to_u32_array(&vblk.codes, device);
    let v_scales_arr = arr(&vblk.scales, &[vblk.scales.len() as i32]);
    pv.append(shape, v_codes_arr, v_scales_arr, device).unwrap();

    let storage = KvStorage::Paged {
        quant: KvQuant::K8V4,
        k: Some(pk),
        v_k8: Some(Box::new(pv)),
        v_planar: None,
        max_seq: 4096,
    };
    let before = dequant_k(&storage, device);

    let layers = vec![storage];
    let path = tmp_path("paged");
    KvBlockWriter::new(MODEL_ID, KvQuant::K8V4, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader.hydrate(MODEL_ID, KvQuant::K8V4, device).unwrap();
    let after = dequant_k(&rebuilt[0], device);
    assert_eq!(before.len(), after.len(), "paged K length");
    let max_err = before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-3, "paged K round-trip error {max_err}");
    let _ = std::fs::remove_file(&path);
}

// Reinterpret a u8 q8/turbo codes blob as a u32 array (4 bytes/word, LE).
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn u8_codes_to_u32_array(codes: &[u8], _device: Device) -> Array {
    assert_eq!(codes.len() % 4, 0, "codes len must be multiple of 4");
    Array::from_bytes(codes, &[(codes.len() / 4) as i32], Dtype::U32).unwrap()
}

/// LinearAttn (GDN): full recurrent state round-trips whole, untruncated.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn roundtrip_linear_attn() {
    let device = Device::Cpu;
    let conv = arr(&lcg(24, 1), &[1, 3, 8]);
    let delta = arr(&lcg(48, 2), &[1, 2, 4, 6]);
    let mut lac = LinearAttnCache::new();
    lac.conv_state = Some(conv);
    lac.delta_state = Some(delta);
    let conv_before = to_vec(lac.conv_state.as_ref().unwrap());
    let delta_before = to_vec(lac.delta_state.as_ref().unwrap());

    let layers: Vec<KvStorage> = vec![KvStorage::None { max_seq: 4096 }];
    let lin = vec![lac];
    let path = tmp_path("gdn");
    KvBlockWriter::new(MODEL_ID, KvQuant::None, &layers, &lin)
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (_layers, _bf16, rebuilt_lin) = reader.hydrate(MODEL_ID, KvQuant::None, device).unwrap();
    assert_eq!(rebuilt_lin.len(), 1, "linear cache count");
    assert_eq!(
        to_vec(rebuilt_lin[0].conv_state.as_ref().unwrap()),
        conv_before
    );
    assert_eq!(
        to_vec(rebuilt_lin[0].delta_state.as_ref().unwrap()),
        delta_before
    );
    let _ = std::fs::remove_file(&path);
}

/// Wrong model_id load returns Err.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn wrong_model_id_rejected() {
    let device = Device::Cpu;
    let layers = vec![KvStorage::None { max_seq: 4096 }];
    let path = tmp_path("wrong_model");
    KvBlockWriter::new(MODEL_ID, KvQuant::None, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let res = reader.hydrate("OtherArch/other-snapshot", KvQuant::None, device);
    match res {
        Err(Error::Mlx(m)) => {
            assert!(m.contains("model_id mismatch"), "wrong error message: {m}");
        }
        Err(other) => panic!("expected model_id mismatch Err, got {other:?}"),
        Ok(_) => panic!("expected Err on wrong model_id, got Ok"),
    }
    let _ = std::fs::remove_file(&path);
}

// ── C3 + C1 + C2: GPU hydrate round-trips (require Metal) ─────────────────

/// C3 + C1: K8V4 write (GPU) → read → hydrate → one-step decode, no panic.
///
/// Without C3 fix: writer dumps full GPU paged capacity (≥512 words for
/// 300-token sequence) → on-disk tensor too large → hydrate reader OOB on
/// `slice_update`.
///
/// Without C1 fix: QuantV allocates zero GPU buffers on hydration → V is
/// all-zeros across history → silent attention corruption.
///
/// Requires Metal / Apple-Silicon GPU.
#[test]
#[ignore = "GPU Metal context — run: cargo test kv_cache -- c3_k8v4_hydrate_round_trip_no_panic --ignored --test-threads=1"]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn c3_k8v4_hydrate_round_trip_no_panic() {
    let device = Device::Gpu;
    // 300 tokens — fills past the 256-token page boundary so paged GPU
    // buffer capacity exceeds the filled prefix.
    let shape = [1i32, 2, 300, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = arr(&lcg(n, 0xC3_1234), &shape);
    let v = arr(&lcg(n, 0xC3_1234 ^ 0xF00D), &shape);

    let mut c = KvCache::with_quant_max_seq(KvQuant::K8V4, 4096);
    c.enter_prefill();
    c.update(&k, &v, device).unwrap();
    c.exit_prefill(device).unwrap();

    let path = tmp_path("c3_k8v4_round_trip");
    write_caches(&path, device, MODEL_ID, KvQuant::K8V4, &[c], &[]).unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader.hydrate(MODEL_ID, KvQuant::K8V4, device).unwrap();
    let storage = rebuilt.into_iter().next().unwrap();

    let mut cache = KvCache::from_storage(storage, KvQuant::K8V4, 300, 0);
    // One decode step — must not OOB-panic.
    // n1 = B*kv_h*S*D = 1*2*1*128 = 256.
    let n1 = 256usize;
    let one_k = arr(&lcg(n1, 0xAABB), &[1, 2, 1, 128]);
    let one_v = arr(&lcg(n1, 0xCCDD), &[1, 2, 1, 128]);
    let (k_out, _) = cache.update(&one_k, &one_v, device).unwrap();
    assert_eq!(k_out.shape()[2], 301, "C3/K8V4: seq should advance to 301");
    let _ = std::fs::remove_file(&path);
}

/// C2 + C3: Planar write (GPU) → read → hydrate → one-step decode, no panic.
///
/// Without C2 fix: QuantPlanarV init_cap = KV_PAGE_SIZE (256) < prev_seq
/// (300) → grow path tries to copy 300 words from a 256-word buffer → OOB
/// slice_update → broadcast error / panic.
///
/// Requires Metal / Apple-Silicon GPU.
#[test]
#[ignore = "GPU Metal context — run: cargo test kv_cache -- c2_planar_hydrate_round_trip_no_panic --ignored --test-threads=1"]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn c2_planar_hydrate_round_trip_no_panic() {
    let device = Device::Gpu;
    let shape = [1i32, 2, 300, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = arr(&lcg(n, 0xC2_5678), &shape);
    let v = arr(&lcg(n, 0xC2_5678 ^ 0xCAFE), &shape);

    let mut c = KvCache::with_quant_max_seq(KvQuant::Planar, 4096);
    c.enter_prefill();
    c.update(&k, &v, device).unwrap();
    c.exit_prefill(device).unwrap();

    let path = tmp_path("c2_planar_round_trip");
    write_caches(&path, device, MODEL_ID, KvQuant::Planar, &[c], &[]).unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader.hydrate(MODEL_ID, KvQuant::Planar, device).unwrap();
    let storage = rebuilt.into_iter().next().unwrap();

    let mut cache = KvCache::from_storage(storage, KvQuant::Planar, 300, 0);
    // n1 = B*kv_h*S*D = 1*2*1*128 = 256.
    let n1 = 256usize;
    let one_k = arr(&lcg(n1, 0xEEFF), &[1, 2, 1, 128]);
    let one_v = arr(&lcg(n1, 0x1122), &[1, 2, 1, 128]);
    let (k_out, _) = cache.update(&one_k, &one_v, device).unwrap();
    assert_eq!(
        k_out.shape()[2],
        301,
        "C2/Planar: seq should advance to 301"
    );
    let _ = std::fs::remove_file(&path);
}

/// Decisive Planar3 V-codec cross-path proof: GPU-quantize → SSD-serialize →
/// CPU-hydrate must reconstruct V within 3-bit PlanarQuant quant-noise.
///
/// This is the real-hardware analogue of the `planarquant.rs` byte-math parity
/// tests. It exercises the actual boundary the codec fix targets:
///
/// 1. A GPU-backed `QuantPlanarV` (bits=3) is built by the real Metal kernel
///    `planar_quantize_v3_gpu`, which packs codes in the shared 10-vals/u32 word
///    convention (4 u32 words / 16 bytes per group).
/// 2. The GPU code/scale/rotation buffers are serialized to a `.kvb` via the
///    live spill writer (`write_caches` → `write_quant_planar_v` GPU branch).
/// 3. The block is hydrated on `Device::Cpu`: `read_quant_planar_v` reinterprets
///    the raw GPU-word bytes as a CPU `PlanarBlocks` and the CPU
///    `planar_dequantize` decodes them.
///
/// Reference is the GPU's own dequant of the same codes (the GPU codec is
/// path-fixed — it was correct before and after the fix). The CPU-hydrated V is
/// compared against it.
///
/// FAIL on the pre-fix CPU codec: it unpacked the GPU-word byte stream with the
/// old dense `bit_offset = elem * 3` layout, scrambling indices — cross-decode
/// error ≈ 1.x. PASS now: both sides share the word convention, so the crossing
/// is quant-noise small.
///
/// Requires Metal / Apple-Silicon GPU.
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd planar3 -- --ignored --test-threads=1"]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the Planar variant only; wildcard panics on shape drift"
)]
fn planar3_v_gpu_spill_cpu_hydrate_cross_path() {
    let device = Device::Gpu;
    // Multi-head GQA-shaped, head_dim 128 (divisible by GROUP_SIZE=32), seq 128.
    let shape = [1i32, 2, 128, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = arr(&lcg(n, 0xF131_AC03), &shape);
    let v = arr(&lcg(n, 0xF131_AC03 ^ 0x5EED), &shape);

    // ── GPU encode via the real Metal planar3 kernel ──────────────────────────
    let mut c = KvCache::with_quant_max_seq(KvQuant::Planar3, 4096);
    c.enter_prefill();
    c.update(&k, &v, device).unwrap();
    c.exit_prefill(device).unwrap();

    // GPU reference V reconstruction (codec is identical pre/post fix on GPU).
    let gpu_ref_v = match c.storage() {
        KvStorage::Planar {
            v: Some(qv), bits, ..
        } => {
            assert_eq!(*bits, 3, "expected Planar3 (3-bit) V storage");
            let (_flat, arr_opt) = qv.dequantize_choice(device, Dtype::F32).unwrap();
            // GPU path always returns Some(Array); None only on the CPU branch.
            to_vec(&arr_opt.unwrap())
        }
        _ => panic!("expected KvStorage::Planar for Planar3"),
    };

    // ── Spill GPU codes to a .kvb (write_quant_planar_v GPU branch) ───────────
    let path = tmp_path("planar3_cross_path");
    write_caches(&path, device, MODEL_ID, KvQuant::Planar3, &[c], &[]).unwrap();

    // ── Hydrate on CPU: GPU-word bytes → PlanarBlocks → CPU planar_dequantize ─
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::Planar3, Device::Cpu)
        .unwrap();
    let storage = rebuilt.into_iter().next().unwrap();

    let cpu_hydrated_v = match &storage {
        KvStorage::Planar {
            v: Some(qv), bits, ..
        } => {
            assert_eq!(*bits, 3, "hydrated Planar3 storage must stay 3-bit");
            let (flat, _) = qv.dequantize_choice(Device::Cpu, Dtype::F32).unwrap();
            flat
        }
        _ => panic!("expected hydrated KvStorage::Planar"),
    };

    assert_eq!(
        gpu_ref_v.len(),
        cpu_hydrated_v.len(),
        "GPU reference and CPU-hydrated V length mismatch"
    );

    // GPU-encode vs CPU-decode of the SAME codes. With a shared word convention
    // this is f32-rounding noise between the two dequant implementations, not the
    // ~1.x scrambled-index error the dense pre-fix CPU codec produced.
    let max_err = gpu_ref_v
        .iter()
        .zip(&cpu_hydrated_v)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 5e-3,
        "planar3 GPU-spill → CPU-hydrate cross-path max abs error {max_err:.6} exceeds 5e-3 — \
         the CPU codec is not reading the GPU 10-vals/u32 word stream; an SSD hydrate corrupts V"
    );

    let _ = std::fs::remove_file(&path);
}

// ── H4 + H5: SWA offset reset on hydration (CPU-runnable) ─────────────────

/// H4 + H5: KvStorage::None (SWA) layer with offset > max_seq must reset
/// gracefully on first decode step without OOB panic.
///
/// The reset path (H5) also emits a tracing::warn! event.
/// This test runs on CPU (no GPU required).
#[test]
fn h4_swa_prev_offset_exceeds_max_seq_reset_no_panic() {
    let device = Device::Cpu;
    // Simulate a SWA layer hydrated with prev_offset=1023, max_seq=512.
    // The SWA ring buffer was not spilled; offset > max_seq triggers reset.
    let storage = KvStorage::None { max_seq: 512 };
    let mut cache = KvCache::from_storage(storage, KvQuant::None, 1023, 0);

    // n = B*kv_h*S*D = 1*2*1*128 = 256.
    let n = 256usize;
    let one_k = arr(&lcg(n, 0x1111), &[1, 2, 1, 128]);
    let one_v = arr(&lcg(n, 0x2222), &[1, 2, 1, 128]);

    // Must not panic; SWA offset is reset to [0..new_seq].
    let result = cache.update(&one_k, &one_v, device);
    assert!(
        result.is_ok(),
        "SWA hydrate with prev_offset > max_seq must not error: {:?}",
        result.err()
    );
}

/// K8VTurbo2Tcq SSD round-trip — V-side codes must survive a write → read
/// cycle byte-for-byte, and the hydrated `QuantV` must carry `use_tcq = true`
/// so any subsequent decode-step encode would re-enter the Viterbi path (not
/// silently fall back to nearest-centroid).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test scaffolding: shape established by build_storage"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by build_storage / writer / reader"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the K8VTurbo2Tcq variant only; wildcard panics on shape drift"
)]
fn roundtrip_k8vturbo2tcq() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::K8VTurbo2Tcq, shape, 0x2B2B_F00D, device);

    let v_codes_before = match &storage {
        KvStorage::K8VTurbo2Tcq { v: Some(qv), .. } => qv.blocks[0].codes.clone(),
        _ => panic!("expected K8VTurbo2Tcq storage"),
    };
    let v_bits_before = match &storage {
        KvStorage::K8VTurbo2Tcq { v: Some(qv), .. } => qv.blocks[0].bits,
        _ => unreachable!(),
    };
    assert_eq!(v_bits_before, 2, "build_storage should produce bits=2 V");

    let layers = vec![storage];
    let path = tmp_path("k8vturbo2tcq");
    KvBlockWriter::new(MODEL_ID, KvQuant::K8VTurbo2Tcq, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::K8VTurbo2Tcq, device)
        .unwrap();
    assert_eq!(rebuilt.len(), 1, "layer count");

    let (v_codes_after, v_bits_after, use_tcq_after) = match &rebuilt[0] {
        KvStorage::K8VTurbo2Tcq { v: Some(qv), .. } => {
            (qv.blocks[0].codes.clone(), qv.blocks[0].bits, qv.use_tcq)
        }
        _ => panic!("expected K8VTurbo2Tcq after hydrate"),
    };
    assert_eq!(
        v_bits_after, 2,
        "K8VTurbo2Tcq V must reconstruct with bits=2"
    );
    assert!(
        use_tcq_after,
        "K8VTurbo2Tcq hydrated QuantV must carry use_tcq=true so post-hydrate \
         decode-step encodes stay on the Viterbi path"
    );
    assert_eq!(
        v_codes_before, v_codes_after,
        "K8VTurbo2Tcq V codes are not byte-identical after SSD round-trip"
    );

    let _ = std::fs::remove_file(&path);
}

/// Wrong kv_quant load returns Err.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn wrong_kv_quant_rejected() {
    let device = Device::Cpu;
    let layers = vec![KvStorage::None { max_seq: 4096 }];
    let path = tmp_path("wrong_quant");
    KvBlockWriter::new(MODEL_ID, KvQuant::None, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let res = reader.hydrate(MODEL_ID, KvQuant::K8V8, device);
    match res {
        Err(Error::Mlx(m)) => {
            assert!(m.contains("kv_quant mismatch"), "wrong error message: {m}");
        }
        Err(other) => panic!("expected kv_quant mismatch Err, got {other:?}"),
        Ok(_) => panic!("expected Err on wrong kv_quant, got Ok"),
    }
    let _ = std::fs::remove_file(&path);
}

// ── K-side IsoQuant SSD round-trips ──────────────────────────────────────────

/// Iso3Sym (K iso3 + V iso3) SSD round-trip.
/// All four K-side buffers and all four V-side buffers survive the spill/
/// hydrate cycle bit-identically. K dequant matches within tolerance.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test scaffolding: shape established by build_storage"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by build_storage / writer / reader"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the IsoSym3 variant only; wildcard panics on shape drift"
)]
fn roundtrip_iso_sym_3() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::Iso3Sym, shape, 0x5722_3A1F, device);

    let (k_codes_before, v_codes_before) = match &storage {
        KvStorage::IsoSym3 {
            k: Some(qk),
            v: Some(qv),
            ..
        } => (qk.blocks[0].codes.clone(), qv.blocks[0].codes.clone()),
        _ => panic!("expected IsoSym3 storage"),
    };

    let layers = vec![storage];
    let path = tmp_path("iso_sym_3");
    KvBlockWriter::new(MODEL_ID, KvQuant::Iso3Sym, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader.hydrate(MODEL_ID, KvQuant::Iso3Sym, device).unwrap();
    assert_eq!(rebuilt.len(), 1, "iso_sym_3: layer count");

    let (k_codes_after, v_codes_after) = match &rebuilt[0] {
        KvStorage::IsoSym3 {
            k: Some(qk),
            v: Some(qv),
            ..
        } => (qk.blocks[0].codes.clone(), qv.blocks[0].codes.clone()),
        _ => panic!("expected IsoSym3 after hydrate"),
    };
    assert_eq!(
        k_codes_before, k_codes_after,
        "iso_sym_3 K codes bit-identical"
    );
    assert_eq!(
        v_codes_before, v_codes_after,
        "iso_sym_3 V codes bit-identical"
    );

    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(k_recon_before.len(), k_recon_after.len());
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_k_err < 1e-3, "iso_sym_3 K dequant err {max_k_err}");

    let _ = std::fs::remove_file(&path);
}

/// Iso4Sym SSD round-trip (4-bit K + 4-bit V).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test scaffolding: shape established by build_storage"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by build_storage / writer / reader"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the IsoSym4 variant only; wildcard panics on shape drift"
)]
fn roundtrip_iso_sym_4() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::Iso4Sym, shape, 0x5722_3A20, device);

    let (k_codes_before, v_codes_before) = match &storage {
        KvStorage::IsoSym4 {
            k: Some(qk),
            v: Some(qv),
            ..
        } => (qk.blocks[0].codes.clone(), qv.blocks[0].codes.clone()),
        _ => panic!("expected IsoSym4 storage"),
    };

    let layers = vec![storage];
    let path = tmp_path("iso_sym_4");
    KvBlockWriter::new(MODEL_ID, KvQuant::Iso4Sym, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader.hydrate(MODEL_ID, KvQuant::Iso4Sym, device).unwrap();
    assert_eq!(rebuilt.len(), 1);

    let (k_codes_after, v_codes_after) = match &rebuilt[0] {
        KvStorage::IsoSym4 {
            k: Some(qk),
            v: Some(qv),
            ..
        } => (qk.blocks[0].codes.clone(), qv.blocks[0].codes.clone()),
        _ => panic!("expected IsoSym4 after hydrate"),
    };
    assert_eq!(k_codes_before, k_codes_after);
    assert_eq!(v_codes_before, v_codes_after);

    let k_recon_after = dequant_k(&rebuilt[0], device);
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_k_err < 1e-3, "iso_sym_4 K dequant err {max_k_err}");

    let _ = std::fs::remove_file(&path);
}

/// IsoKOnly3 SSD round-trip (K iso3, V bf16 off-storage).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test scaffolding: shape established by build_storage"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by build_storage / writer / reader"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the IsoKOnly3 variant only; wildcard panics on shape drift"
)]
fn roundtrip_iso_k_only_3() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::IsoKOnly3, shape, 0x5722_3A21, device);

    let k_codes_before = match &storage {
        KvStorage::IsoKOnly3 { k: Some(qk), .. } => qk.blocks[0].codes.clone(),
        _ => panic!("expected IsoKOnly3 storage"),
    };

    let layers = vec![storage];
    let path = tmp_path("iso_k_only_3");
    KvBlockWriter::new(MODEL_ID, KvQuant::IsoKOnly3, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::IsoKOnly3, device)
        .unwrap();
    assert_eq!(rebuilt.len(), 1);

    let k_codes_after = match &rebuilt[0] {
        KvStorage::IsoKOnly3 { k: Some(qk), .. } => qk.blocks[0].codes.clone(),
        _ => panic!("expected IsoKOnly3 after hydrate"),
    };
    assert_eq!(k_codes_before, k_codes_after);

    let k_recon_after = dequant_k(&rebuilt[0], device);
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_k_err < 1e-3);

    let _ = std::fs::remove_file(&path);
}

/// IsoKOnly4 SSD round-trip.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test scaffolding: shape established by build_storage"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by build_storage / writer / reader"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the IsoKOnly4 variant only; wildcard panics on shape drift"
)]
fn roundtrip_iso_k_only_4() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::IsoKOnly4, shape, 0x5722_3A22, device);

    let k_codes_before = match &storage {
        KvStorage::IsoKOnly4 { k: Some(qk), .. } => qk.blocks[0].codes.clone(),
        _ => panic!("expected IsoKOnly4 storage"),
    };

    let layers = vec![storage];
    let path = tmp_path("iso_k_only_4");
    KvBlockWriter::new(MODEL_ID, KvQuant::IsoKOnly4, &layers, &[])
        .write(&path, device)
        .unwrap();

    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::IsoKOnly4, device)
        .unwrap();
    assert_eq!(rebuilt.len(), 1);

    let k_codes_after = match &rebuilt[0] {
        KvStorage::IsoKOnly4 { k: Some(qk), .. } => qk.blocks[0].codes.clone(),
        _ => panic!("expected IsoKOnly4 after hydrate"),
    };
    assert_eq!(k_codes_before, k_codes_after);

    let k_recon_after = dequant_k(&rebuilt[0], device);
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_k_err < 1e-3);

    let _ = std::fs::remove_file(&path);
}

// ── Rotor K-side SSD round-trip tests ────────────────────────────────────────

/// Rotor3Sym SSD round-trip with QJL OFF.
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_sym_3_no_qjl() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, k_recon_before) = build_storage(KvQuant::Rotor3Sym, shape, 0xA140_2301, device);
    let k_codes_before = match &storage {
        KvStorage::RotorSym3 { k: Some(qk), .. } => qk.blocks[0].codes.clone(),
        _ => panic!("expected RotorSym3 storage"),
    };
    let layers = vec![storage];
    let path = tmp_path("rotor_sym_3_no_qjl");
    KvBlockWriter::new(MODEL_ID, KvQuant::Rotor3Sym, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::Rotor3Sym, device)
        .unwrap();
    let (k_codes_after, use_qjl_after) = match &rebuilt[0] {
        KvStorage::RotorSym3 { k: Some(qk), .. } => (qk.blocks[0].codes.clone(), qk.use_qjl()),
        _ => panic!("expected RotorSym3 after hydrate"),
    };
    assert_eq!(k_codes_before, k_codes_after);
    assert!(!use_qjl_after, "QJL must hydrate as OFF");
    let k_recon_after = dequant_k(&rebuilt[0], device);
    let max_k_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_k_err < 1e-3, "rotor_sym_3 K dequant err {max_k_err}");
    let _ = std::fs::remove_file(&path);
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// Rotor3Sym SSD round-trip with QJL ON (explicitly enabled — QJL is off by
/// default).
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_sym_3_qjl() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::Rotor3Sym, shape, 0xA140_2302, device);
    let (k_codes_before, qjl_codes_before) = match &storage {
        KvStorage::RotorSym3 { k: Some(qk), .. } => {
            (qk.blocks[0].codes.clone(), qk.blocks[0].qjl_codes.clone())
        }
        _ => panic!("expected RotorSym3 storage"),
    };
    assert!(
        !qjl_codes_before.is_empty(),
        "QJL sideband must be present at build time when explicitly enabled"
    );
    let layers = vec![storage];
    let path = tmp_path("rotor_sym_3_qjl");
    KvBlockWriter::new(MODEL_ID, KvQuant::Rotor3Sym, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::Rotor3Sym, device)
        .unwrap();
    let (k_codes_after, qjl_codes_after, use_qjl_after) = match &rebuilt[0] {
        KvStorage::RotorSym3 { k: Some(qk), .. } => (
            qk.blocks[0].codes.clone(),
            qk.blocks[0].qjl_codes.clone(),
            qk.use_qjl(),
        ),
        _ => panic!("expected RotorSym3 after hydrate"),
    };
    assert_eq!(k_codes_before, k_codes_after);
    assert_eq!(
        qjl_codes_before, qjl_codes_after,
        "QJL signs must hydrate bit-identically"
    );
    assert!(use_qjl_after, "use_qjl must be ON after hydrate");
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// Rotor4Sym SSD round-trip with QJL ON (explicitly enabled).
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_sym_4_qjl() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::Rotor4Sym, shape, 0xA140_2303, device);
    let k_codes_before = match &storage {
        KvStorage::RotorSym4 { k: Some(qk), .. } => qk.blocks[0].codes.clone(),
        _ => panic!("expected RotorSym4"),
    };
    let layers = vec![storage];
    let path = tmp_path("rotor_sym_4_qjl");
    KvBlockWriter::new(MODEL_ID, KvQuant::Rotor4Sym, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::Rotor4Sym, device)
        .unwrap();
    let (k_codes_after, use_qjl_after) = match &rebuilt[0] {
        KvStorage::RotorSym4 { k: Some(qk), .. } => (qk.blocks[0].codes.clone(), qk.use_qjl()),
        _ => panic!("expected RotorSym4 after hydrate"),
    };
    assert_eq!(k_codes_before, k_codes_after);
    assert!(use_qjl_after);
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// Rotor4Sym SSD round-trip with QJL OFF.
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_sym_4_no_qjl() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::Rotor4Sym, shape, 0xA140_2304, device);
    let layers = vec![storage];
    let path = tmp_path("rotor_sym_4_no_qjl");
    KvBlockWriter::new(MODEL_ID, KvQuant::Rotor4Sym, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::Rotor4Sym, device)
        .unwrap();
    let use_qjl_after = match &rebuilt[0] {
        KvStorage::RotorSym4 { k: Some(qk), .. } => qk.use_qjl(),
        _ => panic!("expected RotorSym4 after hydrate"),
    };
    assert!(!use_qjl_after);
    let _ = std::fs::remove_file(&path);
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// RotorK3Asym SSD round-trip at (v_bits=4, v_group_size=64). Verifies the
/// layout tag carries the V-side (bits, group) suffix so the reader can
/// dispatch the correct affine V codec on hydrate.
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_k3_asym_v4_g64() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let kq = KvQuant::RotorK3Asym {
        v_bits: 4,
        v_group_size: 64,
    };
    let (storage, k_recon_before) = build_storage(kq, shape, 0xA158_3104, device);
    let layers = vec![storage];
    let path = tmp_path("rotor_k3_asym_v4_g64");
    KvBlockWriter::new(MODEL_ID, kq, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader.hydrate(MODEL_ID, kq, device).unwrap();
    assert_eq!(rebuilt.len(), 1);
    let (vb_after, vg_after) = match &rebuilt[0] {
        KvStorage::RotorKAsym3 {
            v_bits,
            v_group_size,
            ..
        } => (*v_bits, *v_group_size),
        _ => panic!("expected RotorKAsym3 after hydrate"),
    };
    assert_eq!(vb_after, 4);
    assert_eq!(vg_after, 64);
    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(k_recon_before.len(), k_recon_after.len());
    let max_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-3,
        "rotor_k3_asym K dequant round-trip error {max_err} too large"
    );
    let _ = std::fs::remove_file(&path);
}

/// RotorK4Asym SSD round-trip at (v_bits=3, v_group_size=64).
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_k4_asym_v3_g64() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let kq = KvQuant::RotorK4Asym {
        v_bits: 3,
        v_group_size: 64,
    };
    let (storage, k_recon_before) = build_storage(kq, shape, 0xA158_3105, device);
    let layers = vec![storage];
    let path = tmp_path("rotor_k4_asym_v3_g64");
    KvBlockWriter::new(MODEL_ID, kq, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader.hydrate(MODEL_ID, kq, device).unwrap();
    assert_eq!(rebuilt.len(), 1);
    let (vb_after, vg_after) = match &rebuilt[0] {
        KvStorage::RotorKAsym4 {
            v_bits,
            v_group_size,
            ..
        } => (*v_bits, *v_group_size),
        _ => panic!("expected RotorKAsym4 after hydrate"),
    };
    assert_eq!(vb_after, 3);
    assert_eq!(vg_after, 64);
    let k_recon_after = dequant_k(&rebuilt[0], device);
    assert_eq!(k_recon_before.len(), k_recon_after.len());
    let max_err = k_recon_before
        .iter()
        .zip(&k_recon_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-3);
    let _ = std::fs::remove_file(&path);
}

/// RotorKOnly3 SSD round-trip with QJL ON (explicitly enabled).
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_k_only_3_qjl() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::RotorKOnly3, shape, 0xA140_2305, device);
    let layers = vec![storage];
    let path = tmp_path("rotor_k_only_3_qjl");
    KvBlockWriter::new(MODEL_ID, KvQuant::RotorKOnly3, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::RotorKOnly3, device)
        .unwrap();
    let use_qjl_after = match &rebuilt[0] {
        KvStorage::RotorKOnly3 { k: Some(qk), .. } => qk.use_qjl(),
        _ => panic!("expected RotorKOnly3 after hydrate"),
    };
    assert!(use_qjl_after);
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// RotorKOnly3 SSD round-trip with QJL OFF.
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_k_only_3_no_qjl() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::RotorKOnly3, shape, 0xA140_2306, device);
    let layers = vec![storage];
    let path = tmp_path("rotor_k_only_3_no_qjl");
    KvBlockWriter::new(MODEL_ID, KvQuant::RotorKOnly3, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::RotorKOnly3, device)
        .unwrap();
    let use_qjl_after = match &rebuilt[0] {
        KvStorage::RotorKOnly3 { k: Some(qk), .. } => qk.use_qjl(),
        _ => panic!("expected RotorKOnly3 after hydrate"),
    };
    assert!(!use_qjl_after);
    let _ = std::fs::remove_file(&path);
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// RotorKOnly4 SSD round-trip with QJL ON (explicitly enabled).
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_k_only_4_qjl() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::RotorKOnly4, shape, 0xA140_2307, device);
    let layers = vec![storage];
    let path = tmp_path("rotor_k_only_4_qjl");
    KvBlockWriter::new(MODEL_ID, KvQuant::RotorKOnly4, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::RotorKOnly4, device)
        .unwrap();
    let use_qjl_after = match &rebuilt[0] {
        KvStorage::RotorKOnly4 { k: Some(qk), .. } => qk.use_qjl(),
        _ => panic!("expected RotorKOnly4 after hydrate"),
    };
    assert!(use_qjl_after);
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// RotorKOnly4 SSD round-trip with QJL OFF.
#[test]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
fn roundtrip_rotor_k_only_4_no_qjl() {
    let _guard = ROTOR_QJL_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 128];
    let (storage, _) = build_storage(KvQuant::RotorKOnly4, shape, 0xA140_2308, device);
    let layers = vec![storage];
    let path = tmp_path("rotor_k_only_4_no_qjl");
    KvBlockWriter::new(MODEL_ID, KvQuant::RotorKOnly4, &layers, &[])
        .write(&path, device)
        .unwrap();
    let reader = KvBlockReader::open(&path).unwrap();
    let (rebuilt, _bf16, _) = reader
        .hydrate(MODEL_ID, KvQuant::RotorKOnly4, device)
        .unwrap();
    let use_qjl_after = match &rebuilt[0] {
        KvStorage::RotorKOnly4 { k: Some(qk), .. } => qk.use_qjl(),
        _ => panic!("expected RotorKOnly4 after hydrate"),
    };
    assert!(!use_qjl_after);
    let _ = std::fs::remove_file(&path);
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// SSD round-trip preserves `layer_idx` positionally.
///
/// Builds N rotor3 caches with distinct `layer_idx` values via
/// [`KvCache::from_storage`], spills via [`write_caches`], then hydrates via
/// [`read_caches`] and asserts that every hydrated cache's `layer_idx` matches
/// the original. This is the integration-level contract for `write_caches`:
/// the on-disk `.kvb` format does not persist `layer_idx`, so the hydrate path
/// reconstructs it positionally. Out-of-order spill would scramble rotor3
/// seeds at hydrate — this test would catch that regression.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: N small, fixed-size loop"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: panic on unexpected error is the desired failure mode"
)]
fn ssd_roundtrip_preserves_layer_idx_positional() {
    let device = Device::Cpu;
    let shape = &[1i32, 2, 4, 96];
    let n_layers = 4usize;

    // Build n_layers rotor3 storages (distinct seeds), wrap each as a KvCache
    // with `layer_idx = i`. The `offset` argument matches the recorded
    // sequence dimension (shape[2]=4).
    let mut caches: Vec<KvCache> = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let (storage, _) = build_storage(KvQuant::Rotor3, shape, 0xA154_0000 ^ (i as u64), device);
        caches.push(KvCache::from_storage(storage, KvQuant::Rotor3, 4, i));
    }

    // Sanity: pre-spill layer_idx matches.
    for (i, c) in caches.iter().enumerate() {
        assert_eq!(c.layer_idx(), i, "pre-spill: layer_idx mismatch at {i}");
    }

    // Spill (layer-ordered) → hydrate → verify positional layer_idx restoration.
    let path = tmp_path("layer_idx_positional");
    write_caches(&path, device, MODEL_ID, KvQuant::Rotor3, &caches, &[]).unwrap();
    let (hydrated, _lin) = read_caches(&path, device, MODEL_ID, KvQuant::Rotor3).unwrap();

    assert_eq!(
        hydrated.len(),
        n_layers,
        "hydrated layer count must match spill"
    );
    for (i, c) in hydrated.iter().enumerate() {
        assert_eq!(
            c.layer_idx(),
            i,
            "hydrated layer_idx must match positional index (write_caches contract)"
        );
    }

    let _ = std::fs::remove_file(&path);
}

// ── Rotor K-only ring-only tail: SSD spill preserves the full store ──────────

/// Global cosine / max-abs-err helpers for the ring-only-tail round-trip.
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Build a rotor3 K-only cache with QJL pinned OFF (env-independent: a
/// pre-seeded rotor table keeps `qjl_s_matrix == None`). This is the state the
/// fused GPU decode path requires.
#[allow(
    clippy::unwrap_used,
    reason = "test helper: values established by construction"
)]
fn seeded_rotor_k3_cache(kv_h: i32, head_dim: i32, max_seq: i32) -> KvCache {
    let n_groups = rmlx_kv_quant::rotorquant::n_groups_for(head_dim as usize);
    let rotors = rmlx_kv_quant::clifford::make_rotor_table(0, 0, n_groups);
    let storage = KvStorage::RotorKOnly3 {
        k: Some(QuantRotorK3::from_cpu_blocks(
            rotors,
            None,
            Vec::new(),
            vec![1, kv_h, 0, head_dim],
            0,
        )),
        max_seq,
    };
    KvCache::from_storage(storage, KvQuant::RotorKOnly3, 0, 0)
}

/// `(dequant, shape[2], cpu_block_tokens)` for a rotor3 K-only cache.
#[allow(
    clippy::unwrap_used,
    reason = "test helper: values established by construction"
)]
fn rotor_k3_probe(cache: &KvCache) -> (Vec<f32>, i32, usize) {
    match cache.storage() {
        KvStorage::RotorKOnly3 { k: Some(ks), .. } => (
            ks.dequant().unwrap(),
            ks.shape.get(2).copied().unwrap_or(0),
            ks.blocks.iter().map(|b| b.n_tokens).sum(),
        ),
        _ => panic!("expected a live RotorKOnly3 store"),
    }
}

/// The write path refuses to persist a rotor K store whose CPU blocks fall
/// short of `shape[2]` with no ring — a truncated store. Runs on CPU (no Metal).
///
/// Mutation check: delete the `ensure_rotor_k_blocks_cover_shape` guard in
/// `write_quant_rotor_k3` — the writer then silently serializes the short prefix
/// and this assertion flips RED (`write` returns `Ok`).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test: values established by construction earlier in this fn"
)]
fn write_rejects_truncated_rotor_k_store() {
    let kv_h = 2_i32;
    let head_dim = 9_i32; // n_groups = 3, exact
    let data = lcg((kv_h * 2 * head_dim) as usize, 0x51D);

    // A real single block covering 2 tokens.
    let mut src = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 0);
    src.append(&data, &[1, kv_h, 2, head_dim]).unwrap();

    // Claim shape[2] == 4 while the blocks cover only 2 tokens and no ring
    // exists — the ring-only tail with the ring missing.
    let truncated = QuantRotorK3::from_cpu_blocks(
        src.rotors.clone(),
        None,
        src.blocks.clone(),
        vec![1, kv_h, 4, head_dim],
        0,
    );
    let layers = vec![KvStorage::RotorKOnly3 {
        k: Some(truncated),
        max_seq: 64,
    }];
    let path = tmp_path("rotor_k_truncated");
    let res =
        KvBlockWriter::new(MODEL_ID, KvQuant::RotorKOnly3, &layers, &[]).write(&path, Device::Cpu);
    let _ = std::fs::remove_file(&path);
    assert!(
        res.is_err(),
        "writer must reject a truncated rotor K store (short blocks, no ring), got Ok"
    );
}

/// Full SSD spill/hydrate round-trip when the GPU ring is the store's only copy.
///
/// After prefill + N fused decode steps a rotor K-only store holds nothing on
/// the host: the per-step block download is skipped, and the append releases the
/// seeded prefill blocks once the ring is live, so the ring carries the whole
/// prefix. The spill clone (`KvCache::try_deep_clone`) rebuilds it from the ring
/// into complete blocks, so `write_caches` → hydrate restores the **full** store
/// — not a prefix truncated at the last CPU block. Asserted byte-exact against
/// the live store's own `dequant()`.
///
/// The empty-blocks precondition is what makes the round-trip load-bearing: with
/// a host copy still resident the clone could pass without the ring rebuild ever
/// running.
///
/// Mutation check: make `try_deep_clone` clone `self.blocks` directly (drop the
/// `synced_rotor_k_blocks` reconcile) — the clone then carries no blocks at all,
/// the ring having been the sole copy, so `write_caches` trips the
/// `ensure_rotor_k_blocks_cover_shape` guard on 0 != prefill+steps tokens and
/// `.expect("write_caches")` panics (RED).
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd rotor_k_only_ring_only_tail_ssd -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test: values established by construction earlier in this fn"
)]
fn rotor_k_only_ring_only_tail_ssd_round_trip() {
    let device = Device::Gpu;
    let (kv_h, n_q, head_dim, prefill, steps) = (2_i32, 8_i32, 128_i32, 6_i32, 5_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut c = seeded_rotor_k3_cache(kv_h, head_dim, 512);
    let k = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 11),
        &[1, kv_h, prefill, head_dim],
    );
    let v = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 12),
        &[1, kv_h, prefill, head_dim],
    );
    let q = arr(
        &lcg((prefill * n_q * head_dim) as usize, 13),
        &[1, n_q, prefill, head_dim],
    );
    c.update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .unwrap();
    c.exit_prefill(device).unwrap();
    for i in 0..steps {
        let k1 = arr(
            &lcg((kv_h * head_dim) as usize, 31 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let v1 = arr(
            &lcg((kv_h * head_dim) as usize, 41 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let q1 = arr(
            &lcg((n_q * head_dim) as usize, 51 + i as u64),
            &[1, n_q, 1, head_dim],
        );
        c.update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .unwrap()
            .eval()
            .unwrap();
    }

    // Ring-only precondition + live full-prefix dequant: the ring is the store's
    // sole copy of K, so everything below exercises the rebuild-from-ring path.
    let (orig_dq, orig_seq, block_tokens) = rotor_k3_probe(&c);
    assert_eq!(orig_seq, prefill + steps, "shape[2] advanced with the ring");
    assert_eq!(
        block_tokens, 0,
        "CPU blocks must be released once the ring is live (ring is the sole resident copy)"
    );

    // Spill clone (materialises the tail) → write → hydrate.
    let clone = c.try_deep_clone().unwrap();
    let path = tmp_path("rotor_k_ring_only_tail");
    write_caches(&path, device, MODEL_ID, KvQuant::RotorKOnly3, &[clone], &[])
        .expect("write_caches must persist the full materialised store");
    let (hydrated, _lin) = read_caches(&path, device, MODEL_ID, KvQuant::RotorKOnly3).unwrap();
    let _ = std::fs::remove_file(&path);

    let (hy_dq, hy_seq, _) = rotor_k3_probe(&hydrated[0]);
    assert_eq!(
        hy_seq,
        prefill + steps,
        "hydrated store must carry the full decoded length, not a truncated prefix"
    );
    assert_eq!(hy_dq.len(), orig_dq.len(), "hydrated K length mismatch");
    let err = max_abs_err(&orig_dq, &hy_dq);
    assert!(
        err < 1e-6,
        "hydrated K must match the live ring-only-tail dequant byte-for-byte (err={err})"
    );
}

// ── Rotor symmetric (quant-K + quant-V) ring-only tail ───────────────────────

/// Build a rotor3 **symmetric** cache with QJL pinned OFF (pre-seeded rotor
/// tables keep `qjl_s_matrix == None`). Both axes are rotor-quantized; the fused
/// quant-V decode path requires QJL off.
#[allow(
    clippy::unwrap_used,
    reason = "test helper: values established by construction"
)]
fn seeded_rotor_sym3_cache(kv_h: i32, head_dim: i32, max_seq: i32) -> KvCache {
    let n_groups = rmlx_kv_quant::rotorquant::n_groups_for(head_dim as usize);
    let k_rotors = rmlx_kv_quant::clifford::make_rotor_table(0, 0, n_groups);
    let v_rotors = rmlx_kv_quant::clifford::make_rotor_table(0, 0, n_groups);
    let storage = KvStorage::RotorSym3 {
        k: Some(QuantRotorK3::from_cpu_blocks(
            k_rotors,
            None,
            Vec::new(),
            vec![1, kv_h, 0, head_dim],
            0,
        )),
        v: Some(QuantRotorV3::from_cpu_blocks(
            v_rotors,
            Vec::new(),
            vec![1, kv_h, 0, head_dim],
            0,
        )),
        max_seq,
    };
    KvCache::from_storage(storage, KvQuant::Rotor3Sym, 0, 0)
}

/// `(k_dequant, v_dequant, shape[2], k_block_tokens, v_block_tokens)` for a
/// rotor3 symmetric cache.
#[allow(
    clippy::unwrap_used,
    reason = "test helper: values established by construction"
)]
fn rotor_sym3_probe(cache: &KvCache) -> (Vec<f32>, Vec<f32>, i32, usize, usize) {
    match cache.storage() {
        KvStorage::RotorSym3 {
            k: Some(ks),
            v: Some(vs),
            ..
        } => (
            ks.dequant().unwrap(),
            vs.dequant().unwrap(),
            ks.shape.get(2).copied().unwrap_or(0),
            ks.blocks.iter().map(|b| b.n_tokens).sum(),
            vs.blocks.iter().map(|b| b.n_tokens).sum(),
        ),
        _ => panic!("expected a live RotorSym3 store"),
    }
}

/// The write path refuses to persist a rotor **V** store whose CPU blocks fall
/// short of `shape[2]` with no ring — a truncated store. Runs on CPU (no Metal).
///
/// Mutation check: delete the `ensure_rotor_v_blocks_cover_shape` guard in
/// `write_quant_rotor_v3` — the writer then silently serializes the short prefix
/// and this assertion flips RED (`write` returns `Ok`).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test: values established by construction earlier in this fn"
)]
fn write_rejects_truncated_rotor_v_store() {
    let kv_h = 2_i32;
    let head_dim = 9_i32; // n_groups = 3, exact
    let data = lcg((kv_h * 2 * head_dim) as usize, 0x71D);

    // A real single V block covering 2 tokens.
    let mut src = QuantRotorV3::new(vec![1, kv_h, 0, head_dim], 64, 0);
    src.append(&data, &[1, kv_h, 2, head_dim]).unwrap();

    // Claim shape[2] == 4 while the blocks cover only 2 tokens and no ring
    // exists — the ring-only tail with the ring missing. Pair it with a matching
    // K store so the sym layer is well-formed on the K side.
    let mut k_src = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 0);
    let k_data = lcg((kv_h * 4 * head_dim) as usize, 0x72E);
    k_src.append(&k_data, &[1, kv_h, 4, head_dim]).unwrap();
    let truncated_v = QuantRotorV3::from_cpu_blocks(
        src.rotors.clone(),
        src.blocks.clone(),
        vec![1, kv_h, 4, head_dim],
        0,
    );
    let layers = vec![KvStorage::RotorSym3 {
        k: Some(k_src),
        v: Some(truncated_v),
        max_seq: 64,
    }];
    let path = tmp_path("rotor_v_truncated");
    let res =
        KvBlockWriter::new(MODEL_ID, KvQuant::Rotor3Sym, &layers, &[]).write(&path, Device::Cpu);
    let _ = std::fs::remove_file(&path);
    assert!(
        res.is_err(),
        "writer must reject a truncated rotor V store (short blocks, no ring), got Ok"
    );
}

/// Full SSD spill/hydrate round-trip after a **symmetric** ring-only-tail
/// decode: both K and V decode tails live only in their GPU rings, are
/// materialised by the spill clone, and hydrate byte-exact.
///
/// Also the V dequant-full-prefix rebuild proof: the live `vs.dequant()` at
/// `orig_v` covers `prefill + steps` while the V CPU blocks are frozen at
/// prefill — so `synced_rotor_v_blocks` rebuilt the tail from the ring rather
/// than zero-padding.
///
/// Mutation check: make `QuantRotorV3::try_deep_clone` clone `self.blocks`
/// directly (drop the `synced_rotor_v_blocks` reconcile) — the clone carries
/// only the frozen prefill prefix, `write_caches` trips the
/// `ensure_rotor_v_blocks_cover_shape` guard, and `.expect(...)` panics (RED).
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd rotor_sym_ring_only_tail_ssd -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test: values established by construction earlier in this fn"
)]
fn rotor_sym_ring_only_tail_ssd_round_trip() {
    let device = Device::Gpu;
    let (kv_h, n_q, head_dim, prefill, steps) = (2_i32, 8_i32, 128_i32, 6_i32, 5_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut c = seeded_rotor_sym3_cache(kv_h, head_dim, 512);
    let k = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 61),
        &[1, kv_h, prefill, head_dim],
    );
    let v = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 62),
        &[1, kv_h, prefill, head_dim],
    );
    let q = arr(
        &lcg((prefill * n_q * head_dim) as usize, 63),
        &[1, n_q, prefill, head_dim],
    );
    c.update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .unwrap();
    c.exit_prefill(device).unwrap();
    for i in 0..steps {
        let k1 = arr(
            &lcg((kv_h * head_dim) as usize, 71 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let v1 = arr(
            &lcg((kv_h * head_dim) as usize, 81 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let q1 = arr(
            &lcg((n_q * head_dim) as usize, 91 + i as u64),
            &[1, n_q, 1, head_dim],
        );
        c.update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .unwrap()
            .eval()
            .unwrap();
    }

    // Ring-as-sole-store precondition on BOTH axes + live full-prefix dequant.
    // The sym append drops the CPU blocks once the ring is live, so both axes'
    // blocks are empty while the ring holds the full `prefill + steps` prefix —
    // the resident-memory win. `dequant` therefore rebuilds the whole prefix
    // from the ring, not a zero-padded prefix.
    let (orig_k, orig_v, orig_seq, k_block_tokens, v_block_tokens) = rotor_sym3_probe(&c);
    assert_eq!(orig_seq, prefill + steps, "shape[2] advanced with the ring");
    assert_eq!(
        k_block_tokens, 0,
        "K CPU blocks dropped — the ring is the sole resident store"
    );
    assert_eq!(
        v_block_tokens, 0,
        "V CPU blocks dropped — the ring is the sole resident store (dequant rebuilt from ring)"
    );

    // Spill clone (materialises both tails) → write → hydrate.
    let clone = c.try_deep_clone().unwrap();
    let path = tmp_path("rotor_sym_ring_only_tail");
    write_caches(&path, device, MODEL_ID, KvQuant::Rotor3Sym, &[clone], &[])
        .expect("write_caches must persist the full materialised sym store");
    let (hydrated, _lin) = read_caches(&path, device, MODEL_ID, KvQuant::Rotor3Sym).unwrap();
    let _ = std::fs::remove_file(&path);

    let (hy_k, hy_v, hy_seq, _, _) = rotor_sym3_probe(&hydrated[0]);
    assert_eq!(
        hy_seq,
        prefill + steps,
        "hydrated sym store must carry the full decoded length"
    );
    assert_eq!(hy_k.len(), orig_k.len(), "hydrated K length mismatch");
    assert_eq!(hy_v.len(), orig_v.len(), "hydrated V length mismatch");
    assert!(
        max_abs_err(&orig_k, &hy_k) < 1e-6,
        "hydrated K must match the live ring-only-tail dequant byte-for-byte"
    );
    assert!(
        max_abs_err(&orig_v, &hy_v) < 1e-6,
        "hydrated V must match the live ring-only-tail dequant byte-for-byte"
    );
}

// ── Iso symmetric ring-only-tail (sole-store) ─────────────────────────────────

fn seeded_iso_sym3_cache(kv_h: i32, head_dim: i32, max_seq: i32) -> KvCache {
    let storage = KvStorage::IsoSym3 {
        k: Some(QuantIsoK3::from_cpu_blocks(
            Vec::new(),
            vec![1, kv_h, 0, head_dim],
            max_seq,
        )),
        v: Some(QuantIsoV3::from_cpu_blocks(
            Vec::new(),
            vec![1, kv_h, 0, head_dim],
        )),
        max_seq,
    };
    KvCache::from_storage(storage, KvQuant::Iso3Sym, 0, 0)
}

/// `(k_dequant, v_dequant, shape[2], k_block_tokens, v_block_tokens)` for an
/// iso3 symmetric cache.
#[allow(
    clippy::unwrap_used,
    reason = "test helper: values established by construction"
)]
fn iso_sym3_probe(cache: &KvCache) -> (Vec<f32>, Vec<f32>, i32, usize, usize) {
    match cache.storage() {
        KvStorage::IsoSym3 {
            k: Some(ks),
            v: Some(vs),
            ..
        } => (
            ks.dequant().unwrap(),
            vs.dequant().unwrap(),
            ks.shape.get(2).copied().unwrap_or(0),
            ks.blocks.iter().map(|b| b.n_tokens).sum(),
            vs.blocks.iter().map(|b| b.n_tokens).sum(),
        ),
        _ => panic!("expected a live IsoSym3 store"),
    }
}

/// Scalar reference attention over head-major (`[1, kv_h, S, D]`) K/V, for the
/// short-kv_seq fallback correctness check.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by \
              slice length"
)]
fn ref_attn_head_major(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    kv_h: usize,
    s: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let hpk = n_q / kv_h;
    let mut out = vec![0.0_f32; n_q * d];
    for hq in 0..n_q {
        let h = hq / hpk;
        let mut scores = vec![0.0_f32; s];
        for (si, sc) in scores.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for di in 0..d {
                acc += q[hq * d + di] * k[(h * s + si) * d + di];
            }
            *sc = acc * scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut den = 0.0_f32;
        for sc in &mut scores {
            *sc = (*sc - m).exp();
            den += *sc;
        }
        for di in 0..d {
            let mut acc = 0.0_f32;
            for (si, &p) in scores.iter().enumerate() {
                acc += p * v[(h * s + si) * d + di];
            }
            out[hq * d + di] = acc / den;
        }
    }
    out
}

/// Short-prompt smoke (hard rule 6) for the `kv_h == 1` single-KV-head shape
/// (Gemma4 global layers): with the fused `iso3_sym` codec live, a decode at a
/// small `kv_seq` used to abort because MLX binds the ring's tiny per-token
/// `norms` slice in the `constant` address space, which the flash kernel's
/// `if_decode_k_lane` (`device const float*`) rejects. `iso_flash_decode_symv_sdpa`
/// now zero-pads `norms` up to its `NORMS_DEVICE_MIN` floor (16) before
/// dispatch whenever `b*kv_h*kv_seq` is below it, so the fused GPU kernel runs at
/// every `kv_seq >= 1` with no CPU dequant fallback (hard rule 10): the padding
/// is allocated but never read, since the kernel's per-tile loop bound is the
/// real `kv_seq` carried in `dims`, not the buffer length. Asserts no abort AND
/// numerically-correct output (vs a scalar reference over the store's own
/// dequant) for `kv_seq` 2, 3, 4, 7, 15 — all below the compile floor.
///
/// Mutation check: force `iso_flash_decode_symv_sdpa`'s norms-padding helper
/// `pad_norms_to_device_floor` to skip padding (return the unpadded array
/// unconditionally) → the `kv_seq == 2`
/// step aborts with "Unable to build metal library from source … cannot pass
/// pointer to address space 'constant' as a pointer to address space 'device' …
/// if_decode_k_lane" (RED).
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd iso_sym_short_kv_seq_kv_h1 -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test: values established by construction; GPU dispatch failures asserted via \
              .expect messages"
)]
fn iso_sym_short_kv_seq_kv_h1_stays_on_gpu() {
    let device = Device::Gpu;
    let n_q = 8_i32;
    let kv_h = 1_i32; // single KV head — the Gemma4 global-layer shape
    let head_dim = 512_i32;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    for kv_seq in [2_i32, 3, 4, 7, 15] {
        let prefill = kv_seq - 1;
        let mut c = seeded_iso_sym3_cache(kv_h, head_dim, 512);
        let k = arr(
            &lcg((prefill * kv_h * head_dim) as usize, 31),
            &[1, kv_h, prefill, head_dim],
        );
        let v = arr(
            &lcg((prefill * kv_h * head_dim) as usize, 32),
            &[1, kv_h, prefill, head_dim],
        );
        let q = arr(
            &lcg((prefill * n_q * head_dim) as usize, 33),
            &[1, n_q, prefill, head_dim],
        );
        c.update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
            .unwrap();
        c.exit_prefill(device).unwrap();

        let qd = lcg((n_q * head_dim) as usize, 41);
        let q1 = arr(&qd, &[1, n_q, 1, head_dim]);
        let k1 = arr(
            &lcg((kv_h * head_dim) as usize, 42),
            &[1, kv_h, 1, head_dim],
        );
        let v1 = arr(
            &lcg((kv_h * head_dim) as usize, 43),
            &[1, kv_h, 1, head_dim],
        );

        // MUST NOT abort — the kernel dispatcher pads the small norms buffer.
        let out = c
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect(
                "short kv_seq at kv_h==1 must not abort — norms is zero-padded before dispatch",
            );
        let got = to_vec(&out);
        assert_eq!(
            got.len(),
            (n_q * head_dim) as usize,
            "kv_seq={kv_seq}: shape"
        );
        assert!(
            got.iter().all(|x| x.is_finite()),
            "kv_seq={kv_seq}: output must be finite"
        );

        // Correctness: scalar attention over the store's own dequant (which
        // transparently rebuilds the ring-only tail). Both decode the same iso
        // values, so the only divergence is summation order — well inside bf16
        // tolerance.
        let (k_deq, v_deq, seq, _, _) = iso_sym3_probe(&c);
        assert_eq!(seq, kv_seq, "kv_seq={kv_seq}: store length");
        let want = ref_attn_head_major(
            &qd,
            &k_deq,
            &v_deq,
            n_q as usize,
            kv_h as usize,
            kv_seq as usize,
            head_dim as usize,
            scale,
        );
        let err = max_abs_err(&got, &want);
        assert!(
            err < 2e-3,
            "kv_seq={kv_seq}: padded-dispatch output max_abs_err={err} exceeds bf16 tolerance"
        );
    }
}

/// CONTINUITY correctness across the `NORMS_DEVICE_MIN` padding floor (16): a
/// single continuous `kv_h == 1` iso3_sym cache decoded ONE token at a time
/// from below the floor through well above it, on the SAME cache/ring for
/// every step, driven **in parallel** with an independent `KvStorage::None`
/// (unquantised bf16/f32) reference cache fed the identical per-step tokens.
/// Unlike [`iso_sym_short_kv_seq_kv_h1_stays_on_gpu`] above (which builds a
/// **fresh** cache per `kv_seq` via a bulk prefill call, so each `kv_seq` is
/// only ever reached once), this exercises the ring-only decode tail growing
/// continuously through the point where `iso_flash_decode_symv_sdpa` stops
/// padding `norms` (once `b*kv_h*kv_seq >= 16`) and starts passing it through
/// unpadded.
///
/// The reference is deliberately **independent of the store under test**: an
/// earlier version of this test compared the kernel's output against a
/// scalar reference built from `iso_sym3_probe`'s `dequant()` — which
/// rebuilds from the SAME ring the kernel just read. A ring corruption both
/// reads see identically (e.g. a dropped or duplicated append) would pass
/// that check with `err≈0`; "didn't error" is not "correct" for a silent-KV
/// class of bug. The bf16 cache instead runs its own `update()` /
/// `scaled_dot_product_attention` over its own independently-accumulated
/// K/V, sharing no state with the iso ring at all, so a dropped/misordered
/// token in the iso ring surfaces as a real divergence between the two
/// outputs — a gross softmax shift over a materially different key set,
/// against random per-step K/V/Q — not just quantisation noise.
///
/// Mutation check: same as [`iso_sym_short_kv_seq_kv_h1_stays_on_gpu`] —
/// forcing `iso_flash_decode_symv_sdpa`'s norms-padding helper
/// `pad_norms_to_device_floor` to skip padding makes the very first decode
/// step (`kv_seq == 2`) abort before
/// this test can reach the floor at all.
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd iso_sym_transition_across_ring_norms_floor -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test: values established by construction; GPU dispatch failures asserted via \
              .expect/.unwrap_or_else messages"
)]
fn iso_sym_transition_across_ring_norms_floor() {
    let device = Device::Gpu;
    let n_q = 8_i32;
    let kv_h = 1_i32; // single KV head — the Gemma4 global-layer shape
    let head_dim = 512_i32;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    const FLOOR: i32 = 16; // mirrors NORMS_DEVICE_MIN in flash_decode_common.rs
                           // 3-bit iso quant error on both K and V vs the independent bf16
                           // reference: measured max_abs_err ~0.14 at kv_seq=2, staying under 0.2
                           // through kv_seq=24 over this test's random LCG data. A single wrong
                           // token's V fed to the reference (simulating a dropped/misordered token)
                           // measured max_abs_err ~0.57 at the same kv_seq=2 — comfortably above
                           // this floor, so genuine quant noise and a lost-token bug do not overlap.
    const ISO_QUANT_TOL: f32 = 0.3;

    let mut iso = seeded_iso_sym3_cache(kv_h, head_dim, 512);
    let mut bf16 = KvCache::from_storage(KvStorage::None { max_seq: 512 }, KvQuant::None, 0, 0);

    // Identical one-token prefill on both caches, matching a realistic short
    // chat prompt: kv_seq == 1 before the first decode step.
    let prefill = 1_i32;
    let k0 = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 51),
        &[1, kv_h, prefill, head_dim],
    );
    let v0 = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 52),
        &[1, kv_h, prefill, head_dim],
    );
    let q0 = arr(
        &lcg((prefill * n_q * head_dim) as usize, 53),
        &[1, n_q, prefill, head_dim],
    );
    iso.update_and_sdpa(&q0, &k0, &v0, scale, "causal", None, device)
        .unwrap();
    iso.exit_prefill(device).unwrap();
    bf16.update_and_sdpa(&q0, &k0, &v0, scale, "causal", None, device)
        .unwrap();
    bf16.exit_prefill(device).unwrap();

    // Decode one token at a time, kv_seq 2..24 — well below the floor through
    // well above it — feeding the identical tokens into both caches.
    let mut crossed_floor = false;
    for step in 0_i32..23 {
        let kv_seq = 2 + step;
        let qd = lcg((n_q * head_dim) as usize, 1_000 + step as u64);
        let kd = lcg((kv_h * head_dim) as usize, 2_000 + step as u64);
        let vd = lcg((kv_h * head_dim) as usize, 3_000 + step as u64);
        let q1 = arr(&qd, &[1, n_q, 1, head_dim]);
        let k1 = arr(&kd, &[1, kv_h, 1, head_dim]);
        let v1 = arr(&vd, &[1, kv_h, 1, head_dim]);

        let iso_out = iso
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .unwrap_or_else(|e| {
                panic!("kv_seq={kv_seq}: must not abort across the ring-norms floor: {e}")
            });
        let bf16_out = bf16
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("bf16 reference cache must not fail — plain SDPA, no custom kernel");

        let got = to_vec(&iso_out);
        let want = to_vec(&bf16_out);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "kv_seq={kv_seq}: output must be finite"
        );

        // Decisive check: the iso3_sym kernel's output vs the INDEPENDENT
        // bf16 reference cache's output — two separate stores, two separate
        // code paths, sharing no ring — so a dropped or misordered pre-floor
        // token in the iso ring cannot pass silently.
        let err = max_abs_err(&got, &want);
        assert!(
            err < ISO_QUANT_TOL,
            "kv_seq={kv_seq}: iso3_sym vs independent bf16 reference max_abs_err={err} exceeds \
             tolerance {ISO_QUANT_TOL} (floor={FLOOR}) — the ring's per-step append/reseed \
             bookkeeping likely dropped or misordered a token"
        );

        if kv_seq >= FLOOR {
            crossed_floor = true;
        }
    }
    assert!(
        crossed_floor,
        "test did not actually reach kv_seq >= {FLOOR} — padding floor unexercised"
    );
}

/// Rotor sibling of [`iso_sym_transition_across_ring_norms_floor`] — same
/// independent-bf16-reference design, same `kv_h == 1` continuity coverage
/// across [`flash_decode_common::NORMS_DEVICE_MIN`], for the `rotor3_sym`
/// codec (`rotor_flash_decode_symv_sdpa`, `rf_decode_k_group`). Rotor's
/// `norms` buffer hits the identical MLX small-buffer `constant`-binding trap
/// as iso — same root cause, same [`crate::flash_decode_common`] fix.
///
/// Mutation check: forcing `rotor_flash_decode_symv_sdpa`'s norms-padding
/// helper `pad_norms_to_device_floor` to skip padding makes the very first
/// decode step (`kv_seq == 2`) abort with the same address-space-mismatch
/// class of MSL error (`rf_decode_k_group`, `constant` vs `device`) before
/// this test can reach the floor at all.
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd rotor_sym_transition_across_ring_norms_floor -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test: values established by construction; GPU dispatch failures asserted via \
              .expect/.unwrap_or_else messages"
)]
fn rotor_sym_transition_across_ring_norms_floor() {
    let device = Device::Gpu;
    let n_q = 8_i32;
    let kv_h = 1_i32; // single KV head — the Gemma4 global-layer shape
    let head_dim = 512_i32;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    const FLOOR: i32 = 16; // mirrors NORMS_DEVICE_MIN in flash_decode_common.rs
                           // 3-bit rotor quant error on both K and V vs the independent bf16
                           // reference: measured max_abs_err ~0.15 at kv_seq=2, staying under 0.2
                           // through kv_seq=24 over this test's random LCG data — same order as
                           // ISO_QUANT_TOL above, same margin reasoning.
    const ROTOR_QUANT_TOL: f32 = 0.3;

    let mut rotor = seeded_rotor_sym3_cache(kv_h, head_dim, 512);
    let mut bf16 = KvCache::from_storage(KvStorage::None { max_seq: 512 }, KvQuant::None, 0, 0);

    let prefill = 1_i32;
    let k0 = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 61),
        &[1, kv_h, prefill, head_dim],
    );
    let v0 = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 62),
        &[1, kv_h, prefill, head_dim],
    );
    let q0 = arr(
        &lcg((prefill * n_q * head_dim) as usize, 63),
        &[1, n_q, prefill, head_dim],
    );
    rotor
        .update_and_sdpa(&q0, &k0, &v0, scale, "causal", None, device)
        .unwrap();
    rotor.exit_prefill(device).unwrap();
    bf16.update_and_sdpa(&q0, &k0, &v0, scale, "causal", None, device)
        .unwrap();
    bf16.exit_prefill(device).unwrap();

    let mut crossed_floor = false;
    for step in 0_i32..23 {
        let kv_seq = 2 + step;
        let qd = lcg((n_q * head_dim) as usize, 4_000 + step as u64);
        let kd = lcg((kv_h * head_dim) as usize, 5_000 + step as u64);
        let vd = lcg((kv_h * head_dim) as usize, 6_000 + step as u64);
        let q1 = arr(&qd, &[1, n_q, 1, head_dim]);
        let k1 = arr(&kd, &[1, kv_h, 1, head_dim]);
        let v1 = arr(&vd, &[1, kv_h, 1, head_dim]);

        let rotor_out = rotor
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .unwrap_or_else(|e| {
                panic!("kv_seq={kv_seq}: must not abort across the ring-norms floor: {e}")
            });
        let bf16_out = bf16
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("bf16 reference cache must not fail — plain SDPA, no custom kernel");

        let got = to_vec(&rotor_out);
        let want = to_vec(&bf16_out);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "kv_seq={kv_seq}: output must be finite"
        );

        let err = max_abs_err(&got, &want);
        assert!(
            err < ROTOR_QUANT_TOL,
            "kv_seq={kv_seq}: rotor3_sym vs independent bf16 reference max_abs_err={err} exceeds \
             tolerance {ROTOR_QUANT_TOL} (floor={FLOOR}) — the ring's per-step append/reseed \
             bookkeeping likely dropped or misordered a token"
        );

        if kv_seq >= FLOOR {
            crossed_floor = true;
        }
    }
    assert!(
        crossed_floor,
        "test did not actually reach kv_seq >= {FLOOR} — padding floor unexercised"
    );
}

/// The write path refuses to persist an iso **V** store whose CPU blocks fall
/// short of `shape[2]` with no ring — a truncated store. Runs on CPU (no Metal).
///
/// Mutation check: delete the `ensure_iso_blocks_cover_shape` guard in
/// `write_quant_iso_v3` — the writer then silently serializes the short prefix
/// and this assertion flips RED (`write` returns `Ok`).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test: values established by construction earlier in this fn"
)]
fn write_rejects_truncated_iso_store() {
    let kv_h = 2_i32;
    let head_dim = 8_i32; // multiple of ISO_QUAT_BLOCK_SIZE (4); n_groups = 2
    let data = lcg((kv_h * 2 * head_dim) as usize, 0x1D0);

    // A real single V block covering 2 tokens.
    let mut src = QuantIsoV3::new(vec![1, kv_h, 0, head_dim]);
    src.append(&data, &[1, kv_h, 2, head_dim]).unwrap();

    // Claim shape[2] == 4 while the blocks cover only 2 tokens and no ring
    // exists — the ring-only tail with the ring missing. Pair with a well-formed
    // K store so the sym layer is valid on the K side.
    let truncated_v = QuantIsoV3::from_cpu_blocks(src.blocks.clone(), vec![1, kv_h, 4, head_dim]);
    let mut k_src = QuantIsoK3::new(vec![1, kv_h, 0, head_dim], 64);
    let k_data = lcg((kv_h * 4 * head_dim) as usize, 0x2E0);
    k_src.append(&k_data, &[1, kv_h, 4, head_dim]).unwrap();

    let layers = vec![KvStorage::IsoSym3 {
        k: Some(k_src),
        v: Some(truncated_v),
        max_seq: 64,
    }];
    let path = tmp_path("iso_v_truncated");
    let res =
        KvBlockWriter::new(MODEL_ID, KvQuant::Iso3Sym, &layers, &[]).write(&path, Device::Cpu);
    let _ = std::fs::remove_file(&path);
    assert!(
        res.is_err(),
        "writer must reject a truncated iso V store (short blocks, no ring), got Ok"
    );
}

/// Full SSD spill/hydrate round-trip after a **symmetric** iso ring-only-tail
/// decode: both K and V decode tails live only in their GPU rings, are
/// materialised by the spill clone, and hydrate byte-exact. Also proves the V
/// dequant-full-prefix rebuild (`synced_iso_v_blocks`) from the ring.
///
/// Mutation checks:
/// * dequant skip-rebuild: make `synced_iso_v_blocks` return
///   `Cow::Borrowed(blocks)` always — the frozen short prefix trips the loud
///   `refusing to zero-pad` error in `dequant` (RED, not a silent zero-pad).
/// * SSD spill: make `QuantIsoV3::try_deep_clone` clone `self.blocks` directly
///   (drop the synced reconcile) — `write_caches` trips
///   `ensure_iso_blocks_cover_shape` → `TruncatedStore` (RED).
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd iso_sym_ring_only_tail_ssd -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test: values established by construction earlier in this fn"
)]
fn iso_sym_ring_only_tail_ssd_round_trip() {
    let device = Device::Gpu;
    let (kv_h, n_q, head_dim, prefill, steps) = (2_i32, 8_i32, 128_i32, 6_i32, 5_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut c = seeded_iso_sym3_cache(kv_h, head_dim, 512);
    let k = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 61),
        &[1, kv_h, prefill, head_dim],
    );
    let v = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 62),
        &[1, kv_h, prefill, head_dim],
    );
    let q = arr(
        &lcg((prefill * n_q * head_dim) as usize, 63),
        &[1, n_q, prefill, head_dim],
    );
    c.update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .unwrap();
    c.exit_prefill(device).unwrap();
    for i in 0..steps {
        let k1 = arr(
            &lcg((kv_h * head_dim) as usize, 71 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let v1 = arr(
            &lcg((kv_h * head_dim) as usize, 81 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let q1 = arr(
            &lcg((n_q * head_dim) as usize, 91 + i as u64),
            &[1, n_q, 1, head_dim],
        );
        c.update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .unwrap()
            .eval()
            .unwrap();
    }

    // Ring-as-sole-store precondition on BOTH axes + live full-prefix dequant.
    let (orig_k, orig_v, orig_seq, k_block_tokens, v_block_tokens) = iso_sym3_probe(&c);
    assert_eq!(orig_seq, prefill + steps, "shape[2] advanced with the ring");
    assert_eq!(
        k_block_tokens, 0,
        "K CPU blocks dropped — the ring is the sole resident store"
    );
    assert_eq!(
        v_block_tokens, 0,
        "V CPU blocks dropped — the ring is the sole resident store (dequant rebuilt from ring)"
    );

    // Spill clone (materialises both tails) → write → hydrate.
    let clone = c.try_deep_clone().unwrap();
    let path = tmp_path("iso_sym_ring_only_tail");
    write_caches(&path, device, MODEL_ID, KvQuant::Iso3Sym, &[clone], &[])
        .expect("write_caches must persist the full materialised sym store");
    let (hydrated, _lin) = read_caches(&path, device, MODEL_ID, KvQuant::Iso3Sym).unwrap();
    let _ = std::fs::remove_file(&path);

    let (hy_k, hy_v, hy_seq, _, _) = iso_sym3_probe(&hydrated[0]);
    assert_eq!(
        hy_seq,
        prefill + steps,
        "hydrated sym store must carry the full decoded length"
    );
    assert_eq!(hy_k.len(), orig_k.len(), "hydrated K length mismatch");
    assert_eq!(hy_v.len(), orig_v.len(), "hydrated V length mismatch");
    assert!(
        max_abs_err(&orig_k, &hy_k) < 1e-6,
        "hydrated K must match the live ring-only-tail dequant byte-for-byte"
    );
    assert!(
        max_abs_err(&orig_v, &hy_v) < 1e-6,
        "hydrated V must match the live ring-only-tail dequant byte-for-byte"
    );
}

/// `truncate_to` keeps the V ring so a ring-only decode tail survives the
/// speculative-decode rollback: after truncating mid-fused-decode, `dequant`
/// still rebuilds the full `[0, n)` prefix from the ring instead of aborting on
/// a short-blocks store.
///
/// On the sole-store path the CPU blocks are already empty (dropped once the ring
/// went live), so `truncate_to`'s block loop is a no-op and the pre-existing
/// `n_tokens`-vs-`n` unit mismatch (a separate, out-of-scope bug that only bites a
/// non-empty `kv_h > 1` block set) does not engage — the ring supplies the whole
/// kept prefix. `kv_h == 2` here to keep the fused kernel on the same
/// multi-head path the round-trip test exercises.
///
/// Mutation check: re-add `self.gpu.clear()` to `QuantIsoV3::truncate_to` — the
/// ring is dropped, the V blocks fall short of `shape[2]` with no ring, and
/// `dequant()` returns the loud `synced_iso_v_blocks` error (RED).
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd iso_sym_truncate_keeps_ring -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test: values established by construction earlier in this fn"
)]
fn iso_sym_truncate_keeps_ring_tail() {
    let device = Device::Gpu;
    let (kv_h, n_q, head_dim, prefill, steps) = (2_i32, 8_i32, 128_i32, 6_i32, 6_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut c = seeded_iso_sym3_cache(kv_h, head_dim, 512);
    let k = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 101),
        &[1, kv_h, prefill, head_dim],
    );
    let v = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 102),
        &[1, kv_h, prefill, head_dim],
    );
    let q = arr(
        &lcg((prefill * n_q * head_dim) as usize, 103),
        &[1, n_q, prefill, head_dim],
    );
    c.update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .unwrap();
    c.exit_prefill(device).unwrap();
    for i in 0..steps {
        let k1 = arr(
            &lcg((kv_h * head_dim) as usize, 111 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let v1 = arr(
            &lcg((kv_h * head_dim) as usize, 121 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let q1 = arr(
            &lcg((n_q * head_dim) as usize, 131 + i as u64),
            &[1, n_q, 1, head_dim],
        );
        c.update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .unwrap()
            .eval()
            .unwrap();
    }

    // Full-length dequant before truncation (the reference prefix).
    let (_ok, orig_v, orig_seq, _, _) = iso_sym3_probe(&c);
    assert_eq!(orig_seq, prefill + steps);
    let per_tok = (kv_h * head_dim) as usize;

    // Truncate into the ring-only tail and confirm dequant still rebuilds it.
    let keep = prefill + steps - 3;
    c.truncate_to(keep);
    let (_k2, v_after, seq_after, _, _) = iso_sym3_probe(&c);
    assert_eq!(seq_after, keep, "shape[2] lowered to the truncation point");
    assert_eq!(
        v_after.len(),
        (keep as usize) * per_tok,
        "V dequant covers the kept prefix — the ring supplied the tail, no abort"
    );
    // Both dequants are head-major `[1, kv_h, S, D]`, so the sequence axis is in
    // the middle — compare per (head, seq-position) rather than a flat slice.
    let (orig_s, hd) = ((prefill + steps) as usize, head_dim as usize);
    let keep_s = keep as usize;
    for h in 0..kv_h as usize {
        for s in 0..keep_s {
            let a = &orig_v[(h * orig_s + s) * hd..(h * orig_s + s) * hd + hd];
            let b = &v_after[(h * keep_s + s) * hd..(h * keep_s + s) * hd + hd];
            assert!(
                max_abs_err(a, b) < 1e-6,
                "kept V (head {h}, pos {s}) must match the pre-truncation dequant"
            );
        }
    }
}

/// `truncate_to` keeps the V ring so a ring-only decode tail survives the
/// speculative-decode rollback: after truncating mid-fused-decode, `dequant`
/// still rebuilds the full `[0, n)` prefix from the ring instead of aborting on
/// a short-blocks store.
///
/// Mutation check: re-add `self.gpu.clear()` to `QuantRotorV3::truncate_to` —
/// the ring is dropped, the V blocks fall short of `shape[2]` with no ring, and
/// `dequant()` returns the loud `synced_rotor_v_blocks` error (RED).
#[test]
#[ignore = "GPU Metal context — run: cargo test -p rmlx-kv-ssd rotor_sym_truncate_keeps_ring -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test: values established by construction earlier in this fn"
)]
fn rotor_sym_truncate_keeps_ring_tail() {
    let device = Device::Gpu;
    let (kv_h, n_q, head_dim, prefill, steps) = (2_i32, 8_i32, 128_i32, 6_i32, 6_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut c = seeded_rotor_sym3_cache(kv_h, head_dim, 512);
    let k = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 101),
        &[1, kv_h, prefill, head_dim],
    );
    let v = arr(
        &lcg((prefill * kv_h * head_dim) as usize, 102),
        &[1, kv_h, prefill, head_dim],
    );
    let q = arr(
        &lcg((prefill * n_q * head_dim) as usize, 103),
        &[1, n_q, prefill, head_dim],
    );
    c.update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .unwrap();
    c.exit_prefill(device).unwrap();
    for i in 0..steps {
        let k1 = arr(
            &lcg((kv_h * head_dim) as usize, 111 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let v1 = arr(
            &lcg((kv_h * head_dim) as usize, 121 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let q1 = arr(
            &lcg((n_q * head_dim) as usize, 131 + i as u64),
            &[1, n_q, 1, head_dim],
        );
        c.update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .unwrap()
            .eval()
            .unwrap();
    }

    // Full-length dequant before truncation (the reference prefix).
    let (_ok, orig_v, orig_seq, _, _) = rotor_sym3_probe(&c);
    assert_eq!(orig_seq, prefill + steps);
    let per_tok = (kv_h * head_dim) as usize;

    // Truncate into the ring-only tail (below the last CPU block, inside the
    // ring-held region) and confirm dequant still rebuilds the kept prefix.
    let keep = prefill + steps - 3;
    c.truncate_to(keep);
    let (_k2, v_after, seq_after, _, _) = rotor_sym3_probe(&c);
    assert_eq!(seq_after, keep, "shape[2] lowered to the truncation point");
    assert_eq!(
        v_after.len(),
        (keep as usize) * per_tok,
        "V dequant covers the kept prefix — the ring supplied the tail, no abort"
    );
    // The kept prefix must be byte-exact with the pre-truncation dequant.
    // Both are head-major `[1, kv_h, S, D]`, so the sequence axis is in the
    // middle — compare per (head, seq-position) rather than a flat prefix slice.
    let (orig_s, hd) = ((prefill + steps) as usize, head_dim as usize);
    let keep_s = keep as usize;
    for h in 0..kv_h as usize {
        for s in 0..keep_s {
            let a = &orig_v[(h * orig_s + s) * hd..(h * orig_s + s) * hd + hd];
            let b = &v_after[(h * keep_s + s) * hd..(h * keep_s + s) * hd + hd];
            assert!(
                max_abs_err(a, b) < 1e-6,
                "kept V (head {h}, pos {s}) must match the pre-truncation dequant"
            );
        }
    }
}
