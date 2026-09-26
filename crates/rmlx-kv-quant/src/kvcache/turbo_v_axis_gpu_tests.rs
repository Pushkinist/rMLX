//! The V-axis device rule of the symmetric turbo decode update, on Metal.
//!
//! `tsym_update` resolves the V device from the code width: the caller's
//! device at 4 bits, `Device::Cpu` at 3. That rule is invisible to every CPU
//! test in the tree, because on a `Device::Cpu` drive the caller's device *is*
//! `Device::Cpu` and the two answers coincide: routing the 3-bit V axis to the
//! caller's device leaves every cell of `kvcache::turbo_store_bytes_tests`
//! green. `docs/KV_UPDATE_PATH.md` § "TurboQuant" states the rule.
//!
//! This file is the drive that sees it. It is the GPU twin of that oracle's
//! own drive — `KvCache::update` with `in_prefill` false and no bf16 seed,
//! which is the one route that reaches the turbo storage dispatch at all (a
//! served prefill returns before it, and a seeded decode step short-circuits
//! on `decode_fp16_k`).
//!
//! # What it holds
//!
//! * Both widths append and decode on Metal without error. Under the lost
//!   width rule the 3-bit cell fails on its first append: `QuantV::append`
//!   enters its GPU branch on the device alone and then returns `Error::Quant`
//!   for `bits != 4`.
//! * The V store's own disposition per width — a GPU mirror at 4 bits, CPU
//!   blocks and no mirror at 3. That is the other direction of the same rule,
//!   and it is what a body that forced `Device::Cpu` at both widths would
//!   break while still returning correct rows.
//! * The K store carries a GPU mirror at both widths: the K axis always takes
//!   the caller's device.

use rmlx_mlx::{Array, Device, Dtype};

use super::KvCache;
use crate::quant::KvQuant;
use crate::storage::KvStorage;
use crate::test_utils::{lcg_data, skip_if_no_gpu_env, TEST_SEED};

/// `[B, kv_h, S, D]` the drive appends at. `D` is a multiple of the turbo
/// group size, which both MSL encode kernels require.
const KV_H: i32 = 1;
const HEAD_DIM: i32 = 128;
const CHUNK_SEQ: i32 = 6;
const DECODE_STEPS: usize = 3;
const MAX_SEQ: i32 = 512;

/// What one width's GPU drive left behind.
struct StoreDisposition {
    k_gpu_mirror: bool,
    v_gpu_mirror: bool,
    v_cpu_blocks: usize,
}

#[allow(
    clippy::expect_used,
    reason = "test driver: every append is on a shape the spelling accepts, so a failure is the defect under test and the panic names it"
)]
fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32 array")
}

/// Drive one symmetric turbo spelling on Metal, `in_prefill` false throughout.
#[allow(
    clippy::expect_used,
    reason = "test driver: a failing append is the defect under test and the panic names it"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the storage variant is the one the KvQuant selected; any other is a construction bug, and the explicit panic names it sooner than a wrong field would"
)]
fn drive_on_gpu(quant: KvQuant) -> StoreDisposition {
    let device = Device::Gpu;
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ).with_layer_idx(0);

    let chunk_shape = [1_i32, KV_H, CHUNK_SEQ, HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&lcg_data(n_chunk, TEST_SEED), &chunk_shape);
    let v = f32_arr(&lcg_data(n_chunk, TEST_SEED ^ 0x5a5a), &chunk_shape);
    let (k_out, v_out) = cache.update(&k, &v, device).expect("bulk append on Metal");
    assert_eq!(
        k_out.shape(),
        chunk_shape.to_vec(),
        "K rows after the chunk"
    );
    assert_eq!(
        v_out.shape(),
        chunk_shape.to_vec(),
        "V rows after the chunk"
    );

    let step_shape = [1_i32, KV_H, 1, HEAD_DIM];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    for step in 0..DECODE_STEPS {
        let seed = TEST_SEED.wrapping_add(step as u64 + 1);
        let ks = f32_arr(&lcg_data(n_step, seed), &step_shape);
        let vs = f32_arr(&lcg_data(n_step, seed ^ 0x5a5a), &step_shape);
        let (ko, vo) = cache
            .update(&ks, &vs, device)
            .expect("decode step on Metal");
        let seq = CHUNK_SEQ + step as i32 + 1;
        assert_eq!(
            ko.shape(),
            vec![1, KV_H, seq, HEAD_DIM],
            "K rows after decode step {step}"
        );
        assert_eq!(
            vo.shape(),
            vec![1, KV_H, seq, HEAD_DIM],
            "V rows after decode step {step}"
        );
    }

    match &cache.storage {
        KvStorage::TurboSym3 { k, v, .. } => {
            let ks = k.as_ref().expect("3-bit K store after the drive");
            let vs = v.as_ref().expect("3-bit V store after the drive");
            StoreDisposition {
                k_gpu_mirror: ks.gpu_codes_buf.is_some(),
                v_gpu_mirror: vs.gpu_codes_buf.is_some(),
                v_cpu_blocks: vs.blocks.len(),
            }
        }
        KvStorage::TurboSym4 { k, v, .. } => {
            let ks = k.as_ref().expect("4-bit K store after the drive");
            let vs = v.as_ref().expect("4-bit V store after the drive");
            StoreDisposition {
                k_gpu_mirror: ks.gpu_codes_buf.is_some(),
                v_gpu_mirror: vs.gpu_codes_buf.is_some(),
                v_cpu_blocks: vs.blocks.len(),
            }
        }
        other => panic!(
            "not a symmetric turbo storage variant: {}",
            super::helpers::storage_variant_name(other)
        ),
    }
}

/// The 3-bit V axis stays on the CPU when the caller drives Metal.
#[test]
#[ignore = "GPU Metal context — run via `make gpu-test CRATE=rmlx-kv-quant FILTER=turbo`"]
fn turbo_sym3_v_axis_stays_on_the_cpu_under_a_metal_drive() {
    if skip_if_no_gpu_env() {
        return;
    }
    let got = drive_on_gpu(KvQuant::TurboSym3);
    assert!(
        got.k_gpu_mirror,
        "tsym3: the K axis must take the caller's device, and the store carries no GPU mirror"
    );
    assert!(
        !got.v_gpu_mirror,
        "tsym3: the V axis was handed the caller's device. QuantV::append refuses bits != 4 in \
         its GPU branch, so this build only got this far by luck of a shape"
    );
    assert_eq!(
        got.v_cpu_blocks,
        1 + DECODE_STEPS,
        "tsym3: the CPU V path pushes one block per append, and the block count says it did not"
    );
}

/// The 4-bit V axis follows the caller onto Metal.
#[test]
#[ignore = "GPU Metal context — run via `make gpu-test CRATE=rmlx-kv-quant FILTER=turbo`"]
fn turbo_sym4_v_axis_follows_the_caller_to_the_gpu() {
    if skip_if_no_gpu_env() {
        return;
    }
    let got = drive_on_gpu(KvQuant::TurboSym4);
    assert!(
        got.k_gpu_mirror,
        "tsym4: the K axis must take the caller's device, and the store carries no GPU mirror"
    );
    assert!(
        got.v_gpu_mirror,
        "tsym4: the V axis was pinned to the CPU. The 4-bit V store has a GPU encode kernel and \
         the width rule hands it the caller's device"
    );
    assert_eq!(
        got.v_cpu_blocks, 0,
        "tsym4: the GPU V path writes the mirror and pushes no CPU block"
    );
}
