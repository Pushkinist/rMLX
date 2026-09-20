//! Pin on the SSD hydrate path of the two TurboQuant K stores.
//!
//! The CPU store-bytes oracle in
//! `crates/rmlx-kv-quant/src/kvcache/turbo_store_bytes_tests.rs` drives
//! `KvCache::update` and never reaches `from_cpu_blocks`, so the hydrate
//! constructor — the one place where the two width-named K stores take
//! genuinely different arguments — is outside every assertion it makes. This
//! file is that half.
//!
//! # What it holds
//!
//! Two things, kept apart on purpose because only one of them may move.
//!
//! 1. **The hydrated payload, field by field**, against the store that was
//!    written: the packed codes, the scales, the per-block original shape and
//!    bit tag, the accumulated shape and the block count. **This cannot move
//!    at either width.** The collapse changes no byte on the wire and no byte
//!    coming back off it.
//! 2. **`max_seq` on the hydrated K store.** This is the divergence, and today
//!    the two widths answer differently: `read_quant_k_turbo3` forwards the
//!    geometry's `max_seq` into `QuantKTurbo3::from_cpu_blocks`, while
//!    `QuantKTurbo4::from_cpu_blocks` takes no such argument and always sets
//!    `0`. The values below are what the tree does today, not what it should
//!    do; `docs/KV_TURBO_TWINS.md` records which width is the reference and
//!    therefore which of these two numbers the collapse moves.
//!
//! Field-by-field rather than one digest: a digest names the store, and these
//! assertions name the field, which is what a reviewer of a collapse needs.
//! The serialisation helpers the `rmlx-kv-quant` pins share are `pub(crate)`
//! to that crate and cannot be reached from here; copying them in to build a
//! digest would plant the twin this campaign removes.
//!
//! # What it cannot see
//!
//! * **The spill side's GPU trim.** `write_quant_k_turbo{3,4}` takes its
//!   `gpu_codes_buf` branch only when the store carries one, which a CPU-built
//!   store does not. The two write bodies are byte-identical anyway.
//! * **`.eval()` on the loaded tensors.** `read_quant_k_turbo4` calls it and
//!   `read_quant_k_turbo3` does not. Both cells below are green, which is the
//!   measurement: on a safetensors-loaded tensor the call changes no byte. It
//!   is a convention decision, not a behaviour one, and this file cannot make
//!   it for the reader — see the doc.
//! * **The GPU hydrate upload.** `append`'s hydrated-init branch, where the
//!   two stores encode their scale bytes differently (one safe, one `unsafe`),
//!   needs a `Device::Gpu` append after the hydrate. `make gpu-test` owns it.

use super::{KvBlockReader, KvBlockWriter};
use rmlx_kv_quant::storage::{KvStorage, QuantKTurbo3, QuantKTurbo4, QuantV};
use rmlx_kv_quant::turboquant::{turbo_quantize_v, TurboBlocks};
use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;

const MODEL_ID: &str = "Qwen3ForCausalLM/turbo-hydrate-pin";

/// `[B, kv_h, S, D]` every cell is built at. `D` is a multiple of the turbo
/// group size and `kv_h > 1`, so the seq-major store layout is exercised.
const SHAPE: [i32; 4] = [1, 2, 4, 128];

/// Window the spilled cache states. Deliberately not zero and not the default:
/// the 4-bit hydrate restores `0` today whatever was written, so a zero here
/// would make the two widths agree for the wrong reason.
const WRITTEN_MAX_SEQ: i32 = 4096;

/// `max_seq` each width's K store carries after a hydrate, as the tree stands.
///
/// The 4-bit store has no way to receive the value: its `from_cpu_blocks` does
/// not take one. That is the divergence the collapse resolves, and the
/// resolution moves one of these two numbers.
const HYDRATED_K_MAX_SEQ_3BIT: i32 = WRITTEN_MAX_SEQ;
const HYDRATED_K_MAX_SEQ_4BIT: i32 = 0;

/// Deterministic f32 data in [-1, 1].
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

fn tmp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "rmlx_turbo_hydrate_{name}_{}.safetensors",
        std::process::id()
    ));
    p
}

/// One CPU-path symmetric turbo storage at `bits`, plus the K block it holds.
#[allow(
    clippy::expect_used,
    reason = "test fixture: the shape and bit width are ones the codec accepts, so a failure is a defect in the codec and the panic names it"
)]
fn build(bits: u8) -> (KvStorage, TurboBlocks) {
    let n: usize = SHAPE.iter().map(|&d| d as usize).product();
    let k_block = turbo_quantize_v(&lcg(n, 0x1234_5678), bits, &SHAPE).expect("encode K");
    let v_block = turbo_quantize_v(&lcg(n, 0xABCD_4321), bits, &SHAPE).expect("encode V");
    let shape = SHAPE.to_vec();
    let v = Some(QuantV::from_cpu_blocks(vec![v_block], shape.clone(), bits));
    let storage = match bits {
        3 => KvStorage::TurboSym3 {
            k: Some(QuantKTurbo3::from_cpu_blocks(
                vec![k_block.clone()],
                shape,
                bits,
                WRITTEN_MAX_SEQ,
            )),
            v,
            max_seq: WRITTEN_MAX_SEQ,
        },
        _ => KvStorage::TurboSym4 {
            k: Some(QuantKTurbo4::from_cpu_blocks(
                vec![k_block.clone()],
                shape,
                bits,
            )),
            v,
            max_seq: WRITTEN_MAX_SEQ,
        },
    };
    (storage, k_block)
}

/// What one hydrated K store answers.
struct HydratedK {
    shape: Vec<i32>,
    bits: u8,
    max_seq: i32,
    blocks: Vec<TurboBlocks>,
}

/// Spill one symmetric turbo cache and hydrate it back.
#[allow(
    clippy::expect_used,
    reason = "test driver: a cache this file just built writes and reads back, so a failure is the defect under test and the panic names it"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the hydrated variant is the one the KvQuant selected; any other is a hydrate bug, and the explicit panic names it sooner than a wrong field would"
)]
fn spill_and_hydrate(name: &str, quant: KvQuant, bits: u8) -> (HydratedK, TurboBlocks) {
    let device = Device::Cpu;
    let (storage, k_block) = build(bits);
    let layers = vec![storage];
    let path = tmp_path(name);

    KvBlockWriter::new(MODEL_ID, quant, &layers, &[])
        .write(&path, device)
        .expect("spill");
    let reader = KvBlockReader::open(&path).expect("open the spilled block");
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, quant, device).expect("hydrate");
    let _ = std::fs::remove_file(&path);

    assert_eq!(rebuilt.len(), 1, "{name}: layer count");
    let layer = rebuilt.into_iter().next().expect("one layer");
    let hydrated = match layer {
        KvStorage::TurboSym3 { k, max_seq, .. } => {
            assert_eq!(
                max_seq, WRITTEN_MAX_SEQ,
                "{name}: the storage-level window is what the decode path reads, and it did \
                 not survive the round trip"
            );
            let k = k.expect("hydrated K store");
            HydratedK {
                shape: k.shape,
                bits: k.bits,
                max_seq: k.max_seq,
                blocks: k.blocks,
            }
        }
        KvStorage::TurboSym4 { k, max_seq, .. } => {
            assert_eq!(
                max_seq, WRITTEN_MAX_SEQ,
                "{name}: the storage-level window is what the decode path reads, and it did \
                 not survive the round trip"
            );
            let k = k.expect("hydrated K store");
            HydratedK {
                shape: k.shape,
                bits: k.bits,
                max_seq: k.max_seq,
                blocks: k.blocks,
            }
        }
        _ => panic!(
            "{name}: the hydrate returned a storage variant that is not the symmetric turbo \
             one the layout tag names"
        ),
    };
    (hydrated, k_block)
}

/// The spilled K payload comes back byte for byte, at both widths.
///
/// This is the half that cannot move. Every field is named so a collapse that
/// drops one says which.
#[allow(
    clippy::indexing_slicing,
    reason = "test: the block count is asserted immediately above every index"
)]
fn assert_payload_survives(name: &str, hydrated: &HydratedK, written: &TurboBlocks, bits: u8) {
    assert_eq!(hydrated.shape, SHAPE.to_vec(), "{name}: accumulated shape");
    assert_eq!(hydrated.bits, bits, "{name}: store bit tag");
    assert_eq!(hydrated.blocks.len(), 1, "{name}: hydrated block count");
    let block = &hydrated.blocks[0];
    assert_eq!(block.bits, bits, "{name}: per-block bit tag");
    assert_eq!(
        block.original_shape, written.original_shape,
        "{name}: per-block original shape"
    );
    assert_eq!(
        block.codes, written.codes,
        "{name}: the packed code plane changed across the spill/hydrate round trip"
    );
    assert_eq!(
        block.scales, written.scales,
        "{name}: the scale plane changed across the spill/hydrate round trip"
    );
}

/// The 3-bit symmetric turbo cache survives a spill and a hydrate.
#[test]
fn tsym3_hydrate_restores_the_k_payload_and_the_window() {
    let (hydrated, written) = spill_and_hydrate("tsym3", KvQuant::TurboSym3, 3);
    assert_payload_survives("tsym3", &hydrated, &written, 3);
    assert_eq!(
        hydrated.max_seq, HYDRATED_K_MAX_SEQ_3BIT,
        "tsym3: the 3-bit hydrate forwards the geometry's max_seq into the K store. If this \
         moved to 0, the collapse resolved the from_cpu_blocks divergence the other way and \
         the doc says which is the reference"
    );
}

/// The 4-bit symmetric turbo cache survives a spill and a hydrate.
#[test]
fn tsym4_hydrate_restores_the_k_payload_but_not_the_window() {
    let (hydrated, written) = spill_and_hydrate("tsym4", KvQuant::TurboSym4, 4);
    assert_payload_survives("tsym4", &hydrated, &written, 4);
    assert_eq!(
        hydrated.max_seq, HYDRATED_K_MAX_SEQ_4BIT,
        "tsym4: the 4-bit hydrate drops the geometry's max_seq because \
         QuantKTurbo4::from_cpu_blocks takes none. If this moved to the written window, the \
         collapse resolved the from_cpu_blocks divergence and this is the expected change"
    );
}

/// The two widths really do answer differently, and the pin above is reading
/// that and not a constant.
///
/// Without this, both cells could be asserting the same number and the
/// divergence would be invisible — which is the state the collapse creates and
/// which must be a deliberate, visible move rather than a quiet one.
#[test]
fn the_two_widths_hydrate_a_different_window_today() {
    assert_ne!(
        HYDRATED_K_MAX_SEQ_3BIT, HYDRATED_K_MAX_SEQ_4BIT,
        "the two widths now restore the same K-store window. That is the collapse's one \
         intended observable change, and the cell it moved is named by whichever of the two \
         pins above was re-baselined"
    );
}
