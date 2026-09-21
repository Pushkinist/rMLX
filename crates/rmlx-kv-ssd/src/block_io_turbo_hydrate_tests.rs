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
//! 2. **`max_seq` on the hydrated K store.** This was the divergence: the
//!    3-bit hydrate forwarded the geometry's `max_seq` and the 4-bit one
//!    dropped it, because its `from_cpu_blocks` took no such argument. The
//!    K-storage collapse resolved it the 3-bit way, so both widths now restore
//!    the window that was written, and the 4-bit constant below moved from `0`
//!    to it. `docs/KV_TURBO_TWINS.md` records the decision.
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
//! * **`.eval()` on the loaded tensors.** The two readers disagreed on it and
//!   all three cells were green either way, which is the measurement: on a
//!   safetensors-loaded tensor the call changes no byte. The collapsed reader
//!   keeps the call, which is a convention decision this file cannot make for
//!   the reader — see the doc.
//! * **The GPU hydrate upload.** `append`'s hydrated-init branch, which the
//!   collapse reduced to the one safe scale-byte form, needs a `Device::Gpu`
//!   append after the hydrate. `make gpu-test` owns it.

use super::block_io_tests::lcg;
use super::{KvBlockReader, KvBlockWriter};
use rmlx_kv_quant::storage::{KvStorage, QuantKTurbo3, QuantKTurbo4, QuantV};
use rmlx_kv_quant::turboquant::{turbo_quantize_v, TurboBlocks};
use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use tempfile::TempDir;

const MODEL_ID: &str = "Qwen3ForCausalLM/turbo-hydrate-pin";

/// `[B, kv_h, S, D]` every cell is built at. `D` is a multiple of the turbo
/// group size and `kv_h > 1`, so the seq-major store layout is exercised.
const SHAPE: [i32; 4] = [1, 2, 4, 128];

/// Window the spilled cache states. Deliberately not zero and not the default:
/// the 4-bit hydrate restores `0` today whatever was written, so a zero here
/// would make the two widths agree for the wrong reason.
const WRITTEN_MAX_SEQ: i32 = 4096;

/// `max_seq` each width's K store carries after a hydrate.
///
/// The collapse gave both widths one `from_cpu_blocks` that takes the window,
/// so both restore what was written. The 4-bit constant is the one cell this
/// campaign moved, from `0`.
///
/// Two constants holding one value, on purpose: each names the width whose
/// cell it pins, so a future divergence moves one of them and the control
/// below turns red rather than both moving together and nothing saying so.
const HYDRATED_K_MAX_SEQ_3BIT: i32 = WRITTEN_MAX_SEQ;
const HYDRATED_K_MAX_SEQ_4BIT: i32 = WRITTEN_MAX_SEQ;

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
                WRITTEN_MAX_SEQ,
            )),
            v,
            max_seq: WRITTEN_MAX_SEQ,
        },
        _ => KvStorage::TurboSym4 {
            k: Some(QuantKTurbo4::from_cpu_blocks(
                vec![k_block.clone()],
                shape,
                WRITTEN_MAX_SEQ,
            )),
            v,
            max_seq: WRITTEN_MAX_SEQ,
        },
    };
    (storage, k_block)
}

/// What one hydrated K store answers — every field of the store, so a
/// constructor that starts filling one this file does not read cannot pass.
struct HydratedK {
    shape: Vec<i32>,
    bits: u8,
    max_seq: i32,
    blocks: Vec<TurboBlocks>,
    gpu_codes_live: bool,
    gpu_scales_live: bool,
    gpu_words_per_step: i32,
    gpu_scales_per_step: i32,
    gpu_capacity: i32,
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
    // A TempDir, not a named path removed on the success path: a failing
    // assertion below would otherwise leave the block behind, and the next run
    // of the same test in the same process id would read it.
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join(format!("{name}.safetensors"));

    KvBlockWriter::new(MODEL_ID, quant, &layers, &[])
        .write(&path, device)
        .expect("spill");
    let reader = KvBlockReader::open(&path).expect("open the spilled block");
    let (rebuilt, _bf16, _lin) = reader.hydrate(MODEL_ID, quant, device).expect("hydrate");

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
            let QuantKTurbo3 {
                blocks,
                gpu_codes_buf,
                gpu_scales_buf,
                gpu_words_per_step,
                gpu_scales_per_step,
                gpu_capacity,
                shape,
                bits,
                max_seq,
            } = k;
            HydratedK {
                shape,
                bits,
                max_seq,
                blocks,
                gpu_codes_live: gpu_codes_buf.is_some(),
                gpu_scales_live: gpu_scales_buf.is_some(),
                gpu_words_per_step,
                gpu_scales_per_step,
                gpu_capacity,
            }
        }
        KvStorage::TurboSym4 { k, max_seq, .. } => {
            assert_eq!(
                max_seq, WRITTEN_MAX_SEQ,
                "{name}: the storage-level window is what the decode path reads, and it did \
                 not survive the round trip"
            );
            let k = k.expect("hydrated K store");
            let QuantKTurbo4 {
                blocks,
                gpu_codes_buf,
                gpu_scales_buf,
                gpu_words_per_step,
                gpu_scales_per_step,
                gpu_capacity,
                shape,
                bits,
                max_seq,
            } = k;
            HydratedK {
                shape,
                bits,
                max_seq,
                blocks,
                gpu_codes_live: gpu_codes_buf.is_some(),
                gpu_scales_live: gpu_scales_buf.is_some(),
                gpu_words_per_step,
                gpu_scales_per_step,
                gpu_capacity,
            }
        }
        _ => panic!(
            "{name}: the hydrate returned a storage variant that is not the symmetric turbo \
             one the layout tag names"
        ),
    };
    (hydrated, k_block)
}

/// The spilled K payload comes back byte for byte, at both widths, and the
/// hydrated store claims no GPU buffer.
///
/// This is the half that cannot move. Both the store and its one block are
/// destructured exhaustively, so a field added to either — or a constructor
/// that starts filling one it used to leave at its default — fails to compile
/// here rather than passing unread.
fn assert_payload_survives(name: &str, hydrated: &HydratedK, written: &TurboBlocks, bits: u8) {
    assert_eq!(hydrated.shape, SHAPE.to_vec(), "{name}: accumulated shape");
    assert_eq!(hydrated.bits, bits, "{name}: store bit tag");
    assert_eq!(hydrated.blocks.len(), 1, "{name}: hydrated block count");
    let Some(block) = hydrated.blocks.first() else {
        panic!("{name}: the hydrate produced no block");
    };
    let TurboBlocks {
        codes,
        scales,
        original_shape,
        bits: block_bits,
    } = block;
    assert_eq!(*block_bits, bits, "{name}: per-block bit tag");
    assert_eq!(
        *original_shape, written.original_shape,
        "{name}: per-block original shape"
    );
    assert_eq!(
        *codes, written.codes,
        "{name}: the packed code plane changed across the spill/hydrate round trip"
    );
    assert_eq!(
        *scales, written.scales,
        "{name}: the scale plane changed across the spill/hydrate round trip"
    );

    // A hydrate builds a CPU-path store: no GPU mirror exists yet, and the
    // bookkeeping that sizes one is set by the first `append`, not here. A
    // constructor that invented a capacity would hand the next append a
    // geometry no buffer backs.
    assert!(
        !hydrated.gpu_codes_live,
        "{name}: the hydrated store claims a GPU codes buffer"
    );
    assert!(
        !hydrated.gpu_scales_live,
        "{name}: the hydrated store claims a GPU scales buffer"
    );
    assert_eq!(
        (
            hydrated.gpu_words_per_step,
            hydrated.gpu_scales_per_step,
            hydrated.gpu_capacity
        ),
        (0, 0, 0),
        "{name}: the hydrated store carries GPU buffer bookkeeping for a buffer it does not have"
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
fn tsym4_hydrate_restores_the_k_payload_and_the_window() {
    let (hydrated, written) = spill_and_hydrate("tsym4", KvQuant::TurboSym4, 4);
    assert_payload_survives("tsym4", &hydrated, &written, 4);
    assert_eq!(
        hydrated.max_seq, HYDRATED_K_MAX_SEQ_4BIT,
        "tsym4: the 4-bit hydrate forwards the geometry's max_seq into the K store, as the \
         3-bit one always did. This cell read 0 before the K-storage collapse; if it moved \
         back, the collapse's one intended observable change was reverted"
    );
}

/// The two widths answer the same, measured.
///
/// It spills and hydrates both widths and compares what the two engines
/// returned, then holds the agreed value to the window that was written.
/// Comparing the two pinned constants instead would execute no engine code: a
/// `from_cpu_blocks` that stopped forwarding the window at one width would
/// still leave a constant-to-constant control green, which is the one event
/// this test exists for.
#[test]
fn the_two_widths_hydrate_the_same_window() {
    let (three, _) = spill_and_hydrate("tsym3_control", KvQuant::TurboSym3, 3);
    let (four, _) = spill_and_hydrate("tsym4_control", KvQuant::TurboSym4, 4);
    assert_eq!(
        three.max_seq, four.max_seq,
        "the two widths restore a different K-store window. One from_cpu_blocks stopped \
         forwarding the geometry's max_seq"
    );
    assert_eq!(
        three.max_seq, WRITTEN_MAX_SEQ,
        "the two widths agree on a window that is not the one the spill wrote"
    );
}
