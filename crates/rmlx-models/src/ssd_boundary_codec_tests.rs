//! A hydrated boundary layer decodes like the layer it was spilled from.
//!
//! The layer policy floors the boundary layers of a `Mixed` or `RotK` base to
//! the 8-bit form of the same family, so those layers hold a store of the base
//! storage variant at other widths. The test builds the stack through
//! `kv_layer_quants`, as the arch does, spills it through the real writer,
//! hydrates it through `SsdHydrator::lookup`, and then decodes one step on every
//! layer of the hydrated stack and of the spilled stack. The two outputs must
//! agree on every layer.

use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_kv_ssd::{
    cache_seed, chained_block_hashes_seeded, hash_to_hex, write_caches, SsdHydrator, SsdKvIndex,
    BLOCK_TOKENS,
};
use rmlx_mlx::{Array, Device, Dtype};

use super::{arr, lcg, Namespace};
use crate::kv_cache::{kv_layer_quants, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};

/// A width the `Mixed` and `RotK` stores accept at group 64.
const HEAD_DIM: i32 = 128;
const KV_HEADS: i32 = 2;
const Q_HEADS: i32 = 4;
const MODEL_SIG: u64 = 0x0b0d_a2e5_1a7e_c0de;
const LAYOUT_KEY: u64 = 0x5eed_b0da_2e00_0001;

/// Largest difference allowed between a hydrated and a spilled layer's decode
/// output. The hydrated store holds the same packed bytes, so the only
/// difference that can come in is float reordering.
const DECODE_BOUND: f32 = 1e-5;

#[allow(
    clippy::expect_used,
    reason = "test fixture: an output that cannot be read back is a fixture bug and must abort the test loudly"
)]
fn to_f32(a: &Array) -> Vec<f32> {
    let f = a.astype(Dtype::F32, Device::Cpu).expect("astype f32");
    f.eval().expect("eval output");
    f.to_bytes()
        .expect("output readback")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four-byte chunk")))
        .collect()
}

/// The per-layer caches the arch builds for `base`, each driven through one
/// prefill bracket over a whole block.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a prefill on a freshly built cache that fails is a fixture bug and must abort the test loudly"
)]
fn prefilled_stack(base: KvQuant, n_layers: usize) -> Vec<KvCache> {
    let shape = [1_i32, KV_HEADS, BLOCK_TOKENS as i32, HEAD_DIM];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    kv_layer_quants(n_layers, base, false)
        .into_iter()
        .enumerate()
        .map(|(i, q)| {
            let mut c = KvCache::with_quant_max_seq(q, 2 * BLOCK_TOKENS as i32).with_layer_idx(i);
            let seed = 0x5100 ^ (i as u64) << 8;
            c.enter_prefill();
            c.update(
                &arr(&lcg(n, seed), &shape),
                &arr(&lcg(n, seed ^ 0xABCD), &shape),
                Device::Cpu,
            )
            .expect("prefill chunk");
            c.exit_prefill(Device::Cpu).expect("exit_prefill");
            c
        })
        .collect()
}

/// Spill `caches` as one block under `base`, then hydrate it back.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a spill or a hydrate of the block this test just wrote that fails is the failure under test and must abort loudly"
)]
fn spill_and_hydrate(tag: &str, base: KvQuant, caches: &[KvCache]) -> Vec<KvCache> {
    let device = Device::Cpu;
    let ns = Namespace::new(tag);
    let layer_quants = kv_layer_quants(caches.len(), base, false);
    let seed = cache_seed(LAYOUT_KEY, base, &layer_quants, MODEL_SIG);
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();

    let chained = chained_block_hashes_seeded(&prompt_ids, seed);
    let key = hash_to_hex(*chained.last().expect("prompt covers a whole block"));
    let path = ns.dir.join(format!("{key}.kvb"));
    write_caches(&path, device, &ns.name, base, caches, &[]).expect("spill block");
    let size = std::fs::metadata(&path).expect("block metadata").len();
    SsdKvIndex::open(&ns.name)
        .expect("index open")
        .record(&key, LAYOUT_KEY, &path, &ns.name, &base.to_string(), size)
        .expect("index record");

    SsdHydrator::open(&ns.name, LAYOUT_KEY, device)
        .expect("hydrator open")
        .lookup(&prompt_ids, seed, base, DispatchPolicy::default(), false)
        .expect("hydrate must not error")
        .expect("the block this test spilled must be found")
        .kv_caches
}

/// One decode step through the arch's attention entry.
fn decode_step(cache: &mut KvCache, layer: usize) -> Result<Vec<f32>, String> {
    let one = (KV_HEADS * HEAD_DIM) as usize;
    let seed = 0x7700 ^ layer as u64;
    let k = arr(&lcg(one, seed), &[1, KV_HEADS, 1, HEAD_DIM]);
    let v = arr(&lcg(one, seed ^ 0x11), &[1, KV_HEADS, 1, HEAD_DIM]);
    let q = arr(
        &lcg((Q_HEADS * HEAD_DIM) as usize, seed ^ 0x22),
        &[1, Q_HEADS, 1, HEAD_DIM],
    );
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    cache
        .update_and_sdpa(&q, &k, &v, scale, "", None, Device::Cpu)
        .map(|out| to_f32(&out))
        .map_err(|e| e.to_string())
}

fn assert_hydrated_boundary_decodes_like_spilled(tag: &str, base: KvQuant) {
    let n_layers = LAYER_ADAPTIVE_HEAD_N + LAYER_ADAPTIVE_TAIL_N + 2;
    let quants = kv_layer_quants(n_layers, base, false);
    assert!(
        quants.iter().any(|&q| q != base),
        "{base}: no boundary layer differs from the base, so this test reads nothing"
    );
    let mut spilled = prefilled_stack(base, n_layers);
    let mut hydrated = spill_and_hydrate(tag, base, &spilled);
    assert_eq!(hydrated.len(), n_layers, "{base}: hydrated layer count");

    let layers = spilled.iter_mut().zip(hydrated.iter_mut()).zip(&quants);
    for (layer, ((s, h), quant)) in layers.enumerate() {
        let want = decode_step(s, layer)
            .unwrap_or_else(|e| panic!("{base}: layer {layer} spilled decode: {e}"));
        let got = decode_step(h, layer).unwrap_or_else(|e| {
            panic!("{base}: layer {layer} (codec {quant}) hydrated decode: {e}")
        });
        assert_eq!(want.len(), got.len(), "{base}: layer {layer} output length");
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst <= DECODE_BOUND,
            "{base}: layer {layer} (codec {quant}) hydrated decode differs from the spilled \
             layer by {worst} (bound {DECODE_BOUND})"
        );
    }
}

#[test]
fn a_hydrated_mixed_boundary_layer_decodes_like_the_spilled_one() {
    assert_hydrated_boundary_decodes_like_spilled(
        "boundary-mixed",
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
    );
}

#[test]
fn a_hydrated_rot_k_boundary_layer_decodes_like_the_spilled_one() {
    assert_hydrated_boundary_decodes_like_spilled(
        "boundary-rotk",
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        },
    );
}
