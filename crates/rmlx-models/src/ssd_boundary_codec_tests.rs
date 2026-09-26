//! A hydrated boundary layer decodes like the layer it was spilled from.
//!
//! The layer policy floors the boundary layers of a `Mixed` or `RotK` base: to
//! the 8-bit form of the same family on a stack that does not share K/V, and to
//! `K8V8` on a stack that does. Either way a boundary layer's codec differs from
//! the base. The test builds the stack through `kv_layer_quants`, as the arch
//! does, spills it through the real writer, hydrates it through
//! `SsdHydrator::lookup`, and then decodes one step on every layer of the
//! hydrated stack and of the spilled stack, through the attention entry the
//! topology uses. The two outputs must agree on every layer, except one case:
//! on a stack that shares K/V, a hydrated layer at the base `Mixed` / `RotK`
//! codec holds no bf16 mirror, and the shared-source entry refuses it. The test
//! holds that refusal for those layers.

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

const MIXED_K8_V4: KvQuant = KvQuant::Mixed {
    k_bits: 8,
    v_bits: 4,
    k_group_size: 64,
    v_group_size: 64,
};
const ROT_K_V4: KvQuant = KvQuant::RotK {
    v_bits: 4,
    v_group_size: 64,
};

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

/// The per-layer caches the arch builds from `layer_quants`, each driven
/// through one prefill bracket over a whole block.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a prefill on a freshly built cache that fails is a fixture bug and must abort the test loudly"
)]
fn prefilled_stack(layer_quants: &[KvQuant], shares_kv: bool) -> Vec<KvCache> {
    let shape = [1_i32, KV_HEADS, BLOCK_TOKENS as i32, HEAD_DIM];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    layer_quants
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let mut c = KvCache::with_quant_max_seq(q, 2 * BLOCK_TOKENS as i32)
                .with_layer_idx(i)
                .with_shares_kv(shares_kv);
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
fn spill_and_hydrate(
    tag: &str,
    base: KvQuant,
    layer_quants: &[KvQuant],
    shares_kv: bool,
    caches: &[KvCache],
) -> Vec<KvCache> {
    let device = Device::Cpu;
    let ns = Namespace::new(tag);
    let seed = cache_seed(LAYOUT_KEY, base, layer_quants, MODEL_SIG);
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
        .lookup(
            &prompt_ids,
            seed,
            base,
            layer_quants,
            DispatchPolicy::default(),
            shares_kv,
        )
        .expect("hydrate must not error")
        .expect("the block this test spilled must be found")
        .kv_caches
}

/// One decode step through the attention entry an arch of this topology uses.
fn decode_step(cache: &mut KvCache, layer: usize, shares_kv: bool) -> Result<Vec<f32>, String> {
    let one = (KV_HEADS * HEAD_DIM) as usize;
    let seed = 0x7700 ^ layer as u64;
    let k = arr(&lcg(one, seed), &[1, KV_HEADS, 1, HEAD_DIM]);
    let v = arr(&lcg(one, seed ^ 0x11), &[1, KV_HEADS, 1, HEAD_DIM]);
    let q = arr(
        &lcg((Q_HEADS * HEAD_DIM) as usize, seed ^ 0x22),
        &[1, Q_HEADS, 1, HEAD_DIM],
    );
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let out = if shares_kv {
        cache
            .update_and_sdpa_shared_source(&q, &k, &v, scale, "", None, Device::Cpu)
            .map(|(out, _shared)| out)
    } else {
        cache.update_and_sdpa(&q, &k, &v, scale, "", None, Device::Cpu)
    };
    out.map(|o| to_f32(&o)).map_err(|e| e.to_string())
}

fn assert_hydrated_boundary_decodes_like_spilled(tag: &str, base: KvQuant, shares_kv: bool) {
    let n_layers = LAYER_ADAPTIVE_HEAD_N + LAYER_ADAPTIVE_TAIL_N + 2;
    let quants = kv_layer_quants(n_layers, base, shares_kv);
    assert!(
        quants.iter().any(|&q| q != base),
        "{base} (shares_kv={shares_kv}): no boundary layer differs from the base, so this \
         test reads nothing"
    );
    let mut spilled = prefilled_stack(&quants, shares_kv);
    let mut hydrated = spill_and_hydrate(tag, base, &quants, shares_kv, &spilled);
    assert_eq!(hydrated.len(), n_layers, "{base}: hydrated layer count");

    let layers = spilled.iter_mut().zip(hydrated.iter_mut()).zip(&quants);
    for (layer, ((s, h), quant)) in layers.enumerate() {
        let at = format!("{base} (shares_kv={shares_kv}): layer {layer} (codec {quant})");
        let want =
            decode_step(s, layer, shares_kv).unwrap_or_else(|e| panic!("{at} spilled decode: {e}"));
        let hydrated_decode = decode_step(h, layer, shares_kv);
        if shares_kv && quant.uses_mixed_path() {
            // `from_storage` restores no bf16 mirror, and the shared-source
            // entry refuses a `Mixed` / `RotK` producer that holds none.
            let Err(e) = hydrated_decode else {
                panic!("{at}: hydrated decode must refuse");
            };
            assert!(
                e.contains("holds no bf16 K/V mirror"),
                "{at}: hydrated decode failed for another reason: {e}"
            );
            continue;
        }
        let got = hydrated_decode.unwrap_or_else(|e| panic!("{at} hydrated decode: {e}"));
        assert_eq!(want.len(), got.len(), "{at} output length");
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst <= DECODE_BOUND,
            "{at} hydrated decode differs from the spilled layer by {worst} \
             (bound {DECODE_BOUND})"
        );
    }
}

#[test]
fn a_hydrated_mixed_boundary_layer_decodes_like_the_spilled_one() {
    assert_hydrated_boundary_decodes_like_spilled("boundary-mixed", MIXED_K8_V4, false);
}

#[test]
fn a_hydrated_rot_k_boundary_layer_decodes_like_the_spilled_one() {
    assert_hydrated_boundary_decodes_like_spilled("boundary-rotk", ROT_K_V4, false);
}

#[test]
fn a_hydrated_mixed_boundary_layer_on_a_sharing_stack_decodes_like_the_spilled_one() {
    assert_hydrated_boundary_decodes_like_spilled("boundary-mixed-shared", MIXED_K8_V4, true);
}

#[test]
fn a_hydrated_rot_k_boundary_layer_on_a_sharing_stack_decodes_like_the_spilled_one() {
    assert_hydrated_boundary_decodes_like_spilled("boundary-rotk-shared", ROT_K_V4, true);
}
