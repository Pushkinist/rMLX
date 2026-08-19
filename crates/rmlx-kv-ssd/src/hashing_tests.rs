//! Seed-partitioning tests for [`super::cache_seed`].
//!
//! The seed decides which stored blocks a request can even see. Every term in
//! it is a case where reusing another entry's K/V would be wrong, so each one
//! gets a test that fails if that term stops entering the formula.

use rmlx_kv_quant::KvQuant;

use super::cache_seed;

const LK: u64 = 0xa55a_5aa5_a55a_5aa5;
const SIG: u64 = 0x1234_5678_9abc_def0;

/// Two requests whose per-layer mixtures differ must not share a digest stream,
/// even at the same layout key, codec and model.
///
/// This is the term that makes a layer-policy change invalidate stored blocks
/// for **every** codec rather than only for requests running the launch
/// default: the layout key is fixed at attach, the seed is not. The two vectors
/// below are the same codec under the boundary promotion and without it.
#[test]
fn cache_seed_separates_two_layer_mixtures() {
    let n = 36usize;
    let uniform = vec![KvQuant::K8V4; n];
    let promoted: Vec<KvQuant> = (0..n)
        .map(|i| {
            if i < 2 || i >= n - 8 {
                KvQuant::K8V8
            } else {
                KvQuant::K8V4
            }
        })
        .collect();

    assert_ne!(
        cache_seed(LK, KvQuant::K8V4, &uniform, SIG),
        cache_seed(LK, KvQuant::K8V4, &promoted, SIG),
        "a per-layer mixture change must move the seed — otherwise blocks stored \
         under the old policy stay reachable to a request built under the new one"
    );
}

/// The mixture enters positionally: the same multiset of per-layer codecs in a
/// different order is a different layout and must seed differently.
#[test]
fn cache_seed_is_sensitive_to_mixture_order() {
    let head = [KvQuant::K8V8, KvQuant::K8V4, KvQuant::K8V4];
    let tail = [KvQuant::K8V4, KvQuant::K8V4, KvQuant::K8V8];
    assert_ne!(
        cache_seed(LK, KvQuant::K8V4, &head, SIG),
        cache_seed(LK, KvQuant::K8V4, &tail, SIG),
        "promoting the first layer and promoting the last are different layouts"
    );
}

/// Layer count enters too: the same codec on a 36- and a 40-layer model is not
/// the same layout.
#[test]
fn cache_seed_is_sensitive_to_layer_count() {
    assert_ne!(
        cache_seed(LK, KvQuant::K8V8, &vec![KvQuant::K8V8; 36], SIG),
        cache_seed(LK, KvQuant::K8V8, &vec![KvQuant::K8V8; 40], SIG),
        "layer count is part of the layout the seed partitions on"
    );
}

/// The pre-existing terms still partition: model identity and codec.
#[test]
fn cache_seed_still_separates_model_and_codec() {
    let mix_k8v8 = vec![KvQuant::K8V8; 8];
    let mix_k8v4 = vec![KvQuant::K8V4; 8];
    assert_ne!(
        cache_seed(LK, KvQuant::K8V8, &mix_k8v8, SIG),
        cache_seed(LK, KvQuant::K8V8, &mix_k8v8, SIG ^ 0xff),
        "two models must not share a digest stream"
    );
    assert_ne!(
        cache_seed(LK, KvQuant::K8V8, &mix_k8v8, SIG),
        cache_seed(LK, KvQuant::K8V4, &mix_k8v4, SIG),
        "two codecs must not share a digest stream"
    );
    assert_ne!(
        cache_seed(LK, KvQuant::K8V8, &mix_k8v8, SIG),
        cache_seed(LK ^ 1, KvQuant::K8V8, &mix_k8v8, SIG),
        "two layouts must not share a digest stream"
    );
}
