"""Regenerate the DFlash 2 reference fixtures from the upstream MLX reference.

rMLX runs no Python. This script exists so the two committed fixtures are
reproducible rather than magic blobs: it drives `dflash/model_mlx.py` from
`z-lab/dflash` at the pinned commit below and writes what that reference
produces.

    pip install mlx mlx-lm numpy
    python generate.py scale
    RMLX_O_MODELS_ROOT=<models-root> python generate.py published

`scale` writes, beside this file:

  * `model.safetensors` + `config.json` — a synthetic drafter at the same tensor
    names and relationships as the published checkpoint, small enough to commit.
  * `reference.safetensors` — the inputs and the hidden states the reference
    produces from them, one per case, plus the chains its candidate selector
    traces over the `wide` case.

`published` writes `../dflash2_published_reference.safetensors`: the hidden
states and selector chains the reference produces from the real
`z-lab/Qwen3.8-27B-DFlash2` weights. Its inputs are not committed — both sides
generate them from the integer recurrence in `synthetic` and from
`selector_logits`, which `tests/dflash2_loader.rs` repeats.

The consumers are `speculative/dflash2/forward_tests.rs` and
`speculative/dflash2/selector_tests.rs` (scale) and `tests/dflash2_loader.rs`
(published).
"""

import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent

# z-lab/dflash, the commit whose model_mlx.py carries the DFlash 2 forward.
REFERENCE_COMMIT = "07ebd93db9f472af339b644bb70221ad8428328a"
REFERENCE_URL = (
    f"https://raw.githubusercontent.com/z-lab/dflash/{REFERENCE_COMMIT}/dflash/model_mlx.py"
)


def load_reference():
    """Fetch the upstream reference into a temp dir and import it.

    Deliberately not cached inside the repo: the reference is not rMLX's to
    vendor, and a stale copy beside the fixtures would be indistinguishable from
    the pinned one.
    """
    scratch = Path(tempfile.mkdtemp(prefix="dflash2-reference-"))
    with urllib.request.urlopen(REFERENCE_URL) as r:
        (scratch / "model_mlx.py").write_bytes(r.read())
    sys.path.insert(0, str(scratch))
    import model_mlx

    return model_mlx


# --- the scale model -------------------------------------------------------

H = 64
GROUP = 16
LAYERS = 2
N_HEADS = 4
N_KV = 2
HEAD_DIM = 16
INTER = 32
VOCAB = 40
RANK = 8
TOPK = 4
BLOCK = 4
THETA = 1.0e7
EPS = 1e-6
TARGETS = [0, 1]
SELECTOR_WEIGHT_SCALE = 0.25

# --- the stand-in verifier head --------------------------------------------

# The selector's unary term is the verifier's LM head over the drafter's hidden
# states. Neither model here has one: the scale drafter has no verifier at all,
# and the published pair's is a 27 B 4-bit head whose output nothing could
# commit. The cases below stand one in -- a floor with `spikes` peaks a fixed
# distance apart -- and both sides build it with the same exact arithmetic, so
# the comparison never depends on a matmul kernel.
#
# The spacing is what makes the case decidable, and it is chosen per model
# rather than shared: the peaks are `step` apart, which is dozens of bf16 places
# at these magnitudes, so the k-th and (k+1)-th candidate cannot tie and the
# candidate set does not depend on how a partition orders equals. The spread
# across the kept candidates -- `(spikes - 1) * step` -- is set comparable to
# the model's own pairwise term, so neither term decides every position. Both
# models were measured: a spread far above the pairwise term makes the chain the
# per-position argmax, and one far below makes it the pairwise argmax, and
# either way one of the two terms could be dropped without the case noticing.
SELECTOR_FLOOR = -8.0
SELECTOR_POS_STRIDE = 12289
SELECTOR_ID_STRIDE = 18
SCALE_SELECTOR_STEP = 0.3125
PUBLISHED_SELECTOR_STEP = 0.09375
SCALE_ANCHORS = (3, 27)
PUBLISHED_ANCHORS = (100, 50000)


def selector_logits(positions, vocab, spikes, step):
    """Stand-in verifier logits, `[1, positions, vocab]` bf16.

    `tests/dflash2_loader.rs::stand_in_logits` repeats this, value for value.
    """
    lg = np.full((1, positions, vocab), SELECTOR_FLOOR, dtype=np.float32)
    for t in range(positions):
        for j in range(spikes):
            token = (t * SELECTOR_POS_STRIDE + j * SELECTOR_ID_STRIDE) % vocab
            lg[0, t, token] = j * step
    return mx.array(lg).astype(mx.bfloat16)


def selector_case(model, hidden_block, spikes, step, anchors, vocab):
    """The reference's chain over `hidden_block`'s drafted positions.

    `hidden_block` is the whole block; the drafted positions are the rest of it
    after the seed, which is the reference's own `logits_start = 1`.
    """
    hidden = hidden_block[:, 1:]
    logits = selector_logits(hidden.shape[1], vocab, spikes, step)
    out = {
        "selector_logits": logits,
        "selector_top1": mx.argmax(logits, axis=-1).astype(mx.uint32)[0],
    }
    for i, anchor in enumerate(anchors):
        path, _, _ = model.candidate_selector.select(
            hidden, logits, mx.array([anchor]), 0.0
        )
        mx.eval(path)
        out[f"selector_chain_{i}"] = path.astype(mx.uint32)[0]
        print(f"  selector anchor {anchor} -> {path.tolist()[0]}")
    print(f"  selector top-1      -> {out['selector_top1'].tolist()}")
    return out


CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "is_causal": False,
    "dflash_config": {
        "block_size": BLOCK,
        "conv_group_size": GROUP,
        "conv_kernel_size": 2,
        "mask_token_id": 39,
        "selector_rank": RANK,
        "selector_top_k": TOPK,
        "target_layer_ids": TARGETS,
    },
    "head_dim": HEAD_DIM,
    "hidden_size": H,
    "intermediate_size": INTER,
    "layer_types": ["sliding_attention"] * LAYERS,
    "model_type": "qwen3",
    "num_attention_heads": N_HEADS,
    "num_hidden_layers": LAYERS,
    "num_key_value_heads": N_KV,
    "rms_norm_eps": EPS,
    "rope_parameters": {"rope_theta": THETA, "rope_type": "default"},
    "sliding_window": 64,
    "vocab_size": VOCAB,
}


def scale_config(model_mlx, sliding_window):
    return model_mlx.DFlashConfig(
        hidden_size=H,
        num_hidden_layers=LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        intermediate_size=INTER,
        vocab_size=VOCAB,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        max_position_embeddings=262144,
        block_size=BLOCK,
        target_layer_ids=tuple(TARGETS),
        num_target_layers=LAYERS,
        mask_token_id=39,
        rope_scaling={"rope_theta": THETA, "rope_type": "default"},
        layer_types=tuple(["sliding_attention"] * LAYERS),
        sliding_window=sliding_window,
        conv_kernel_size=2,
        conv_group_size=GROUP,
        selector_rank=RANK,
        selector_top_k=TOPK,
        is_causal=False,
    )


def scale_weights(rng):
    """Checkpoint-named tensors at the shapes the config predicts."""
    w = {}

    def normal(shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_w(shape):
        return (1.0 + rng.standard_normal(shape) * 0.1).astype(np.float32)

    w["fc.weight"] = normal((H, len(TARGETS) * H))
    w["hidden_norm.weight"] = norm_w((H,))
    w["norm.weight"] = norm_w((H,))
    # The selector's three tensors are drawn wider than the rest. Their product
    # is a rank-RANK triple product, so at the trunk's 0.05 the pairwise term
    # lands around 0.009 -- below one bf16 place at any logit scale that leaves
    # the candidate set unambiguous, which would make a bf16 reference case for
    # the selector unable to separate anything. The draw consumes the same
    # stream either way, so every other tensor here is unchanged by this scale.
    w["candidate_selector.hidden_projection.weight"] = normal((RANK, H), SELECTOR_WEIGHT_SCALE)
    w["candidate_selector.predecessor_codebook"] = normal((VOCAB, RANK), SELECTOR_WEIGHT_SCALE)
    w["candidate_selector.successor_codebook"] = normal((VOCAB, RANK), SELECTOR_WEIGHT_SCALE)
    groups = H // GROUP
    for i in range(LAYERS):
        p = f"layers.{i}"
        w[f"{p}.input_layernorm.weight"] = norm_w((H,))
        w[f"{p}.post_attention_layernorm.weight"] = norm_w((H,))
        w[f"{p}.self_attn.q_proj.weight"] = normal((N_HEADS * HEAD_DIM, H))
        w[f"{p}.self_attn.k_proj.weight"] = normal((N_KV * HEAD_DIM, H))
        w[f"{p}.self_attn.v_proj.weight"] = normal((N_KV * HEAD_DIM, H))
        w[f"{p}.self_attn.o_proj.weight"] = normal((H, N_HEADS * HEAD_DIM))
        w[f"{p}.self_attn.q_norm.weight"] = norm_w((HEAD_DIM,))
        w[f"{p}.self_attn.k_norm.weight"] = norm_w((HEAD_DIM,))
        w[f"{p}.mlp.gate_proj.weight"] = normal((INTER, H))
        w[f"{p}.mlp.up_proj.weight"] = normal((INTER, H))
        w[f"{p}.mlp.down_proj.weight"] = normal((H, INTER))
        for which in ("attention_conv", "mlp_conv"):
            # Both taps at the same magnitude: a base near 1 on tap 0 and near 0
            # on tap 1 would make a dropped tap almost invisible.
            w[f"{p}.{which}.base_kernel"] = normal((2, 2, H), scale=0.5)
            w[f"{p}.{which}.kernel_projection.weight"] = normal((2 * 2 * groups, H))
    return w


def reference_hidden(model, block, target_hidden, cache_offset=0):
    """`DFlash2DraftModel.hidden_states`, minus the embedding lookup."""
    cache = model.make_cache()
    for c in cache:
        c.offset = cache_offset
    h = block
    h_ctx = model.hidden_norm(model.fc(target_hidden))
    for layer, c in zip(model.layers, cache):
        h = layer(h, h_ctx, model.rope, c)
    return model.norm(h)


def for_load(weights):
    """The codebooks carry no `.weight` suffix on disk; `load_draft` renames."""
    out = dict(weights)
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        out[f"{key}.weight"] = out.pop(key)
    return out


def build_scale(model_mlx):
    rng = np.random.default_rng(20260906)
    bf16 = {k: mx.array(v).astype(mx.bfloat16) for k, v in scale_weights(rng).items()}
    mx.save_safetensors(str(HERE / "model.safetensors"), bf16)
    (HERE / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    loadable = for_load(bf16)

    def randn(shape):
        return mx.array((rng.standard_normal(shape) * 0.5).astype(np.float32)).astype(
            mx.bfloat16
        )

    block = randn((1, BLOCK, H))
    target_hidden = randn((1, 6, len(TARGETS) * H))
    target_hidden_short = randn((1, 2, len(TARGETS) * H))
    out = {
        "block_hidden": block,
        "target_hidden": target_hidden,
        "target_hidden_short": target_hidden_short,
    }
    # (name, sliding_window, conditioning, cache offset seeded before the call).
    # The last case exists to measure the position-shift invariance the port
    # relies on; nothing asserts equality against it.
    cases = [
        ("wide", 64, target_hidden, 0),
        ("window_trims", 4, target_hidden, 0),
        ("short_context", 64, target_hidden_short, 0),
        ("wide_at_offset", 64, target_hidden, 17),
    ]
    for name, window, ctx, offset in cases:
        model = model_mlx.DFlash2DraftModel(scale_config(model_mlx, window))
        model.eval()
        model.load_weights(list(loadable.items()))
        mx.eval(model.parameters())
        hidden = reference_hidden(model, block, ctx, offset)
        mx.eval(hidden)
        out[f"hidden_{name}"] = hidden
        print(f"{name}: window {window}, ctx {ctx.shape[1]}, offset {offset} -> {hidden.shape}")

    selector_model = model_mlx.DFlash2DraftModel(scale_config(model_mlx, 64))
    selector_model.eval()
    selector_model.load_weights(list(loadable.items()))
    mx.eval(selector_model.parameters())
    print("selector, over the `wide` case's hidden states:")
    out.update(
        selector_case(
            selector_model,
            out["hidden_wide"],
            TOPK + 2,
            SCALE_SELECTOR_STEP,
            SCALE_ANCHORS,
            VOCAB,
        )
    )

    mx.save_safetensors(str(HERE / "reference.safetensors"), out)
    print("wrote", HERE / "reference.safetensors")


# --- the published checkpoint ----------------------------------------------

SLUG = "z-lab__Qwen3.8-27B-DFlash2"
PUBLISHED_BLOCK_LEN = 8
PUBLISHED_CTX_LEN = 3


def synthetic(count, seed):
    """`synthetic_input` in `tests/dflash2_loader.rs`, value for value."""
    return np.array(
        [
            (((seed + i) * 1103515245 + 12345) % 2147483648) / 2147483648.0 - 0.5
            for i in range(count)
        ],
        dtype=np.float32,
    )


def build_published(model_mlx):
    root = os.environ.get("RMLX_O_MODELS_ROOT")
    if not root:
        raise SystemExit("set RMLX_O_MODELS_ROOT to the directory holding " + SLUG)
    path = Path(root) / SLUG
    raw = json.loads((path / "config.json").read_text())
    d = raw["dflash_config"]
    config = model_mlx.DFlashConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_parameters"]["rope_theta"],
        max_position_embeddings=raw["max_position_embeddings"],
        block_size=d["block_size"],
        target_layer_ids=tuple(d["target_layer_ids"]),
        num_target_layers=raw["num_target_layers"],
        mask_token_id=d["mask_token_id"],
        rope_scaling=raw["rope_parameters"],
        layer_types=tuple(raw["layer_types"]),
        sliding_window=raw["sliding_window"],
        conv_kernel_size=d["conv_kernel_size"],
        conv_group_size=d["conv_group_size"],
        selector_rank=d["selector_rank"],
        selector_top_k=d["selector_top_k"],
        is_causal=raw["is_causal"],
    )
    weights = {k: v for f in path.glob("*.safetensors") for k, v in mx.load(str(f)).items()}
    model = model_mlx.DFlash2DraftModel(config)
    model.eval()
    model.load_weights(list(for_load(weights).items()))
    mx.eval(model.parameters())

    h_dim = config.hidden_size
    concat = len(config.target_layer_ids) * h_dim
    block = mx.array(
        synthetic(PUBLISHED_BLOCK_LEN * h_dim, 0).reshape(1, PUBLISHED_BLOCK_LEN, h_dim)
    ).astype(mx.bfloat16)
    ctx = mx.array(
        synthetic(PUBLISHED_CTX_LEN * concat, 1000003).reshape(1, PUBLISHED_CTX_LEN, concat)
    ).astype(mx.bfloat16)

    hidden = reference_hidden(model, block, ctx)
    mx.eval(hidden)
    f = hidden.astype(mx.float32)
    print("hidden", hidden.shape, "absmax", float(mx.max(mx.abs(f))))
    saved = {"hidden": hidden}
    print("selector, over that block's drafted positions:")
    saved.update(
        selector_case(
            model,
            hidden,
            config.selector_top_k + 4,
            PUBLISHED_SELECTOR_STEP,
            PUBLISHED_ANCHORS,
            config.vocab_size,
        )
    )
    # The stand-in logits are rebuilt on the Rust side from the same arithmetic;
    # a vocabulary-wide array is not a thing to commit.
    del saved["selector_logits"]
    out = HERE.parent / "dflash2_published_reference.safetensors"
    mx.save_safetensors(str(out), saved)
    print("wrote", out)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    model_mlx = load_reference()
    if which == "scale":
        build_scale(model_mlx)
    elif which == "published":
        build_published(model_mlx)
    else:
        raise SystemExit("usage: generate.py {scale|published}")


if __name__ == "__main__":
    main()
