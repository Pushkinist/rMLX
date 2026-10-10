# Weight Quantization Reference

The weight formats rMLX loads: bit layout, scale and bias encoding, and how
each one reaches a matmul. KV-cache codecs are a separate axis; see
`docs/KV_QUANT.md`.

---

## 1. Overview

| Format | Bits | Group size | Scale | Bias | Inference path |
|---|---|---|---|---|---|
| `bf16` | 16 | — | — | — | plain matmul |
| `mxfp8` | 8 | 32 | E8M0, 1 B | — | MLX `quantized_matmul`, mode `mxfp8` |
| `mxfp4` | 4 | 32 | E8M0, 1 B | — | MLX `quantized_matmul`, mode `mxfp4` |
| `nvfp4` | 4 | 16 | E4M3, 1 B | — | MLX `quantized_matmul`, mode `nvfp4` |
| affine `qN_gG` | 2, 3, 4, 5, 6, 8 | 32, 64, 128 | float (bf16 or f16 on disk), cast to bf16 | same as scale (additive) | MLX `quantized_matmul`, mode `affine` |
| ternary (BitLinear) | 2 per trit | tensor-wide | bf16 scalar | — | unpacked to bf16 at load, plain matmul |
| ParoQuant | 4 (affine) | per checkpoint | f16 | f16 | activation rotation, then `quantized_matmul` |
| Prism Hadamard pack | 2 (affine) | 128 | f16, cast to bf16 | f16, cast to bf16 | signed block Hadamard of the input, then `quantized_matmul` (section 10) |

---

## 2. bf16 — unquantized baseline

bf16 is the upper 16 bits of an IEEE 754 f32: 1 sign bit, 8 exponent bits
(bias 127), 7 mantissa bits, stored little-endian.

```
bf16 byte layout (LE):
  byte[0] = bits[7:0]
  byte[1] = bits[15:8]   (sign + exponent + mantissa MSBs)

Decode: f32 = reinterpret_cast<f32>(u32::from(u16::from_le_bytes) << 16)
```

Source: `crates/rmlx-quant/src/bf16.rs`

---

## 3. MXFP family — Microscaling FP (OCP spec)

A per-group scale byte and element bytes or nibbles. The element encoding
differs per variant.

### 3.1 E8M0 scale (shared)

8 bits, unsigned exponent, bias 127. No sign, no mantissa.

```
value = 2^(e - 127)   for e in [0, 254]
0xFF  → NaN
0x00  → 2^(-127)
```

The CPU decoder turns a `0xFF` scale into NaN for its whole group and warns
once per call.

### 3.2 mxfp8 — E8M0 scale + E4M3 elements

- **Group size**: 32
- **Scale storage**: 1 E8M0 byte per group, shape `[rows, cols/32]`
- **Elements**: 1 byte each, OCP E4M3: 1 sign + 4 exponent (bias 7) +
  3 mantissa

```
E4M3 layout: s[7] | e[6:3] | m[2:0]

Normal   (1 ≤ e ≤ 14): (-1)^s × 2^(e-7) × (1 + m/8)
Subnormal (e == 0)    : (-1)^s × 2^(-6) × (m/8)
NaN                   : e == 0xF, m == 0x7  (bytes 0x7F and 0xFF)
No infinity (OCP E4M3 FN).

Dequant: w = e4m3_decode(element_byte) × e8m0_decode(scale_byte)
```

**Loader scale-dtype contract.** The E8M0 scale byte is an exponent, not a
float. MLX `dequantize` and `quantized_matmul` reject any scale dtype other
than `uint8` for `mxfp8`/`mxfp4`. So loaders keep those `.scales` at their
on-disk `uint8`. `load_util::bf16_scales` casts only float scales (bf16, f16,
f32) to bf16 and passes every integer dtype through. A bf16-cast E8M0 scale
fails at the first prefill with `Scale type must be uint8`.

Source: `crates/rmlx-quant/src/mxfp.rs`, `crates/rmlx-quant/src/fp8.rs`

### 3.3 mxfp4 — E8M0 scale + E2M1 elements

- **Group size**: 32
- **Scale storage**: 1 E8M0 byte per group, shape `[rows, cols/32]`
- **Elements**: two nibbles per byte, `rows × (cols/2)` bytes, low nibble
  first

```
E2M1 layout (4-bit nibble): s[3] | e[2:1] | m[0], exponent bias 1.

Normal   (e ≥ 1): (-1)^s × 2^(e-1) × (1 + m/2)
Subnormal (e==0): (-1)^s × m/2

All 16 values:
  0x0=+0.0  0x1=+0.5  0x2=+1.0  0x3=+1.5
  0x4=+2.0  0x5=+3.0  0x6=+4.0  0x7=+6.0
  0x8=-0.0  0x9=-0.5  0xA=-1.0  0xB=-1.5
  0xC=-2.0  0xD=-3.0  0xE=-4.0  0xF=-6.0
No NaN, no infinity.

Byte packing: element[2i] = byte & 0xF, element[2i+1] = byte >> 4

Dequant: w = e2m1_decode(nibble) × e8m0_decode(scale_byte)
```

Source: `crates/rmlx-quant/src/mxfp.rs`, `crates/rmlx-quant/src/fp4.rs`

### 3.4 nvfp4 — E4M3 scale + E2M1 elements

- **Elements**: E2M1 nibbles, packed as mxfp4
- **Group size**: 16
- **Scale storage**: 1 byte per group, shape `[rows, cols/16]`

The spec defines the scale as unsigned UE4M3. MLX implements it as signed
E4M3, and MLX-produced snapshots follow MLX. The CPU decoder supports both
through `compat_mlx_signed_scale`:

```
UE4M3 (compat_mlx_signed_scale = false, the decoder default):
  e = (byte >> 3) & 0xF, m = byte & 0x7, bias 7
  Normal (e ≥ 1): 2^(e-7) × (1 + m/8), max 480
  Subnormal (e==0): 2^(-6) × (m/8)
  No NaN; 0xFF decodes to 480.0

E4M3 (compat_mlx_signed_scale = true):
  signed OCP E4M3, range ±240

Dequant: w = e2m1_decode(nibble) × scale_decode(scale_byte)
```

KV calibration decodes nvfp4 with `compat_mlx_signed_scale = true`, matching
MLX.

Source: `crates/rmlx-quant/src/mxfp.rs`, `crates/rmlx-quant/src/fp8.rs`

#### Group 16 and MLX's split-K partition

nvfp4's group (16) is narrower than the 32-wide K tile of MLX's
`qmm_t_splitk` kernels. The linked MLX aligns each split-K partition to
`group_size` alone. At group 16 it can hand the kernel a partition that is not
a whole number of tiles. The kernel then reads past it into the next group's
codes and scales and returns wrong values for every element, with no error.

Shape alone decides whether it fires: `transpose=true`, a 2-D weight, a batch
at or above MLX's vector-kernel limit, and a partition `K / split_k` that is
not a multiple of 32. So it hits prefill and speculative verify; single-token
decode runs the vector kernel.

`rmlx_mlx::ops::quantized_matmul` mirrors MLX's split-K arithmetic
(`splitk_safe_rows`). When the partition would not be tile-whole, it grows the
batch with zero rows onto one that is, then slices them off. Zero rows cannot
change the kept rows. The guard is inert for any group of 32 or more. It is
also inert below `qmv_batch_limit_floor`, the smallest vector-kernel limit any
Apple GPU uses for the shape. mlx-c exposes no GPU-architecture query, so the
floor is the minimum over MLX's branches; the guard may pad batches the
device would have run on the vector kernel. `matmul_tests.rs` bounds that
over-pad.

Upstream aligns the partition to `max(group_size, 32)` after MLX 0.32.
`linked_mlx_still_carries_the_misaligned_split_k_partition` fails once the
linked MLX is past 0.32. The guard is then to be deleted, not carried.

---

## 4. Affine quants — block-affine weight quantization

The affine family is every `qN_gG` with bits ∈ {2, 3, 4, 5, 6, 8} and group
size ∈ {32, 64, 128}. One codec, `affine.rs`, spells all of them, but its
CPU decoder only actually matches MLX's byte layout at 2, 4 and 8 bits —
see § 4.2.

### 4.1 Dequant formula

```
w_fp = scale × code + bias
```

`bias` is additive and already carries its sign: it equals
`-zero_point × scale`.

### 4.2 Packed-code layout

MLX stores the codes as a `uint32` tensor, bit-packed LSB-first with no
padding. The pack unit depends on the width (MLX `get_pack_factor` /
`get_bytes_per_pack`):

| bits | codes per pack | bytes per pack |
|---|---|---|
| 2, 4, 8 | 32 / bits | 4 |
| 3 | 8 | 3 |
| 5 | 8 | 5 |
| 6 | 4 | 3 |

So a row of `cols` codes takes `cols × bits / 8` bytes at every width.

The CPU codec in `affine.rs` reads `floor(32 / bits)` codes per 32-bit word
at every width (`CodeStorage::U32Le`), leaving the top bits of a word unused.
That matches MLX at 2, 4 and 8 bits only. It also reads a byte-packed
`CodeStorage::U8` layout, used in round-trip tests.

### 4.3 Scale / bias storage

`scales` and `biases` are each `[rows, cols / group_size]`, row-major. The
loader casts float scales to bf16 (`load_util::bf16_scales`).

### 4.4 Supported combinations

Every pairing of bits {2, 3, 4, 5, 6, 8} with group size {32, 64, 128}. The
tag is `q<bits>_g<group>`, for example `q4_g64`.

**Load-time bit-width gate.** A `bits` value outside
`rmlx_quant::affine::SUPPORTED_BITS` has no kernel in the linked mlx-c.
`rmlx_models::arch::loader` checks the model's declared `quantization.bits`,
the global default and every `quantization.tensor_overrides` entry, before any
tensor I/O. An unsupported width fails the load with one error, instead of
failing at the first prefill. See `docs/ADDING_A_MODEL.md` for how the gate
composes with a new architecture.

**Hadamard packs.** A Prism Hadamard pack stores affine 2-bit weights folded
with a signed block Hadamard transform. Section 10 covers it.

Source: `crates/rmlx-quant/src/affine.rs`

---

## 5. Where quantized weights are decoded

A quantized `Linear` or `Embedding` keeps its packed codes, scales and biases
as loaded; `QuantMode` (`crates/rmlx-models/src/layers/quant.rs`) names the
mode. MLX decodes them on the GPU inside `quantized_matmul`, `dequantize` and
`gather_qmm`. Only ternary weights are unpacked at load (section 9).

The CPU decoders in `crates/rmlx-quant` (`bf16`, `mxfp`, `fp8`, `fp4`,
`affine`) serve KV calibration (`rmlx-loader::calibration`) and the
`rmlx info` probe. `rmlx-quant::awq` converts AWQ-packed ParoQuant tensors at
load.

---

## 6. Adding a new weight-quant format

A format reaches inference only through a kernel the linked MLX has. It needs
a `QuantMode` variant whose string MLX's `quantized_matmul` accepts, loader
support for its scale dtype, and a smoke probe (`rmlx serve`, reject
incoherent output). A CPU decoder in `crates/rmlx-quant` is needed only for
calibration or `rmlx info`.

---

## 7. ParoQuant — weight rotation, MLX-native INT4

ParoQuant (`z-lab/paroquant`) rotates the **input activations** in channel
pairs before a standard affine INT4 matmul. The weights are affine INT4; the
rotation is on the activation side.

### 7.1 How it differs from PlanarQuant

| | PlanarQuant (KV codec) | ParoQuant |
|---|---|---|
| What is rotated | Stored values, in pairs | Input activations, in channel pairs |
| When | At dequant | Before each matmul |
| Storage | Codebook codes + per-pair scale | Affine INT4 |

### 7.2 Loading

A PARO checkpoint carries, per linear, `qweight`, `scales`, `qzeros`,
`theta`, `pairs` and `channel_scales`. `qweight` and `qzeros` are AWQ-packed;
`rmlx_quant::awq` converts them to MLX's affine layout and additive biases at
load. `cos_theta` and `sin_theta` are computed from `theta` at load, as F16.
The result is `Linear::Paro`.

### 7.3 Rotation kernel inputs

`paro_rotate_gpu` takes:

- `x`: `[batch, hidden]` activations, F16 or BF16.
- `packed_pairs`: `[krot, hidden/2]` I32; low 16 bits `i_local`, high 16 bits
  `j_local`, within the group.
- `cos_theta`, `sin_theta`: `[krot, hidden/2]` F16.
- `channel_scales`: `[hidden]` F16.
- `krot`: rotation rounds, at most `MAX_KROT` (16).
- `group_size`: channel group width, at most `MAX_GROUP_SIZE` (256).

A `krot` or `group_size` over its bound is an `Error::Quant`.

### 7.4 Rotation algorithm

For each group of `group_size` channels and each tile of `ROWS_PER_TILE`
rows:

```
x[c] = x[c] × channel_scales[c]        for every channel c in the group
for round in 0..krot:
    for each pair (i, j) of the group in packed_pairs[round]:
        (x[i], x[j]) = (x[i] × cos + x[j] × sin, x[j] × cos - x[i] × sin)
```

The rotated activations then go through `quantized_matmul` with group size,
4 bits and mode `affine`.

`ROWS_PER_TILE` (1 for a single row, 4 otherwise), `MAX_KROT` and
`MAX_GROUP_SIZE` are MLX template ints supplied at dispatch. One registration
covers both variants, and the bounds come from the Rust consts the validation
checks.

Dispatcher: `crates/rmlx-models/src/paroquant_msl.rs`
MSL body: `crates/rmlx-models/src/metal/paroquant_rotate.metal`
External reference: `z-lab/paroquant/paroquant/kernels/metal/rotation.metal`

The `z-lab/` snapshots (e.g. `z-lab__Qwen3.6-27B-PARO`) use this format.

---

## 8. Storage-size reference

Bytes on disk per weight element:

| Format | Codes | Metadata |
|---|---|---|
| bf16 | 2.00 | — |
| mxfp8 | 1.00 | 1 B / 32 elements |
| mxfp4 | 0.50 | 1 B / 32 elements |
| nvfp4 | 0.50 | 1 B / 16 elements |
| affine N-bit, group G | N / 8 | 4 B / G elements (16-bit scale + bias) |
| ternary | 0.25 | one bf16 scalar per tensor |

---

## 9. Ternary / BitLinear (`BitNetForCausalLM`)

BitNet b1.58 stores each linear weight as a U8 tensor `[N/4, K]` holding four
2-bit fields per byte, LSB first. The value is `raw - 1`:

```
bits [1:0] → field 0      raw 0 → -1
bits [3:2] → field 1      raw 1 →  0
bits [5:4] → field 2      raw 2 → +1
bits [7:6] → field 3      raw 3 → +2 (never in valid data)
```

Field `t` of packed row `r` is logical row `t × (N/4) + r`: a strided
interleave, not four consecutive rows.

Each weight has a sibling `*.weight_scale`, a BF16 scalar `[1]`. At load
`dequant_trit_u8` multiplies it in and returns a BF16 `[N, K]` matrix, so
inference is a plain BF16 matmul with no special kernel.

Source: `crates/rmlx-models/src/bitnet/loader.rs`

---

## 10. Prism Hadamard pack

A Prism Hadamard pack (`model_type` `prism_hadamard_qwen35`) stores affine
2-bit weights (section 4) of a dense Qwen3.5 model, folded with a signed
block Walsh-Hadamard transform. Its `config.json` names no `architectures`;
the pack dispatches on the class of its `base_model_type` (`docs/MODELS.md`,
Overview).

### 10.1 Contract

- `config.json` names the contract file in `hadamard_config` and lists each
  folded module in `modules` (`path`, `block`, `embedding`, `dtype`).
- The contract file (keys `prism.hadamard.*`, version 1) names the transform
  `normalized-sylvester-walsh-hadamard`, the axis `input-last-dimension`, the
  sign mode `explicit` and `gdn_v_grouped` true. It gives the block size, one
  `±1` sign vector per input width, the forward tensors (`weight_names`) and
  the inverse tensors (`inverse_weight_names`, the embedding).
- Each module has a `<module>.signs` tensor (F32, one value per input column)
  equal to the contract's vector of its width.
- A linear layer computes `y = W' . H(D x)`. The embedding computes
  `e = D H dequant(E'[id])`. `H` is the block-diagonal normalized Sylvester
  matrix and `D` the diagonal of the signs. Both directions compute in f32 and
  round once to the input dtype. No layer has an output-side transform, so
  the residual stream, the norms, the delta-net recurrence, RoPE, attention
  and the KV cache see ordinary values.
- Not folded: the norms, `A_log`, `dt_bias`, `conv1d` and the dense F32
  `in_proj_a` / `in_proj_b`.

### 10.2 Where it is applied

`HadamardContract::read` (`crates/rmlx-loader/src/hadamard.rs`) parses and
checks the contract. The Qwen3.5 loader (`load_from_path` in
`crates/rmlx-models/src/qwen3_5_moe/loader.rs`) folds each contract module
through `HadamardFold` (`crates/rmlx-models/src/qwen3_5_moe/layers.rs`): the
quantized layer becomes `Linear::Hadamard` or `Embedding::Hadamard`, with one
`SignedBlockHadamard` (`crates/rmlx-models/src/hadamard.rs`) per input width,
over MLX's `hadamard_transform` (`rmlx_mlx::hadamard_transform`). The load
fails on a contract module the loader built no layer for, and logs
`Qwen3_5: Hadamard pack, folding its modules` with the module counts.

### 10.3 What is checked

`check_hadamard_pack` (`crates/rmlx-models/src/arch/loader.rs`) is the one
check. `load_model`, the `serve` startup check (`--model`, `--draft-model`
and each model a `--registry` file names) and the draft-model reader
(`draft_kind_of`) call it. Embedding, Whisper and TTS loaders do not. The
embedding route takes only the `JinaEmbeddingsV4Model` class, which a pack
never resolves to; Whisper and TTS load only the snapshot their own flag
names (`--whisper-model-path`, `--tts-model-path`).

For a snapshot with no `hadamard_config` whose directory cannot be listed,
the check returns a loader error that names the directory, not a pack
refusal.

A snapshot is a pack when `config.json` names `hadamard_config`, or when a
visible `*.safetensors` file holds a tensor whose last name component is
`signs`. rMLX serves a pack when all of these hold:

- `base_model_type` is `qwen3_5` and the dispatch class is
  `Qwen3_5ForConditionalGeneration`;
- the checkpoint is not ParoQuant (`quantization_config.quant_method`);
- `HadamardContract::read` accepts the contract: the version and the three
  strings above, `gdn_v_grouped` true, a block of 512, 1024, 2048 or 4096
  that divides every width, `±1` sign values of the length the widths need,
  `modules` equal to the two contract lists, each record `float16` at the
  contract block, each `.signs` tensor in exactly one shard of the load and
  equal to the contract's vector, no `.signs` tensor without a module
  record, no quantized linear under the contract's tensor prefix without a
  module record, `components.mtp` and `tie_word_embeddings` false,
  and `gdn_activation_layout` `grouped` when present.

Any other pack is refused with one error that names the fact and the file
that holds it. A `.signs` tensor with no `hadamard_config` is refused by its
shard and its name. That scan skips hidden files, and it logs and skips a
file it cannot parse.

Two more refusals come later. The Qwen3.5 loader refuses a sparse MoE layer
in a pack: a Hadamard linear has no expert dispatch. `Eagle3Drafter::load`
refuses a drafter with an active restricted-vocabulary head over a pack
verifier (`Architecture::refuse_restricted_vocab_head`): that head gathers
folded output rows and multiplies them with an unrotated hidden state.
`serve` makes the same refusal at startup, exit 1, for `--model` a pack and
`--draft-model` an EAGLE-3 drafter whose config declares a reduced
vocabulary (`Eagle3Config::restricted_vocab`), so the server does not start
to refuse every request.

### 10.4 Limits

- Text only, as every Qwen3.5 snapshot: the loader does not read the vision
  tower the checkpoint carries.
- The EAGLE-3 restricted-vocabulary head, sparse MoE layers, a tied head and
  an MTP component are refused (section 10.3).
- Each module transforms its own input. One decode token runs 402 transforms
  where 258 distinct inputs would do. The cost is not measured.

### 10.5 Activation dtype and the noise floor

The pack's `modules` records name `float16`. The vendor runtime (the
`runtime/` directory of the snapshot) rounds the embedding output to f16 and
keeps f16 scales, but its F32 norm weights promote the hidden state to f32
after the first norm: the hidden state after layers 0 and 63, and the
logits, are f32. rMLX accepts the record and computes the pack as every
affine model: bf16 activations and bf16-cast scales (`bf16_scales` in
`crates/rmlx-models/src/load_util.rs`).

The noise floor is the error of bf16 activations alone. A change in
precision is judged by its error as a multiple of that floor. Per
projection, the root-gated CPU test
`the_bf16_scale_cast_is_bounded_by_the_bf16_activation_floor`
(`crates/rmlx-models/src/qwen3_5_moe/layers_tests.rs`) measures the scale
cast against f16 activations, the dtype the record names, on real rows, in
f64 with exact accumulation, and bounds rMLX's path by a fixed multiple of
the floor. For the logits, see 10.6.

`in_proj_a` and `in_proj_b` stay F32, as stored: the loader builds an
unquantized linear in its stored dtype. With bf16 activations their outputs
are f32, so `a + dt_bias` and the softplus run in f32. The root-gated CPU
test `a_bf16_cast_of_the_gate_projections_raises_the_delta_net_gate_error_above_the_floor`
(`crates/rmlx-models/src/qwen3_5_moe/layers_tests.rs`) is a design
measurement, not a check of the loader. By the same method on four
gated-delta-net layers, it shows the cost the F32 choice avoids: a
`bf16_param` cast of the two weights with bf16 outputs would raise the
log-gate error to 1.46 to 2.44 times the floor. The F32 arm is the floor plus
one f32 rounding, so it stays at the floor by construction. The structure
test of the synthetic pack checks that the loader builds both as plain F32
layers.

**The seam tests.** `load_model_serves_a_pack_as_the_plain_model_its_contract_unfolds`
(`crates/rmlx-models/src/qwen3_5_moe/pack_serve_tests.rs`) loads a synthetic pack
through `load_model` and compares its logits with the plain model the
contract unfolds to, `D H dequant(W')` per module, at input widths 512 and
1024 with block 512. On the CPU (GPU forbidden, attention layer only: the
gated-delta-net recurrence is a Metal kernel) it is a wiring check, not a
precision check. MLX's CPU `quantized_matmul` accumulates a bf16 input in
bf16: the pack misses the F32 model by 0.283 of the largest logit, and by
0.016 when the inner matmul is given an f32 input, so the accumulation is
about 94 percent of the gap. A transform run in bf16 instead of f32 stays
inside the CPU bound. The test requires the two controls, no transform and
the other direction, to miss the bound by 2 times. Its GPU copy,
`load_model_serves_a_pack_as_the_plain_model_its_contract_unfolds_on_gpu`,
adds the gated-delta-net layer, compares with a bf16 oracle, and bounds the
difference at 0.03 of the largest logit.

### 10.6 Logits against the vendor runtime

`crates/rmlx-models/tests/fixtures/hadamard_pack_vendor/logits.safetensors`
holds the vendor runtime's logits, computed on the GPU, for three sequences
of 64 token ids of `prism-ml__Ternary-Bonsai-2-27B-mlx-2bit` (a chat turn,
Python code, a word problem). For each position it holds the 20 largest
logits of the runtime as shipped, their ids and the log-sum-exp over the
vocabulary (the reference). It also holds the same runtime after mlx-vlm's
own bf16 cast, at those ids (the floor). That cast is the one
`mlx_vlm.convert` applies: every floating parameter but `A_log`. The script
`generate.py` beside the file writes it, with the pins of the runtime's
`requirements.txt`.

The GPU test
`the_real_pack_meets_the_vendor_logits_and_misses_them_without_its_transforms_on_gpu`
(`crates/rmlx-models/src/qwen3_5_moe/pack_reference_tests.rs`) loads the pack
through `load_model` and runs two arms, each with fresh caches: one prefill
forward over each sequence (192 positions), and a decode arm that prefills
the first 48 ids and decodes the other 16 one at a time (positions 47 to 63,
51 positions; the reference there is the vendor's prefill). It reads four
statistics against the reference:

- `reach_max`: the largest difference over the ranks a sampler reaches, the
  ranks whose reference probability is at least 1e-4, and rank 0. A logit
  error `d` moves the probability of its token by at most `p (e^d - 1)`, and
  the KL statistic weighs the rest.
- `top_rms`: the rms difference over the reference's top 20.
- `kl`: a KL divergence on the top 20 plus one bucket for the rest of the
  vocabulary.
- the positions where the argmax over the vocabulary differs.

Each arm passes when:

- `reach_max`, `top_rms` and `kl` are each at most 2 times the floor's, over
  the same positions;
- the argmax agrees at each position whose reference top-2 margin is above
  the floor's `reach_max` (the rule of the proof plan in the issue);
- the argmax differs at no more than 2 times as many positions as the
  floor's argmax. The proof plan has no such count; it only makes the gate
  stricter.

Controls, in the same run: the model with the transform of one module
removed (the `down_proj` of layer 0, then of layer 63, each put back after)
must break the gate, and the model with each Hadamard layer replaced by the
folded layer it wraps must exceed each of the floor's three statistics by
more than 10 times. The CPU tests in the same file check the gate on planted
cases and check the fixture reader, and pin the committed floor's figures.

**A change after measurement.** The gate first read the largest difference
over all 20 ranks. Measured, that statistic sat at 1.91 of the 2 allowed. Its
floor value was one logit at rank 20 with a reference probability of 9e-6,
which no sampler draws, so the margin depended on a single unreachable
logit. The statistic was then changed to `reach_max`, and the argmax rule
was changed from twice the floor's largest difference to the proof plan's
rule. The factor 2 did not change. The largest difference over all 20 ranks
is still printed.

The test prints its figures, as multiples of the floor's (`reach_max`,
`top_rms`, `kl`):

| arm | floor (absolute) | rMLX | argmax differs |
|---|---|---|---|
| prefill, 192 positions | 0.482, 0.0440, 1.75e-4 | 1.67, 1.29, 1.34 | 1 position, as the floor, margin 0.003 |
| decode, 51 positions | 0.482, 0.0488, 2.11e-4 | 1.26, 1.22, 1.03 | none |
| control: layer 0 `down_proj` | | 20.9, 24.9, 411 | 15 positions; gate broken |
| control: layer 63 `down_proj` | | 23.1, 53.9, 1321 | 25 positions; gate broken |
| control: no transform | | 66, 255, 70 000 | every position |

The resolution of the gate: one transform missing out of 402 moves each
statistic to more than 10 times its bound. An error smaller than 2 times
the floor's is not seen. Over the full vocabulary (a run outside the tests),
rMLX's prefill rms difference is 1.31 times the floor's, and its largest
difference is below the floor's.

Source: `crates/rmlx-loader/src/hadamard.rs`,
`crates/rmlx-models/src/qwen3_5_moe/layers.rs`,
`crates/rmlx-models/src/arch/loader.rs`
