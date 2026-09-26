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
size ∈ {32, 64, 128}. One codec, `affine.rs`, covers them all.

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
