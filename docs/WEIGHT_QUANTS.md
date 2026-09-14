# Weight Quantization Reference

Formats supported by rMLX for quantized weight storage. Covers bit layout,
scale / zero-point encoding, packed-byte representation, dequant cost, and
Metal kernel references where applicable.

See also: [KV_CACHE.md](KV_CACHE.md) for KV-side quantization, and
[MODELS.md](MODELS.md) for per-arch weight-quant usage.

---

## 1. Overview

| Format | Bits | Group size | Scale dtype | Zero-point | MSL kernel |
|---|---|---|---|---|---|
| `bf16` | 16 | — | — | — | — |
| `mxfp8` | 8 | 32 | E8M0 (1 B) | — | — |
| `mxfp4` | 4 | 32 | E8M0 (1 B) | — | — |
| `nvfp4` | 4 | 16 | UE4M3 / E4M3 (1 B) | — | — |
| `q8_g128` | 8 | 128 | bf16 | bf16 (additive) | `q8_msl.rs` |
| `q8_g64` | 8 | 64 | bf16 | bf16 (additive) | — |
| `q8_g32` | 8 | 32 | bf16 | bf16 (additive) | — |
| `q6_g64` | 6 | 64 | bf16 | bf16 (additive) | — |
| `q5_g64` | 5 | 64 | bf16 | bf16 (additive) | — |
| `q4_g128` | 4 | 128 | bf16 | bf16 (additive) | — |
| `q4_g64` | 4 | 64 | bf16 | bf16 (additive) | — |
| `q4_g32` | 4 | 32 | bf16 | bf16 (additive) | — |
| `q3_g64` | 3 | 64 | bf16 | bf16 (additive) | — |
| `q2_g64` | 2 | 64 | bf16 | bf16 (additive) | — |
| `turboquant` | 1–4 | 32 | f32 per block | — (codebook) | `turboquant_msl.rs` |
| `planarquant` | 1–4 | 32 | f32 per pair | — (codebook) | `planarquant_msl.rs` |
| `paroquant` | 4 (INT4 affine) | varies | pre-rotation | — | `paroquant_msl.rs` |
| `ternary` (BitLinear) | 2 (eff. ~1.58) | tensor-wide | bf16 scalar | — | none (BF16 at load) |

---

## 2. bf16 — unquantized baseline

bf16 (Brain Float 16) is the native MLX weight dtype for unquantized models.
The layout is the upper 16 bits of an IEEE 754 f32: 1 sign bit, 8 exponent
bits (bias 127), 7 mantissa bits. On Apple Silicon, tensors are stored
little-endian.

```
bf16 byte layout (LE):
  byte[0] = bits[7:0]   (low byte of the 16-bit pattern)
  byte[1] = bits[15:8]  (high byte, carries sign + exponent + mantissa MSBs)

Decode: f32 = reinterpret_cast<f32>(u32::from(u16::from_le_bytes) << 16)
```

Dequant cost: zero — bf16 is already the compute dtype; no conversion
kernel runs. Weights are mmap'd and used directly.

Source: `crates/rmlx-quant/src/bf16.rs`

---

## 3. MXFP family — Microscaling FP (OCP spec)

All three variants share the same outer structure: a per-group scale byte
followed by element bytes (or nibbles). The scale is an E8M0 exponent-only
value. Element encoding differs per variant.

### 3.1 E8M0 scale (shared)

8 bits, unsigned exponent, bias 127. No sign, no mantissa.

```
value = 2^(e - 127)   for e in [0, 254]
0xFF  → NaN           (broken snapshot; rMLX warns once and propagates NaN)
0x00  → 2^(-127)      (valid but very small; ~5.9e-39)
```

One E8M0 byte per group. For mxfp8/mxfp4 the group is 32 elements; for
nvfp4 the group is 16 elements.

### 3.2 mxfp8 — E8M0 scale + E4M3 elements

- **Bits per element**: 8 (1 byte, unpacked)
- **Group size**: 32
- **Scale storage**: 1 E8M0 byte per group, shape `[rows, cols/32]`
- **Element storage**: `rows × cols` bytes, 1 byte per element
- **Element encoding**: OCP E4M3 — 1 sign + 4 exponent (bias 7) + 3 mantissa

```
E4M3 layout: s[7] | e[6:3] | m[2:0]

Normal   (1 ≤ e ≤ 14): (-1)^s × 2^(e-7) × (1 + m/8)
Subnormal (e == 0)    : (-1)^s × 2^(-6) × (m/8)
NaN                   : e == 0xF, m == 0x7  (bytes 0x7F and 0xFF)
No Infinity in OCP E4M3 FN profile.

Dequant: w = e4m3_decode(element_byte) × e8m0_decode(scale_byte)
```

**Loader scale-dtype contract.** The E8M0 scale byte is an exponent, not a
float — MLX `dequantize`/`quantized_matmul` reject any scale dtype other than
`uint8` for the `mxfp8`/`mxfp4` modes. Loaders must keep mxfp8/mxfp4 `.scales`
at their on-disk `uint8` dtype; the float-uniformity cast that lifts fp16 affine
scales to bf16 (`load_util::bf16_scales`) is gated on `scales.dtype()` and skips
non-float scales. Casting an E8M0 scale to bf16 corrupts the exponent and crashes
the dequant kernel at first prefill (`Scale type must be uint8`).

Primary test target: `mlx-community__gemma-4-e4b-it-mxfp8`.

Source: `crates/rmlx-quant/src/mxfp.rs`, `crates/rmlx-quant/src/fp8.rs`

### 3.3 mxfp4 — E8M0 scale + E2M1 elements

- **Bits per element**: 4 (two nibbles per byte)
- **Group size**: 32
- **Scale storage**: 1 E8M0 byte per group, shape `[rows, cols/32]`
- **Element storage**: `rows × (cols/2)` bytes, low nibble first

```
E2M1 layout (4-bit nibble): s[3] | e[2:1] | m[0]
Exponent bias = 1.

Normal   (e ≥ 1): (-1)^s × 2^(e-1) × (1 + m/2)
Subnormal (e==0): (-1)^s × 2^0 × (m/2) = (-1)^s × m/2

Value table (exhaustive — only 16 values exist):
  0x0=+0.0  0x1=+0.5  0x2=+1.0  0x3=+1.5
  0x4=+2.0  0x5=+3.0  0x6=+4.0  0x7=+6.0
  0x8=-0.0  0x9=-0.5  0xA=-1.0  0xB=-1.5
  0xC=-2.0  0xD=-3.0  0xE=-4.0  0xF=-6.0

No NaN, no Infinity in E2M1.

Byte packing: byte = (hi_nibble << 4) | lo_nibble
  element[2i]   ← byte & 0xF   (low nibble)
  element[2i+1] ← byte >> 4    (high nibble)

Dequant: w = e2m1_decode(nibble) × e8m0_decode(scale_byte)
```

Source: `crates/rmlx-quant/src/mxfp.rs`, `crates/rmlx-quant/src/fp4.rs`

### 3.4 nvfp4 — UE4M3 scale + E2M1 elements

- **Bits per element**: 4 (two nibbles per byte, same as mxfp4)
- **Group size**: 16 (half of mxfp4)
- **Scale storage**: 1 byte per group (UE4M3 or E4M3), shape `[rows, cols/16]`
- **Scale encoding**: two modes

```
UE4M3 (Blackwell-correct, default):
  e = (byte >> 3) & 0xF   (4 bits, unsigned)
  m = byte & 0x7           (3 bits)
  bias = 7
  Normal (e ≥ 1): 2^(e-7) × (1 + m/8)   — always positive, max ≈ 480
  Subnormal (e==0): 2^(-6) × (m/8)
  No NaN reservation. 0xFF → finite 480.0

E4M3 compat mode (compat_mlx_signed_scale = true):
  Standard signed OCP E4M3 (bit 7 treated as sign).
  Range ≈ ±240, 137× less than UE4M3.
  Use for MLX-produced nvfp4 snapshots (ml-explore/mlx#2962 parity).

Dequant: w = e2m1_decode(nibble) × scale_decode(scale_byte)
```

The default (`compat_mlx_signed_scale = false`) uses UE4M3 per the Blackwell
spec. Pass `compat_mlx_signed_scale = true` only to load MLX-produced snapshots
that used the signed-E4M3 path.

Source: `crates/rmlx-quant/src/mxfp.rs`, `crates/rmlx-quant/src/fp8.rs`

#### Group 16 and MLX's split-K partition

nvfp4 is the only codec here whose group (16) is narrower than the 32-wide K
tile MLX's `qmm_t_splitk` kernels step the contracted dimension by. MLX aligns
each split-K partition to `group_size` alone, so at group 16 it can hand the
kernel a partition that is not a whole number of tiles — the kernel then reads
past it into the following group's codes and scales and silently returns wrong
values for **every** element, at full magnitude, with no error raised.

Whether it fires is decided by shape, not by the weights: it needs
`transpose=true`, a 2-D weight (more precisely `out.size() / M / N == 1`), a
batch at or above MLX's vector-kernel limit, and a partition `K / split_k` that
is not a multiple of 32. That makes it a
**prefill** and speculative-verify defect — single-token decode runs the vector
kernel and is unaffected. On a gemma-4-class nvfp4 checkpoint
(`hidden_size` 2560) the `k_proj` / `v_proj` pair at `N=512` is corrupt for
batches of 10–32 tokens and the per-layer input gate at `N=256` for 33–64, so
any prompt of roughly 10–64 tokens is affected.

`rmlx_mlx::ops::quantized_matmul` mirrors MLX's split-K arithmetic and, when
the partition would not be tile-whole, grows the batch with zero rows onto a
partition that is — then slices them off. Zero rows cannot change the rows that
are kept. It is inert for every group at or above 32, and for batches below the
smallest vector-kernel limit any Apple GPU uses for the shape (14, 10 or 6,
from the minimum over every branch of MLX's `get_qmv_batch_limit`).

That floor is deliberately pessimistic, and the cost is worth stating plainly:
mlx-c exposes no GPU-architecture query, and MLX's real crossover is 10–32
depending on architecture and shape, so between the floor and the device's
actual limit the guard pads batches MLX would have run on the vector kernel.
Erring low is the only safe direction — the opposite error leaves a tiled
matmul unguarded and silently wrong. The over-pad is bounded at 4.64x of an
already-small batch and is pinned by a test.

Reproduction, per-shape exposure tables and the end-to-end measurements are in
`.rmlx/nvfp4-splitk-partition.md`, which is runtime state and not checked in;
the mechanism above is the checked-in account, and the guard's own tests
(`crates/rmlx-mlx/src/ops/matmul_tests.rs`) pin every number it depends on.

Upstream fixed this by aligning the partition to `max(group_size, 32)`; that fix
is on MLX `main` and in neither 0.31.2 nor 0.32.0. When a release carrying it is
pinned, `linked_mlx_still_carries_the_misaligned_split_k_partition` fails and
the guard should be deleted rather than carried.

---

## 4. Affine quants — block-affine weight quantization

The affine family covers all `qN_gGS` combinations rMLX supports:
bits ∈ {2, 3, 4, 5, 6, 8} × group_size ∈ {32, 64, 128}. The variants
share a single codec in `affine.rs`; the per-variant names are loader
conventions.

### 4.1 Dequant formula

```
w_fp = scale × code + bias
```

`bias` is additive and already carries the correct sign. This is identical
to `scale × (code - zero_point)` where the AWQ→MLX conversion pre-baked
`bias = -zero_point × scale` into the stored bf16 field.

### 4.2 Packed-code layout

All production MLX affine snapshots use U32Le storage: codes packed
LSB-first into 32-bit little-endian words.

```
bits = N, per_word = floor(32 / N)

Word index for element (row, col):
  word_idx  = row × words_per_row + col / per_word
  shift     = (col % per_word) × N
  mask      = (1 << N) - 1
  code      = (word >> shift) & mask

words_per_row = ceil(cols / per_word)
total bytes   = rows × words_per_row × 4
```

For 3-bit elements `per_word = 10` — the top 2 bits of each u32 word are
unused padding.

A U8 storage variant (codes packed across byte boundaries, LSB-first) is
supported for completeness and round-trip testing but has not been observed
in any production snapshot.

### 4.3 Scale / bias storage

Shape `[rows, cols / group_size]`, bf16 LE, row-major.
Both `scales` and `biases` use the same shape. Each bf16 value is 2 bytes
stored little-endian.

### 4.4 Supported combinations

| Tag | Bits | Group size | Typical use |
|---|---|---|---|
| `q8_g128` | 8 | 128 | K-cache affine, heavy INT8 weight models |
| `q8_g64` | 8 | 64 | higher-quality INT8 |
| `q8_g32` | 8 | 32 | highest-quality INT8 |
| `q6_g64` | 6 | 64 | 6-bit weight models |
| `q5_g64` | 5 | 64 | 5-bit weight models |
| `q4_g128` | 4 | 128 | standard 4-bit (mlx-community default) |
| `q4_g64` | 4 | 64 | higher-quality 4-bit |
| `q4_g32` | 4 | 32 | highest-quality 4-bit |
| `q3_g64` | 3 | 64 | aggressive 3-bit |
| `q2_g64` | 2 | 64 | extreme compression |

The `q8_g128` path has a GPU (Metal) kernel; all other affine variants
dequant on the CPU path during model load.

**Load-time bit-width gate.** `bits` outside `{2,3,4,5,6,8}` (e.g. an
unreleased 1-bit affine checkpoint) has no dequant kernel in either the CPU
codec (`affine.rs::validate_params`) or the linked mlx-c's GPU
`affine_dequantize_*` / `quantized_matmul` kernels. `rmlx_models::arch::loader`
pre-flights the model's declared `quantization.bits` — the global default
*and* every `quantization.tensor_overrides` entry — against
`rmlx_quant::affine::SUPPORTED_BITS` before any tensor I/O — an unsupported
bit-width fails the load immediately with one clear error instead of
"loading" successfully and then dying per-token at first prefill with a
buried Metal kernel-load error. See `docs/ADDING_A_MODEL.md` for how this
gate composes with a new architecture.

### 4.5 Dequant cost

Affine dequant runs once at model load, not per-token. The cost is linear
in `rows × cols` and is dominated by the cache-coherent scan of
`packed_codes`. Scale and bias are loaded once per group (one bf16 decode
per field per group), then amortized across `group_size` elements.

Source: `crates/rmlx-quant/src/affine.rs`
MSL kernel (q8_g128): `crates/rmlx-kv-quant/src/q8_msl.rs`

---

## 5. TurboQuant — Lloyd-Max codebook, **no rotation**

TurboQuant is used in rMLX primarily for the V side of the KV cache
(4-bit) and for weight quantization experiments. It uses a fixed-codebook
scalar quantizer with a per-block scale rather than affine scale + bias.

**The name promises a rotation this implementation does not have.** Upstream
TurboQuant decorrelates before quantizing — that is the point of the family, and
it is what IsoQuant, PlanarQuant and RotorQuant all do. rMLX's turbo encoder
applies no transform on either axis at any width: §5.2 and §5.3 below are the
whole codec, and the source contains no Hadamard or Walsh-Hadamard code. The
`tsym3` / `tsym4` SSD layout tags name the Lloyd-Max codebook the encoder does
apply — `tsym3_lloyd_3_3` / `tsym4_lloyd_4_4`.

What the missing transform would be worth has been measured, by a controlled
test-side ablation around the shipped encoder
(`crates/rmlx-kv-quant/src/turbo_rotation_fidelity_tests.rs`). The short form:
it is worth roughly a bit to two bits on **K**-shaped data with channel
outliers, roughly a hundredth of a bit on **V**-shaped i.i.d. data, and the
payoff shrinks as the codebook widens. Since rMLX uses turbo primarily on the V
axis, and the V-side implementation is the expensive one — it needs an explicit
inverse transform after the SV accumulation, across the P2 kernel, four dequant
kernels and both fused-QK kernels — the value and the cost are inverted with
respect to each other. Full account, including the scope limits:
docs/KV_QUANT.md, "The turbo family's missing rotation — what it is worth, and
where".

### 5.1 Codebook

Lloyd-Max optimal centroids for N(0,1), derived by
`turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0, n_iter=100)`.
Hardcoded as f32 constants; regenerated via `scripts/gen_lloyd_codebook.py`.

```
1-bit (2 entries): [-0.7979, +0.7979]
2-bit (4 entries): [-1.51, -0.453, +0.453, +1.51]
3-bit (8 entries): [-2.152, -1.344, -0.756, -0.245, +0.245, +0.756, +1.344, +2.152]
4-bit (16 entries): [-2.718, -2.052, -1.601, -1.240, -0.928, -0.646, -0.381, -0.126,
                     +0.126, +0.381, +0.646, +0.928, +1.240, +1.601, +2.052, +2.718]
```

### 5.2 Block layout

Group size = 32. Each block:

```
scale = max(|x_i|) / max_centroid    (f32, one per block)
code_i = nearest_centroid(x_i / scale)

Nearest centroid: count midpoint boundary crossings from left.
  Boundaries: (cb[k] + cb[k+1]) / 2  for k in 0..n-2

Packed storage:
  bits per element: 1–4
  bytes per block: ceil(32 × bits / 8)
  packing: LSB-first; elements 0..N in positions 0..N within the byte stream
```

### 5.3 Dequant

```
w = codebook[code] × scale
```

### 5.4 8-bit

8-bit TurboQuant is not supported. Use affine q8_g128 (the `q8_msl.rs`
path) for 8-bit K-cache or weight quantization.

### 5.5 Supported archs

All — TurboQuant is arch-agnostic. Primary production use: V-cache at 4-bit
in the K8V4 asymmetric config; see [KV_CACHE.md](KV_CACHE.md).

Source: `crates/rmlx-kv-quant/src/turboquant.rs`
MSL kernel: `crates/rmlx-kv-quant/src/turboquant_msl.rs`
TurboFlash kernel (K8+V4, split-K FlashAttention): `crates/rmlx-kv-quant/src/turbo_flash_msl.rs`

---

## 6. PlanarQuant — Givens-rotation codebook, per-pair scale

PlanarQuant extends TurboQuant with two improvements:

1. **Rotation**: each pair of elements is first rotated by the Givens rotation
   that minimises reconstruction error, chosen from a 16-entry codebook.
2. **Per-pair scale**: scale is computed after rotation for each 2-element pair,
   not once per 32-element block. This alone reduces quantization error
   relative to TurboQuant on any input; the rotation provides additional
   gain when pairs are correlated.

rMLX is the first working Apple Silicon PlanarQuant implementation (2026-05).
The llama.cpp-turboquant Metal kernels for PlanarQuant fall back to CPU
(upstream issue #7). Reference: `ParaMind2025/isoquant`.

### 6.1 Rotation codebook

16 Givens rotations at angles `θ_k = k × π/16` for k ∈ 0..15.
Stored as `[cos θ, -sin θ, sin θ, cos θ]` (row-major 2×2 matrix).
Covering `[0, π)` is sufficient: `R(θ + π) = -R(θ)` and the sign is
absorbed by the max-abs pair scale.

### 6.2 Block layout

Group size = 32 (16 pairs per block).

```
Per block:
  codes     : ceil(32 × bits / 8) bytes, bit-packed LSB-first
  scales    : 16 × f32 = 64 bytes  (one f32 per pair, not per block)
  rotations : 8 bytes               (4-bit rotation index per pair, 2 per byte, LSB-first)

scales layout: one f32 per pair = total_elems/2 f32 values over the full tensor
rotations layout: pair p in block b → byte = b × 8 + p/2, nibble = p%2 × 4
```

### 6.3 Quantize

For each pair `(a, b)` in a block:

1. Try all 16 rotations. For rotation k:
   - Apply `R_k`: `ya = c×a - s×b`, `yb = s×a + c×b`
   - Compute pair scale: `max(|ya|, |yb|) / max_centroid`
   - Quantize + dequantize both; measure max absolute reconstruction error.
2. Choose k with minimum error.
3. Store: rotation index (4 bits), pair scale (f32), two codebook indices (bits each).

### 6.4 Dequant

```
rot_idx  = (rotation_byte >> shift) & 0xF
(ya, yb) = (codebook[idx_a] × scale, codebook[idx_b] × scale)
(a, b)   = R_{rot_idx}^T × (ya, yb)
         = (c×ya + s×yb, -s×ya + c×yb)
```

Uses the same Lloyd-Max N(0,1) codebook as TurboQuant (§5.1).

### 6.5 Dequant cost vs TurboQuant

PlanarQuant stores `total_elems/2` f32 scale values vs TurboQuant's
`total_elems/32` f32 scales — 16× more scale storage per tensor. The
rotation indices add `total_elems/4` bytes. Against this cost, max
reconstruction error is guaranteed to be ≤ TurboQuant on any input.

### 6.6 Supported archs

All — PlanarQuant is arch-agnostic. Primarily used for the V side of the
KV cache; see [KV_CACHE.md](KV_CACHE.md).

Source: `crates/rmlx-kv-quant/src/planarquant.rs`
MSL kernel: `crates/rmlx-kv-quant/src/planarquant_msl.rs`

---

## 7. ParoQuant — weight rotation, MLX-native INT4

ParoQuant (`z-lab/paroquant`) applies pairwise Givens rotations to the
**input activations** before a standard affine INT4 weight matmul. The
weights themselves are stored as ordinary affine INT4 (`q4_g*`); the
rotation is a pre-matmul step on the activation side.

### 7.1 How it differs from PlanarQuant

| | PlanarQuant | ParoQuant |
|---|---|---|
| What is rotated | Weight tensor elements (pairs) | Input activations (channel pairs) |
| Rotation application | At dequant time | Before each matmul |
| Weight storage | Codebook codes + per-pair scale | Standard affine INT4 |
| Scale grain | One f32 per 2 weight elements | Standard affine (one per group) |

### 7.2 Rotation kernel inputs

The Metal kernel (`paro_rotate_gpu`) takes:

- `x`: `[batch, hidden]` input activations (F16 or BF16).
- `packed_pairs`: `[krot, hidden/2]` I32 — pair indices (i, j) per round.
  Lo 16 bits = i_local, hi 16 bits = j_local within group.
- `cos_theta`: `[krot, hidden/2]` F16 — cosines pre-computed at model load.
- `sin_theta`: `[krot, hidden/2]` F16 — sines pre-computed at model load.
- `channel_scales`: `[hidden]` F16 — per-channel scale factors.
- `krot`: number of rotation rounds (≤ `MAX_KROT = 16`).
- `group_size`: channel group width (≤ `MAX_GROUP_SIZE = 256`).

### 7.3 Rotation algorithm

For each group of `group_size` channels and each `ROWS_PER_TILE` activation rows:

```
for round in 0..krot:
    (i, j) = packed_pairs[round]
    (a', b') = (a × cos + b × sin, b × cos - a × sin)
    x[i] = a'
    x[j] = b'
```

After `krot` rounds the rotated activations are written back and fed into
the standard affine INT4 dequant + matmul.

### 7.4 Supported archs

ParoQuant checkpoints are produced by `z-lab/paroquant`. The `z-lab/`
Open Models snapshots (e.g. `z-lab__Qwen3.6-27B-PARO`) use this format.

Dispatcher: `crates/rmlx-models/src/paroquant_msl.rs`
MSL body: `crates/rmlx-models/src/metal/paroquant_rotate.metal`
External reference: `z-lab/paroquant/paroquant/kernels/metal/rotation.metal`

`ROWS_PER_TILE` (1 for a decode step, 4 for prefill / batch), `MAX_KROT` and
`MAX_GROUP_SIZE` are MLX template ints supplied at dispatch, so one registration
covers both variants and the bounds have a single source of truth in the Rust
consts the validation checks use.

---

## 8. Storage-size reference

Bytes on disk per `rows × cols` weight matrix:

| Format | Bytes per weight element | Extra metadata |
|---|---|---|
| bf16 | 2.00 | — |
| mxfp8 | 1.00 | 1 B / 32 elements (E8M0 scale) |
| mxfp4 | 0.50 | 1 B / 32 elements |
| nvfp4 | 0.50 | 1 B / 16 elements |
| q8_g128 | 1.00 | 4 B / 128 elements (scale + bias, each bf16) |
| q4_g128 | 0.50 | 4 B / 128 elements |
| q4_g64 | 0.50 | 4 B / 64 elements |
| q4_g32 | 0.50 | 4 B / 32 elements |
| q3_g64 | 0.375 | 4 B / 64 elements |
| q2_g64 | 0.25 | 4 B / 64 elements |
| turboquant 4-bit | 0.50 | 4 B / 32 elements (f32 scale) |
| planarquant 4-bit | 0.50 | 4 B / 2 elements (f32 per-pair scale) + 0.5 B / 2 elements (rotation) |

---

## 9. Dequant path summary

```
Model load (once):
  mxfp8    → e4m3_decode(elem) × e8m0_decode(scale)     [CPU, per element]
  mxfp4    → e2m1_decode(nibble) × e8m0_decode(scale)   [CPU, per nibble]
  nvfp4    → e2m1_decode(nibble) × ue4m3_decode(scale)  [CPU, per nibble]
  affine   → scale × code + bias                         [CPU or GPU (q8_g128)]
  bf16     → passthrough (no conversion)

Per-token inference:
  TurboQuant V  → codebook[idx] × scale                 [Metal, group_size=32]
  PlanarQuant V → codebook[idx] × scale → R_k^T         [Metal, group_size=32]
  ParoQuant     → R_{krot}(x) before affine INT4 matmul [Metal, activation-side]
  TurboFlash    → fused K8+V4 inside split-K FlashAttn  [Metal, K group_size=128, V group_size=32]
```

---

## 9. Ternary / BitLinear (`BitNetForCausalLM`)

BitNet b1.58 stores each linear weight as a ternary tensor: values ∈ `{-1, 0, +1}`.
Four ternary values are packed into one U8 byte using 2 bits per trit (LSB first).

```
Byte layout (one byte → 4 trits):
  bits [1:0] → trit 0
  bits [3:2] → trit 1
  bits [5:4] → trit 2
  bits [7:6] → trit 3

Encoding:
  raw 0 → ternary  0  → 0.0
  raw 1 → ternary +1  → +weight_scale
  raw 2 → ternary -1  → -weight_scale
  raw 3 → ternary -1  → -weight_scale  (alias; treated same as 2)
```

Storage shape: U8 `[N//4, K]` → logical BF16 `[N, K]` after unpacking.

Each weight tensor has a sibling `*.weight_scale` tensor: a single BF16 scalar `[1]`
that gives the absolute magnitude of the non-zero trits. rMLX multiplies the scale
in at load time — the resulting BF16 matrix carries the scale baked in.

Effective bits per parameter: ~1.58 (log2(3)). Storage overhead at fp32 → ternary
is ~20× compression versus bf16.

#### Dequantization algorithm

```rust
for r in 0..packed_rows {
    for c in 0..cols {
        let byte = u8_bytes[r * cols + c];
        for t in 0..4 {
            let raw = (byte >> (t * 2)) & 0x3;
            let val = match raw { 0 => 0.0, 1 => +scale, _ => -scale };
            out_bf16[(r * 4 + t) * cols + c] = bf16(val);
        }
    }
}
```

#### Cost and inference path

Dequant runs once at load time (measured ~6 090 ms for the 2B model, 30 layers,
on Apple Silicon M-series). After dequant, inference uses plain BF16 matmul with
no special Metal kernel.

Source: `crates/rmlx-models/src/bitnet/loader.rs` — `dequant_trit_u8()`.

---

## 10. Adding a new weight-quant format

1. Implement `dequant_to_f32` in `crates/rmlx-quant/src/<format>.rs`.
2. Add a variant to the quant tag enum in the loader.
3. Add a smoke probe (`rmlx serve`, check for `!!!!!!`-style corruption).
4. Register the format in `rmlx info --list-weight-quants`.
5. Update this file and [KV_CACHE.md](KV_CACHE.md) if it applies to KV cache as well.

---

## See also

- [`KV_QUANT.md`](KV_QUANT.md) — KV cache quantization variants (parallel axis).
- [`KV_CACHE.md`](KV_CACHE.md) — KV cache architecture and storage layout.
- [`MODELS.md`](MODELS.md) — per-architecture weight quant defaults and support.
- [`FFI.md`](FFI.md) — mlx-c FFI surface for the Metal kernel dispatch.
