# KV codec fidelity

This doc gives the measured fidelity of the KV codecs: incoherence, the turbo
family's missing rotation, and rate-distortion.

The other KV quantization docs: [`KV_QUANT.md`](KV_QUANT.md) (the contract:
API, CLI flags, the auto default, bit rates, codec disposition);
[`KV_LAYER_POLICY.md`](KV_LAYER_POLICY.md) (which codec each layer gets);
[`KV_CODECS.md`](KV_CODECS.md) (storage per `KvStorage` variant, TurboQuant
calibration); [`KV_ROTATION_CODECS.md`](KV_ROTATION_CODECS.md) (the iso and
rotor codecs); [`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) (fused-QK, fused
flash-decode, sparse attention);
[`KV_STORE_TRUNCATION.md`](KV_STORE_TRUNCATION.md) (`truncate_to` per store).

---

## Codec fidelity — measured

Two CPU-only measurement surfaces in `rmlx-kv-quant`, both deterministic from
`TEST_SEED`, both inside `make model-check`. Neither needs a model snapshot or
the GPU. See `docs/TESTING.md` for how to run them and for the helper list.

### Incoherence — does the rotation do anything

`crates/rmlx-kv-quant/src/rotation_fidelity_tests.rs`.

The incoherence of a row is `mu = sqrt(d)·max|x_i|/||x||_2`. It is 1 for a
flat row and `sqrt(d)` for a one-hot row.

The per-codec cosine gates run on the i.i.d.-uniform LCG fixture. For i.i.d.
data `mu` is near `max|x_i|` in sigma units. At `head_dim = 128` that is
about 1.72 for uniform on [-1, 1] (`max|x_i|` near 1, sigma `1/sqrt(3)`) and
about 2.8 for Gaussian. A
Hadamard pushes uniform toward Gaussian and so raises `mu`. **The LCG cosine
gates therefore carry no information about rotation quality: an identity
rotation passes every one of them.**

The outlier fixture is i.i.d. Gaussian with 4 of 128 channels scaled 20x. It
models the per-channel Key outliers reported by KIVI (arXiv:2402.02750) and
KVQuant (arXiv:2401.18079). The ratio is the magnitude reported for emergent
outlier features (arXiv:2208.07339).
`outlier_fixture_is_adversarial_and_iid_fixtures_are_not`
(`test_utils_tests.rs`) asserts a mean `mu` of at least 5.0 for it. The
i.i.d. fixtures stay under 5.0.

The channel **count** is not from the literature. 4 of 128 is denser than the
reported outlier fraction. It puts an outlier in every affine group of 64,
where `rot_k` sets its scale. Tests sweep both fixture parameters. `mu` rises
monotonically with the ratio. Against the channel count it rises, peaks, then
returns exactly to the i.i.d. value once every channel is scaled.

A block-`b` orthogonal transform can reduce `mu` by at most `sqrt(b)`. The
peak coordinate's block keeps its L2 norm, and a `b`-vector's maximum is at
least its norm over `sqrt(b)`. On the outlier fixture at `head_dim = 128`:

| Family | Transform | Block | `mu` ceiling | Gate |
|---|---|---|---:|---:|
| `rot_k` / `RotK` | Walsh-Hadamard, full `head_dim` | 128 | 11.31x | ≥ 3.0x |
| `iso3` / `iso4` | isoclinic SO(4), fixed quaternion | 4 | 2.00x | pinned 1.3846x |
| `planar3` | Givens, 16-entry codebook, per-pair search | 2 | 1.41x | pinned 1.1910x |
| `planar4` | Givens, 16-entry codebook, per-pair search | 2 | 1.41x | pinned 1.1518x |
| `rotor3` / `rotor4` | Cl(3,0) rotor sandwich, static per (layer, head) | 3 | 1.73x | pinned 1.0815x |

A pinned family must stay under its ceiling and above its pin minus 0.05. The
`rot_k` gate at 3.0x needs an effective block of at least 9. No block-local
family and no block-4 truncation of the Hadamard can reach it.

The rotor pin is the weakest of eight `(layer, head)` rotor tables. Only the
groups that hold an outlier channel move `mu`, four of 43, so one table is a
four-sample estimate.

Only `rot_k` applies a full-dimension transform. The block-local families stay
well under their ceilings. Their rotations are fixed (iso, rotor) or fitted to
reconstruction error rather than to incoherence (planar). **This is not a
defect in them**: they buy packing efficiency, a different axis. The iso
quaternion keeps `phi/sqrt(5) = 0.72` of a lone large coordinate, hence about
1.38x.

`rot_k_hadamard_buys_bits_on_outlier_data_and_costs_them_on_iid_data`
compares `rot_k` with the same `affine q8 group=64` quantizer without the
Hadamard. It asserts at least 1.5 bits of SQNR gained on the outlier fixture.
It also asserts a loss on the i.i.d. uniform fixture.

The gain is exactly `log2(peak_plain / peak_rotated)` over the affine group.
The block ceiling bounds the same quantity: a block-`b` transform can buy at
most `0.5·log2(b)` bits. So the 1.5-bit gate demands an effective block of 8
or more. `non_full_dimension_rotations_fail_the_rot_k_gain_gate` checks that
the block-4 Hadamard and the iso quaternion stay under it.

### The turbo family's missing rotation — what it is worth, and where

TurboQuant is named for a rotation this tree does not apply. The shipped codec
quantizes raw KV against a Lloyd-Max codebook with no decorrelating transform,
at any width, on either axis. `crates/rmlx-kv-quant/src/turboquant.rs` has no
Hadamard code. The layout tags `TURBOSYM3_LAYOUT_TAG` / `TURBOSYM4_LAYOUT_TAG`
are `tsym3_lloyd_3_3` / `tsym4_lloyd_4_4` and name the codebook. SSD hydrate
dispatches on exact tag equality, so a tag is a stored format. Changing one
needs a `SCHEMA_VERSION` bump.

`crates/rmlx-kv-quant/src/turbo_rotation_fidelity_tests.rs` measures what the
transform would be worth. It holds the codec, the width and the group size
fixed. It moves only a full-`head_dim` normalized FWHT in and out around the
shipped CPU encoder.

What its gates assert, at every width the codebook accepts (1 to 4 bits):

- **K-shaped data** (outlier fixture): the full Hadamard beats a block-4
  Hadamard, which beats no transform. The gain shrinks as the codebook widens.
  It clears the 1.5-bit `ROT_K_MIN_OUTLIER_GAIN_BITS` threshold at 1, 2 and 3
  bits, and misses it at 4 bits.
- **V-shaped data** (i.i.d. Gaussian): the gain stays under 0.1 bits. An
  isotropic Gaussian is rotation-invariant, so there is nothing to recover.
- **i.i.d. uniform**: the transform loses bits.
- Against `iso` at the same width, the missing rotation is more than half of
  turbo's cosine gap on the outlier fixture. Scale cadence is the smaller term.

Turbo is primarily a **V** codec. So a rotation would pay on the smaller half
of the family, and at the narrow widths. The V side would also need an inverse
transform after the SV accumulation in the flash and dequant kernels.

These gates run on a **model** of K-cache structure, not on a K/V tensor
captured in a forward pass. For a real checkpoint they are an estimate.

### Rate-distortion — is the bit width delivering

`crates/rmlx-kv-quant/src/rate_distortion_tests.rs`.

The anchor is the fixed-rate Lloyd-Max SQNR for the standard normal (Max 1960,
Table I): 4.396 / 9.300 / 14.616 / 20.224 dB at 1–4 bits. It is **not** the
rate-distortion bound (`6.02·b` dB). It assumes a quantizer matched to the
source that spends no rate on its scale. Every codec here stores a per-group
scale, so it can land above the anchor. The stored rate of each codec is under
`docs/KV_QUANT.md` §"Memory and bit-rate summary".

Wasted bits are `(anchor − measured) / 6.02`; negative is ahead of the anchor.
On the i.i.d. Gaussian fixture, 256 x 128, each cell is pinned:

| Codec | bits | anchor | wasted bits (pinned) |
|---|---:|---:|---:|
| turbo | 2 | 9.300 dB | +0.34 |
| turbo | 3 | 14.616 dB | −0.06 |
| turbo | 4 | 20.224 dB | −0.23 |
| tcq | 2 | 9.300 dB | +0.34 |
| tcq | 3 | 14.616 dB | −0.06 |
| planar | 3 | 14.616 dB | −4.32 |
| planar | 4 | 20.224 dB | −2.74 |
| iso | 3 | 14.616 dB | −0.78 |
| iso | 4 | 20.224 dB | −0.86 |
| rotor | 3 | 14.616 dB | −0.97 |
| rotor | 4 | 20.224 dB | −1.05 |

Each cell's budget in `CELLS` is its pin plus 0.10 bits.
`scalar_codebook_rate_distortion_report` fails a cell past its budget, and
`pinned_budgets_sit_one_slack_above_the_measurement` fails a budget more than
0.105 bits above the measurement.

**The small-group scale is not a loss.** iso, rotor and planar derive the
scale from the maximum of 3 or 4 samples. Yet these small groups come out
*ahead* of the anchor. The group maximum is
a strong conditioning statistic and is reconstructed near-exactly. The two or
three remaining elements are then known to be smaller than it. The shortfall
is at the other end: large groups at low bit widths. `turbo` and `tcq` at 2
bits spend one f32 scale per 32 values and lose 0.34 bits.

**Planar's 3-bit and 4-bit widths cost byte-identical storage.** A 32-value
planar block takes 4 words at either width, under the `32 / bits`
vals-per-word convention. A width is dominated when it costs the same bytes
as the other width and loses quality. Each family therefore has one strictly
dominated width only if its two widths cost the same bytes. Planar is the only
such family: `planar4` loses about 3.9 dB against `planar3`.

Per pair, planar pins the larger element to the outermost centroid. The smaller
lands on the grid `centroid / max_centroid`. Its outermost gap is
`(2.152 − 1.344)/2.152 = 0.375` at 3 bits and `(2.718 − 2.052)/2.718 = 0.245`
at 4 bits. That is only 1.5x finer, while the 16-angle Givens search that must
land *both* elements on centroids gets no larger. The extra bit does not pay
for itself.

iso and rotor pack their codes into the dense code plane, where a code costs
`bits`. Their 4-bit width costs one more bit per stored code and wins on
quality. `planar_widths_are_byte_identical_and_the_others_pay_for_their_bits`
pins all three families.

**TCQ's claw-back measures 0.000 dB.** See the trellis degeneracy note under
`docs/KV_CODECS.md` §"`KvStorage::K8VTurbo3Tcq` — q8_0 K, TurboQuant 3-bit V with Viterbi trellis".
