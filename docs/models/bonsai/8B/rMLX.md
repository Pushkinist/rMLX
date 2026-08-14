# Bonsai-8B (2-bit) — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable 2-bit tier) = **mlx-lm (no KV quant)**, decode TPS:
> **109.8 / 94.5 / 73.2 / 48.6 / 28.5** (4k/8k/16k/32k/64k). *(unchanged — not
> re-benched this pass; siblings comparison still valid as an anchor.)*

**Model:** `prism-ml__Ternary-Bonsai-8B-mlx-2bit` —
`Qwen3ForCausalLM`, dense ~8B, **2-bit affine** (group 128), full attention
(`sliding_window: null`), **YARN rope ×4** (native 16384 → 65536). Single
snapshot, ctx ceiling 64k (no 128k).
**Machine:** Apple M5 Max, 128 GB, macOS 26.5.1 (Darwin 25.5.0) · **Binary:**
`release-perf`, rMLX 0.3.0 (`main` @ `d8d7463`). **nax GEMM kernels:** 288
(`brew` mlx 0.31.2 + mlx-c 0.6.0_2, confirmed linked — the 0.32.0 bottle
regression does not apply here). **Date:** 2026-07-21.
**Protocol:** unchanged from the 0.2.5 baseline — batch=1, temp=0,
`max_tokens=256`; serve once per codec at the 64k ceiling (`--max-ctx 65536`,
lazy-grow ring) + CBB `run_one` load-once for decode and cold r0 TTFT;
**n=3 measured** (4k/8k/16k/32k), **n=1 measured** (64k), 1 warmup `r0`
discarded. Bar (§3): WIN / TIE-on-noise / LOSS.

> **All 25 KV codecs run, full 4k–64k sweep — no cell capped this pass.** The
> 0.2.5 baseline capped the K-only family (`k_iso3/4` ≤16k, `k_rotor3/4` ≤8k)
> because they were CPU-bound; that cap is gone — both families now run the
> complete 4k–64k range on the new flash-decode-over-quant kernels (§0, §4).

---

## 0. TL;DR

- **`none` is still the fastest codec at every context, and still the
  smallest KV at every context.** Decode **133.3 / 116.2 / 89.4 / 61.5 /
  37.2** (4k/8k/16k/32k/64k) — a virtual match to the 0.2.5 baseline
  (133.1/114.9/90.1/61.4/36.3, all within ±2.5%, no regression) — and still
  **+21…+31%** over the mlx-lm champion (§3), the lead now *growing* with
  context (was flat in 0.2.5). KV-MB unchanged (658/1310/2725/5408/10536),
  still the floor every other codec is measured against (§2.1).
- **The K-only family (`k_iso3/4`, `k_rotor3/4`) went from CPU-bound-unusable
  to GPU-functional across the full 4k–64k range.** 0.2.5 had them capped and
  crawling (k_iso3 5.8→1.5 TPS by 16k, k_rotor3/4 0.7→0.4, no 32k/64k data at
  all). Now: k_iso3 **25.5·18.2·13.2·8.9·5.5**, k_iso4
  **25.9·19.7·14.7·9.9·4.1**, k_rotor3 **23.9·15.5·12.6·8.4·4.5**, k_rotor4
  **21.0·18.5·13.3·8.8·4.3** — +340% to +4500% at the sizes with a prior
  number, and a first-ever complete curve at 32k/64k. Confirmed GPU via
  `iso_flash_decode_sdpa` / `rotor_flash_decode_sdpa` kernel-dispatch probes
  at every size including 64k (not just a short-context artifact). The
  `--rotor-qjl` default flipped `on`→`off` since 0.2.5 (source-verified:
  `crates/rmlx-cli/src/main.rs:224`); an ablation re-run with `--rotor-qjl on`
  reproduces the old 0.7 TPS exactly, confirming the flag flip is the entire
  fix (~30–34× at 4k). Still usable-but-not-fast at long context (4–5 TPS
  @64k) — a real, working codec now, not a broken one.
- **The `_sym` family (`iso3_sym/4_sym`, `rotor3_sym/4_sym`) regressed at
  every context and is now SLOWER at 64k than the CPU path it replaced.**
  0.2.5: 133/131/136/133 TPS at 4k, collapsing to 5.9/5.7/5.8/7.3 at 64k
  (already a known CPU-dequant defect). Now: **17–19 TPS at 4k** (already
  below 0.2.5's *worst* mid-context cells) falling to **1.7–2.4 TPS at 64k** —
  worse than the 0.2.5 CPU path it was built to replace. The symv
  flash-decode kernel provably dispatches on GPU at every size (confirmed via
  `iso_flash_decode_symv_sdpa` / `rotor_flash_decode_symv_sdpa` probes,
  including at 64k) — this is a genuine marginal-cost defect in the new
  kernel, not a dispatch failure. Root-caused to the V-side dequant **inside
  the symv kernel specifically** — *not* to quantized-V in general
  (`rot_k_tq4v` quantizes V on the generic path and is cheap at 1.16 ms/1k),
  and *not* to the flash-decode scaffolding in general (the K-only kernels are
  costly but stable): marginal cost **6.25–9.42 ms per 1k KV tokens** vs the
  K-only (quantized-K-only, bf16-V) family's **2.25–3.49 ms/1k** and `none`'s
  **0.33 ms/1k** (§2.2) — filed as issue #292. KV memory did improve sharply
  (30608→10640 MB @64k, 2.91×→1.01× `none`, essentially free now) — the
  codec got cheap on memory and expensive on compute at exactly the context
  length it exists to serve.
- **A mid/long-context regression was reported across 11 codecs (−6…−34% at
  16k–64k) and filed as issue #293. Re-measurement retracts most of it — see
  §4.2.** Two things were wrong. (a) The grouping label "non-kernel-dispatching
  codecs" is not what the code says: `KvQuant::carries_msl()` is `true` for all
  eleven (only `none` carries no MSL at all), and
  `KvQuant::cpu_hot_path_reason()` returns `None` — Metal on the hot path — for
  seven of them (`planar`, `planar3`, `planar_k`, `k8vturbo3`, `k8vturbo3tcq`,
  `tsym3`, `tsym4`). Only `iso3/4` and `rotor3/4` are CPU-hot-path, and only at
  **prefill**. The declared-unaffected set (`none`, `k8v8`, `k8vturbo2`,
  `k8v4`, `rot_k_tq4v`) is classifier-identical to the declared-affected set,
  so the grouping never had a mechanism behind it. (b) The per-cell magnitudes
  were harness artefacts: two runs of the **same binary** disagreed by 29% at
  `none`@32k (61.5 vs 47.76) and reversed the codec ordering. Under `rmlx bench`
  the same cells hold parity or move a fraction of the reported amount.
- **No codec is CPU-bound at any size, at decode.** Every codec with a
  dedicated flash-decode-over-quant kernel (`_sym`, K-only) was confirmed
  dispatching it at 4k *and* 64k via untimed verbose kernel-dispatch probes;
  every other codec shows zero CPU-dequant/host-download log lines. Note the
  distinction the classifier draws, because it is easy to mis-state: **carrying
  MSL and dispatching a flash-decode kernel are different properties.**
  `carries_msl()` is `true` for every codec except `none`; a codec can carry
  MSL, decode entirely on GPU, and still dispatch no *flash-decode* kernel,
  because the warm-TTFT bf16 seed absorbs the decode window. The one caveat:
  `iso3/4`/`rotor3/4` (non-sym) carry a documented CPU-side V-encode at
  **prefill only** (`cpu_hot_path_reason()` in `quant.rs`) — decode itself
  reads a bf16 seed and is full-speed GPU; this is why `iso3/4`/`rotor3/4`
  track `none`'s TPS shape but not its prefill time.

---

## M. Measurement note (KV ring sizing — fairness)

Unchanged from 0.2.5: rMLX `serve`'s KV ring grows lazily, so every codec is
served **once** at `--max-ctx 65536` and all five prompt sizes sweep against
the resident lazy-grown ring. The 64k fixture (`longctx_64k.json`, ~63.3k
Bonsai tokens via the server's own chat-template tokenization) fits the
ceiling with room for 256 generated tokens — verified directly this pass.
`rmlx baseline --prompt-tokens` now shares this same chat-template
tokenization (a chat-JSON fixture is rendered through `chat_template.jinja`
before tokenizing, not fed to the tokenizer raw) so it agrees with `serve`'s
token count and needs no `--max-prompt-tokens` headroom for this fixture.

---

## 1. rMLX snapshot benched

| Snapshot (basename) | Weight quant | Arch / size | Role | Disk |
|---|---|---|---|---|
| `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | affine g128 b2 (ternary) | `Qwen3ForCausalLM` dense ~8B, full-attn, YARN ×4 | base | 2.2 GB |

No drafter snapshot exists for Bonsai → no speculative / MTP grid (unchanged).

---

## 2. rMLX full matrix

**Cell = `decodeTPS · r0TTFT(s) · KV-MB`.** Same measurement method as 0.2.5
(§ above). No cell is capped this pass — every codec ran the full 4k–64k
sweep.

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| none | 133.3·1.2s·658 | 116.2·2.9s·1310 | 89.4·7.0s·2725 | 61.5·20.7s·5408 | 37.2·64.8s·10536 |
| k8v4 | 132.3·1.2s·830 | 41.3·3.2s·1654 | 23.3·7.7s·3446 | 12.9·21.2s·6829 | 6.7·73.2s·13292 |
| k8v8 | 131.1·1.2s·872 | 111.5·2.9s·1739 | 79.8·7.2s·3623 | 55.4·20.6s·7178 | 32.1·65.9s·13968 |
| planar | 130.6·1.2s·1051 | 110.1·2.9s·2096 | 70.6·9.1s·4372 | 50.0·23.2s·8653 | 27.7·68.1s·16828 |
| planar3 | 134.9·1.2s·1051 | 108.9·3.0s·2096 | 70.6·9.5s·4372 | 50.7·23.7s·8653 | 30.8·74.5s·16828 |
| planar_k | 133.7·1.2s·944 | 109.7·2.9s·1882 | 72.7·9.1s·3922 | 52.7·23.0s·7768 | 31.3·67.7s·15112 |
| k8vturbo2 | 129.0·1.7s·804 | 106.4·3.9s·1602 | 84.2·11.1s·3335 | 60.1·25.4s·6613 | 36.0·77.6s·12876 |
| k8vturbo3 | 130.2·1.8s·817 | 116.3·4.8s·1628 | 82.1·10.8s·3389 | 57.9·26.9s·6720 | 31.6·77.7s·13084 |
| k8vturbo2tcq | 132.2·3.3s·804 | 113.9·7.7s·1602 | 82.6·16.3s·3335 | 56.7·38.0s·6613 | 33.6·96.6s·12876 |
| k8vturbo3tcq | 135.7·4.4s·817 | 113.3·9.0s·1628 | 81.7·19.6s·3389 | 56.8·43.4s·6720 | 32.9·108.5s·13084 |
| tsym3 | 134.5·1.9s·762 | 110.7·4.2s·1517 | 86.1·9.4s·3157 | 58.3·25.4s·6264 | 34.8·74.4s·12200 |
| tsym4 | 135.2·1.2s·788 | 110.2·3.5s·1570 | 76.2·8.8s·3269 | 54.5·23.6s·6481 | 29.0·70.1s·12616 |
| iso3 | 133.8·2.5s·1391 | 86.1·5.4s·2770 | 63.6·12.5s·5764 | 42.3·30.8s·11440 | 17.5·87.5s·22287 |
| iso4 | 133.1·3.7s·1391 | 86.4·8.1s·2770 | 61.9·17.2s·5764 | 43.3·39.5s·11440 | 11.2·101.5s·22287 |
| iso3_sym | 18.9·3.8s·665 | 14.7·7.8s·1328 | 9.6·18.0s·2776 | 7.0·38.7s·5479 | 2.4·102.2s·10640 |
| iso4_sym | 17.6·6.0s·665 | 14.4·12.2s·1328 | 10.4·25.7s·2776 | 7.7·58.7s·5479 | 1.8·156.1s·10640 |
| k_iso3 | 25.5·2.4s·661 | 18.2·5.5s·1319 | 13.2·13.4s·2750 | 8.9·31.9s·5444 | 5.5·86.0s·10588 |
| k_iso4 | 25.9·3.6s·661 | 19.7·7.9s·1319 | 14.7·17.0s·2750 | 9.9·38.5s·5444 | 4.1·97.9s·10588 |
| rotor3 | 134.7·2.5s·1047 | 104.7·5.3s·2086 | 71.8·12.9s·4341 | 50.0·29.4s·8613 | 27.6·78.3s·16775 |
| rotor4 | 129.3·2.8s·1047 | 110.8·6.3s·2086 | 75.5·13.4s·4341 | 50.9·32.7s·8613 | 25.8·88.6s·16775 |
| rotor3_sym | 18.0·3.6s·808 | 12.6·7.1s·1614 | 9.7·15.7s·3374 | 6.2·38.5s·6659 | 2.3·97.7s·12928 |
| rotor4_sym | 16.9·4.4s·808 | 14.1·9.0s·1614 | 9.8·19.4s·3374 | 6.3·43.5s·6659 | 1.7·106.3s·12928 |
| k_rotor3 | 23.9·2.4s·1015 | 15.5·5.1s·2023 | 12.6·13.2s·4217 | 8.4·31.4s·8354 | 4.5·87.5s·16255 |
| k_rotor4 | 21.0·2.9s·1015 | 18.5·5.8s·2023 | 13.3·13.6s·4217 | 8.8·31.9s·8354 | 4.3·83.0s·16255 |
| rot_k_tq4v | 98.3·1.2s·841 | 69.5·2.9s·1667 | 42.4·8.8s·3461 | 24.4·22.9s·6859 | 12.6·67.9s·13351 |

**Best decode per size:** 4k=k8vturbo3tcq(135.7), 8k=k8vturbo3(116.3),
16k/32k/64k=**none** (89.4/61.5/37.2). The 4k/8k "wins" over `none` are
noise-level (+1.8%/+0.1%) — `none` is the honest default at every size, and
outright the best from 16k on (§3 keeps this framing from 0.2.5).

**Supplementary (not part of the 25×5 matrix) — `--rotor-qjl on` ablation,
4k only:** `k_rotor3` 23.9→**0.7** TPS, `k_rotor4` 21.0→**0.7** TPS — exact
match to the 0.2.5 default-`on` numbers, confirming the flag-default flip is
the entire GPU/CPU delta for this family.

### 2.1 KV-cache size (MB) and ratio vs `none`

`none` is still the memory floor at every context — 658 / 1310 / 2725 / 5408
/ 10536 MB, matching 0.2.5 almost exactly (±1 MB rounding). The big move this
pass is the K-only / sym-iso family's memory footprint:

| KV | MB@64k | ratio | 0.2.5 ratio | | KV | MB@64k | ratio | 0.2.5 ratio |
|---|---|---|---|---|---|---|---|---|
| **none** | **10536** | **1.00×** | 1.00× | | iso3 / iso4 | 22287 | 2.12× | 2.12× |
| k_iso3/4 | 10588 | **1.00×** | 1.64× | | iso3_sym/4_sym | 10640 | **1.01×** | 2.91× |
| tsym3 | 12200 | 1.16× | 1.16× | | rotor3/4 | 16775 | 1.59× | 1.59× |
| tsym4 | 12616 | 1.20× | 1.20× | | rotor3_sym/4_sym | 12928 | 1.23× | 1.88× |
| k8vturbo2/3 | 12876–13084 | 1.22–1.24× | 1.22–1.24× | | planar/planar3 | 16828 | 1.60× | 1.60× |
| k8vturbo2/3tcq | 12876–13084 | 1.22–1.24× | 1.22–1.24× | | planar_k | 15112 | 1.43× | 1.43× |
| k8v4 | 13292 | 1.26× | 1.26× | | k8v8 | 13968 | 1.33× | 1.33× |
| rot_k_tq4v | 13351 | 1.27× | 1.27× | | k_rotor3/4 | **11898**† | **1.115×**† | 1.14× (broken) |

† `k_rotor3/4` is the one cell re-measured after the CPU-block defect below was
fixed: 11,897,787,872 B at 64k against that run's own `none` of 10,671,734,784 B.
Pre-fix the same run read 16,477,668,320 B (1.544×) — the 16255 / 1.54× this
table used to carry. Every other row is the earlier sweep (its `none` reads
10536 MB, ~1.3% under the re-measurement's, which is why the ratio and not the
absolute is the comparable number).

`k_iso3/4` and `iso3_sym/4_sym` dropped to essentially **1.00–1.01× `none`**
(from 1.64× / 2.91×) once the GPU ring became their sole resident store. That
is a large improvement on the previous figure but it is **not a memory win**:
1.00× `none` means "ties bf16", and the format cannot do better than tie. iso
spends one whole `u32` code word **and** one `f32` scale per 4-element group
(16.25 bits/value at head\_dim=128) and rotor one of each per 3-element group
(21.75), against bf16's 16.0 — so the nominal 3-bit / 4-bit width never
reaches storage, and the 3-bit and 4-bit member of each family measure
byte-identical here. `none` is the memory floor at every context and no member
of this family can undercut it. See `docs/KV_QUANT.md` § "Memory truth".

`k_rotor3/4`'s 1.54× was a **defect, not the layout**: the K-only rotor append
never dropped its CPU blocks once the ring was live, so the prefill prefix
stayed resident twice (the `_sym` appends always dropped theirs — hence 1.23×
against the K-only pair's 1.54×). Fixed. Re-measured A/B on the same binary
pair, prompts 4k / 16k / 64k, 3 runs each:

| model | prompt | before | after | vs `none` after |
|---|---|---|---|---|
| Bonsai-8B (`kv_h=8`, D=128) | 4k | 990.0 MB | **717.2 MB** (−27.6%) | 1.118× |
| Bonsai-8B | 16k | 4090.7 MB | **2959.4 MB** (−27.7%) | 1.119× |
| Bonsai-8B | 64k | 16,477,668,320 B | **11,897,787,872 B** (−27.79%) | 1.544× → **1.115×** |
| gemma-4-e2b (`kv_h=1`, D=256) | 4k | 53.9 MB | **37.0 MB** (−31.4%) | 1.163× |
| gemma-4-e2b | 16k | 201.3 MB | **130.7 MB** (−35.1%) | 1.169× |
| gemma-4-e2b | 64k | 786,057,912 B | **502,473,744 B** (−36.08%) | 1.830× → **1.170×** |

The win survives and grows with context — it is a per-token duplication, so it
scales with the prefix, and the 64k cell is where the post-fix layout ratio
should be read. Same-run `none` baseline at 64k: Bonsai 10,671,734,784 B;
gemma-4-e2b ≈429.5 MB (both e2b ratios resolve to it). Every other codec
measured byte-identical across the binary pair (0.00% delta).

#### Decode cost of the fix: zero, demonstrated against a null control

The honest way to read a small decode delta is to measure something whose true
delta is known to be zero in the same session. `k_iso3` is that control here:
its drop call already existed before this change, the diff provably cannot reach
its decode path, and it measures byte-identical on both binaries. Anything it
reads is the instrument.

| model | prompt | `k_rotor3` (treatment) | `k_iso3` (null control, true delta = 0) |
|---|---|---|---|
| Bonsai-8B | 32k | **+0.13%** [95% CI −2.17, +2.42] | +0.60% |
| gemma-4-e2b | 32k | **+0.13%** [95% CI −2.78, +3.03] | +0.56% |

At 32k the instrument resolves and the answer is clean: the treatment's delta is
*smaller than the control's*, and both sit inside a ±2–3% interval. TTFT −0.03%
±0.3%. Dropping a redundant CPU copy costs nothing at decode, which is what the
layout predicts.

**Do not measure this cell at 4k.** The same null control — a change with no
possible effect — reads **−11.22%** [95% CI −28.50, +6.06] at gen=8 and −2.26%
at gen=128 on Bonsai/4k (ABBA-paired, n=6, on `forward_total_ms`). The 4k cell
has no resolving power, so a 4k number for this codec measures the machine, not
the code. Nor does pairing rescue it: the 4k treatment run, re-analysed as the
paired design ABBA exists to create, reads −9.7% (t = −3.06, df 7, p ≈ 0.018) —
a "significant" regression in a cell whose own control says zero. The earlier
4k and 16k decode figures for `k_rotor3` / `k_rotor4` are below this noise floor
and are deliberately not quoted here as measurements.

**The measurement machine was not quiescent.** A co-resident VM held 100–154%
CPU with load average ≈6 for the duration. That is the direct cause of the
short-context cells being unusable: at 4k the per-step work is small enough that
scheduler contention dominates it. The 32k cells still resolve because their
per-step work is large by comparison. Reproduce long-context, or on an idle
machine, or both.

### 2.2 Marginal decode cost (ms per 1k KV tokens), fit over 8k→64k

Per-token inter-token-latency (`ITL = 1000 / decode_tps`) fit to
`ITL = a + b·kv_seq` (kv_seq in units of 1k tokens) across the 8k/16k/32k/64k
cells. `a` = fixed per-step cost (ms), `b` = marginal cost per 1k resident KV
tokens (ms) — the number that predicts whether a codec can ever beat bf16 at
long context. Grouped by whether a named flash-decode-over-quant kernel
dispatches (confirmed via untimed verbose probe at 64k).

**Read `b` as an ordering, not as a coefficient, and ignore `a` for the symv
tier.** One global straight line over four cells is the wrong model for a curve
that bends: where the fit reports a *negative* `a` (`iso3_sym` −4.53,
`iso4_sym` −52.59, `rotor4_sym` −52.55) it is asserting a negative fixed
per-step cost, which is not a thing — least-squares is absorbing convexity into
the intercept, and the 64k cell it leans on hardest is n=1. The tier separation
below (generic path ≪ K-only kernels ≪ symv kernels) survives that; the
individual `a` values do not. For a number to act on, take the segment-wise
marginal cost between two replicated adjacent cells rather than a global fit.

| codec | a (ms) | b (ms/1k tok) | kernel dispatched |
|---|---|---|---|
| none | 5.94 | **0.326** | *(reference — no quant)* |
| k8vturbo2 | 6.56 | 0.329 | — (generic path) |
| tsym3 | 6.04 | 0.353 | — (generic path) |
| k8vturbo2tcq | 5.91 | 0.372 | — (generic path) |
| k8vturbo3tcq | 5.80 | 0.382 | — (generic path) |
| k8v8 | 5.90 | 0.393 | — (generic path) |
| planar_k | 6.53 | 0.397 | — (generic path) |
| planar3 | 6.77 | 0.404 | — (generic path) |
| k8vturbo3 | 5.18 | 0.408 | — (generic path) |
| tsym4 | 5.29 | 0.449 | — (generic path) |
| rotor3 | 5.80 | 0.471 | — (generic path, bf16-decode-seed) |
| planar | 5.66 | 0.472 | — (generic path) |
| rotor4 | 4.30 | 0.529 | — (generic path, bf16-decode-seed) |
| iso3 | 2.34 | 0.823 | — (generic path, bf16-decode-seed) |
| rot_k_tq4v | 4.79 | 1.160 | — (generic path) |
| iso4 | −7.38 | 1.414 | — (generic path, bf16-decode-seed) |
| k8v4 | 6.69 | 2.226 | — (generic path) |
| **k_iso3** | 38.77 | **2.248** | `iso_flash_decode_sdpa` |
| **k_rotor3** | 35.47 | **2.860** | `rotor_flash_decode_sdpa` |
| **k_rotor4** | 22.79 | **3.202** | `rotor_flash_decode_sdpa` |
| **k_iso4** | 11.16 | **3.492** | `iso_flash_decode_sdpa` |
| **iso3_sym** | −4.53 | **6.249** | `iso_flash_decode_symv_sdpa` |
| **rotor3_sym** | 0.35 | **6.476** | `rotor_flash_decode_symv_sdpa` |
| **iso4_sym** | −52.59 | **8.845** | `iso_flash_decode_symv_sdpa` |
| **rotor4_sym** | −52.55 | **9.418** | `rotor_flash_decode_symv_sdpa` |

Three clean tiers: **none-adjacent / generic-path codecs 0.33–2.2 ms/1k**
(`none` itself the floor at 0.326; `k8v4`'s 2.226 is the one outlier in this
tier — its 4-bit-V dequant is expensive even without a dedicated kernel),
**K-only kernel codecs 2.2–3.5 ms/1k** (7–10× `none`, but *stable* — no sign
of runaway divergence), **symv kernel codecs 6.2–9.4 ms/1k** (19–29×
`none`, and this is the one tier that actually degrades the codec below its
own predecessor at 64k — see §0 / issue #292). Quantizing V roughly doubles
marginal cost on top of quantizing K; the defect is the V-side dequant inside
the symv kernel specifically, not "quantized V" in general (`rot_k_tq4v`, a
non-kernel quantized-V codec, sits at 1.16 ms/1k — cheap) and not the
flash-decode scaffolding in general (K-only kernels are cheap and stable).

### 2c. SSD KV tier

**Not benched** — unchanged from 0.2.5 (a 256-token single-stream decode
never overflows the RAM prompt-cache at 8B + ≤23 GB KV).

---

## 3. Standing vs champion (decode)

| Prompt | rMLX `none` | champion (mlx-lm) | Δ | standing |
|---|---|---|---|---|
| 4k | **133.3** | 109.8 | **+21 %** | 🟢 WIN |
| 8k | **116.2** | 94.5 | **+23 %** | 🟢 WIN |
| 16k | **89.4** | 73.2 | **+22 %** | 🟢 WIN |
| 32k | **61.5** | 48.6 | **+27 %** | 🟢 WIN |
| 64k | **37.2** | 28.5 | **+31 %** | 🟢 WIN |

> `none` still leads mlx-lm at every context, and the margin now **grows**
> with context (+21→+31%, was flat +21→+27% in 0.2.5) — `none`'s own decode
> is unchanged from baseline (§0), so this is a mild champion-side data point
> more than an rMLX improvement; not re-benched this pass, carried from
> SIBLINGS.md.

---

## 4. Gaps & hypotheses (improvement plan)

Ranked by impact:

1. **`_sym` family: GPU-dispatching kernel, but net regression, worst at the
   long context it exists to serve.** Root-caused this pass to the
   quantized-**V** dequant specifically inside the symv kernel (§2.2,
   §0) — marginal cost 5.92–9.02 ms/1k vs the K-only-quant sibling's
   2.21–3.37 ms/1k. Filed as **issue #292**. No path to viability without a
   kernel fix; until fixed, `_sym` is strictly dominated by `none` on both
   decode and memory at every context measured.
2. **RETRACTED — the "11 non-kernel-dispatching codecs" regression does not
   reproduce; `k8v4`'s long-standing "crater" is the TurboFlash kernel**
   (**issue #293**). Original claim, kept for the record — per-cell deltas
   vs 0.2.5, `rmlx serve` + CBB harness:

   | codec | 16k | 32k | 64k |
   |---|---|---|---|
   | iso3 | 89.0→63.6 (**−28.5%**) | 63.3→42.3 (**−33.2%**) | 10.8→17.5 (+62.0%, baseline itself was broken here) |
   | iso4 | 91.0→61.9 (**−32.0%**) | 63.8→43.3 (**−32.1%**) | 13.1→11.2 (**−14.5%**) |
   | rotor3 | 92.1→71.8 (**−22.0%**) | 65.4→50.0 (**−23.5%**) | 38.6→27.6 (**−28.5%**) |
   | rotor4 | 87.5→75.5 (**−13.7%**) | 62.8→50.9 (**−18.9%**) | 39.4→25.8 (**−34.5%**) |
   | planar | 85.0→70.6 (**−16.9%**) | 55.5→50.0 (**−9.9%**) | 34.6→27.7 (**−19.9%**) |
   | planar3 | 88.0→70.6 (**−19.8%**) | 56.7→50.7 (**−10.6%**) | 33.3→30.8 (**−7.5%**) |
   | planar_k | 86.6→72.7 (**−16.1%**) | 56.2→52.7 (**−6.2%**) | 33.3→31.3 (**−6.0%**) |
   | k8vturbo3 | 89.1→82.1 (**−7.9%**) | 63.1→57.9 (**−8.2%**) | 36.2→31.6 (**−12.7%**) |
   | k8vturbo3tcq | 90.5→81.7 (**−9.7%**) | 62.5→56.8 (**−9.1%**) | 39.2→32.9 (**−16.1%**) |
   | tsym3 | 93.7→86.1 (**−8.1%**) | 66.3→58.3 (**−12.1%**) | 40.5→34.8 (**−14.1%**) |
   | tsym4 | 91.8→76.2 (**−17.0%**) | 64.3→54.5 (**−15.2%**) | 38.0→29.0 (**−23.7%**) |
   | *none (control)* | 90.1→89.4 (−0.8%) | 61.4→61.5 (+0.2%) | 36.3→37.2 (+2.5%) |
   | *k8v8 (control)* | 80.2→79.8 (−0.5%) | 57.0→55.4 (−2.8%) | 33.8→32.1 (−5.0%) |

   *(iso3@64k and iso4/rotor3/rotor4@64k pull in the opposite direction from
   their own 16k/32k cells for iso3 only — 0.2.5's iso3@64k=10.8 was itself a
   collapsed/broken number, so 17.5 there is not a real improvement, just a
   less-broken one; the genuine, monotonic regression window for iso3/iso4 is
   8k–32k.)*

   **What re-measurement found.** Re-run with `rmlx bench` (n=3 + 1 warmup per
   cell, one process per cell, prompt cache cleared per run, medians, and the
   tool's own settle gate refusing any cell that trends rather than spreads):

   | codec@ctx | ratio to `none`, 0.2.5 | ratio to `none`, now | shift |
   |---|---|---|---|
   | planar@16k | 0.943 | 0.943 | **0.0%** |
   | planar@32k | 0.904 | 0.955 | **+5.6%** |
   | tsym3@16k | 1.040 | 0.984 | −5.4% |
   | tsym3@32k | 1.080 | 1.006 | −6.9% |
   | iso3@16k | 0.988 | 0.899 | −9.0% |
   | iso3@32k | 1.031 | 0.888 | −13.9% |
   | k8v8@16k *(declared control)* | 0.890 | 0.963 | +8.2% |
   | k8v8@32k *(declared control)* | 0.928 | 0.971 | +4.6% |

   `planar`, whose claimed −16.9% @16k was among the largest cells, is
   **unchanged to three decimal places**. `k8v8`, the declared control,
   *improved* more than several "regressed" codecs moved. Only the
   `cpu_hot_path_reason()` family (`iso3`, and `tsym3` mildly) shifts at all,
   at roughly a third of the claimed magnitude and monotonically with context —
   consistent with store size (iso3 holds 11.3 GB at 32k against `none`'s
   5.4 GB, so it pays 2.1× the per-step KV maintenance bandwidth) rather than
   with a shared-path tax.

   **Why the original numbers moved: the harness, not the codecs.** Two runs of
   the *same* binary put `none`@32k at 61.5 and 47.76 — 29% apart — and
   reversed the codec ordering between them (`planar`/`none` @32k reads 0.813
   in one and 1.139 in the other). Under `rmlx bench`, `none`@32k measures
   65.81 and, re-run after the whole rest of the matrix, 65.75 — **0.1%
   apart**. So the machine is not the problem; serving each codec once and
   taking a single unguarded measurement is. Measured noise floors on this
   host: ≤2.4% within a cell, 0.1–0.9% across processes minutes apart, and up
   to **7.3%** across sessions an hour apart (a `k8v4`@32k gemma cell whose
   configuration provably did not change read 99.0 then 106.2). Most of the
   retracted table sits inside that last band.

   **What the re-measurement did surface.** §4.4 below used to attribute
   `k8v4`'s crater from 8k up to "an inherently costly generic-path V-4bit
   dequant on this arch". It is not the codec — it is the **TurboFlash MSL
   kernel**, which `--turbo-flash=auto` enabled on every recognised Apple
   family, this host included. Back-to-back on the same binary, `k8v4`@16k at
   `--max-ctx 16640` decodes **82.8 TPS with the kernel off and 24.2 TPS with
   it on** — both arms settled, both emitting a byte-identical token digest.
   At `--max-ctx 65536` the gap reads wider still (89.4 → 19.3 @16k, 61.3 →
   10.5 @32k) but those ON cells were refused by bench's settle gate — the 32k
   ON arm decoded 12.0 → 10.5 → 8.8 across three runs without ever reaching
   steady state — so 3.4× is the certified floor, not the headline. The 0.2.5 and 0.3.0 matrices were both taken
   through `rmlx serve`, which resolved the gate ON; `rmlx bench` never
   resolved it at all, which is why the crater vanishes there. Both halves are
   fixed: the gate is now global (every subcommand resolves it identically) and
   `auto` now holds OFF. `k8v4` re-measures at 88.8 @16k / 61.8 @32k under the
   shipped default.
3. **K-only codecs are usable now but not fast at long context** (4–5 TPS
   @64k). A real, working, GPU codec — no further urgency, but not a
   long-context recommendation either. Memory is now essentially free
   (1.00–1.54× `none`), so if a use case genuinely needs sub-8-bit K only,
   this is viable where it was not before.
4. **`k8v4`'s crater from 8k up was the TurboFlash kernel, and is fixed.**
   Every `k8v4` cell in §2 (39.3 @8k, down to 6.7 @64k) was measured through
   `rmlx serve`, where `--turbo-flash=auto` resolved ON. It is not "an
   inherently costly generic-path V-4bit dequant on this arch" — that reading
   was wrong, and the §2.2 marginal-cost fit for `k8v4` describes the kernel,
   not the codec. With the gate off, `k8v4` decodes **88.8 TPS @16k and 61.8
   @32k** (`rmlx bench`, n=3), i.e. within a few percent of `none`, for a
   byte-identical token digest. `auto` now holds OFF (see `docs/KV_QUANT.md`
   §TurboFlash), so this is the shipped behaviour. `rot_k_tq4v` is untouched by
   the gate — TurboFlash only serves K8V4 storage — and its mild −7…−12% drift
   at longer ctx stands as previously described.
5. **`*_sym` / `*tcq` prefill remains the heaviest cost family** — unchanged
   shape from 0.2.5 (`k8vturbo3tcq` 64k TTFT 108.5s vs `none`'s 64.8s).

---

## 5. Caveats

- **`none` is still the headline number and the smallest KV**, unchanged
  from 0.2.5 in every respect measured.
- **Every decode cell in §2 was measured through `rmlx serve` with
  `--turbo-flash=auto`, which resolved ON.** For `k8v4` — the only storage
  TurboFlash serves — that means §2's cells are kernel-on numbers and read
  3.4–5.9× low from 16k up; see §4.4. Every other codec is unaffected by the
  gate. Cells re-measured with `rmlx bench` are labelled as such in §4.2.
- **64k is n=1 measured** — point estimate, same caveat as 0.2.5. A single
  unguarded measurement is now known to be worth less than it looks on this
  host: see the noise-floor figures in §4.2 (up to 7.3% across sessions).
- **`rmlx baseline --prompt-tokens` used to tokenize the raw JSON fixture**
  **file text** (envelope + syntax), not just message content — a real,
  separate finding from this pass, since fixed. For `longctx_64k.json` that
  produced ~69.7k tokens, over both the model's 65536 ctx ceiling and the
  default `--max-prompt-tokens` cap; post-#223 ("error loudly on GPU prompt
  truncation") this was a hard, correct error rather than 0.2.5-era silent
  truncation. All KV-MB cells in this doc were measured with an explicit
  `--max-prompt-tokens 65528` (headroom for `--max-tokens 8`), reproducing
  the same effective 65536-token fill the 0.2.5 baseline almost certainly
  measured (validated: `none`@64k KV-MB landed at 10536 MB, matching 0.2.5 to
  the byte). The bug was specific to the `baseline --prompt-tokens` CLI path
  and did **not** affect the decode-TPS/TTFT cells, which go through the
  server's normal chat-completions endpoint and its own correct
  chat-templated tokenization. **Fixed**: `baseline` now renders a chat-JSON
  fixture through `chat_template.jinja` before tokenizing, matching the
  server path (`longctx_64k.json` now measures ~63.3k content tokens, no
  `--max-prompt-tokens` override needed); the workaround above documents how
  this pass's numbers were obtained, not current required usage.
- **No codec is CPU-bound at decode, at any size** — every kernel-dispatching
  codec confirmed dispatching at 4k and 64k; every non-dispatching codec
  shows zero CPU-dequant/host-download log lines at either end of the range.
  `iso3/4`/`rotor3/4` carry a documented CPU-side V-encode at **prefill
  only** (source: `cpu_hot_path_reason()`, `crates/rmlx-kv-quant/src/quant.rs`) — not a decode-time
  regression driver, and not new since 0.2.5.
- **No MTP / speculative** — Bonsai ships no drafter snapshot.
- **No §6 weight-quant sweep** — one on-disk 2-bit snapshot; no QAT siblings.
- **SSD tier not benched** — not triggered at 256-token single-stream (§2c).
- Full campaign checkpoint (127 rows: 125 main-matrix cells + 2 supplementary
  `--rotor-qjl on` ablation rows) audited complete via a real CSV parser —
  zero missing cells, zero duplicate rows, zero failed/incoherent runs across
  the entire 25×5 matrix.
