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
- **NEW finding this pass: a broad mid/long-context regression across 11
  non-kernel-dispatching codecs**, all -6…-34% at 16k–64k while `none` and
  `k8v8` hold exact parity (§4, detailed table). Affected:
  `iso3, iso4, rotor3, rotor4, planar, planar3, planar_k, k8vturbo3,
  k8vturbo3tcq, tsym3, tsym4` (**tsym4 added** — as regressed as tsym3, e.g.
  64k **38.0→29.0, −23.7%**, worse than tsym3's −14.1%). Not present in
  `k8v4`, `k8vturbo2`, `k8vturbo2tcq`, `rot_k_tq4v` (flat or only borderline
  cells). Filed as **issue #293** — see §4 for the shared-cause hypothesis and
  exact per-cell deltas.
- **No codec is CPU-bound at any size, at decode.** Every codec that has a
  dedicated flash-decode-over-quant kernel (`_sym`, K-only) was confirmed
  dispatching it at 4k *and* 64k via untimed verbose kernel-dispatch probes;
  every codec without one (`none`-adjacent, plain/turbo family, `iso3/4`,
  `rotor3/4`, `k8v4`, `rot_k_tq4v`) shows zero CPU-dequant/host-download log
  lines. The one caveat: `iso3/4`/`rotor3/4` (non-sym) carry a documented
  CPU-side V-encode at **prefill only** (`cpu_hot_path_reason()` in
  `quant.rs`) — decode itself reads a bf16 seed and is full-speed GPU; this
  is why `iso3/4`/`rotor3/4` track `none`'s TPS shape but not its prefill
  time.

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
| rot_k_tq4v | 13351 | 1.27× | 1.27× | | k_rotor3/4 | 16255 (stale) | 1.54× (defect; **1.12×** re-measured at 4k/16k — see below) | 1.14× (broken) |

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
pair, prompt 4k and 16k, 3 runs each:

| model | prompt | before | after | vs `none` after |
|---|---|---|---|---|
| Bonsai-8B (`kv_h=8`, D=128) | 4k | 990.0 MB | **717.2 MB** (−27.6%) | 1.118× |
| Bonsai-8B | 16k | 4090.7 MB | **2959.4 MB** (−27.7%) | 1.119× |
| gemma-4-e2b (`kv_h=1`, D=256) | 4k | 53.9 MB | **37.0 MB** (−31.4%) | 1.163× |
| gemma-4-e2b | 16k | 201.3 MB | **130.7 MB** (−35.1%) | 1.169× |

Every other codec measured byte-identical across that pair (0.00% delta), and
decode TPS moved only within run-to-run spread: a position-balanced A/B at 16k
(n=6 per side, 128 generated tokens) put `k_rotor3` at +0.23% and `k_rotor4` at
−1.76%, with TTFT flat to 0.2%.

**Measure this cell at 16k or longer, not at 4k.** On an M5 Max the 4k
`k_rotor3` decode is bimodal — it lands at either ≈19.5 or ≈23.9 TPS, a 20%
swing, on either binary, with TTFT rock-steady at 2340 ± 8 ms. n=8 per side is
not enough to see through that; the 16k/32k cells hold a 1–3% spread and are
where an A/B on this codec is resolvable.

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
2. **NEW — broad mid/long-context regression (16k–64k) across 11
   non-kernel-dispatching codecs, `none`/`k8v8` unaffected** (**issue #293**).
   Exact per-cell deltas vs 0.2.5:

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

   **Hypothesis (not verified by bisection — flagging as a starting point,
   not a conclusion):** every regressed codec in this table shows
   `NO_NAMED_KERNEL_EVENT` on the kernel-dispatch probe — i.e. all of them
   decode through the *same generic, non-codec-specific path* that `none`
   also uses. `none` (no quant at all) and `k8v8` (symmetric 8-bit K+V, the
   most heavily-trodden codec) are the two codecs in that shared-path group
   that are *not* regressed; everything else sharing the path is. That
   points at the shared decode/attention scaffolding picking up per-step
   overhead for quantized-but-kernel-less codecs specifically, rather than
   ten independent per-codec regressions. It is not a clean bit-width split
   either: `k8vturbo2`/`k8vturbo2tcq` (mostly unaffected) sit right next to
   `k8vturbo3`/`k8vturbo3tcq` (clearly regressed) in the same family, and
   `k8v4` (4-bit V, unaffected) sits next to `rot_k_tq4v` (4-bit V, mildly
   regressed −7…−12%) — so whatever changed is sensitive to specific
   codec/shape combinations within the shared path, not a uniform per-request
   tax. Recommend a `git bisect` across the recent shared-path work (the
   flash-decode shell rewrite and GPU-native norms-padding commits are the
   two most likely candidates given they touch common ring/mask
   construction) before assuming a single-line fix.
3. **K-only codecs are usable now but not fast at long context** (4–5 TPS
   @64k). A real, working, GPU codec — no further urgency, but not a
   long-context recommendation either. Memory is now essentially free
   (1.00–1.54× `none`), so if a use case genuinely needs sub-8-bit K only,
   this is viable where it was not before.
4. **`k8v4` / `rot_k_tq4v` (4-bit-V) are unchanged from 0.2.5's crater from
   8k up** (§2 table; k8v4 39.3→41.3 @8k is noise, still craters to 6.7 @64k;
   rot_k_tq4v 74.9→69.5 @8k, mild −7…−12% drift at longer ctx, same shape as
   before). Marginal cost (§2.2) confirms this is a real, expensive,
   *non*-kernel GPU dequant path (1.16–2.23 ms/1k) — not CPU-bound, just an
   inherently costly generic-path V-4bit dequant on this arch. Unfixed since
   0.2.5; still the lowest-priority item since `k8v8` (8-bit V) tracks `none`
   at every size and is the honest recommendation whenever V compression is
   wanted.
5. **`*_sym` / `*tcq` prefill remains the heaviest cost family** — unchanged
   shape from 0.2.5 (`k8vturbo3tcq` 64k TTFT 108.5s vs `none`'s 64.8s).

---

## 5. Caveats

- **`none` is still the headline number and the smallest KV**, unchanged
  from 0.2.5 in every respect measured.
- **64k is n=1 measured** — point estimate, same caveat as 0.2.5.
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
