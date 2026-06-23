# Bonsai-8B (2-bit) — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable 2-bit tier) = **mlx-lm (no KV quant)**, decode TPS:
> **109.8 / 94.5 / 73.2 / 48.6 / 28.5** (4k/8k/16k/32k/64k).

**Model:** `prism-ml__Ternary-Bonsai-8B-mlx-2bit` —
`Qwen3ForCausalLM`, dense ~8B, **2-bit affine** (group 128), full attention
(`sliding_window: null`), **YARN rope ×4** (native 16384 → 65536). Single
snapshot, ctx ceiling 64k (no 128k).
**Machine:** Apple M5 Max, 128 GB, macOS 26.5.1 (Darwin 25.5.0) · **Binary:**
`release-perf`, branch `bench/bonsai-siblings` @ `aa9eb31` (= main `0b6b825`
0.2.5 code + the bonsai docs; no engine diff vs released 0.2.5).
**Protocol:** batch=1, temp=0, `max_tokens=256`; serve once per codec at the 64k
ceiling (`--max-ctx 65536`, lazy-grow ring) + CBB `run_one` load-once for decode
and cold r0 TTFT; **n=3 measured** (4k/8k/16k/32k), **n=1 measured** (64k), 1
warmup `r0` discarded. **Same harness as SIBLINGS**, so rMLX cells compare
directly. Bar (§3): WIN / TIE-on-noise / LOSS.

> **Status: full 25-codec KV sweep (2026-06-23).** All 25 KV codecs run — Bonsai
> is `Qwen3ForCausalLM` **dense**, and `is_qwen_moe()` matches only the MoE
> classes, so the sub-8-bit-K arch-guard (`validate_resolved`) does **not** fire.
> No MTP / speculative grid: Bonsai ships **no drafter** snapshot (unlike the
> Gemma4 assistant / Qwen3.6 MTP-DFlash-Eagle families). No §6 weight-quant sweep:
> one on-disk 2-bit snapshot, no QAT siblings.

## 0. TL;DR

- **rMLX trails the mlx-lm champion at every size, and the gap WIDENS with
  context:** `none` decode **101 / 78 / 54 / 34 / 19** vs champ 110 / 95 / 73 /
  49 / 29 → **−8 / −17 / −27 / −30 / −32 %** (§3). This is the opposite of the
  Gemma4 small-dense result (#44 made e2b/e4b WIN); Bonsai 2-bit has not had that
  fix ported.
- **The cause is an f32 `none` KV.** `none` KV is **≈ 4.3 bytes/element**
  (1231 MB @4k → **19752 MB @64k**) — roughly **2× a bf16 baseline**. The widening
  loss is the textbook f32-KV signature (2× KV bandwidth, worst at long ctx) — the
  same class #44 fixed for Gemma4's global decode K/V but **never ported to the
  Qwen3 path**. **Porting the bf16 decode-KV stream to Qwen3 is the headline
  lever**, expected to roughly halve `none` KV and recover most of the long-ctx
  loss.
- **KV codec is a decode no-op at best, a loss at worst, and memory-inflating.**
  Mainstream codecs (planar, tsym, rotor, k8v8) ≈ `none` at short ctx; **none is
  the honest number** (the per-codec spread is thermal noise and the faster-looking
  cells carry a *larger* KV). Every codec inflates KV (1.08–2.02× `none`); the only
  sub-`none` codec is `k_rotor` (0.91×) which is **broken** (below).
- **4-bit-V codecs crater decode (~½ `none`).** `k8v4` and `rot_k_tq4v` (both tq4
  on V) fall to **42 / 27 / 15 / 8 TPS** (8k→64k) — the V-4bit dequant path is
  expensive on Bonsai. `k8v8` (8-bit V) tracks `none`. **Avoid 4-bit V here.**
- **iso* / *_sym collapse to ~4 TPS at 64k** (fine at 32k) — long-ctx CPU-dequant
  collapse, same class as Gemma4-31b iso_sym. **K-only codecs are outright
  broken:** `k_iso*` / `k_rotor*` decode at 0.4–5.5 TPS **and emit incoherent
  output** (repetition loop) — sub-8-bit K on a 2-bit GQA model is a correctness
  failure, not just slow. Capped early (k_iso ≤16k, k_rotor ≤8k).
- **Prefill/TTFT is heavy and exposes per-codec cost:** `none` 64k cold = **111 s**;
  `rotor*_sym` catastrophic (**246–261 s @64k**, QJL prefill); `*tcq` elevated;
  K-only crawl. All decode cells (except the K-only family) are coherent at temp=0.

---

## M. Measurement note (KV ring sizing — fairness)

rMLX `serve` historically pre-allocated the KV ring to `--max-ctx`, penalizing
small prompts under a high ceiling. Post-#25 the ring **grows lazily**, so a high
ceiling is free. Every codec here is served **once** at `--max-ctx 65536` (the
Bonsai ceiling) and all five prompt sizes sweep against the resident lazy-grown
ring — matching the dynamic-KV siblings, no per-size relaunch. The 64k fixture
(`longctx_64k.json`, ~63.3k Bonsai tokens) fits the ceiling with room for 256
generated tokens.

---

## 1. rMLX snapshot benched

| Snapshot (basename) | Weight quant | Arch / size | Role | Disk |
|---|---|---|---|---|
| `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | affine g128 b2 (ternary) | `Qwen3ForCausalLM` dense ~8B, full-attn, YARN ×4 | base | 2.2 GB |

No drafter snapshot exists for Bonsai → no speculative / MTP grid (§0).

---

## 2. rMLX full matrix

**Baseline cell = `decodeTPS · r0TTFT(s) · KV-MB`.** decode + cold r0 TTFT from
serve + `run_one` (load-once, chat-templated); `KV-MB` from `rmlx baseline
--record` filled-prefix `kv_cache_bytes` (captured per run via an events-table
high-water-mark). Markers: `—·—·MB` = decode capped (codec too slow to measure
that size; KV-MB still captured at `max_tokens=8`).

All 25 codecs run (no arch-guard on dense Qwen3). The K-only family
(`k_iso* / k_rotor*`) is capped (k_iso ≤16k, k_rotor ≤8k) and measured at
`max_tokens=64`, n=2 — its decode is CPU-bound *and* incoherent (§coherence).

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| none | 101.2·1.6s·1231 | 78.4·3.9s·2453 | 53.7·10.6s·5102 | 33.8·33.3s·10134 | 19.4·111.0s·19752 |
| k8v4 | 90.0·1.8s·1404 | 42.0·4.7s·2797 | 26.9·11.8s·5824 | 15.4·33.5s·11555 | 8.0·110.4s·22508 |
| k8v8 | 93.3·1.8s·1446 | 77.9·4.7s·2882 | 53.9·12.2s·6000 | 33.0·34.3s·11904 | 18.2·111.9s·23184 |
| planar | 101.6·1.9s·1625 | 78.0·4.9s·3239 | 52.7·12.3s·6749 | 33.7·34.3s·13378 | 18.9·112.3s·26044 |
| planar3 | 102.1·2.2s·1625 | 75.7·4.7s·3239 | 47.4·13.7s·6749 | 30.8·37.6s·13378 | 17.0·122.0s·26044 |
| planar_k | 91.4·2.1s·1517 | 70.3·5.4s·3025 | 47.9·13.5s·6300 | 30.5·37.6s·12494 | 17.5·121.6s·24328 |
| k8vturbo2 | 92.5·2.5s·1378 | 71.7·6.2s·2745 | 49.6·15.2s·5712 | 32.1·42.0s·11339 | 17.5·130.8s·22092 |
| k8vturbo3 | 95.0·2.7s·1391 | 73.3·6.6s·2770 | 50.6·15.9s·5766 | 32.3·42.3s·11446 | 17.8·132.4s·22300 |
| k8vturbo2tcq | 95.1·3.8s·1378 | 73.5·9.0s·2745 | 48.9·20.8s·5712 | 31.5·52.9s·11339 | 18.0·150.9s·22092 |
| k8vturbo3tcq | 94.9·4.6s·1391 | 79.8·10.3s·2770 | 55.4·22.4s·5766 | 35.3·53.9s·11446 | 20.0·150.2s·22300 |
| tsym3 | 101.7·2.1s·1335 | 79.9·4.9s·2660 | 55.9·12.8s·5535 | 35.2·36.4s·10990 | 19.5·117.3s·21416 |
| tsym4 | 98.2·1.8s·1361 | 76.6·4.5s·2713 | 53.0·11.8s·5647 | 33.9·33.6s·11206 | 18.7·109.9s·21832 |
| iso3 | 101.8·3.5s·1964 | 78.3·7.2s·3913 | 54.2·17.0s·8141 | 33.3·44.3s·16166 | 4.7·136.4s·31504 |
| iso4 | 98.1·4.1s·1964 | 76.8·9.0s·3913 | 52.3·20.6s·8141 | 33.4·52.6s·16166 | 5.0·153.9s·31504 |
| iso3_sym | 98.7·4.0s·2483 | 73.3·8.8s·4944 | 52.3·20.8s·10282 | 30.1·55.9s·20428 | 4.5·177.5s·39824 |
| iso4_sym | 100.7·6.3s·2483 | 77.5·13.4s·4944 | 52.4·30.3s·10282 | 32.5·72.1s·20428 | 4.1·199.8s·39824 |
| rotor3 | 100.3·2.7s·1621 | 78.1·6.3s·3229 | 54.0·16.4s·6719 | 33.3·43.2s·13339 | 19.2·136.6s·25992 |
| rotor4 | 99.8·3.2s·1621 | 76.4·7.3s·3229 | 54.0·16.6s·6719 | 34.1·45.0s·13339 | 19.2·134.9s·25992 |
| rotor3_sym | 102.3·9.7s·1813 | 78.0·20.8s·3610 | 53.1·43.5s·7506 | 34.3·100.4s·14910 | 4.2·245.8s·29062 |
| rotor4_sym | 101.8·10.4s·1813 | 75.9·22.1s·3610 | 54.6·47.6s·7506 | 34.2·106.7s·14910 | 4.5·260.7s·29062 |
| k_iso3 | 5.5·3.4s·1442 | 2.7·7.2s·2872 | 1.5·15.2s·5975 | —·—·11868 | —·—·23132 |
| k_iso4 | 2.1·3.9s·1442 | 1.0·8.7s·2872 | 0.5·20.2s·5975 | —·—·11868 | —·—·23132 |
| k_rotor3 | 0.7·8.7s·1116 | 0.4·18.4s·2222 | —·—·4621 | —·—·9176 | —·—·17882 |
| k_rotor4 | 0.7·9.0s·1116 | 0.4·19.0s·2222 | —·—·4621 | —·—·9176 | —·—·17882 |
| rot_k_tq4v | 77.5·1.6s·1415 | 49.6·3.9s·2817 | 28.1·11.4s·5859 | 15.6·33.9s·11632 | 8.2·112.5s·22666 |

**Apparent best decode per size** (rotor3_sym 4k / tsym3 8k·16k / k8vturbo3tcq
32k·64k) is **thermal noise around `none`** — those cells are within ~±2 % of
`none` *and* carry a larger KV (k8vturbo3tcq @64k = 1.13× `none` KV for a +0.6 TPS
"win"). There is no real KV-quant decode win on Bonsai. `none` is the headline.

### 2.1 KV-cache size (MB) and ratio vs `none`

`none` KV is **f32** (≈4.3 bytes/element); every codec keeps a bf16/packed seed
*alongside* its blocks, so all but the broken `k_rotor` are **larger** than
`none`. (Once the Qwen3 decode-KV stream is moved to bf16 — §4 — `none` should
roughly halve and the ratios shift further against the codecs.)

| KV | MB@64k | ratio | | KV | MB@64k | ratio |
|---|---|---|---|---|---|---|
| none | 19752 | 1.00× | | iso3 / iso4 | 31504 | 1.59× |
| k_rotor3/4 | 17882 | **0.91×** (broken) | | iso3_sym / iso4_sym | 39824 | 2.02× |
| tsym3 | 21416 | 1.08× | | rotor3 / rotor4 | 25992 | 1.32× |
| tsym4 | 21832 | 1.11× | | rotor3_sym / rotor4_sym | 29062 | 1.47× |
| k8vturbo2/3 | 22092–22300 | 1.12–1.13× | | planar / planar3 | 26044 | 1.32× |
| k8v4 | 22508 | 1.14× | | planar_k | 24328 | 1.23× |
| rot_k_tq4v | 22666 | 1.15× | | k8v8 | 23184 | 1.17× |
| k_iso3/4 | 23132 | 1.17× | | | | |

### 2c. SSD KV tier

**Not benched.** As on Gemma4 (`SIBLINGS`/`rMLX` §2c), a 256-token single-stream
decode never overflows the RAM prompt-cache — at 8B + ≤20 GB KV the SSD tier
would not spill, so it is decode-neutral and untriggered at these sizes. SSD is a
capacity feature; exercising it needs a multi-turn / >RAM-KV scenario. Left out
rather than reported as a no-op cell.

---

## 3. Standing vs champion (decode)

rMLX `none` decode vs the SIBLINGS mlx-lm champion. `none` is the honest number —
the per-codec spread is noise (§2), so no cherry-picked "best codec" is used.

### Bonsai-8B — 🔴 LOSS at every size, widening with context

| Prompt | rMLX `none` | champion (mlx-lm) | Δ | standing |
|---|---|---|---|---|
| 4k | 101.2 | 109.8 | −8 % | 🔴 LOSS |
| 8k | 78.4 | 94.5 | −17 % | 🔴 LOSS |
| 16k | 53.7 | 73.2 | −27 % | 🔴 LOSS |
| 32k | 33.8 | 48.6 | −30 % | 🔴 LOSS |
| 64k | 19.4 | 28.5 | −32 % | 🔴 LOSS |

> **Verdict: rMLX loses on Bonsai 2-bit at every context, and the loss grows with
> context** — the signature of a bandwidth (not kernel) deficit. No KV codec
> recovers it: the axis is a decode no-op at short ctx and a loss at long ctx, and
> it inflates memory. The lever is the **f32 → bf16 decode-KV stream port to
> Qwen3** (§4), not KV quant.

---

## 4. Gaps & hypotheses (improvement plan)

Ranked by impact:

1. **`none` decode-KV is f32 on the Qwen3 path — port the #44 bf16 stream.** The
   single highest-value lever. `none` KV measures ≈4.3 bytes/element (19.75 GB
   @64k) — about double a bf16 baseline — and the decode loss widens with context
   exactly as a 2× KV-bandwidth penalty predicts (−8 % @4k → −32 % @64k). #44
   moved Gemma4's global decode K/V to bf16 and flipped e2b/e4b from losses to
   wins; the Qwen3 decode path never received it. Expected payoff: ~halve `none`
   KV and recover most of the long-ctx gap toward mlx-lm parity.
2. **4-bit-V dequant is expensive on Bonsai — fix or avoid tq4-V.** `k8v4` and
   `rot_k_tq4v` (both tq4 on V) crater to ~½ `none` from 8k up (42/27/15/8 TPS),
   while `k8v8` (8-bit V) tracks `none`. The V-4bit dequant on this 2-bit/GQA-8
   model costs more than the bandwidth it saves. Either a faster V-4bit decode
   kernel or steering `auto` away from 4-bit V on this arch.
3. **Long-ctx codec collapse (iso*, *_sym @64k = ~4 TPS).** Fine at 32k, collapse
   at 64k — the CPU-dequant / cold-codec path scaling with KV length (same class
   as Gemma4-31b iso_sym). No path to viability without a Metal dequant kernel.
4. **K-only codecs are a correctness failure, not just slow.** `k_iso* / k_rotor*`
   decode at 0.4–5.5 TPS **and produce incoherent output** (repetition loop) on
   Bonsai 2-bit — sub-8-bit K on a high-GQA 2-bit model is the PPL-disaster path
   the Qwen-MoE arch-guard already rejects. Recommend extending the guard (or an
   `auto` skip) to dense 2-bit Qwen3, or at minimum a loud resolve warning.
5. **Prefill/TTFT is heavy.** `none` 64k cold = 111 s; `rotor*_sym` 246–261 s
   (QJL prefill); `*tcq` elevated. Prefill is the second big-ticket lever after
   the KV-stream fix, independent of decode.

---

## 5. Caveats

- **rMLX loses to mlx-lm on Bonsai at every size** (−8…−32 %) — this is the honest
  standing, driven by the f32 decode-KV (§4.1), not a measurement artifact.
- **`none` is the headline number.** Per-codec "best" cells are thermal noise
  around `none` and carry a larger KV — not real wins.
- **64k is n=1 measured** (single run after the discarded warmup) — point estimate.
- **K-only codecs (`k_iso* / k_rotor*`) are capped AND incoherent** — decode
  measured at `max_tokens=64`, n=2, capped at 16k/8k; their output is a repetition
  loop. Do not use on Bonsai. KV-MB still captured (baseline `max_tokens=8`).
- **`k8v4` 4k (90) vs 8k (42) cliff** — the tq4-V cost appears from 8k up;
  reproduced independently by `rot_k_tq4v` (same tq4-V), so it is a real codec
  cost, not a warmup artifact.
- **No MTP / speculative** — Bonsai ships no drafter snapshot.
- **No §6 weight-quant sweep** — one on-disk 2-bit snapshot; no QAT siblings.
- **SSD tier not benched** — not triggered at 256-token single-stream (§2c).
- rMLX decode/TTFT cells recorded in CBB `metrics/runs/*.jsonl` (backend=rmlx);
  KV-MB in `runs.db` events (`baseline_kv_cache_bytes`). Aggregator:
  `Cross-Backend-Bench/scripts/agg_bonsai_kvsweep_full.py`.
