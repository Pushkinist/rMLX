# Bonsai-8B (2-bit) — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable 2-bit tier) = **mlx-lm (no KV quant)**, decode TPS:
> **109.8 / 94.5 / 73.2 / 48.6 / 28.5** (4k/8k/16k/32k/64k).

**Model:** `prism-ml__Ternary-Bonsai-8B-mlx-2bit` —
`Qwen3ForCausalLM`, dense ~8B, **2-bit affine** (group 128), full attention
(`sliding_window: null`), **YARN rope ×4** (native 16384 → 65536). Single
snapshot, ctx ceiling 64k (no 128k).
**Machine:** Apple M5 Max, 128 GB, macOS 26.5.1 (Darwin 25.5.0) · **Binary:**
`release-perf`, rMLX 0.2.5 (`main` @ `22b8ba1`). **Date:** 2026-06-24.
**Protocol:** batch=1, temp=0, `max_tokens=256`; serve once per codec at the 64k
ceiling (`--max-ctx 65536`, lazy-grow ring) + CBB `run_one` load-once for decode
and cold r0 TTFT; **n=3 measured** (4k/8k/16k/32k), **n=1 measured** (64k), 1
warmup `r0` discarded. **Same harness as SIBLINGS**, so rMLX cells compare
directly. Bar (§3): WIN / TIE-on-noise / LOSS.

> **All 25 KV codecs run.** Bonsai is `Qwen3ForCausalLM` **dense**, and
> `is_qwen_moe()` matches only the MoE classes, so the sub-8-bit-K arch-guard
> (`validate_resolved`) does **not** fire. No MTP / speculative grid: Bonsai
> ships **no drafter** snapshot (unlike the Gemma4 assistant / Qwen3.6
> MTP-DFlash-Eagle families). No §6 weight-quant sweep: one on-disk 2-bit
> snapshot, no QAT siblings.

## 0. TL;DR

- **rMLX `none` LEADS the mlx-lm champion at every context.** `none` decode
  **133 / 115 / 90 / 61 / 36** (4k/8k/16k/32k/64k) vs champ **110 / 95 / 73 / 49
  / 29** → **+21 / +22 / +23 / +26 / +27 %** (§3). The lead is roughly flat across
  context — no bandwidth penalty that grows with KV size.
- **`none` KV is bf16 (≈2 B/element) and the smallest KV of any codec.** 657 MB
  @4k → 10536 MB @64k. Every quantized codec carries a *larger* resident KV than
  `none` (1.14×–2.91×), so `none` is both the fastest honest number and the
  memory winner (§2.1).
- **Prefill is light at short/mid context** (`none` cold TTFT 1.3 / 2.9 / 7.3 s at
  4k/8k/16k) and **62 s at 64k** — heavy but tractable; `*_sym` codecs are the
  prefill outliers (§4).
- **KV codec buys no meaningful decode win and costs memory.** `none` is the
  headline. A cluster of non-4bit-V codecs (`tsym3/4`, `rotor3`, `k8vturbo3`) sits
  a small, consistent **+4…+12 %** above `none` at 16k–64k (best-per-size:
  `tsym3` 93.7 / 66.3 / 40.5 at 16k/32k/64k), a modest long-ctx K-bandwidth
  effect — but each carries **1.16–1.24×** the KV, so `none` stays the default.
- **4-bit-V codecs crater decode (~⅓ `none` from 8k up).** `k8v4`
  (128→39→24→13→7) and `rot_k_tq4v` (98→75→47→26→14) — the tq4-V dequant path is
  expensive on this 2-bit/GQA-8 model. `k8v8` (8-bit V) tracks `none`. **Avoid
  4-bit V here.**
- **iso* / *_sym collapse at 64k** (~6–13 TPS; healthy ~62–64 at 32k) — long-ctx
  CPU-dequant collapse, same class as Gemma4-31b iso_sym. **K-only codecs
  (`k_iso* / k_rotor*`) are unusable:** 0.4–5.8 TPS, CPU-bound (capped k_iso ≤16k,
  k_rotor ≤8k).
  > **Superseded for `k_rotor3/4` (with `--rotor-qjl off`).** Their per-step
  > full-prefix CPU dequant is replaced by a fused MSL flash-decode over the
  > packed rotor store (`rotor_flash_decode`, `docs/KV_QUANT.md`). Re-measured on
  > this model at 4k: `k_rotor3` 1.34 → **16.2** TPS, `k_rotor4` 1.36 → **17.0**
  > (24× over the `--rotor-qjl on` default). The rows below are a pre-kernel
  > snapshot. `k_iso*` and the default `--rotor-qjl on` path are unchanged.

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

The K-only family (`k_iso* / k_rotor*`) is capped (k_iso ≤16k, k_rotor ≤8k) and
measured at `max_tokens=64`, n=2 — its decode is CPU-bound (§0).

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| none | 133.1·1.3s·657 | 114.9·2.9s·1309 | 90.1·7.3s·2724 | 61.4·21.2s·5407 | 36.3·62.0s·10536 |
| k8v4 | 128.5·1.5s·829 | 39.3·3.6s·1653 | 23.7·8.7s·3445 | 13.3·22.7s·6828 | 7.0·66.9s·13292 |
| k8v8 | 126.5·1.6s·871 | 110.4·3.9s·1738 | 80.2·9.1s·3622 | 57.0·22.5s·7177 | 33.8·66.0s·13968 |
| planar | 130.4·1.5s·1050 | 108.1·3.8s·2095 | 85.0·8.8s·4371 | 55.5·22.5s·8652 | 34.6·66.2s·16828 |
| planar3 | 127.9·1.5s·1050 | 114.5·3.8s·2095 | 88.0·9.0s·4371 | 56.7·22.5s·8652 | 33.3·66.5s·16828 |
| planar_k | 130.2·1.4s·943 | 114.1·3.8s·1881 | 86.6·9.0s·3921 | 56.2·22.8s·7767 | 33.3·67.5s·15112 |
| k8vturbo2 | 128.4·1.9s·803 | 112.2·4.7s·1601 | 88.3·10.8s·3334 | 63.0·26.8s·6612 | 35.4·74.8s·12876 |
| k8vturbo3 | 129.3·2.1s·816 | 114.9·4.9s·1627 | 89.1·10.9s·3388 | 63.1·27.2s·6719 | 36.2·76.2s·13084 |
| k8vturbo2tcq | 131.4·3.4s·803 | 113.3·7.6s·1601 | 83.7·15.9s·3334 | 57.4·38.2s·6612 | 38.4·95.7s·12876 |
| k8vturbo3tcq | 131.2·4.2s·816 | 110.5·8.6s·1627 | 90.5·19.5s·3388 | 62.5·42.5s·6719 | 39.2·105.3s·13084 |
| tsym3 | 133.7·1.8s·761 | 115.4·4.2s·1516 | **93.7**·9.3s·3156 | **66.3**·23.8s·6263 | **40.5**·67.4s·12200 |
| tsym4 | 133.4·1.2s·787 | **117.5**·3.0s·1569 | 91.8·7.3s·3268 | 64.3·19.5s·6480 | 38.0·59.2s·12616 |
| iso3 | 134.4·2.6s·1390 | 115.3·5.4s·2769 | 89.0·12.2s·5763 | 63.3·30.5s·11439 | 10.8·79.8s·22288 |
| iso4 | 134.2·3.6s·1390 | 113.2·7.5s·2769 | 91.0·16.8s·5763 | 63.8·37.5s·11439 | 13.1·94.3s·22288 |
| iso3_sym | 133.2·3.5s·1908 | 114.0·7.4s·3800 | 87.4·16.0s·7904 | 62.8·37.1s·15702 | 5.9·93.7s·30608 |
| iso4_sym | 131.4·5.6s·1908 | 114.4·12.0s·3800 | 89.4·25.4s·7904 | 61.5·56.9s·15702 | 5.7·150.2s·30608 |
| rotor3 | **136.2**·2.4s·1046 | 116.2·5.1s·2085 | 92.1·12.0s·4341 | 65.4·30.9s·8612 | 38.6·76.9s·16776 |
| rotor4 | 133.8·2.8s·1046 | 115.4·6.2s·2085 | 87.5·13.4s·4341 | 62.8·32.5s·8612 | 39.4·82.8s·16776 |
| rotor3_sym | 136.2·9.2s·1239 | 116.1·19.2s·2466 | 90.0·39.5s·5128 | 64.1·94.8s·10183 | 5.8·214.4s·19846 |
| rotor4_sym | 132.7·9.9s·1239 | 113.3·20.7s·2466 | 90.1·43.3s·5128 | 64.4·101.9s·10183 | 7.3·225.3s·19846 |
| k_iso3 | 5.8·2.4s·1075 | 2.7·5.5s·2141 | 1.5·11.6s·4455 | —·—·8848 | —·—·17244 |
| k_iso4 | 2.1·3.5s·1075 | 1.1·7.3s·2141 | 0.5·16.1s·4455 | —·—·8848 | —·—·17244 |
| k_rotor3 | 0.7·8.2s·749 | 0.4·17.0s·1491 | —·—·3101 | —·—·6156 | —·—·11994 |
| k_rotor4 | 0.7·8.5s·749 | 0.4·17.7s·1491 | —·—·3101 | —·—·6156 | —·—·11994 |
| rot_k_tq4v | 98.4·1.2s·834 | 74.9·2.8s·1660 | 47.3·7.2s·3454 | 26.4·20.5s·6852 | 14.4·59.6s·13346 |

**Best decode per size** (bold above): `rotor3` 4k (136.2), `tsym4` 8k (117.5),
`tsym3` 16k·32k·64k (93.7 / 66.3 / 40.5). These beat `none` by **+0.5…+12 %**
(largest at 32k/64k) — a small, *consistent* long-ctx K-bandwidth effect (an
8-bit-K codec reads less per decode step than bf16). But each carries
**1.16–1.24×** the `none` KV, and the 64k cells are n=1, so `none` stays the
headline / default.

### 2.1 KV-cache size (MB) and ratio vs `none`

`none` KV is **bf16** (≈2 bytes/element): 657 / 1309 / 2724 / 5407 / 10536 MB at
4k…64k. Every quantized codec keeps a bf16/packed seed *alongside* its blocks, so
all are **larger** than `none` — `none` is the clear memory winner at every
context.

| KV | MB@64k | ratio | | KV | MB@64k | ratio |
|---|---|---|---|---|---|---|
| **none** | **10536** | **1.00×** | | iso3 / iso4 | 22288 | 2.12× |
| k_rotor3/4 | 11994 | 1.14× (broken) | | iso3_sym / iso4_sym | 30608 | 2.91× |
| tsym3 | 12200 | 1.16× | | rotor3 / rotor4 | 16776 | 1.59× |
| tsym4 | 12616 | 1.20× | | rotor3_sym / rotor4_sym | 19846 | 1.88× |
| k8vturbo2/3 | 12876–13084 | 1.22–1.24× | | planar / planar3 | 16828 | 1.60× |
| k8vturbo2/3tcq | 12876–13084 | 1.22–1.24× | | planar_k | 15112 | 1.43× |
| k8v4 | 13292 | 1.26× | | k8v8 | 13968 | 1.33× |
| rot_k_tq4v | 13346 | 1.27× | | k_iso3/4 | 17244 | 1.64× |

### 2c. SSD KV tier

**Not benched.** As on Gemma4 (`SIBLINGS`/`rMLX` §2c), a 256-token single-stream
decode never overflows the RAM prompt-cache — at 8B + ≤11 GB KV the SSD tier
would not spill, so it is decode-neutral and untriggered at these sizes. SSD is a
capacity feature; exercising it needs a multi-turn / >RAM-KV scenario. Left out
rather than reported as a no-op cell.

---

## 3. Standing vs champion (decode)

rMLX `none` decode vs the SIBLINGS mlx-lm champion, **same serve + `run_one`
harness** on both sides (directly comparable). `none` is the honest number — the
per-codec spread (§2) is small and memory-costly, so no cherry-picked "best
codec" is used.

| Prompt | rMLX `none` | champion (mlx-lm) | Δ | standing |
|---|---|---|---|---|
| 4k | **133.1** | 109.8 | **+21 %** | 🟢 WIN |
| 8k | **114.9** | 94.5 | **+22 %** | 🟢 WIN |
| 16k | **90.1** | 73.2 | **+23 %** | 🟢 WIN |
| 32k | **61.4** | 48.6 | **+26 %** | 🟢 WIN |
| 64k | **36.3** | 28.5 | **+27 %** | 🟢 WIN |

> **rMLX leads mlx-lm on Bonsai 2-bit at every context** (+21…+27 %), and the
> lead is roughly flat with context — there is no KV-bandwidth penalty that grows
> with sequence length. `none` is the champion-beating cell; KV quant adds no
> decode and costs memory (§2).

---

## 4. Gaps & hypotheses (improvement plan)

Ranked by impact:

1. **4-bit-V dequant is expensive on Bonsai — fix or avoid tq4-V.** `k8v4` and
   `rot_k_tq4v` (both tq4 on V) crater to ~⅓ `none` from 8k up (39/24/13/7 and
   75/47/26/14 TPS), while `k8v8` (8-bit V) tracks `none`. The V-4bit dequant on
   this 2-bit/GQA-8 model costs more than the bandwidth it saves. Either a faster
   V-4bit decode kernel or steering `auto` away from 4-bit V on this arch.
2. **Long-ctx codec collapse (iso*, *_sym @64k = ~6–13 TPS).** Healthy at 32k
   (~62–64), collapse at 64k — the CPU-dequant / cold-codec path scaling with KV
   length (same class as Gemma4-31b iso_sym). No path to viability without a Metal
   dequant kernel.
3. **K-only codecs are unusably slow.** `k_iso* / k_rotor*` decode at 0.4–5.8 TPS
   (CPU-bound) — sub-8-bit K on a high-GQA 2-bit model. Recommend an `auto` skip /
   loud resolve warning for sub-8-bit-K on dense 2-bit Qwen3, the way the Qwen-MoE
   arch-guard already rejects them.
4. **`*_sym` prefill is heavy.** `none` 64k cold = 62 s; `rotor*_sym` 214–225 s
   @64k (QJL prefill); `*tcq` elevated. Prefill is the main remaining lever, and
   it is codec-bound — the `*_sym` and `*tcq` families pay a large per-chunk
   prefill cost that `none` and the plain codecs do not.

---

## 5. Caveats

- **`none` is the headline number** and the smallest KV. The non-4bit-V codec
  cluster (`tsym3/4`, `rotor3`, `k8vturbo3`) is a real but small +4…+12 % at
  16k–64k, at 1.16–1.24× the KV — not a free win.
- **64k is n=1 measured** (single run after the discarded warmup) — point estimate.
  The 64k codec-vs-`none` deltas in particular should be read as a trend, not a
  CI-bounded comparison.
- **K-only codecs (`k_iso* / k_rotor*`) are capped and unusably slow** — decode
  measured at `max_tokens=64`, n=2, capped at 16k/8k. KV-MB still captured
  (baseline `max_tokens=8`). Do not use on Bonsai.
- **`k8v4` 4k (128) vs 8k (39) cliff** — the tq4-V cost appears from 8k up;
  reproduced independently by `rot_k_tq4v` (same tq4-V), so it is a real codec
  cost, not a warmup artifact.
- **No MTP / speculative** — Bonsai ships no drafter snapshot.
- **No §6 weight-quant sweep** — one on-disk 2-bit snapshot; no QAT siblings.
- **SSD tier not benched** — not triggered at 256-token single-stream (§2c).
- rMLX decode/TTFT cells recorded in CBB `metrics/runs/*.jsonl` (backend=rmlx);
  KV-MB in `runs.db` events (`baseline_kv_cache_bytes`). Aggregator:
  `Cross-Backend-Bench/scripts/agg_bonsai_kvsweep_full.py`.
