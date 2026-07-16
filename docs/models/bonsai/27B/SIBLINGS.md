# Bonsai-27B (2-bit) — Sibling-Backend Champions

> Companion (planned): `rMLX.md` — rMLX full KV-quant matrix + standing-vs-champion.
> This file covers sibling backends only.
> Mirrors [`../8B/SIBLINGS.md`](../8B/SIBLINGS.md) in protocol and shape; read the
> deltas from the 8B campaign in §0 and §3 — the 27B is a **different architecture**
> and the sibling story changes materially.

**Model:** `prism-ml__Ternary-Bonsai-27B-mlx-2bit` —
`Qwen3_5ForConditionalGeneration`, dense ~27B **text tower** of a VLM-shaped
checkpoint (nested `text_config`/`vision_config`; text-only bench), **2-bit
affine** weights (group 128), **GatedDeltaNet hybrid attention**
(`full_attention_interval: 4` → 16 full-attention + 48 linear/GDN layers of 64),
`head_dim: 256`, **native 262144 context** (no YARN — plain rope, high theta,
mRoPE interleaved). Carries an MTP head declaration (`mtp_num_hidden_layers: 1`)
but ships **no `mtp.*` weights**. Single snapshot.
**Collected:** 2026-07-15
**Machine:** Apple M5 Max, 128 GB unified memory, macOS 26.5.1 (Darwin 25.5.0)
**Protocol:** batch=1 single-stream; temp=0 (greedy); `max_tokens=256`;
n=5 requested per cell (1 warmup `r0` discarded → **n=4 measured**) at
4k/8k/16k/32k; **n=2 requested → n=1 measured** at 64k/128k. Decode-TPS = median
of the measured runs; ranges shown are [min–max of measured] (not bootstrap CI —
runs collected per-cell via direct `runners.run_one` calls, see §5). The siblings
were each **pulled to latest** before their run (per-backend SHAs in §5).

---

## 0. TL;DR

- **On the GDN-hybrid 27B the sibling decode field is a near-tie — the opposite
  of the 8B.** Where the 8B (full-attention `Qwen3ForCausalLM`) had mlx-lm
  clearly leading and oMLX collapsing at long context, here all three benched
  backends cluster within run-to-run noise at every size. The reason is
  architectural: only **16 of 64 layers** are full-attention (growing KV); the
  other 48 are GatedDeltaNet linear-attention with **fixed-size recurrent
  state**, so KV bandwidth barely scales with context and no backend is
  KV-bandwidth-starved.
- **Decode is flat with context.** mlx-lm (the clean same-method reference)
  runs **45 → 42 → 41 → 37 → 30 → 23 TPS** at 4k/8k/16k/32k/64k/128k — a ~2×
  falloff across a **32×** context range. The 8B dropped ~4× across just a 4×
  range (110→28 at 4k→64k). The GDN hybrid is the difference.
- **KV-8,4 (turboquant) is NOT a net loss here — it is at parity.** On the 8B,
  tq's KV-8,4 lost 2–12 %. On the 27B it is within noise (43.4 / 43.8 / 40.4 /
  31.0 / 27.9 / 21.5 TPS; 8k even edges mlx-lm). Two reasons: (a) KV-8,4 only
  touches the **16 full-attn layers** (type dispatch skips the 48 GDN layers);
  (b) in this fork's single-serve path the quantized cache is used **only for
  LRU prompt-cache storage** — it is dequantized before `generate_step`, so
  neither prefill nor decode ever computes against quantized K/V (§3).
- **oMLX does NOT collapse at long context here** (it cratered to 4 TPS at 64k
  on the 8B). RSS stayed a shallow 9.2 → 15.4 GB across 4k→128k, decode degraded
  gracefully. Same GDN reason — shallow KV growth, no forced paging.
  **BUT the oMLX numbers carry heavy measurement caveats** (SSE-chunk undercount
  recovered from server logs; version-advantaged on MLX 0.32 vs the others' 0.31.2;
  different measurement path) — reported, but **not crowned** (§3, §5).
- **Coherence holds to 128k on every backend.** Every benched cell produced the
  same on-task thinking-model output under native 262144-context rope — no
  degeneration at 128k.
- **No 256k cell** — the 128k fixture (`longctx_128k.json`, ~131k tokens) fits
  the 262144 ceiling and is the top cell run; a 256k fixture does not exist and
  was not generated (§4).
- **ollama, paroquant, isoquant, mlx-vlm, rMLX** are skipped — see §4.

---

## 1. Snapshot in Scope

One snapshot; all three benched siblings load the **same 2-bit affine weights**
→ directly weight-comparable. `mlx-lm-turboquant` additionally quantizes the
full-attention KV to 8,4.

| Snapshot (basename) | Weight quant | Arch / size | Role | Disk | Max ctx |
|---|---|---|---|---|---|
| `prism-ml__Ternary-Bonsai-27B-mlx-2bit` | affine g128 b2 | `Qwen3_5ForConditionalGeneration` dense, ~27B text tower | **bench target** (dense sibling of the Qwen3.6-35B-A3B MoE) | 7.9 GB | 262144 |

Architecture facts (from `config.json` → `text_config`, drive the bench design):

| Fact | Value | Consequence |
|---|---|---|
| `num_hidden_layers` | 64 | — |
| attention | **GDN hybrid**, `full_attention_interval: 4` | **16 full-attention + 48 linear (GatedDeltaNet)** layers |
| heads (attn / kv) | 24 / 4 (GQA), `head_dim` 256 | full-attn KV is 4 heads × 256; only on 16/64 layers |
| linear (GDN) state | `linear_conv_kernel_dim: 4`, key/value head-dim 128 | **fixed-size recurrent state** — does not grow with context |
| `max_position_embeddings` | **262144** | ctx ceiling 256k; top cell run is 128k |
| rope | default (no YARN), `rope_theta 1e7`, mRoPE interleaved | native 262144, not a scaled extension (contrast 8B YARN ×4) |
| `mtp_num_hidden_layers` | 1 | MTP head declared, **no `mtp.*` weights in checkpoint** → inert on every backend |
| config shape | nested `text_config`/`vision_config`, `image_token_id` | VLM wrapper; **text-only bench** (vision dropped by every loader) |
| weight quant | 2-bit affine, group 128 | tiny weight stream; decode is not weight-bandwidth-bound |

All three siblings load this arch cleanly — mlx-lm and its turboquant fork ship
`qwen3_5.py` + `gated_delta.py`; oMLX serves it through its batched engine. There
is **no missing-module blocker** (contrast the Gemma4 KV-shared `sanitize()` story).

---

## 2. Sibling Champion Table

Decode TPS = median of measured runs (r0 warmup dropped), temp=0, `max_tokens=256`,
single-stream. **All three benched siblings carry affine g128 b2 weights** →
weight-comparable. `mlx-lm-tq` additionally quantizes the 16 full-attn KV to 8,4
("fake-asym"). **oMLX numbers are measurement-caveated** (§3.4, §5) — shown for
completeness, not crowned. Champion per row (among the same-method siblings
mlx-lm / tq) in **bold**.

### 2a. Decode TPS

| Prompt | mlx-lm (no KV) | mlx-lm-tq (KV 8,4) | oMLX (caveated) | Champion |
|---|---|---|---|---|
| 4k | **45.1** [44.3–46.3] | 43.4 [42.2–45.5] | 51.0 ¹ | **mlx-lm** |
| 8k | 41.7 [40.4–44.6] | **43.8** [43.1–44.1] | 46.8 ¹ | **tq ≈ mlx-lm** ² |
| 16k | **40.6** [39.9–41.9] | 40.4 [40.0–40.9] | 45.7 ¹ | **mlx-lm ≈ tq** |
| 32k | **36.8** [35.9–37.0] | 31.0 [27.9–35.7] ³ | 40.5 ¹ | **mlx-lm** |
| 64k⁴ | **30.2** | 27.9 | 30.3 ¹ | **mlx-lm ≈ oMLX** |
| 128k⁴ | **23.0** | 21.5 | 21.6 ¹ | **mlx-lm** |

¹ **oMLX decode recovered from server logs, not the harness.** At oMLX 0.5.1 the
`/v1/chat/completions` SSE stream batches multiple real tokens per delta event, so
the harness `decode_tps` undercounts by ~8–13× (it counts events/sec). True rates
were read from oMLX's own per-request log line (`256 tokens in {elapsed}s`) and
cross-validated against a manual non-streaming request. These are steady-state
(warm-prefix-cache) decode rates — same *kind* of number as the mlx-lm/tq columns
— but a **different measurement path**, and oMLX ran on **MLX 0.32.0** vs the
others' **0.31.2** (pulled-to-latest, per the run request). So oMLX's nominal lead
at 4k–32k is not a clean same-basis comparison; **not crowned**.

² 8k is a statistical tie: tq's median (43.8) edges mlx-lm (41.7) but inside the
overlapping measured spreads. tq is not systematically ahead — the pattern is
non-monotonic across sizes (§3.2), consistent with run-to-run variance between two
separately-launched servers, not a KV-quant decode effect.

³ tq 32k has the widest spread of the sweep (27.9–35.7, ~28 %); r2/r3 are slow
with no cause visible in the per-run records (RSS/TTFT unremarkable). Since this
fork never computes against quantized K/V (§3.1), this reads as system/thermal
noise, reported raw and unsmoothed.

⁴ **64k and 128k: n=1 measured** (single run after the discarded warm-up) — point
estimates, no range. Prefill-dominated and expensive; decode is stable across the
short-ctx n=4 runs, so the point estimates are trustworthy trend anchors, not
range-bounded comparisons.

### 2b. Prefill (cold, same-method siblings)

Cold first-request TTFT (`r0`, or the standalone 4k smoke — genuine uncached
prefill). Warm (post-r0) requests hit each backend's LRU prompt cache and return
cache-hit TTFTs (excluded). **oMLX prefill is excluded entirely** — at 0.5.1 it
streams a near-empty first SSE delta, so the harness `ttft_ms` measures
time-to-first-SSE-byte (~19–292 ms at every size), not real prefill (§3.4).

| Backend | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| mlx-lm (no KV) | 4.4 s / 932 t·s | 10.7 s / 767 | 20.6 s / 796 | 43.7 s / 751 | 109.9 s / 596 | **317.8 s / 412** |
| mlx-lm-tq (KV 8,4) | 4.4 s / 940 | 9.6 s / 853 | 19.8 s / 828 | 45.1 s / 726 | 121.5 s / 539 | 339.4 s / 386 |

The **128k cold prefill is the headline long-context cost** — ~318 s TTFT
(~412 tok/s) for mlx-lm, ~339 s for tq — independent of decode. tq is at/ahead of
plain through 16k and only 3–11 % slower at 32k–128k (contrast the 8B, where tq
prefill was 22–42 % slower — the 27B's 48 non-KV GDN layers dilute the quant-cache
boundary cost).

### 2c. Peak RSS (no paging on any backend)

| Backend | 4k | 128k | trajectory |
|---|---|---|---|
| mlx-lm | ~7.7 GB | ~7.8 GB | flat 7.65–7.76 GB across the sweep |
| mlx-lm-tq | ~7.7 GB | ~7.8 GB | flat 7.64–7.77 GB |
| oMLX | ~9.2 GB | ~15.4 GB | shallow ~1.7× climb, **no collapse** |

No backend paged. The 8B's oMLX paged-KV collapse (3.3→16.8 GB, decode → 4 TPS at
64k) **does not reproduce** — the GDN hybrid keeps KV growth shallow, so oMLX's
tiering never forces a paged cache. (oMLX separately grew a 20 GB on-disk SSD
prefix-cache tier under `~/.omlx/cache` — disk, not RAM; decode-neutral; see §5.)

---

## 3. Decode-efficiency analysis

**1. KV-8,4 is compute-neutral on this fork/arch — parity, not loss.** Two
mechanisms, both from code inspection of the turboquant fork:
   - `TextModel.make_cache()` returns `ArraysCache` for the 48 GDN layers and
     `KVCache` only for the 16 full-attn layers; `_maybe_quantize_cache()` only
     converts `isinstance(c, KVCache)` entries, so the 48 GDN recurrent-state
     layers are **mechanically excluded** from 8,4 quantization — no GDN-awareness
     needed, it falls out of the type dispatch.
   - In the single-serve path the quantized cache is used **only for LRU
     prompt-cache storage**: a fetched quantized cache is dequantized *before*
     `stream_generate`/`generate_step`, and quantization is applied once at
     request end (after the client stream closes) to compress before caching.
     **Neither prefill nor decode ever runs against quantized K/V.** So the 8B
     doc's "fake-asym dequant adds per-step compute" explanation does **not** hold
     for this fork's single-serve path; the observed decode gaps (§2a note ²) are
     run-to-run variance, not a quant compute penalty.

**2. The sibling decode spread is within noise.** tq vs mlx-lm: −3.7 % (4k),
**+5.1 %** (8k), −0.6 % (16k), −15.8 % (32k, the noisy outlier), −7.6 % (64k),
−6.6 % (128k) — non-monotonic, sign-changing, and (per point 1) not attributable
to a compute-path difference between the two. On a GDN hybrid with tiny 2-bit
weights and shallow KV, decode is neither weight- nor KV-bandwidth-starved, so
there is little headroom for any backend to separate.

**3. Flat decode is the architectural headline.** mlx-lm decode falls only ~2×
(45→23) across a 32× context range (4k→128k). Contrast the 8B full-attention
model (~4× over just 4k→64k). Only 16/64 layers grow KV; the 48 GDN layers hold
fixed-size state, so per-step attention cost grows far more slowly than a
full-attention model of comparable size.

**4. oMLX — no collapse, but three comparability caveats (why it is not crowned).**
   - *No paging collapse* (§2c) — genuinely different from the 8B; a real,
     positive result for oMLX on this arch.
   - *Decode measurement path differs.* Its true decode was recovered from
     server-log `256 tokens / elapsed` lines (the harness SSE count is an ~8–13×
     undercount at 0.5.1). This is a warm-cache steady-state decode rate, the same
     *kind* of number as mlx-lm/tq, but obtained differently and including a
     slightly different slice of per-request overhead.
   - *Version advantage.* oMLX was pulled to **MLX 0.32.0 / omlx 0.5.1**; mlx-lm
     and tq run on **MLX 0.31.2**. Its nominal +6…+13 % lead at 4k–32k is
     plausibly a kernel-version effect, not a backend-quality result.
   Given these, oMLX is reported but **mlx-lm (no-KV) is the honest same-method,
   same-MLX-version reference champion**, mirroring the 8B doc's treatment of
   oMLX as caveated / non-championable.

**5. MTP is inert everywhere.** The checkpoint declares `mtp_num_hidden_layers: 1`
but ships no `mtp.*` weights; mlx-lm strips them in `sanitize()`, oMLX logs
"config declares mtp heads but checkpoint ships no mtp.* weights; attachment
skipped." No speculative path engaged on any backend — decode is plain.

---

## 4. Skips — non-benched backends / cells

| Backend / cell | Reason |
|---|---|
| **ollama** | No Bonsai (ternary 2-bit affine) MLX tag; ollama is GGUF, weight-format-incompatible. Same skip as 8B. |
| **paroquant** / **isoquant** | Rotation-quant weight schemes, non-weight-comparable to affine 2-bit; no live OpenAI-compatible CBB endpoint for Bonsai. isoquant additionally has no `.venv`. Same pre-classified skips as 8B / Gemma4 / Qwen3.6. |
| **mlx-vlm** | This is a text-only bench (the 27B's vision tower is dropped by every loader); mlx-vlm is a VLM reference with no applicable text-decode path. |
| **256k cell** | The 262144 ceiling would allow it, but no 256k fixture exists and none was generated (the builder's targets top out at 131072). 128k is the top cell run. |
| **rMLX (all cells)** | This file is siblings-only; the rMLX full KV-quant matrix belongs in a companion `rMLX.md`. |

**Note on the 128k fixture (not a skip).** `longctx_128k.json` (~131,052 tokens
under the shared qwen3.6-family tokenizer, which is the same vocab family as
Bonsai-27B, so counts transfer within ~1 %) fits the 262144 ceiling with ample
headroom for the chat template + 256 generated tokens. It is the genuine
top-context cell, comparable to the other sizes.

---

## 5. Notes

### Machine and environment
- **Chip:** Apple M5 Max, 128 GB unified memory · **OS:** macOS 26.5.1 (Darwin 25.5.0) · **Date:** 2026-07-15.
- **Single-MLX discipline:** the three siblings ran **strictly serially**, one MLX
  process at a time (Metal is exclusive per process). Each run preflighted
  (`pkill -f "rmlx serve"; pkill -f mlx_lm; pkill -f paroquant; pkill -f omlx;
  rm -f /tmp/rmlx.*.claim`) and confirmed MLX released before the next started.
- **Pulled to latest** per backend before its run (SHAs below).

### Backend SHAs (post-pull, as run)
| Backend | SHA (used) | MLX | Notes |
|---|---|---|---|
| mlx-lm (stock) | `15b522f` (from `e476a22`) | 0.31.2 | `qwen3_5.py` unchanged by the pull; text tower only, vision+MTP dropped in `sanitize()`. **No-KV same-method reference champion.** |
| mlx-lm-turboquant | `67db9af` (already latest) | 0.31.2 | KV-8,4 via `--kv-cache-quantization 8,4 --quantized-kv-start 0`; quantizes only the 16 full-attn KV, and only for LRU storage (§3.1). |
| oMLX | `5a39ba3a` (from `6aee461`, 422-file pull) | **0.32.0** | omlx 0.5.1; routed through the **VLM batched engine** (VLM-shaped config), text-served correctly; decode recovered from server logs (§3.4). |
| ollama | — | — | no Bonsai MLX tag |

### Harness
- Generic driver `Cross-Backend-Bench/scripts/bench_longctx_gemma4.sh` (the
  `bench_longctx_bonsai.sh` wrapper is env-overridable but hard-codes an 8B header
  + 64k ceiling). Cells were driven via **direct `runners.run_one` calls** with
  `--request-timeout 1800` — the shared driver hard-codes run_one's 600 s default,
  too tight for the 27B's ~318 s cold 128k prefill.
- Prompt fixtures: the shared, content-addressed `prompts/longctx_{4k,8k,16k,32k,64k,128k}.json`
  (identical in `rMLX/` and `Cross-Backend-Bench/`), reused verbatim so cells are
  cross-backend and cross-model comparable. No prompts regenerated.
- Data sink: `Cross-Backend-Bench/metrics/runs/*.jsonl` (backend / model_id /
  quant_signature / decode_tps / ttft_ms / peak_rss_mb per run). As on the 8B
  campaign, these are **not** in rMLX `runs.db` — the CBB record shape is rejected
  by the rMLX metrics buffer (known landmine; the benign `recorder rejected record`
  warning appears on every `run_one` and is unrelated).

### Coherence gate
Every benched cell — all three backends, all six sizes — passed temp=0 greedy
coherence: non-empty, `success=true`, on-task, no repetition/degeneration.
Bonsai-27B is a thinking model; every completion begins with the identical
`"Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - **"` preamble
(chain-of-thought in the `reasoning` field). **Coherence holds at 128k** — the
model stays on-task at the top benched context under native 262144 rope.

### Caveats carried into the rMLX matrix / synthesis
- **64k/128k are n=1 measured** — point estimates; bump to n≥2 if a close call
  decides a champion.
- **oMLX decode is server-log-recovered + version-advantaged (MLX 0.32 vs 0.31.2)
  + measurement-path-different** — treat as caveated, do not crown. A clean
  same-method re-run would need a harness fix (`stream_options.include_usage` or
  non-streaming token counting) and, for a like-for-like MLX comparison, version
  alignment across siblings.
- **KV-8,4 is parity, not loss, on this arch** (unlike 8B) — and in this fork's
  single-serve path never touches live compute (§3.1). Read the tq column as
  "same weights, KV quant is decode-neutral here," not as a KV-codec speedup.
- **The mlx-lm decode reference** (~45 / 42 / 41 / 37 / 30 / 23 TPS at
  4k/8k/16k/32k/64k/128k) is the number the rMLX matrix will be measured against.
- **Followup (housekeeping):** the oMLX run left a ~20 GB SSD prefix-cache under
  `~/.omlx/cache` (reclaim with `rm -rf ~/.omlx/cache` if disk matters); and any
  future oMLX bench needs the SSE-token-count fix or its `decode_tps` will
  misreport by ~8–13×.
