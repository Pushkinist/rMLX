# Gemma4 — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable mxfp8 tier) = **mlx-lm (no KV quant)**, decode TPS:
> e2b 116/115/111/106/97 · e4b 75/71/66/58/46 · 26b 74/72/67/60/49/36 ·
> 31b 14.1/13.6/12.9/11.7/9.9/7.6 (4k/8k/16k/32k/64k[/128k]).

**Family:** `gemma4` (`Gemma4ForConditionalGeneration`, text core `gemma4_text`)
**Machine:** Apple M5 Max, 128 GB, macOS 26.4.1 · **Binary:** `release-perf`, rMLX
0.1.1 (`c308b24`)
**Protocol:** batch=1, temp=0, `max_tokens=256`; **n=3 measured** (4k/8k/16k/32k),
**n=2 → n=1 measured** (64k/128k), 1 warmup `r0` discarded; decode-TPS median +
95% CI (bootstrap). **Same harness as SIBLINGS** (CBB `run_one`, chat-templated
serve), so rMLX cells compare directly. Bar (§0.1): WIN / TIE-on-CI-overlap / LOSS.

> **Status: Stage 2 COMPLETE** (Phases B–E). rMLX serve cells use **per-prompt
> right-sized `--max-ctx`** — see the §M measurement note; this is essential for a
> fair number (rMLX pre-allocates the KV ring to `--max-ctx`).

## 0. TL;DR

- **rMLX LOSES decode to the mlx-lm champion on Gemma4 at almost every cell**,
  −5 % to −26 %, widening with context. The **only TIE is e2b @ 4k** (116.8 vs
  116.4). This is the **opposite of Qwen3.6** (where rMLX won +12–15 %).
- **KV quantization does NOT recover the gap** (Phase C). On e2b/e4b/26b the codec
  overhead ≈ the bandwidth saved; **only 31b gains** (k8vturbo2 **+4.1 %** @ 64k).
  So the deficit is **not KV-read-bandwidth** — it's the SWA-attention decode path
  (per-step kernel / snapshot cost), confirmed by KV quant being a no-op.
- **rMLX carries a much larger KV footprint than mlx-lm under Gemma4 SWA**
  (e2b @64k RSS **30.5 GB** vs mlx-lm **5.0 GB**; e4b @64k **40.8 GB** vs
  **7.7 GB**) — yet quantizing it doesn't speed decode, so the footprint is an
  allocation/capacity issue, not the decode bottleneck.
- **Speculative decoding is BROKEN on Gemma4** (Phase E) — 2 bugs: the
  `--draft-kind mtp` dispatch misroutes a plain-gemma4 draft to the Qwen3.5 MTP
  path, and the dedicated assistant drafter (only pairs with the **e2b** verifier)
  hits a Metal SDPA `Invalid mask_mode additive` crash → 0 decode. **File both.**
- **SSD tier** initializes cleanly + is decode-neutral (−2 % @64k, within noise),
  but **no spill is triggered** at 256-token single-stream (capacity feature, not
  exercised at these sizes).
- **Prefill/TTFT is brutal on the big models** (31b @128k > 600 s, times out;
  26b @128k ≈ 403 s) — a separate, large deficit from decode.

---

## M. Measurement note (READ FIRST — fairness)

**rMLX `serve` pre-allocates the KV ring to `--max-ctx`, and decode-step cost
scales with that capacity, not the filled length.** Benching a 4k prompt under
`--max-ctx 140000` penalized rMLX ~20–25 % (empty-ring bandwidth). Cross-check:
e2b @4k decode = **95 TPS** at max_ctx 140000 vs **119 TPS** (`rmlx baseline`,
auto-sized) vs **116.8** here (serve, max_ctx 6144). So every cell below launches
serve with `--max-ctx` right-sized to that prompt (4k→6144 … 64k→79872 →
128k→151552), matching the dynamic-KV siblings. Even so, rMLX's KV footprint stays
larger than mlx-lm's under SWA (§0) — that is a real characteristic, not a sizing
artifact, and tighter sizing would not flip the long-ctx losses.

---

## 1. rMLX snapshots benched

| Snapshot (basename) | Weight quant | Arch / size | Role | Resident (4k→64k) |
|---|---|---|---|---|
| `…gemma-4-e2b-it-mxfp8` | mxfp8 g32 b8 | dense, ~2B eff, SWA 512, kv-shared 20 | base | 5.6 → 30.5 GB |
| `…gemma-4-e4b-it-mxfp8` | mxfp8 g32 b8 | dense, ~4B eff, SWA 512, kv-shared 18 | base | 8.5 → 40.8 GB |
| `…gemma-4-26b-a4b-it-mxfp8` | mxfp8 g32 b8 | **MoE** 26B/~4B act, SWA 1024 | base | 26 → 35 GB |
| `…gemma-4-31b-it-mxfp8` | mxfp8 g32 b8 | dense 31B, SWA 1024 | base | 31 → 36 GB |
| `…gemma-4-E2B-it-assistant-bf16` | bf16 | assistant drafter (hidden 1536) | speculative (Phase E) | — |

---

## 2. rMLX full matrix

### 2a. Phase B — baseline (`--kv-quant none`)

decode TPS median [95% CI]; cold-`r0` prefill TTFT / tok·s; peak RSS.

**e2b** (ceiling 64k — 128k fixture 137,920 tok > 131072 ctx):

| Prompt | decode TPS | cold prefill | peak RSS |
|---|---|---|---|
| 4k | **116.8** [116.6–117.1] | 0.3 s / 14 847 | 5.6 GB |
| 8k | **110.2** [108.9–111.5] | 0.5 s / 15 005 | 6.2 GB |
| 16k | **106.3** [106.2–106.4] | 1.2 s / 13 955 | 8.3 GB |
| 32k | **96.7** [96.5–96.9] | 3.3 s / 10 019 | 16.7 GB |
| 64k¹ | **76.4** | 11.6 s / 5 658 | 30.5 GB |

**e4b** (ceiling 64k):

| Prompt | decode TPS | cold prefill | peak RSS |
|---|---|---|---|
| 4k | **71.2** [70.9–71.4] | 0.8 s / 5 335 | 8.5 GB |
| 8k | **65.5** [64.6–66.4] | 1.4 s / 5 672 | 8.9 GB |
| 16k | **60.9** [60.9–61.0] | 3.0 s / 5 475 | 10.6 GB |
| 32k | **52.3** [52.2–52.4] | 6.6 s / 4 935 | 17.3 GB |
| 64k¹ | **39.6** | 18.3 s / 3 577 | 40.8 GB |

**26b-a4b MoE** (ceiling 128k):

| Prompt | decode TPS | cold prefill | peak RSS |
|---|---|---|---|
| 4k | **66.6** [66.0–67.2] | 8.1 s / 505 | 26.0 GB |
| 8k | **64.6** [64.3–64.9] | 17.1 s / 478 | 26.3 GB |
| 16k | **61.9** [61.6–62.2] | 37.0 s / 442 | 27.1 GB |
| 32k | **53.0** [52.9–53.2] | 87.2 s / 376 | 30.6 GB |
| 64k¹ | **41.6** | 181.7 s / 361 | 35.1 GB |
| 128k¹ | **28.1** | 403.3 s / 325 | 34.7 GB |

**31b dense** (ceiling 128k):

| Prompt | decode TPS | cold prefill | peak RSS |
|---|---|---|---|
| 4k | **11.0** [11.0–11.1]² | 8.9 s / 461 | 31.3 GB |
| 8k | **12.5** [12.5–12.5]² | 17.0 s / 482 | 31.6 GB |
| 16k | **11.5** [11.5–11.5] | 38.8 s / 422 | 32.4 GB |
| 32k | **9.8** [9.8–9.9] | 90.0 s / 364 | 35.7 GB |
| 64k¹ | **8.0** | 228.2 s / 287 | 35.3 GB |
| 128k¹ | **5.6** (r1, prefix-cache) | r0 **TIMEOUT** (>600 s, ~138k tok prefill) | 34.2 GB |

¹ 64k/128k = **n=1 measured** (single run post-warmup) — point estimate, no CI.
² **31b 4k (11.0) < 8k (12.5)** — anomalous (decode should fall with ctx). Likely
thermal/warmup noise on the n=2 4k cell; the executor also flagged 31b kv-none 4k
as ~1 TPS below a historical `eed133c` reading (12.2) → a **possible minor 31b
dense regression** worth a separate check. Treat 31b 4k as soft.

All cells coherent (temp=0): e2b `"llama.cpp: Longest README content…"`, e4b
`"llama.cpp: Contains extensive feature descriptions…"`, 26b `"llama.cpp: longest
README provided…"`, 31b `"llama.cpp: 178 lines"`.

### 2b. Phase C — KV-variant sweep

**Ranking @ 8k on e4b** (representative dense SWA; n=3; none@8k = 65.5):

| KV variant | decode @8k | Δ vs none | coherence |
|---|---|---|---|
| k8vturbo3 | 67.3 | +2.7% | PASS |
| rotor4 | 67.1 | +2.5% | PASS |
| k8vturbo2 | 66.9 | +2.1% | PASS |
| k8v8 | 66.9 | +2.1% | PASS |
| k8v4 | 66.8 | +2.0% | PASS |
| planar3 | 66.6 | +1.7% | PASS |
| rotor3 | 66.6 | +1.7% | PASS |
| iso4 | 66.5 | +1.4% | PASS |
| iso3 | 66.4 | +1.4% | PASS |
| planar | 65.6 | +0.1% | PASS |
| none | 65.5 | — | PASS |

At 8k the KV cache is tiny (SWA window 512) so all codecs sit within +3% — **not
predictive** of the long-ctx outcome (same caveat as Qwen3.6). **No arch-guard
rejects** — none of these are in the MoE-guarded sub-8-bit-K set, so all ran on the
26b MoE too. Dense e2b/e4b/31b admit the full codec set; the guarded set
(`tsym3/4`, `iso*_sym`, `k_iso*`, `rotor*_sym`, `k_rotor*`, `rotor_k_*_asym`,
`planar_k`) was not swept (sub-8-bit-K; low value here).

**Carry to 64k** (the KV-dominated point; top-4 + k8v4/k8v8; vs none@64k):

| Model | none@64k | best KV variant | best @64k | Δ vs none | verdict |
|---|---|---|---|---|---|
| e2b | 76.4 | k8vturbo2 | 76.2 | −0.3% | none wins (noise) |
| e4b | 39.6 | k8vturbo3 | 39.6 | +0.0% | tie |
| 26b | 41.6 | rotor4 | 40.7 | −2.1% | none wins |
| **31b** | 8.0 | **k8vturbo2** | **8.33** | **+4.1%** | **KV wins** |

26b @128k: rotor4 = 27.7 vs none 28.1 (−1.6%) → none wins.

**Finding: KV quant does not recover rMLX's decode on Gemma4 — except 31b.** On the
small / MoE models the per-token codec dequant cost cancels the bandwidth saved
(decode isn't KV-read-bound there; see §0). **31b dense is the exception** — its
KV footprint at 64k is heavy enough that k8vturbo2's 2-bit-V compression nets
+4.1 % (n=2 — worth a 3-run confirm before adopting as the 31b long-ctx default).

### 2c. Phase D — SSD KV tier (off / on)

`--kv-ssd-cache-gb 8`. Two cells:

| Config | decode | Δ vs SSD-off | SSD activity |
|---|---|---|---|
| 31b k8vturbo2 @64k, SSD on | 8.17 / 8.16 | −2.0% | tier init OK (drain thread, layout_key); **no block spilled** |
| 26b none @128k, SSD on | 26.0 (r0 only) | −7.5% (single-run) | tier init OK; **no block spilled** |

**Finding:** SSD tier wires up cleanly (drain thread, `layout_key`, `index.db`) and
is **decode-neutral** (within noise), but **256-token single-stream decode never
overflows the ~25–30 GB RAM prompt-cache, so no spill/hydrate is exercised**. SSD
is a **capacity** feature; at these sizes it isn't triggered. Not stress-tested
here (would need a multi-turn / >RAM-KV scenario). _(The Qwen3.6 §9
hydrate-doesn't-skip-prefill gap is untested on Gemma4 for the same reason.)_

### 2d. Phase E — speculative (Gemma4 assistant MTP) — **BROKEN**

Every speculative cell failed. Three distinct failures (verifier @4k, kv none):

| Pairing | result |
|---|---|
| 31b + `e2b-mxfp8`, `--draft-kind mtp` | **HTTP 500** — misroutes to the Qwen3.5 MTP path (checks `model_type=="gemma4_assistant"`; plain `gemma4` falls through) → `text_config missing num_experts`. |
| e4b / 31b + `assistant-bf16`, mtp | **HTTP 500** — `backbone_hidden_size 1536 != verifier hidden 2560 / 5376`. The on-disk assistant was built for the **e2b** verifier only. |
| **e2b** + `assistant-bf16`, mtp (correct pair) | **Loads**, enters `mtp_assistant_generate_greedy`, then **Metal crash**: `scaled_dot_product_attention: Invalid mask_mode additive` → TTFT ~220 ms, **0 decode tokens**. |

**Two bugs to file:**
1. **Spec dispatch** doesn't recognize a plain-`gemma4` draft for the two-model
   path — `--draft-kind mtp` routes any non-`gemma4_assistant` draft to the
   Qwen3.5 MTP sidecar. The docs' "31B verifier + E2B draft" example is not
   reachable through the current flag. (`speculative` dispatch.)
2. **`Gemma4AssistantDrafter` SWA mask** uses `mask_mode = additive`, which the
   mlx-c Metal SDPA kernel rejects (`fast.cpp:629`: must be `causal`/`array`/`''`).
   The only valid drafter pairing (e2b + assistant) therefore crashes on every
   decode step. (`speculative/gemma4_assistant.rs` SWA-mask construction.)

**Phase E verdict:** Gemma4 speculative is **non-functional** in rMLX 0.1.1 — no
accept-rate or speedup measurable. (Mirrors the Qwen3.6 Eagle3-crash situation pre-
0.1.1; same "file + fix later" path.)

---

## 3. Standing vs champion (decode)

rMLX best (any KV) vs the SIBLINGS mlx-lm champion. WIN / TIE-on-CI / LOSS (§0.1).

### e2b
| Prompt | rMLX best | champion (mlx-lm) | standing |
|---|---|---|---|
| 4k | 116.8 [116.6–117.1] | 116.4 | 🟡 **TIE** (CI overlaps) |
| 8k | 110.2 | 114.7 | 🔴 LOSS −4% |
| 16k | 106.3 | 110.7 | 🔴 LOSS −4% |
| 32k | 96.7 | 106.1 | 🔴 LOSS −9% |
| 64k | 76.4 | 97.4 | 🔴 LOSS −22% |

### e4b
| Prompt | rMLX best | champion | standing |
|---|---|---|---|
| 4k | 71.2 | 74.6 | 🔴 LOSS −5% |
| 8k | 65.5 | 71.3 | 🔴 LOSS −8% |
| 16k | 60.9 | 66.1 | 🔴 LOSS −8% |
| 32k | 52.3 | 58.1 | 🔴 LOSS −10% |
| 64k | 39.6 | 45.9 | 🔴 LOSS −14% |

### 26b-a4b
| Prompt | rMLX best | champion | standing |
|---|---|---|---|
| 4k | 66.6 | 74.1 | 🔴 LOSS −10% |
| 8k | 64.6 | 71.5 | 🔴 LOSS −10% |
| 16k | 61.9 | 67.2 | 🔴 LOSS −8% |
| 32k | 53.0 | 59.5 | 🔴 LOSS −11% |
| 64k | 41.6 | 49.2 | 🔴 LOSS −15% |
| 128k | 28.1 | 35.8 | 🔴 LOSS −22% |

### 31b
| Prompt | rMLX best | champion | standing |
|---|---|---|---|
| 4k | 11.0² | 14.1 | 🔴 LOSS −22% (4k soft, §2a²) |
| 8k | 12.5 | 13.6 | 🔴 LOSS −8% |
| 16k | 11.5 | 12.9 | 🔴 LOSS −11% |
| 32k | 9.8 | 11.7 | 🔴 LOSS −16% |
| 64k | 8.33 (k8vturbo2) | 9.9 | 🔴 LOSS −16% |
| 128k | 5.6 | 7.6 | 🔴 LOSS −26% |

> **Verdict: rMLX trails the mlx-lm champion on Gemma4 decode everywhere except
> e2b @4k (TIE).** The loss widens with context. Note the champion comparison uses
> **decode only**; on prefill rMLX is also far behind (§2a TTFT). Unlike Qwen3.6
> (MoE, no SWA — rMLX won +12–15 %), Gemma4's interleaved SWA exposes a weakness in
> rMLX's sliding-window attention decode path.

---

## 4. Gaps & hypotheses (Phase F synthesis → improvement plan)

Ranked by impact:

1. **SWA-attention decode path — the primary deficit.** rMLX loses 5–26 % on
   Gemma4 decode, widening with context, and **KV quant does NOT recover it**
   (Phase C) → the cost is **not KV-read-bandwidth** but the per-step
   sliding-window attention compute (mask construction, ring snapshot/restore, or
   attention-over-window kernel). The widening-with-ctx slope is the signature.
   Profile the Gemma4 decode attention vs mlx-lm's SWA path. **Highest value.**
2. **KV footprint under SWA.** rMLX resident KV is ~6× mlx-lm at 64k (e2b 30.5 vs
   5.0 GB; e4b 40.8 vs 7.7 GB) — mlx-lm shrinks KV to the SWA window for
   windowed layers; rMLX appears to allocate full-ctx KV per layer. It doesn't
   gate decode speed directly (KV quant was a no-op) but it caps deployable context
   and feeds (1). Worth shrinking KV allocation to the window for SWA layers.
3. **Speculative is broken (Phase E, 2 bugs).** (a) spec dispatch misroutes a
   plain-gemma4 draft to the Qwen3.5 MTP path; (b) the assistant drafter's
   `additive` SWA mask crashes the Metal SDPA kernel. Fix both → the 31b verifier
   (11 TPS) + e2b/assistant draft is exactly where spec should pay off. File as
   bugs.
4. **Prefill/TTFT on the big models.** 31b @128k prefill > 600 s (times out);
   26b @128k ≈ 403 s; prefill tok/s sinks to ~300 at long ctx. Same prefill class
   the Qwen3.6 campaign flagged — separate from decode, large.
5. **Possible 31b dense regression** — kv-none 4k 11.0 is below a historical 12.2
   (`eed133c`); 31b 4k < 8k is non-physical. Confirm with a clean n≥3 4k re-run
   and bisect if real.

**The one bright spot:** **31b + k8vturbo2 (+4.1 % @64k)** — the only KV-quant win,
the rotation/turbo-V codec rMLX uniquely ships paying off on the most
KV-bandwidth-pressured model. Everything else trails. Coherence was solid across
all KV variants and all baseline cells.

---

## 5. Caveats

- **64k/128k are n=1 measured** (single post-warmup run) — point estimates.
- **31b 4k is soft** (anomalous vs 8k; possible regression — §2a²).
- **SSD not stress-tested** — no spill triggered at 256-token single-stream.
- **Speculative unmeasured** — broken (Phase E); no accept-rate / speedup exists.
- **e2b/e4b 128k absent** — fixture (137,920 gemma tok) exceeds 131072 ctx.
- rMLX cells recorded in CBB `metrics/runs/*.jsonl` (backend=rmlx); **not** in
  `runs.db` (CBB schema rejected by the rMLX buffer — known harness landmine).
- Aggregator: `Cross-Backend-Bench/scripts/agg_gemma4_siblings.py`.
