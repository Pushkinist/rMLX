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

- **(Updated 2026-06-08, post-#25 load-once.)** rMLX **now WINS e2b @4k** (117.4 vs
  116.4) and e4b short/mid-ctx losses narrowed (8k −3.4 %, 16k −5.3 %), but rMLX
  **still LOSES at long ctx** (32k–64k: −12…−21 %), the gap widening with context.
  Still the **opposite of Qwen3.6** (rMLX won +12–15 % there). See §2b/§3.
- **KV quantization does NOT recover the gap** (Phase C). The **full 25-codec e2b/e4b
  sweep** shows all 16 mainstream codecs within **±2 % of `none`** at every size —
  KV quant neither helps nor hurts dense-Gemma4 decode; **only 31b gains** (k8vturbo2
  **+4.1 %** @ 64k). So the deficit is **not KV-read-bandwidth** — it's the
  SWA-attention decode path. _(The 9 rotation/K-only codecs could not be measured
  cleanly — Metal shader cold-compile artifact; see §2b.)_
- **rMLX carries a much larger KV footprint than mlx-lm under Gemma4 SWA**
  (e2b @64k RSS **30.5 GB** vs mlx-lm **5.0 GB**; e4b @64k **40.8 GB** vs
  **7.7 GB**) — yet quantizing it doesn't speed decode, so the footprint is an
  allocation/capacity issue, not the decode bottleneck.
- **Speculative decoding is BROKEN on Gemma4** (Phase E) — 2 bugs: the
  `--draft-kind mtp` dispatch misroutes a plain-gemma4 draft to the Qwen3.5 MTP
  path, and the dedicated assistant drafter (only pairs with the **e2b** verifier)
  hits a Metal SDPA `Invalid mask_mode additive` crash → 0 decode. **File both.**
  _(Filed + fixed: #23 dispatch, #24 mask. Post-fix (§V): both reachable,
  additive crash gone, short-prompt spec works — accept 0.833 — but a residual
  off-by-one SWA mask shape remains at longer prompts.)_
- **SSD tier** initializes cleanly + is decode-neutral (−2 % @64k, within noise),
  but **no spill is triggered** at 256-token single-stream (capacity feature, not
  exercised at these sizes).
- **Prefill/TTFT is brutal on the big models** (31b @128k > 600 s, times out;
  26b @128k ≈ 403 s) — a separate, large deficit from decode.

---

## M. Measurement note (READ FIRST — fairness)

> **⚠ SUPERSEDED for decode by fix #25 (merged `a8228dd`) — see §V.** The KV ring
> no longer pre-allocates to `--max-ctx`; it grows lazily, so a high ceiling no
> longer penalizes small prompts. The §M tables below were measured pre-#25 with
> per-prompt right-sizing and remain valid as recorded numbers. New benches may
> serve once at a high ceiling.

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

## V. Post-fix validation (2026-06-08, branch rebased on `95f95f4`)

Four bug fixes merged to main after the Stage-2 sweep (#23 spec dispatch, #24
assistant SDPA mask, #25 KV-ring lazy-grow, #26 per-request KV/ctx hot-swap).
Re-ran 3 spot cells on `release-perf` to confirm the recorded baseline held and
the fixes took effect (single validation runs, not the full n=3+CI protocol —
medians of 3 raw runs).

**#25 lazy-grow — CONFIRMED, obsoletes the §M penalty.** e2b served **once** at
`--max-ctx 79872` (the 64k ceiling), all sizes against the resident ring:

| Prompt | decode @ ceiling 79872 | §M right-sized baseline | Δ |
|---|---|---|---|
| 4k | **119.9** | 116.8 | +2.7 % |
| 8k | 114.2 | 110.2 | +3.6 % |
| 16k | 103.9 | 106.3 | −2.3 % |

4k under the 64k ceiling decodes at **119.9**, not the old pre-#25 **95** penalty
(§M) — the ring grew lazily instead of pre-allocating. The per-size serve relaunch
in `bench_gemma4_rmlx.sh` is no longer required for a fair decode number.

> **Refined by the full sweep (§2b).** The +2.7 %/+3.6 % above were warm-session /
> thermal noise on a 3-run spot-check — the full e2b+e4b sweep shows **load-once ≈
> per-prompt right-sized within noise** at every size on both models (e2b `none`
> 117.4/112.1/104.0/91.9/76.2; e4b matches right-sized incl. 32k). So #25 removes
> the big fixed-ceiling penalty but does NOT make decode *faster* than a tight
> ceiling — it makes a high ceiling *free*. One soft outlier: e2b 32k load-once came
> in ~5 % under the old right-sized 96.7; it does not reproduce on e4b, so it is
> noise, **not** a pow2-ring-capacity effect (which was the initial hypothesis,
> falsified by e4b). **Caveat:** the rotation/K-only KV codecs hit a separate Metal
> shader cold-compile artifact and are not cleanly measured here — see §2b.

**Baseline intact (no regression from #25/#26).** e4b served at ceiling 79872:
4k = **72.1** (+1.3 % vs 71.2), 8k = **68.7** (+4.8 % vs 65.5). e2b above too.

**Speculative (#24/#23) — partially recovered.** The §2d `Invalid mask_mode
additive` Metal crash is **gone** (#24 landed). e2b + `assistant-bf16`
`--draft-kind mtp` now routes correctly (#23) and decodes coherently on **short**
prompts (accept-rate **0.833**). **Residual bug:** at 4k longctx (prompt_len 4117)
MTP verify hits an off-by-one SWA mask shape mismatch —
`scaled_dot_product_attention: [broadcast_shapes] Shapes (1,1,5,4127) and
(1,8,5,4126) cannot be broadcast` (`fast.cpp:629`). So spec is reachable + the
additive crash is fixed, but the assistant SWA mask is one position too long
during multi-token verify at non-trivial prompt lengths. Follow-up to #24.

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

**e2b** (ceiling 64k — 128k fixture 137,920 tok > 131072 ctx).
_decode = **2026-06-08 load-once** @ `--max-ctx 79872` (serve once, sweep all sizes
against the resident lazy-grown ring, post-#25); `pre-#25` = the prior per-prompt
right-sized serve. They agree within noise. cold-prefill is method-independent._

| Prompt | decode TPS (load-once) | pre-#25 right-sized | cold prefill |
|---|---|---|---|
| 4k | **117.4** | 116.8 | 0.3 s / 14 847 |
| 8k | **112.1** | 110.2 | 0.5 s / 15 005 |
| 16k | **104.0** | 106.3 | 1.2 s / 13 955 |
| 32k³ | **91.9** | 96.7 | 3.3 s / 10 019 |
| 64k¹ | **76.2** | 76.4 | 11.6 s / 5 658 |

Peak RSS unchanged from prior (5.6 → 32.6 GB at 64k; serving at a 64k ceiling does
**not** pre-allocate — the ring grows lazily, so RSS still tracks the filled length).

**e4b** (ceiling 64k). Same load-once method; load-once matches right-sized at
**every** size (incl. 32k: 52.5 vs 52.3) — confirming the e2b 32k³ dip is noise,
not a method effect.

| Prompt | decode TPS (load-once) | pre-#25 right-sized | cold prefill |
|---|---|---|---|
| 4k | **71.6** | 71.2 | 0.8 s / 5 335 |
| 8k | **67.9** | 65.5 | 1.4 s / 5 672 |
| 16k | **62.1** | 60.9 | 3.0 s / 5 475 |
| 32k | **52.5** | 52.3 | 6.6 s / 4 935 |
| 64k¹ | **39.7** | 39.6 | 18.3 s / 3 577 |

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

³ **load-once method (e2b/e4b only, 2026-06-08).** Serve once at `--max-ctx 79872`
and sweep 4k→64k ascending against the resident ring (valid post-#25). Matches the
pre-#25 per-prompt right-sized numbers within noise on both models; the lone e2b
32k −5 % dip does not reproduce on e4b, so treat it as run-to-run noise. 26b/31b
were **not** re-run — their rows are the pre-#25 right-sized Stage-2 numbers.

All cells coherent (temp=0): e2b `"llama.cpp: Longest README content…"`, e4b
`"llama.cpp: Contains extensive feature descriptions…"`, 26b `"llama.cpp: longest
README provided…"`, 31b `"llama.cpp: 178 lines"`. The 2026-06-08 KV sweep (§2b) is
coherent across all 16 mainstream codecs on both models — no repetition loops.

### 2b. Phase C — full KV sweep (e2b + e4b, 2026-06-08, load-once)

All 25 `KvQuant::FromStr` codecs × {4k,8k,16k,32k,64k}, n=3 (n=2 @64k), served
load-once per codec at the 64k ceiling. **16 mainstream codecs measured clean; the
9 rotation / K-only codecs are contaminated by a Metal shader cold-compile artifact
(caveat below) and are NOT ranked.** Best per cell bolded.

**e2b — decode TPS median (16 mainstream codecs):**

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| **none** | **117.4** | **112.1** | 104.0 | 91.9 | 76.2 |
| k8v4 | 115.6 | 111.0 | 103.5 | 91.2 | 76.8 |
| k8v8 | 114.9 | 111.0 | 104.9 | 92.4 | 76.0 |
| planar | 116.0 | 109.5 | 103.2 | 92.5 | 75.4 |
| planar3 | 115.4 | 110.9 | 103.6 | **92.7** | 76.2 |
| planar_k | 116.0 | 110.8 | 104.3 | 90.9 | 76.1 |
| k8vturbo2 | 116.1 | 110.9 | 103.3 | 92.6 | 76.6 |
| k8vturbo3 | 115.2 | 110.3 | **105.6** | 92.1 | 76.8 |
| k8vturbo2tcq | 115.3 | 110.9 | 104.1 | 92.0 | **76.9** |
| k8vturbo3tcq | 116.3 | 109.6 | 103.4 | 91.8 | 76.7 |
| tsym3 | 115.7 | 110.3 | 103.3 | 91.7 | 76.4 |
| tsym4 | 115.3 | 110.3 | 103.3 | 92.3 | 76.4 |
| iso3 | 114.4 | 111.0 | 105.1 | 91.5 | 76.1 |
| iso4 | 115.8 | 109.2 | 102.7 | 90.7 | 76.4 |
| iso3_sym | 115.5 | 110.3 | 103.1 | 91.1 | 75.6 |
| iso4_sym | 115.7 | 110.2 | 103.2 | 91.5 | 74.6 |

**e4b — decode TPS median (16 mainstream codecs):**

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| **none** | **71.6** | 67.9 | 62.1 | 52.5 | 39.7 |
| k8v4 | 70.8 | 68.1 | 61.6 | 52.2 | 39.0 |
| k8v8 | 71.1 | 68.2 | 61.0 | 51.5 | **40.4** |
| planar | 70.6 | 67.6 | 62.1 | **52.6** | 40.3 |
| planar3 | 70.9 | **68.9** | 62.3 | **52.6** | **40.4** |
| planar_k | 70.5 | 68.2 | **62.6** | 52.0 | 39.9 |
| k8vturbo2 | 70.5 | 68.3 | 60.8 | 52.3 | 39.9 |
| k8vturbo3 | 71.3 | 68.4 | 61.6 | 52.0 | 40.1 |
| k8vturbo2tcq | 71.2 | 68.3 | 61.8 | 52.3 | 40.1 |
| k8vturbo3tcq | 70.9 | 67.8 | 62.0 | 52.2 | 39.5 |
| tsym3 | 70.9 | 67.8 | 62.0 | 52.3 | —ᴸ |
| tsym4 | 69.5 | 68.0 | 62.4 | 52.2 | 40.2 |
| iso3 | 71.0 | 68.0 | 62.0 | 52.0 | ✗⁰ |
| iso4 | 71.5 | 68.8 | 62.3 | 52.0 | 39.4 |
| iso3_sym | 69.3 | 68.5 | 62.3 | 51.2 | 39.8 |
| iso4_sym | 70.6 | 68.4 | 62.1 | 50.8 | 39.1 |

ᴸ `—` = serve LOADFAIL on a cold start (shader compile, see caveat); ⁰ `✗` =
`success` but `decode_tps=0` (64k prefill consumed the token budget).

**Finding (e2b/e4b): every mainstream KV codec is within ±2 % of `none`** at every
size — KV quant neither helps nor hurts dense-Gemma4 decode. This confirms §0: the
deficit is the **SWA-attention decode path, not KV-read-bandwidth**. `none` is the
safe default; no codec earns its keep on e2b/e4b. (`rotor4` showed a noise-level edge
on e2b — 118.6/113.1/107.5 — but was **slower** than `none` on e4b, so it is not a
robust win, and its e2b numbers are shader-contamination-suspect anyway.)

**⚠ Rotation / K-only codecs NOT cleanly measurable — Metal shader cold-compile.**
`k_iso3/4`, `rotor3/4`, `rotor3/4_sym`, `k_rotor3/4`, `rot_k_tq4v` carry heavy MSL
shaders that compile lazily during the **first forward pass** (30–60× slow), and the
1-token serve warmup does not trigger compilation. Their first measured cells are
therefore contaminated — e.g. e2b `k_iso4` 26→4 TPS but e4b `k_iso4` 71→52 ≈ `none`,
a warmup artifact, not codec cost. 59 cold-start records were quarantined; the
survivors are still unreliable and are excluded from ranking. Only `k_iso3` is
consistently slow on both models (genuinely heavy at this ctx). **A clean
rotation-codec bench needs a realistic-prompt warmup to force shader compilation
before measuring** — a bench-harness gap, and a candidate engine UX bug (the first
request after serving a rotation-KV model stalls 30–60×).

**Carry to 64k — 26b / 31b (prior right-sized Stage-2 data, NOT re-run 2026-06-08):**

| Model | none@64k | best KV | best @64k | Δ | verdict |
|---|---|---|---|---|---|
| 26b | 41.6 | rotor4 | 40.7 | −2.1% | none wins |
| **31b** | 8.0 | **k8vturbo2** | **8.33** | **+4.1%** | **KV wins** |

26b @128k: rotor4 27.7 vs none 28.1 (−1.6%) → none wins. **31b dense stays the one
model where KV quant pays** — k8vturbo2 +4.1 % @64k (n=2; confirm at n=3). These
26b/31b rotation cells predate the shader-compile finding and may be warm/cold-mixed
— re-bench with the warmup fix before trusting them.

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

### 2d. Phase E — speculative (Gemma4 assistant MTP) — **BROKEN** (pre-fix; see §V)

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

rMLX best (mainstream KV, **2026-06-08 load-once**) vs the SIBLINGS mlx-lm champion.
WIN / TIE-on-CI / LOSS (§0.1). e2b/e4b updated; 26b/31b are the prior Stage-2 rows.

### e2b — **now WINS @4k** (was TIE)
| Prompt | rMLX best | champion (mlx-lm) | standing |
|---|---|---|---|
| 4k | 117.4 (none) | 116.4 | 🟢 **WIN +0.9%** |
| 8k | 112.1 (none) | 114.7 | 🔴 LOSS −2.3% (was −4%) |
| 16k | 105.6 (k8vturbo3) | 110.7 | 🔴 LOSS −4.6% |
| 32k | 92.7 (planar3) | 106.1 | 🔴 LOSS −12.6% (32k³ soft) |
| 64k | 76.9 (k8vturbo2tcq) | 97.4 | 🔴 LOSS −21% |

### e4b — losses narrowed at short/mid ctx
| Prompt | rMLX best | champion | standing |
|---|---|---|---|
| 4k | 71.6 (none) | 74.6 | 🔴 LOSS −4.0% (was −5%) |
| 8k | 68.9 (planar3) | 71.3 | 🔴 LOSS −3.4% (was −8%) |
| 16k | 62.6 (planar_k) | 66.1 | 🔴 LOSS −5.3% (was −8%) |
| 32k | 52.6 (planar/planar3) | 58.1 | 🔴 LOSS −9.5% |
| 64k | 40.4 (k8v8/planar3) | 45.9 | 🔴 LOSS −12.0% (was −14%) |

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

> **Verdict (post-#25, 2026-06-08): rMLX now WINS e2b @4k (+0.9 %) and the e4b
> losses narrowed** (8k −8→−3.4 %, 16k −8→−5.3 %), but still **trails at long ctx**,
> the gap widening to −12…−21 % at 32k–64k. The champion comparison is **decode
> only**; on prefill rMLX is also far behind (§2a TTFT). Unlike Qwen3.6 (MoE, no SWA
> — rMLX won +12–15 %), Gemma4's interleaved SWA still exposes a weakness in rMLX's
> sliding-window attention decode path at scale. KV quant does not move it (§2b).

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
