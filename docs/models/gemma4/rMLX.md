# Gemma4 — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable mxfp8 tier) = **mlx-lm (no KV quant)**, decode TPS:
> e2b 116/115/111/106/97 · e4b 75/71/66/58/46 · 26b 74/72/67/60/49/36 ·
> 31b 14.1/13.6/12.9/11.7/9.9/7.6 (4k/8k/16k/32k/64k[/128k]).

**Family:** `gemma4` (`Gemma4ForConditionalGeneration`, text core `gemma4_text`)
**Machine:** Apple M5 Max, 128 GB, macOS 26.4.1 · **Binary:** `release-perf`,
post-fix campaign (`bench/gemma4-siblings` on main `4da2000`; #32/#33/#34/#35/#36/#39 merged)
**Protocol:** batch=1, temp=0, `max_tokens=256`; **n=3 measured** (4k/8k/16k/32k),
**n=2 → n=1 measured** (64k/128k), 1 warmup `r0` discarded; decode-TPS median +
95% CI (bootstrap). **Same harness as SIBLINGS** (CBB `run_one`, chat-templated
serve), so rMLX cells compare directly. Bar (§0.1): WIN / TIE-on-CI-overlap / LOSS.

> **Status: Stage 2 COMPLETE + post-fix dual-axis re-bench (2026-06-08).** Post-#25
> the KV ring grows lazily — serve once at a high ceiling, no per-prompt right-sizing
> needed (§M/§V). This sweep adds the **`kv_cache_bytes`** axis (#33/#39).

## 0. TL;DR

- **(Updated 2026-06-08, post-fix dual-axis sweep — 6 fixes merged: #32/#33/#34/#35/#36/#39.)**
  Decode standing is **unchanged within noise** — the fixes were KV-reporting,
  speculative, and codec-classification work; they do **not** touch the decode
  kernel. e2b **TIE @4k** (116.6 vs 116.4), losses **widen with ctx** to −21 % @64k;
  e4b **−4…−12 %** (4k→64k). The **SWA-attention decode path remains the deficit** —
  opposite of Qwen3.6 (MoE, no SWA — rMLX won +12–15 %). See §2b/§3.
- **NEW — KV-cache-SIZE axis (now measurable: #33/#39 give accurate `kv_cache_bytes`).**
  The headline: on Gemma4 SWA, **KV "quant" mostly _inflates_ resident KV, not
  shrinks it.** A quantized layer keeps a bf16 warm-TTFT decode seed _alongside_ the
  quantized blocks + scales, so vs `none` @64k: k8v4 **1.20×**, k8v8 1.25×,
  planar/rotor ~1.46×, iso 1.86×, **iso_sym 2.47× (worst)**. **Only K-only rotor
  compresses K (0.87×)** — but it is CPU-bound and decode-uncompetitive. So KV quant
  buys **neither decode nor memory** on dense Gemma4; `none` is the right default.
  The per-layer net-benefit `warn!` (#34) now flags these net-negative configs at
  resolve time.
- **KV-footprint claim CORRECTED.** The earlier "rMLX carries ~6× mlx-lm KV under
  SWA (30.5 GB)" was an artifact: windowed SWA layers are **already window-bounded**
  (flat ring, #35), and `kv_cache_bytes` was reading the **ceiling-sized** buffer
  (#39, now filled-prefix). True live-inference KV @64k (`none`): **e2b 780 MB /
  e4b 2088 MB**. _Note:_ the None-path global decode seed is **f32, not bf16** — a
  real ~2× allocation-reduction candidate, tracked as **#44** (numerics-gated, not done).
- **Rotation/K-only codecs now CLEANLY MEASURABLE** (#36 load-time MSL precompile —
  the prior "shader cold-compile" contamination is **gone**: r0≈r1≈r2 for every
  rotation codec, no LOADFAILs). Honest verdict: **iso/rotor V-only** fire a
  CPU-hot-path `warn!` (prefill encode is CPU) but **decode ≈ `none`** (decode reads
  the bf16 seed); **k_iso / k_rotor (K-only)** are **genuinely CPU-bound** — decode
  craters (e2b k_iso3 52→3 TPS across 4k→64k) and is not competitive.
- **Speculative decoding is FIXED on Gemma4** (#32, was broken). The verify-step SWA
  mask off-by-one — in **both** the producer and consumer attention branches — is
  resolved; e2b + assistant-bf16 `--draft-kind mtp` now runs a 4.6k-token prompt with
  a partial-accept round, **no SDPA broadcast crash** (§V). #23/#24 earlier fixed the
  dispatch + additive-mask crash; #32 closes the residual.
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
_decode = **2026-06-08 post-fix load-once** @ `--max-ctx 79872` (serve once, sweep
all sizes against the resident lazy-grown ring); `kv_cache_bytes` = live-inference
KV (filled-prefix, #39-accurate), from `rmlx baseline`, in MB. cold-prefill is
method-independent. Decode matches the pre-fix sweep within ±1 % (no regression)._

| Prompt | decode TPS (`none`) | kv_cache_bytes (`none`) | cold prefill |
|---|---|---|---|
| 4k | **116.6** | 64 MB | 0.3 s / 14 847 |
| 8k | **110.9** | 115 MB | 0.5 s / 15 005 |
| 16k | **104.4** | 229 MB | 1.2 s / 13 955 |
| 32k | **92.1** | 440 MB | 3.3 s / 10 019 |
| 64k¹ | **75.5** | 780 MB | 11.6 s / 5 658 |

`kv_cache_bytes` is the **live-inference** KV (12 windowed SWA layers flat at the
window + 3 global full-attention layers growing with ctx); the global seed is f32
(#44). Peak RSS @64k ≈ 32.6 GB (weights + activations + Metal, KV is a small part).

**e4b** (ceiling 64k). Same load-once method; decode within ±1 % of the pre-fix sweep.

| Prompt | decode TPS (`none`) | kv_cache_bytes (`none`) | cold prefill |
|---|---|---|---|
| 4k | **70.7** | 178 MB | 0.8 s / 5 335 |
| 8k | **68.2** | 314 MB | 1.4 s / 5 672 |
| 16k | **61.8** | 618 MB | 3.0 s / 5 475 |
| 32k | **52.5** | 1181 MB | 6.6 s / 4 935 |
| 64k¹ | **39.9** | 2088 MB | 18.3 s / 3 577 |

**26b-a4b MoE** (ceiling 128k) — **not re-run in the 2026-06-08 post-fix sweep**
(e2b/e4b only); rows are the prior Stage-2 right-sized numbers, decode-only:

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

**Load-once method (e2b/e4b, 2026-06-08).** Serve once at `--max-ctx 79872` and
sweep 4k→64k ascending against the resident lazy-grown ring (valid post-#25). Matches
per-prompt right-sized within noise on both models. 26b/31b were **not** re-run —
their rows are the prior right-sized Stage-2 numbers.

All cells coherent (temp=0): e2b `"llama.cpp: Longest README content…"`, e4b
`"llama.cpp: Contains extensive feature descriptions…"`, 26b `"llama.cpp: longest
README provided…"`, 31b `"llama.cpp: 178 lines"`. The 2026-06-08 KV sweep (§2b) is
coherent across **all 25 codecs** on both models — no repetition loops.

### 2b. Phase C — full dual-axis KV sweep (e2b + e4b, 2026-06-08 post-fix)

All 25 `KvQuant::FromStr` codecs × {4k,8k,16k,32k,64k}, n=3 (n=2 @64k). **Two axes:**
decode TPS (serve+`run_one`, load-once at the 64k ceiling) **and `kv_cache_bytes`**
(`rmlx baseline`, filled-prefix, #39-accurate). **Rotation codecs are now cleanly
measured** — #36's load-time MSL precompile killed the prior cold-compile
contamination (r0≈r1≈r2 every codec, no LOADFAILs). Best mainstream per cell bolded.

**Decode — every mainstream codec is within ±2 % of `none` (decode no-op), both models.**

**e2b — decode TPS median (16 mainstream):**

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| **none** | **116.6** | 110.9 | 104.4 | 92.1 | 75.5 |
| k8v4 | 116.3 | 110.6 | **104.9** | 92.2 | 76.0 |
| k8v8 | 115.2 | **111.1** | **104.9** | 90.9 | 76.4 |
| planar | 116.0 | 109.5 | 103.9 | 92.3 | 75.1 |
| planar3 | 115.4 | 110.7 | 104.4 | **92.6** | 76.3 |
| planar_k | 115.7 | 111.0 | 103.6 | 91.2 | 76.3 |
| k8vturbo2 | 116.0 | **111.1** | 104.2 | 92.5 | 76.1 |
| k8vturbo3 | 115.0 | 110.2 | 104.8 | 92.0 | 76.5 |
| k8vturbo2tcq | 115.3 | 110.9 | 104.0 | 92.2 | 76.5 |
| k8vturbo3tcq | 116.3 | 110.4 | 103.6 | 91.5 | **76.6** |
| tsym3 | 116.1 | 110.3 | 103.3 | 91.9 | **76.6** |
| tsym4 | 114.6 | 111.0 | 103.8 | 92.4 | 76.5 |
| iso3ᵂ | 114.9 | 110.2 | 104.4 | 91.2 | 75.9 |
| iso4ᵂ | 115.4 | 109.2 | 103.7 | 90.7 | 75.5 |
| iso3_symᵂ | 116.4 | 110.0 | 102.9 | 91.1 | 75.5 |
| iso4_symᵂ | 115.7 | 110.1 | 103.2 | 90.5 | 74.9 |

**e4b — decode TPS median (16 mainstream):**

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| **none** | 70.7 | 68.2 | 61.8 | 52.5 | 39.9 |
| k8v4 | 70.8 | 67.7 | 61.8 | 52.3 | 39.4 |
| k8v8 | **71.3** | 68.2 | 62.4 | 52.5 | 40.1 |
| planar | 70.6 | 67.6 | 62.1 | 52.4 | 40.3 |
| planar3 | 71.0 | 68.6 | 62.4 | **52.7** | **40.4** |
| planar_k | 70.1 | 68.2 | **62.6** | 52.4 | 40.0 |
| k8vturbo2 | **71.3** | **68.7** | 61.4 | 52.2 | 40.0 |
| k8vturbo3 | **71.4** | 68.4 | 61.6 | 52.1 | 40.2 |
| k8vturbo2tcq | 71.2 | 68.6 | 61.6 | 52.5 | 40.1 |
| k8vturbo3tcq | 71.1 | 68.0 | 62.1 | **52.7** | 39.9 |
| tsym3 | 70.9 | 67.5 | 62.1 | 52.4 | 39.9 |
| tsym4 | 69.7 | 68.0 | 62.4 | 52.2 | 40.3 |
| iso3ᵂ | 71.0 | 68.0 | 62.4 | 52.0 | 38.8 |
| iso4ᵂ | 70.7 | 68.6 | 62.3 | 52.2 | 39.2 |
| iso3_symᵂ | 69.8 | 68.3 | 62.3 | 51.5 | 38.9 |
| iso4_symᵂ | 70.8 | 68.4 | 62.2 | 51.4 | 38.6 |

ᵂ fires the per-layer CPU-hot-path `warn!` (#36) — the iso/rotor V-encode is on CPU
at **prefill**, but **decode reads the bf16 seed**, so decode stays ≈ `none`. The
warn is about memory/prefill, not decode TPS.

**NEW — `kv_cache_bytes` (the KV-SIZE axis). KV "quant" _inflates_ resident KV.**
Ratio vs `none` is ~codec-intrinsic (near-constant across ctx); shown @64k with the
absolute MB. `none` absolute grows with ctx (e2b 64/115/229/440/**780** MB;
e4b 178/314/618/1181/**2088** MB at 4k/8k/16k/32k/64k):

| KV | e2b @64k | e4b @64k | ratio vs none | why |
|---|---|---|---|---|
| **k_rotor3 / k_rotor4** | **677** | **1818** | **0.87×** | only real K compression (K-only, no bf16 seed) — but CPU-bound (below) |
| none | 780 | 2088 | 1.00× | bf16 K+V, baseline |
| tsym3 | 876 | 2344 | 1.12× | |
| k8v4 | 939 | 2512 | 1.20× | quant blocks **+ bf16 seed** |
| k8v8 | 978 | 2616 | 1.25× | |
| planar / rotor* | 1136–1143 | 3038–3056 | 1.46× | + planar/rotation scratch |
| iso3 / iso4 | 1456 | 3890 | 1.86× | + quaternion buffers |
| **iso3_sym / iso4_sym** | **1934** | **5164** | **2.47×** | worst — sym rotation tables |

**Finding (the headline): on Gemma4 SWA, KV quant costs decode nothing AND grows
memory.** A quantized layer keeps its bf16 warm-TTFT decode seed _alongside_ the
packed blocks + scales (+ rotation/quaternion scratch), so every mainstream codec is
**larger** than `none` — up to 2.47×. The only genuine K compression (k_rotor, 0.87×)
is CPU-bound. So **`none` is the right default on dense Gemma4** — confirmed on both
axes. (#34's net-benefit `warn!` now surfaces this at resolve time.)

**Rotation / K-only — now honestly classified (#36), no cold-compile artifact.**
- **iso/rotor V-only** (`iso*`, `rotor*`): decode ≈ `none` (above); the CPU cost is
  prefill-side. Memory-negative (1.46–2.47×). Not worth it.
- **K-only `k_iso* / k_rotor*` are genuinely CPU-bound** — K dequant runs on the host
  every decode step (no bf16 seed to shadow it). Decode craters with ctx:
  e2b `k_iso3` **52.4/30.9/20.5/12.3/3.4**, `k_iso4` 26.7/15.4/8.2/4.2/2.0;
  e4b `k_iso3` 23.4/14.6/7.6/3.1/1.8. `k_rotor*` are ~1–4 TPS (32k/64k cells capped —
  too slow to measure). They are the **only KV-shrinking** codecs (0.87×) but are
  decode-uncompetitive; a Metal K-dequant kernel would be needed to make them viable.
- **`rot_k_tq4v`** is Metal but its TQ-4 K-kernel cost **scales with hidden dim**:
  ≈`none` on e2b (111.7/109.5/99.9/90.3/69.5) but **−15…−27 %** on e4b
  (67.9/58.9/54.4/43.8/29.3) — a real K-side cost on the wider model, not an artifact.

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

### 2d. Phase E — speculative (Gemma4 assistant MTP) — **FIXED** (#23/#24/#32; see §V)

> **Resolved 2026-06-08.** All three failures below are fixed: #23 (dispatch), #24
> (additive→array mask), and **#32** (the verify-step SWA mask off-by-one in both the
> producer and consumer attention branches). e2b + `assistant-bf16` `--draft-kind
> mtp` now serves a 4.6k-token prompt with a partial-accept round, coherent output,
> **no SDPA broadcast crash**. The pre-fix failure analysis is kept below for history.

Pre-fix, every speculative cell failed. Three distinct failures (verifier @4k, kv none):

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

**Phase E verdict (post-fix):** Gemma4 speculative is **functional** — e2b +
assistant-bf16 MTP runs at real prompt lengths without crashing (#23/#24/#32 all
merged). Accept-rate / net speedup is not yet swept here (correctness gate only —
the 4.6k repro exercised a partial-accept round, no crash); a speed sweep is
follow-up. (Mirrors the Qwen3.6 path: crash fixed first, speed characterized later.)

---

## 3. Standing vs champion (decode)

rMLX best **mainstream** KV (2026-06-08 post-fix sweep) vs the SIBLINGS mlx-lm
champion. WIN / TIE-on-CI / LOSS (§0.1). Decode is **unchanged within noise** from
the prior sweep — the 6 fixes don't touch the decode kernel. e2b/e4b refreshed;
26b/31b are the prior Stage-2 rows (not re-run).

### e2b — TIE @4k, losses widen with ctx
| Prompt | rMLX best (mainstream) | champion (mlx-lm) | standing |
|---|---|---|---|
| 4k | 116.6 (none) | 116.4 | 🟡 **TIE +0.2%** |
| 8k | 111.1 (k8v8) | 114.7 | 🔴 LOSS −3.1% |
| 16k | 104.9 (k8v4/k8v8) | 110.7 | 🔴 LOSS −5.2% |
| 32k | 92.6 (planar3) | 106.1 | 🔴 LOSS −12.7% |
| 64k | 76.6 (tsym3) | 97.4 | 🔴 LOSS −21.4% |

### e4b — LOSS −4…−12 %, widening
| Prompt | rMLX best (mainstream) | champion | standing |
|---|---|---|---|
| 4k | 71.4 (k8vturbo3) | 74.6 | 🔴 LOSS −4.3% |
| 8k | 68.7 (k8vturbo2) | 71.3 | 🔴 LOSS −3.6% |
| 16k | 62.6 (planar_k) | 66.1 | 🔴 LOSS −5.3% |
| 32k | 52.7 (planar3) | 58.1 | 🔴 LOSS −9.3% |
| 64k | 40.4 (planar3) | 45.9 | 🔴 LOSS −12.0% |

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

> **Verdict (post-fix, 2026-06-08): rMLX TIEs e2b @4k (+0.2 %) and trails at long
> ctx**, the gap widening to −12…−21 % at 32k–64k (e4b −4…−12 %). Decode is unchanged
> by the 6 fixes — they were KV-reporting, spec, and classification work. The champion
> comparison is **decode only**; on prefill rMLX is also far behind (§2a TTFT). Unlike
> Qwen3.6 (MoE, no SWA — rMLX won +12–15 %), Gemma4's interleaved SWA still exposes a
> weakness in rMLX's sliding-window attention decode path at scale. KV quant moves
> neither decode (±2 %) nor memory (it _inflates_ KV — §2b).

---

## 4. Gaps & hypotheses (Phase F synthesis → improvement plan)

Ranked by impact:

1. **SWA-attention decode path — the primary deficit.** rMLX loses 5–26 % on
   Gemma4 decode, widening with context, and **KV quant does NOT recover it**
   (Phase C) → the cost is **not KV-read-bandwidth** but the per-step
   sliding-window attention compute (mask construction, ring snapshot/restore, or
   attention-over-window kernel). The widening-with-ctx slope is the signature.
   Profile the Gemma4 decode attention vs mlx-lm's SWA path. **Highest value.**
2. **KV memory: quant inflates, and the global seed is f32.** Two corrected facts
   replace the old "6× full-ctx KV" claim: windowed SWA layers are **already
   window-bounded** (#35), and the prior 6× was a reporting artifact (#39). The real
   levers now: (a) KV "quant" **grows** resident KV (bf16 seed alongside blocks,
   1.2–2.47× — §2b), so it is net-negative on dense Gemma4; (b) the None-path **global
   decode seed is f32** — storing it bf16 is a ~2× KV-residency reduction on the only
   growing layers (**#44**, numerics-gated). Neither gates decode speed (KV quant is a
   no-op there) but both cap deployable context.
3. **Speculative — FIXED (#23/#24/#32).** Dispatch (#23), additive-mask crash (#24),
   and the verify-step SWA mask off-by-one in both producer + consumer branches (#32)
   are all resolved; e2b + assistant MTP runs at 4.6k without crashing. Remaining:
   sweep accept-rate / net speedup (the 31b verifier + e2b/assistant draft is where
   spec should pay off) — not yet measured.
4. **Prefill/TTFT on the big models.** 31b @128k prefill > 600 s (times out);
   26b @128k ≈ 403 s; prefill tok/s sinks to ~300 at long ctx. Same prefill class
   the Qwen3.6 campaign flagged — separate from decode, large.
5. **K-only codecs are CPU-bound (#36).** `k_iso* / k_rotor*` are the only
   KV-shrinking codecs (0.87×) but run K dequant on the host every decode step →
   decode craters with ctx (§2b). A Metal K-dequant kernel would make the one genuine
   compressor viable; until then they are decode-uncompetitive and warned at resolve.

**No KV-quant bright spot on e2b/e4b** — `none` wins both axes. The prior 31b
k8vturbo2 +4.1 % @64k note (KV-bandwidth-pressured dense model) stands as the lone
KV-quant win, but 31b was not re-run this sweep (decode-only; no KV-byte axis).
Coherence was solid across all 25 codecs and all baseline cells.

---

## 5. Caveats

- **64k/128k are n=1 measured** (single post-warmup run) — point estimates.
- **31b 4k is soft** (anomalous vs 8k; possible regression — §2a²).
- **SSD not stress-tested** — no spill triggered at 256-token single-stream.
- **Speculative fixed (#23/#24/#32), speed not swept** — runs without crashing;
  accept-rate / net speedup not yet measured (correctness gate only).
- **K-only codecs decode capped at 32k/64k** — `k_iso* / k_rotor*` too slow
  (~1–4 TPS) to measure at long ctx; their KV-byte axis is captured at all sizes.
- **e2b/e4b 128k absent** — fixture (137,920 gemma tok) exceeds 131072 ctx.
- **`kv_cache_bytes` ingest whitelist** — `rmlx metrics record` only ingests
  none/k8v4/k8v8/planar into `observations`; the other 21 codecs' bytes were read
  from the `events` table (improvement-plan item: widen `canonicalize_kv_quant`).
- rMLX cells recorded in CBB `metrics/runs/*.jsonl` (backend=rmlx); **not** in
  `runs.db` (CBB schema rejected by the rMLX buffer — known harness landmine).
- Aggregator: `Cross-Backend-Bench/scripts/agg_gemma4_siblings.py`.
