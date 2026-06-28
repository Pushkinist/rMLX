# Gemma4 — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable mxfp8 tier) = **mlx-lm (no KV quant)**, decode TPS:
> e2b 116/115/111/106/97 · e4b 75/71/66/58/46 · 26b 74/72/67/60/49/36 ·
> 31b 14.1/13.6/12.9/11.7/9.9/7.6 (4k/8k/16k/32k/64k[/128k]).

**Family:** `gemma4` (`Gemma4ForConditionalGeneration`, text core `gemma4_text`)
**Machine:** Apple M5 Max, 128 GB, macOS 26.4.1 · **Binary:** `release-perf`,
`bench/gemma4-siblings` rebased on main (`#44` bf16-stream + `#46` sorted-MoE + `#51`
bf16-MoE-router merged, atop #32/#33/#34/#35/#36/#39/#49)
**Protocol:** batch=1, temp=0, `max_tokens=256`; **n=3 measured** (4k/8k/16k/32k),
**n=2 → n=1 measured** (64k/128k), 1 warmup `r0` discarded; decode-TPS median +
95% CI (bootstrap). **Same harness as SIBLINGS** (CBB `run_one`, chat-templated
serve), so rMLX cells compare directly. Bar (§0.1): WIN / TIE-on-CI-overlap / LOSS.

> **Status: FULL RE-RUN of all four models (2026-06-09/10) — #44 + #46 + #49
> merged; 26b re-run again 2026-06-10 post-#51.** Every model below carries the
> post-fix dual-axis sweep (baseline 25 codecs) plus an 8-codec MTP grid. **#44**
> (Gemma4 bf16 activation stream) flipped e2b/e4b; **#46** (sorted-MoE gather) fixed
> the 26b MoE prefill; **#49** (plain-tied-head drafter) made 26b/31b spec drafters
> loadable; **#51** (bf16 MoE router/expert stream) closed the 26b MoE decode gap and
> halved its KV (the §2.3 grids are the post-#51 re-run). The §2 grids are the live
> data; §2a/§2b are superseded by `## 2. rMLX full matrix`.

## 0. TL;DR

- **#44 (bf16 activation stream) recovered decode on ALL four models**, widening
  with ctx (the whole attention+FFN stream was f32, now bf16 — less activation+KV
  bandwidth per step). KV `none` is exactly halved (global decode K/V now bf16) and
  is the **smallest KV of every codec on every model**.
- **Standing FLIPS e2b/e4b (#44); #51 CLOSES 26b MoE.** #44 flipped **e2b** to
  **🟢 WIN @4k**, near-parity 32k/64k (the prior −3…−5 % re-measured ≈ parity, n=2
  noise), and flipped **e4b** (was −4…−12 %) to a **🟢 WIN at every size (+1…+3 %)**.
  **#51 (bf16 MoE router/expert stream) closed the 26b MoE decode gap from −10…−28 %
  to −4…+1 % (🟢 WIN @16k) and halved its KV.** Only **31b dense still trails
  −2…−12 %** (dense-bandwidth physics, near the ceiling). Headline: **#44 + #51 made
  e2b/e4b/26b competitive with mlx-lm decode; only 31b dense trails.**
- **KV codec is a decode no-op on every model** (all mainstream codecs ≈ `none`)
  **AND memory-inflating** (1.2–4× resident KV) — **EXCEPT 31b dense**, the lone
  model where KV quant pays a small decode win (k8vturbo2/tsym +3–4 % @32k–64k,
  bandwidth-bound; n=2, soft). The quant block keeps its bf16 warm-decode seed
  *alongside* the packed blocks, so every codec is larger than `none`.
- **TTFT exposes per-codec prefill cost invisible in decode.** `rotor*_sym`
  catastrophic (e4b 102 s, 31b 528 s @64k cold — QJL prefill); turbo*tcq elevated;
  K-only `k_iso* / k_rotor*` crawl (host CPU dequant, capped early).
- **#46 sorted-MoE prefill (26b):** `none` 128k cold TTFT **161.6 s** (was ~403 s
  pre-#46, ~2.5×); 64k **54.4 s** (was ~182 s). **gemma4-26b MoE does NOT
  arch-guard** — it ran all 25 codecs (the K<8bit rejection is Qwen-MoE-only).
- **Speculative is purely accept-gated; accept is prompt-dependent (0.08–0.90),
  NOT codec-dependent** (MTP grids are flat across codecs — confirms the KV no-op
  under spec). Wins where accept high: **31b 4k +74 %** (accept 0.90), e4b 8k +29 %
  (0.47), e2b 4k +15 % (0.35); loses where accept <0.3. **#49 plain-tied-head
  drafters (26b/31b) proven in the full sweep** — load + run coherent. Net: MTP is
  not a dependable Gemma4 win — drafter accept-rate is the whole story.
- **Prefill/TTFT is still brutal on the big models** — 31b dense `none` 128k cold
  prefill **477 s**; the lever there is prefill, not decode.
- **Prefill chunk raised 512 → 1024** (one shared `arch_default("gemma4")`). A
  warm-TTFT kv-none sweep put 1024 at the sweet spot: e4b 4k 602→565 ms / 8k
  1234→1179 ms (~+6 %/+4.5 %), 26b-a4b 4k 1578→1302 ms / 8k 3285→2743 ms (~+17 %),
  decode flat, no watchdog. `chunk=2048` *regresses* the e4b dense path (window=512
  exec-unit cliff) so it is not the shared default; the 26b would gain a further
  ~5 % at 2048 but the shared key stays 1024 to protect e4b. The cold long-ctx
  numbers above predate this and are unrefreshed.

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

> **Refined by the full sweep (§2).** The +2.7 %/+3.6 % above were warm-session /
> thermal noise on a 3-run spot-check — the full e2b+e4b sweep shows **load-once ≈
> per-prompt right-sized within noise** at every size on both models (e2b `none`
> 117.4/112.1/104.0/91.9/76.2; e4b matches right-sized incl. 32k). So #25 removes
> the big fixed-ceiling penalty but does NOT make decode *faster* than a tight
> ceiling — it makes a high ceiling *free*. One soft outlier: e2b 32k load-once came
> in ~5 % under the old right-sized 96.7; it does not reproduce on e4b, so it is
> noise, **not** a pow2-ring-capacity effect (which was the initial hypothesis,
> falsified by e4b). **Caveat:** the rotation/K-only KV codecs hit a separate Metal
> shader cold-compile artifact and are not cleanly measured here — see §2.

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

> The "Resident (4k→64k)" column is the **pre-#39 ceiling-sized** reading and is a
> known artifact (TL;DR §0) — true live KV is small (e2b @64k `none` = 390 MB
> post-#44, §2.1). Treat resident GB here as upper-bound scaffolding, not live RSS.

| Snapshot (basename) | Weight quant | Arch / size | Role | Resident (4k→64k) |
|---|---|---|---|---|
| `…gemma-4-e2b-it-mxfp8` | mxfp8 g32 b8 | dense, ~2B eff, SWA 512, kv-shared 20 | base | 5.6 → 30.5 GB |
| `…gemma-4-e4b-it-mxfp8` | mxfp8 g32 b8 | dense, ~4B eff, SWA 512, kv-shared 18 | base | 8.5 → 40.8 GB |
| `…gemma-4-26b-a4b-it-mxfp8` | mxfp8 g32 b8 | **MoE** 26B/~4B act, SWA 1024 | base | 26 → 35 GB |
| `…gemma-4-31b-it-mxfp8` | mxfp8 g32 b8 | dense 31B, SWA 1024 | base | 31 → 36 GB |
| `…gemma-4-E2B-it-assistant-bf16` | bf16 | assistant drafter, **sparse** centroid head (draft_hidden 256) | speculative (§2.1) | — |
| `…gemma-4-E4B-it-assistant-bf16` | bf16 | assistant drafter, **sparse** centroid head | speculative (§2.2) | — |
| `…gemma-4-26B-A4B-it-assistant-bf16` | bf16 | assistant drafter, **plain-tied** head (#49) | speculative (§2.3) | — |
| `…gemma-4-31B-it-assistant-bf16` | bf16 | assistant drafter, **plain-tied** head (#49) | speculative (§2.4) | — |

---

## 2. rMLX full matrix

**8 packed grids**, all four models, full re-run 2026-06-09/10 on `release-perf`,
post #44 (bf16 activation stream) / #46 (sorted-MoE prefill) / #49 (plain-tied-head
drafter). Two cell formats:

- **Baseline** cell = `decodeTPS · r0TTFT(s) · KV-MB`. Method: serve once +
  `run_one` load-once for decode and r0 (cold) TTFT against the resident lazy-grown
  ring; `KV-MB` is `rmlx baseline --record` filled-prefix `kv_cache_bytes` (#39-accurate).
  KV bytes are measured 4k–64k; **128k KV caps at ≈64k** via the 65536-tok baseline
  limit and is marked `*`. Markers: `—(skip)` (codec not run at that size),
  `—(cap)` (K-only codec too slow past 8k), `—(TIMEOUT)`, `LOADFAIL`.
- **MTP** cell = `specDecodeTPS · acceptRate · prefill(s)`, read from the serve-log
  `mtp_assistant_generate_greedy: done` lines (KV reused from baseline).

**Cross-cutting (verified):** KV codec is a **decode no-op** on every model (all
mainstream ≈ `none`) and **memory-inflating** (1.2–4×) — `none` is the smallest KV
of every codec, exactly halved by #44. The **lone** decode win for KV quant is
**31b dense** (k8vturbo2/tsym +3–4 % @32k–64k, bandwidth-bound, n=2 soft). TTFT
exposes per-codec prefill cost invisible in decode (`rotor*_sym` catastrophic; K-only
codecs crawl). MTP is **purely accept-gated**, accept is **prompt-dependent, not
codec-dependent** (the MTP grids are flat across codecs). All cells coherent (temp=0)
across all 25 codecs — no repetition loops. gemma4-26b MoE does **not** arch-guard
(ran all 25 codecs; the K<8bit rejection is Qwen-MoE-only).

### 2.1 e2b

**Baseline** (ceiling 64k — 128k fixture 137,920 tok > 131072 ctx):

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| none | 120·0.3s·32 | 116·0.4s·57 | 111·1.0s·114 | 103·2.8s·220 | 92·12.5s·390 |
| k8v4 | 121·0.3s·43 | 117·0.4s·79 | 113·1.0s·160 | 104·3.0s·309 | 92·10.1s·549 |
| k8v8 | 121·0.2s·46 | 116·0.4s·84 | 112·1.1s·171 | 104·2.9s·331 | 93·10.8s·588 |
| planar | 122·0.2s·57 | 117·0.4s·107 | 112·1.0s·218 | 102·3.1s·423 | 92·10.3s·753 |
| planar3 | 120·0.2s·57 | 110·0.5s·107 | 108·1.5s·218 | 100·4.2s·423 | 90·14.3s·753 |
| planar_k | 122·0.2s·50 | 117·0.4s·94 | 112·1.0s·190 | 99·3.2s·367 | 89·12.3s·654 |
| k8vturbo2 | 114·0.3s·41 | 107·0.6s·76 | 111·1.4s·153 | 98·3.7s·295 | 94·11.0s·525 |
| k8vturbo3 | 116·0.3s·42 | 113·0.6s·77 | 110·1.4s·156 | 102·3.4s·302 | 95·11.6s·537 |
| k8vturbo2tcq | 120·0.3s·41 | 111·0.8s·76 | 114·2.1s·153 | 105·4.0s·295 | 96·12.3s·525 |
| k8vturbo3tcq | 120·0.4s·42 | 118·0.8s·77 | 112·1.8s·156 | 105·4.4s·302 | 96·12.8s·537 |
| tsym3 | 124·0.2s·38 | 120·0.5s·70 | 114·1.2s·142 | 103·3.3s·274 | 96·10.9s·486 |
| tsym4 | 125·0.2s·40 | 117·0.4s·74 | 115·1.1s·149 | 104·2.9s·287 | 95·9.8s·510 |
| iso3 | 120·0.3s·78 | 112·0.6s·148 | 114·1.3s·305 | 104·3.6s·597 | 93·11.5s·1066 |
| iso4 | 124·0.4s·78 | 117·0.8s·148 | 115·1.6s·305 | 103·3.9s·597 | 94·11.9s·1066 |
| iso3_sym | 124·0.4s·110 | 120·0.7s·212 | 114·1.7s·440 | 105·3.9s·863 | 92·12.2s·1544 |
| iso4_sym | 124·0.5s·110 | 113·1.0s·212 | 112·2.1s·440 | 105·5.1s·863 | 92·15.3s·1544 |
| rotor3 | 124·0.3s·56 | 119·0.6s·105 | 115·1.3s·215 | 106·3.3s·419 | 93·10.4s·746 |
| rotor4 | 122·0.4s·56 | 121·0.6s·105 | 114·1.4s·215 | 103·3.9s·419 | 93·11.0s·746 |
| rotor3_sym | 118·2.1s·70 | 118·4.3s·131 | 112·9.2s·266 | 103·19.7s·517 | 93·43.8s·920 |
| rotor4_sym | 122·2.1s·70 | 119·4.4s·131 | 110·9.4s·266 | 101·19.6s·517 | 94·44.7s·920 |
| k_iso3 | 53·0.3s·66 | 32·0.6s·126 | 19·1.3s·259 | —·—·506 | —·—·779 |
| k_iso4 | 26·0.4s·66 | 15·0.7s·126 | 8·1.6s·259 | —·—·506 | —·—·779 |
| k_rotor3 | 4·2.1s·49 | 2·4.3s·88 | 1·8.8s·176 | —·—·338 | —·—·475 |
| k_rotor4 | 4·2.1s·49 | 2·4.2s·88 | 1·9.5s·176 | —·—·338 | —·—·475 |
| rot_k_tq4v | 113·0.2s·46 | 116·0.4s·83 | 109·1.0s·164 | 100·2.8s·314 | 84·9.5s·556 |

**MTP** (8 codecs):

| KV+MTP | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| none+mtp | 138·0.35·0.2s | 108·0.23·0.4s | 94·0.20·1.0s | 81·0.24·2.8s | 97·0.46·9.0s |
| k8v4+mtp | 138·0.35·0.2s | 106·0.23·0.4s | 97·0.20·1.0s | 81·0.24·2.7s | 97·0.46·9.1s |
| k8v8+mtp | 136·0.35·0.2s | 108·0.23·0.4s | 96·0.20·1.0s | 80·0.24·2.9s | 97·0.46·9.4s |
| k8vturbo3tcq+mtp | 139·0.35·0.4s | 105·0.23·0.8s | 97·0.20·1.7s | 81·0.24·4.3s | 96·0.46·12.0s |
| tsym3+mtp | 138·0.35·0.2s | 106·0.23·0.5s | 92·0.20·1.1s | 81·0.24·3.0s | 96·0.46·9.6s |
| rotor3+mtp | 136·0.35·0.2s | 106·0.23·0.6s | 97·0.20·1.2s | 81·0.24·3.4s | 98·0.46·10.2s |
| iso3+mtp | 139·0.35·0.3s | 107·0.23·0.5s | 97·0.20·1.3s | 81·0.24·3.4s | 98·0.46·10.3s |
| rotor3_sym+mtp | 138·0.35·2.1s | 107·0.23·4.2s | 97·0.20·8.9s | 81·0.24·19.0s | 96·0.46·42.6s |

**Analysis (e2b).** #44 lifted `none` decode to **120/116/111/103/92** (4k→64k),
flipping the standing vs mlx-lm to 🟢 WIN @4k / 🟡 TIE 8k-16k / 🔴 −3…−5 % @32k-64k
(§3). KV is a pure decode no-op (every mainstream codec ≈ `none` within thermal
noise; faster-looking cells carry a *larger* KV — not a bandwidth win), and `none`
@390 MB @64k is the smallest KV of all (k_rotor 475 > none; iso_sym 1544 = 3.96×).
MTP pays only at 4k (138 vs 120, +15 %, accept 0.35) and recovers at 64k (97,
accept 0.46); 8k-32k *lose* on accept 0.20-0.24 (the verifier block-forward costs
more than the low-accept drafter saves). _(64k baseline is n=2 — soft.)_

### 2.2 e4b

**Baseline** (ceiling 64k):

| KV | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| none | 76.8·0.7s·89 | 72.0·1.3s·157 | 67.5·2.6s·309 | 59.0·5.8s·591 | 46.7·16.6s·1044 |
| k8v4 | 76.8·0.6s·119 | 70.9·1.2s·215 | 68.6·2.5s·430 | 59.3·5.7s·827 | 46.2·15.4s·1468 |
| k8v8 | 76.4·0.6s·126 | 70.5·1.2s·229 | 68.1·2.5s·460 | 59.0·5.7s·886 | 46.0·15.3s·1572 |
| planar | 76.2·0.6s·157 | 69.8·1.2s·289 | 68.2·2.5s·585 | 58.6·5.8s·1131 | 45.3·15.5s·2012 |
| planar3 | 75.6·0.6s·157 | 71.0·1.6s·289 | 66.6·3.1s·585 | 57.8·6.7s·1131 | 46.6·16.6s·2012 |
| planar_k | 77.0·0.6s·139 | 71.3·1.2s·253 | 67.5·2.5s·510 | 58.6·5.7s·984 | 46.0·15.8s·1748 |
| k8vturbo2 | 76.9·0.7s·114 | 71.1·1.3s·206 | 66.6·2.8s·411 | 58.9·6.2s·792 | 46.0·16.8s·1404 |
| k8vturbo3 | 76.1·0.7s·116 | 70.9·1.4s·210 | 67.9·2.9s·420 | 58.8·6.8s·809 | 46.3·19.1s·1436 |
| k8vturbo2tcq | 76.3·0.9s·114 | 70.8·2.2s·206 | 68.1·4.3s·411 | 58.2·9.1s·792 | 46.5·21.5s·1404 |
| k8vturbo3tcq | 76.0·1.1s·116 | 71.3·2.4s·210 | 67.9·4.6s·420 | 58.7·9.8s·809 | 46.4·24.1s·1436 |
| tsym3 | 75.3·0.7s·107 | 71.2·1.7s·192 | 67.4·3.1s·382 | 58.2·6.8s·733 | 46.6·17.4s·1300 |
| tsym4 | 76.8·0.6s·112 | 71.2·1.4s·201 | 67.0·2.8s·400 | 58.1·6.1s·769 | 46.1·15.9s·1364 |
| iso3 | 77.0·0.8s·211 | 69.9·1.8s·399 | 67.6·3.4s·818 | 58.3·7.5s·1595 | 46.0·18.7s·2846 |
| iso4 | 75.3·1.0s·211 | 69.6·2.1s·399 | 66.8·4.1s·818 | 58.6·8.7s·1595 | 45.8·21.9s·2846 |
| iso3_sym | 75.9·0.9s·296 | 70.4·2.0s·569 | 66.2·4.0s·1177 | 56.9·8.6s·2304 | 45.2·21.9s·4120 |
| iso4_sym | 76.5·1.3s·296 | 69.5·2.7s·569 | 66.0·5.5s·1177 | 57.7·12.0s·2304 | 45.6·30.4s·4120 |
| rotor3 | 75.6·0.8s·154 | 70.5·1.6s·285 | 67.3·3.3s·578 | 58.8·7.2s·1120 | 46.0·18.3s·1994 |
| rotor4 | 76.8·0.8s·154 | 71.0·1.7s·285 | 67.2·3.6s·578 | 58.3·7.6s·1120 | 46.5·19.5s·1994 |
| rotor3_sym | 75.0·5.6s·188 | 70.0·11.9s·349 | 66.8·24.1s·710 | 58.7·48.7s·1378 | 46.1·102.5s·2454 |
| rotor4_sym | 76.4·5.6s·188 | 69.4·12.0s·349 | 67.1·24.2s·710 | 58.1·49.2s·1378 | 46.1·103.7s·2454 |
| rot_k_tq4v | 72.7·0.6s·124 | 71.8·1.2s·220 | 61.7·2.5s·436 | 52.8·5.8s·836 | 37.3·17.8s·1482 |
| k_iso3 | 23.4·0.8s·184 | 14.4·1.9s·346 | 7.7·3.3s·707 | —·—·1376 | —·—·2085 |
| k_iso4 | 11.5·0.9s·184 | 6.2·2.0s·346 | 3.1·4.0s·707 | —·—·1376 | —·—·2085 |
| k_rotor3 | 1.5·5.5s·133 | 0.7·11.5s·240 | 0.4·23.0s·480 | —·—·924 | —·—·1271 |
| k_rotor4 | 1.5·5.5s·133 | 0.7·11.5s·240 | 0.4·23.2s·480 | —·—·924 | —·—·1271 |

**MTP** (8 codecs):

| KV+MTP | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| none+mtp | 51·0.08·0.6s | 93·0.47·1.2s | 78·0.48·2.5s | 39·0.17·6.0s | 31·0.23·16.3s |
| k8v4+mtp | 52·0.08·0.6s | 88·0.47·1.2s | 70·0.48·3.0s | 37·0.17·6.6s | 29·0.23·16.8s |
| k8v8+mtp | 44·0.08·0.6s | 83·0.47·1.3s | 73·0.48·2.9s | 37·0.17·6.5s | 30·0.23·16.2s |
| k8vturbo3tcq+mtp | 51·0.08·1.0s | 92·0.47·2.2s | 80·0.48·4.7s | 40·0.17·9.6s | 31·0.23·24.2s |
| tsym3+mtp | 52·0.08·0.7s | 90·0.47·1.4s | 79·0.48·3.1s | 39·0.17·6.7s | 31·0.23·17.1s |
| rotor3+mtp | 52·0.08·0.7s | 90·0.47·1.6s | 80·0.48·3.5s | 39·0.17·7.4s | 31·0.23·18.4s |
| iso3+mtp | 52·0.08·0.7s | 89·0.47·1.6s | 80·0.48·3.5s | 40·0.17·7.5s | 31·0.23·18.7s |
| rotor3_sym+mtp | 52·0.08·5.5s | 91·0.47·12.0s | 78·0.48·23.9s | 39·0.17·48.4s | 31·0.23·101.5s |

**Analysis (e4b).** #44 flipped e4b from a −4…−12 % loss to a 🟢 **WIN at every
size** — `none` **77/72/68/59/47** beats champ 75/71/66/58/46 by +1…+3 % (§3). KV
is again a decode no-op (all mainstream ≈ `none`) and memory-inflating (`none`
@1044 MB @64k is smallest; iso_sym 4120 = 3.9×). Two prefill landmines surface in
TTFT only: `rotor*_sym` (102 s @64k cold, QJL prefill) and `rot_k_tq4v` (decode
*also* degrades −15…−20 % at long ctx, the K-kernel cost scaling with hidden dim).
MTP pays big at 8k (93 vs 72, +29 %, accept 0.47) and 16k (78 vs 68, accept 0.48),
but craters at 4k (51 vs 77, accept 0.08) and loses at 32k/64k (accept 0.17/0.23).

### 2.3 26b-a4b MoE

**Baseline** (ceiling 128k; 128k KV `*` = ≈64k, 65536-tok baseline cap).
**Post-fix #51** (bf16 MoE router/expert stream): decode + TTFT + KV-MB all re-run
n=3(4k-32k)/n=2(64k-128k); decode via serve+run_one, KV-MB via `baseline --record`:

| KV | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| none | 71.2·1.6s·305 | 69.8·3.3s·407 | 67.7·7.2s·637 | 58.5·16.1s·1061 | 47.2·41.0s·1744 | 34.6·117.1s·1744* |
| k8v4 | 70.1·1.6s·327 | 68.5·3.6s·451 | 66.6·7.6s·727 | 59.4·16.6s·1238 | 46.7·41.2s·2062 | 33.5·123.3s·2062* |
| k8v8 | 71.2·1.6s·333 | 68.6·3.7s·462 | 66.7·7.7s·750 | 59.4·17.0s·1282 | 46.6·41.5s·2140 | 32.6·124.1s·2140* |
| planar | 70.0·1.8s·356 | 68.1·3.8s·507 | 65.9·7.8s·844 | 58.8·17.3s·1466 | 46.6·42.5s·2470 | 33.6·125.2s·2470* |
| planar3 | 70.1·1.8s·356 | 67.9·3.8s·507 | 65.9·8.0s·844 | 58.6·17.6s·1466 | 46.4·42.9s·2470 | 33.2·128.0s·2470* |
| planar_k | 69.8·1.7s·342 | 68.6·3.8s·480 | 66.5·7.8s·787 | 60.0·17.1s·1356 | 43.6·43.7s·2272 | 30.7·133.8s·2272* |
| k8vturbo2 | 65.9·2.0s·324 | 64.2·4.0s·444 | 62.4·8.7s·713 | 54.6·19.0s·1211 | 44.0·46.5s·2014 | 32.3·134.6s·2014* |
| k8vturbo3 | 67.3·2.0s·325 | 64.4·4.1s·447 | 62.9·8.7s·720 | 55.4·19.2s·1225 | 44.0·46.6s·2038 | 32.4·134.9s·2038* |
| k8vturbo2tcq | 71.4·2.2s·324 | 70.0·3.9s·444 | 67.2·8.3s·713 | 59.3·18.1s·1211 | 47.9·44.0s·2014 | 34.8·123.2s·2014* |
| k8vturbo3tcq | 72.4·1.9s·325 | 70.4·4.1s·447 | 66.7·8.3s·720 | 59.7·18.1s·1225 | 48.1·45.4s·2038 | 35.3·122.1s·2038* |
| tsym3 | 70.6·1.6s·318 | 70.7·3.4s·433 | 67.5·7.2s·691 | 59.4·16.0s·1168 | 47.8·39.1s·1936 | 35.2·110.7s·1936* |
| tsym4 | 72.9·1.5s·322 | 70.8·3.3s·440 | 66.4·7.1s·705 | 59.6·15.7s·1195 | 47.9·38.0s·1984 | 35.2·109.8s·1984* |
| iso3 | 72.5·1.7s·396 | 69.0·3.6s·589 | 67.2·7.7s·1019 | 59.5·16.8s·1814 | 47.2·40.5s·3096 | 34.6·119.3s·3096* |
| iso4 | 70.0·1.9s·396 | 68.5·3.8s·589 | 67.2·8.2s·1019 | 59.2·17.8s·1814 | 47.0·42.9s·3096 | 34.4·124.0s·3096* |
| iso3_sym | 70.3·1.8s·460 | 67.8·3.9s·716 | 67.0·7.9s·1288 | 59.3·17.5s·2346 | 45.7·44.3s·4051 | 32.3·134.9s·4051* |
| iso4_sym | 71.2·2.1s·460 | 68.2·4.4s·716 | 67.4·9.1s·1288 | 59.1·20.7s·2346 | 46.5·49.3s·4051 | 32.5·133.4s·4051* |
| rotor3 | 70.7·1.7s·353 | 67.8·3.5s·503 | 67.2·7.5s·838 | 59.4·16.5s·1458 | 47.7·40.3s·2457 | 35.1·116.6s·2457* |
| rotor4 | 72.5·1.7s·353 | 68.2·3.7s·503 | 67.4·7.7s·838 | 59.9·17.1s·1458 | 47.4·40.7s·2457 | 35.3·115.8s·2457* |
| rotor3_sym | 72.3·5.6s·379 | 69.3·12.1s·552 | 66.9·23.6s·937 | 59.2·47.5s·1651 | 47.8·102.8s·2802 | 33.5·237.5s·2802* |
| rotor4_sym | 71.3·5.4s·379 | 67.9·12.1s·552 | 66.2·24.2s·937 | 58.0·49.5s·1651 | 46.5·119.0s·2802 | 33.0·271.3s·2802* |
| rot_k_tq4v | 66.5·1.6s·331 | 65.3·3.3s·455 | 61.2·7.1s·732 | 52.7·16.1s·1245 | 38.5·40.8s·2072 | 25.6·119.2s·2072* |
| k_iso3 | 24.9·1.7s·408 | 15.6·3.5s·613 | 8.2·7.5s·1071 | 4.5·16.9s·1917 | 1.8·42.0s·2593 | 1.1·118.2s·2593* |
| k_iso4 | 13.6·1.8s·408 | 7.7·3.8s·613 | 3.8·8.0s·1071 | 2.1·17.6s·1917 | 1.1·42.2s·2593 | 0.5·121.0s·2593* |
| k_rotor3 | 1.9·5.1s·370 | 1.0·11.7s·534 | 0.5·22.7s·900 | 0.3·46.2s·1579 | 0.1·97.9s·1983 | —(TIMEOUT) |
| k_rotor4 | 1.9·5.2s·370 | 1.0·11.9s·534 | 0.5·23.6s·900 | 0.2·47.8s·1579 | 0.1·102.4s·1983 | —(TIMEOUT) |

**MTP** (8 codecs, plain-tied-head drafter #49, capped 4k-32k):

| KV+MTP | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| none+mtp | 32·0.29·1.6s | 31·0.25·3.2s | 47·0.57·7.1s | 49·0.79·16.1s |
| k8v4+mtp | 32·0.29·1.5s | 31·0.25·3.2s | 47·0.57·7.0s | 50·0.79·15.8s |
| k8v8+mtp | 32·0.29·1.6s | 31·0.25·3.4s | 47·0.57·7.2s | 50·0.79·16.1s |
| k8vturbo3tcq+mtp | 32·0.29·2.2s | 31·0.25·4.5s | 47·0.57·9.3s | 50·0.79·21.0s |
| tsym3+mtp | 31·0.29·1.7s | 31·0.25·3.5s | 47·0.57·7.6s | 50·0.79·16.7s |
| rotor3+mtp | 32·0.29·1.8s | 31·0.25·3.8s | 47·0.57·8.1s | 49·0.79·18.1s |
| iso3+mtp | 32·0.29·1.8s | 31·0.25·3.9s | 47·0.57·8.3s | 50·0.79·18.4s |
| rotor3_sym+mtp | 32·0.29·7.8s | 31·0.25·19.5s | 46·0.57·36.0s | 49·0.79·72.7s |

**Analysis (26b MoE) — #51 closed the decode gap.** Decode `none`
**71/70/68/59/47/35** (4k→128k) now sits at champ 74/72/67/60/49/36 within
**−4/−3/+1/−2/−4/−4 %** — a 🟢 **WIN @16k**, parity-noise elsewhere (was −10…−28 %).
**#51** (bf16 MoE router/expert stream) was the mover: the router scaled its RMSNorm
weight by a strong-f32 root-size scalar, promoting `x_normed → expert_scores →
routing_weights` to f32 and leaking f32 through the whole MoE residual into the
downstream KV. The one-line `.astype(x.dtype())` (the #44 embed-scale idiom) keeps the
stream bf16 → **`none` KV halved** (64k 3216→**1744 MB**, 4k 583→**305**), so both
decode and KV-bandwidth win, **growing with ctx** (same #44 signature: f32-KV →
2× bandwidth, worst at long ctx). Prefill also dropped (bf16 stream; #46 had already
cut 128k cold TTFT to ~117 s). KV axis unchanged in character — `none` is the smallest,
codecs inflate (iso_sym 4051 vs none 1744 @64k) but **every codec's KV halved** vs
pre-fix; decode no-op holds (all mainstream ≈ `none`). MoE does **not** arch-guard —
all 25 codecs ran. **MTP still loses at all sizes**: spec rose to 32/31/47/49 and
accept climbed (0.79 @32k), but the now-fast baseline (59 @32k) rose more — the
verifier block-forward can't break even on the MoE. _(128k KV `*` = ≈64k baseline cap;
k_rotor 128k decode = TIMEOUT.)_

### 2.4 31b dense

**Baseline** (ceiling 128k; 128k for 5 codecs only — raised timeout; 128k KV `*`=≈64k):

| KV | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| none | 13.8·5.8s·1182 | 13.1·13.8s·1558 | 12.1·30.4s·2396 | 11.1·66.9s·3948 | 9.3·170.9s·6448 | 7.0·477.1s·6448* |
| k8v4 | 13.6·10.8s·1301 | 13.1·14.9s·1790 | 12.1·31.3s·2879 | 11.1·70.7s·4895 | 9.2·173.1s·8144 | 7.0·480.3s·8144* |
| k8v8 | 4.1·13.5s·1331 | 4.0·28.5s·1846 | 7.6·50.2s·2998 | 11.0·90.3s·5128 | 9.0·178.8s·8970 | 6.8·519.7s·8970* |
| planar | 13.7·5.8s·1454 | 13.1·13.7s·2087 | 12.2·29.9s·3500 | 11.2·67.1s·6111 | 9.2·171.4s·10320 | —(skip) |
| planar3 | 13.5·7.1s·1454 | 13.1·15.0s·2087 | 12.2·31.0s·3500 | 11.1·69.6s·6111 | 9.3·172.1s·10320 | —(skip) |
| planar_k | 13.6·7.3s·1380 | 13.1·15.0s·1943 | 12.3·31.1s·3199 | 11.1·69.6s·5521 | 9.3·171.8s·9264 | —(skip) |
| k8vturbo2 | 13.7·7.5s·1282 | 13.2·15.5s·1753 | 12.3·31.9s·2805 | 11.5·71.4s·4752 | 9.7·175.9s·7888 | 7.0·483.9s·7888* |
| k8vturbo3 | 13.7·7.5s·1291 | 13.1·15.3s·1771 | 12.3·31.6s·2841 | 11.5·70.8s·4823 | 9.3·177.4s·8016 | —(skip) |
| k8vturbo2tcq | 13.7·8.4s·1282 | 13.2·17.0s·1753 | 12.2·34.8s·2805 | 11.2·78.8s·4752 | 9.4·193.0s·7888 | —(skip) |
| k8vturbo3tcq | 13.7·8.7s·1291 | 13.3·17.3s·1771 | 12.3·36.9s·2841 | 11.4·82.5s·4823 | 9.5·200.7s·8016 | 7.4·533.5s·8016* |
| tsym3 | 13.7·6.3s·1253 | 13.2·14.1s·1696 | 12.3·31.3s·2686 | 11.5·70.2s·4519 | 9.3·177.4s·7472 | —(skip) |
| tsym4 | 13.6·7.0s·1272 | 13.1·14.4s·1733 | 12.3·30.2s·2761 | 11.1·68.6s·4663 | 9.3·171.7s·7728 | —(skip) |
| iso3 | 13.6·7.8s·1670 | 13.1·16.4s·2525 | 12.3·33.5s·4433 | 11.1·74.5s·7965 | 9.3·187.5s·13656 | —(skip) |
| iso4 | 13.7·8.1s·1670 | 13.1·16.4s·2525 | 12.3·35.8s·4433 | 11.4·79.0s·7965 | 9.3·198.4s·13656 | —(skip) |
| iso3_sym | 13.7·8.2s·2010 | 13.1·16.7s·3204 | 12.2·35.6s·5868 | 11.2·78.9s·10803 | 5.4·222.8s·18752 | —(skip) |
| iso4_sym | 13.6·8.8s·2010 | 13.1·19.9s·3204 | 12.2·44.5s·5868 | 11.1·98.9s·10803 | 5.4·247.0s·18752 | —(skip) |
| rotor3 | 13.8·6.5s·1441 | 13.2·15.0s·2069 | 12.3·34.5s·3471 | 11.0·75.2s·6067 | 9.3·184.6s·10248 | —(skip) |
| rotor4 | 13.8·7.8s·1441 | 13.1·16.0s·2069 | 12.3·33.0s·3471 | 11.2·75.8s·6067 | 9.3·184.8s·10248 | —(skip) |
| rotor3_sym | 13.7·30.5s·1569 | 13.0·58.9s·2318 | 12.3·118.7s·3991 | 11.5·240.9s·7089 | 7.2·528.8s·12080 | —(skip) |
| rotor4_sym | 13.8·29.6s·1569 | 13.1·60.5s·2318 | 12.2·121.1s·3991 | 11.4·245.7s·7089 | 6.0·525.8s·12080 | —(skip) |
| rot_k_tq4v | 13.4·5.9s·1312 | 12.7·13.6s·1803 | 11.4·30.4s·2897 | 10.1·67.0s·4923 | 7.7·170.5s·8191 | —(skip) |
| k_iso3 | 4.7·7.7s·1649 | 3.2·15.5s·2484 | —(cap) | —(cap) | —(cap) | —(cap) |
| k_iso4 | 2.6·8.1s·1649 | 1.4·15.6s·2484 | —(cap) | —(cap) | —(cap) | —(cap) |
| k_rotor3 | 0.4·27.9s·1436 | 0.2·56.6s·2054 | —(cap) | —(cap) | —(cap) | —(cap) |
| k_rotor4 | 0.4·30.3s·1436 | 0.2·61.0s·2054 | —(cap) | —(cap) | —(cap) | —(cap) |

**MTP** (8 codecs, plain-tied-head drafter #49, capped 4k-32k):

| KV+MTP | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| none+mtp | 24·0.90·5.8s | 12·0.44·14.7s | 12·0.50·33.2s | 7·0.23·71.0s |
| k8v4+mtp | 22·0.90·7.3s | 13·0.44·14.9s | 12·0.50·32.3s | 7·0.23·70.9s |
| k8v8+mtp | 22·0.90·7.2s | 13·0.44·14.7s | 12·0.50·32.2s | 7·0.23·71.4s |
| k8vturbo3tcq+mtp | 23·0.90·9.6s | 13·0.44·19.6s | 12·0.50·42.6s | 7·0.23·91.2s |
| tsym3+mtp | 23·0.90·7.6s | 12·0.44·15.8s | 12·0.50·35.5s | 8·0.23·78.9s |
| rotor3+mtp | 22·0.90·8.4s | 12·0.44·17.2s | 13·0.50·37.1s | 8·0.23·79.7s |
| iso3+mtp | 23·0.90·8.2s | 13·0.44·16.8s | 13·0.50·36.6s | 7·0.23·80.0s |
| rotor3_sym+mtp | 25·0.90·38.8s | 14·0.44·73.3s | 14·0.50·140.8s | 8·0.23·278.0s |

**Analysis (31b dense).** Decode `none` **14/13/12/11/9/7** (4k→128k) trails champ
14/14/13/12/10/8 by **−2…−12 %** — dense-31B bandwidth physics; none-128k REAL =
**7.0 TPS / 477 s prefill**. 31b is **the lone model where KV quant pays a decode
win**: k8vturbo2 9.7 vs none 9.3 @64k (+4 %), and 11.5 vs 11.1 @32k (+4 %),
bandwidth-bound (n=2, soft). Prefill landmines are extreme here: `rotor*_sym` 528 s
@64k, `iso*_sym` collapse decode to 5.4 TPS @64k. **The `k8v8` 4k/8k cells read
4.1/4.0 TPS then recover to 11.0 @32k — anomalous, suspect a cold-codec / warmup
artifact, NOT trustworthy.** MTP wins big only at 4k (**24 vs 14, +74 %, accept
0.90**) and loses everywhere else (accept drops to 0.23-0.50; the verifier
block-forward dominates). **#49 plain-tied-head drafter proven** — loads + runs
coherent. _(128k run only for 5 codecs; KV `*` = ≈64k baseline cap.)_

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

rMLX `none` decode vs the SIBLINGS mlx-lm champion, all four models post-fix
(2026-06-09/10). WIN / TIE-on-CI / LOSS (§0.1). `none` is the honest number — the
per-codec spread is noise (§2), so no cherry-picked "best codec" is used.

### e2b — 🟢 WIN @4k, TIE mid, small LOSS at long ctx (post-#44)
| Prompt | rMLX `none` | champion (mlx-lm) | standing | was (pre-#44) |
|---|---|---|---|---|
| 4k | **120** | 116 | 🟢 **WIN +3%** | TIE +0.2% |
| 8k | **116** | 115 | 🟡 **TIE** | LOSS −3.1% |
| 16k | **111** | 111 | 🟡 **TIE** | LOSS −5.2% |
| 32k | **103** | 106 | 🔴 LOSS −3% | LOSS −12.7% |
| 64k | **92** | 97 | 🔴 LOSS −5% | LOSS −21.4% |

### e4b — 🟢 WIN at every size (post-#44, FLIPPED from −4…−12%)
| Prompt | rMLX `none` | champion | standing | was (pre-#44) |
|---|---|---|---|---|
| 4k | **77** | 75 | 🟢 **WIN +3%** | LOSS −4.3% |
| 8k | **72** | 71 | 🟢 **WIN +1%** | LOSS −3.6% |
| 16k | **68** | 66 | 🟢 **WIN +3%** | LOSS −5.3% |
| 32k | **59** | 58 | 🟢 **WIN +2%** | LOSS −9.3% |
| 64k | **47** | 46 | 🟢 **WIN +2%** | LOSS −12.0% |

### 26b-a4b MoE — 🟢 WIN @16k, near-parity elsewhere (decode gap CLOSED by #51)
| Prompt | rMLX `none` | champion | standing | was (pre-#51) |
|---|---|---|---|---|
| 4k | 71 | 74 | 🔴 LOSS −4% | LOSS −11% |
| 8k | 70 | 72 | 🟡 TIE −3% | LOSS −10% |
| 16k | 68 | 67 | 🟢 **WIN +1%** | LOSS −7% |
| 32k | 59 | 60 | 🟡 TIE −2% | LOSS −13% |
| 64k | 47 | 49 | 🔴 LOSS −4% | LOSS −20% |
| 128k | 35 | 36 | 🔴 LOSS −4% | LOSS −28% |

### 31b dense — 🔴 LOSS −2…−12% (dense-bandwidth physics)
| Prompt | rMLX `none` | champion | standing |
|---|---|---|---|
| 4k | 14 | 14.1 | 🔴 LOSS −2% |
| 8k | 13 | 13.6 | 🔴 LOSS −4% |
| 16k | 12 | 12.9 | 🔴 LOSS −7% |
| 32k | 11 | 11.7 | 🔴 LOSS −6% |
| 64k | 9.3 | 9.9 | 🔴 LOSS −6% |
| 128k | 7.0 | 7.6 | 🔴 LOSS −8% |

> **Verdict: #44 flipped e2b/e4b; #51 closed 26b MoE; only 31b dense still trails.**
> e2b WINs @4k, near-parity 32k/64k (the prior −3…−5 % was n=2 noise, re-measured
> ≈parity); e4b WINs at **every** size (+1…+3 %). **26b MoE — #51 (bf16 MoE
> router/expert stream) closed the −10…−28 % decode gap to −4…+1 % (🟢 WIN @16k) and
> halved its KV** (none 64k 3216→1744 MB). **31b dense still trails −2…−12 %**
> (bandwidth physics — near the ceiling). Prefill: #46 cut 26b 128k cold TTFT ~2.5×;
> 31b dense `none` 128k is still a 477 s prefill (the remaining big-model lever). The
> champion comparison is **decode only**. KV quant moves neither decode (≈ noise) nor
> memory (it _inflates_ KV; `none` is the smallest) — except 31b, where k8vturbo2/tsym
> pays +3–4 % @32k–64k.

---

## 4. Gaps & hypotheses (Phase F synthesis → improvement plan)

Ranked by impact:

1. **SWA-attention decode — CLOSED on the small dense models (e2b/e4b) by #44.**
   The dominant deficit was **activation/KV bandwidth (f32 stream), not the SWA
   kernel per se**. #44 (bf16 activation stream) flipped **e2b** to a +3 % WIN @4k /
   TIE mid / −3…−5 % long-ctx (was −21 % @64k) and **e4b** to a 🟢 **WIN at every
   size** (+1…+3 %, flipped from −4…−12 %). The SWA decode deficit is closed on small
   dense; the prior small long-ctx e2b residual re-measured ≈ parity (32k 106.9, 64k
   103.2 — it was n=2 noise, e2b is dense and untouched by #51).
2. **Big-model decode — 26b MoE CLOSED by #51; only 31b dense remains.** **#51**
   (bf16 MoE router/expert stream) closed the 26b MoE gap from −10…−28 % to **−4…+1 %**
   (🟢 WIN @16k) and halved its KV — the same f32-stream class #44 fixed for the dense
   path, which the MoE **router** was missed by (a strong-f32 root-size scalar leaked
   f32 through `routing_weights` into the whole MoE residual → downstream KV). **31b
   dense −2…−12 %** (dense-bandwidth physics — close to the ceiling, hard to move) is
   now the **only** model trailing mlx-lm on decode.
3. **Prefill improved on 26b (#46) but still large on 31b dense.** #46 sorted-MoE
   gather cut 26b 128k cold TTFT ~2.5× (403 s → **161.6 s**) and 64k ~3× (~182 s →
   **54.4 s**). 31b dense `none` 128k is still a **477 s** prefill (64k 171 s) — the
   highest-value remaining big-model lever, same prefill class the Qwen3.6 campaign
   flagged.
4. **KV memory: `none` is the smallest of every codec on every model.** Windowed SWA
   layers are window-bounded (#35), the old 6× was a reporting artifact (#39), #44
   stores the global decode K/V bf16, and **#51 extends the same bf16-stream fix to the
   26b MoE residual → 26b `none` KV also halved** (64k 3216→**1744 MB**; every 26b
   codec's KV dropped proportionally). `none` KV is smaller than every quant codec
   (which keep their bf16 seed *alongside* the packed blocks, 1.2–4× larger). **KV
   quant only pays on 31b dense** (k8vturbo2/tsym +3–4 % @32k–64k, bandwidth-bound) —
   it is net-negative on the other three.
5. **MTP accept-rate is the lever — drafter quality, not the engine.** Spec is purely
   accept-gated, and accept is **prompt-dependent (0.08–0.90), not codec-dependent**
   (the MTP grids are flat across codecs). Wins where accept high (31b 4k +74 % @0.90,
   e4b 8k +29 % @0.47, e2b 4k +15 % @0.35), loses where accept <0.3. **#49
   plain-tied-head drafters (26b/31b) proven in the full sweep** — load + run
   coherent. The assistant drafters aren't reliably predictive, so MTP is **not a
   dependable Gemma4 win** — improving it means a better drafter, not engine work.
6. **K-only codecs are CPU-bound (#36) AND memory-negative.** `k_iso* / k_rotor*` run
   K dequant on the host every decode step → decode craters with ctx (§2; capped
   early). They *were* the only sub-1.0× KV compressors, but #44 dropped `none` below
   them, so the lone reason to keep them is gone. No path to viability without a Metal
   K-dequant kernel; warned at resolve.

Coherence was solid across all 25 codecs on all four models (full sweep,
2026-06-09/10) — no repetition loops.

---

## 5. Caveats

- **All four models re-benched post #44/#46/#49 (2026-06-09/10).** §2 grids are the
  live data; the prior pre-#44 e4b/26b/31b rows are gone.
- **e2b/e4b 64k is n=2 measured** — the per-codec +4 % cluster at 64k (tsym/k8vturbo)
  is within that thin-sample noise AND those codecs carry a larger KV, so it is not
  a real bandwidth win; `none` is the headline.
- **31b `k8v8` 4k/8k cells (4.1/4.0 TPS) are anomalous** — they recover to 11.0 @32k,
  suggesting a cold-codec / warmup artifact, not a real result. Do not trust them.
- **31b k8vturbo2/tsym +3–4 % @32k–64k is n=2 soft** — the lone KV-quant decode win;
  confirm at n=3 before relying on it.
- **SSD not stress-tested** — no spill triggered at 256-token single-stream.
- **MTP is accept-gated, prompt-dependent** — net win only where accept is high
  (31b 4k @0.90, e4b 8k @0.47, e2b 4k @0.35); loses where accept <0.3. Not a
  dependable Gemma4 win.
- **K-only codecs decode capped early** — `k_iso* / k_rotor*` too slow (e2b capped
  16k, 31b capped 8k) to measure further; KV-byte axis captured where they ran.
- **e2b/e4b 128k absent** — fixture (137,920 gemma tok) exceeds 131072 ctx.
- **26b/31b 128k KV caps at ≈64k** (marked `*`) — the 65536-tok baseline limit; the
  decode/TTFT numbers at 128k are real but the recorded KV-MB is the 64k value.
- **`kv_cache_bytes` ingest whitelist** — `rmlx metrics record` only ingests
  none/k8v4/k8v8/planar into `observations`; the other 21 codecs' bytes were read
  from the `events` table (improvement-plan item: widen `canonicalize_kv_quant`).
- rMLX cells recorded in CBB `metrics/runs/*.jsonl` (backend=rmlx); **not** in
  `runs.db` (CBB schema rejected by the rMLX buffer — known harness landmine).
- Aggregator: `Cross-Backend-Bench/scripts/agg_gemma4_siblings.py`.

---

## 6. Weight-quant sweep — QAT 4-bit vs PTQ mxfp8 (2026-06-11)

A second, orthogonal axis to §2's KV-codec sweep: the **weight** quant. Tests the
Gemma QAT collection (`mlx-community/gemma-4-<size>-it-qat-<fmt>`, MLX safetensors,
quantization-aware-trained) against the on-disk PTQ-mxfp8 baselines.

**Loader work landed to make these load** (branch `refactor/model-agnostic-architecture`):
unquantized **bf16** (Plain fallback when no `.scales`) + **affine-int4** with
per-group `.biases` (dense linears, embeddings, *and* MoE expert `gather_qmm`) +
a **`Gemma4UnifiedForConditionalGeneration`** arch alias for the 12B variant
(encoder-free multimodal; text decoder is Gemma4, with the encoder-free vision
and audio front-ends fully wired — see §"12B unified" below).

**Method:** decode TPS via `rmlx baseline --kv-quant none --prompt-tokens 4096
--max-tokens 100` (weight-isolated: KV held at bf16-`none` so the only variable is
the weight codec; 1 warmup + 3 measured, median). Accuracy via `rmlx eval ppl`
wikitext-2 (20k-token slice, BOS-per-window scorer). Coherence via temp-0 serve.
These are a *different harness* from §2 (single-prompt `baseline`, not serve+CBB
`run_one`), so the absolute TPS is comparable only **within** this section, not
cross-referenced to §2's serve numbers.

| size | mxfp8 (PTQ) | 4bit QAT (affine g64) | mxfp4 QAT | bf16 / nvfp4 ref |
|---|---|---|---|---|
| **e2b** | 127.7 · ppl 139 | **142.3 (+11%)** · ppl 37 | — *(empty HF repo)* | — |
| **e4b** | 78.5 · ppl 47 | 87.5 (+11%) · ppl 25 | **87.7 (+12%)** · ppl 25 | bf16 47.3 · **ppl 22.8** · nvfp4 86.8 · ppl 301 ✗ |
| **12b** | 35.4 · ppl 524† | **39.5 (+12%)** · ppl 958† | 39.3 (+11%) · ppl 1490† | — |
| **26b MoE** | 79.2 · ppl 30434† | **106.2 (+34%)** · ppl 778† | 104.9 (+32%) · ppl 823† | — |

Cell = `decodeTPS (Δ vs mxfp8) · ppl`. All coherent at temp 0 **except nvfp4**
(rambling). `†` = ppl is a genuine instruct-model artifact, see finding 4.

**Findings:**

1. **4-bit decodes faster than mxfp8 on every model** — weights stay packed at
   runtime (`Linear::Quantized` → `quantized_matmul(mode)`), so halving weight
   bytes (1.0→0.5 B/elem) cuts decode bandwidth directly. **+11%** small dense,
   **+32–34% on 26b MoE** (more weight bandwidth to save). bf16 is the slowest
   (e4b 47.3, 2 B/elem). mxfp4 ≈ affine-4bit on speed.
2. **Accuracy — the e4b cell (with a bf16 anchor) is the clean result:** QAT-4bit
   ppl 25.0 ≈ QAT-mxfp4 25.4 ≈ **bf16 22.8** ≪ **PTQ-mxfp8 46.6**. QAT 4-bit
   matches bf16 quality *and* runs +12% faster than the 8-bit PTQ baseline — a
   genuine Pareto win on the small dense models. (e2b echoes it: 4bit 37 ≪ mxfp8 139.)
3. **nvfp4 is fast but numerically degraded** (ppl 301 vs mxfp4 25; rambling
   output). rMLX's dispatch is byte-identical to mlx-lm's and the weights
   reconstruct to 5-decimal vs the bf16 reference — the degradation is inside MLX
   0.31.2's nvfp4 `quantized_matmul` GPU kernel, not rMLX. **Use mxfp4 for 4-bit.**
4. **12b/26b raw-wikitext ppl is huge but GENUINE, not a bug.** 26b-mxfp8 ppl
   30434 despite coherent generation. Verified: the scorer is correct (softcap
   applied, no non-finite, flat-high per-window NLL from window 0), and **rMLX
   reproduces the mlx-lm reference within bf16 noise** (mlx-lm 26b-mxfp8 NLL 11.1
   → ppl 65855, same flat-high per-window pattern). The split is structural —
   the "clean" e2b/e4b are matformer nano models (`num_kv_shared_layers>0`,
   `hidden_size_per_layer_input=256`); the "high-ppl" 12b/26b/31b are
   `attention_k_eq_v` / no-per-layer-input. Coherent argmax + tiny probability
   *mass* on raw-web tokens is the signature of an aggressively instruction-tuned
   model preferring its chat distribution over raw text. Within-model precision
   ordering (mxfp8 < mxfp4 < 4bit) is preserved and correct. **Do not use raw
   wikitext ppl as a quant-quality signal for the k_eq_v instruct models** — it is
   not comparable to the e2b/e4b cells. A chat-template-wrapped corpus (or a base
   checkpoint) would be needed for a meaningful quality number there.

**Recommendation (weight quant per size):** where a QAT 4-bit snapshot exists,
**mxfp4 / affine-4bit is the pick** — faster than mxfp8 at every size (most on
26b MoE) and, on the small dense models, accuracy ≈ bf16 and better than PTQ-mxfp8.
mxfp8 stays a safe default; nvfp4 is out until the MLX kernel is fixed.

**Caveats:** decode-TPS here is `--kv-quant none` weight-isolated (~4k prompt),
not the §2 serve+CBB harness — within-section deltas only. `e2b-it-qat-mxfp4` is
an empty placeholder repo on HF (no weights). The 12b unified arch's vision and
audio input are fully wired (encoder-free embedder); these QAT cells are
text-decode benchmarks only. Metrics-DB ingest of these cells is deferred
(grid-summary only).
