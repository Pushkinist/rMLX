# Qwen3.6 — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable affine int8 35B tier) = **mlx-lm-turboquant**,
> decode TPS @ 4k/8k/16k/32k/64k/128k = 84.88 / 83.93 / 79.20 / 73.56 / 65.09 / 50.81.

**Family:** `qwen3_5_moe` · **Machine:** Apple M5 Max, 128 GB, macOS 26.4.1
**Protocol:** batch=1, temp=0, `max_tokens=256`; n=5 measured (4k/8k/16k/32k),
n=2 (64k/128k), 1 warmup discarded; decode-TPS median + 95% CI (bootstrap);
binary `release-perf`. Bar (§0.1): WIN / TIE-on-CI-overlap / LOSS.
Step-by-step execution — one cell per run.

> **Status: Stage 2 COMPLETE** (Phases B–E + PARO). Step-by-step, one cell per run.

## 0. TL;DR

- **rMLX WINS decode vs the champion (mlx-lm-turboquant) at every prompt size**,
  +12–15% (kv-none), confirmed path-equivalent to the sibling harness (−0.2%).
- **Best KV: rotor3 / iso3** (rotation codecs) — +1.9% over none at 128k. Turbo
  codecs *lose* at long ctx (V-dequant overhead). At 8k all 8-bit-K codecs tie.
- **Prefill/TTFT — FIXED, now at parity with mlx-lm.** The GatedDeltaNet
  recurrence flipped from the `gated_delta_step_gpu` kernel to a lazy ops-graph at
  T≥256, which pinned the prefill chunk at 64 and made rMLX ~7× slower than its
  own potential. Fix: GDN always uses the kernel → chunk rises 64→2048 (mlx-lm's
  `prefill_step_size`). A kv-none warm-TTFT sweep measured **4.0–4.2× lower TTFT**
  (4k 4240→1065ms, 8k 9008→2136ms) with no decode change, no Metal watchdog
  through 64k. **The earlier "~40–50× slower than mlx-lm / 4k TTFT 144ms" claim
  was a measurement error** — a direct mlx-lm 0.31.3 run on this exact snapshot +
  prompts measures **2711–3606 prompt tok/s** (4k/8k), i.e. ~1.1–1.2× of rMLX's
  ~3050 tok/s; the cited 28000 tok/s exceeds the M5 Max bandwidth ceiling and is
  non-physical. Prefill is bandwidth-bound; both backends sit at ~the same level.
  The §2a grid below is the pre-fix `rmlx baseline` record. See §4 ①.
- **Speculative barely helps:** MTP +4.2% (best drafter), DFlash −37%, Eagle3
  −39% (now works after the crash fix, but low accept ~0.27 → net loss). Spec-loop
  overhead eats the accept gain.
- **SSD tier** decode-neutral (−0.5% / −3.1%); spill now persists correctly; the
  **hydrate-doesn't-skip-prefill** gap remains (#9, deferred).
- **PARO 27B:** rMLX runs it (26.3 TPS) but −5.6% vs the paroquant reference.
- **Bugs filed + FIXED in 0.1.1:** Eagle3 MoE crash (#8), SSD spill GPU-stream
  (#10), baseline prompt cap (#11). Still open: SSD hydrate-no-prefill-skip (#9,
  enhancement). CBB→runs.db schema reject (harness). See plan §10.

---

## 1. rMLX snapshots benched

| Snapshot (basename) | Weight quant | Role | Resident |
|---|---|---|---|
| `mlx-community__Qwen3.6-35B-A3B-8bit` | affine int8 (g64) | base / comparable | ~35 GB |
| `z-lab__Qwen3.6-27B-PARO` | ParoQuant 4-bit krot=8 (27B) | alt-weight (non-comparable) | ~14 GB |
| `…-MTP-5bit` / `…-DFlash` / `…-eagle3` | drafters | speculative (Phase E) | _pending_ |

---

## 2. rMLX full matrix

### 2a. Phase B — baseline (`--kv-quant none`, base model)

| Prompt | decode TPS (median [95% CI]) | TTFT | prefill TPS | peak RSS |
|---|---|---|---|---|
| 4k | **97.68** [96.94–97.91] | 6857 ms | 596 | 35.4 GB |
| 8k | **94.28** [89.01–94.93] | 13 847 ms | 586 | 35.4 GB |
| 16k | **89.33** [88.88–89.94] | 30 619 ms | 552 | 35.8 GB |
| 32k | **83.32** [82.93–83.75] | 61 138 ms | 549 | 36.4 GB |
| 64k | **74.07** [74.07–74.08] | 132 359 ms | 495 | 41.5 GB |
| 128k | **57.12** [57.12–57.13] | 312 314 ms (cold) | 419 | 40.8 GB |

_Cells 4k–64k via `rmlx baseline`; 128k via `rmlx serve`+`run_one` (baseline caps
prompts at 64k — see plan §10). 128k prefill ≈ 5m12s for 130 810 tokens._
_**Paths validated equivalent**: 4k baseline 97.68 vs serve+run_one 97.47 (−0.2%)
— so these rows compare directly against the serve+run_one siblings; the wins are
real, not a measurement artifact._

### 2b. Phase C — KV-variant sweep

**Ranking pass @ 8k** (n=3, `baseline`), sorted by decode TPS. All coherence PASS,
no skips (planar/planar3 ARE MoE-allowed):

| KV variant | decode TPS @8k (median [range]) | ttft_ms | note |
|---|---|---|---|
| k8vturbo2tcq | 96.10 [95.41–96.52] | 14 074 | tcq ttft penalty, no decode gain |
| k8vturbo3tcq | 96.04 [95.42–96.31] | 14 352 | " |
| k8vturbo3 | 96.02 [95.92–96.17] | 13 617 | tightest variance |
| rotor3 | 96.02 [95.50–96.15] | 13 891 | |
| iso3 | 96.00 [95.87–96.30] | 13 881 | |
| k8vturbo2 | 95.87 [95.68–96.24] | 13 579 | |
| planar3 | 95.86 [95.42–96.08] | 13 529 | |
| rotor4 | 95.85 [95.78–96.16] | 13 912 | |
| planar | 95.72 [95.05–96.03] | 13 468 | |
| iso4 | 95.71 [94.97–96.23] | 15 598 | higher prefill cost |
| k8v4 | 95.50 [94.88–95.66] | 13 567 | |
| k8v8 | 95.23 [94.92–95.87] | 13 519 | |
| none (bf16) | 94.28 [89.01–94.93] | 13 847 | unquantized KV |
| rot_k_tq4v | 89.60 [89.20–90.26] | 13 398 | ~7% slower (tq4 V cost) |
| rot_k_v4g64 | 58.21 [58.02–58.34] | 13 475 | V-side dequant decode cost |
| mixed_k8g128_v4g64 | 57.44 [57.30–57.69] | 13 471 | " |
| mixed_k8g128_v8g128 | 56.50 [56.26–56.51] | 13 449 | " |

**Read:** at 8k the KV cache is tiny vs the weights, so all 8-bit-K codecs are
within <1% (bandwidth-bound on weights). The 8k ranking is NOT predictive of the
long-context winner — KV-codec savings only pay off where the KV cache is large.

**Arch-guard pre-skips (MoE, no probe):** tsym3, tsym4, iso3_sym, iso4_sym,
k_iso3, k_iso4, rotor3_sym, rotor4_sym, k_rotor3, k_rotor4, rotor_k_3_asym_*,
rotor_k_4_asym_*, planar_k.

**Carry plan (step-by-step, adaptive prune — plan §7.0):** the informative size
for KV choice is **128k** (KV-dominated). Carry the top 3-bit-V codecs to 128k
(serve+run_one); confirm across codec families, then conclude.

**Carry results @ 128k** (vs none@128k = 57.12):

| KV variant | decode TPS @128k (median) | Δ vs none | coherence |
|---|---|---|---|
| none (bf16) | 57.12 | — | PASS |
| **rotor3** | **58.19** | **+1.9%** | PASS |
| **iso3** | **58.21** | **+1.9%** | PASS |
| k8vturbo3 | 55.57 | −2.7% | PASS |
| k8vturbo2tcq / k8vturbo3tcq | _pruned_ | (turbo family — tracks turbo3 −2.7%) | |

**Finding (forming): codec FAMILY decides the long-ctx outcome, not bit-width.**
- **rotor3 (rotation-based V) BEATS none at 128k (+1.9%)** — the rotation codec's
  dequant is cheap enough that the KV-bandwidth saving nets positive. This is the
  rotation-KV advantage rMLX uniquely ships.
- **k8vturbo3 (turbo V) LOSES (−2.7%)** — its per-token V-dequant over 131k
  positions costs more than the bandwidth saved (compute-bound).
- At 8k (KV tiny) all codecs ≈ none; the split only appears at long ctx.
- `--fused-qk` kernels are default-OFF — turbo's loss may shrink with them on
  (improvement-plan lead).
- **KV-cache memory numbers were inconsistent across runs** (none KV reported
  3.0 GB in one cell, a "bf16≈40 GB" framing in another) → NOT asserting
  compression ratios here; verify KV footprint separately.

### 2c. Phase D — SSD KV tier (off/on)

Enabled via `--kv-ssd-cache-gb 8`. rotor3 @128k:

| Config | decode TPS @128k | Δ | notes |
|---|---|---|---|
| rotor3, SSD off | 58.19 | — | |
| rotor3, SSD on | 57.90 / 56.41 | **−0.5% / −3.1%** | spill async (no decode stall) |

**Finding:** SSD decode overhead is **negligible** (−0.5% original, −3.1% on
re-test, both within noise) — the drain-thread spill doesn't stall decode. SSD's
purpose is **capacity** (KV that outgrows RAM), not speed; at 128k single-stream
(KV ~2.2 GB) it isn't needed. Two bugs were found; one is fixed:
1. **SSD hydrate does not bypass prefill** — `prompt_cache_hits=0` even after a
   successful hydrate (510 blocks, prefix_len 130560); the full 342s prefill still
   ran. The cross-restart prefill-reuse benefit is **not materializing** (the
   hydrated KV isn't matched to the active request's prefix). **Open — #9,
   deferred (feature-sized; ExactOnly policy vs block-aligned hydrate).**
2. ~~Spill GPU-stream WARN~~ — **FIXED in 0.1.1 (#10).** Was
   `eval-for-spill failed: no Stream(gpu,N)` → block not persisted. Re-tested
   post-fix: WARN gone, `kv-spill: block written + indexed` confirmed (2.34 GB
   blocks persisted), decode 56.41 TPS (−3.1%, within noise).

### 2d. Phase E — speculative (drafter × KV anchor)

Verifier = 35B-A3B-8bit, kv none, 4k, temp 0 (greedy accept). Baseline none@4k = 97.68.

| Drafter | decode TPS @4k | speedup vs none | accept-rate | vs champion 84.88 |
|---|---|---|---|---|
| MTP-5bit (block 3, k=4) | 101.77 | +4.2% (1.04×) | 66.4% | +20% |
| DFlash (block 16, k=4) | 61.86 | **−37% (regression)** | 47.2% | −27% |
| Eagle3 (block 5) | 59.61 | **−39% (regression)** | 26.5% | −30% |

_DFlash: the `z-lab__…-DFlash` snapshot loads as a heavy ~full-size MoE (72.6 GB
dual-model peak) — draft cost ≈ verify cost, so spec backfires. Decode-only
number; engine overall (incl. 6.5s prefill) was ~23 TPS. Dead end for this snapshot._

_Eagle3: **the crash is fixed** (PR #8 / 0.1.1 — the drafter KV cache was
hardcoded to 4096; now sized to the verifier `max_seq`). Re-tested post-fix:
256 tokens complete, coherent, no `slice_update` crash. But decode is **59.61 TPS
(−39% vs none)** at a low **26.5% accept-rate** — the small accept can't offset the
spec-loop overhead, so Eagle3 is a net decode loss for this model (same shape as
DFlash). Functional now, but not worth using here without a higher-accept drafter._

**Phase E verdict (post-0.1.1):** all three drafters now RUN (Eagle3 crash fixed),
but speculative is **not a net win** for Qwen3.6: MTP +4.2% (only positive),
DFlash −37%, Eagle3 −39%. The accept-rates (MTP 66%, DFlash 47%, Eagle3 27%)
don't clear the spec-loop overhead. Gains live behind (a) reducing per-round
overhead, (b) a higher-accept / truly-lightweight drafter, (c) block-size / KV-
anchor tuning — deferred to the improvement plan.

**Finding (forming):** MTP spec gives only **+4.2%** despite a 66% accept rate —
the speculative round overhead (draft-model forwards + multi-position verify)
nearly cancels the decode-step savings. With 66% accept + block 3, theory is
~1.5×; rMLX realizes 1.04×. → **rMLX's spec path has overhead to recover**
(improvement-plan target). Spec currently adds little over rMLX's already-winning
kv-none. Testing DFlash / Eagle3 + block-size tuning to see if any drafter clears
the overhead.

---

## 3. Standing vs champion (decode)

| Prompt | rMLX (kv none) | champion (mlx-lm-tq) | standing |
|---|---|---|---|
| 4k | 97.68 [96.94–97.91] | 84.88 | 🟢 **WIN +15%** (CI clear) |
| 8k | 94.28 [89.01–94.93] | 83.93 | 🟢 **WIN +12%** (CI clear) |
| 16k | 89.33 [88.88–89.94] | 79.20 | 🟢 **WIN +13%** (CI clear) |
| 32k | 83.32 [82.93–83.75] | 73.56 | 🟢 **WIN +13%** (CI clear) |
| 64k | 74.07 [74.07–74.08] | 65.09 | 🟢 **WIN +14%** (CI clear) |
| 128k | 58.19 (rotor3) / 57.12 (none) | 50.81 | 🟢 **WIN +14.5%** (rotor3) |

> **Phase B verdict: rMLX baseline (kv none) WINS decode at every prompt size**
> (+12–15%) vs the champion mlx-lm-turboquant. KV-quant variants (Phase C) may
> push further. Prefill/TTFT (once the apparent deficit) is now at mlx-lm parity
> after the chunk fix — see §4 ①.

---

## 4. Gaps & hypotheses (Phase F synthesis → improvement plan)

Ranked by impact:

1. **Prefill / TTFT — RESOLVED; rMLX is at mlx-lm parity.** rMLX prefill was ~7×
   slower than its own potential (NOT the "40–50× vs mlx-lm" originally claimed —
   see below), degrading with length. Root cause: the GatedDeltaNet recurrence
   flipped from the `gated_delta_step_gpu` Metal kernel to a lazy ops-graph at
   T≥256 (~184K nodes at T=256, ~1.47M at T=2048 across 30 GDN layers), which
   pinned the prefill chunk at 64 — so a 4k prompt ran ~64 forward passes + 63
   KV-state evals where mlx-lm runs ~2. Fix: the GDN now always uses the kernel
   (one dispatch, T-loop in registers; chaining across chunks is f32-state-exact
   and matches mlx-lm's `use_kernel=True` default), unblocking a chunk rise to
   2048 (mlx-lm's `prefill_step_size`). A kv-none warm-TTFT sweep measured TTFT 4k
   4240→1065ms (4.0×), 8k 9008→2136ms (4.2×), 16k 19489→4712ms (4.1×); 32k/64k
   complete with no Metal watchdog; the 3-family decode canary is unchanged.
   **Baseline correction:** the original "~40–50× slower / 4k TTFT 144ms" was a
   measurement error. A direct mlx-lm 0.31.3 run on this exact snapshot + prompts
   measures **2711/3301 tok/s @4k (cold/warm), 3606 @8k** — vs rMLX's ~3050 tok/s,
   i.e. mlx-lm is only **1.08–1.18×** faster, not 48×. The cited 28000 tok/s is
   non-physical (exceeds M5-Max bandwidth ~5×). Prefill is bandwidth-bound and
   both backends sit at ~the same level; the residual ~10–18% is ordinary
   cross-impl kernel variance (host-side mask build is one small contributor),
   not a structural deficit. **Investigated + dropped:** moving the post-prefill
   prompt-cache store off the TTFT path — measured at 2.9ms (0.17% of TTFT), no
   measurable win, reverted.
2. **Decode kernel headroom.** rMLX wins the tier but sits ~half of M5-Max roofline
   (≈340 GB/s of ~600). llama.cpp/ollama saturate more. Tighter MoE decode kernels
   (expert gather + dequant fusion) could extend the lead. `--fused-qk` is
   default-OFF — turning it on may also let quant codecs win on decode.
3. **Speculative overhead.** MTP only +4.2% despite 66% accept — the spec-round
   cost (draft forwards + multi-position verify) nearly cancels the savings. The
   Eagle3 MoE KV crash is **fixed (#8 / 0.1.1)** — Eagle3 now runs but at −39%
   (27% accept). Remaining work: reduce per-round overhead; source a higher-accept
   / truly-lightweight drafter (the DFlash snapshot is full-size).
4. **SSD prompt-cache not skipping prefill (#9, open).** Hydrate restores KV but
   `prompt_cache_hits=0` → full prefill still runs. Wiring the hydrated prefix into
   the cache-hit path would turn SSD into a real cross-restart TTFT win (and helps
   #1 for repeated long prefixes). _(The spill GPU-stream bug is fixed — #10 /
   0.1.1.)_
5. **PARO decode** −5.6% vs the paroquant reference — rMLX's PARO path trails the
   native rotation kernels. Lower priority (non-comparable, niche format).

**What's already good (don't "fix"):** decode TPS leads the MLX int8 tier at every
size; rotation KV codecs (rotor3/iso3) are a genuine long-ctx edge; SSD decode
overhead is negligible; coherence solid across all KV variants.

## 5. PARO 27B (non-comparable alt-weight)

rMLX runs `z-lab__Qwen3.6-27B-PARO` (4-bit krot=8, ~14 GB): decode **26.31 TPS @8k**
[26.28–26.31], coherent. vs paroquant sibling 27.97 → **−5.6%**. Different model
(27B) + format — not comparable to the 8-bit champion table; recorded for coverage.
