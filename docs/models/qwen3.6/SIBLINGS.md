# Qwen3.6 — Sibling-Backend Champions (Phase A)

> Companion: [`rMLX.md`](rMLX.md) — rMLX full matrix + standing-vs-champion.
> This file covers sibling backends only.

**Family:** `qwen3_5_moe` (`Qwen3_5MoeForCausalLM` / `Qwen3_5MoeForConditionalGeneration`)
**Stage 1 (Phase A) collected:** 2026-06-06
**Machine:** Apple M5 Max, 128 GB unified memory, macOS 26.4.1 (Darwin 25.4.0)
**Protocol:** batch=1 single-stream; temp=0 (greedy); `max_tokens=256`;
n=5 measured (4k/8k/16k/32k), n=2 measured (64k/128k); 1 warmup before each backend.
All medians computed from full n; 95% CI via bootstrap (2000 samples).

---

## 1. Snapshots in Scope

| Snapshot (basename) | Weight quant | Arch | Role | Disk |
|---|---|---|---|---|
| `mlx-community__Qwen3.6-35B-A3B-8bit` | affine int8 (g64) | Qwen3.5-MoE | **Base / bench target** | 35 GB |
| `mlx-community__Qwen3.6-35B-A3B-MTP-5bit` | affine 5-bit | Qwen3.5-MoE | MTP drafter _(Stage 2)_ | 572 MB |
| `z-lab__Qwen3.6-35B-A3B-DFlash` | DFlash weights | Qwen3.5-MoE | DFlash drafter _(Stage 2)_ | 904 MB |
| `Dogacel__specdrift-qwen3.6-35b-a3b-eagle3` | Eagle3 weights | Qwen3.5-MoE | Eagle3 drafter _(Stage 2)_ | 398 MB |
| `z-lab__Qwen3.6-27B-PARO` | ParoQuant rotation (~4-bit effective) | Qwen3.5-MoE | Alt-weight (non-comparable) | 18 GB |

**Notes on alt-weight:**
- `z-lab__Qwen3.6-27B-PARO` is a **different model** (27B parameters, PARO rotation quantization) — not weight-comparable to the 35B 8-bit base. Benched as a separate non-comparable row.
- `mlx-community__Qwen3.6-35B-A3B-8bit` is the **weight-comparable anchor** for mlx-lm, mlx-lm-turboquant, and oMLX.
- ollama uses `qwen3.6:35b-a3b-coding-mxfp8` (mxfp8, not affine int8) — flagged as non-weight-comparable.
- All drafter snapshots listed above are Stage 2 only.

---

## 2. Sibling Champion Table (Phase A — Decode TPS)

**Weight-comparable group (affine int8, 35B):** mlx-lm, mlx-lm-turboquant, oMLX.
**Non-comparable:** ollama (mxfp8, 35B), paroquant (PARO ~4-bit, 27B). Shown as separate rows; excluded from champion designation.

Decode TPS = median of n measured runs at steady-state (tokens 2…end).
95% CI via bootstrap. All runs: `max_tokens=256`, temp=0.

### 2a. Decode TPS — weight-comparable tier (affine int8 35B)

| Prompt | mlx-lm (`04a1910`) | mlx-lm-turboquant (`67db9af`) | oMLX (`2169285`) | Champion |
|---|---|---|---|---|
| 4k (4 096 tok) | 82.49 [82.07–83.09] | **84.88 [83.05–85.73]** | 61.75 [51.06–63.27]¹ | **mlx-lm-turboquant** |
| 8k (8 192 tok) | 81.89 [80.82–83.09] | **83.93 [82.59–84.28]** | 60.98 [46.99–61.56]¹ | **mlx-lm-turboquant** |
| 16k (16 381 tok) | 76.38 [71.40–78.43] | **79.20 [78.42–80.49]** | 57.01 [35.10–57.71]¹ | **mlx-lm-turboquant** |
| 32k (32 764 tok) | 71.12 [70.28–72.39] | **73.56 [72.50–74.43]** | 51.10 [19.44–51.88]¹ | **mlx-lm-turboquant** |
| 64k (65 528 tok) | 63.33 [62.63–64.03] | **65.09 [64.54–65.64]** | 25.51 [8.30–42.72]¹² | **mlx-lm-turboquant** |
| 128k (131 052 tok) | 49.51 [49.23–49.78] | **50.81 [50.18–51.44]** | 16.85 [3.57–30.12]¹² | **mlx-lm-turboquant** |

¹ oMLX run-1 of each prompt size shows drastically lower TPS (first-run metal kernel warm-up / KV growth).
The median includes run-1; CI lower bound reflects this. Runs 2–5 are stable (~62/61/57/51 TPS).
² Only 2 runs for 64k/128k; oMLX run-1 is especially bad (~8 TPS and ~3.6 TPS), making both the median and CI unreliable for these cells. Treat oMLX at 64k+ as anomalous.

### 2b. Prefill TPS (first-TTFT only, from run-1 cold prefill)

oMLX enables auto-prefix caching — runs 2+ show TTFT of ~30–140ms regardless of prompt size (cache hit).
Only run-1 TTFT reflects actual prefill. All other backends show increasing TTFT with context.

| Prompt | mlx-lm | mlx-lm-turboquant | oMLX (run-1 only) | ollama (non-comp.) | paroquant (non-comp.) |
|---|---|---|---|---|---|
| 4k | 2 687 t/s | 2 654 t/s | ~139k t/s (cached)³ | 1 945 t/s | 884 t/s |
| 8k | 3 824 t/s | 3 788 t/s | ~260k t/s (cached)³ | 3 342 t/s | 739 t/s |
| 16k | 3 297 t/s | 3 072 t/s | ~403k t/s (cached)³ | 3 645 t/s | 655 t/s |
| 32k | 2 804 t/s | 2 774 t/s | ~633k t/s (cached)³ | 3 608 t/s | 595 t/s |
| 64k | 2 102 t/s | 2 035 t/s | ~883k t/s (cached)³ | 2 250 t/s | 490 t/s |
| 128k | 1 179 t/s | 1 227 t/s | ~1.1M t/s (cached)³ | 1 761 t/s | 354 t/s |

³ oMLX prompt-caching makes TTFT 29–116ms regardless of context length. These numbers reflect cache-hit speed,
not actual prefill throughput. Real first-prefill TPS not measured for oMLX in this bench.

### 2c. Non-comparable rows (for reference only)

| Prompt | ollama 0.23.2 (mxfp8 35B) | paroquant `c049a8a` (PARO ~4-bit 27B) |
|---|---|---|
| 4k | 179.15 [175.75–179.82] | 28.55 [28.08–28.65] |
| 8k | 165.12 [163.26–166.06] | 27.97 [27.88–28.05] |
| 16k | 143.94 [141.71–144.44] | 26.51 [26.44–26.71] |
| 32k | 147.15 [146.93–148.85] | 24.39 [24.30–24.41] |
| 64k | 127.68 [126.02–129.33] | 21.61 [21.59–21.64] |
| 128k | 105.80 [105.44–106.16] | 17.80 [17.73–17.87] |

**ollama:** ~2× apparent decode TPS, but the comparison is **contaminated** (see §2d). ollama is `mxfp8`, **37 GB on disk ≈ the 35 GB int8 tier** — roughly the same bytes/param, so the earlier "mxfp8 halves the bytes" claim is **wrong**. The apparent advantage is NOT a quant/bandwidth-footprint effect.

**paroquant:** 27B model with Python/MLX rotation-kernel dispatch (no compiled Metal kernel for PARO rotations). Low TPS (~17–28) reflects Python overhead in the rotation pass, not model size alone. Stable and coherent.

---

## 2d. Decode-efficiency analysis — the ollama "2×" investigated (2026-06-06)

Probe goal: explain ollama's ~2× decode TPS and decide whether rMLX can match it.
Findings (from `Cross-Backend-Bench/metrics/runs/*.jsonl` + a 3-run rMLX
`baseline` probe on `mlx-community__Qwen3.6-35B-A3B-8bit`, 4k, max_tokens=256):

**The 2× is contaminated, not a clean win:**

1. **Same byte footprint.** ollama = `mxfp8`, **37 GB**; MLX int8 = **35 GB**.
   ~1 byte/param both. The 2× is **not** a quant/bandwidth-footprint effect.
2. **ollama ignored `max_tokens`.** It generated ~455–485 tokens to EOS every
   cell (thinking model; CBB runner doesn't pass `num_predict`), while mlx-lm /
   mlx-lm-tq / rMLX honored 256. Unequal work — decode *rate* is still a rate,
   but the cells aren't equal-effort.
3. **ollama TTFT is cache-flat** (~72–131 ms for 4k→64k, then 74 s at 128k) →
   prefix caching / context not reprocessed per run. Its prefill numbers are
   meaningless, and its decode may attend a shorter effective KV (cheaper/token)
   at long context.

**Apples-to-apples decode TPS @ 4k (same 8-bit weights, controlled 256-token
runs where the backend obeyed the cap):**

| Backend | decode TPS @ 4k | vs mlx-lm |
|---|---|---|
| mlx-lm | 82.1 | — |
| mlx-lm-turboquant | 85.7 | +4% |
| **rMLX (kv none)** | **98.2** | **+20%** |
| ollama (uncontrolled, ~485 tok, cache) | 175.8 | not trustworthy |

**rMLX already leads the MLX tier on decode** (+20% vs mlx-lm). At ~3.5 GB/token
active and ≈98 TPS, rMLX uses ~340 GB/s — roughly half of the M5 Max roofline,
so headroom remains. ollama's implied ~610 GB/s is near/above roofline →
consistent with its number being inflated by (2)+(3), not a real 2× of honest
work. **Conclusion: ollama's edge, where real, is llama.cpp's Metal MoE
bandwidth utilization (kernel efficiency), NOT quant — recoverable in rMLX on the
same weights. The literal "2×" is mostly an artifact of uncontrolled generation
+ caching.**

**Prefill gap — investigated and RESOLVED (was a chunk-size bug + a flawed
baseline):** rMLX `baseline` @ 4k once showed **TTFT ≈ 6.9 s, prefill ≈ 580
tok/s**, attributed at the time to a "~40–50× vs mlx-lm (144 ms)" deficit. Both
numbers were wrong. The 6.9 s was a GatedDeltaNet chunk-size bug (prefill pinned
at chunk=64); fixing it (GDN kernel-always → chunk 2048) brought 4k TTFT to
≈ 1.06 s (~3050 tok/s). The "mlx-lm 144 ms / 28000 tok/s" baseline is
non-physical — a direct mlx-lm 0.31.3 run on this snapshot measures ≈ 2.7k–3.6k
prompt tok/s, i.e. mlx-lm is only ~1.1–1.2× faster. Prefill is bandwidth-bound;
rMLX is at mlx-lm parity. No structural prefill deficit remains.

**To settle ollama honestly (later):** re-measure with `num_predict=256`, prompt
caching disabled, and confirm full-context attention — then its decode TPS is
directly comparable. Until then treat ollama's column as uncontrolled.

---

## 3. Skips — Non-Benched Backends

| Backend | Reason |
|---|---|
| `mistral.rs` | No CBB runner; no bench script. Would require building a new runner — out of scope for Phase A data collection. |
| `isoquant` | No CBB runner; no bench script. Skip-with-reason: no live server endpoint available. |
| `rotorquant` | No CBB runner; no bench script. Skip-with-reason: no live server endpoint available. |
| `llama.cpp` | GGUF runtime — weight format incompatible (GGUF ≠ MLX safetensors). Weight-quant mismatch vs affine-int8 baseline. No CBB runner. |
| `mlx-vlm` | Qwen3.6 is a text-only model family; mlx-vlm is a VLM reference. No text-only serving path applicable here. |
| `dynamo` | Not verified as OpenAI-compatible; no CBB runner. Pre-classified skip. |
| `1bit-eval-scratch` | Eval harness, not a serving backend. No live inference server. |
| `experiments-kv-cache-compression` | Research / data-only repo; no serving endpoint. |
| `llama-cpp-turboquant` | TurboQuant Metal kernel variant (llama.cpp base). GGUF runtime — weight incompatible. No CBB runner. |
| `johndpope-llama-cpp-turboquant` | Same as above. |
| `turboquant_plus` | TurboQuant Metal kernel variant. No serving endpoint. No CBB runner. |
| `multi-turboquant` | TurboQuant Metal kernel variant. No serving endpoint. No CBB runner. |

---

## 4. Stage 2 Placeholder — rMLX Full Matrix

**Pending Stage 2. Do not fill this section until Stage 2 is executed.**

```
rMLX full matrix (Stage 2):
  - Base: mlx-community__Qwen3.6-35B-A3B-8bit, kv-quant none → baseline
  - KV sweep: all arch-allowed named variants (MoE arch-guard pre-skips applied)
  - SSD tier: off/on at 64k/128k for top-3 KV cells
  - Speculative: MTP-5bit, DFlash, Eagle3 × {none, k8v4, k8v8, top-1 from sweep}
  - Prompt sizes: 4k / 8k / 16k / 32k / 64k / 128k
  - Standing-vs-champion table: decode + prefill, per prompt size, 🟢/🟡/🔴
```

**Arch-guard pre-skips for MoE (Qwen3.5-MoE):** the following KV variants are hard-rejected
by `k_below_8bit()` in `rmlx-kv-quant/src/quant.rs` and will be logged as skip(arch-guard)
without a probe: `tsym3, tsym4, iso3_sym, iso4_sym, k_iso3, k_iso4, rotor3_sym, rotor4_sym,
k_rotor3, k_rotor4, rotor_k_3_asym_*, rotor_k_4_asym_*` (~13 variants), plus `planar_k`.

---

## 5. Notes

### Machine and environment
- **Chip:** Apple M5 Max, 128 GB unified memory
- **OS:** macOS 26.4.1 (Darwin 25.4.0)
- **Date:** 2026-06-06
- **Bench start:** ~03:18 UTC; all four comparable backends + paroquant completed in ~140 min wall-clock.

### Backend SHAs
| Backend | SHA / Version |
|---|---|
| mlx-lm | `04a1910` (pulled ff-only 2026-06-06) |
| mlx-lm-turboquant | `67db9af` (already up to date) |
| oMLX | `2169285` (pulled ff-only 2026-06-06) |
| ollama | version `0.23.2` |
| paroquant | `c049a8a` (pulled ff-only 2026-06-06) |

### Coherence gate results
All five backends passed the coherence gate at 8k prompt size:

| Backend | 8k output sample | Pass/Fail |
|---|---|---|
| mlx-lm | `The user wants to find the top three projects by README length…` | PASS |
| mlx-lm-turboquant | `The user wants to find the top three projects by README length…` | PASS |
| oMLX | `\nThe user wants to find the top three projects by README length…` | PASS |
| ollama | `Here's a thinking process:\n\n1.  **Analyze User Input:**…` | PASS |
| paroquant | `Here's a thinking process:\n\n1.  **Analyze User Input:**…` | PASS |

No backend produced empty output, repetition loops, or broken punctuation.
Note: ollama and paroquant output reasoning/thinking preambles (the model emits `<think>…</think>` style output). Both are coherent extended-thinking outputs, not degeneration.

### oMLX caching anomaly
oMLX appears to have auto-prefix caching enabled. From run-2 onward of each prompt size,
TTFT is 27–140ms regardless of prompt length (4k → 128k), indicating the full KV prefix is
served from cache. Consequence: run-1 TPS at each new prompt size is much lower (first-time KV
allocation + prefill), while runs 2-5 are stable but do NOT reflect independent full-context
prefill. The 5-run median is dominated by the stable cached runs; CI lower bound reflects the
single cold-start outlier. The oMLX decode TPS numbers are valid (steady-state decode is after
TTFT regardless of caching); the prefill TPS numbers from oMLX are not meaningful in this bench.

### Thermal / caching caveats
- No `purge` was run between backends. macOS buffer cache may hold recently-accessed safetensors pages.
- Bench ran sequentially without thermal cooling pauses. M5 Max sustained performance is well-characterized; no thermal throttle detected (decode TPS stable within each backend's run series).
- The mlx-lm-turboquant KV cache flag `--kv-cache-quantization 8,4 --quantized-kv-start 0` is a "fake asymmetric" flag (the CBB script note) — mlx-lm-turboquant's 8,4 is not true asymmetric K8/V4 as implemented in rMLX. The decode TPS advantage over plain mlx-lm is real but modest (~2–4%), reflecting reduced KV bandwidth at large context.

### paroquant notes
- Paroquant `_serve_mlx()` path wraps `mlx_lm.server` with a patched `load()` that applies PARO rotation at load time. The rotation kernels run through Python/MLX (not compiled Metal). This explains the ~28 TPS decode at 4k for a 27B model (expected ~85+ with a compiled Metal kernel). The rotation overhead dominates each forward pass.
- Paroquant 128k prefill takes ~370 seconds (TTFT 370 600ms). This is expected given the Python rotation overhead on 131k tokens.
- Paroquant is benched purely for data completeness. It is non-weight-comparable (27B, different quant scheme) and non-competitive at current Python-kernel speed.

### Prompt sizes
All six fixtures used: `longctx_{4k,8k,16k,32k,64k,128k}.json` (4 096 / 8 192 / 16 381 / 32 764 / 65 528 / 131 052 prompt tokens). Sub-4k and 256k prompts are not in the fixture set — deferred, not silently omitted.
