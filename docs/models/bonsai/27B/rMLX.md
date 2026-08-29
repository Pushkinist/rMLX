# Bonsai-27B (2-bit) — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable 2-bit tier) = **mlx-lm (no KV quant)**, decode TPS:
> **45.1 / 41.7 / 40.6 / 36.8 / 30.2 / 23.0** (4k/8k/16k/32k/64k/128k).

> **⚠️ Correction (2026-07-18) — the prefill / TTFT numbers below were a
> Homebrew-MLX artifact; "prefill loses 2.6–3.4×" is FALSE.** This matrix was
> benched (2026-07-15/16) on a Homebrew `mlx` 0.32.0 bottle that silently shipped
> **zero** Neural-Accelerator (NAX) GEMM kernels on this M5 box — a ~3.8× GPU-matmul
> loss that inflates **prefill only** (decode is matvec/bandwidth-bound and
> unaffected). §3/§4 originally concluded rMLX prefill is 2.6–3.4× slower than the
> siblings and blamed the GDN sequential-in-T recurrence. **Both are wrong.** After
> pinning MLX back to the last NAX-present build (`steel_gemm_fused_nax_*` kernels
> asserted present in the loaded `mlx.metallib`) and re-measuring the `none`-row
> cold TTFT on the identical `longctx_Nk` fixtures (same binary, same prompts, cold
> single prefill), the gap **largely closes**:
>
> | ctx | published (no-NAX brew mlx) | re-measured (pinned, NAX present) | factor |
> |---|---|---|---|
> | 4k | 14.8 s | **4.8 s** | 3.08× |
> | 8k | 32.0 s | **10.7 s** | 3.00× |
> | 16k | 67.9 s | **24.4 s** | 2.79× |
> | 32k | 147.5 s | **53.4 s** | 2.76× |
> | 64k | 335.0 s | **136.6 s** | 2.45× |
> | 128k | 815.4 s | **408.8 s** | 1.99× |
>
> (4k–32k n=3 median, 64k/128k n=1; every run generated the full 256 tokens.) The
> factor shrinks with context because GEMM is a larger fraction of short-prompt
> prefill. Against the (NAX-correct, PyPI-wheel) mlx-lm champion prefill (4.4 / 10.7
> / 20.6 / 43.7 / 109.9 / 317.8 s), rMLX prefill is now **~1.0–1.3× the champion** —
> parity at 4k–8k, a small residual growing to ~1.3× at 128k — **not** a 3× collapse.
> The NAX-regression root-cause profile puts the GDN recurrence at **~2 % of
> prefill** (MLP dominates), so the original "48×256 serial GDN ⇒ 3.2×" arithmetic
> was numerology — and the measured near-parity above confirms it independently.
>
> **Corrected here:** the `none`-row TTFT in §2, the §3 prefill table, the §0
> prefill bullet, and the §4 prefill gap. **Unchanged (NAX-independent, still
> valid):** every decode-TPS and KV-MB figure, and the `k8v4` decode correction
> below. **Still an artifact — re-measure pending:** the TTFT (middle number) of the
> other 24 codec rows in §2; prefill is ~codec-independent, so scale those down by
> the per-column `none` factor above (~3.1× @4k → ~2.0× @128k) for the true
> magnitude.

**Model:** `prism-ml__Ternary-Bonsai-27B-mlx-2bit` —
`Qwen3_5ForConditionalGeneration`, dense ~27B **text tower** of a VLM-shaped
checkpoint (text-only bench), **2-bit affine** (group 128), **GatedDeltaNet
hybrid attention** (`full_attention_interval: 4` → 16 full-attn + 48 linear/GDN
of 64 layers), `head_dim: 256`, **native 262144 context** (plain rope, no YARN).
MTP head declared but ships no `mtp.*` weights → inert. Single snapshot.
**Machine:** Apple M5 Max, 128 GB, macOS 26.5.1 (Darwin 25.5.0) · **Binary:**
`release-perf`, rMLX 0.3.0 (`bench/bonsai-27b` @ `3d83e6f`). **Date:** 2026-07-15/16.
**Protocol:** batch=1, temp=0, `max_tokens=256`; serve once per codec at the 256k
ceiling (`--max-ctx 262144`, lazy-grow ring) + CBB `run_one` load-once for decode
and cold r0 TTFT; **n=3 measured** (4k/8k/16k/32k), **n=1 measured** (64k/128k), 1
warmup `r0` discarded. **Same harness as SIBLINGS**, so rMLX cells compare
directly. KV-MB from serve events `op='kv_cache_bytes'` high-water (the `baseline
--record` path truncates prompts >65536 tokens, so it cannot measure 128k — see
§M). Bar (§3): WIN / TIE-on-noise / LOSS. Cell = `decodeTPS · r0TTFT(s) · KV-MB`.

> **All 25 KV codecs run.** Bonsai-27B is dense `Qwen3_5ForConditionalGeneration`;
> `head_dim=256` satisfies the K-side bit-packing constraint for every codec, and
> the sub-8-bit-K arch-guard did **not** fire on any of the 25 (all loaded
> cleanly). No MTP / speculative grid: the checkpoint declares an MTP head but
> ships no `mtp.*` weights (inert). No §6 weight-quant sweep: one on-disk 2-bit
> snapshot, no QAT siblings.

## 0. TL;DR

- **rMLX `none` LEADS the mlx-lm champion at every context, but by a smaller
  margin than the 8B.** `none` decode **50.9 / 47.7 / 41.7 / 37.9 / 31.3 / 23.7**
  (4k…128k) vs champ **45.1 / 41.7 / 40.6 / 36.8 / 30.2 / 23.0** → **+12.8 / +14.4
  / +2.7 / +2.9 / +3.6 / +3.0 %** (§3). The lead is decisive at short context and
  narrows to ~+3 % (borderline run-to-run noise) from 16k on — nothing like the
  8B's flat +21…+27 %. **Decode is flat with context** (50.9→23.7 across a **32×**
  range, ~2× falloff), because only 16/64 layers are full-attention (KV-growing);
  the other 48 GDN layers hold fixed-size recurrent state, which also compresses
  the whole codec spread.
- **rMLX prefill — near parity (corrected 2026-07-18; first reported as a 2.6–3.4×
  loss).** The original prefill numbers were measured on a Homebrew MLX bottle that
  shipped zero NAX GEMM kernels (~3.8× matmul loss, prefill-only). Re-measured on the
  pinned NAX-present MLX, `none` cold TTFT is **4.8 / 10.7 / 24.4 / 53.4 / 136.6 /
  408.8 s** (4k…128k) — **at parity to ~1.3× the mlx-lm champion** (4.4 / 10.7 / 20.6
  / 43.7 / 109.9 / 317.8 s), not the 2.6–3.4× loss this doc first reported. The GDN
  recurrence is ~2 % of prefill, so the "sequential-in-T GDN ⇒ 3.2×" story was
  numerology (see the correction at the top, §3, §4). Decode is unaffected.
- **Three codec tiers** (§2, §4):
  - **Tier 1 — GPU-fused, fast, viable** (`none`, `k8v8`, `planar`/`planar_k`/
    `planar3`, `k8vturbo2/3`, `k8vturbo2tcq/3tcq`, `tsym3/4`, `iso3/4`, `rotor3/4`):
    real MSL kernels, decode 25–53 TPS. **Several BEAT `none` by +11…+15 % at
    long ctx** — the **tcq pair** (`k8vturbo2tcq` +15 % @128k) and **rotor3/4**
    (+11…+13 % @128k) are the fastest memory-sane codecs. The champion-beating
    decode lives here.
  - **Tier 2 — bf16-mirror, works but no memory win** (the `*_sym` family:
    `iso3_sym`, `iso4_sym`, `rotor3_sym`, `rotor4_sym`): decode reads a full bf16
    seed (huge KV, 2.2–3.4× `none`), the quant path is dormant. Fast raw decode at
    4k–64k, but all four show a **reproducible 128k warm-cache decode stall**
    (aggregate craters to ~1–10 TPS while `itl_p50` implies ~28 TPS) — the matrix
    uses the **cold r0** number as the cell value and footnotes the stall.
  - **Tier 3 — CPU-bound, unusable** (the K-only family: `k_iso3/4`,
    `k_rotor3/4`): sub-8-bit rotation/iso K with **no Metal kernel** → CPU dequant
    fallback → **0.05–8.8 TPS**, GPU idle. Capped.
    > **Superseded for `k_rotor3/4` (run with `--rotor-qjl off`).** The rotor
    > K-only decode is now a fused MSL flash-decode over the packed rotor store
    > (`rotor_flash_decode`, see `docs/KV_QUANT.md`), so the per-step full-prefix
    > CPU dequant that produced these numbers is gone. Re-measured at 4k: Bonsai-8B
    > 1.34 → 17.0 TPS, medgemma-4B 7.37 → 51.8 TPS. The numbers in this table are a
    > pre-kernel snapshot and were **not** re-run on the 27B. Two caveats stand:
    > `--rotor-qjl on` still takes the CPU path (the kernel cannot reproduce the
    > QJL residual) — it was the default when this was recorded and is opt-in
    > now, so the shipped path is the kernel — and `k_iso3/4` is untouched: it keeps its own
    > per-step host restaging.
- **No long-ctx collapse (unlike the 8B).** On the 8B, `iso*`/`*_sym` cratered to
  ~6–13 TPS at 64k; on the 27B they hold **30–36 TPS at 64k**. GDN's shallow KV
  growth avoids the CPU-dequant collapse entirely — a big divergence from the 8B doc.
- **`planar` direction-flip.** `planar` **beats** `none` on the 27B (+1…+9 %,
  growing with ctx) but **lost** to `none` on the 8B. Same build, opposite sign —
  flagged, not diagnosed (§5).
- **4-bit-V is slow *and* broken (see the 2026-07-17 correction below).** The
  `k8v4` decode cost is real — a clean re-measurement that keeps generation
  *below* the KV boundary gives **50.4 / 15.5 / 5.0** TPS at 4k/32k/128k vs a
  same-machine `k8v8` control **45.1 / 37.3 / 21.4** (a genuine, context-growing
  tq4-V cost: `k8v4` ties `k8v8` at 4k, is 2.4× slower at 32k, 4.3× slower at
  128k). `rot_k_tq4v` (48→…→11.7) shares it — that codec has since been
  retired (see `docs/KV_QUANT.md`); every mention of it on this page is a
  historical measurement. `k8v8` (8-bit V) tracks `none`. But
  `k8v4` decode *also* **crashes at the next power-of-two KV boundary** (still
  live), so it cannot generate across a `2^k` boundary. **Avoid 4-bit V here.**

> **Correction (2026-07-17, `626381e`, re: #241).** The `k8v4` row was suspected
> to be a swallowed-crash artifact (#233/#235). Re-measurement refutes that for
> the *numbers* and confirms it for the *runs*. The decode **rates are genuine**:
> clean runs kept below the KV boundary (`max_tokens=200`, every step completes,
> `finish_reason=length`) reproduce the recorded curve, and a same-machine `k8v8`
> control confirms a real tq4-V cost — **so the tq4-V explanation stands.** But
> every recorded `k8v4` cell was a **truncated crashing run**: `k8v4` decode dies
> exactly when `prompt_len + generated` reaches `2^k`
> (`reshape: array of size 0 into (1,4,1,64)`, the #233 class; #238 did **not**
> fix the paged 4-bit-V path), delivering only 242–250 of the 256 requested
> tokens. The per-token rate survived because the death is at the tail. The
> failure is masked to the streaming client (`finish_reason=null` via the
> retry-replay envelope), a #235-class gap on the streaming path. Net: `k8v4` is
> **broken, not merely slow** — a stronger reason to avoid 4-bit V than this doc
> originally gave. The boundary crash needs its own tracking bug. (`rot_k_tq4v`
> shares the tq4-V decode cost; its boundary-crash behaviour was not re-verified.)
- **`none` is the smallest KV of any codec** (bf16, ≈2 B/element): 419 MB @4k →
  8865 MB @128k. Every quantized codec carries a *larger* resident KV (1.20×–3.43×),
  so `none` is both the honest headline number and the memory winner (§2.1).

---

## M. Measurement note (serve-once at the 256k ceiling)

Every codec is served **once** at `--max-ctx 262144` (the Bonsai-27B ceiling) and
all six prompt sizes sweep against the resident **lazy-grown** ring — no per-size
relaunch, matching the dynamic-KV siblings. **The lazy ring is confirmed free**:
serve startup is ~2.5 s despite the 262144 ceiling (`eager preload complete
load_ms=2536`), and the ring grows in chunk-sized increments during prefill
(`KV prefill buffer grow from=65536 to=131072`), not eagerly to the ceiling — a
1-token warmup leaves only a ~149 MB resident floor. The 128k fixture
(`longctx_128k.json`, ~131,052 tokens; ~130,810 actually filled) fits the ceiling
with room for 256 generated tokens.

**`--max-timeout-secs 1800` is required for 128k.** `rmlx serve` enforces an
independent **server-side per-request wall-clock cap**, default **600 s**, applied
to SSE streams too and *not* overridden by the CBB client's `--request-timeout`.
On the no-NAX brew mlx used for this campaign the cold 128k prefill exceeded 600 s
(`none` r0 = 815 s), so the first 128k attempt was killed at exactly
`e2e_ms=600007` with HTTP 408. Every long-context serve here was relaunched with
`--max-timeout-secs 1800`; all cold prefills landed inside that budget (worst case
`iso4_sym` 903 s @128k). *(Correction 2026-07-18: those seconds are the no-NAX
artifact — re-measured on the pinned NAX-present MLX the 128k `none` prefill is
408.8 s, comfortably under the 600 s cap; the extended timeout was a consequence of
the missing GEMM kernels, not the true prefill cost. See the correction at the top.)*

**KV-MB capture** uses the serve-side per-request events-table high-water-mark
(`op='kv_cache_bytes'`, one row per request through the shared engine loop). The
`rmlx baseline --record` path is **not** usable at 128k — it hardcodes a 65536
prompt-token cap and silently truncates, so the 131k fixture would read as a
65k-length KV. The events path has no such cap; the 128k `none` reading (8865 MB)
is 1.97× the 64k reading (4507 MB), matching the ~2× token ratio — confirming a
genuine full-length prefill.

> ⚠️ **KV-MB is SUSPECT (under-reported) for the `k_iso3` / `k_iso4` /
> `k_rotor3` / `k_rotor4` rows.** These four codecs are the ring-backed K-only
> family: their flash-decode path stands up a GPU ring during decode that the
> byte accounting did not count when this matrix was captured, so the published
> figure is CPU blocks only. The ring is roughly **+34% on top of the blocks**;
> measured on the 8B sibling at 4k the decode-time total for `k_iso3` moved
> 25.3 → 42.9 MB/layer once the ring was counted. The accounting is fixed, but
> **these cells have not been re-measured** — treat them as a lower bound, not a
> reading. Re-capture is tracked separately. Decode-TPS and TTFT in those rows
> are unaffected (the ring was always allocated; only the metric was blind).
> All other codecs' KV-MB is unaffected within this column's reading precision:
> they allocate no ring, and their bytes already came from their real buffers.
> The one exception is immaterial — the TurboQuant V store's optional TCQ
> codebook and calibration indices are now counted too, which is O(100 B)
> against an MB column.

---

## 1. rMLX snapshot benched

| Snapshot (basename) | Weight quant | Arch / size | Role | Disk |
|---|---|---|---|---|
| `prism-ml__Ternary-Bonsai-27B-mlx-2bit` | affine g128 b2 (ternary) | `Qwen3_5ForConditionalGeneration` dense ~27B text tower, GDN hybrid (16 full-attn + 48 GDN of 64), head_dim 256, native 262144 ctx | base | 7.9 GB |

No drafter snapshot exists for Bonsai (MTP head declared, no `mtp.*` weights) →
no speculative / MTP grid (§0).

---

## 2. rMLX full matrix

**Cell = `decodeTPS · r0TTFT(s) · KV-MB`.** decode + cold r0 TTFT from serve +
`run_one` (load-once, chat-templated); `KV-MB` from the serve events-table
`kv_cache_bytes` high-water-mark (§M).

> **⚠️ TTFT column (the middle number) correction.** All r0TTFT values in this
> table were measured on the no-NAX Homebrew MLX bottle and are inflated ~2.0–3.1×
> (see the correction at the top). The **`none` row TTFT has been re-measured on the
> pinned NAX-present MLX and replaced** (4.8 / 10.7 / 24.4 / 53.4 / 136.6 / 408.8 s).
> The other 24 codec rows' TTFT is **still the no-NAX artifact — re-measure pending**;
> prefill is ~codec-independent, so scale them down by the per-column `none` factor
> (~3.1× @4k → ~2.0× @128k). Decode-TPS and KV-MB in every row are NAX-independent
> and unchanged.

Markers: `†` = 128k value is the **cold r0**
number (warm-cache decode stalls — Tier-2 `*_sym`, see below / §5). `‡` = decode
**crashes at the next power-of-two KV boundary** — the cell is a *truncated
crashing run* (242–250 of 256 tokens; the per-token rate is genuine, but the run
does not complete — see the §0 correction / §5 / §4.3). `—·—·—` = not captured.
K-only rows (`k_iso* / k_rotor*`) are **capped, CPU-bound** — their decode is a
reduced-token probe (`max_tokens 8–64`, n=1), not a steady-state 256-token rate.

| KV | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| none | 50.9·4.8s·419 | 47.7·10.7s·692 | 41.7·24.4s·1237 | 37.9·53.4s·2327 | 31.3·136.6s·4507 | 23.7·408.8s·8865 |
| k8v4‡ | 51.1·14.9s·519 | 29.7·32.2s·1063 | 22.4·70.2s·1979 | 15.7·144.9s·3811 | 10.0·315.4s·7477 | 5.6·773.6s·14795 |
| k8v8 | 51.0·14.9s·535 | 47.6·32.1s·923 | 40.8·69.4s·1699 | 37.0·148.5s·3251 | 31.4·331.0s·6356 | 23.7·813.3s·12555 |
| planar | 51.6·14.9s·631 | 48.4·32.1s·1115 | 44.9·64.8s·2084 | 40.2·139.9s·4021 | 33.6·314.4s·7895 | 25.8·768.0s·15630 |
| planar3 | 50.3·14.9s·662 | 47.6·32.3s·1170 | 44.2·65.1s·2185 | 39.6·142.0s·4216 | 33.0·319.5s·8278 | 25.0·785.6s·16387 |
| planar_k | 52.1·14.8s·601 | 48.1·31.7s·1048 | 45.7·64.4s·1943 | 40.2·138.4s·3732 | 33.8·310.2s·7309 | 25.2·781.7s·14453 |
| k8vturbo2 | 51.0·15.1s·497 | 48.2·32.0s·848 | 44.1·66.1s·1551 | 39.6·144.3s·2956 | 33.3·325.3s·5766 | 25.1·798.4s·11380 |
| k8vturbo3 | 51.7·15.1s·503 | 47.9·33.0s·862 | 42.1·69.4s·1578 | 38.8·147.9s·3011 | 33.6·327.5s·5877 | 25.6·789.7s·11604 |
| k8vturbo2tcq | 52.5·15.9s·497 | 49.9·33.8s·848 | 46.6·69.0s·1551 | 40.1·148.6s·2956 | 33.8·330.4s·5766 | **27.3**·815.7s·11380 |
| k8vturbo3tcq | **52.6**·16.3s·503 | 49.3·34.8s·862 | 46.5·71.0s·1578 | 40.6·150.9s·3011 | 34.6·340.1s·5877 | 27.2·827.7s·11604 |
| tsym3 | 51.7·15.1s·474 | 49.1·32.0s·802 | 44.6·66.0s·1459 | 40.2·144.1s·2773 | 33.6·321.3s·5401 | 25.7·786.6s·10654 |
| tsym4 | 51.6·14.8s·489 | 47.9·32.1s·832 | 43.5·66.1s·1517 | 39.7·142.4s·2887 | 33.5·318.6s·5627 | 25.5·778.4s·11101 |
| iso3 | 52.5·15.5s·794 | 49.6·32.5s·1461 | **47.2**·67.4s·2795 | 40.6·144.5s·5464 | 34.1·325.2s·10801 | 25.4·797.1s·21467 |
| iso4 | 52.5·16.1s·794 | 49.4·34.6s·1461 | 44.6·69.8s·2795 | 40.7·150.1s·5464 | 35.0·335.4s·10801 | 25.4·823.5s·21467 |
| iso3_sym | 52.6·16.0s·1053 | 49.1·34.9s·2000 | 44.1·71.4s·3892 | 40.3·151.9s·7676 | 31.6·354.3s·15246 | 24.1†·857.1s·30382 |
| iso4_sym | 52.5·17.3s·1053 | 50.6·36.3s·2000 | 47.8·74.7s·3891 | 41.2·159.5s·7676 | 30.7·382.9s·15246 | 23.8†·903.5s·30382 |
| rotor3 | 51.3·15.5s·619 | 49.1·33.5s·1101 | 44.6·67.0s·2064 | 40.4·144.8s·3990 | 33.5·325.7s·7844 | 26.4·797.7s·15544 |
| rotor4 | 51.6·15.6s·619 | 48.5·33.8s·1101 | 45.0·67.8s·2064 | 40.7·145.9s·3990 | 33.9·326.2s·7844 | 26.8·804.4s·15544 |
| rotor3_sym | 51.9·22.9s·750 | **50.8**·48.3s·1361 | 45.7·102.4s·2584 | **43.2**·211.4s·5030 | **36.2**·462.4s·9921 | 25.6†·1075.2s·19702 |
| rotor4_sym | 52.2·23.3s·750 | 50.7·49.6s·1361 | **47.9**·102.4s·2584 | 42.5·213.1s·5030 | 35.9·467.2s·9921 | 24.7†·1061.3s·19702 |
| k_iso3 *(capped)* | 8.8·15.4s·758 | 4.7·32.3s·1388 | 3.0·68.9s·2649 | 1.5·144.6s·5171 | 0.7·323.6s·10207 | 0.2·799.5s·20293 |
| k_iso4 *(capped)* | 3.6·16.1s·758 | 1.9·33.7s·1388 | 1.0·71.5s·2649 | 0.5·151.2s·5163 | 0.2·344.5s·10207 | 0.1·886.4s·20293 |
| k_rotor3 *(capped)* | 0.8·22.4s·577 | 0.4·46.7s·1022 | 0.2·100.3s·1910 | 0.1·209.1s·3687 | 0.05·456.4s·7241 | —·—·— |
| k_rotor4 *(capped)* | 0.8·22.6s·577 | 0.4·47.5s·1022 | —·101.4s·1480 | —·209.8s·2821 | —·462.3s·5503 | —·—·— |
| rot_k_tq4v *(retired, see `docs/KV_QUANT.md`)* | 48.0·14.9s·531 | 42.3·32.5s·916 | 33.8·70.5s·1685 | 26.8·144.9s·3225 | 19.0·322.6s·6303 | 11.7·788.1s·12456 |

**Best decode per size** (bold above): 4k `k8vturbo3tcq` (52.57 — a near-tie with
`iso3_sym` 52.55 / `iso4` 52.49; the whole 4k column is 52.5–52.6, i.e. noise);
8k `rotor3_sym` (50.75); 16k `rotor4_sym` (47.86); 32k `rotor3_sym` (43.23); 64k
`rotor3_sym` (36.20); 128k `k8vturbo2tcq` (27.28).

> **Read the bolds with the tiers.** The 8k/16k/32k/64k winners are the `*_sym`
> **bf16-mirror family (Tier 2)** — genuinely the fastest *raw* decode at those
> sizes, but at **2.0–2.2× the `none` KV**, **~1.4–2.0× heavier cold prefill**
> (`rotor3_sym` 64k TTFT 462 s vs `none` 335 s — both no-NAX-artifact seconds; the
> ratio is codec-relative and preserved even though the absolute seconds inflate),
> and a **128k warm-cache stall**
> (§5) — so **not** the recommended pick. Among the memory- and prefill-sane
> **Tier-1** codecs, the **tcq pair** and **rotor3/rotor4** lead at long context
> (`k8vturbo2tcq` +15.2 %, `k8vturbo3tcq` +14.8 %, `rotor4` +13.1 %, `rotor3`
> +11.3 % vs `none` at 128k) — the champion-beating cells that cost the least
> memory and prefill.

### 2.1 KV-cache size (MB) and ratio vs `none`

`none` KV is **bf16** (≈2 bytes/element): 419 / 692 / 1237 / 2327 / 4507 / 8865 MB
at 4k…128k. Every quantized codec keeps a bf16/packed seed *alongside* its blocks,
so all are **larger** than `none` — `none` is the memory winner at every context.
Ratios below are at **128k**.

| KV | MB @128k | ratio vs none |
|---|---|---|
| **none** | **8865** | **1.00×** |
| tsym3 | 10654 | 1.20× |
| tsym4 | 11101 | 1.25× |
| k8vturbo2 / k8vturbo2tcq | 11380 | 1.28× |
| k8vturbo3 / k8vturbo3tcq | 11604 | 1.31× |
| rot_k_tq4v *(retired, see `docs/KV_QUANT.md`)* | 12456 | 1.41× |
| k8v8 | 12555 | 1.42× |
| planar_k | 14453 | 1.63× |
| k8v4 | 14795 | 1.67× |
| rotor3 / rotor4 | 15544 | 1.75× |
| planar | 15630 | 1.76× |
| planar3 | 16387 | 1.85× |
| rotor3_sym / rotor4_sym | 19702 | 2.22× |
| k_iso3 / k_iso4 *(capped)* | 20293 | 2.29× |
| iso3 / iso4 | 21467 | 2.42× |
| iso3_sym / iso4_sym | 30382 | 3.43× |
| k_rotor3 *(capped)* | — | — (1.61× @64k; 128k not captured) |
| k_rotor4 *(capped)* | — | — (1.22× @64k; 128k skipped) |

The lightest quantized tier is `tsym3/4` + the `k8vturbo`/`tcq` family (1.20–1.31×);
the heaviest is `iso*_sym` (3.43×). Note the byte-for-byte identical KV between
3-/4-bit variants of the same family (`iso3`=`iso4`, `rotor3`=`rotor4`,
`rotor3_sym`=`rotor4_sym`, `k_iso3`=`k_iso4`): the variant selects a dequant path,
not a smaller byte layout. (`k_rotor3` vs `k_rotor4` is the one exception —
`k_rotor4` runs ~20–24 % lighter from 16k up.)

### 2c. SSD KV tier

**Not benched.** As on Gemma4 and the 8B (`SIBLINGS`/`rMLX` §2c), a 256-token
single-stream decode never overflows the RAM prompt-cache, so the SSD tier does
not spill and is decode-neutral / untriggered at these sizes. Even the heaviest
cell here (`iso*_sym` ~30 GB KV at 128k) fits the 128 GB unified memory with no
paging. SSD is a capacity feature; exercising it needs a multi-turn / >RAM-KV
scenario. Left out rather than reported as a no-op cell. (Note: the `*_sym` 128k
warm-cache stall in §5 is a *RAM* prompt-cache eviction artifact — the 1 GiB RAM
prompt-cache cap vs ~20–30 GB raw KV — not an SSD-tier event.)

---

## 3. Standing vs champion (decode)

rMLX `none` decode vs the SIBLINGS mlx-lm champion (no-KV, same-method reference),
**same serve + `run_one` harness** on both sides (directly comparable). `none` is
the honest number — the per-codec spread (§2) is small and memory-costly, so no
cherry-picked "best codec" is used.

| Prompt | rMLX `none` | champion (mlx-lm) | Δ | standing |
|---|---|---|---|---|
| 4k | **50.9** | 45.1 | **+12.8 %** | 🟢 WIN |
| 8k | **47.7** | 41.7 | **+14.4 %** | 🟢 WIN |
| 16k | **41.7** | 40.6 | **+2.7 %** | 🟢 WIN (narrow) |
| 32k | **37.9** | 36.8 | **+2.9 %** | 🟢 WIN (narrow) |
| 64k | **31.3** | 30.2 | **+3.6 %** | 🟢 WIN (narrow) |
| 128k | **23.7** | 23.0 | **+3.0 %** | 🟢 WIN (narrow) |

> **rMLX decode leads mlx-lm on Bonsai-27B at every context (+2.7…+14.4 %)**, but
> the lead is **decisive only at short context** and shrinks to ~+3 % (borderline
> run-to-run noise given n=1 at 64k/128k) from 16k on — a much smaller margin than
> the 8B's flat +21…+27 %. The GDN hybrid's flat decode-vs-context curve
> compresses everyone toward a near-tie at long context (same reason mlx-lm-tq is
> at parity, not a loss, in `SIBLINGS`). `none` is the champion-beating cell;
> KV quant adds a little decode at long ctx (§2) but costs memory.

**Prefill — near parity (corrected 2026-07-18).** rMLX `none` cold TTFT re-measured
on the pinned NAX-present MLX (the published column was a no-NAX brew-mlx artifact —
see the correction at the top) vs the champion's prefill (SIBLINGS §2b, measured on
the NAX-correct PyPI wheel, so unaffected: 4.4 / 10.7 / 20.6 / 43.7 / 109.9 /
317.8 s):

| Prompt | rMLX `none` TTFT (pinned) | was (no-NAX brew) | champion TTFT | ratio | standing |
|---|---|---|---|---|---|
| 4k | **4.8 s** | 14.8 s | 4.4 s | 1.09× | 🟢 ~TIE |
| 8k | **10.7 s** | 32.0 s | 10.7 s | 1.00× | 🟢 TIE |
| 16k | **24.4 s** | 67.9 s | 20.6 s | 1.18× | 🟡 near |
| 32k | **53.4 s** | 147.5 s | 43.7 s | 1.22× | 🟡 near |
| 64k | **136.6 s** | 335.0 s | 109.9 s | 1.24× | 🟡 near |
| 128k | **408.8 s** | 815.4 s | 317.8 s | 1.29× | 🟡 near |

rMLX prefill is **at parity to ~1.3× the champion** — tied at 4k–8k, a small
residual (1.2–1.3×) that grows modestly with context. The earlier "2.6–3.4× loss"
was entirely the missing-NAX GEMM artifact, not the backend. Decode wins, prefill is
competitive — the standing verdict is **decode-favourable, prefill-competitive**.
The small long-context residual is a minor follow-up (§4), no longer the headline gap.

---

## 4. Gaps & hypotheses (improvement plan)

Ranked by impact:

1. **Fused MSL flash-decode-over-quant kernels for the Tier-2 (`*_sym`) and
   Tier-3 (`k_*`) families — built, and the ROI argument did not survive it.**
   The original text here read: *the iso / rotor codecs are the fastest in the
   whole sweep when they have a GPU decode kernel (`rotor3/4` +7…+13 % vs `none`;
   `iso3` +13 % at 16k), their sub-8-bit and symmetric variants have no Metal
   kernel, so a real fused flash-decode-over-quant kernel would convert 8
   currently useless / mirror cells (4 `*_sym` + 4 `k_*`) into viable ones.*

   Those kernels were then written — `iso_flash_decode`, `rotor_flash_decode` and
   the `_symv` pair — and all 8 cells now decode straight off the packed store
   with no mirror. **They did not become viable.** Two independent reasons, both
   measured after the fact:

   * The stores were not smaller than bf16 at the time of this measurement
     (16.25 bits/value for iso, 21.75 for rotor, against bf16's 16.0), so there
     was never a bandwidth prize to collect. **Both figures are superseded:**
     narrowing the ring's scale and norm planes to `KV_SIDEBAND_DTYPE` took iso
     to **12.125** (a genuine memory win) and rotor to **16.25** (still above
     the floor). The kernel-shell reason below is unaffected and remains the
     binding one — see "Memory truth" in `docs/KV_QUANT.md`.
   * The kernel shell itself achieves only 4–14 % of MLX `sdpa_vector`'s
     per-byte throughput. Its P1 grid is indexed by *query* head and so re-reads
     the whole KV stream `heads_per_kv` times, which caps the shell at
     `1/heads_per_kv` — but that cap is **not** where the loss is: the fused
     kernel is issue-bound, not memory-bound, and removing the entire
     query-head class leaves it 3.2× slower than the generic path. Even the
     densest store in the tree is too fat to clear the bar either way. The
     `ρ < ε` arithmetic, the two-architecture measurement, the corollary for
     `kv_h == 1` and the negative result on lifting ε are in
     `docs/KV_QUANT.md` § "Fused flash-decode over a quant store — the break-even
     condition".

   The premise that misled this item was *"the GPU-kerneled members of the same
   families already win, so a kernel is the missing piece"*. `rotor3/4` and
   `iso3` win because they decode through the **bf16 mirror**, i.e. through MLX's
   own SDPA — not because their codec math is fast at decode. They were never
   evidence about a hand-written kernel.
2. **rMLX prefill — corrected to near-parity (was misdiagnosed as a 2.6–3.4× GDN
   loss).** The original text here concluded prefill was 2.6–3.4× slower than the
   siblings and pinned the blame on the GDN recurrence kernel
   (`gated_delta_msl.rs`, a sequential per-timestep scan), arguing the 27B's
   `48×256` serial cost was `3.2×` the Qwen3.6-35B's `30×128`. **That was numerology
   on artifact data.** The campaign ran on a Homebrew MLX bottle missing every NAX
   GEMM kernel (~3.8× matmul loss); re-measured on the pinned NAX-present MLX,
   prefill is **at parity to ~1.3× the mlx-lm champion** (§3) — a measured
   refutation of the 3× claim — and the NAX-regression root-cause profile puts the
   GDN recurrence at **~2 % of prefill time** (MLP dominates), so GDN was never the
   3× lever. The merged GDN-kernel-always + prefill-chunk
   64→2048 fix (active in this binary) already brought this arch close. A small
   residual long-context gap (1.2–1.3× at 32k–128k) remains and could still benefit
   from a chunkwise-parallel delta-rule prefill kernel, but it is a **minor**
   follow-up — not the headline weakness this section originally claimed.
3. **4-bit-V is broken *and* slow — the boundary crash is the P0, the dequant
   cost the P1.** *(a)* `k8v4` decode **crashes at the next power-of-two KV
   boundary**: generation dies exactly when `prompt_len + generated` reaches
   `2^k` (`reshape: array of size 0 into (1,4,1,64)`, the #233 class, still live
   on `626381e` — #238 did not cover the paged 4-bit-V path). The retry envelope
   then masks it as `finish_reason=null` to the streaming client (a #235-class
   gap). This is a shippable-blocker bug, not a perf item — it needs its own
   ticket. *(b)* Independently, the tq4-V **dequant cost is real** (clean,
   boundary-safe re-measurement 2026-07-17: `k8v4` 50.4/15.5/5.0 vs a `k8v8`
   control 45.1/37.3/21.4 at 4k/32k/128k — a tie at 4k growing to 4.3× slower at
   128k); `rot_k_tq4v` shares it, `k8v8` (8-bit V) tracks `none`. Either a faster
   V-4bit decode kernel, or steer `auto` away from 4-bit V on this arch.
4. **`*_sym` 128k warm-cache decode stall — investigate the prompt-cache rebuild.**
   All four `*_sym` codecs show a reproducible single-large-stall on **warm-cache**
   128k requests: `itl_p50` stays healthy (~36 ms/tok ⇒ ~28 TPS) but the aggregate
   craters to ~1 TPS (`iso*_sym`) / ~6–10 TPS (`rotor*_sym`). Cold r0 is clean
   (23.8–25.6 TPS, in line with `none`). Likely a RAM prompt-cache eviction/rebuild
   at ~20–30 GB raw KV vs the 1 GiB `resolved_ram_prompt_cache_gb` cap — one very
   slow reconstructed token dominates e2e time. Worth a follow-up ticket; the
   matrix reports cold r0 for these cells.
5. **K-only codecs are unusably slow — add an `auto` skip / loud resolve warning.**
   `k_iso* / k_rotor*` decode at 0.05–8.8 TPS (CPU-bound, no Metal kernel), the
   same class the Qwen-MoE arch-guard already rejects. Recommend a resolve-time
   warning (or `auto` skip) for sub-8-bit-K on dense 2-bit Qwen3_5, so nobody
   selects one expecting a usable rate.

---

## 5. Caveats

- **Prefill/TTFT numbers were a no-NAX Homebrew-MLX artifact (corrected
  2026-07-18).** The whole matrix was benched on an MLX bottle missing the NAX GEMM
  kernels (~3.8× matmul loss, prefill-only). The `none`-row TTFT has been re-measured
  on the pinned NAX-present MLX and replaced (§2/§3); the other codec rows' TTFT is
  still the inflated artifact (scale by the per-column `none` factor, ~3.1× @4k →
  ~2.0× @128k). Decode-TPS and KV-MB are NAX-independent and stand. The "prefill
  loses 2.6–3.4×" verdict and the "GDN sequential-in-T ⇒ 3.2×" root-cause were both
  false; corrected prefill is at parity to ~1.3× the champion.
- **`none` is the headline number** and the smallest KV. The long-ctx codec win
  (tcq pair, `rotor3/4`, `planar*`: +5…+15 % at 128k) is real but memory-costly
  (1.28×–1.85× the `none` KV) and prefill-costly — not a free win.
- **64k and 128k are n=1 measured** (single run after the discarded warmup) —
  point estimates. The ~+3 % `none`-vs-champion margins at 16k–128k in particular
  are inside plausible run-to-run noise; read them as "at least a tie, small win,"
  not a CI-bounded lead.
- **`*_sym` 128k cells use the cold r0 decode** (`iso3_sym` 24.1, `iso4_sym` 23.8,
  `rotor3_sym` 25.6, `rotor4_sym` 24.7 — marked `†`). The warm-cache measured run
  stalls to ~1–10 TPS on every attempt (reproduced 2× each), an artifact of a
  single mid-decode stall (bf16-mirror prompt-cache rebuild at ~20–30 GB KV), not
  a steady-state rate — `itl_p50` implies ~28 TPS throughout. See §4 #4.
- **No long-ctx `iso*`/`*_sym` collapse** (unlike the 8B, where they cratered to
  ~6–13 TPS at 64k). On the 27B they hold 30–36 TPS at 64k — GDN's shallow KV
  growth (only 16/64 layers) avoids the CPU-dequant collapse. Major divergence
  from the 8B doc.
- **`planar` direction-flip** — `planar` beats `none` on the 27B (+1…+9 %) but
  lost to `none` on the 8B, same build. Architecture-dependent (GDN vs
  full-attention) or an intervening kernel change; flagged, not diagnosed.
- **K-only codecs (`k_iso* / k_rotor*`) are capped and unusable** — decode is a
  reduced-token probe (`max_tokens 8–64`, n=1); the 27B is milder than the 8B
  (measurable through 16k–32k vs the 8B's ≤16k cap) thanks to GDN, but still
  CPU-bound. **Data gaps:** `k_rotor3` 128k **not captured** (client-timeout margin
  miss — the request *did* complete server-side in ~1415 s, but the harness
  `timeout 1400` killed the client ~15 s early, so no KV-MB/decode row exists);
  `k_rotor4` 128k **skipped** (time budget) and `k_rotor4` 16k/32k/64k are
  **KV-MB-only** (`max_tokens=1` fills, decode not measured).
- **`k8v4` is a real tq4-V cost *on top of* a boundary crash** (2026-07-17
  re-measurement, #241). The rate curve is genuine — the 4k→8k cliff (51→30) then
  smooth crater is a real tq4-V dequant cost (confirmed clean vs a same-machine
  `k8v8` control), reproduced by `rot_k_tq4v`. But the recorded cells are also
  **truncated crashing runs**: `k8v4` decode dies when `prompt_len + generated`
  hits the next power of two (242–250 of 256 tokens delivered), so `k8v4` cannot
  generate across a `2^k` boundary. The rate survived because the death is at the
  tail; the codec is nonetheless broken for real use — see the §0 correction.
- **No MTP / speculative** — Bonsai declares an MTP head but ships no `mtp.*`
  weights (inert).
- **No §6 weight-quant sweep** — one on-disk 2-bit snapshot; no QAT siblings.
- **SSD tier not benched** — not triggered at 256-token single-stream (§2c).
- **Metrics landmines (benign):** the CBB `recorder rejected record` warning fires
  on every `run_one` (known §8.5 shape mismatch); the iso/rotor/tsym codec names
  are absent from the server metrics-drainer identity allow-list (`metrics_drainer:
  … is not a valid kv_quant`) — neither affects decode/TTFT/KV-MB capture, all
  sourced from `run_one` stdout + the events-table `kv_cache_bytes` query. Two
  K-only cells carried a `db24cf4` `run_one` `backend_version` tag (a harness-label
  artifact — the binary was the same `release-perf` `3d83e6f` build throughout).
