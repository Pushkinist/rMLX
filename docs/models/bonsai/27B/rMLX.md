# Bonsai-27B (2-bit) — rMLX Full Matrix (Stage 2)

> Companion: [`SIBLINGS.md`](SIBLINGS.md) — sibling-backend champions.
> Champion (weight-comparable 2-bit tier) = **mlx-lm (no KV quant)**, decode TPS:
> **45.1 / 41.7 / 40.6 / 36.8 / 30.2 / 23.0** (4k/8k/16k/32k/64k/128k).

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
- **rMLX prefill is the weak spot.** Cold TTFT is **~2.6–3.4× slower** than the
  mlx-lm/tq siblings at every size (815 s vs 318 s at 128k; 14.8 s vs 4.4 s at 4k).
  Decode wins, prefill loses — the top perf follow-up (§4, **#216**): the GDN
  recurrence kernel is sequential-in-T; the merged #155 chunk fix (64→2048) is
  already spent, so this needs a chunkwise-parallel delta-rule prefill kernel.
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
    > **`rotor3_sym` / `rotor4_sym` no longer belong in this tier.** Their decode
    > is now a fused MSL flash over **both** packed rotor rings
    > (`rotor_flash_decode_symv`, see `docs/KV_QUANT.md`), and `exit_prefill`
    > seeds no bf16 K or V mirror for them — so the "no memory win" claim is
    > retired: measured KV is **−34%** (Bonsai-8B 590 → 391 MB; gemma-4-e2b
    > 36.1 → 23.5 MB). The mirror that fed the 128k warm-cache stall is gone.
    > **The trade is not free:** the bf16 mirror was MLX's native bf16 flash and
    > was *fast*; the fused kernel costs **−50…−90% decode** on the same shapes
    > (Bonsai-8B ~145 → ~15 TPS; gemma-4-e2b ~135 → ~56). The bottleneck is the
    > shared kernel shell — the bf16-V `k_rotor3` sibling is already ~23 TPS on
    > that shape — not the V unpack. `iso3_sym` / `iso4_sym` are untouched and
    > remain genuine Tier 2. Numbers here are a **pre-kernel snapshot** and were
    > not re-run on the 27B.
  - **Tier 3 — CPU-bound, unusable** (the K-only family: `k_iso3/4`,
    `k_rotor3/4`): sub-8-bit rotation/iso K with **no Metal kernel** → CPU dequant
    fallback → **0.05–8.8 TPS**, GPU idle. Capped.
    > **Superseded for `k_rotor3/4` (run with `--rotor-qjl off`).** The rotor
    > K-only decode is now a fused MSL flash-decode over the packed rotor store
    > (`rotor_flash_decode`, see `docs/KV_QUANT.md`), so the per-step full-prefix
    > CPU dequant that produced these numbers is gone. Re-measured at 4k: Bonsai-8B
    > 1.34 → 17.0 TPS, medgemma-4B 7.37 → 51.8 TPS. The numbers in this table are a
    > pre-kernel snapshot and were **not** re-run on the 27B. Two caveats stand:
    > the default `--rotor-qjl on` still takes the CPU path (the kernel cannot
    > reproduce the QJL residual), and `k_iso3/4` is untouched — it keeps its own
    > per-step host restaging.
- **No long-ctx collapse (unlike the 8B).** On the 8B, `iso*`/`*_sym` cratered to
  ~6–13 TPS at 64k; on the 27B they hold **30–36 TPS at 64k**. GDN's shallow KV
  growth avoids the CPU-dequant collapse entirely — a big divergence from the 8B doc.
- **`planar` direction-flip.** `planar` **beats** `none` on the 27B (+1…+9 %,
  growing with ctx) but **lost** to `none` on the 8B. Same build, opposite sign —
  flagged, not diagnosed (§5).
- **4-bit-V still craters.** `k8v4` (51→30→22→16→10→**5.6**) and `rot_k_tq4v`
  (48→42→34→27→19→**11.7**) — the tq4-V dequant cost, same class as the 8B. `k8v4`
  is the worse of the two (4-bit-V is expensive on this arch). `k8v8` (8-bit V)
  tracks `none`. **Avoid 4-bit V here.**
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
The genuine cold 128k prefill exceeds 600 s (`none` r0 = 815 s), so the first 128k
attempt was killed at exactly `e2e_ms=600007` with HTTP 408. Every long-context
serve here was relaunched with `--max-timeout-secs 1800`; all cold prefills landed
inside that budget (worst case `iso4_sym` 903 s @128k).

**KV-MB capture** uses the serve-side per-request events-table high-water-mark
(`op='kv_cache_bytes'`, one row per request through the shared engine loop). The
`rmlx baseline --record` path is **not** usable at 128k — it hardcodes a 65536
prompt-token cap and silently truncates, so the 131k fixture would read as a
65k-length KV. The events path has no such cap; the 128k `none` reading (8865 MB)
is 1.97× the 64k reading (4507 MB), matching the ~2× token ratio — confirming a
genuine full-length prefill.

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
`kv_cache_bytes` high-water-mark (§M). Markers: `†` = 128k value is the **cold r0**
number (warm-cache decode stalls — Tier-2 `*_sym`, see below / §5). `—·—·—` =
not captured. K-only rows (`k_iso* / k_rotor*`) are **capped, CPU-bound** — their
decode is a reduced-token probe (`max_tokens 8–64`, n=1), not a steady-state
256-token rate.

| KV | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| none | 50.9·14.8s·419 | 47.7·32.0s·692 | 41.7·67.9s·1237 | 37.9·147.5s·2327 | 31.3·335.0s·4507 | 23.7·815.4s·8865 |
| k8v4 | 51.1·14.9s·519 | 29.7·32.2s·1063 | 22.4·70.2s·1979 | 15.7·144.9s·3811 | 10.0·315.4s·7477 | 5.6·773.6s·14795 |
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
| rot_k_tq4v | 48.0·14.9s·531 | 42.3·32.5s·916 | 33.8·70.5s·1685 | 26.8·144.9s·3225 | 19.0·322.6s·6303 | 11.7·788.1s·12456 |

**Best decode per size** (bold above): 4k `k8vturbo3tcq` (52.57 — a near-tie with
`iso3_sym` 52.55 / `iso4` 52.49; the whole 4k column is 52.5–52.6, i.e. noise);
8k `rotor3_sym` (50.75); 16k `rotor4_sym` (47.86); 32k `rotor3_sym` (43.23); 64k
`rotor3_sym` (36.20); 128k `k8vturbo2tcq` (27.28).

> **Read the bolds with the tiers.** The 8k/16k/32k/64k winners are the `*_sym`
> **bf16-mirror family (Tier 2)** — genuinely the fastest *raw* decode at those
> sizes, but at **2.0–2.2× the `none` KV**, **~1.4–2.0× heavier cold prefill**
> (`rotor3_sym` 64k TTFT 462 s vs `none` 335 s), and a **128k warm-cache stall**
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
| rot_k_tq4v | 12456 | 1.41× |
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

**Prefill loses.** rMLX cold TTFT vs the champion's prefill (SIBLINGS §2b:
4.4 / 10.7 / 20.6 / 43.7 / 109.9 / 317.8 s):

| Prompt | rMLX `none` TTFT | champion TTFT | ratio | standing |
|---|---|---|---|---|
| 4k | 14.8 s | 4.4 s | 3.37× | 🔴 LOSS |
| 8k | 32.0 s | 10.7 s | 2.99× | 🔴 LOSS |
| 16k | 67.9 s | 20.6 s | 3.30× | 🔴 LOSS |
| 32k | 147.5 s | 43.7 s | 3.38× | 🔴 LOSS |
| 64k | 335.0 s | 109.9 s | 3.05× | 🔴 LOSS |
| 128k | 815.4 s | 317.8 s | 2.57× | 🔴 LOSS |

Decode wins, prefill loses (~2.6–3.4× slower) at every size — the standing verdict
is **decode-favourable, prefill-adverse**. Prefill is the single biggest rMLX gap
on this arch (§4).

---

## 4. Gaps & hypotheses (improvement plan)

Ranked by impact:

1. **Fused MSL flash-decode-over-quant kernels for the Tier-2 (`*_sym`) and
   Tier-3 (`k_*`) families — the headline.** The matrix proves the ROI: the iso /
   rotor codecs are the **fastest in the whole sweep when they have a GPU decode
   kernel** (`rotor3/4` +7…+13 % vs `none`; `iso3` +13 % at 16k). Their sub-8-bit
   and symmetric variants have **no Metal kernel**, so they fall back to a CPU
   dequant seed (Tier 3) or a dormant bf16-mirror (Tier 2). A real fused
   flash-decode-over-quant kernel for these would convert **8 currently
   useless / mirror cells** (4 `*_sym` + 4 `k_*`) into viable ones. This is the
   known blocker: *per-step `quantized_matmul` can't beat MLX bf16 flash at long
   ctx — it needs an MSL flash-decode-over-quant kernel*. The 27B is the best
   argument yet for building it, because the GPU-kerneled members of the same
   families already win.
2. **rMLX prefill is the weak spot — cold TTFT 2.6–3.4× slower than the siblings**
   (**#216**). `none` 815 s vs mlx-lm 318 s at 128k, same ratio at every size (§3).
   Root cause is **not** the chunk size: #155 (GDN kernel-always + prefill chunk
   64→2048) is already merged and was active in this benched binary. The residual
   cost is the **GDN recurrence kernel itself** (`gated_delta_msl.rs:110`) — a
   *sequential* per-timestep scan (`for t in 0..T`, loop-carried state), so prefill
   runs T=2048 serial steps/chunk over 48 GDN layers. Serial cost scales as
   `num_gdn_layers × head_dim`: the 27B's `48×256` is **3.2×** the Qwen3.6-35B's
   `30×128` (where #155 reached mlx-lm parity) — matching the observed gap. Fix =
   a chunkwise-parallel delta-rule prefill kernel (Bonsai-8B, dense 2-bit, *no GDN*,
   prefills fast on the same quant-GEMM path — confirming GDN is the delta). Top perf
   follow-up.
3. **4-bit-V dequant is expensive on this arch — avoid tq4-V.** `k8v4`
   (−38…−76 % vs `none` from 8k up) and `rot_k_tq4v` (−11…−50 %) both crater
   progressively; `k8v8` (8-bit V) tracks `none`. `k8v4` is the worse of the two
   (5.6 TPS @128k vs `rot_k_tq4v` 11.7 — the rotation K-side partially offsets the
   shared tq4-V cost). Either a faster V-4bit decode kernel, or steer `auto` away
   from 4-bit V on this arch.
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
- **`k8v4` 4k→8k cliff** (51→30) then a smooth crater — the tq4-V cost appears from
  8k up; reproduced by `rot_k_tq4v` (same tq4-V), so it is a real codec cost.
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
