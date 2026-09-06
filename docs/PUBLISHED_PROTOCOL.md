<!-- GENERATED FILE — do not edit by hand. -->
<!-- Regenerate: make published-table -->

# Published-protocol speculative-decoding results

> [!WARNING]
> **THIS TABLE HOLDS NO PUBLISHABLE MEASUREMENT.**
>
> - `plain`, `mtp_sidecar/block=7`, `dflash/block=16,dflash/depth=accept_rate`: the arms were **synthetic**. A stub answered every request and no model was ever served, so every rate under those modes is a fixture value and bounds nothing.

Measured by `scripts/spec_bench_published.sh` under the protocol
third-party on-device speculative-decoding posts report, so these numbers
can sit beside theirs. Bounds by `scripts/perf_ceiling.py`.

## The protocol, and the parts of it we chose

The published on-device protocol leaves several things unstated. They are
pinned here and printed with the numbers rather than left to a reader to
assume.

| choice | value |
|---|---|
| max output tokens | 1024 for every dataset; MATH-500 also at 4096, as a column beside the headline |
| thinking tokens | on, counted as output |
| warmup | 1 untimed request per pass, on a prompt in no sample set |
| passes | 3 consecutive, each score their mean |
| resident memory | peak `phys_footprint` (`docs/PROFILING.md` §9), sampled every 250 ms, so it is a lower bound on the true peak |
| sampling | the checkpoint's own — the request carries no sampling field. Read back from the engine: `min_p`=0.0, `seed`=42919, `temperature`=0.7, `top_k`=20, `top_p`=0.95 |
| seed | engine default, identical in all three passes |
| run-to-run range refusal | a mean whose three passes span more than 5% of the mean is withheld, not averaged |

### Three things a reader comparing this to a published figure must know

1. **MT-Bench questions are two-turn; only the first turn is measured.**
   The second turn is preserved verbatim in
   `prompts/published/mt_bench.json` and is simply not sent, so nothing is
   lost — but a published MT-Bench figure that measures both turns is
   measuring a longer context than this one, and the two are not
   interchangeable.
2. **The macro average is one cell per dataset**, at the 1024-token budget. MATH-500's 4096-token cell is a
   column beside the headline, not a fourth dataset: folding it in would
   give MATH-500 twice the weight of the other two.
3. **The seed is held fixed across all three passes**, so the run-to-run
   range is a reading of machine stability and not of sampling variance.
   That is checked rather than asserted — `diverged` counts the samples
   that did not generate the same length in all three passes, and for
   those the range carries sampling variance too.

## Where the bounds come from

Every measured figure is printed beside the figure it cannot exceed. The
bounds are `scripts/perf_ceiling.py` — a static census over the snapshot's
`config.json` and safetensors headers, no GPU and no model — at
614 GB/s on `m5_max_128gb`, with the KV
priced at `none` and a ring preallocated to `--max-ctx 8192`. Its KV byte model is held to the engine's own by
`make check-kv-byte-model-parity`, so the KV term is not a second opinion.

- **decode ceiling** — weights streamed per step plus the KV bytes a step
  reads, over the host's memory bandwidth. Evaluated at the middle of each
  cell's decode window; the `ctx` column names it.
- **resident floor** — the weights text decode must hold plus the KV the
  cache holds at that context.
- **there is no speculative ceiling here.** `perf_ceiling.py` models one
  autoregressive forward per token and has no drafter, no block and no
  accept rate. A speculative arm is printed against the *same*
  autoregressive ceiling, and a value above 100% is the point of
  speculative decoding rather than a defect. None is invented.
- **percent-of-ceiling is not scale-free.** It is
  `1 / (1 + overhead / ideal)`, so the same fixed per-step cost reads worse
  on a small model than on a large one. Compare it down a column, within
  one model — not across models.

## Output speed — `fixture/published-table-16L` (bf16, KV `none`)

| cell | mode | samples | max out | t/s (mean of 3) | range % | AR ceiling t/s | % of AR ceiling | × plain | ctx | worst sample % | diverged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mt_bench:1024` | `plain` | 80 | 1024 | 62.42 | 0.70 | 82.55 | 75.6% | 1.000 | 276 | 3.4 | 0 of 80 |
| `mt_bench:1024` | `mtp_sidecar/block=7` | 80 | 1024 | 104.83 | 0.70 | 82.55 | 127.0% | 1.679 | 276 | 3.4 | 1 of 80 |
| `mt_bench:1024` | `dflash/block=16,dflash/depth=accept_rate` | 80 | 1024 | 88.23 | 0.70 | 82.55 | 106.9% | 1.413 | 276 | 3.4 | 0 of 80 |
| `math_500:1024` | `plain` | 128 | 1024 | 61.72 | 0.70 | 82.34 | 75.0% | 1.000 | 564 | 3.4 | 2 of 128 |
| `math_500:1024` | `mtp_sidecar/block=7` | 128 | 1024 | 99.33 | 0.70 | 82.34 | 120.6% | 1.609 | 564 | 3.4 | 2 of 128 |
| `math_500:1024` | `dflash/block=16,dflash/depth=accept_rate` | 128 | 1024 | 84.93 | 0.70 | 82.34 | 103.1% | 1.376 | 564 | 3.4 | 1 of 128 |
| `math_500:4096` | `plain` | 128 | 4096 | 60.22 | 0.70 | 81.73 | 73.7% | 1.000 | 1415 | 3.4 | 3 of 128 |
| `math_500:4096` | `mtp_sidecar/block=7` | 128 | 4096 | 96.13 | 0.70 | 81.73 | 117.6% | 1.596 | 1415 | 3.4 | 4 of 128 |
| `math_500:4096` | `dflash/block=16,dflash/depth=accept_rate` | 128 | 4096 | 82.73 | 0.70 | 81.73 | 101.2% | 1.374 | 1415 | 3.4 | 2 of 128 |
| `humaneval:1024` | `plain` | 128 | 1024 | 63.12 | 0.70 | 82.51 | 76.5% | 1.000 | 338 | 3.4 | 0 of 128 |
| `humaneval:1024` | `mtp_sidecar/block=7` | 128 | 1024 | 118.64 | 0.70 | 82.51 | 143.8% | 1.880 | 338 | 3.4 | 0 of 128 |
| `humaneval:1024` | `dflash/block=16,dflash/depth=accept_rate` | 128 | 1024 | 96.43 | 0.70 | 82.51 | 116.9% | 1.528 | 338 | 3.4 | 1 of 128 |
| **MACRO** | `plain` | 3 cells | 1024 | 62.42 | 0.70 | 82.47 | 75.7% | 1.000 | — | — | — |
| **MACRO** | `mtp_sidecar/block=7` | 3 cells | 1024 | 107.60 | 0.70 | 82.47 | 130.5% | 1.724 | — | — | — |
| **MACRO** | `dflash/block=16,dflash/depth=accept_rate` | 3 cells | 1024 | 89.86 | 0.70 | 82.47 | 109.0% | 1.440 | — | — | — |

MACRO is the mean over `humaneval:1024`, `math_500:1024`, `mt_bench:1024`
— one cell per dataset. Its ceiling column is the mean of those cells'
ceilings, so both sides of the ratio are averaged the same way.
`range %` is over the three pass means. `worst sample %` is the widest
across-pass range of any one sample, which a pass-mean range cannot see;
it is reported and never refused, because at a sampled temperature one
prompt generates different text of different length on each pass.

## The round loop

`tokens_per_round` is `1 + accept_rate × (block − 1)` while every round
drafts the configured block, so `block` is its maximum and `block kept` is
the fraction of the drafted block the verifier kept (`docs/SPECULATIVE.md`).
A loop that resized its block instead says so on its own `done` line, and
for those `block kept` is left empty rather than quoting a fraction of a
block the drafter did not always propose.

| cell | mode | tokens/round | block | block kept | accepted/step | accept rate |
|---|---|---:|---:|---:|---:|---:|
| `mt_bench:1024` | `mtp_sidecar/block=7` | 3.412 | 7 | 48.7% | 2.412 | 0.402 |
| `math_500:1024` | `mtp_sidecar/block=7` | 3.412 | 7 | 48.7% | 2.412 | 0.402 |
| `math_500:4096` | `mtp_sidecar/block=7` | 3.412 | 7 | 48.7% | 2.412 | 0.402 |
| `humaneval:1024` | `mtp_sidecar/block=7` | 3.412 | 7 | 48.7% | 2.412 | 0.402 |
| `mt_bench:1024` | `dflash/block=16,dflash/depth=accept_rate` | 2.874 | 16 | — (adaptive block) | 1.874 | 0.125 |
| `math_500:1024` | `dflash/block=16,dflash/depth=accept_rate` | 2.874 | 16 | — (adaptive block) | 1.874 | 0.125 |
| `math_500:4096` | `dflash/block=16,dflash/depth=accept_rate` | 2.874 | 16 | — (adaptive block) | 1.874 | 0.125 |
| `humaneval:1024` | `dflash/block=16,dflash/depth=accept_rate` | 2.874 | 16 | — (adaptive block) | 1.874 | 0.125 |

## The fixed-length prompt — 1355 tokens, plain decode

One prompt of a stated length, 1024 output budget, three
runs. The protocol's second figure is the *autoregressive* one, so this
block is not measured on a speculative arm at all. The body is cut from
`longctx_4k.json` to hit the token target exactly against this
checkpoint's tokenizer, so it is not checked in — it travels in the result
file with the measurement.

| figure | measured | bound | % of bound |
|---|---:|---:|---:|
| output speed (tok/s) | 61.83 | 81.41 | 76.0% |
| input speed (tok/s) | 4818.67 | — | — |
| peak `phys_footprint` (GB) | 9.64 | 8.80 | 91.3% |
| peak RSS (GB) | 9.42 | 8.80 | 93.4% |

The two resident rows read the other way round — their bound is a
**floor**, so the last column is how much of what the process held is
accounted for by the 8.67 GB of weights
text decode must hold plus 0.13 GB of KV at
1867 context. The remainder is allocator slack, activations,
the prompt cache and everything else a process holds; it is not waste and
this does not say it is. Both peaks are a sampled gauge, so both are a
lower bound on the true peak.

The input-speed bound is empty on purpose. `perf_ceiling.py` projects
prefill from a measured anchor row in `runs.db` and reports nothing at
all rather than guessing when it has none: a single achieved-GEMM
constant is not defensible on this host, where the recorded rows span a
7× range across models.

## Provenance

| field | value |
|---|---|
| backend | rmlx 0.4.1 (release-perf) |
| binary | `sha256:327e90b6f5e4f41e` |
| snapshot | `fixture__published-table-16L` |
| KV codec | `none` (read back from the engine) |
| hardware | `m5_max_128gb` |

| mode | run | thermal | host interference |
|---|---|---|---|
| `plain` | 2026-09-06T09:00:00Z | — | synthetic — no reading taken off this machine |
| `mtp_sidecar/block=7` | 2026-09-06T10:00:00Z | — | synthetic — no reading taken off this machine |
| `dflash/block=16,dflash/depth=accept_rate` | 2026-09-06T11:00:00Z | — | synthetic — no reading taken off this machine |

Every per-sample row behind these means is recordable into `runs.db` by
`scripts/ingest/published_ingest.py`, which is a separate and explicit
step: a measurement and a record are different acts, and `observations`
is append-only.
