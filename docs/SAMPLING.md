# Sampling reference

Per-token sampling in rMLX: from raw model logits to the next token id.

---

## Overview

Every autoregressive decode step produces a `[1, vocab]` logit tensor on the
GPU. The sampler converts it to a single token id, returned as a `[1] I32`
array. Two paths exist:

**Fast greedy** — `temperature == 0` and no penalties and no constraint mask.
One GPU `argmax` call, no host transfer, byte-identical across runs.

**Host path** — everything else. One GPU-to-host transfer, then pure-Rust
work covering constraint masking, logit biasing, penalties, temperature
scaling, nucleus and top-k filtering, and an inverse-CDF draw.

Both paths return the same shape (`[1] I32`) so all downstream call sites
(materialise via `to_bytes` → `i32::from_le_bytes`) are unchanged.

The two paths are not equally fast, and the gap is not small — see
[Cost of the host path](#cost-of-the-host-path) for the measured figures.

---

## Pipeline

```text
┌─────────────────────────────────────────────────────────────────────┐
│  [1, vocab]  logits  (F32 or BF16, on GPU)                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │  Fast-greedy gate                        │
        │  temp == 0  AND  !penalties_active()     │
        │  AND  no constraint mask                 │
        └──────────┬──────────────────┬────────────┘
                   │ yes              │ no
                   ▼                  ▼
        ┌──────────────────┐  ┌───────────────────────────────────────┐
        │  GPU argmax      │  │  GPU→host transfer  (single per step) │
        │  (no host work)  │  └────────────────┬──────────────────────┘
        └──────────────────┘                   │
                                               │
                                  ┌────────────▼───────────────┐
                                  │  1. constraint mask        │
                                  │     forbidden ids → -inf   │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  2. logit_bias             │
                                  │     logit[id] += bias      │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  3. repetition_penalty     │
                                  │     sign-aware multiplicat.│
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  4. presence_penalty       │
                                  │     subtract once per uid  │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  5. frequency_penalty      │
                                  │     subtract penalty×count │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  6. temperature scale      │
                                  │     + numerically-stable   │
                                  │       softmax              │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  7. top-p (nucleus)        │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  8. min-p                  │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  9. top-k                  │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │ 10. renormalise             │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │ 11. inverse-CDF sample     │
                                  │     (PCG32 per request)    │
                                  └────────────┬───────────────┘
                                               │
                                  ┌────────────▼───────────────┐
                                  │  [1] I32  token id         │
                                  └────────────────────────────┘
```

Application order for steps 2–5 (`apply_penalties`) and 7–9 (`filter_top_p`
→ `filter_min_p` → `filter_top_k`) matches mlx-lm `make_logits_processors`
and `make_sampler` exactly.

---

## Per-stage deep dive

### Greedy (temperature == 0)

When `SamplerConfig::temperature <= 0.0`, `sampling_active()` returns
`false`. The decode loop branches to the GPU `argmax`:

```
argmax(&logits_flat, -1, device)   →  [1] I32
```

No host transfer. No random draw. The result is byte-identical for the same
model, prompt, and weights — exact reproducibility is a hard guarantee.

Three sub-cases refine the greedy path:

| Condition | Path |
|---|---|
| No constraint, no penalties | Pure GPU `argmax` |
| Constraint mask, no penalties | `apply_mask_argmax` — additive `-inf` bias on GPU, then `argmax` |
| Penalties active, with or without a constraint | `argmax_with_penalties` — GPU→host, mask, penalties, host argmax |

`apply_mask_argmax` builds a F32 bias buffer on the host (0.0 for allowed,
`-inf` for forbidden), wraps it as a `[1, vocab]` MLX array, adds it to the
logits (GPU op, promotes BF16 to F32 automatically), then calls `argmax`.
Overhead versus unconstrained: approximately 0.05 ms host-side bias fill for
a 262K-token vocabulary.

#### Tie-break contract

**Selection and filtering resolve an exact tie to the lowest token id, on the
host and on the device alike.** MLX's `argmax` reduces with a strict `>`, so an
equal value never displaces the earlier index; the host scan in
`argmax_with_penalties` (`host_argmax`) is written to mirror it. Three
consequences follow from the same rule and are pinned by tests:

- equal logits → lowest id,
- a `NaN` never displaces a real maximum,
- an all-`-inf` row (every token forbidden) yields id 0.

The rule matters because the three greedy sub-cases above must be
interchangeable: which one a request lands in is decided by whether a
constraint or a penalty happens to be set, and that must never change the
token on a row where the top logits tie. `Iterator::max_by` returns the *last*
maximum and so cannot be used here.

Ties are not exotic — see the measurement under
[Host selection is not bit-identical to the GPU argmax](#host-selection-is-not-bit-identical-to-the-gpu-argmax):
on a realistic 262144-wide BF16-derived softmax row, 259416 of 262143 adjacent
pairs are exactly equal.

**The primary evidence for the device half of the rule is the GPU test
`mlx_argmax_breaks_ties_to_lowest_index_gpu`**, which is mutation-verified: it
goes red when the rule is inverted, so it is a gate that can actually fail.

A census of real streams **corroborates** it. Two pure-GPU greedy 512-token
generations contained exact top-2 ties at 2 of 275 steps on Ternary-Bonsai-8B
(steps 70, 253) and at 4 of 200 on gemma-4-e2b (steps 29, 69, 108, 132). The
device emitted the **lower** tied id in every one — **6 of 6**, zero exceptions.
Step 132 of the gemma run is the divergence analysed under
[Host selection is not bit-identical to the GPU argmax](#host-selection-is-not-bit-identical-to-the-gpu-argmax).

Read that as corroboration, not as the basis of the contract. Six tied steps
from two runs on a single prompt is a small sample, and every tie in it comes
from the same pair of greedy streams. What it adds is that the rule holds on
real logits, which a fixture cannot show; it does not carry the contract alone.

**Scope.** The rule covers everything that selects or filters a token:

| Site | Rule |
|---|---|
| `argmax` / `apply_mask_argmax` (device) | equal logits → lowest id (MLX's own reduction) |
| `argmax_with_penalties` (host greedy) | mirrors it via `host_argmax` |
| `filter_top_k` | equal probabilities → lowest ids survive the cut, so `top_k = 1` is the argmax on tied rows too |
| `filter_top_p` | equal probabilities → lowest ids survive the nucleus |
| `compute_top_logprobs` | equal logits → ascending id, so rank 0 is the token `argmax` would pick |

It does **not** cover the inverse-CDF draw: a categorical sample from a tied
distribution is supposed to be random, and forcing it to the lowest id would
make `temperature > 0` biased.

Neither filter uses a comparator. Folding an unordered `NaN` pair to `Equal`
makes a `NaN` compare equal to everything, which is intransitive, and
`sort_unstable_by` may detect that and panic with "user-provided comparison
function does not correctly implement a total order" — mid-decode, aborting the
step. Adding an id tiebreak on top widens the window rather than closing it.
Both filters order integers under the standard `Ord` instead, so no comparator
exists to be intransitive.

They do it differently, because the two have different jobs:

- `filter_top_k` **partitions** `select_nth_unstable` over packed `u64` keys
  (inverted total-order bits above the token id). It needs a set, not an order,
  which is what mlx-lm's `argpartition` says too.
- `filter_top_p` needs the full ascending order for the cumulative sum, so it
  sorts the **values alone** and applies the id rule once, to the single tied
  group the cut lands in. Packing the id into the sort key would make every key
  distinct and destroy the equal-element partition that a tie-dense row hands
  the sort — that formulation is *slower* than no tie rule at all.

Measured at a 262144-token vocabulary (best-of-9, three process runs), against
the index sorts these replace:

| Filter | fixture | previous | now |
|---|---|---:|---:|
| `top_k` (k=64) | tie-dense (1233 distinct) | 2.02–2.13 ms | 0.30–0.33 ms |
| `top_k` (k=64) | all-distinct | 3.67–4.31 ms | 0.32–0.41 ms |
| `top_p` (0.95) | tie-dense | 2.02–2.06 ms | 1.31–1.34 ms |
| `top_p` (0.95) | all-distinct | 3.73–4.32 ms | 2.17–2.68 ms |

So pinning the tie order is a net speedup on every *served* distribution
measured, not a cost to be justified. Details on `rank_key_desc` and
`total_order_bits` in `sampler.rs`.

**Two shapes are marginally slower**, and neither is a distribution a model
produces:

| Shape | before | after |
|---|---:|---:|
| Perfectly uniform row (every probability identical) | 0.27 ms | 0.40 ms |
| Heavily constraint-masked row (nearly all mass at `0.0`) | 0.419 ms | 0.474 ms |

Both are cases where the sort it replaces gets a near-free equal-element
partition over one or two distinct values. The uniform row is also the only
shape where `filter_top_p`'s `tied` vector grows to the full vocabulary — on a
realistic tie-dense row it holds 808 entries (6 KiB), and on the masked row one.

The keys use the IEEE total-order flip rather than the raw bit pattern, so they
order every `f32` including negatives and `-0.0`. The raw pattern is monotone
only over non-negative values, and `probs` are non-negative today — but the
failure mode if that stopped holding is a silently wrong token (`-0.0` outranks
every positive, so `top_k = 1` would keep it and zero the real maximum), and
`release-perf` disables debug assertions, so an assert would not catch it.

**Where MLX does not match itself.** MLX seeds its CPU reduction with element 0
and its Metal reduction with `-inf`, so a `NaN` at index 0 returns 0 on CPU and
the first real maximum on Metal. No host rule can match both. `host_argmax`
follows Metal, which is the production stream; the CPU-stream unit tests avoid
that one shape. Everywhere else the two MLX backends agree with each other and
with `host_argmax`.

That seeding difference is also why the CPU tests are not sufficient. Three
`#[ignore]`d `Device::Gpu` mirrors re-run the contract on Metal, one per claim:
equal-logits-lowest-id, `NaN`-never-displaces, and the all-`-inf` row. The last
carries the most weight — it is the only shape where the `-inf` seed is never
displaced, so the answer is a property of the reduction's *seeding* rather than
of its comparisons, and its CPU test **cannot fail** (MLX's CPU backend returns
0 by construction whatever Metal does). Until that mirror has run, "an all-`-inf`
row yields id 0" is a claim about Metal taken from reading MLX's kernel, not a
measured one. It is no longer reachable through a constraint mask — that case
errors — but `host_argmax` still has to answer.

### Rows the sampler refuses

Two inputs are defects rather than modes, and both now return `Err` on the
channel the decode loop already propagates.

**An all-`false` constraint mask.** No token satisfies the grammar, so every
token the selection could return violates it. Returning one anyway is the worst
option: the engine state that produced the empty mask is persistent, so the
stream would emit the same arbitrary token (id 0) for the rest of the generation
while the request reports success. Logging instead is no better — the check sits
on the per-token decode path, so a `warn!` would fire once per emitted token
and, at a few hundred bytes a line, evict the whole log directory under
`RMLX_LOG_CAP_MB` within hours, deleting the evidence it exists to provide. All
three mask-accepting entry points (`apply_mask_argmax`,
`argmax_with_penalties`, `sampling_distribution`) refuse it.

**That guard is unit-tested only; it has no demonstrated production trigger.**
An attempt to construct an all-forbidden mask through the HTTP surface did not
find one. `{"enum": []}` is rejected at schema parse with HTTP 400, before any
mask is built; and because the tokenizer is byte-level BPE, exotic `const` or
single-`enum` values still leave a byte that continues them, so they never
starve the mask either. The constraint engine itself does engage on these
requests, so the guarded path is live — but the all-`false` state was not
reachable from outside. Read the guard as a defence against a future
constraint-engine defect, proven by construction in tests and by argument from
the code path, not as a reproduction of an observed served failure.

**A non-finite logits row, on the sampling path.** `softmax_scaled` errors when
the exponentials do not sum to a finite value, which happens exactly when a
logit is `NaN` or `+inf`. This is the decode-step half of the rule the prefill
guard already enforces, and it has to live here because that guard is a
*prefill* guard: on every test-target architecture it runs once before the loop
and never again, so nothing downstream reports a `NaN` arriving at decode step
300. The check is free — `sum` is already computed.

Sampling such a row is silent, not degraded: the `NaN` reaches `probs`,
`renormalise` no-ops (`total > 0.0` is false on `NaN`), `sample_inverse_cdf`'s
`total <= 0.0` guard is also false, its `cum > target` never fires, and it
returns `last_nonzero` — the same id every step, **independent of the RNG**.
Measured on a 16-wide row with one `NaN`: a constant token for every seed tried,
against a varied healthy control.

**Greedy deliberately does not refuse a `NaN` row.** It mirrors the device
reduction, which skips `NaN` and returns the largest real logit; erroring there
would re-create the host/device split this contract exists to close, since the
pure-GPU `argmax` cannot refuse anything without an extra per-token reduction.
The asymmetry is pinned by a test. The sampling path refuses because its failure
mode is a constant stream; the greedy path does not because its answer is the
device's.

### Temperature scaling and softmax

When `temperature > 0.0` the host path runs. After the constraint mask and
penalty steps, logits are scaled by `1 / temperature` and passed through a
numerically stable softmax:

```
scaled[i] = logit[i] * (1 / temperature)
max        = max(scaled)
prob[i]    = exp(scaled[i] - max) / sum_j(exp(scaled[j] - max))
```

The subtract-max shift prevents overflow for logits of large magnitude. The
result is a proper probability vector summing to 1.0.

Positions already set to `-inf` (from the constraint mask) softmax to
exactly 0.0 and contribute nothing to the renormalisation sum.

### Top-p (nucleus) truncation

Applied after softmax. Mirrors mlx-lm `apply_top_p` (`sample_utils.py`
L205–237).

Sort probabilities ascending. Walk them in ascending order, accumulating an
inclusive cumulative sum. A token survives iff its inclusive cumulative
probability is strictly greater than `1 - top_p`:

```
survived tokens: those where cum_ascending > (1 - top_p)
```

Tokens below the threshold go to 0.0. No-op unless `0 < top_p < 1`.

The ascending-sort + inclusive-cumsum direction is exact mlx-lm parity and
is not interchangeable with descending order.

Among equal probabilities the **lowest ids** survive the cut — the same
lowest-id-wins rule the device `argmax` and `top_k` use. Unlike `top_k`, this
filter is on by default on the served path: several `generation_config.json`
snapshots ship a `top_p`. Without the rule the survivor set is an artefact of
the sort's pivot choice — on a row of 64 with one 0.4 and 63 identical tail
values at `top_p = 0.5`, the unordered version keeps `{0, 29, 54..63}`, id 29
surviving while ids 30..53 are zeroed from a bit-identical value.

The *number* of survivors is unaffected by the rule: probabilities are
non-negative, so the drop set is a prefix of the ascending order, and every
member of a tied group contributes the same value to the cumulative sum. Only
*which* members are dropped depends on the order within a group. That is what
lets the implementation sort the values alone — keeping the equal-element
partition a tie-dense row hands the sort — and then split just the one tied
group the cut lands in, dropping its highest ids.

### Min-p truncation

Applied after top-p. Mirrors mlx-lm `apply_min_p` (`sample_utils.py`
L155–201).

Remove any token whose probability is below `max_prob * min_p`:

```
cutoff = max(probs) * min_p
prob[i] = 0.0  if prob[i] < cutoff
```

`min_tokens_to_keep` is 1; the maximum-probability token always satisfies
the threshold so no special floor-of-1 case is needed. No-op when
`min_p <= 0.0`.

### Top-k truncation

Applied after min-p. Mirrors mlx-lm `apply_top_k` (`sample_utils.py`
L130–151).

Keep exactly the `k` highest-probability tokens; set the rest to 0.0. No-op
when `k == 0` or `k >= vocab`.

Equal probabilities are ranked by **ascending token id**, so a cut that falls
inside a tied group keeps the lowest ids — the same rule the device `argmax`
uses, which is what makes `top_k = 1` reduce to greedy on a tied row as well
as on an untied one. mlx-lm's `argpartition` leaves the tied order
unspecified; rMLX pins it rather than inheriting whatever the sort's pivot
choice produces.

### Renormalisation

After top-p / min-p / top-k filtering the surviving probabilities no longer
sum to 1.0. A single normalisation pass restores the invariant before the
CDF draw.

### Repetition penalty

Part of `apply_penalties`, applied before softmax. Mirrors mlx-lm
`make_repetition_penalty`. Sign-aware multiplicative penalty over the
last-20-token window:

```
if logit[id] < 0:  logit[id] *= rep_penalty
else:              logit[id] /= rep_penalty
```

Applied once per unique token id in the window regardless of count. Identity
value: `1.0`.

**Window note.** The window is the trailing 20 generated tokens
(`context_size=20` in mlx-lm). This diverges from OpenAI's full-context
semantics: tokens repeated more than 20 positions ago are not penalised.

### Presence penalty

Applied after repetition penalty. Subtracts a flat value once for every
token id that appears at least once in the last-20 window:

```
logit[id] -= presence_penalty  (once per unique id)
```

Identity value: `0.0`.

### Frequency penalty

Applied after presence penalty. Subtracts a value proportional to the count
of each token id in the last-20 window:

```
logit[id] -= frequency_penalty * count(id, window)
```

Identity value: `0.0`.

### Logit bias

Applied first in `apply_penalties`, before repetition/presence/frequency:

```
logit[id] += bias  for each (id, bias) pair
```

Out-of-vocabulary ids are silently skipped. The bias is additive and
unbounded; setting `bias = -inf` hard-bans a token.

### Hot-path discriminant

`PenaltyConfig::penalties_active()` returns `false` when all four fields are
at their identity values (`rep_penalty == 1.0`, `presence_penalty == 0.0`,
`frequency_penalty == 0.0`, `logit_bias.is_empty()`). The decode loop checks
this once per step; if false and temperature is also 0, the entire host path
is skipped and no GPU-to-host transfer occurs.

---

### Inverse-CDF sample

Given the filtered, renormalised probability vector, one `Pcg32` draw
produces the chosen token:

```
r = rng.next_f32()          // uniform [0, 1)
target = r * sum(probs)
chosen = first i where cumsum(probs[0..=i]) > target
```

Falls back to the last nonzero index on floating-point drift (degenerate
case; at least one nonzero probability is guaranteed by the constraint-mask
invariant).

### RNG — PCG32

A minimal PCG32 (O'Neill 2014, PCG-XSH-RR variant) is instantiated per
request from `SamplerConfig::seed_or_default()`. The instance is threaded
through every decode step so the random stream is contiguous. No `rand`
crate dependency.

Seeding:
```
seed = request.seed.unwrap_or(0xA7A7)
```

Absent-seed requests always use `0xA7A7`, making them deterministic per
model at temperature > 0. This is a documented contract; callers that want
true stochasticity must supply a distinct seed per request.

---

## Cost of the host path

The host path is not a cheap variant of the greedy one. It costs a
`vocab`-sized GPU-to-host transfer plus `O(vocab)` host arithmetic per token,
and — because the next forward cannot be dispatched until the row has been read
back and a token chosen — it also gives up the software pipelining the greedy
path gets for free. Greedy returns a *lazy* GPU argmax and dispatches the next
step while the GPU is still busy; the host path runs strictly serial.

**It is the served default, not an opt-in.** `resolve_sampling_params` takes
temperature from, in order: the request, `--default-temperature`,
`generation_config.json`, then a hard-coded `1.0`. So a
`/v1/chat/completions` request that omits sampling fields never lands on the
greedy path unless an operator put it there. Worse for cost, the snapshots
carry filters too — gemma-4 ships `temperature 1.0, top_p 0.95, top_k 64` and
Qwen3.6 `1.0 / 0.95 / 20` — so the served default is the *most* expensive shape
below, not the cheapest.

### The instrument

The shared decode loop emits a `sampler_profile` event once per generation,
covering only the steps that took the host path. A purely greedy run emits
nothing and takes no extra clock readings.

| Field | Meaning |
|---|---|
| `sync_per_step_ms` | Wait for the forward before the row can be read. GPU latency, not sampler cost. |
| `sample_per_step_ms` | Readback plus all host arithmetic — mask, penalties, softmax, filters, draw, and logprob capture. |
| `step_per_step_ms` | Whole step, host-path steps only. |
| `sample_share_pct` | `sample / step`. |

Drive the path from the bench harness with `rmlx bench --temperature`,
`--top-p`, `--top-k` and `--repetition-penalty`. `scripts/perf_canary.sh` is
greedy-only and cannot observe this class at all.

**`sample_share_pct` is not a bound in either direction.** Two errors act on
it with opposite signs:

- It *understates*, by omitting cost the host path causes but does not spend
  inside the window: the forfeited pipelining, and — on the constraint-mask-only
  path — the GPU add + argmax that `apply_mask_argmax` merely schedules, which
  execute at the next step's `eval` and are billed to `sync`. Every other path
  forces the row to the host inside the window and is fully accounted.
- It *overstates* on a contended host, because `sample` is pure host CPU while
  `sync` is dominated by GPU execution, so CPU steal stretches the numerator
  only. Every figure below was taken under such contention.

The end-to-end figure is a decode-TPS comparison against a greedy control at the
same shape; the share is the component attributable to host work.

### Measured, 2026-08-16, M5 Max — PROVISIONAL

`release-perf`, `rmlx bench --max-tokens 100`, 1 warmup + 3 measured runs at 4k
and 1 + 2 at 16k/64k. `share%` is the **median over the measured generations**
(the warmup's event is discarded); it is a within-step ratio, so it survives a
cell whose `decode_tps` the harness refused to median.

**These are single-arm medians on a contended host, not an interleaved A/B.**
`scripts/perf_ab.sh` refused the host (exit 125: a VM at 114 % of a core, later
joined by an npm process at 131 %), and the threshold was not raised. Treat the
column separations as the result and the third digit as noise. The dataset's own
noise floor is about 2.5 points: at 4k, gemma-4-e2b's `temp 0.7 + rep 1.1` cell
measured *faster* than its `temp 0.7` cell (113.86 vs 111.03 tok/s) despite
doing strictly more work. No throughput delta below that is reported as an
effect.

#### Share versus context

`sample` is `O(vocab)` and context-invariant; `step` grows with context. The
share is therefore a falling curve, and a single short-context point is its
maximum rather than its value. `kv_quant=auto`:

| model | vocab | attention | cell | 4k | 16k | 64k |
|---|---:|---|---|---:|---:|---:|
| gemma-4-e2b | 262144 | sliding-window | `--temperature 0.7` | 10.00 % | 9.29 % | 8.30 % |
| gemma-4-e2b | 262144 | sliding-window | `--repetition-penalty 1.1` | 2.76 % | 2.63 % | 2.24 % |
| gemma-4-e2b | 262144 | sliding-window | served default | 24.82 % | 24.35 % | 22.30 % |
| Qwen3.6-35B-A3B | 248320 | global | `--temperature 0.7` | 6.98 % | 6.56 % | 4.58 % |
| Qwen3.6-35B-A3B | 248320 | global | `--repetition-penalty 1.1` | 2.04 % | 1.94 % | 1.41 % |
| Qwen3.6-35B-A3B | 248320 | global | served default | 22.55 % | 21.15 % | (cell did not complete) |
| Ternary-Bonsai-8B | 151669 | global | `--temperature 0.7` | 5.80 % | 0.65 % | 0.18 % |
| Ternary-Bonsai-8B | 151669 | global | `--repetition-penalty 1.1` | 1.78 % | 0.19 % | 0.06 % |
| Ternary-Bonsai-8B | 151669 | global | served default | 5.88 % | 0.65 % | 0.18 % |

`sample` itself is flat across context, as predicted: gemma-4-e2b holds
0.901 / 0.854 / 0.879 ms per step at temperature 0.7, Bonsai 0.498 / 0.499 /
0.510, Qwen3.6 0.796 / 0.811 / 0.807. Everything the curve does, it does through
the denominator.

**Gemma-4 barely falls at all** — 10.0 % to 8.3 % over a 16× context increase —
because its sliding-window attention keeps step time nearly context-independent
(8.92 → 10.59 ms). An arch that does not pay for context does not dilute a
context-invariant cost either.

**Bonsai's collapse is a codec artifact, not attention.** Its `auto` codec
(`mixed_k8g64_v4g64`) decodes at 13.2 tok/s at 16k and 3.6 at 64k — step times
of 76 and 282 ms, against 11.1 and 24.7 ms for the same model and contexts on
`--kv-quant none` below. The extra time is host-side work inside the forward
(`sync_per_step_ms` stays at 2.8 ms while the step runs 282 ms), so the share
there is being divided by a separate defect rather than by attention. The `none`
arm is the one to read for that model. The codec gap itself is worth its own
look and is not this section's subject.

Bonsai's "served default" row is not the expensive shape the other two models
show, for a mundane reason: it ships no `generation_config.json`, so its served
default is the hard-coded `temperature 1.0` with no filters — the same cost as
the temperature cell. The filters are what make gemma-4's and Qwen3.6's served
defaults three to four times dearer.

#### The same models on a healthy codec (`--kv-quant none`)

Re-run with `--kv-quant none`. Its greedy control decodes Bonsai at 95.6 tok/s
at 16k and 41.4 at 64k, against the ~83 and ~38 recorded for that model and
codec in `docs/PERF_BASELINE.md` — the same regime, on a busier host, rather
than the 13.2 / 3.6 above. The denominator is legitimate attention cost:

| model | ctx | cell | `sample` ms/step | `step` ms/step | `share%` | decode_tps |
|---|---:|---|---:|---:|---:|---:|
| Ternary-Bonsai-8B | 16k | `--temperature 0.7` | 0.505 | 11.13 | 4.54 % | 87.8 |
| Ternary-Bonsai-8B | 16k | `--repetition-penalty 1.1` | 0.150 | 10.55 | 1.42 % | 92.8 |
| Ternary-Bonsai-8B | 64k | `--temperature 0.7` | 0.519 | 24.70 | 2.10 % | 39.5 |
| Ternary-Bonsai-8B | 64k | `--repetition-penalty 1.1` | 0.143 | 23.76 | 0.60 % | 41.1 |
| Qwen3.6-35B-A3B | 16k | `--temperature 0.7` | 0.828 | 12.39 | 6.68 % | 80.7 |
| Qwen3.6-35B-A3B | 16k | `--repetition-penalty 1.1` | 0.239 | 11.30 | 2.12 % | 88.5 |
| Qwen3.6-35B-A3B | 64k | `--temperature 0.7` | 0.838 | 16.00 | 5.25 % | (unsettled) |
| Qwen3.6-35B-A3B | 64k | `--repetition-penalty 1.1` | 0.239 | 16.28 | 1.47 % | (unsettled) |

The two Qwen3.6 64k cells had their `decode_tps` refused for not settling under
host contention; their `share` is a within-step ratio and is unaffected.

#### Per-context verdict against the issue's kill criterion

The criterion — host share under 3 % for both temperature and repetition
penalty — evaluated per context rather than once:

| model | codec | 4k | 16k | 64k |
|---|---|---|---|---|
| gemma-4-e2b (sliding-window, vocab 262144) | auto | **NOT MET** 10.00 / 2.76 | **NOT MET** 9.29 / 2.63 | **NOT MET** 8.30 / 2.24 |
| Qwen3.6-35B-A3B (global, 248320) | auto | **NOT MET** 6.98 / 2.04 | **NOT MET** 6.56 / 1.94 | **NOT MET** 4.58 / 1.41 |
| Qwen3.6-35B-A3B | none | — | **NOT MET** 6.68 / 2.12 | **NOT MET** 5.25 / 1.47 |
| Ternary-Bonsai-8B (global, 151669) | auto | **NOT MET** 5.80 / 1.78 | (MET 0.65 / 0.19 — collapsed denominator, see above) | (MET 0.18 / 0.06 — same) |
| Ternary-Bonsai-8B | none | — | **NOT MET** 4.54 / 1.42 | **MET** 2.10 / 0.60 |

(`temperature 0.7` / `repetition-penalty 1.1`, in that order.)

**The criterion is not met.** It is met in exactly one of the nine legitimate
(model, context, codec) cells — Ternary-Bonsai-8B at 64k on `none` — and missed
everywhere else, including at 64k on both other architectures. The two Bonsai
`auto` cells that clear it do so against a step time inflated by the codec, not
by attention.

What the sweep does narrow is the shape of the claim. The cost falls with
context on globally-attending models, so it is worst at short context; and it
falls hardly at all on sliding-window attention, where gemma-4-e2b still pays
8.3 % at 64k. It also scales with vocabulary. "Sampling is expensive" is
therefore too broad: it is expensive at large vocabularies, at short-to-medium
context, and on architectures whose step time does not grow with context — and
it is expensive at *every* context measured once the ordering filters the served
defaults enable are switched on.

#### Cost by knob, at 4k

| Model | vocab | cell | `sample` ms/step | `share%` | decode TPS vs greedy |
|---|---:|---|---:|---:|---:|
| gemma-4-e2b | 262144 | greedy | — | — | 129.22 (control) |
| gemma-4-e2b | 262144 | `--repetition-penalty 1.1` | 0.228 | 2.75 | −4.9 % |
| gemma-4-e2b | 262144 | `--temperature 0.7` | 0.880 | 9.80 | −14.1 % |
| gemma-4-e2b | 262144 | `--temperature 0.7 --repetition-penalty 1.1` | 0.874 | 9.88 | −11.9 % |
| gemma-4-e2b | 262144 | `--temperature 0.7 --top-p 0.95` | 2.525 | 22.89 | −29.9 % |
| Ternary-Bonsai-8B | 151669 | greedy | — | — | 134.50 (control) |
| Ternary-Bonsai-8B | 151669 | `--repetition-penalty 1.1` | 0.141 | 1.72 | −12.4 % |
| Ternary-Bonsai-8B | 151669 | `--temperature 0.7` | 0.487 | 5.82 | −13.5 % |
| Ternary-Bonsai-8B | 151669 | `--temperature 0.7 --repetition-penalty 1.1` | 0.491 | 5.87 | −14.2 % |
| Ternary-Bonsai-8B | 151669 | `--temperature 0.7 --top-p 0.95` | 1.665 | 17.15 | −26.0 % |
| Qwen3.6-35B-A3B | 248320 | greedy | — | — | 98.61 (control) |
| Qwen3.6-35B-A3B | 248320 | `--repetition-penalty 1.1` | 0.218 | 2.04 | −5.3 % |
| Qwen3.6-35B-A3B | 248320 | `--temperature 0.7` | 0.790 | 7.09 | −9.2 % |
| Qwen3.6-35B-A3B | 248320 | `--temperature 0.7 --top-p 0.95` | 2.847 | 20.67 | −26.4 % |

Reading it:

- **Temperature alone** is 5.8–9.8 % of step time in host work at 4k. The work
  is the transfer plus a full-vocabulary softmax.
- **A repetition penalty alone** is the cheapest host cell — 1.7–2.8 % — because
  the penalty arithmetic touches at most 20 ids; the transfer and the host argmax
  are the whole cost. Its throughput deltas (−4.9 / −12.4 / −5.3 %) are *not*
  explained by `sample`: subtracting it from the greedy step leaves 0.17 ms
  (gemma), 0.64 ms (Bonsai) and 0.35 ms (Qwen3.6) unaccounted, a spread too wide
  for a structurally identical change. The forfeited pipelining is the obvious
  candidate and it is untested — no measurement here isolates it.
- **The ordering filters are the dominant term.** `--top-p 0.95` roughly triples
  the host work, to 17–23 % of step time, because it sorts the entire
  vocabulary; `--top-k` does the same. Anyone optimising this path should start
  there and should not assume the other stages behave alike.
- The cost tracks vocabulary size within a stage: at temperature 0.7 the
  262144-token vocabulary pays 0.880 ms/step against 0.487 ms/step for the
  151669-token one.

### Host selection is not bit-identical to the GPU argmax

At `--temperature 0.0001` the host categorical sampler is *close to* an argmax,
so its token stream might be expected to match the greedy GPU `argmax` stream.
It does not. The divergence is reproducible and every run within a cell agrees,
so it is a deterministic property of the two paths and not a race.

**Which architecture exhibits it is a property of the prompt and the window
length, not of the architecture, and must not be used to scope the defect.**
The original filing saw gemma-4-e2b agree with greedy for all 100 tokens while
Ternary-Bonsai-8B agreed through 32 and diverged by 64. A later run at 512 max
tokens on a different prompt (`--kv-quant none --max-ctx 4096`, `release-perf`)
inverted that exactly: Ternary-Bonsai-8B was identical to greedy for all 275
tokens it emitted, and gemma-4-e2b diverged at step 132. Both observations are
real and neither is a fact about the model. A window that ends before the first
tied row shows agreement and nothing more, so an arm that matches is evidence of
a short window, not of an unaffected architecture.

#### What is proven

**A near-zero temperature is not an argmax, and is not a valid oracle for
greedy.** Two separate reasons, both pinned by unit tests:

- The underflow window is computable. `exp` in f32 returns exactly zero below
  about `-104`, so a logit keeps non-zero probability iff it is within
  `104 * temperature` of the maximum — 0.0104 at `temperature = 1e-4`. Inside
  that window the distribution is genuinely mixed and the draw is genuinely
  random.
- At an **exact** tie the post-softmax distribution is uniform over the tied
  ids, and the inverse-CDF draw picks among them by the RNG. It matches the
  device `argmax` — which takes the lowest tied id — only about `1/k` of the
  time. That is a categorical sampler behaving correctly.

So the argument "temperature ~0 should match greedy, therefore a mismatch is a
bug" does not hold, and neither does the reverse inference: a mismatch there is
not evidence of a defect in either path.

**Exact ties are common, not exotic.** Logits are BF16 on most snapshots — 8
mantissa bits, so 0.125 spacing at magnitude 16. Measured on a 262144-wide
BF16-derived softmax row, 259416 of the 262143 adjacent pairs are exactly
equal. Tie behaviour is the common case in this code, which is why the rank
rules below are pinned rather than left to a sort's internals.

**Host greedy was resolving ties the opposite way from the device.**
`argmax_with_penalties` selected with `Iterator::max_by` (last maximum wins)
against MLX `argmax`'s first-maximum-wins, let a `NaN` reset the running best,
and returned the last id rather than 0 on a fully masked row. Fixed, and pinned
by tests that use MLX's own reduction as the oracle. See the tie-break contract
under [Greedy](#greedy-temperature--0).

#### The mechanism, established

**The divergence is the exact-tie mechanism. The residual is closed.** The
closure condition this section set for itself was the gap between the top two
logits at the first divergent step, read via `logprobs: true` with
`top_logprobs: 2` (which reports `logit - lse` per rank): a gap of `0` or below
`104 * temperature` confirms the tie/near-tie mechanism, a gap well above it
kills the tie explanation. That was run, on the gemma-4-e2b divergence above.

At step 132 the greedy stream emits 16939 (`▁featured`) and the
temperature-`1e-4` stream emits 22420 (`▁incorporated`). Both ranks report the
same logprob, `-1.0606343`, bit for bit — a gap of **exactly `0.0`**. That is an
exact tie, not a near-tie: the post-softmax distribution over the two ids is
uniform, the inverse-CDF draw takes whichever the RNG lands on, and the device
`argmax` takes the lower id. Both paths are correct and there is no third
mechanism left to look for. The near-tie candidate (a gap inside the 0.0104
window, same symptom without an exact tie) did not need to be invoked, and
"something outside the sampler" is excluded by a gap that is identically zero.

One control makes this a measurement rather than a coincidence: the
temperature-`1e-4` streams are **byte-identical across the two arms** of this
change. That confirms on real output what the code path already implies — the
temperature path never enters `argmax_with_penalties`, so the greedy tie fix
neither introduced this divergence nor masked it.

The original filing also offered `--repetition-penalty 1.1` producing the same
divergent stream as corroboration. It is not: that flag divides positive logits
of the trailing-20 ids by 1.1, which is a different objective, not another
route to the same argmax. Its agreement with the temperature stream is evidence
for nothing either way.

What this does **not** license is treating a near-zero temperature as an oracle
for greedy. The mechanism is understood; the two streams still legitimately
differ, for the reasons under [What is proven](#what-is-proven) above.

#### Consequence for a future fused GPU sampler

The merge gate cannot be "the token stream matches a near-zero-temperature CPU
run", and it cannot be "the streams match" at any `temperature > 0` unless the
RNG matches too. The gate that survives is two-part: on the greedy path, exact
token identity against the device `argmax` including the lowest-id tie rule; on
the stochastic path, exact token identity against the CPU path **given the same
`Pcg32` draw sequence** — a GPU RNG must reproduce `Pcg32` bit-for-bit, or the
kernel ships behind a dispatch policy with the CPU path retained as the oracle.

## Special tokens

**EOS (end-of-sequence).** Each architecture exposes `eos_token_ids()` from
its `config.json`. The field may be a single integer or an array; both are
normalised to `Vec<u32>`. The decode loop checks `eos_ids.contains(&token_id)`
after each step and sets `finish_reason = "stop"` on the first match.

**BOS (beginning-of-sequence).** Used only at prompt construction (smoke
probe, bare-BOS seeding). Not involved in the decode loop itself.

**PAD.** Not used at inference time; rMLX does not pad within a single decode
stream.

**Think markers.** Qwen3-family models use `<think>` and `</think>` as
plain string tokens in the generated output. These pass through the normal
decode loop unchanged. The `ThinkSplitter` (server layer, `engine.rs`) reads
the decoded piece strings and routes them to `reasoning_content` vs `content`
in the OpenAI response — it does not intercept token ids in the sampler.

The `thinking_end_token_id` (the `</think>` token id) is resolved from the
tokenizer at request setup and forwarded to the decode loop exclusively for
the budget-forced injection path described below.

---

## Thinking-budget enforcement

Whether the assistant turn begins inside a `<think>` block is a property of
the **checkpoint's chat template**, not of the architecture. Templates in the
Qwen3 family do all three: prefill an open `<think>\n`, prefill a *closed*
`<think>\n\n</think>\n\n` so the model answers directly (Ternary-Bonsai does
this unconditionally, ignoring `enable_thinking`), or prefill nothing and let
the model emit its own `<think>`.

The server therefore reads the initial channel off the rendered prompt —
`engine::think::prompt_leaves_think_open`, which is `true` iff the last
thinking-start delimiter in the prompt comes after the last thinking-end
delimiter — and threads it as `GenerationRequest::prompt_think_open`. That is
the `ThinkSplitter`'s initial state and the seed of the constraint engine's
`is_thinking` handle.

Getting it from the architecture instead does not self-correct: a splitter
started open against a template that already closed the block never sees the
`</think>` that would close it, so `is_thinking` stays `true` for the whole
request. Everything downstream latches with it — all output is reported as
`reasoning_content`, and a `json_schema` constraint whose engage gate defers
while thinking never engages at all.

From that initial channel the decode loop emits reasoning tokens until the
model produces `</think>`, then switches to the answer channel.

`ThinkSplitter` (server layer) tracks the count of pieces routed to the
thinking channel. When `thinking_budget` is set and the count exceeds the
cap, `ThinkSplitter::account_thinking_piece()` latches `force_close = true`.

The decode loop checks `step_fn` return values for a forced token id after
each emitted step. When `take_force_close()` returns `true`, the server's
`step_fn` callback returns `Some(thinking_end_token_id)`. The decode loop:

1. Discards the GPU-pipelined successor token for this step.
2. Constructs a `[1] I32` array from the forced token id.
3. Feeds it as both the next decode input (`y`) and the pending output token.
4. Clears `forced_next` so the injection fires exactly once.

The model then continues from the `</think>` state, producing answer tokens.
Forced injection tokens carry no logprobs (`pending_logprobs = None`).

This path is zero-overhead when `thinking_budget == None`: a single
`Option` discriminant check in `account_thinking_piece()` returns
immediately.

---

## Constrained decoding

`response_format: json_object` and `response_format: json_schema` activate
a `ConstraintEngine` for the request. The engine is stateful; two methods
are called per decode step:

- `step_mask(vocab_size) -> &[bool]` — returns a vocabulary-sized boolean
  allow-mask. `true` = token may be sampled; `false` = logit set to `-inf`.
- `advance(token_id)` — informs the engine which token was chosen so it can
  advance its grammar state.

The mask is consulted before penalties and softmax, so constraint filtering
composes with all other sampling parameters.

**No-op path.** When `response_format` is absent, no `ConstraintEngine` is
constructed. The per-arch decode loops pattern-match on
`Option<&mut dyn ConstraintEngine>` and take the fast branch on `None` — one
discriminant check, no allocation, output byte-identical to unconstrained.

**Greedy + constraint.** When `temperature == 0` and a constraint is active,
`apply_mask_argmax` keeps the argmax on the GPU (additive bias, then
`argmax`). No host transfer occurs beyond what the GPU op needs.

**Stochastic + constraint.** When `temperature > 0`, the host path runs.
`step_mask` output is passed as the `mask` argument to `sample_token_array`
and applied before softmax.

**`wants_mask()` hint.** A `ConstraintEngine` may return `false` from
`wants_mask()` during a warm-up phase (for example, while the Qwen3
`<think>…</think>` block is still open). When `wants_mask() == false` the
decode loop skips `step_mask()` and uses the unconstrained fast path for
that step.

**JSON engine.** `constraint_json::JsonObjectConstraint` (free-form
`json_object`) and `constraint_json::SchemaConstraint` (`json_schema`)
implement `ConstraintEngine`. Construction decodes every token in the
vocabulary once (approximately 600 ms for a 152K-token vocabulary — one-time
cost, hidden behind TTFT). Per-step mask build is an O(vocab) byte-set
membership sweep: for each candidate token a shared scratch grammar is reset
to the current state and the token's bytes are fed through it. The reset is
the per-token cost driver, so both engines keep their grammar's *immutable*
part cheap to copy — `JsonObjectConstraint` uses `Copy` frames, and
`SchemaConstraint` holds the parsed schema (property lists, element schemas,
literal-alternative sets) behind `Arc` and refills only the tiny mutable
progress (`emitted` keys, trie `viable`/`pos`, phase) in place. A reset is
therefore a handful of refcount bumps plus small buffer refills rather than a
deep copy of the schema subtree. The free-form engine is ~1 ms/step on
Qwen3.6; schema masking is heavier (the residual is the O(vocab) sweep and
the literal-trie step-work, not the grammar copy) but no longer pays a
per-token schema deep-clone.

EOS tokens are naturally masked out mid-JSON (special tokens decode to empty
or zero first byte; the state machine never allows byte value 0). At
terminal states (valid JSON complete) EOS ids are explicitly forced to `true`
in the mask so the decode loop's EOS stop predicate can fire normally.

### Whitespace is bounded, on purpose

Withholding EOS until the value is complete is what makes an over-permissive
grammar dangerous. Any byte the grammar accepts without making progress is a
cycle the decoder can sit in, and at `temperature == 0` it will: the mask
keeps offering that byte and keeps refusing EOS, so the request runs to
`max_tokens` and returns HTTP 200 carrying nothing usable. Both engines
therefore cap a run of *insignificant* whitespace at
`constraint_json::MAX_INSIGNIFICANT_WS_RUN` (16) bytes; any content or
structural byte resets the counter. No JSON document becomes unreachable —
only indentation deeper than the cap is clipped. This is the same reasoning
behind the bounded `space` rule llama.cpp generates from a JSON schema.

Two positions are *not* insignificant whitespace and reject it outright: raw
C0 control bytes inside any JSON string, including an object **key** string
(RFC 8259 requires them escaped), and whitespace before the root value.

`make schema-constraint-canary` (`scripts/schema_constraint_canary.sh`) is the
real-model proof for both properties, on Bonsai and gemma-4-e2b. Its PASS/FAIL
rule is fixed at the top of the script, and `EXPECT=baseline` inverts the exit
code so a harness too weak to see the defect fails as loudly as a broken fix.

### Non-enforcement is reported

Both JSON engines have a warm-up phase and only start masking once the model
emits something the grammar can latch onto — a value-starter byte for a
container root, the first post-reasoning token for a scalar root.
`ConstraintEngine::engaged()` exposes whether that ever happened. A
generation that ends with it still `false` was never constrained, and its
output is byte-for-byte indistinguishable from output the grammar inspected
and permitted, so the decode loop emits a `warn!` naming the model and
session. The response itself is still returned with HTTP 200: the tokens have
already been streamed by the time the engine's terminal state is known, so
there is no honest way to refuse mid-flight. Callers that must not accept an
unenforced result should validate the response body against their schema.

---

## Logprobs

Logprob capture is disabled by default (`top_logprobs_k == 0`). Setting
`top_logprobs_k > 0` (via the OpenAI `logprobs: true` + `top_logprobs: k`
request fields, cap 20) enables `compute_top_logprobs` after each decode
step.

`compute_top_logprobs` performs one GPU-to-host transfer (shared with the
`temp > 0` path when both are active; otherwise an additional transfer). It
computes a numerically stable log-softmax over the raw logits:

```
lse = max(logits) + ln(sum(exp(logits - max)))
logprob(i) = logits[i] - lse
```

The top-k selection is a partial selection sort, O(vocab × k), appropriate
for the k ≤ 20 cap.

**Semantics.** Logprobs are computed from the raw model logits before
temperature, penalties, or nucleus filtering. This matches OpenAI's contract:
the reported probability is the model's own per-token log-likelihood, not the
post-sampling distribution.

**Zero-overhead contract.** When `top_logprobs_k == 0` (the default), no
log-softmax or allocation occurs on the decode hot path. The decode loops
gate all logprob work behind an explicit `if lp_k > 0` check.

The captured `TokenLogprobs` struct carries:
- `token_id` — the sampled or argmax token id for this step.
- `token_logprob` — `ln P(token_id)` under the raw-logit softmax.
- `top` — the `k` highest-probability `(token_id, logprob)` pairs,
  descending by logprob. May or may not include the chosen token.

Equal logits rank by **ascending token id**, so rank 0 is the token the device
`argmax` would select and ranks `1..k` are reproducible rather than an artefact
of the selection's swaps. That makes `top_logprobs: 2` usable as the tie probe
described under
[Host selection is not bit-identical to the GPU argmax](#host-selection-is-not-bit-identical-to-the-gpu-argmax):
`top[0].logprob - top[1].logprob` is the top-2 gap at that step.

---

## Determinism

**Greedy (`temperature == 0`).** Byte-identical across runs. No RNG is
consulted. Given fixed weights, a fixed prompt, and the same KV cache state,
the token sequence is fully determined — and it is the same sequence in all
three greedy sub-cases, because they share one tie rule (lowest id wins; see
[Greedy](#greedy-temperature--0)). Adding a constraint or a penalty moves a
request between sub-cases, and that move must not change the token on a tied
row.

**Stochastic (`temperature > 0`).** Deterministic when the seed is fixed.
The PCG32 RNG is seeded from `SamplerConfig::seed_or_default()`:

- Seed provided by the caller → use as given.
- No seed → `0xA7A7` (fixed default, documented contract).

Absent-seed requests at `temperature > 0` are therefore deterministic per
model and prompt. Callers that need distinct random sequences across retries
must provide explicit differing seeds.

**Speculative decoding.** Temperature > 0 is rejected for speculative
decoding at the server layer (HTTP 400). Speculative candidates must be
greedy; the verifier uses stochastic acceptance only when both draft and
verifier distributions are sampled identically via `sampling_distribution()`,
which replicates the full pipeline (mask → penalties → softmax → filters →
renormalise) so acceptance is unbiased (Leviathan 2023, §2.3).

---

## Key types and functions

| Symbol | Crate / module | Purpose |
|---|---|---|
| `SamplerConfig` | `rmlx-models::sampler` | Temperature, top-p, top-k, min-p, seed, top_logprobs_k |
| `PenaltyConfig` | `rmlx-models::sampler` | rep_penalty, presence_penalty, frequency_penalty, logit_bias |
| `Pcg32` | `rmlx-models::sampler` | Per-request RNG |
| `sample_token_array` | `rmlx-models::sampler` | Host categorical sampler entry point |
| `argmax_with_penalties` | `rmlx-models::sampler` | Greedy + penalties (host argmax) |
| `apply_mask_argmax` | `rmlx-models::sampler` | Greedy + constraint mask (GPU argmax) |
| `apply_penalties` | `rmlx-models::sampler` | Logit processors: bias → rep → presence → freq |
| `sampling_distribution` | `rmlx-models::sampler` | Build post-sampling probability vector (shared with speculative path) |
| `compute_top_logprobs` | `rmlx-models::sampler` | Per-step logprob capture |
| `ConstraintEngine` | `rmlx-models::constraint` | Trait: per-step allow-mask + state advance |
| `ThinkSplitter` | `rmlx-server::engine` | Think/answer channel routing + budget enforcement |

---

## See also

- `docs/KV_CACHE.md` — KV cache layout and quantisation; how the decode loop
  manages cache growth and prefix reuse.
- `docs/TESTING.md` — golden-token tests that pin sampling output for
  thinking-budget injection and exact-hit cache paths.
- `docs/SPECULATIVE.md` — speculative decoding; how the draft and verifier
  share `sampling_distribution` for stochastic acceptance.
