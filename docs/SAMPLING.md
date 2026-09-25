# Sampling reference

Per-token sampling in rMLX: from raw model logits to the next token id.

---

## Overview

Every decode step produces a `[1, vocab]` logit row on the GPU. The sampler
turns it into one token id, returned as a `[1] I32` array. Two paths exist:

- **Fast greedy**: `temperature <= 0`, no penalties and no constraint mask.
  One GPU `argmax`, no host transfer.
- **Host path**: everything else. One GPU-to-host transfer, then Rust code
  applies the mask, logit bias, penalties, temperature, filters and an
  inverse-CDF draw.

Both return the same `[1] I32` shape, so every caller reads the token the
same way. The host path costs far more; see
[Cost of the host path](#cost-of-the-host-path).

---

## Pipeline

The host path runs these steps in order:

1. Constraint mask: forbidden ids go to `-inf`.
2. `logit_bias`: `logit[id] += bias`.
3. Repetition penalty (sign-aware, multiplicative).
4. Presence penalty.
5. Frequency penalty.
6. Temperature scale and a stable softmax.
7. Top-p, then min-p, then top-k.
8. Renormalise.
9. Inverse-CDF draw from the per-request `Pcg32`.

Steps 2–5 are `apply_penalties`, in the order of mlx-lm
`make_logits_processors`. Step 7 is `filter_top_p` → `filter_min_p` →
`filter_top_k`, in the order of mlx-lm `make_sampler`.

---

## Per-stage reference

### Greedy (temperature == 0)

`SamplerConfig::sampling_active()` is `temperature > 0.0`. When it is
`false`, the greedy path runs and no RNG is consulted. It has three cases:

| Condition | Path |
|---|---|
| No constraint, no penalties | GPU `argmax` |
| Constraint mask, no penalties | `apply_mask_argmax`: add a `0` / `-inf` bias row on the GPU, then `argmax` |
| Penalties, with or without a constraint | `argmax_with_penalties`: read back, mask, penalties, host argmax |

#### Tie-break contract

**Selection and filtering resolve an exact tie to the lowest token id, on the
host and on the device.** MLX's `argmax` reduces with a strict `>`, so an equal
value never displaces an earlier index. `host_argmax` mirrors it. Tests pin
three consequences:

- equal logits → lowest id;
- a `NaN` never displaces a real maximum;
- an all-`-inf` row yields id 0.

The three greedy cases must agree, because setting a constraint or a penalty
moves a request between them. Exact ties are common: BF16 logits carry 8
mantissa bits, so many adjacent values in a softmax row are equal.
`mlx_argmax_breaks_ties_to_lowest_index_gpu` pins the device half on Metal and
goes red when the rule is inverted.

The rule covers every site that selects or filters a token:

| Site | Rule |
|---|---|
| `argmax` / `apply_mask_argmax` (device) | equal logits → lowest id |
| `argmax_with_penalties` (host greedy) | mirrors it through `host_argmax` |
| `filter_top_k` | equal probabilities → lowest ids survive, so `top_k = 1` is the argmax on tied rows too |
| `filter_top_p` | equal probabilities → lowest ids survive the nucleus |
| `compute_top_logprobs` | equal logits → ascending id, so rank 0 is the token `argmax` picks |

It does not cover the inverse-CDF draw. A draw from a tied distribution is
meant to be random; forcing the lowest id would bias `temperature > 0`.

Neither filter uses a float comparator. A comparator that folds a `NaN` pair
to `Equal` is intransitive, and `sort_unstable_by` may panic on it mid-decode.
Both filters order integers under `Ord` instead:

- `filter_top_k` runs `select_nth_unstable` over packed `u64` keys: the
  inverted total-order bits of the probability above the token id.
- `filter_top_p` needs the ascending order for its cumulative sum. It sorts
  the values alone, then applies the id rule to the one tied group the cut
  lands in.

The keys use the IEEE total-order flip, not the raw bit pattern, so they order
every `f32`, including negatives and `-0.0`.

MLX seeds its CPU reduction with element 0 and its Metal reduction with
`-inf`. So a `NaN` at index 0 returns 0 on CPU and the first real maximum on
Metal. `host_argmax` follows Metal, the production stream. Three `#[ignore]`
`Device::Gpu` tests re-run the contract on Metal, one per consequence. The
all-`-inf` CPU test cannot fail, because MLX's CPU backend returns 0 by
construction; only its Metal mirror tests the claim.

### Rows the sampler refuses

Two inputs return `Err`, which the decode loop propagates.

**An all-`false` constraint mask.** No token satisfies the grammar. The
engine state is persistent, so returning a token would repeat one arbitrary
id for the rest of the generation. `apply_mask_argmax`,
`argmax_with_penalties` and `sampling_distribution` all refuse it. No HTTP
request is known to reach it; the guard defends against a constraint-engine
defect.

**A non-finite logit row on the sampling path.** `softmax_scaled` errors when
the exponentials do not sum to a finite value, which happens when a logit is
`NaN` or `+inf`. Sampling such a row would return the same last nonzero id
every step, whatever the seed.

**Greedy does not refuse a `NaN` row.** It mirrors the device reduction, which
skips `NaN` and returns the largest real logit. The pure-GPU `argmax` cannot
refuse anything without an extra reduction per token. A test pins the
asymmetry.

### Temperature scaling and softmax

```
scaled[i] = logit[i] * (1 / temperature)
prob[i]   = exp(scaled[i] - max(scaled)) / sum_j exp(scaled[j] - max(scaled))
```

A position masked to `-inf` gets probability exactly 0.

### Top-p (nucleus) truncation

Mirrors mlx-lm `apply_top_p`. Sort the probabilities ascending and take an
inclusive cumulative sum. A token survives if its cumulative probability is
strictly greater than `1 - top_p`. It is a no-op unless `0 < top_p < 1`.

The number of survivors does not depend on the tie rule: every member of a
tied group adds the same value to the sum. Only which members drop depends on
the order inside a group, and the highest ids drop.

### Min-p truncation

Mirrors mlx-lm `apply_min_p`. A token drops when its probability is below
`max(probs) * min_p`. The top token always survives. It is a no-op when
`min_p <= 0`.

### Top-k truncation

Mirrors mlx-lm `apply_top_k`. Keep the `k` most probable tokens and zero the
rest. It is a no-op when `k == 0` or `k >= vocab`. mlx-lm's `argpartition`
leaves the tied order unspecified; rMLX pins it to ascending id.

### Renormalisation

After the filters, one pass rescales the survivors to sum to 1.

### Repetition penalty

Mirrors mlx-lm `make_repetition_penalty`, over the last 20 generated tokens:

```
if logit[id] < 0:  logit[id] *= rep_penalty
else:              logit[id] /= rep_penalty
```

It applies once per unique id in the window. The identity is `1.0`. The
window is 20 tokens (`context_size=20` in mlx-lm), not OpenAI's full context:
a repeat older than 20 tokens is not penalised.

### Presence penalty

Subtracts `presence_penalty` once for each id in the 20-token window. The
identity is `0.0`.

### Frequency penalty

Subtracts `frequency_penalty * count(id, window)`. The identity is `0.0`.

### Logit bias

Applied first in `apply_penalties`: `logit[id] += bias` for each pair. An id
outside the vocabulary is skipped. The HTTP routes refuse a non-finite bias
with 400 (`parse_logit_bias`), so a request cannot hard-ban a token with
`-inf`.

### Hot-path discriminant

`PenaltyConfig::penalties_active()` is `false` when all four fields hold their
identity values. When it is `false` and the temperature is 0, the step does no
host work and no transfer.

### Inverse-CDF sample

```
target = rng.next_f32() * sum(probs)
chosen = first i where cumsum(probs[0..=i]) > target
```

On float drift it returns the last nonzero index.

### RNG — PCG32

A PCG32 (PCG-XSH-RR) is built per request from
`SamplerConfig::seed_or_default()`, which is `seed.unwrap_or(0xA7A7)`. One
instance runs through every decode step, so the stream is contiguous. There is
no `rand` dependency.

---

## Defaults and resolution order

`resolve_sampling_params` (`crates/rmlx-server/src/openai/errors.rs`) resolves
each field per request:

| Field | Order | Hard default |
|---|---|---|
| `temperature` | request, `--default-temperature`, `generation_config.json` | `1.0` |
| `top_p` | request, `generation_config.json` | `1.0` |
| `top_k` | request, `generation_config.json` | `0` (off) |
| `repetition_penalty` | request, `generation_config.json` | `1.0` |
| `min_p`, `frequency_penalty`, `presence_penalty` | request | `0.0` |
| `logit_bias`, `seed` | request | none |

`top_logprobs` is resolved by the route from `logprobs` and `top_logprobs`,
capped at 20. So a request that omits the sampling fields samples at
temperature 1.0 unless the operator sets `--default-temperature 0`. It also
takes any `top_p` and `top_k` the snapshot's `generation_config.json`
carries.

---

## Cost of the host path

The host path is not a cheap variant of greedy. Per token it moves a
`vocab`-wide row to the host and does `O(vocab)` host work. It also gives up
pipelining: greedy returns a lazy GPU argmax and dispatches the next step
while the GPU is busy, but the host path cannot dispatch until it has chosen
a token. The served default is a host-path shape (see the table above).
`top_p` and `top_k` are the dearest stages, because they order the whole
vocabulary.

### The instrument

The shared decode loop emits a `sampler_profile` event once per generation,
over the steps that took the host path. A greedy run emits nothing.

| Field | Meaning |
|---|---|
| `sync_per_step_ms` | Wait for the forward before the row can be read. GPU latency, not sampler cost. |
| `sample_per_step_ms` | Readback plus all host work: mask, penalties, softmax, filters, draw and logprob capture. |
| `step_per_step_ms` | The whole step, host-path steps only. |
| `sample_share_pct` | `sample / step`. |

`rmlx bench --temperature`, `--top-p`, `--top-k` and `--repetition-penalty`
drive the path. `scripts/perf_canary.sh` is greedy-only and cannot see it.

`sample_share_pct` is not a bound in either direction:

- It understates, because it omits the lost pipelining. On the mask-only
  greedy path, `apply_mask_argmax` only schedules the GPU add and argmax.
  They run at the next step's `eval` and are billed to `sync`.
- It overstates on a busy host. `sample` is host CPU while `sync` is mostly
  GPU time, so CPU contention stretches only the numerator.

The end-to-end cost is a decode-rate comparison against a greedy control at
the same shape. The share is the part attributable to host work. `sample`
does not grow with context and `step` does, so the share falls with context.
It falls little on sliding-window attention, whose step time barely grows.

### Host selection is not bit-identical to the GPU argmax

A near-zero temperature is not an argmax, and is not an oracle for greedy.
Two reasons, both pinned by unit tests:

- `exp` in f32 underflows to zero below about `-104`. A logit keeps nonzero
  probability when it is within `104 * temperature` of the maximum: 0.0104 at
  `temperature = 1e-4`. Inside that window the draw is random.
- At an exact tie the softmax is uniform over the tied ids, and the draw
  picks among them by the RNG. The device `argmax` takes the lowest id.

So a stream at `temperature = 1e-4` can diverge from the greedy stream at the
first tied row, and both paths are correct. A matching window shows only that
no tie fell inside it. `top_logprobs: 2` shows the top-2 gap at a step (see
Logprobs).

A future fused GPU sampler therefore needs two gates. On the greedy path:
exact token identity against the device `argmax`, lowest-id ties included. On
the sampling path: exact identity against the CPU path given the same `Pcg32`
draws. A GPU RNG must reproduce `Pcg32` bit for bit, or the kernel ships
behind a dispatch policy with the CPU path kept as the oracle.

## Special tokens

**EOS.** Each architecture reads `eos_token_id` from `config.json` through
`eos_token_ids()`. An integer or an array both become `Vec<u32>`. The decode
loop stops with `finish_reason = "stop"` on the first match.

**BOS.** Used only when building a prompt.

**PAD.** Not used; a decode stream is never padded.

**Think markers.** `<think>` and `</think>` pass through the decode loop as
ordinary tokens. The server's `ThinkSplitter` (`engine/think.rs`) routes the
decoded pieces to `reasoning_content` or `content`; the sampler does not look
at them. The route resolves the `</think>` token id at request setup, and the
decode loop uses it only for budget enforcement.

---

## Thinking-budget enforcement

Whether the assistant turn starts inside a `<think>` block depends on the
checkpoint's chat template, not the architecture. A Qwen3-family template may
leave `<think>\n` open, prefill a closed `<think>\n\n</think>\n\n`, or prefill
nothing.

The server reads the initial channel off the rendered prompt.
`engine::think::prompt_leaves_think_open` is `true` when the last
thinking-start delimiter comes after the last thinking-end delimiter. The
result travels as `GenerationRequest::prompt_think_open`. It sets the
`ThinkSplitter`'s initial state and seeds the constraint engine's
`is_thinking` handle. A splitter started open against a closed block would
never see a `</think>`: all output would go to `reasoning_content`, and a
`json_schema` constraint that waits for the answer would never engage.

`ThinkSplitter` counts the pieces routed to the thinking channel. When
`thinking_budget` is set and the count passes it, `account_thinking_piece()`
latches `force_close`. The server's step callback then returns the `</think>`
id. The decode loop:

1. Discards the pipelined successor token for this step.
2. Builds a `[1] I32` array from the forced id.
3. Feeds it as the next decode input and as the pending output token.
4. Clears `forced_next`, so the injection fires once.

The model continues from `</think>` with answer tokens. The forced token
carries no logprobs. With no `thinking_budget`, `account_thinking_piece()`
returns at once.

---

## Constrained decoding

`response_format: json_object` and `response_format: json_schema` build a
`ConstraintEngine` for the request. Each step calls:

- `step_mask(vocab_size) -> &[bool]`: `true` means the token may be chosen.
- `advance(token_id)`: moves the grammar state past the chosen token.

The mask is applied before penalties and softmax, so it composes with every
other parameter. Without `response_format` no engine is built. At
`temperature == 0` the mask goes through `apply_mask_argmax` on the GPU; above
it, through the host path. While `wants_mask()` is `false`, for example while
a `<think>` block is open, the step skips `step_mask()`.

`constraint_json::JsonObjectConstraint` (`json_object`) and
`constraint_json::SchemaConstraint` (`json_schema`) implement the trait.
Construction decodes every vocabulary token once. Each step's mask is an
`O(vocab)` sweep: a scratch grammar is reset to the current state and fed the
token's bytes. Both engines keep the immutable part of the grammar cheap to
copy: `Copy` frames in the object engine, and the parsed schema behind `Arc`
in the schema engine.

EOS is masked out mid-JSON, since special tokens decode to no byte the
grammar accepts. At a terminal state the engine sets the EOS ids to `true`, so
the decode loop's EOS stop can fire.

Speculative decoding refuses a constrained request.

### Whitespace is bounded, on purpose

Withholding EOS until the value is complete makes an over-permissive grammar
dangerous. A byte the grammar accepts without progress is a cycle. At
`temperature == 0` the decoder can sit in it until `max_tokens` and return 200
with nothing usable. Both engines cap a run of insignificant whitespace at
`constraint_json::MAX_INSIGNIFICANT_WS_RUN` (64) bytes; any content or
structural byte resets the count. No JSON value becomes unreachable; only
indentation past the cap is clipped. llama.cpp bounds its JSON-schema `space`
rule for the same reason.

Both engines refuse raw C0 control bytes inside any string, keys included
(RFC 8259 requires them escaped). The schema engine also refuses whitespace
before the root value. The `json_object` engine accepts it there, under the
same cap.

`make schema-constraint-canary` (`scripts/schema_constraint_canary.sh`) is the
real-model check, on Bonsai and gemma-4-e2b with two probes each. Its PASS and
FAIL rule is fixed at the top of the script. `EXPECT=baseline` asserts a
per-cell table, so a harness too weak to see a defect fails too. Each probe
runs its own server process. A cell with missing evidence reports HARNESS
ERROR and fails both arms. In the cell whose property name holds a space,
the grammar alone would force the whitespace cycle on both models. The
single-word cell is a no-regression check.

### Non-enforcement is reported

Both engines start masking only once the model emits something the grammar
can latch onto: a value-start byte for a container root, or the first token
after reasoning for a scalar root. `ConstraintEngine::engaged()` reports
whether that happened. `engaged_handle()` gives the route a handle that
outlives the move of the engine into the decode thread. A generation that
ends unengaged was never constrained.

- **Non-streaming refuses.** No byte has reached the client yet, so the route
  returns HTTP 502 `constraint_not_engaged`.
- **Streaming cannot refuse.** The deltas are already sent. It logs the warn
  and completes.
- Only `response_format` is checked. A forced `tool_choice` also builds a
  constraint, but it has a text-parsing fallback.

Either way the decode loop emits a `warn!` carrying the route's `request_id`.
The span is carried across `spawn_blocking` explicitly.

---

## Logprobs

Capture is off by default (`top_logprobs_k == 0`). `logprobs: true` with
`top_logprobs: k` (at most 20) makes the decode loop call
`compute_top_logprobs` after each step. It reads the row back and computes a
stable log-softmax over the raw logits:

```
lse        = max(logits) + ln(sum(exp(logits - max)))
logprob(i) = logits[i] - lse
```

The top `k` come from a partial selection, `O(vocab × k)`. Logprobs come from
the raw logits, before temperature, penalties and filters, as in OpenAI's
contract. With `k == 0` no log-softmax runs and nothing is allocated.

`TokenLogprobs` carries:

- `token_id`: the chosen id.
- `token_logprob`: `ln P(token_id)` under the raw softmax.
- `top`: the `k` most likely `(token_id, logprob)` pairs, descending. It may
  not include the chosen token.

Equal logits rank by ascending id, so rank 0 is the token the device `argmax`
selects. `top[0].logprob - top[1].logprob` is the top-2 gap at that step.

---

## Determinism

**Greedy.** No RNG is consulted. With fixed weights, prompt and cache state
the token sequence is fixed, and it is the same in all three greedy cases,
because they share the tie rule.

**Sampling.** Deterministic for a fixed seed. With no seed the RNG uses
`0xA7A7`, so an unseeded request at `temperature > 0` repeats per model and
prompt. A caller who wants different draws across retries sends different
seeds.

**Speculative decoding.** Every speculative arm honours `temperature > 0`,
and the emitted distribution is the verifier's own. At `temperature == 0`
every arm stays on the argmax path.

- **Two-model.** The drafter samples, so the round accepts each proposal with
  probability `min(1, p(x)/q(x))`. On the first rejection it draws the
  correction from the residual `normalize((p − q)+)`, the rule of Leviathan
  et al. (2023). `p` and `q` both come from `sampling_distribution()`, which
  runs the full pipeline, so the two match.
- **Sidecar drafters** (MTP sidecar, Gemma4 MTP assistant, DFlash, DFlash 2,
  EAGLE-3). These propose an argmax, a point mass. With a point-mass proposal
  the same rule needs no drafter distribution: draw the verifier's token from
  `sampling_distribution()` at each verified position, accept the prefix the
  proposals agree with, and emit the verifier's draw either way.

Consequences:

- **A sampled request costs more than a greedy one on a sidecar arm.** Greedy
  is one device reduction over the verified block. A draw is a host round
  trip and a full-vocabulary softmax per position, rejected tail included.
  EAGLE-3's restricted read-back covers only the drafter's vocabulary and
  cannot be normalised, so sampled requests take the full-vocabulary branch.
- **Penalties and `logit_bias` reach no speculative arm.** No round loop
  rebuilds the penalty window over each drafted prefix. A request that sets
  one gets a `warn` naming the fields it did not get.
- **No speculative arm captures logprobs.**

---

## Key types and functions

| Symbol | Crate / module | Purpose |
|---|---|---|
| `SamplerConfig` | `rmlx-models::sampler` | Temperature, top-p, top-k, min-p, seed, `top_logprobs_k` |
| `PenaltyConfig` | `rmlx-models::sampler` | Repetition, presence and frequency penalties, `logit_bias` |
| `Pcg32` | `rmlx-models::sampler` | Per-request RNG |
| `sample_token_array` | `rmlx-models::sampler` | Host sampling entry point |
| `argmax_with_penalties` | `rmlx-models::sampler` | Greedy with penalties (host argmax) |
| `apply_mask_argmax` | `rmlx-models::sampler` | Greedy with a constraint mask (GPU argmax) |
| `apply_penalties` | `rmlx-models::sampler` | Bias → repetition → presence → frequency |
| `sampling_distribution` | `rmlx-models::sampler` | The post-filter probability vector, shared with the speculative path |
| `compute_top_logprobs` | `rmlx-models::sampler` | Per-step logprob capture |
| `ConstraintEngine` | `rmlx-models::constraint` | Per-step allow-mask and state advance |
| `ThinkSplitter` | `rmlx-server::engine::think` | Think/answer routing and budget enforcement |
| `resolve_sampling_params` | `rmlx-server::openai::errors` | Per-request defaults and resolution order |

---

## See also

- `docs/PROMPT_CACHE.md`: prefix reuse across requests.
- `docs/SPECULATIVE.md`: the speculative drafters and the round loop.
