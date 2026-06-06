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
| No constraint, penalties active | `argmax_with_penalties` — GPU→host, penalties, host argmax |

`apply_mask_argmax` builds a F32 bias buffer on the host (0.0 for allowed,
`-inf` for forbidden), wraps it as a `[1, vocab]` MLX array, adds it to the
logits (GPU op, promotes BF16 to F32 automatically), then calls `argmax`.
Overhead versus unconstrained: approximately 0.05 ms host-side bias fill for
a 262K-token vocabulary.

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

Keep exactly the `k` highest-probability tokens; set the rest to 0.0. Ties
are broken by index rank (descending probability sort, argpartition
semantics). No-op when `k == 0` or `k >= vocab`.

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

Qwen3-family models begin their assistant turn with a `<think>` block
prefilled by the chat template. The decode loop emits reasoning tokens until
the model produces `</think>`, then switches to the answer channel.

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

**JSON engine.** `constraint_json::JsonObjectConstraint` /
`JsonSchemaConstraint` implement `ConstraintEngine`. Construction decodes
every token in the vocabulary once (approximately 600 ms for a 152K-token
vocabulary — one-time cost, hidden behind TTFT). Per-step mask build is
O(vocab) byte-set membership, approximately 1 ms on Qwen3.6.

EOS tokens are naturally masked out mid-JSON (special tokens decode to empty
or zero first byte; the state machine never allows byte value 0). At
terminal states (valid JSON complete) EOS ids are explicitly forced to `true`
in the mask so the decode loop's EOS stop predicate can fire normally.

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

---

## Determinism

**Greedy (`temperature == 0`).** Byte-identical across runs. No RNG is
consulted. Given fixed weights, a fixed prompt, and the same KV cache state,
the token sequence is fully determined.

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
