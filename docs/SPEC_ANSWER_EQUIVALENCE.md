# Answer equivalence for speculative decoding

`crates/rmlx-models/tests/spec_greedy_equivalence.rs`. Greedy speculative
decoding emits the verifier's own argmax at every position. So at temperature
0 a speculative run and a plain run of the same verifier are two ways of
computing one answer. This gate checks that a round loop keeps it that way.

It is not bit-identity. The verify pass scores a whole block in one forward,
where plain decode steps one token at a time: a different reduction order. No
correct speculative arm is byte-identical to plain greedy in general.

**This is not the bench's arm-equality refusal.** `scripts/spec_bench.sh`
digests each measured completion and refuses to file a speculative row whose
answer differs from the plain arm's at all. That decides whether a throughput
row is worth recording, and it is right to be absolute. This gate answers what
byte equality cannot: whether a difference is a near-tie the arithmetic
permits or a defect in the round loop.

The thresholds below are set from runs of the gate against the shipped
engine and against deliberately broken ones. `scripts/spec_broken_engine.sh`
applies each broken engine by name; take the readings with it.

## What it runs

Each pair runs every prompt in `PROMPTS`:

| verifier | drafter | round loop | rollback |
|---|---|---|---|
| `gemma-4-e2b-it-mxfp8` | `gemma-4-E2B-it-assistant-bf16` | shared-K/V assistant | KV truncation, SWA ring included |
| `Qwen3.8-27B-mxfp8` | `Qwen3.8-27B-MTP-mxfp8` | MTP sidecar | KV truncation + recurrent snapshot/replay |
| `Qwen3.8-27B-4bit` | `Qwen3.8-27B-MTP-4bit` | MTP sidecar | KV truncation + recurrent snapshot/replay |
| `Qwen3.8-27B-4bit` | `Qwen3.8-27B-DFlash2` | DFlash 2 block drafter | KV truncation + recurrent snapshot/replay |
| `Qwen3.6-35B-A3B-8bit` | `Qwen3.6-35B-A3B-DFlash` | DFlash 1, adaptive block | KV truncation + recurrent snapshot/replay |
| `Qwen3.6-35B-A3B-8bit` | `specdrift-qwen3.6-35b-a3b-eagle3` | EAGLE-3 | KV truncation + recurrent snapshot/replay |
| `Qwen3.8-27B-mxfp8` | `ornith-1.0-9b-mxfp8-mlx` | two full models, greedy | both models' KV + recurrent state |

Both halves resolve by slug from `RMLX_O_MODELS_ROOT`, so every pair runs
under `make gpu-test` wherever its snapshots are. `RMLX_KV_TEST_MODEL`
overrides the verifier only when it names an architecture the pair covers.
`RMLX_DRAFT_TEST_MODEL` overrides a drafter only where
the models root does not hold its slug. One variable cannot name several
drafters, which is why the slug outranks it.

Before loading, a pair refuses a drafter of the wrong kind, a drafter
quantized unlike its verifier, and a drafter declaring another backbone width.
The two-model pair cannot be pinned by kind, since every full model is a
possible draft. It is exempt from the format check, and the engine's
`vocab_pairing` compares the two tokenizers id by id instead.

A pair runs at a named block, or at the block a request that names none is
served (`default_block_for` of the drafter's declared depth). The gate asserts
the loop ran that block. The block decides how many positions the verify
forward scores, how many the acceptance walk reads and how long a rejected
tail the rollback drops. So two blocks of one loop are two runs, and some
loops carry a second pair at another block.

The prompts all ask for continuous prose. A tokenizer that declares `<think>`
gets an empty reasoning block, and turn markers are read from the tokenizer's
added tokens, so a pair is always served inside its own template.

## Coverage

The engine has seven drafter paths. Six have a pair here. The seventh cannot
have one: `spec_generate_stochastic_cached` is the two-model Leviathan
acceptance rule, which runs only at `temperature > 0`. There it has no
temperature-0 arm to compare, and neither arm is a function of the model
alone. `crates/rmlx-models/tests/two_model_stochastic.rs` gates it on a
different property: one seed reproduces one sequence, while a second seed and
temperature 0 do not.

| round loop | test | block |
|---|---|---|
| Gemma4 shared-K/V assistant | `the_assistant_round_loop_reproduces_plain_greedy` | served |
| MTP sidecar | `the_recurrent_round_loop_reproduces_plain_greedy` (mxfp8 verifier) | served |
| MTP sidecar | `the_recurrent_round_loop_reproduces_plain_greedy_at_the_declared_block` (4-bit verifier) | served |
| MTP sidecar | `the_recurrent_round_loop_reproduces_plain_greedy_past_the_declared_block` (4-bit verifier) | 8 |
| DFlash 2 block | `the_block_round_loop_reproduces_plain_greedy` | served |
| DFlash 2 block | `the_block_round_loop_reproduces_plain_greedy_at_the_declared_block` | 8 |
| DFlash 1 adaptive block | `the_adaptive_round_loop_reproduces_plain_greedy` | served |
| DFlash 1 adaptive block | `the_adaptive_round_loop_reproduces_plain_greedy_over_its_whole_schedule` | 16 |
| EAGLE-3 | `the_restricted_vocab_round_loop_reproduces_plain_greedy` | served |
| two full models, greedy | `the_two_model_round_loop_reproduces_plain_greedy` | served |
| two full models, stochastic | none, and none is possible | `two_model_stochastic.rs` |

The property is not transitive across loops: each has its own rollback and
its own acceptance walk. Nor is it transitive across blocks. The MTP head
chains on its own output hidden, so a request may name a block past the depth
the checkpoint declares. The DFlash 1 schedule sets each round's block from
recent accept rates; at its declared 16 a run truncates an 8-wide append and
follows it with a 4- or 6-wide one, a sequence the served block never reaches.

The per-round event stream of the six pairs at the served block is pinned
separately, in `crates/rmlx-models/tests/fixtures/spec_round_baseline/`, and
compared by `scripts/spec_round_stream_compare.py`. This gate reads the
answer; that comparison reads which round moved.

## The oracle: where a correct pair diverges

A reduction-order difference is a relative perturbation of order `1e-3` on a
logit. It can flip a decision the verifier was already nearly indifferent
about, and nothing else. So the gate reads the verifier's top-two logprob
margin at the position where the two arms **first** differ. It returns the rank
of that margin in the same arm's own margin distribution
(`divergence_confidence`). Both arms saw the same context up to that
position, so this judges the pair, not an arm. It needs no per-prompt
calibration: it is a rank, not a number of nats.

`MAX_DIVERGENCE_CONFIDENCE` is 0.12. A first divergence ranked above it is
refused: the round loop fed the verifier a different state, not the same state
in a different order. The value sits between the worst correct reading and
the lowest broken one above it.

No broken engine is refused on every prompt by this oracle alone. So the gate
runs **every** prompt, not one, and the repetition control runs beside it. A
pair with no prompt it could judge fails: a run that asserted nothing is not a
pass.

## Why agreement cannot be thresholded

The obvious oracle, how much of one answer the two arms share, cannot be
thresholded. How much two correct arms share is decided by where their first
near-tie lands, and by nothing else. An exact tie early in an answer sends two
correct arms into well-formed, different prose. Broken engines read in the
same range. Nothing bounds the correct minimum from below, so no floor fits in
the gap.

The gate prints `lcs_ratio`, the weakest tail window, the first divergence,
the margin there and both arms' decoded text on every run. They are evidence,
not a gate. `WORST_CORRECT_TAIL_AGREEMENT` records the worst tail reading a
correct pair reached; the gate does not apply it. The ragged-loop sweep reads
it to show that the two populations overlap.

## The second oracle: the repetition control

The first oracle has nothing to read when both arms are degenerate: there is no
healthy reference whose margins mean anything. So every run also checks that
neither arm repeats at a short period across more than `MAX_CYCLE_FRACTION`
(0.20) of its tokens. It checks the whole stream and each of the
`TAIL_WINDOWS` (4) tail cuts, at every period up to `MAX_CYCLE_PERIOD` (64)
that leaves `MIN_CYCLE_SAMPLES` (32) comparisons.

The ceiling holds only for prose. Healthy structured output, such as a
markdown table, overlaps ragged loops on this measure, which is why the
prompts ask for prose. The upper side is swept, not sampled: two arms in one
period-8 loop are walked from 0% to 100% raggedness, and the control refuses
every pair up to 60%. Past that the arms are more noise than loop, and nothing
takes over.

Its declared blind spot is its own sample floor. The narrowest window is
`len / TAIL_WINDOWS`, so at the 256-token budget it can evidence no period
above 32. A collapse confined to the last quarter at a longer period is read
only over a wider window, where healthy text dilutes it.
`a_cycle_confined_to_a_window_too_narrow_to_read_it_is_a_declared_blind_spot`
pins that from both sides.

Which arm collapsed decides the verdict. A degenerate speculative arm against
a healthy reference is refused. A degenerate reference arm says the prompt
did not come back as prose the control can read: it is reported as
unjudgeable, not failed. Plain greedy is the control.

## Which arm is short

`MIN_ANSWER_TOKENS` is 160, read against both arms but not symmetrically:

- Both arms under the floor: the prompt produced no answer. Unjudgeable.
- The reference under the floor: plain greedy is the control, so the prompt is
  unjudgeable. The one exception is a speculative arm that reproduced the whole
  reference answer and ran on: the loop did not stop where the verifier
  stopped, and that is refused.
- The speculative arm under the floor while the reference ran on: the round
  loop cut its own run. Refused.

`MIN_LENGTH_RATIO` (0.60) refuses two arms whose lengths differ that much;
`lcs_ratio` divides by the shorter arm and would otherwise score a truncated
prefix as 1.0.

## A declared boundary: EAGLE-3's restricted vocabulary

EAGLE-3's verify pass takes the argmax over the drafter's reduced vocabulary
(its target ids plus the verifier's stop ids) at every position. It computes
the verifier's full-vocabulary argmax at one position only: the first the
draft missed, or the bonus when it missed none. An accepted position is
therefore the draft's token. That is the verifier's own argmax only when the
verifier's argmax is inside the reduced set.

That mirrors the upstream implementation, so it is a design boundary, not a
port defect. It is still an answer change at temperature 0.
`docs/SPECULATIVE.md` says the restricted read-back is sound at temperature 0
only; that is a claim about sampling. This section is about what remains at
temperature 0: the reduced argmax is the true argmax only on the ids the
drafter can name.

### The rule, and why the token alone will not carry it

Widening the same inexactness to the correction changes an answer at the same
kind of token, with the same margin. Nothing in the two token streams tells
the two apart. What separates them is which position the loop was at: the
restriction reaches an accepted position by design and cannot reach the
correction. So the EAGLE-3 round loop records a `DecidedBy` for each emitted
token, and the gate's `Restriction` reads it:

- A first divergence at a token the loop emitted from the drafter's own
  argmax (`RestrictedVocab`), where the reference token is one the drafter's
  vocabulary cannot name: **the boundary**. Reported as unjudgeable, not
  refused.
- A first divergence anywhere else: judged as any other pair's.

A rule keyed on the unnameable token alone would waive the widened
correction, the defect the pair exists to catch. Every run of the pair prints
`unnameable` (how many reference tokens the drafter cannot name),
`divergence_unnameable` and `divergence_decided_by`.

## Rules for changing the gate

- **A fixture must judge the pair, not an arm.** The gate's question is about
  two arms sharing a real prefix and how its oracles interact. A fixture built
  from one arm, or from independently seeded heads, cannot test that.
- **Do not characterise a population from points you chose.** Sweep the
  parameter, assert the property, and let the sweep pick the points.

One limitation stands. `Rng::prose` is an i.i.d. word model with no
autocorrelation, so it reads lower on a self-similarity measure than real
prose. `prose_clears_the_control_at_every_length_the_gate_can_hand_it` is a
hard gate on it and bounds `MAX_CYCLE_FRACTION` from below, so the synthetic
headroom overstates the true headroom. The real arms are measured too, and
the constant rests on those readings.
