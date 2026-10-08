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
healthy reference whose margins mean anything. So every run also reads how
much of each arm repeats at a short period. It reads the whole stream and each
of the `TAIL_WINDOWS` (4) tail cuts, at every period up to `MAX_CYCLE_PERIOD`
(64) that leaves `MIN_CYCLE_SAMPLES` (32) comparisons.

`MAX_CYCLE_FRACTION` (0.20) is the ceiling, and the only threshold. The two
arms are read differently:

- The reference arm is read as it is. Above the ceiling it is unjudgeable.
- The speculative arm is read without the repeats the reference arm wrote
  itself (`repeats_beyond`). Above the ceiling it is refused.

What "wrote itself" means, exactly. At one period, the positions that repeat
the token one period before them are read as runs. A run of consecutive
repeats, together with the tokens of one period before it, is one stretch of
the arm: a phrase and the place where the arm said it before. The run is not
counted when the reference arm holds that whole stretch, token for token, in
the tail cut the stretch starts in. A stretch that starts in the last quarter
of the speculative arm is looked for in the last quarter of the reference
arm; one that starts in the first quarter is looked for in all of it. A run
is excused whole or not at all. In one reading, a repeat of the reference arm
excuses one repeat of the speculative arm, once: a second copy of the same
stretch needs a second place in the reference arm. A run is read inside the
window of the reading, so a loop that begins before a cut is cut there. The
reading is still the strongest one over every window and period, taken after
the runs are excused.

A run takes the first free place: the first one from the cut on that holds
its stretch and whose repeats no run has taken. A later place is not tried
first, and a place is not kept for a longer run. So the verdict can depend on
the order of the runs. The measured pair is pinned
(`a_run_takes_the_first_free_place_so_the_order_of_the_runs_can_decide`). A
reference arm holds a burst of 9 tokens and then a burst of 5 tokens in its
last window, 12 of 63. A speculative arm that holds the same two bursts and a
6-token burst of its own reads 17 of 63 raw. In the reference order it is
agreed, with 5 repeats counted. With the two bursts exchanged, the burst of 5
takes the start of the 9-token place, the burst of 9 fits no free place, and
the arm is refused at 13 of 63. This was found on burst arms only.

The position bound exists because a judgeable reference arm can hold a long
loop. Over the whole reference arm a 57-token loop at period 8 reads 49 of
248, under the ceiling. The same loop at the tail of the speculative arm
reads 49 of 56. A stretch matched anywhere would excuse it; the last quarter
of the reference arm does not hold it
(`a_loop_the_reference_arm_holds_somewhere_else_is_refused`).

The reason is a measurement. An answer about one subject repeats that subject,
in both arms. A healthy arm of a correct engine reads 9 of 39 at period 25 in
its last window, where a period-8 loop at 58% raggedness can read 0.20. Five
of the nine are one phrase, and the reference arm wrote the same phrase after
the same 25 tokens. Without them the arm reads 4 of 39.
`a_healthy_arm_that_repeats_no_more_than_its_reference_is_not_a_collapse`
holds both arms as a fixture.

What the tests hold, and against what:

- Twenty-one constructed arms against judgeable reference arms are each
  refused: eight whose reference arm reads near the ceiling, seven whose
  reference arm holds the same loop somewhere else, and six that hold a burst
  or a loop of the reference arm twice, in full or in part.
- An exact loop at the end of the measured arm is refused past 13 tokens at
  period 1 and past 19 at period 8. The test asserts that each of those
  verdicts is the one a fixed ceiling gives on the raw reading of the last
  window at the loop's period. At period 25 the loop is refused when it goes
  one token past what the reference arm wrote, at 28 tokens. A fixed ceiling
  refuses the measured arm itself, so there is no raw verdict to compare with
  there.
- A looping arm against a healthy reference arm, walked from 0% to 100%
  raggedness, is refused under 58%, against synthetic prose and against two
  reference arms that read near the ceiling. Each verdict of these sweeps is
  asserted equal to a fixed ceiling on the raw reading.
- Two arms in one period-8 loop, walked the same way, are refused under 60%,
  with the same assertion on the two raw readings. Past that the arms are
  more noise than loop, and nothing takes over.

A healthy arm can still be refused. Only a run the reference arm wrote token
for token is excused. A healthy arm that repeats its subject above the
ceiling in other words, or by single-token coincidences alone, is refused:
the measured arm against a reference arm with one token of that stretch
changed is refused
(`a_healthy_arm_above_the_ceiling_is_refused_when_the_reference_did_not_write_its_phrase`).
Position is a cause too. A healthy arm whose reference arm said the phrase in
an earlier tail cut is refused. The measured pair has 7 tokens of margin: its
stretch starts at token 223, the reference arm holds it at token 199, and the
last quarter starts at token 192. With the reference arm's phrase 7 tokens
earlier the pair is agreed; at 8 tokens earlier it is refused
(`the_measured_pair_is_agreed_with_seven_tokens_of_position_margin`).
The declared blind spot is the converse: a loop that the reference arm also
wrote in full, in the same tail cut. Its size is bounded. In one reading, the
repeats left out of the speculative arm are at most the raw repeats of the
reference arm from the same cut at the same period, and for a judgeable
reference arm of the same length those are at most the ceiling. So the control
admits a raw reading of at most twice the ceiling in one window. The edge is
pinned: a reference arm holds at most 19 tokens of a period-8 loop in its last
window (11 of 56); a speculative arm that holds that loop and a second one of
its own, up to the ceiling again, reads 22 of 56 raw and is agreed; one more
token of its own loop and it is refused
(`the_control_admits_at_most_the_ceiling_twice_in_one_window`). One burst in
the reference arm does not excuse two copies of itself; it excuses at most its
own 12 repeats. A speculative arm that holds the reference arm's 13-token
burst twice and one of its own reads 36 of 63 and is refused
(`one_burst_in_the_reference_arm_excuses_one_burst_and_no_more`). A place is
not one run: two 7-token bursts take 6 repeats each of that 13-token burst,
and both are excused. The same test pins that row at 0, and pins the rows
where the second burst needs one repeat more than is left. For two arms of
different lengths the windows differ in length, and the bound is on the count
of repeats, not on the fraction.

The control reads only prose. Healthy structured output, such as a markdown
table, overlaps ragged loops on this measure, which is why the prompts ask for
prose.

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
hard gate on it and bounds `MAX_CYCLE_FRACTION` from below, but no synthetic
healthy stream reaches the ceiling, and a real one does. So the excused
repeat rests on one measured pair.
