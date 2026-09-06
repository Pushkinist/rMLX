# Answer equivalence for speculative decoding

`crates/rmlx-models/tests/spec_greedy_equivalence.rs`. Greedy speculative
decoding emits the verifier's own argmax at every position, so at temperature 0
a speculative run and a plain run of the same verifier are two ways of computing
one answer. Nothing in a throughput number says whether that still holds. Before
this gate the only checks were four alignment suites comparing a **48-token**
shared prefix, and one Gemma4-assistant test that asserted only that the accept
rate was high.

The gate found two defects that had been invisible to those checks for as long
as they have existed, and both are fixed. What follows is the oracle it settled
on, why the obvious oracle does not work, and the numbers every constant is set
against.

## What it runs

Six pairs, each over every prompt in the file. The assistant pair resolves both
halves by slug from `RMLX_O_MODELS_ROOT` and so runs wherever the snapshots are;
the other five have their drafter named by the operator and their verifier is the
reason (below):

| verifier | drafter | round loop | rollback |
|---|---|---|---|
| `gemma-4-e2b-it-mxfp8` | `gemma-4-E2B-it-assistant-bf16` | shared-K/V assistant | KV truncation, SWA ring included |
| `Qwen3.8-27B-mxfp8` | `Qwen3.8-27B-MTP-mxfp8` | MTP sidecar | KV truncation + recurrent snapshot/replay |
| `Qwen3.8-27B-4bit` | `Qwen3.8-27B-DFlash2` | DFlash 2 block drafter | KV truncation + recurrent snapshot/replay |
| `Qwen3.6-35B-A3B-8bit` | `Qwen3.6-35B-A3B-DFlash` | DFlash 1 block drafter | KV truncation + recurrent snapshot/replay |
| `Qwen3.6-35B-A3B-8bit` | `specdrift-qwen3.6-35b-a3b-eagle3` | EAGLE-3 | KV truncation + recurrent snapshot/replay |
| `Qwen3.8-27B-mxfp8` | `ornith-1.0-9b-mxfp8-mlx` | two full models, greedy | both models' KV + recurrent state |

`RMLX_DRAFT_TEST_MODEL` names one drafter, so a pair whose loop does not drive
the kind that snapshot declares stands down naming both rather than handing
another drafter's tensors to a loader. The two-model loop is the one kind that
cannot be pinned that way — `two_model` is an inference from the architecture
registry and every full model satisfies it — so that pair reads the vocabulary
both snapshots declare and stands down when they differ.

The DFlash 1 pair is the only one whose verify width is not fixed: its loop sets
each round's block from the accept rate of the recent ones, so a run truncates an
8-wide append and follows it with a 4- or 6-wide one. Every individual width is
exercised by some other pair; that sequence is not.

Six prompts, all asking for continuous prose. A tokenizer that declares `<think>`
gets an empty reasoning block — its own template's `enable_thinking=false` — and
turn markers are read off the tokenizer's added tokens rather than hard-coded, so
a pair is never served outside its own template.

**One pair is selected differently from the other five, and the reason is the
census.** The
assistant pair resolves both halves by slug, produces **zero** shader-validation
hits (measured, narrowed run), and runs under `make gpu-test` on any machine
holding the snapshots — about 3.5 minutes. Every other pair's verifier drives
MLX's mxfp8 or affine quantized matmul, whose `load_safe` bound is the same one
`scripts/gpu_validation_census.txt` records for the affine instantiation: a
narrowed run reports 1344 invalid **loads** (no stores) from
`mxfp8_qmm_t_splitk_bfloat16_t_gs_32_b_8_alN_false`, a kernel this repo does not
compile. The census pins one exact count per originating test, and a count taken
from a 256-token generation moves with every prompt — pinning it would make the
census brittle rather than informative. So those pairs resolve their drafter from
`RMLX_DRAFT_TEST_MODEL` only, `make gpu-test` reports them as skipped naming the
variable, and running one is:

```
RMLX_DRAFT_TEST_MODEL=<...>/mlx-community__Qwen3.8-27B-MTP-mxfp8 \
cargo test -p rmlx-models --test spec_greedy_equivalence -- --ignored --nocapture \
  the_recurrent_round_loop
```

The verifier comes from `RMLX_O_MODELS_ROOT` by slug; `RMLX_KV_TEST_MODEL` still
overrides it when it names a snapshot of an architecture the pair covers.

That is a real gap: five of the six pairs — including the case the second defect
was found in — are not in a gate that runs by default. Closing it needs either
the census analysis extended to these kernels and models, or a stable count.

## Coverage

Seven round loops exist. Six have a pair here. The seventh **cannot have one**:
`spec_generate_stochastic_cached` is the two-model loop's Leviathan acceptance
rule, which only runs at `temperature > 0`. There is no temperature-0 arm of it
to compare against plain greedy, and at any other temperature neither arm is a
function of the model alone. Its own gate is
`crates/rmlx-models/tests/two_model_stochastic.rs`, which pins that one seed
reproduces one sequence while a second seed and temperature 0 do not — a
different property, and the only one available.

| round loop | pair | how it is gated |
|---|---|---|
| Gemma4 shared-K/V assistant | `the_assistant_round_loop_reproduces_plain_greedy` | slug-resolved, runs under `make gpu-test` |
| Qwen3.5-family MTP sidecar | `the_recurrent_round_loop_reproduces_plain_greedy` | `RMLX_DRAFT_TEST_MODEL` |
| DFlash 2 block | `the_block_round_loop_reproduces_plain_greedy` | `RMLX_DRAFT_TEST_MODEL` |
| DFlash 1 block | `the_adaptive_round_loop_reproduces_plain_greedy` | `RMLX_DRAFT_TEST_MODEL` |
| EAGLE-3 | `the_restricted_vocab_round_loop_reproduces_plain_greedy` | `RMLX_DRAFT_TEST_MODEL` |
| two full models, greedy | `the_two_model_round_loop_reproduces_plain_greedy` | `RMLX_DRAFT_TEST_MODEL` |
| two full models, stochastic | none, and none is possible | `two_model_stochastic.rs`, a different property |

The property is not transitive across loops — each has its own rollback and its
own acceptance walk, which is what the gate reads — so each of the six is its own
pair rather than an inference from a neighbour.

## The divergence this gate was named for, settled

Served at temperature 0 on the code prompt, `Qwen3.8-27B-4bit` drafted by
`z-lab/Qwen3.8-27B-DFlash2` at block 8 diverged from the same verifier's
no-drafter answer near the start — "We need to respond to user:" against "We need
answer user's request:" — and stayed diverged, while the MTP sidecar on the same
verifier and prompt was byte-identical to the no-drafter arm. Read as
byte-equality against plain greedy, that says the loop changed the answer.

**Byte-equality is the oracle this file rejects**, and the measurement says why.
Plain greedy on that verifier and that prompt, at 160 tokens with the top two
logprobs per position, reproduces the no-drafter arm exactly and gives the margin
at every decision in it. The arms part at "answer" against "to", the third token;
the top-two gap there is 0.2500 nats — the **smallest** of all 160, against a
median of 10.3750. Its rank in the arm's own margins is 0.0000, against a ceiling
of 0.12. The position after it reads 2.1250 nats and rank 0.0875, also inside the
band. Whichever of the two positions the report meant, the divergence sits at the
floor of the distribution correct pairs occupy. It is a near-tie flip, not a
changed answer.

The arm that produced it also no longer exists. At that commit the checkpoint's
`DFlash2DraftModel` was read as a DFlash 1 drafter, and the DFlash 1 loader built
58 of its 81 tensors — every dynamic convolution and the whole candidate selector
went unread. That is refused outright now: the checkpoint declares itself a
DFlash 2 drafter and routes to its own loader and its own round loop, and a
loader that leaves a tensor unread refuses rather than serving an architecture
that is not the checkpoint's.

What the observation was evidence about is the DFlash 1 loop, and that now has a
pair. On `Qwen3.6-35B-A3B-8bit` drafted by `z-lab/Qwen3.6-35B-A3B-DFlash` at its
own block of 16, with the adaptive schedule running, it agrees with plain greedy
on the five prompts it judges and reproduces the sixth exactly. Its first
divergences read 0.0000 to 0.0234.

## The oracle: where a correct pair diverges

A reduction-order difference is a relative perturbation of order `1e-3` on a
logit. It can flip a decision the verifier was already nearly indifferent about
and it can flip nothing else. So the gate reads the verifier's own top-two
logprob margin at the position the two arms **first** differ, and returns where
that sits in the same arm's own margin distribution. Both arms saw the same
context up to that position, so this judges the pair rather than an arm, and it
needs no per-prompt calibration: it is a rank, not a number of nats.

Measured over six pairs and six prompts each, against the shipped engine and
against a deliberately broken one per pair — two on the block pair and two on the
restricted-vocabulary one:

| engine | percentile of the reference arm's own margins | refused on |
|---|---|---|
| assistant pair, as shipped | 0.0000 to 0.0820 | — |
| recurrent pair, as shipped | 0.0000 to 0.0234 | — |
| block pair, as shipped | 0.0000 to 0.0273 | — |
| adaptive pair, as shipped | 0.0000 to 0.0234 | — |
| restricted-vocabulary pair, as shipped | 0.0000 to 0.0703 | — |
| two-model pair, as shipped | 0.0000 to 0.0234 | — |
| assistant pair, SWA ring keeping its rejected block tail | 0.4219 to 0.9258 | 4 of 6 |
| recurrent pair, acceptance walk without the final norm | 0.0000 to 0.5000 | 4 of 6 |
| block pair, rejected tail never rolled off | 0.0117 to 0.9297 | 6 of 6 |
| block pair, one rejected draft kept every partial round | 0.1758 to 0.6406 | 6 of 6 |
| adaptive pair, one rejected draft kept every partial round | 0.0703 to 0.8320 | 5 of 5 judged |
| restricted-vocabulary pair, one rejected draft kept every partial round | 0.0000 to 0.8320 | 5 of 5 judged |
| restricted-vocabulary pair, correction left on the restricted argmax | 0.0000 to 0.8828 | 1 of 6 |
| two-model pair, one rejected draft kept every partial round | 0.0000 to 0.6680 | 6 of 6 |

`MAX_DIVERGENCE_CONFIDENCE` is 0.12, inside the band the measurement leaves:
above the worst correct cell (0.0820, 1.46×) and under the lowest broken cell
above it (0.1445, 1.20×). It clears every correct cell, and no broken engine is
refused on every prompt by this oracle alone — which is why the gate runs **every**
prompt rather than one, and why the repetition control runs beside it. The
block pair's two broken engines are refused on six of six; one of their twelve
cells reads 0.0117, inside the ceiling, and it is the control that refuses it,
reading 1.0000. On the four broken engines added with the new pairs the split
runs the same way: the confidence ceiling fires on 4, 3, 1 and 4 of the judged
cells and the control takes the rest.

The last row is the exception and it is a property of the defect, not of the
ceiling — see the boundary below.

## A declared boundary: EAGLE-3's restricted vocabulary

EAGLE-3's verify pass does not score the whole block against the verifier's
vocabulary. It takes the argmax over the drafter's reduced one — 32000 target
ids, plus the verifier's stop ids so a turn can still end — at every position,
and computes the verifier's full-vocabulary argmax at exactly one: the first
position the draft missed, or the bonus when it missed none. An accepted position
is therefore emitted as the *draft's* token, and that token is the verifier's own
argmax only when the verifier's argmax is inside the restricted set. A restricted
argmax equals the true one exactly when the true one is in the set, so the whole
inexactness lives on the positions where it is not.

That mirrors the upstream implementation, so it is a design boundary rather than
a port defect. It is still an answer change at temperature 0, which is what this
gate reads, so the gate measures the exposure rather than assuming it away. Every
run of that pair prints two figures per prompt: `unnameable`, how many of the
reference arm's own tokens the drafter's vocabulary cannot say, and
`divergence_unnameable`, whether the token the arms parted on is one of them.

Measured over the six prompts: `unnameable` reads 1, 2, 2, 3, 4 and 5 tokens —
under 2% of a 256-token answer — and `divergence_unnameable` is **false on all
six**. The boundary exists and did not fire at any first divergence here; every
one of them is at a token the drafter can name, so the restriction cannot explain
it and the confidence oracle judges it as it judges any other pair.

The gate has power over the boundary when it does bite. Left with the correction
position on the restricted argmax as well — which widens the same inexactness
from "sometimes at an accepted position" to "always at the correction" — the pair
is refused on one prompt of six, at confidence 0.8828, with
`divergence_unnameable` true on exactly that cell. One of six is what the
exposure predicts, and it is why that pair carries a second broken engine: with
its rollback target off by one it is refused on five of the five prompts it
judges.

## Which arm is short

`MIN_ANSWER_TOKENS` is 160 and both arms are read against it, but not
symmetrically, and the restricted-vocabulary pair is what forced the asymmetry to
be written down.

A **speculative** arm under the floor while the reference ran on is the round
loop cutting its own run, and is refused. A **reference** arm under the floor is
the prompt: plain greedy is the control here, exactly as it is for the repetition
control, and a prompt it answered in 52 tokens leaves no answer to compare. The
one shape that is still about the loop is a speculative arm that reproduced the
whole reference answer and then carried on — a loop that did not stop where the
verifier stopped — and that is separable by inspection: the arms are identical up
to the reference's end.

The case: on `Qwen3.6-35B-A3B-8bit` the 4k document is summarised in 52 tokens.
The MTP sidecar and DFlash 1 loops reproduce that answer exactly. EAGLE-3 parts
from it at the fourth token — the single lowest-margin decision in those 52, rank
0.0000 — and writes a longer, well-formed summary instead. Under the old rule
that was refused as a truncation; it is two different answers, and the shorter of
them is one the gate cannot read.

The class this gives up is a loop that runs past the verifier's stop *on a prompt
whose reference answer is under the floor*. It costs nothing on the prompts whose
reference arms run the full budget, which is five of six on every pair here, and
the identical-prefix test keeps the plain form of it. It was also checked
directly: served with the EAGLE-3 drafter, the loop stops on the verifier's stop
ids and returns `finish_reason: stop`.

## Why there is no subsequence floor

The obvious oracle — how much of one answer the two arms share — cannot be
thresholded, and this is measured rather than asserted.

How much two *correct* arms share is decided by where their first near-tie lands
and by nothing else. The figures here are `lcs_ratio` over the whole arm, not the
tail readings `WORST_CORRECT_TAIL_AGREEMENT` is set from — the same runs read
lower on that. On the assistant pair the same engine, model and prompt
family read 0.9375 on the 4k document and 0.4766 on a prompt whose arms flip an
**exact** tie (top-two margin 0.0000) at token 37 — after which both write
well-formed, correct, different prose. The broken engines read 0.2188 to 0.4615
on the same measure — three per cent under the worst correct cell, and nothing
bounds that correct minimum from below, because it is set by where an exact tie
happens to land. No floor can be placed in a gap that narrow and that arbitrary.

The figure is printed on every run, together with the weakest tail window, the
first divergence, the margin there, and both arms' decoded text. It is evidence,
not a gate.

**The three pairs added since made that concrete rather than merely argued.**
`WORST_CORRECT_TAIL_AGREEMENT` — the worst tail reading a correct pair reached —
was 0.2344, measured over two pairs. Over all six it is 0.1094, and the worst
reading of the whole population belongs to the block pair, which had never been
added to it. At the old figure the ragged-loop sweep could assert that everything
the repetition control admits agrees no better than the worst correct pair. At
the true one it cannot: past 60% raggedness the control admits pairs agreeing up
to 0.2188, twice the worst correct reading. The sweep now pins both edges and
asserts the overlap. That is the same claim this section makes, arrived at from
the other side.

## The second oracle: the repetition control

The first has nothing to read when *both* arms are degenerate: two arms in the
same loop have no healthy reference arm whose margins mean anything. So every
run also checks that neither arm repeats at a short period across more than
`MAX_CYCLE_FRACTION` of its tokens, over the whole stream and over each tail cut,
at every period up to `MAX_CYCLE_PERIOD` that leaves `MIN_CYCLE_SAMPLES`
comparisons.

`MAX_CYCLE_FRACTION` is 0.20. Healthy prose from these prompts reads 0.0426 to
0.1351 on the real arms, and 1000 synthetic streams at each of six lengths peak
at 0.1351 and trip the ceiling none — 1.48× of headroom. The other side is swept
rather than sampled: two arms in the same period-8 loop are walked from 0% to
100% raggedness over four seed pairs, and the control refuses every pair up to
60%, past which the arms are more noise than loop. An earlier 0.50, placed from
six sampled points, left 34% to 52% passing. **Nothing takes over past 60%** —
see the subsequence section above for what the pairs the control admits there
agree at.

The control's declared blind spot is its own sample floor: the narrowest window
is `len / TAIL_WINDOWS`, so at this budget it can evidence no period above 32,
and past that the reading comes from a wider window the collapse only partly
fills. A last-quarter loop reads 0.8750 at period 32, 0.2500 at 40 and 0.1875 at
48; the ceiling is crossed between 40 and 48, and the test pins it from both
sides.

Which arm collapsed decides what the gate is entitled to say. A degenerate
speculative arm against a healthy reference is a verdict about the round loop. A
degenerate *reference* arm is a verdict about the input — plain greedy is the
control — and is reported as unjudgeable rather than failed. So is a prompt
neither arm answered: the recurrent pair answers the 4k summary in 13 and 26
tokens, which says nothing about the round loop. The length floor reads the same
asymmetry — see "Which arm is short" above.

## The two defects

**The sliding-window ring kept its rejected block tail.** A speculative round
writes its whole verify block into every layer and then rolls the rejected tail
off. The full-attention layers dropped it; the SWA ring's rollback was a
documented no-op past its wrap, so it kept the rejected drafts and an offset the
rest of the stack had left behind — and because the round loop read its rollback
target off layer 0, which is sliding, the full-attention layers then rolled back
to the wrong place too. A 4k prompt wraps the ring; a short one does not, which
is why nothing showed for as long as it did. See `docs/KV_CACHE.md` for the
lossless-rollback rule the ring now implements.

**The acceptance walk scored an un-normed hidden.**
`Architecture::logits_from_hidden` is documented as taking a pre-final-norm
hidden. Gemma4 applied the norm; Qwen3.5-MoE applied only the LM head. The MTP
round loop's acceptance walk hands it a raw capture, so on that verifier every
verify position was scored with the final RMSNorm's weight vector missing — a
silent reweighting of the vocabulary that agreed with the right answer most of
the time and not always. `logits_from_final_hidden` is now the head-only entry
point, and the callers that hold an already-normed hidden name it.

The issue's hypothesis for the second case — rejected drafts leaving recurrent
state behind — is **falsified**. `rollback_round_caches` already snapshots the
recurrent state before the verify forward and replays the accepted prefix, and it
was never the cause. With the norm applied, three of six prompts come back
bit-identical to plain greedy.

## Structural rules this file was rebuilt under

Both were learned the hard way, over four rounds of review that each planted the
next defect.

- **A fixture must judge the pair, not an arm.** The gate's question is about two
  arms sharing a real prefix and the interaction between its oracles. Every
  fixture written before those rounds tested one arm in isolation, or a pair
  built from independently seeded heads, and a false claim about the interaction
  went unnoticed for two rounds.
- **Do not characterise a population from points you chose.** Sweep the
  parameter, assert the property, and let the sweep pick the points. Both the
  "healthy output ≤ 0.69" claim and the "no gap across the ragged range" claim
  were drawn from a handful of self-authored samples and both were false.

One limitation is worth carrying forward, and it is not only a weakness of the
synthetic argument. `Rng::prose` is an i.i.d. word model with zero
autocorrelation, and real prose is not — so synthetic prose reads **lower** on a
self-similarity measure than the real thing. That generator is not merely
illustrative: `prose_clears_the_control_at_every_length_the_gate_can_hand_it` is
a hard gate on it, and `MAX_CYCLE_FRACTION`'s lower bound is derived from it. The
1.48× headroom over 6000 synthetic streams therefore over-estimates the true
headroom by an unmeasured factor. The real arms are measured too and read 0.0426
to 0.1351, which is what the constant actually rests on; a generator with
realistic autocorrelation, or a corpus of arms captured from runs, would close
the gap.
