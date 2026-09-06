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

Three pairs, each over every prompt in the file. The assistant pair resolves both
halves by slug from `RMLX_O_MODELS_ROOT` and so runs wherever the snapshots are;
the other two have their drafter named by the operator and their verifier is the
reason (below):

| verifier | drafter | round loop | rollback |
|---|---|---|---|
| `gemma-4-e2b-it-mxfp8` | `gemma-4-E2B-it-assistant-bf16` | shared-K/V assistant | KV truncation, SWA ring included |
| `Qwen3.8-27B-mxfp8` | `Qwen3.8-27B-MTP-mxfp8` | MTP sidecar | KV truncation + recurrent snapshot/replay |
| `Qwen3.8-27B-4bit` | `Qwen3.8-27B-DFlash2` | DFlash 2 block drafter | KV truncation + recurrent snapshot/replay |

`RMLX_DRAFT_TEST_MODEL` names one drafter, so a pair whose loop does not drive
the kind that snapshot declares stands down naming both rather than handing
another drafter's tensors to a loader.

Six prompts, all asking for continuous prose. A tokenizer that declares `<think>`
gets an empty reasoning block — its own template's `enable_thinking=false` — and
turn markers are read off the tokenizer's added tokens rather than hard-coded, so
a pair is never served outside its own template.

**The two pairs are selected differently, and the reason is the census.** The
assistant pair resolves both halves by slug, produces **zero** shader-validation
hits (measured, narrowed run), and runs under `make gpu-test` on any machine
holding the snapshots — about 3.5 minutes. The recurrent pair's verifier drives
MLX's mxfp8 quantized matmul, whose `load_safe` bound is the same one
`scripts/gpu_validation_census.txt` records for the affine instantiation: a
narrowed run reports 1344 invalid **loads** (no stores) from
`mxfp8_qmm_t_splitk_bfloat16_t_gs_32_b_8_alN_false`, a kernel this repo does not
compile. The census pins one exact count per originating test, and a count taken
from a 256-token generation moves with every prompt — pinning it would make the
census brittle rather than informative. So that pair resolves its drafter from
`RMLX_DRAFT_TEST_MODEL` only, `make gpu-test` reports it as skipped naming the
variable, and running it is:

```
RMLX_KV_TEST_MODEL=<...>/mlx-community__Qwen3.8-27B-mxfp8 \
RMLX_DRAFT_TEST_MODEL=<...>/mlx-community__Qwen3.8-27B-MTP-mxfp8 \
cargo test -p rmlx-models --test spec_greedy_equivalence -- --ignored --nocapture \
  the_recurrent_round_loop
```

That is a real gap: the case the second defect was found in is not in a gate that
runs by default. Closing it needs either the census analysis extended to this
kernel and model, or a stable count — neither of which this change is the place
for.

**Three of the round loops that exist are outside the gate entirely.** It
covers the Gemma4 assistant loop, the Qwen3.5-family MTP sidecar loop and the
DFlash 2 block loop. The DFlash 1, EAGLE-3 and two-model loops have no pair
here, and the property is not transitive across loops — each one has its own
rollback and its own acceptance walk, which is what the gate reads.

That boundary is not hypothetical. Served at temperature 0 on the code prompt,
`Qwen3.8-27B-4bit` drafted by `z-lab/Qwen3.8-27B-DFlash2` at block 8 diverged
from the same verifier's no-drafter answer at the fourth token — "We need to
respond to user:" against "We need answer user's request:" — and stayed
diverged; the MTP sidecar on the same verifier, same prompt and same 160-token
budget is byte-identical to it. A greedy speculative loop can only emit the
verifier's own argmax, so a drafter proposing badly costs throughput and cannot
change the answer. A changed answer is the loop, not the drafter.

**That arm was the DFlash 1 loop wearing the checkpoint's name**, and it no
longer exists: the checkpoint declares itself a DFlash 2 drafter and routes to
its own loader and its own round loop, which is now a pair in this gate and
agrees with plain greedy on six of six prompts. What the observation is still
evidence about is DFlash 1, which drives `z-lab/Qwen3.6-35B-A3B-DFlash` on its
own verifier and remains uncovered — that is the pair a DFlash 1 case here would
be built on.

## The oracle: where a correct pair diverges

A reduction-order difference is a relative perturbation of order `1e-3` on a
logit. It can flip a decision the verifier was already nearly indifferent about
and it can flip nothing else. So the gate reads the verifier's own top-two
logprob margin at the position the two arms **first** differ, and returns where
that sits in the same arm's own margin distribution. Both arms saw the same
context up to that position, so this judges the pair rather than an arm, and it
needs no per-prompt calibration: it is a rank, not a number of nats.

Measured over three pairs and six prompts each, against the shipped engine and
against a deliberately broken one per pair — two on the block pair:

| engine | percentile of the reference arm's own margins |
|---|---|
| assistant pair, as shipped | 0.0000 to 0.0820 |
| recurrent pair, as shipped | 0.0000 to 0.0234 |
| block pair, as shipped | 0.0000 to 0.0273 |
| assistant pair, SWA ring keeping its rejected block tail | 0.4219 to 0.9258 |
| recurrent pair, acceptance walk without the final norm | 0.0000 to 0.5000 |
| block pair, rejected tail never rolled off | 0.0117 to 0.9297 |
| block pair, one rejected draft kept every partial round | 0.1758 to 0.6406 |

`MAX_DIVERGENCE_CONFIDENCE` is 0.12, inside the band the measurement leaves:
above the worst correct cell (0.0820, 1.46×) and under the lowest broken cell
above it (0.1538, 1.28×). It clears every correct cell and refuses ten of the
first twelve broken ones.
The two it does not are covered by running **every** prompt rather than one:
each broken engine is refused on at least four of its six, so a gate pinned to
one prompt would be a coin toss and this one is not.

The block pair's two broken engines are refused on six of six each. One of their
twelve cells reads 0.0117, inside the ceiling, and is refused by the repetition
control instead — which reads 1.0000 on it. The two oracles cover each other and
neither alone gives that recall.

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
100% raggedness over four seed pairs, and every pair the control lets through
agrees no better than the worst a correct pair reached. An earlier 0.50, placed
from six sampled points, left 34% to 52% passing.

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
tokens, which says nothing about the round loop. One arm short while the other
ran on is refused.

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
