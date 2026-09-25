# The speculative round loop and its drafter interface

Every speculative drafter runs one round loop, `run_rounds` in
`crates/rmlx-models/src/speculative/round_loop.rs`. A drafter meets it through
one trait, `RoundDrafter`. This doc states what the loop does, what a drafter
declares, what the interface cannot express, and what no check can see.
`docs/SPECULATIVE.md` describes the drafters and how to run them.

Seven drafter paths enter the loop, each through its own entry function:

| Entry | Drafter | File |
|---|---|---|
| `mtp_generate` | `SidecarRound` | `crates/rmlx-models/src/speculative/mtp.rs` |
| `mtp_assistant_generate` | `AssistantRound` | `crates/rmlx-models/src/speculative/gemma4_assistant.rs` |
| `dflash_generate` | `AdaptiveRound` | `crates/rmlx-models/src/speculative/dflash/round.rs` |
| `dflash2_generate` | `BlockRound` | `crates/rmlx-models/src/speculative/dflash2/round.rs` |
| `eagle3_generate` | `Eagle3Round` | `crates/rmlx-models/src/speculative/eagle3/round.rs` |
| `spec_generate_greedy_cached` | `TwoModelRound`, `Acceptance::Prefix` | `crates/rmlx-models/src/speculative/mod.rs`, `two_model.rs` |
| `spec_generate_stochastic_cached` | `TwoModelRound`, `Acceptance::Stochastic` | `crates/rmlx-models/src/speculative/mod.rs`, `two_model.rs` |

`spec_generate_greedy` is not a path. It validates a two-model request,
resolves the proposal count and routes to one of the two two-model entries.

## The loop

An entry refuses a prompt under two tokens and resolves the request's block,
both before any cache stack exists. It then builds a `RoundCfg` and calls
`run_rounds`. The loop:

1. builds the verifier's cache stack and, on a hybrid, its recurrent stack;
2. calls `prefill`, and emits the seed if the drafter drew one;
3. per round: narrows the block (`block`), proposes (`propose`), refuses an
   empty chain, arms the recurrent tape, verifies (`verify`), bounds the
   reduced-vocabulary prefix, emits the commit, rolls the verifier back to the
   accepted prefix, then calls the drafter's `rollback`, `condition` and
   `carry`, and writes one round event;
4. writes one request record and reports the verifier's resident KV.

It returns the emitted steps and the widest block any round ran.

The loop owns every step that is the same for all drafters. The rollback
target is always computed from the offset read before the verify forward
(`rollback_target_from_head`). The carry token for the next round has one
producer: the loop reads it off the commit and hands it over in
`RoundOutcome::carry`. The recurrent tape is armed by the loop, because the
loop's own rollback is the tape's only consumer.

## The interface

`RoundDrafter` has six methods and two associated constants. The doc comments
in `round_loop.rs` are the reference; this is the summary.

| Item | What the drafter states |
|---|---|
| `prefill` | Its prefill, the seed (`Seed::Emitted` or `Seed::Carried`), its own `prefill_ns`, whether the request takes a reduced read-back, whether it projects conditioning |
| `block` | The block this round runs. Default: `round_block` |
| `propose` | This round's proposals |
| `verify` | The round's `Verdict`: accept count, commit, timings, reduced prefix length |
| `rollback` | Its own state back to the accepted prefix, as a `CacheSpan`. Default: `None` |
| `condition` | What the next round conditions on, as a `Conditioning` |
| `carry` | Every array the next round's drafter reads, handed to a callback |
| `KV_REPORT_SKIPPED_BY` | Which of the two exits skips the resident-KV report |
| `VERIFIER_OFFSET_BASIS` | Which read of the verifier's offset the round line reports |

`RoundCtx` carries the verifier, its two cache stacks, the context ceiling and
codec they were built at, the request's `VerifierDraw`, the charge decision and
the device. `rollback` and `condition` take it immutably, so only `prefill` and
`verify` can advance the verifier's caches or the draw.

`carry` must invoke its callback. The round event is written inside it, and the
loop refuses a round whose drafter returned without calling it.

### What each drafter declares

| Drafter | Seed | Report skipped by | Offset basis | Projects conditioning | `rollback` | `condition` | `block` | Charges phases |
|---|---|---|---|---|---|---|---|---|
| MTP sidecar | emitted | seed exit | after | no | its KV | one sliced row | default | yes |
| Gemma4 assistant | emitted | seed exit | before | no | default | `None` | default | yes |
| DFlash 1 | emitted | seed exit | after | yes | default | committed rows | adaptive | no |
| DFlash 2 | emitted | seed exit | after | yes | default | `accept + 1` rows | default | yes |
| EAGLE-3 | emitted | seed exit | after | no | its KV | `None` | default | no |
| two-model, both rules | carried | in-round exit | after | no | the draft model's stacks | `None` | default | no |

"Charges phases" is the entry's `charged` field: `phases_charged()` where yes,
`false` where no. `docs/SPECULATIVE.md` § "Charged and uncharged phases" says
what a charged phase is.

### Why the report fields are paired

A round event has two halves. The loop fills its own half: round index, accept,
committed count, charge and phases. The drafter builds the other half as two
structs, `CacheSpan` and `Conditioning`. A field added to either struct is a
compile error in every drafter that builds one, and nowhere else. A drafter
that keeps no cache answers `None` and owes no value. A reviewer checks that a
`None` drafter really carries nothing.

`Conditioning::projected` is optional because the MTP sidecar slices one
verifier row per round and projects nothing. `Conditioning::rows` is read off
the buffer's shape, not off what the round meant to put in it.

## What differs per drafter, and where it lives

None of these is a branch in `run_rounds`.

**The acceptance rule.** The greedy walk compares tokens; the stochastic rule
compares the two distributions each proposal was drawn from. Every path but
the two-model stochastic one uses the greedy walk, `accept_prefix`. The
two-model drafter holds the rule as the field `two_model::Acceptance`, read
only by `propose` and `verify`. `Prefix` is the greedy walk. `Stochastic` runs
Leviathan's test in `stochastic_prefix`. Both fill the same `Verdict`: the
accepted prefix plus the one token the verifier stands behind.

**The capture.** The forward and what it captures are inside `verify`:
multi-layer hidden states for the sidecar, both DFlash drafters and EAGLE-3,
shared K/V plus a raw hidden for the assistant, nothing for the two-model
drafter. The loop never sees a capture.

**The reduced-vocabulary read-back.** EAGLE-3 can score accepted positions over
its draft vocabulary. It decides once per request, in `prefill`: the drafter's
hot path is active and the draw is not sampling. The loop logs that decision on
its starting line. `Verdict::restricted` is the reduced prefix length.
`guard_restricted_prefix` refuses a round whose prefix exceeds its acceptance,
or that reports one on a request that declared none. The loop writes one
`DecidedBy` entry per emitted token, and attributes the seed to the full
vocabulary.

**The adaptive block.** DFlash 1 overrides `block` with
`dflash_next_block_size`. Its history of `(accepted, drafted)` pairs is written
in `verify` and read at the head of the next round.

**What a round conditions on.** DFlash 1 conditions on the committed count and
DFlash 2 on `accept + 1`, at the same two calls, `committed_rows` and
`guard_round_conditioning`. The two agree except when the budget cuts the
block. The MTP sidecar advances its drafting position by the committed count.
That is why `condition` runs after the emission and receives `RoundOutcome`.

**The conditioned-rows figure.** `projects_conditioning` decides whether the
request record carries `conditioned_rows` as `Some` or `None`. The loop opens
the count at zero and adds each round's projection. A drafter cannot supply an
opening value. The loop refuses a round that projects rows the request
declared it would not.

**The verifier offset on the round line.** The assistant reports the read
before the verify forward; the others report the read after it. Both name the
same position. The loop computes the target from the pre-forward read in
either case.

**`prefill_ns`.** Each drafter times its own prefill, because what the span
covers differs. The loop does not time the call.

**The resident-KV report.** The loop has two early exits. The seed exit returns
when the seed is an EOS. The in-round exit returns when a round emits an EOS.
Each drafter skips the report at exactly one of them, and declares which. The
report has one writer, so a skipped report leaves the previous request's
figure readable.

**The block figure.** The loop returns the widest block that ran. The seed exit
returns the resolved block instead, because no round has run there. The
two-model entries return `drafts_per_round` of the figure, and
`spec_generate_greedy` adds the verifier's token back.

**One draw stream per request.** `VerifierDraw` holds the request's one
generator, seeded from the sampler. A drafter draws through it:
`block_tokens` and `block_distributions` for the verifier, `proposal` for a
drafted token, `rng` for the coins the stochastic rule tosses. A drafter that
seeded a second generator from the same seed would produce two correlated
streams.

**The empty-chain refusal.** One refusal, in the loop, before the verify
forward. Its message says an empty chain is a broken drafter, not the end of
the request.

## The two two-model entries

`spec_generate_greedy_cached` and `spec_generate_stochastic_cached` share most
of their lines. Each makes three statements alone: its `SpecLoop`, its
`Acceptance` and its refusal text. `make check-spec-charge` and
`make check-spec-sampling` both pin seven drafter paths, one entry per path.
Folding the two into one entry with a parameter would leave six entries and a
wrapper in no population. They stay apart for that reason.
`each_two_model_entry_names_its_own_rule_its_own_loop_kind_and_its_own_block`
in `crates/rmlx-models/src/speculative/tests.rs` holds the three statements
apart.

## The oracle

Five observables, each blind to something different:

- **The per-round event stream**, against
  `crates/rmlx-models/tests/fixtures/spec_round_baseline/MANIFEST.sha256`,
  compared by `scripts/spec_round_stream_compare.py`. It sees each round's
  shape. It is blind to the values inside a round, to `charged` (every pinned
  run is uncharged), to the resident-KV figure, and to a request that ran no
  round.
- **The request record**, one `done` line per request. It is the only
  observable that covers a request that stopped on its seed, and the only one
  that carries `conditioned_rows`. It cannot say which round moved anything.
  `RoundStats::conditioning_violation` and `emission_violation` pass on `None`.
- **The equivalence pairs** in
  `crates/rmlx-models/tests/spec_greedy_equivalence.rs`, judged by
  `docs/SPEC_ANSWER_EQUIVALENCE.md`. They see the answer and assert the
  returned block on every arm and prompt. They are blind to sampled arms and to
  the stochastic rule.
- **`make check-spec-charge`**, whose census reads `charge_phases:3 false:4`
  over seven entries. It sees a charge decision that moved, not whether the
  decision is right.
- **`make check-spec-sampling`**, which sees that every path hands the loop the
  request's sampler. `crates/rmlx-models/tests/spec_sampled_distribution.rs`
  checks a sampled distribution, on one pair.

A byte-identical greedy answer is not evidence for a draft-side change; see
`docs/SPECULATIVE.md` § "Judging a draft-side change".

`crates/rmlx-models/src/speculative/round_skeleton_tests.rs` reads the loop's
source. `DISPOSITIONS` states, per drafter, which exit skips the report, how the
empty chain is refused and which offset the round line reports. The marker
reading holds `round_loop.rs` to the pattern `SPEWGRIP`: the seed guard, the
seed exit's report and return, the round loop, the empty-chain refusal, the
tail record, its guard and its report. It reads statement order, not
reachability or arguments.

## What no runtime check sees

Each row is a defect that leaves every observable above green.

| Defect | Why nothing sees it |
|---|---|
| The seed exit stops returning early | No round runs, so no round line is written; no gate prompt stops on its seed |
| The resident-KV report reads the drafter's caches, or sits in a branch that never runs | The figure reaches only a server metrics row. The marker reading pins where the call is written, not what it reads |
| `conditioned_rows` on a seed-exit record falls from `Some(0)` to `None` | `conditioning_violation` passes on `None`; no pair stops on its seed |
| A drafter declares `projects_conditioning` and projects nothing | The record moves from `None` to `Some(0)` and both pass |
| The seed exit returns the widest block that ran | No gate prompt stops on its seed |
| The empty-chain refusal lost | The acceptance walk answers an empty chain instead of refusing it, and the round emits one token. The marker reading catches the deleted text, not the behaviour |
| The loop times `prefill` itself | Five of seven records move, and the figure has no bound |
| The seed attributed to the reduced vocabulary | The seed is the verifier's own argmax on both arms, and no reader checks `decided_by[0]` |

The resident-KV row would need a request-level test with two loaded models,
under `make gpu-test`.
