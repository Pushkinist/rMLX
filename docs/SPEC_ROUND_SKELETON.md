# One speculative round loop: the proposed interface

**Status: proposal.** No engine code exists for any of it. This file is what a
reviewer judges before the first drafter is migrated, and what each migration
chunk is held to afterwards.

Seven round loops in `crates/rmlx-models/src/speculative/` run one algorithm:

| loop | file |
|---|---|
| `mtp_generate` | `crates/rmlx-models/src/speculative/mtp.rs` |
| `dflash_generate` | `crates/rmlx-models/src/speculative/dflash/mod.rs` |
| `dflash2_generate` | `crates/rmlx-models/src/speculative/dflash2/round.rs` |
| `eagle3_generate` | `crates/rmlx-models/src/speculative/eagle3/mod.rs` |
| `mtp_assistant_generate` | `crates/rmlx-models/src/speculative/gemma4_assistant.rs` |
| `spec_generate_greedy_cached` | `crates/rmlx-models/src/speculative/mod.rs` |
| `spec_generate_stochastic_cached` | `crates/rmlx-models/src/speculative/mod.rs` |

`spec_generate_greedy` in the same file is the two-model entry guard and
dispatcher, not a loop: it validates a request, resolves the draft count and
delegates.

## The shared skeleton, read off the seven bodies

Every one of them, in this order: refuse a prompt under two tokens; resolve the
request's block; build the verifier's cache stack and its recurrent stack;
prefill; emit a seed or not; then per round — narrow the block against the
remaining budget, propose, arm the recurrent tape, verify, walk the acceptance,
emit what the round committed, roll the verifier's caches back to the accepted
prefix, roll the drafter's own state back, condition the next round, log one
round event; then, after the loop — one request record, one report of the
verifier's resident KV, and the widest block any round ran.

What differs is five things: how a drafter proposes, what it conditions on, how
it rolls its own state back, which of the verifier's outputs it needs captured,
and — for one loop — what "the verifier accepted this" means.

## The interface

One loop function in a new `round_loop.rs` beside
`crates/rmlx-models/src/speculative/round_common.rs`, and one trait each drafter
implements in its own module.

```rust
pub(crate) trait RoundDrafter {
    /// Prefill, and the token the request emits before its first round.
    /// `None` is a pair that emits nothing until a round has run.
    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled>;

    /// The block this round runs. Default `round_block(block_total, remaining)`.
    fn block(&self, block_total: usize, remaining: usize) -> usize;

    /// This round's proposals. An empty chain is the loop's refusal, not this one's.
    fn propose(&mut self, ctx: &mut RoundCtx<'_>, carry: u32, block: usize) -> Result<Vec<u32>>;

    /// Score the round and say what it commits.
    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32], remaining: usize) -> Result<Verdict>;

    /// Return the drafter's own state to the accepted prefix. `None` is a
    /// drafter that keeps nothing across rounds.
    fn rollback(&mut self, ctx: &RoundCtx<'_>, v: &Verdict) -> Result<Option<CacheSpan>>;

    /// Condition the next round on what this one committed. `None` is a drafter
    /// that carries no conditioning buffer.
    fn condition(&mut self, ctx: &RoundCtx<'_>, v: &Verdict) -> Result<Option<Conditioning>>;
}
```

`Verdict` carries the accepted count, the tokens the round commits, and — for a
restricted-vocabulary drafter — how long this round's restricted prefix is.
`RoundCtx` carries the verifier, its cache stacks, the device and the charge
decision; `rollback` and `condition` take it immutably, so only `prefill` and
`verify` can advance the verifier's caches.

### Why the fields are paired

Today seven `RoundReport` literals each state every field, which is what makes a
new field seven compile errors. One literal makes it one, and the value then has
nowhere to come from. Pairing restores the property by moving the per-drafter
half of the report into structs the *drafter* constructs:

```rust
pub(crate) struct CacheSpan { pub before: i32, pub target: i32 }
pub(crate) struct Conditioning { pub rows: i32, pub projected: i32 }
```

A field added to either is a compile error in every drafter that builds one. A
field added to the loop's own half — the round index, the accept, the committed
count, the charge, the phases — is one compile error at one site, correctly, so
no drafter is asked for a fact it does not have.

Both are returned as `Option`, and the `None` arm is a word rather than a
literal of `None` fields. That is deliberate: a drafter that keeps no cache and
one that keeps one are two different statements, and only the second owes a
value when the struct grows. The cost is that a `None` drafter is silent about a
new field — which is right, since it has none — and the reviewer's job is to
check that a drafter answering `None` really carries nothing.

## What the interface cannot express, and what is proposed for it

Seven things. Five are expressible with no branch in the loop; two are decisions
the owner has to take.

1. **The stochastic acceptance rule.** `spec_generate_stochastic_cached` does not
   compare tokens: it builds the verifier's post-sampling distribution at every
   position, tests each proposal against the drafter's own, and samples the
   extra token. Its `verify` therefore cannot share a body with the other six.
   Proposed: `verify` owns the acceptance rule; six drafters delegate to one
   shared greedy body (forward, read back the verifier's own tokens,
   `accept_prefix`), one has its own. That is a second implementation of one
   method, not a branch in the loop and not a twin — the shared body exists once.
2. **The four capture shapes.** Multi-layer hidden capture (the MTP sidecar,
   both DFlash loops, EAGLE-3 cold), shared K/V plus a raw hidden (the Gemma4
   assistant), no capture (both two-model loops), and EAGLE-3's restricted
   read-back with one full-vocabulary correction. Proposed: the forward is inside
   `verify`, so "what to capture" never reaches the loop as a value it has to
   branch on.
3. **EAGLE-3's per-token attribution.** The `&mut Vec<DecidedBy>` is in the
   public signature and the equivalence gate reads it. Proposed: the loop carries
   it as one `Option` and hands it to `round_common::emit_round_tokens`, which
   already takes exactly that argument, with `Verdict::restricted` as the prefix
   length. The seed's `DecidedBy::FullVocab` becomes a general rule — a loop with
   an attribution buffer and a seed attributes the seed to the full vocabulary.
4. **DFlash 1's adaptive block.** `dflash_next_block_size` opens with
   `round_block` and then moves the result by the accept rate of the recent
   rounds. Proposed: the `block` method, six defaults and one override.
5. **The widest block a run reached.** The five sidecar loops return the widest
   block; the two two-model loops return the widest *proposal count*, which is
   one less. Proposed: the loop returns the widest block and the two-model entry
   subtracts one, so the unit difference stays in the module that owns it.
6. **The verifier offset a round reports.** The Gemma4 assistant reads it before
   its verify forward and counts forward over the accepted prefix; the other six
   read it after and count back from the tail. The two spellings name the same
   position — pinned by `the_two_rollback_spellings_name_the_same_position` in
   `crates/rmlx-models/src/speculative/round_skeleton_tests.rs` — but they are
   two different numbers on the round line, and the pinned baseline holds both.
   Proposed: the loop reads the offset before *and* after the forward, computes
   the rollback target from the head spelling once, refuses the round when the
   two spellings disagree, and reports `CacheSpan::before` from a two-valued
   basis each drafter declares. The alternative — one basis for all seven and a
   re-blessed baseline — moves 36 cells for a field whose meaning does not
   change, and costs a full capture.

   The refusal is worth stating on its own: it is a check nothing in the tree
   performs today, it is free, and it is the only runtime catcher a rollback off
   by one would have that does not need a GPU.

7. **The exit that reports the verifier's resident KV.** Each loop has exactly
   one exit that calls `round_common::report_verifier_kv_bytes` and exactly one
   that does not, and *which* exit differs: the five sidecar loops skip it on the
   seed-EOS exit and reach it after an in-round EOS; the two two-model loops
   reach it on the normal exit and return before it on an in-round EOS. One loop
   has one exit shape.

   **Proposed, and it is a behaviour change: every exit reports.** The figure has
   one writer — a speculative request never goes through
   `Architecture::generate_greedy` — so a request that skips it leaves the
   previous request's figure readable to a caller that samples around the call.
   The two exits that skip it today are the two that would read stale.

   This changes no baseline cell, no pair reading and no gate. That is exactly
   why it must be written down here rather than merged: see "the mutation I
   could not catch" below.

## Migration order

One drafter per chunk after this one, each its own PR, each proving its own pair
and a byte-identical round stream.

1. **The Gemma4 assistant.** Its pair is the only one that resolves both halves
   by slug and runs under `make gpu-test` on any machine holding the snapshots —
   every other pair needs an operator-named `RMLX_DRAFT_TEST_MODEL`, so this is
   the only first choice whose proof a reviewer can reproduce without being told
   which snapshot to fetch. It exercises the most optional arms of the interface
   per unit of proof cost: the charged arm (`phases_charged()` and
   `RoundPhases`), the head-basis offset of item 6, the `None` recurrent stack,
   a drafter that keeps no cache of its own, and the sliding-window ring's
   rollback. This chunk also carries the gate re-key below — a migrated entry
   moves out of the charge gate's derived population the moment its body goes.
2. **The MTP sidecar.** The complement: a recurrent refold, a drafter cache
   rolled back by offset, a single conditioning row, and a `d_offset`/`d_target`
   pair on the round line. Two pairs on one verifier at two blocks.
3. **DFlash 2.** A conditioning window that slides, a prompt-window capture, and
   the second charged loop.
4. **DFlash 1.** The adaptive block, and the one loop whose verify width is not
   fixed.
5. **EAGLE-3.** The restricted vocabulary, the attribution buffer, and a drafter
   that rolls back by re-running.
6. **The two-model greedy loop.** Two full models, a draft-side rollback with its
   own tape, no seed, and the resync that prepends the last draft token on a full
   accept.
7. **The two-model stochastic loop.** Last, because it is the only acceptance
   rule that is not the shared one and the only loop with no equivalence pair —
   its gate is `crates/rmlx-models/tests/two_model_stochastic.rs`, which pins
   that one seed reproduces one sequence.

## The oracle

Four observables, and each is blind to something different.

- **The per-round event stream** against the checked-in baseline:
  `python3 scripts/spec_round_stream_compare.py verify <capture>`, 36 of 36 cells
  at every chunk, digests from
  `crates/rmlx-models/tests/fixtures/spec_round_baseline/MANIFEST.sha256`. It
  sees the round's shape — block, accept, proposals, committed rows, both cache
  spans, the conditioning pair, `refolded`. It is blind to the values inside a
  round: two runs agreeing on every count can be scoring different logits. It is
  blind to `charged`, which reads `false` on both sides by construction. It is
  blind to anything a loop does not report, the resident-KV figure included.
- **The equivalence pairs**, judged by the divergence-confidence oracle in
  `docs/SPEC_ANSWER_EQUIVALENCE.md` at the readings recorded there. They see the
  answer. They are blind to every sampled arm, blind to the stochastic loop
  entirely, and blind wherever a pair's drafter is not resolvable on the machine
  running them.
- **`make check-spec-charge`**, whose census must read `charge_phases:3 false:4`
  at every step of the campaign — see the gate section. It sees a decision that
  moved. It cannot see whether the schedule it describes is the right one.
- **`make check-spec-sampling`**, which sees that a driver was handed the
  request's sampler and read it. It cannot tell a right distribution from a
  wrong one; `crates/rmlx-models/tests/spec_sampled_distribution.rs` does that,
  on one pair.

**The greedy digest is not evidence.** Greedy verification emits the verifier's
own argmax at every position whatever the drafter proposed, so a byte-identical
answer after a draft-side change says that run's near-ties happened not to move.

## What cannot move

- Every cell of the pinned round stream: 36 cells, their round counts, and the
  digest of each. A cell that moves for a legitimate reason is re-blessed in the
  same commit with the loop and the reason named.
- Every equivalence-pair reading in `docs/SPEC_ANSWER_EQUIVALENCE.md`, recorded
  again in each migration PR's body.
- The exit behaviour of every loop: the seed-EOS exit that returns before any
  round, the in-round EOS exit that stops the request, the empty-chain refusal,
  and the token budget that caps the emission without capping the acceptance.
- The sampler read: every driver takes the request's sampler and reads it.
- `charged` per drafter — three loops charge, four do not, and no migration
  changes which.
- The two request shapes that reach no loop: a request carrying a sampler
  constraint is refused at the speculative entry, and no loop consults or
  publishes a prompt-cache slot. A shared loop that accepted a pre-filled cache
  would open a path no observable here can see.

## The mutations, and what catches each

| mutation | catcher | exists today |
|---|---|---|
| the rollback target off by one at the loop's call site | round stream `v_target`; the pairs (this is the "rejected tail never rolled off" broken engine, refused 6 of 6); and, proposed above, the loop's own head-against-tail refusal | stream and pairs yes, the refusal is new |
| a drafter's condition called on the wrong row | round stream `condition_rows` / `projected_rows`; `guard_round_conditioning` refuses a projection that does not match the commit | yes |
| a capture handed to the wrong drafter | the type system — each drafter takes its own capture inside its own `verify` | new, structural |
| the block narrowing dropped, or the adaptive schedule applied to the wrong drafter | round stream `num_draft` | yes |
| the recurrent tape not armed before the verify | `refold_lin_tapes` refuses a tape that does not describe the round | yes |
| a seed emitted where none was, or dropped | round stream `emitted_total` shifts on every line; `RoundStats::emission_violation` | yes |
| the restricted prefix reported for a loop that has no restricted vocabulary | the EAGLE-3 boundary rule in the equivalence gate reads `DecidedBy` per token | yes |
| the charge token hard-wired across the collapse | `make check-spec-charge`, re-keyed as below | needs the re-key |
| the in-round EOS exit stops recording the request | round stream ends early; the request record is absent | yes |
| **the seed-EOS exit stops returning early** | nothing at runtime: no round runs, so no round line is written, and no gate prompt has an EOS seed | **new, none** |
| **the resident-KV report moved to another exit, or computed from the wrong caches** | nothing at runtime | **new, none** |
| **the widest-block return changes unit for the two-model entry** | nothing: the server discards it and no two-model test reads it | **new, none** |
| the empty-chain refusal lost | nothing at runtime — a drafter that proposes nothing makes the acceptance walk answer rather than refuse, and the round silently emits one token | **new, none** |

Two of the four uncatchable rows get a text-level catcher in this chunk, in
`crates/rmlx-models/src/speculative/round_skeleton_tests.rs`: one test pins that
every loop has exactly one exit that reports the verifier's resident KV and
exactly one that does not, and one pins that every loop refuses an empty
proposal chain before its acceptance walk. Both read source text, so both are
weaker than a runtime check by exactly the distance between "the call is still
written" and "the call still does the right thing". What they buy is that a
migration cannot change either disposition without editing an assertion and
saying why.

### The mutation I could not catch

**The resident-KV report.** Nothing observable in this crate reads it: it is not
on the round line, the equivalence pairs do not touch it, the accept counters do
not carry it, and the value only becomes visible in an append-only metrics row
written by the server. A migration that keeps the call and reports the wrong
caches — the drafter's instead of the verifier's, say, which is the mistake the
function's own documentation warns against — produces a plausible figure, a
green `make ci`, a green `make gpu-test` and 36 of 36 matching round-stream
cells.

The proposal in item 7 above is itself in this class, which is the reason it is
written here as a decision to take rather than merged as a tidy-up.

Naming what would close it, since it is out of this chunk's scope: the figure
would need a second reader — a request-level assertion that the verifier's
resident bytes are non-zero after any request that ran a round, and that they
describe the verifier's stack rather than the drafter's. That is a test needing
two loaded models, so it belongs with the equivalence pairs under `make
gpu-test`, not on the CPU side.

## Gate changes the skeleton needs

**`make check-spec-charge`.** Its population is derived as "carries the `step_fn`
signature **and** constructs a `RoundTotals`". After the first migration the
migrated entry keeps its signature and loses its totals, so it drops out of the
population, the census reads `charge_phases:2 false:4` and RULE 3 fires. The
re-key therefore lands in the same PR as the first migration, not later, and it
is a re-key of the *population*, not of any rule's needle:

- Population (a), the loop: `step_fn` plus a `RoundTotals`. Exactly one fn.
  RULE 1 reads it as today — the `charge` argument of its `rollback_round` call
  and the `charged:` of the record it builds are one token, which is the
  parameter it was handed. RULE 7's "exactly one `log_round(`" reads it too.
- Population (b), the entries: a fn that constructs the loop's configuration
  literal. Seven fns, each naming the decision once, at the call that runs the
  loop, in the drafter's own module. RULE 2 (a `charge_phases` token binds
  exactly `phases_charged()`), RULE 3 (the census) and RULE 5 (nothing else
  names a `charged:`) read this population, with the same needles they use now,
  because the entry writes `charged:` in a literal exactly as a loop does today.
- During the campaign both populations coexist with the unmigrated loops, which
  are read by the rules unchanged. **The census is invariant at every step:
  seven charge sites, `charge_phases:3 false:4`.** What falls, 7 to 1, is the
  count of fns in population (a), and that is a second number the gate should
  state rather than infer.
- RULE 6 is untouched, provided the loop does **not** live in
  `crates/rmlx-models/src/speculative/round_common.rs` — that file's exemption
  for naming the low-level rollback is anchored to its path, and a loop moved
  into it would inherit an exemption it must not have. Hence `round_loop.rs`.
- RULE 7's directory scope is untouched for the same reason: the loop stays
  under `crates/rmlx-models/src/speculative/`.
- The fixtures for each re-keyed rule move with it, asserting the reason and not
  only the exit code.

**`make check-spec-sampling`.** RULE 1's population is `pub fn` drivers. The
seven entries stay `pub` and keep mentioning `sampler_cfg` when they pass it, so
the gate stays green — while the loop that actually reads the sampler is
`pub(crate)` and outside it. Widen RULE 1 to any visibility, which is the rule
`check_spec_charge.sh` already uses, so the one loop is enumerated.

**`make debt-report`.** Its driver group is discovered by the `step_fn`
signature alone, so it already lists `emit_step`, `emit_round_tokens` and
`emit_seed_token` beside the loops — eleven fns where the campaign measures
seven. Narrowing it to the charge gate's population is an open item in its own
right. At the end of the campaign the group must list one driver plus the
per-drafter `propose` / `condition` / `rollback` implementations, whose pairwise
similarity is what their real differences warrant.

## Duplication at the base of the campaign

Measured at the merge base with `scripts/lib/debt_report.py`'s own extractor
over the seven loop bodies — 2326 lines summed, 21 pairs:

```
dflash_generate      <-> dflash2_generate                : 58.2%
dflash_generate      <-> eagle3_generate                 : 47.0%
dflash_generate      <-> mtp_assistant_generate          : 45.3%
dflash_generate      <-> spec_generate_greedy_cached     :  5.7%
dflash_generate      <-> spec_generate_stochastic_cached :  6.0%
dflash_generate      <-> mtp_generate                    : 62.2%
dflash2_generate     <-> eagle3_generate                 : 38.4%
dflash2_generate     <-> mtp_assistant_generate          : 43.7%
dflash2_generate     <-> spec_generate_greedy_cached     :  5.3%
dflash2_generate     <-> spec_generate_stochastic_cached :  5.6%
dflash2_generate     <-> mtp_generate                    : 55.6%
eagle3_generate      <-> mtp_assistant_generate          : 35.5%
eagle3_generate      <-> spec_generate_greedy_cached     :  8.9%
eagle3_generate      <-> spec_generate_stochastic_cached :  9.4%
eagle3_generate      <-> mtp_generate                    : 45.3%
mtp_assistant_generate <-> spec_generate_greedy_cached   :  4.8%
mtp_assistant_generate <-> spec_generate_stochastic_cached: 5.1%
mtp_assistant_generate <-> mtp_generate                  : 53.1%
spec_generate_greedy_cached <-> spec_generate_stochastic_cached: 72.7%
spec_generate_greedy_cached <-> mtp_generate             :  3.8%
spec_generate_stochastic_cached <-> mtp_generate         :  4.1%
```

**Summed matched lines over the 21 pairs: 2032.** Two extraction chunks raised
this figure while lowering nothing — a shared helper called from seven places
leaves seven identical call sites where seven different bodies used to be. Each
migration chunk reports the figure again, and a chunk that raises it has moved
duplication rather than removed it. The figure only falls when a loop body is
deleted, which is what this campaign is for: after the last migration there is
one body and no pair.
