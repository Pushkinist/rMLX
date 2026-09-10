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
verifier's resident KV, and a block figure the caller reads back.

What differs is five things: how a drafter proposes, what it conditions on, how
it rolls its own state back, which of the verifier's outputs it needs captured,
and — for one loop — what "the verifier accepted this" means.

## The interface

One loop function in a new `round_loop.rs` beside
`crates/rmlx-models/src/speculative/round_common.rs`, and one trait each drafter
implements in its own module.

```rust
pub(crate) trait RoundDrafter {
    /// Which of this loop's two exits skips the report of the verifier's
    /// resident KV. A declared per-loop skip — see item 7.
    const KV_REPORT_SKIPPED_BY: ReportSkippedBy;

    /// Prefill, the token the request emits before its first round, and how long
    /// the prefill took. `None` is a pair that emits nothing until a round runs.
    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled>;

    /// The block this round runs. Default `round_block(block_total, remaining)`.
    fn block(&self, block_total: usize, remaining: usize) -> usize;

    /// This round's proposals. An empty chain is the loop's refusal, not this one's.
    fn propose(&mut self, ctx: &mut RoundCtx<'_>, carry: u32, block: usize) -> Result<Vec<u32>>;

    /// Score the round and say what it commits.
    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32], remaining: usize) -> Result<Verdict>;

    /// Return the drafter's own state to the accepted prefix. `None` is a
    /// drafter that keeps nothing across rounds.
    fn rollback(&mut self, ctx: &RoundCtx<'_>, v: &Verdict, out: RoundOutcome)
        -> Result<Option<CacheSpan>>;

    /// Condition the next round on what this one committed. `None` is a drafter
    /// that carries no conditioning buffer.
    fn condition(&mut self, ctx: &RoundCtx<'_>, v: &Verdict, out: RoundOutcome)
        -> Result<Option<Conditioning>>;
}
```

`Prefilled` carries the seed, this drafter's own `prefill_ns`, and whether the
drafter projects a conditioning buffer and reports the rows it accumulated — see
items 9 and 10 below for why the last two are the drafter's to state. `Verdict`
carries the accepted count, the tokens the round commits, and — for a
restricted-vocabulary drafter — how long this round's restricted prefix is.

```rust
pub(crate) struct RoundOutcome { pub emit: RoundEmit, pub verifier_target: i32 }
```

`RoundEmit` is what `round_common::emit_round_tokens` already returns: the
committed count and whether a token stopped the request. `verifier_target` is
where the loop's own rollback left the verifier's caches, and it is handed over
because the Gemma4 assistant reads it — its next round's shared K/V offset *is*
that target. It has one consumer and that consumer is `condition`, so `rollback`
takes the `RoundEmit` alone until a migration finds a second one. Without it in the signature the first migration would re-derive
`rollback_target_from_head` inside a drafter, which is the duplication this
campaign exists to remove. One producer, the loop; one consumer today, the
assistant's `condition`.

`RoundCtx` carries the verifier, its two cache stacks, the device, the request's
`VerifierDraw` and the charge decision. The draw is in it because every `verify`
draws the verifier's tokens through it and EAGLE-3 additionally reads
`draw.sampling()` to decide whether the restricted read-back may run at all;
without it in the context, the loop would hold the sampler and hand it to nobody.
`rollback` and `condition` take the context immutably, so only `prefill` and
`verify` can advance the verifier's caches or the draw.

### Why the fields are paired

Today seven `RoundReport` literals each state every field, which is what makes a
new field seven compile errors. One literal makes it one, and the value then has
nowhere to come from. Pairing restores the property by moving the per-drafter
half of the report into structs the *drafter* constructs:

```rust
pub(crate) struct CacheSpan { pub before: i32, pub target: i32 }
pub(crate) struct Conditioning { pub rows: Option<i32>, pub projected: Option<i32> }
```

A field added to either is a compile error in every drafter that builds one. A
field added to the loop's own half — the round index, the accept, the committed
count, the charge, the phases — is one compile error at one site, correctly, so
no drafter is asked for a fact it does not have.

`projected` is optional because the MTP sidecar reports a conditioning row and
no projection: it slices one verifier row per round rather than projecting one,
so `condition_rows` is `Some` and `projected_rows` is `None` on every line it
writes. A non-optional `projected` would put a number on the six pinned cells of the
recurrent pair, which carry none. `rows` is optional for a weaker reason: every loop reads it off
the buffer's shape and reports whatever that read returned, and making it
non-optional would introduce either an unwrap or a refusal on a path no cell
exercises.

Both structs are returned as `Option`, and the `None` arm is a word rather than a
literal of `None` fields. That is deliberate: a drafter that keeps no cache and
one that keeps one are two different statements, and only the second owes a
value when the struct grows. The cost is that a `None` drafter is silent about a
new field — which is right, since it has none — and the reviewer's job is to
check that a drafter answering `None` really carries nothing.

**Both structs land in migration chunk 2, not chunk 1.** The Gemma4 assistant
answers `None` to both, so introducing them with it would land two types with no
producer. Chunk 1 passes the report's per-drafter half as the `None`s the
assistant already writes; chunk 2 introduces the structs with the MTP sidecar,
their first producer, and re-keys the report onto them.

## What the interface cannot express, and what is proposed for it

Eleven things. Nine are expressible with no branch in the loop; one is a declared
per-loop skip; one is a decision the owner has to take, marked as such.

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
5. **The block figure the caller reads back, and the five seed exits.** The five
   sidecar loops return the widest block any round ran; the two two-model loops
   return the widest *proposal count*, one less, and
   `spec_generate_greedy` adds one back on the way out — the unit boundary is
   that one `map` at the end of the dispatcher, and the two-model per-loop entry
   is where the subtraction goes so the value the dispatcher receives is
   unchanged.

   **This figure is gated, on every arm and every prompt.** The equivalence
   harness asserts the driver's returned block against the block the pair runs
   at, for all six pairs, so it is not a figure the collapse may quietly move:
   it must be byte-identical per loop, and no pair reading is re-blessed for it.
   The server discards it, and
   `crates/rmlx-models/tests/qwen3_5_mtp_drafter_alignment.rs` reads it too, but
   the pairs are what pin it.

   What is *not* gated is the five sidecar **seed** exits, which return the
   resolved block rather than the widest that ran — a round has not run there, so
   the two differ, and no gate prompt stops on its seed. The two in-round EOS
   exits already return the widest that ran. Proposed: the loop returns the
   widest block that ran on every exit, which changes those five values alone.
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

   The refusal is a new runtime error path taken on every round, and its
   correctness rests on an invariant nothing asserts: that the verifier's
   post-forward offset is exactly the pre-round offset plus the positions the
   round fed. Six loops read that offset as a maximum across the stack precisely
   because a recurrent layer's `KvCache` never advances, and the assistant's
   stack mixes sliding-window rings with full-attention layers. The chunk that
   adds the refusal owes two controls: a positive one, that it fires on a planted
   off-by-one, and a negative one, that all seven loops pass 36 of 36 cells with
   it armed. Without both it is a new way for a correct round to fail.
7. **The exit that reports the verifier's resident KV.** Each loop has exactly
   one exit that calls `round_common::report_verifier_kv_bytes` and exactly one
   that does not, and *which* exit differs: the five sidecar loops return on the
   seed EOS before reaching the report and fall through to it after an in-round
   EOS; the two two-model loops do the opposite, having no seed. One loop has one
   exit shape.

   **Proposed: preserved exactly, as a declared per-loop skip.**
   `KV_REPORT_SKIPPED_BY` is the declaration, and the loop reads it at its two
   exits. It is the one per-drafter flag the loop branches on, and it is here
   rather than hidden because the alternative is a behaviour change the campaign
   is not for.

   The alternative is worth stating, and it is **not adopted**: the figure has
   one writer — a speculative request never goes through
   `Architecture::generate_greedy` — so a request that skips the report leaves
   the previous request's figure readable to a caller that samples around the
   call, and the two exits that skip it are the two that would read stale.
   Unifying them changes no baseline cell, no pair reading and no gate, which is
   exactly why it is not folded into a refactor whose contract is that nothing
   moves. It is a separate change, on its own evidence, after the collapse.

   The declaration is also what keeps the test below alive across the campaign:
   the seven-row table is re-keyed onto `KV_REPORT_SKIPPED_BY` in migration
   chunk 1, and each chunk drops its own file from the source scan in the same
   commit. Without the declaration the first migration would delete a loop the
   test still reads and leave nothing in its place.

   **A declaration needs two readers, and chunk 1 owes the second.** That each
   drafter declares an exit is one fact; that the loop honours what it declared
   is another, and a drafter can declare the exit the loop ignores. The second
   reader is the same marker reading, over `round_loop.rs`: its two exits, and
   the two arms the constant is read at. So the source scan gains that file in
   the very commit that drops the first migrated one, and its declared pattern is
   the loop's own, not a drafter's.
8. **What a round conditions on.** DFlash 1 conditions on the round's *committed*
   count and DFlash 2 on `accept + 1`, at the same two calls — the row count
   handed to `committed_rows` and the bound handed to `guard_round_conditioning`.
   The two agree except when the request's budget cuts the block, and neither
   derives from the other. The MTP sidecar advances its drafting position by the
   committed count for the same reason. The committed count is known only after
   the emission, which happens between `verify` and `condition`. Proposed: the
   loop's order is verify, emit, verifier rollback, drafter rollback, condition,
   and the `RoundEmit` the emission returned is passed to the last two.
9. **Whether a drafter projects a conditioning buffer, before its first round.**
   The request record's `conditioned_rows` is `Some(0)` on the seed-EOS record of
   DFlash 1 and DFlash 2 and `None` on the other five — a statement about the
   drafter, made before any round has run. Fourteen `RoundTotals` literals state
   it today, and one loop-built literal cannot know it. Proposed: `Prefilled`
   declares it, and the loop reads that declaration for both the seed record and
   the tail record.

   The discriminator is **"projects its conditioning and reports the rows it
   accumulated"**, not "holds a conditioning buffer". The MTP sidecar holds one —
   a single verifier row it slices per round — and reports `None`, because it
   projects nothing and accumulates nothing. A declaration keyed on holding a
   buffer would move the MTP sidecar's record, which is a cell that may not move.

   The declaration decides `Some` against `None` and nothing else: the value
   inside is an accumulator the two block loops add each round's projected rows
   to, under a `.max(0)` clamp, and stays the loop's own running figure.

   Without the declaration the two `Some(0)`s become `None` and nothing anywhere
   sees it: `RoundStats::conditioning_violation` is guarded on `None` and passes.
10. **What `prefill_ns` covers.** The three loops that prefill the prompt less
    its last token close the span before their round-0 carry forward; DFlash 2
    closes it after its whole-prompt capture, trim and projection but before the
    seed draw; EAGLE-3 closes it after the seed draw *and* after its drafter's
    own KV prefill, which conditions on the verifier's capture. Proposed:
    `prefill` returns its own `prefill_ns` and the loop does not time it. A loop
    that timed the call would move the figure on five of seven records and no
    gate would see it.
11. **What the empty-chain refusal says.** The two spellings are the same test:
    a two-model round's verifier carry is always one token, so `v_k < 2` holds
    exactly when the proposal chain is empty, and both families reach their check
    after every drafting forward the round takes. Neither family stops anywhere
    the other would not. What differs is the `Error::Model` message — seven
    texts, each naming its own loop and its own reason. Proposed: one refusal
    with one message, and the seven texts are the cost. They are not on any
    gate's path, so the disposition is recorded here rather than defended: the
    message that survives must still say that an empty chain is a broken drafter
    and not the end of the request, which is the part a reader acts on.

## Migration order

One drafter per chunk after this one, each its own PR, each proving its own pair
and a byte-identical round stream.

1. **The Gemma4 assistant.** Its pair is the only one that resolves both halves
   by slug and runs under `make gpu-test` on any machine holding the snapshots —
   every other pair needs an operator-named `RMLX_DRAFT_TEST_MODEL`, so this is
   the only first choice whose proof a reviewer can reproduce without being told
   which snapshot to fetch. It exercises the charged arm (`phases_charged()` and
   `RoundPhases`), the head-basis offset of item 6, the `None` recurrent stack,
   a drafter that keeps no cache of its own, and the sliding-window ring's
   rollback, and it is the one consumer of `RoundOutcome::verifier_target`. It
   answers `None` to both `rollback` and `condition`, so it proves the loop and
   the two optional arms and nothing about the two paired structs — which is why
   those wait for chunk 2. This chunk also carries two re-keys: the charge gate
   below, because a migrated entry leaves its derived population the moment its
   body goes; and the disposition test below, which moves onto
   `KV_REPORT_SKIPPED_BY` and drops `gemma4_assistant.rs` from its source scan in
   the same commit. Every later chunk drops its own file the same way, and the
   seven-row table is what does not change.
2. **The MTP sidecar.** The complement, and the first producer of `CacheSpan` and
   `Conditioning`: a recurrent refold, a drafter cache rolled back by offset, a
   single conditioning row with no projection beside it, and a
   `d_offset`/`d_target` pair on the round line.
3. **DFlash 2.** A conditioning window that slides, a prompt-window capture, the
   third charged loop — so the charged arm is exercised by chunk 1 and chunk 2
   before it reaches here — and the first drafter to declare a conditioning
   buffer before its first round.
4. **DFlash 1.** The adaptive block, the one loop whose verify width is not
   fixed, and the other half of item 8 — it conditions on the committed count
   where DFlash 2 conditions on the acceptance.
5. **EAGLE-3.** The restricted vocabulary, the attribution buffer, a drafter that
   rolls back by re-running, and the only `verify` that reads `draw.sampling()`
   to decide which of two read-backs it may take. Its `accept_and_reseed` is a
   rollback, a conditioning and a reseed in one call, so all of it goes in
   `rollback`, `condition` returns `None`, and the drafter-side target is read
   back off the cache afterwards rather than computed — which is what makes it
   the cross-check on the verifier's target that the round line calls it.
6. **The two-model greedy loop.** Two full models, a draft-side rollback with its
   own tape, no seed, and the resync that prepends the last draft token on a full
   accept. It is also where the raw `argmax` read-back becomes the context's
   `VerifierDraw`; the two are the same at temperature 0, and
   `the_two_model_round_loop_reproduces_plain_greedy` is what says so rather than
   the claim.
7. **The two-model stochastic loop.** Last, because it is the only acceptance
   rule that is not the shared one and the only loop with no equivalence pair —
   its gate is `crates/rmlx-models/tests/two_model_stochastic.rs`, which pins
   that one seed reproduces one sequence.

   It also closes the campaign in the docs, and names what it deletes: the
   `CLAUDE.md` documentation-map row and the paragraph in `docs/SPECULATIVE.md`
   both stop calling this file a proposal and describe the loop the tree has, the
   seventh loop body goes, the sampling gate's `spec_generate_greedy_cached`
   exception goes with chunk 6's, and the migration order above goes — a
   completed order is a stale plan, not a reference.

## The oracle

Five observables, and each is blind to something different.

- **The per-round event stream** against the checked-in baseline:
  `python3 scripts/spec_round_stream_compare.py verify <capture>`, 36 of 36 cells
  at every chunk, digests from
  `crates/rmlx-models/tests/fixtures/spec_round_baseline/MANIFEST.sha256`. It
  sees the round's shape — block, accept, proposals, committed rows, both cache
  spans, the conditioning pair, `refolded`. It is blind to the values inside a
  round: two runs agreeing on every count can be scoring different logits. It is
  blind to `charged`, which reads `false` on both sides by construction. It is
  blind to anything a loop does not report, the resident-KV figure included, and
  it sees no request that ran no round.
- **The request record**, one `done` line per request, assembled from the
  `RoundTotals` a loop closes on: `block_size`, `rounds`, `emitted`,
  `seed_emitted`, `emitted_in_rounds`, `conditioned_rows`, the accept counters,
  the four spans and `charged`. This is the only observable that covers a request
  which stopped on its seed, and the only one that carries `conditioned_rows`.
  Fourteen literals produce it today — five seed-EOS, two in-round EOS, seven
  tails — and collapsing them to one is owed to this campaign. It is blind to
  which round moved anything, and its `conditioning_violation` and
  `emission_violation` are guarded, so a figure that becomes `None` passes.
- **The equivalence pairs**, judged by the divergence-confidence oracle in
  `docs/SPEC_ANSWER_EQUIVALENCE.md` at the readings recorded there. They see the
  answer. They are blind to every sampled arm, blind to the stochastic loop
  entirely, and blind wherever a pair's drafter is not resolvable on the machine
  running them.
- **`make check-spec-charge`**, whose census must read `charge_phases:3 false:4`
  over seven charge sites at every step of the campaign — see the gate section.
  It sees a decision that moved. It cannot see whether the schedule it describes
  is the right one.
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
- Every field of the request record, per loop — `conditioned_rows` on the two
  seed-EOS records that carry `Some(0)` included.
- The exit behaviour of every loop: the seed-EOS exit that returns before any
  round, the in-round EOS exit that stops the request, the empty-chain refusal,
  and the token budget that caps the emission without capping the acceptance.
- The driver's returned block on every normal exit, which the equivalence
  harness asserts for every arm on every prompt.
- What the empty-chain refusal tells a reader: seven texts become one, and the
  one that survives still has to say that an empty chain is a broken drafter and
  not the end of the request.
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
| the rollback target off by one at the loop's call site | round stream `v_target`; the pairs (this is the "rejected tail never rolled off" broken engine, refused 6 of 6); and, proposed above, the loop's own head-against-tail refusal | stream and pairs yes, the refusal is new and owes two controls |
| a drafter's condition called on the wrong row | round stream `condition_rows` / `projected_rows`; `guard_round_conditioning` refuses a projection that does not match the commit | yes |
| a round conditioned on the acceptance where the loop conditions on the commit | round stream `projected_rows`, on a prompt whose budget cuts a block | yes, on that prompt only |
| a capture handed to the wrong drafter | the type system — each drafter takes its own capture inside its own `verify` | new, structural |
| the block narrowing dropped, or the adaptive schedule applied to the wrong drafter | round stream `num_draft` | yes |
| the recurrent tape not armed before the verify | `refold_lin_tapes` refuses a tape that does not describe the round | yes |
| a seed emitted where none was, or dropped | round stream `emitted_total` shifts on every line; `RoundStats::emission_violation` | yes |
| the restricted prefix reported for a loop that has no restricted vocabulary | the EAGLE-3 boundary rule in the equivalence gate reads `DecidedBy` per token | yes |
| the charge token hard-wired across the collapse | `make check-spec-charge`, re-keyed as below — the census on one reading, RULE 8 on the other | yes |
| the in-round EOS exit stops recording the request | round stream ends early; the request record is absent | yes |
| **the seed-EOS exit stops returning early** | nothing at runtime: no round runs, so no round line is written, and no gate prompt has an EOS seed | **new, none** |
| **the resident-KV report moved to another exit, or computed from the wrong caches** | nothing at runtime | **new, none** |
| **`conditioned_rows` on a seed-EOS record falls from `Some(0)` to `None`** | nothing: `conditioning_violation` is guarded on `None`, the round stream sees no request that ran no round, and no pair has an EOS seed | **new, none** |
| the block figure changes unit on a normal exit | the equivalence harness asserts the driver's returned block on every arm and every prompt | yes |
| **a seed exit returns the widest that ran instead of the resolved block** | nothing: no gate prompt stops on its seed | **new, none** |
| the empty-chain refusal lost | nothing at runtime — a drafter that proposes nothing makes the acceptance walk answer rather than refuse, and the round silently emits one token | **new, none** |
| `prefill_ns` recomposed by the loop | nothing: five of seven records would move and the figure has no bound to fail | **new, none** |

Two of those rows get a text-level catcher in this chunk, in
`crates/rmlx-models/src/speculative/round_skeleton_tests.rs`. What holds them is
a **seven-row table** naming, per loop, which exit skips the resident-KV report
and how the loop refuses an empty proposal chain; the source scan is a *reading*
of that table against today's tree, positional rather than by count, so a report
moved to the other exit fails rather than passing on an unchanged total. The
table is the part that survives the collapse: `KV_REPORT_SKIPPED_BY` is the same
statement in the engine, chunk 1 re-keys the test onto it, and each chunk drops
its own file from the scan.

Five markers, not four. The reading is of the early exit, the head of the round
loop, the empty-chain refusal, the tail request record and the resident-KV
report — measured, `EWGRP` for a loop that skips the report on its seed and
`WGRERP` for one that skips it in a round, because an in-round EOS exit writes
its own record before it returns. Four markers left an undeclared escape: a
report moved *into* the round loop after the refusal read `EWGP`, unchanged,
because there was no marker for where the loop ends.

**The residual, declared.** The reading is of statement order and not of
reachability. A report at the position the table names, inside a branch that
never runs, reads identical — and so does one whose arguments are wrong, which is
the row below.

### The mutation I could not catch

**The resident-KV report.** Nothing observable in this crate reads it: it is not
on the round line, the equivalence pairs do not touch it, the accept counters do
not carry it, and the value only becomes visible in an append-only metrics row
written by the server. A migration that keeps the call, at the right exit, and
reports the wrong caches — the drafter's instead of the verifier's, which is the
mistake the function's own documentation warns against — produces a plausible
figure, a green `make ci`, a green `make gpu-test` and 36 of 36 matching
round-stream cells. The table above pins where the call is, and cannot pin what
it reads.

The unification item 7 declines is itself in this class, which is the reason it
is written there as a change to take on its own evidence rather than folded into
a refactor.

Naming what would close it, since it is out of this chunk's scope: the figure
would need a second reader — a request-level assertion that the verifier's
resident bytes are non-zero after any request that ran a round, and that they
describe the verifier's stack rather than the drafter's. That is a test needing
two loaded models, so it belongs with the equivalence pairs under `make
gpu-test`, not on the CPU side.

## Gate changes the skeleton needs

### `make check-spec-charge`

Its population is derived as "carries the `step_fn` signature **and** constructs
a `RoundTotals`", and the census is built from that population alone. After the
first migration the migrated entry keeps its signature and loses its totals, so
it leaves the population, and the shared loop enters it carrying a token that is
neither `phases_charged()` nor a literal. Both readings of the naive re-key fail:

- Keep the loop in the census and the census reads `charge:1 charge_phases:2
  false:4` mid-campaign. RULE 3 fires on a correct tree.
- Drop the loop from the census and RULE 1 inside it compares a forwarded token
  against itself. A loop that writes `let charge = false;` at both its
  `rollback_round` site and its `RoundTotals` is green while the seven entries
  still spell 3:4 — which is the "charge token hard-wired" mutation this gate is
  the only reader of.

**The decision: the token travels as a value, and a new RULE 8 holds the
forwarded side, exactly as RULE 2 holds the deciding side.** This is what the
scanner does; the configuration type it resolves the forwarded parameter by is
named `RoundCfg`, and the engine owes it that name.

Three populations, all derived, none a name list:

- **(a) round loops** — the signature plus a `RoundTotals`, unchanged. Within it,
  a loop is **forwarded** when its *parameter list* carries the configuration
  type that holds the charge field, and **classic** otherwise.
- **(b) entries** — a fn carrying the driver signature that constructs no
  `RoundTotals`, calls no `rollback_round`, **and constructs the loop's
  configuration type**. Seven of them at the end of the campaign, each naming the
  decision once, at the call that runs the loop, in the drafter's own module.
  The last conjunct is the discriminator, not a detail: without it the population
  admits `spec_generate_greedy`, which validates a request and delegates, and the
  three emit helpers, none of which decides anything — and the rule below would
  report four fns as entries missing their decision, for ever. An entry that
  stops constructing the configuration leaves the population, which is what makes
  a dropped decision read as a lost entry rather than as a clean scan.
- **(c) drafter rollbacks** — a fn outside (a) that calls `rollback_round` and
  neither carries the driver signature nor names a `charged:` field of its own.
  Those two exclusions are what keep the old reading alive: a fn that drives a
  generation, or states a decision in a record, is a loop the derivation lost
  rather than a drafter rolling its own state back, and stays exit 2. Without
  them a lost loop would be read under RULE 4's argument rule and could pass.

**Membership is read off the signature; the census is keyed on the binding, and
the two must not be the same reading.** A loop that hard-wires its charge still
takes the configuration, so it is still forwarded and RULE 8 still reaches it;
what changes is its binding, which is what puts it back in the census. Defining
"forwarded" by the binding instead — the first shape this file carried — makes
RULE 8 unfireable: a loop failing it stops being forwarded and leaves the rule's
population rather than failing it, and only the census is left.

The rules over them:

- **RULE 1** reads a classic loop as today, and reads a forwarded loop the same
  way: one token at every `rollback_round` argument and every `charged:` field.
- **RULE 2** is unchanged and reads populations (a)-classic and (b): a
  `charge_phases` token binds to exactly `phases_charged()`.
- **RULE 3**, the census, is computed over **every charge token that is not the
  forwarded configuration field — whether named directly at its sites or bound to
  it in the same fn** — which is the classic loops' and population (b)'s: seven
  sites, `charge_phases:3 false:4`, at every step of the campaign. A forwarded
  loop that names or binds the field it was handed contributes nothing; the same
  loop hard-wiring a literal contributes one and the census reads eight.
- **RULE 4** gains population (c): a rollback outside a round loop that looks
  like neither a loop nor a record is no longer exit 2 outright, but its
  `charge` argument carries what the loop handed it. Where the fn is handed the
  round's context — a parameter whose declared type is `RoundCtx` — that is
  `<ctx>.charged` and nothing else; where it is not, any field of any of its own
  parameters will do. The strict arm closes the loose one's hole: a second
  parameter with a field of the same name satisfies "a field of one of its own
  parameters" while carrying a different decision, and nothing downstream can
  see which was read. A literal or a `phases_charged()` there is a second
  decision made where nothing can hold it to the loop that ordered it, and is
  exit 1; an argument the scan cannot read back stays exit 2.
- **RULE 5** keeps its needle and inverts its verdict for population (b) alone:
  a `charged:` in a fn with no driver signature is exit 1 exactly as now. Over
  (b) it becomes a rule rather than a membership test — an entry states exactly
  one `charged:`; zero or two is exit 1 naming the entry, and a field the scan
  cannot read back is exit 2. Folding the count into membership instead would
  put an entry with two decisions in no population at all, where no rule reaches
  it.
- **RULE 6** and **RULE 7** are untouched, provided the loop does **not** live in
  `crates/rmlx-models/src/speculative/round_common.rs` — that file's exemption
  for naming the low-level rollback is anchored to its path, and a loop moved
  into it would inherit an exemption it must not have. Hence `round_loop.rs`,
  under `crates/rmlx-models/src/speculative/`, which is where RULE 7's directory
  scan reaches.
- **RULE 8 (the forwarded decision is the one that was handed over).** A
  forwarded loop binds its charge token, in the same fn, to exactly the
  configuration field it was handed — the whole right-hand side, not a prefix —
  or names that field directly at every site. *Every* binding of that token must
  be that field, so a second one is exit 1 naming it. Anything else is exit 1.
  The parameter is resolved by its **declared type**, the configuration type, not
  by RULE 4's looser "a field of one of its own parameters": a loop that took a
  second parameter with a `charged` field could otherwise satisfy the rule from
  the wrong one.

  **And the configuration the loop was handed is read-only inside it.** RULE 8
  constrains the token; without this it constrains nothing, because the value can
  be moved instead. A `let` that rebinds the parameter's own name — the loop
  building its own configuration and forwarding faithfully from that — writes no
  `charged:` site and passes every reading. So does an assignment to the charge
  field of a `mut` parameter. Two more spell the same move: an assignment to
  the whole parameter through its reference, and a `mem::` call that swaps it
  out. All four are **exit 2**: the decision the entry made is no longer the
  decision the loop applies, and no reading downstream can see it.

The two readings are complementary and neither alone covers the hard-wire. A
forwarded loop that writes `let charge = false;` is caught **twice**: by RULE 8,
because the binding is not the configuration field, and by RULE 3, because a
token bound to something other than that field is in the census and the census
then reads eight sites. RULE 8 can be edited out of the script; the census
cannot, since it is what the gate exists to state. And the census alone names
only a count, where RULE 8 names the line.

**Two extractor changes went with it, not one.**

1. The token reader reported `?` for anything that was not a bare identifier, so
   `charged: cfg.charged` was unreadable and would have exited 2. It now reads a
   field access as a token — and **keeps the delimiter anchor it already had**:
   `<ident>.<ident>` followed by `,`, `}` or the end of the line, and nothing
   else. `cfg.charged()`, `cfg.charged.into()`, `cfg.charged as bool` and
   `!cfg.charged` stay `?` and stay exit 2. The anchor is what stops the reader
   turning a call, a cast or a negation into the field it resembles, which is
   precisely how a decision gets inverted under a spelling the census still
   counts as forwarded.
2. The binding reader records **per binding**, keyed to the loop's own charge
   token. It used to set one flag per function with a good binding outranking a
   bad one, so a shadow passed. **Measured on this tree, before the re-key**: a
   `mtp.rs` that binds
   `charge_phases` to `super::phases_charged()` and then rebinds it to `false`
   on the next line exits 0 with `OK: 7 speculative round loops … census
   charge_phases:3 false:4` — a loop that charges nothing, counted among the
   three that ask. That was a live hole on `main`, not a consequence of the
   re-key, and the re-key chunk closed it: the same edit now exits 2, naming the
   loop that binds its charge token more than once.

   Under the per-binding reading a token bound twice in one fn is **exit 1** for
   a forwarded loop, where RULE 8 has something exact to say about the second
   binding, and **exit 2** for a classic loop and **for an entry**, where the
   same-token reading has become vacuous: with a shadow, the token at the
   `rollback_round` argument and the token in the `charged:` field can be two
   different values under one spelling, and nothing in the scan can say which
   binding governs which site. The entry arm is not a corner: at the end state
   all seven decisions live in entries, so an unread shadow there is the whole
   census.

   **What counts as a readable binding**, since a reader that guesses is worse
   than one that refuses. The only readable form is a body-level
   `let <token> = <rhs>;` whose statement closes on its own line. `let mut`, a
   type annotation, a right-hand side spanning more than one line, a destructure
   in the body or in the parameter list, and a binding introduced by `if let` or
   `match` are each **exit 2** — a shape this gate cannot read, reported rather
   than skipped. A fn with zero readable bindings whose sites name a bare
   identifier is **exit 1**: the token means something the scan never saw.

   Two hazards go with that. A right-hand side long enough for rustfmt to break
   over two lines turns a readable binding into an unreadable one on a
   reformatting commit, so the configuration field's name has to stay short
   enough that it cannot happen — a constraint on the engine, not on the scanner.
   And a destructure of the configuration in the parameter list —
   `let RoundCfg { charged: charge, .. }` — registers a `charged:` site today,
   which the census would count as an eighth decision; the exit-2 disposition
   above is what keeps that from being a silent miscount.

#### The fixture cases the re-key passes

Twenty-three cases, each a scan root in `scripts/check_spec_charge_fixtures.sh`
built on one of two tree shapes the campaign passes through — mid-campaign, with
one drafter migrated, and the end state, with all seven decisions in entries.
They are stated here with the exit and the reason each produces. Two of them
assert two reasons on the one tree, and case 16 is run on both arms, so the
twenty-three are twenty-six runs of the harness, beside the thirty-eight the
gate already had and six more: a token whose sites name it and whose binding the
scan never saw, the three further ways a forwarded loop can move its
configuration rather than read it, a rollback that charges off the wrong
parameter, and a comment in a parameter list, which is prose.

Cases 1 and 12 ask for a reason the old success line did not carry: it printed
the loop count and the census and nothing about populations. The line now names
all three — `N classic, M forwarded, K entries; census … (N sites).` and the
forwarded loops by name — so a tree that passes says which shape it passed as,
and a fixture asserting that a loop is forwarded has a line to assert against.
On today's tree, which has no shared loop yet, it reads `7 classic, 0 forwarded,
0 entries; census charge_phases:3 false:4 (7 sites).`

| # | the tree | exit | the reason it must give |
|---|---|---|---|
| 1 | mid-campaign: one forwarded loop, six classic loops, one entry | 0 | seven charge sites, census `charge_phases:3 false:4`, one forwarded loop named |
| 2 | the forwarded loop writes `let charge = false;` at both sites | 1 | RULE 8, the binding is not the configuration field — and the census reads eight sites, both readings on one tree |
| 3 | the forwarded loop writes `let charge = cfg.charged \|\| x;` | 1 | RULE 8, the whole right-hand side |
| 4 | the forwarded loop names `cfg.charged` at both sites, no binding | 0 | forwarded, read as one token |
| 5 | one of the seven entries drops its `charged:` | 1 | both reasons on one tree: RULE 5's zero-arm names the entry, and the census reads six sites — excluding the loop must not hide a lost decision |
| 6 | an entry binds `charge_phases` to a literal | 1 | RULE 2, unchanged |
| 7 | an entry moves a token from `false` to `phases_charged()` | 1 | RULE 3, census `charge_phases:4 false:3` |
| 8 | a fn with no driver signature writes `charged:` | 1 | RULE 5, unchanged |
| 9 | a drafter `rollback()` passes `ctx.charged` | 0 | RULE 4's new arm |
| 10 | a drafter `rollback()` passes `false` | 1 | RULE 4, a second decision beside the loop's |
| 11 | a drafter `rollback()` passes an expression the scan cannot read | 2 | unreadable site |
| 12 | the end state: one forwarded loop, seven entries, no classic loop | 0 | seven charge sites, census `charge_phases:3 false:4` |
| 13 | the end state with the loop deleted | 2 | no round loop found; a scan that finds nothing must not pass |
| 14 | the forwarded loop names `rollback_round_caches` | 2 | RULE 6, unchanged |
| 15 | the forwarded loop calls `log_round` twice | 2 | RULE 7, unchanged |
| 16 | the forwarded loop shadows its binding — `let charge = cfg.charged;` then `let charge = false;` | 1 | RULE 8 names the second binding; the same shape on a classic loop is exit 2 |
| 17 | an entry states two `charged:` fields | 1 | RULE 5 over (b): an entry names one decision |
| 18 | an entry shadows its binding — `charge_phases` bound to `phases_charged()` then rebound to `false` | 2 | a shape this gate cannot read, the entry arm of case 16 |
| 19 | an entry stops constructing the configuration and calls the loop some other way | 1 | it leaves (b), so the census reads six sites — a dropped decision is a lost entry, not a clean scan |
| 20 | the forwarded loop builds its own configuration — `let cfg = RoundCfg::uncharged();` — and forwards from that | 2 | the configuration handed over is read-only inside the loop; a rebinding of its name writes no site any reading can see |
| 21 | the forwarded loop assigns `cfg.charged = false;` on a `mut` parameter | 2 | the same, by assignment rather than by rebinding |
| 22 | the forwarded loop writes `charged: cfg.charged()` | 2 | the delimiter anchor holds: a call is not the field it resembles |
| 23 | the forwarded loop writes `let mut charge = cfg.charged;` | 2 | not a readable binding form; a reader that guesses is worse than one that refuses |

### `make check-spec-sampling`

RULE 1's population was `pub fn` drivers, which found six fns and would have kept
finding the seven entries after the collapse — while the one loop that reads the
sampler is `pub(crate)` and would have been outside the gate entirely. Widening to
"any visibility" alone is not free: it enumerates `spec_generate_greedy_cached`, which
takes no sampler at all, and `emit_step`, `emit_round_tokens` and
`emit_seed_token`, which are helpers. All four fail condition (a) on the day the
rule changes.

It is widened to **any visibility, over the charge gate's populations (a) and (b)
and the fns that call one of them** — the one loop, the seven entries, and the
two-model entry guard that routes a request to one of two loops by reading
whether the sampler is active. That third clause is not tidiness: the guard is
where a sampled request can be routed to the greedy arm, which is this gate's
own defect class, and it is in the gate today only because it happens to be
`pub`. Narrowing to (a) alone is the
tempting move and it is wrong: at the end of the campaign the entries are the
only place a request's sampler can be dropped, since each takes `sampler_cfg`
and hands it to the loop's configuration, and a gate that stops reading them
loses its own defect class at the one site that can still commit it. That is the
rule this campaign already learned once — a re-keyed gate follows a defect class,
it does not shed one.

Over (a), condition (b) — the parameter is read, not merely declared — is
satisfied in the shared loop by the `VerifierDraw::new(...)` construction in its
body, and the needle is that construction rather than a mention of
`sampler_cfg`. Over (b) it is satisfied by the sampler reaching the loop call:
the needle is the configuration the entry builds carrying the sampler, not the
name appearing somewhere in the body. A sampler assigned into a `RoundCtx` field
and read by nobody is exactly the shape (b) exists to refuse, and so is a
`sampler_cfg` an entry accepts and leaves out of the configuration it hands over.

Two things the re-key had to settle, since the shared loop does not take a
sampler parameter of its own — it is handed one inside `RoundCfg`. Condition (a)
is met by either, and the draw needle reads a `VerifierDraw::new(...)` naming
`sampler_cfg`, so `VerifierDraw::new(sampler_cfg)` and
`VerifierDraw::new(cfg.sampler_cfg)` are one needle. **That names the
configuration's field**: it is `sampler_cfg`, the same constraint on the engine
that the charge field's short name is. And the two-model stochastic loop builds
no `VerifierDraw` at all — its acceptance rule is its own, and it seeds the whole
draw stream with `Pcg32::new(sampler_cfg.seed_or_default())`. That is still a
draw constructed from the request's sampler, so it is a second needle any loop
may satisfy rather than a second name the gate exempts.

The census belongs on the success line for the same reason the charge gate's
does: **one loop, seven entries, one guard**. What is *pinned* there is the one
figure that does not move across the campaign — the drafter paths, being the
loops that are not the shared one plus the entries, seven of them, because a
migrated drafter's loop body becomes its entry one for one. A lost entry is then
exit 2, and so is an eighth path: a scan that finds six drafter paths where the
tree has seven has not passed, it has stopped looking. The loop and guard counts
are printed beside it and not pinned, since the loop count is exactly what each
migration moves.

The three emit helpers are excluded without an exception: none constructs a
`RoundTotals`, none constructs the loop's configuration, and none calls a fn that
does — `emit_round_tokens` calls `emit_step`, which is in no population either. `spec_generate_greedy_cached` is
the one real exception: it is the two-model greedy loop, it takes no sampler at
all, and it runs only at temperature 0 where the verifier's argmax is the draw.
It is recorded with that reason, as a constant and not an environment knob, and
no caller is asked to pass it a sampler it does not take. The recall test holds
it in three directions — the loop it names passes, the same body renamed does
not, and a copy of the gate with the name struck out refuses the clean tree by
name — and the exception is deleted in migration chunk 6, where that loop
becomes a drafter whose `verify` draws through the context.

### `make debt-report`

Its driver group is discovered by the `step_fn` signature alone, so it already
lists `emit_step`, `emit_round_tokens` and `emit_seed_token` beside the loops —
eleven fns where the campaign measures seven. Narrowing it to the charge gate's
population (a) is an open item in its own right. At the end of the campaign the
group must list one driver plus the per-drafter `propose` / `condition` /
`rollback` implementations, whose pairwise similarity is what their real
differences warrant.

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
