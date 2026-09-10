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
    fn rollback(&mut self, ctx: &RoundCtx<'_>, v: &Verdict, emit: RoundEmit)
        -> Result<Option<CacheSpan>>;

    /// Condition the next round on what this one committed. `None` is a drafter
    /// that carries no conditioning buffer.
    fn condition(&mut self, ctx: &RoundCtx<'_>, v: &Verdict, emit: RoundEmit)
        -> Result<Option<Conditioning>>;
}
```

`Prefilled` carries the seed, this drafter's own `prefill_ns`, and whether the
drafter carries a conditioning buffer — see items 9 and 10 below for why the
last two are the drafter's to state. `Verdict` carries the accepted count, the
tokens the round commits, and — for a restricted-vocabulary drafter — how long
this round's restricted prefix is. `RoundEmit` is what
`round_common::emit_round_tokens` already returns: the committed count and
whether a token stopped the request.

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
writes. A non-optional `projected` would put a number on twelve pinned cells
that carry none. `rows` is optional for a weaker reason: every loop reads it off
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

Ten things. Seven are expressible with no branch in the loop; three are decisions
the owner has to take, marked as such.

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
5. **The block figure the caller reads back, and its two exits.** The five
   sidecar loops return the widest block any round ran; the two two-model loops
   return the widest *proposal count*, one less. Both families return something
   else again on the seed or in-round EOS exit — the *resolved* block, not the
   widest that ran, which contradicts every one of their doc comments. The server
   discards the figure for every loop; only
   `crates/rmlx-models/tests/qwen3_5_mtp_drafter_alignment.rs` reads it, for the
   MTP sidecar's normal exit. Proposed: the loop returns the widest block that
   ran on every exit including the EOS ones, and the two-model entry subtracts
   one, so the unit stays in the module that owns it. Changing the EOS exits'
   value is a change to a figure with one reader on one loop, and that reader
   does not exercise an EOS exit — which is why it is listed here rather than
   done quietly.
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

   **Proposed, and it is a behaviour change: every exit reports.** The figure has
   one writer — a speculative request never goes through
   `Architecture::generate_greedy` — so a request that skips it leaves the
   previous request's figure readable to a caller that samples around the call.
   The two exits that skip it today are the two that would read stale.

   This changes no baseline cell, no pair reading and no gate. That is exactly
   why it must be written down here rather than merged: see "the mutation I
   could not catch" below.
8. **What a round conditions on.** DFlash 1 conditions on the round's *committed*
   count and DFlash 2 on `accept + 1`, at the same two calls — the row count
   handed to `committed_rows` and the bound handed to `guard_round_conditioning`.
   The two agree except when the request's budget cuts the block, and neither
   derives from the other. The MTP sidecar advances its drafting position by the
   committed count for the same reason. The committed count is known only after
   the emission, which happens between `verify` and `condition`. Proposed: the
   loop's order is verify, emit, verifier rollback, drafter rollback, condition,
   and the `RoundEmit` the emission returned is passed to the last two.
9. **Whether a drafter carries a conditioning buffer, before its first round.**
   The request record's `conditioned_rows` is `Some(0)` on the seed-EOS record of
   DFlash 1 and DFlash 2 and `None` on the other five — a statement about the
   drafter, made before any round has run. Fourteen `RoundTotals` literals state
   it today, and one loop-built literal cannot know it. Proposed: `Prefilled`
   declares it, and the loop reads that declaration for both the seed record and
   the tail record. Without the declaration the two `Some(0)`s become `None` and
   nothing anywhere sees it: `RoundStats::conditioning_violation` is guarded on
   `None` and passes.
10. **What `prefill_ns` covers.** The three loops that prefill the prompt less
    its last token close the span before their round-0 carry forward; DFlash 2
    closes it after its whole-prompt capture, trim and projection but before the
    seed draw; EAGLE-3 closes it after the seed draw *and* after its drafter's
    own KV prefill, which conditions on the verifier's capture. Proposed:
    `prefill` returns its own `prefill_ns` and the loop does not time it. A loop
    that timed the call would move the figure on five of seven records and no
    gate would see it.

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
   rollback. It answers `None` to both `rollback` and `condition`, so it proves
   the loop and the two optional arms and nothing about the two paired structs —
   which is why those wait for chunk 2. This chunk also carries the gate re-key
   below: a migrated entry leaves the charge gate's derived population the moment
   its body goes.
2. **The MTP sidecar.** The complement, and the first producer of `CacheSpan` and
   `Conditioning`: a recurrent refold, a drafter cache rolled back by offset, a
   single conditioning row with no projection beside it, and a
   `d_offset`/`d_target` pair on the round line.
3. **DFlash 2.** A conditioning window that slides, a prompt-window capture, the
   second charged loop, and the first drafter to declare a conditioning buffer
   before its first round.
4. **DFlash 1.** The adaptive block, the one loop whose verify width is not
   fixed, and the other half of item 8 — it conditions on the committed count
   where DFlash 2 conditions on the acceptance.
5. **EAGLE-3.** The restricted vocabulary, the attribution buffer, a drafter that
   rolls back by re-running, and the only `verify` that reads `draw.sampling()`
   to decide which of two read-backs it may take.
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
| the charge token hard-wired across the collapse | `make check-spec-charge`, re-keyed as below — the census on one reading, RULE 8 on the other | needs the re-key |
| the in-round EOS exit stops recording the request | round stream ends early; the request record is absent | yes |
| **the seed-EOS exit stops returning early** | nothing at runtime: no round runs, so no round line is written, and no gate prompt has an EOS seed | **new, none** |
| **the resident-KV report moved to another exit, or computed from the wrong caches** | nothing at runtime | **new, none** |
| **`conditioned_rows` on a seed-EOS record falls from `Some(0)` to `None`** | nothing: `conditioning_violation` is guarded on `None`, the round stream sees no request that ran no round, and no pair has an EOS seed | **new, none** |
| **the block figure changes unit, or an EOS exit returns the widest instead of the resolved block** | nothing for six of the seven loops: the server discards it, and the one test that reads it reads the MTP sidecar's normal exit | **new, none** |
| the empty-chain refusal lost | nothing at runtime — a drafter that proposes nothing makes the acceptance walk answer rather than refuse, and the round silently emits one token | **new, none** |
| `prefill_ns` recomposed by the loop | nothing: five of seven records would move and the figure has no bound to fail | **new, none** |

Two of those rows get a text-level catcher in this chunk, in
`crates/rmlx-models/src/speculative/round_skeleton_tests.rs`. What holds them is
a **seven-row table** naming, per loop, which exit skips the resident-KV report
and how the loop refuses an empty proposal chain; the source scan is a *reading*
of that table against today's tree, positional rather than by count, so a report
moved to the other exit fails rather than passing on an unchanged total. The
table is the part that survives the collapse: when the disposition becomes a
per-drafter declaration in the engine, the test re-keys onto the declaration and
keeps the same seven rows.

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

The proposal in item 7 is itself in this class, which is the reason it is written
here as a decision to take rather than merged as a tidy-up.

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
forwarded side, exactly as RULE 2 holds the deciding side.**

Three populations, all derived, none a name list:

- **(a) round loops** — the signature plus a `RoundTotals`, unchanged. Within it,
  a loop whose charge token binds to a field of one of its own parameters is
  **forwarded**; every other member is **classic**.
- **(b) entries** — a fn carrying the driver signature that writes exactly one
  `charged:` field, constructs no `RoundTotals` and calls no `rollback_round`.
  Seven of them at the end of the campaign, each naming the decision once, at
  the call that runs the loop, in the drafter's own module.
- **(c) drafter rollbacks** — a fn outside (a) that calls `rollback_round`.

The rules over them:

- **RULE 1** reads a classic loop as today, and reads a forwarded loop the same
  way: one token at every `rollback_round` argument and every `charged:` field.
- **RULE 2** is unchanged and reads populations (a)-classic and (b): a
  `charge_phases` token binds to exactly `phases_charged()`.
- **RULE 3**, the census, is computed over **the classic loops' tokens and
  population (b)'s tokens** — seven sites, `charge_phases:3 false:4`, at every
  step of the campaign. A forwarded loop's token is not in it.
- **RULE 4** gains population (c): a rollback outside a round loop is no longer
  exit 2 outright, but its `charge` argument must be a field of one of its own
  parameters. A literal or a `phases_charged()` there is a second decision made
  where nothing can hold it to the loop that ordered it, and is exit 1; an
  argument the scan cannot read back stays exit 2.
- **RULE 5** keeps its needle and inverts its verdict for population (b) alone:
  a `charged:` in a fn with no driver signature is exit 1 exactly as now.
- **RULE 6** and **RULE 7** are untouched, provided the loop does **not** live in
  `crates/rmlx-models/src/speculative/round_common.rs` — that file's exemption
  for naming the low-level rollback is anchored to its path, and a loop moved
  into it would inherit an exemption it must not have. Hence `round_loop.rs`,
  under `crates/rmlx-models/src/speculative/`, which is where RULE 7's directory
  scan reaches.
- **RULE 8 (the forwarded decision is the one that was handed over).** A
  forwarded loop binds its charge token, in the same fn, to exactly the
  configuration field it was handed — the whole right-hand side, not a prefix —
  or names that field directly at every site. Anything else is exit 1.

The two readings are complementary, and the second is what makes the first
fail-closed: a loop that hard-wires stops binding to a parameter's field, so it
is no longer forwarded, so it is classic, so its token joins the census and the
census reads eight sites. RULE 8 fires beside it and names the binding. Neither
reading alone covers the case — RULE 8 could be edited out, and the census alone
cannot tell a forwarded token from a hard-wired one, since both are spelled by
the local's name.

One extractor change goes with it: the token reader reports `?` for anything that
is not a bare identifier, so `charged: cfg.charged` is unreadable today and would
exit 2. It must learn to read a field access as a token.

#### The fixture cases the re-key must pass

None of these can run against today's scanner: populations (b) and (c) and
RULE 8 do not exist in it, and a fixture root asserting them would fail
`make check-spec-charge-fixtures` today. They are stated here with the exit and
the reason each must produce, and the re-key chunk turns each into a scan root.

| # | the tree | exit | the reason it must give |
|---|---|---|---|
| 1 | mid-campaign: one forwarded loop, six classic loops, one entry | 0 | seven charge sites, census `charge_phases:3 false:4`, one forwarded loop named |
| 2 | the forwarded loop writes `let charge = false;` at both sites | 1 | RULE 8 names the binding; the census reads eight sites |
| 3 | the forwarded loop writes `let charge = cfg.charged \|\| x;` | 1 | RULE 8, the whole right-hand side |
| 4 | the forwarded loop names `cfg.charged` at both sites, no binding | 0 | forwarded, read as one token |
| 5 | one of the seven entries drops its `charged:` | 1 | census of six sites — excluding the loop must not hide a lost entry |
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

### `make check-spec-sampling`

RULE 1's population is `pub fn` drivers, which finds six fns today and will keep
finding the seven entries after the collapse — while the one loop that reads the
sampler is `pub(crate)` and outside the gate entirely. Widening to "any
visibility" alone is not free: it enumerates `spec_generate_greedy_cached`, which
takes no sampler at all, and `emit_step`, `emit_round_tokens` and
`emit_seed_token`, which are helpers. All four fail condition (a) on the day the
rule changes.

Widen it to **any visibility and constructs a `RoundTotals`** — the charge gate's
population (a). That admits the shared loop and excludes the three emit helpers
by construction. `spec_generate_greedy_cached` is then the one exception, and it
is a principled one: it is the two-model greedy loop, which runs only at
temperature 0 and reads the verifier's argmax directly. Record it as an exception
with that reason, and delete the exception in migration chunk 6, where that loop
becomes a drafter whose `verify` draws through the context.

Condition (b) — the parameter is read, not merely declared — is satisfied in the
shared loop by the `VerifierDraw::new(sampler_cfg)` construction in its body, and
the needle must be that construction rather than a mention of `sampler_cfg`. A
sampler assigned into a `RoundCtx` field and read by nobody is exactly the shape
(b) exists to refuse.

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
