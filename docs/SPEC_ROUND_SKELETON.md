# One speculative round loop: the interface

**Status: the collapse is done — migration chunks 1 to 7 have landed.** The loop
is `crates/rmlx-models/src/speculative/round_loop.rs` and every drafter runs on
it. This file is what a reviewer judged before the first drafter was migrated,
and what each migration chunk was held to afterwards. What each chunk landed
differently from the proposal below is listed under "What chunk 1 landed"
through "What chunk 7 landed".

Seven drafter paths in `crates/rmlx-models/src/speculative/` run one algorithm,
and each is an entry onto the one loop:

| drafter path | file | body |
|---|---|---|
| `mtp_generate` | `crates/rmlx-models/src/speculative/mtp.rs` | entry; `run_rounds` in `crates/rmlx-models/src/speculative/round_loop.rs` |
| `dflash_generate` | `crates/rmlx-models/src/speculative/dflash/round.rs` | entry; `run_rounds` in `crates/rmlx-models/src/speculative/round_loop.rs` |
| `dflash2_generate` | `crates/rmlx-models/src/speculative/dflash2/round.rs` | entry; `run_rounds` in `crates/rmlx-models/src/speculative/round_loop.rs` |
| `eagle3_generate` | `crates/rmlx-models/src/speculative/eagle3/round.rs` | entry; `run_rounds` in `crates/rmlx-models/src/speculative/round_loop.rs` |
| `mtp_assistant_generate` | `crates/rmlx-models/src/speculative/gemma4_assistant.rs` | entry; `run_rounds` in `crates/rmlx-models/src/speculative/round_loop.rs` |
| `spec_generate_greedy_cached` | `crates/rmlx-models/src/speculative/mod.rs`, drafter in `two_model.rs` | entry; `run_rounds` in `crates/rmlx-models/src/speculative/round_loop.rs` |
| `spec_generate_stochastic_cached` | `crates/rmlx-models/src/speculative/mod.rs`, drafter in `two_model.rs` | entry; `run_rounds` in `crates/rmlx-models/src/speculative/round_loop.rs` |

`spec_generate_greedy` in the same file is the two-model entry guard and
dispatcher, not a loop: it validates a request, resolves the draft count and
delegates.

## The shared skeleton, read off the seven bodies it replaced

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
and — for one drafter, under one of its two rules — what "the verifier accepted
this" means.

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

`RoundCtx` carries the verifier, its two cache stacks, the context ceiling they
were built at, the device, the request's `VerifierDraw` and the charge decision. The draw is in it because every `verify`
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

**Both structs landed in migration chunk 2, not chunk 1.** The Gemma4 assistant
answers `None` to both, so introducing them with it would have landed two types
with no producer. Chunk 1 passed the report's per-drafter half as the `None`s
the assistant already wrote; chunk 2 introduced the structs with the MTP
sidecar, their first producer, and re-keyed the report onto them.

## What chunk 1 landed

Eight differences from the proposal above. Seven are because the shape it
proposes has no producer yet and landing it would mean a type or a field nothing
writes; the eighth preserves a value on `main` the proposal would have moved.
They are the chunk-2 agenda as much as they are a record.

- **`Prefilled::seed` is a `u32`, not an `Option<u32>`.** The `None` arm is the
  two-model pair, which migrates in chunk 6. The cost of landing the `Option`
  early is not a missing round-0 carry — it is that the loop emits the seed
  unconditionally, through one `emit_seed_token` call that is also where the
  seed-EOS record is written, so a `None` would have to be given a token to emit
  and would emit an invented one. The arm arrives with the pair that needs it.

  **Closed in chunk 6, and not as an `Option`.** The two statements — which
  token the first round carries, and whether the request emits it — are one
  two-valued `Seed`, because a token beside a flag can be a flag about another
  token.
- **`RoundOutcome` is not a type.** Its `emit` half has no reader in chunk 1 —
  the assistant's `condition` reads the verifier target and nothing else — and a
  struct field no one reads is a `dead_code` warning, not a seam. `rollback` and
  `condition` take `verifier_target: i32` and the struct arrives with its second
  reader.
- **`Verdict` carries `verify_ns` and `walk_ns` and no `restricted`.** The
  verify forward and the acceptance walk are two spans of the round line, and
  the forward is inside `verify`, so the split is the drafter's to report.
  `restricted` is EAGLE-3's and arrives with it — together with the per-token
  attribution buffer of item 3, which the loop has no carrier for either. Those
  two are one chunk's work and neither has a producer before it.
- **`RoundDrafter` has a seventh method, `carry`, and it takes a callback.**
  `log_round` takes every array a round leaves for the next round's drafter,
  under the name that drafter calls it, and checks on a charged round that they
  are forced. Only the drafter knows what it carries. It hands the list to a
  `&mut dyn FnMut(&[(&str, &Array)])` rather than returning one: the consumer is
  `log_round`'s charged arm, which is off on the default path, so a returned
  collection would be allocated every round of every request and dropped unread.
  It returns a `Result` for the same reason the method exists — a drafter that
  lost its carry refuses, where an empty slice would pass the charged round's
  forcing check on exactly the state that check is for.
- **The loop always reports `RoundPhases`.** It times every phase it runs, so it
  has the figures on every request. Four loops report `None` today and will stop
  doing so as they migrate; the pinned round stream drops every `*_ms` field, so
  it is blind to this.
- **The verifier offset has one basis, the head spelling.** The assistant is the
  only loop on the shared body and reads its offset before the verify forward.
  The two-valued basis of item 6 arrives with the first tail-spelling drafter.
- **No head-against-tail refusal.** Item 6 proposes one and owes it two controls;
  chunk 1 neither adds it nor owes them.
- **The seed exit returns the resolved block, not the widest that ran.** This is
  the one deviation that is a preserved value rather than a missing producer:
  item 5 proposes the loop return the widest that ran on every exit, which would
  move the four remaining sidecar seed exits and this one. The shared loop keeps
  `main`'s value — no
  round has run at that exit, so `widest_bs` is zero there and returning it would
  report a block of nothing. Item 5 is qualified below accordingly: the change it
  proposes is still owed, and it is owed as its own change, because the row in
  the mutation table says nothing catches it. Migrating a drafter and moving that
  value in the same chunk would be a behaviour change riding inside a refactor
  whose contract is that nothing moves.

Two things chunk 1 does carry as proposed: `Prefilled` declares
`conditioned_rows` (the assistant answers `None`), and `KV_REPORT_SKIPPED_BY` is
read at both of the loop's exits, so the disposition test's seven-row table is
now a reading of the constant for the migrated loop and of the source for the
six that are not. The loop's own two guards on that constant are read by the
same marker sequence, which carries each guard's negation and not only the arm
it names — with the arm alone, dropping either `!` leaves every marker where the
table says it should be while the loop reports on the exit it declared it would
skip.

Two things stay outside the loop and neither is a deviation: the refusal of a
prompt under two tokens and the resolution of the request's block are both in
the drafter's entry, beside the verifier-pairing check. They are there because
they refuse before anything is built — no cache stack, no prefill, no round —
and because the loop is handed a block that is already resolved and a prompt it
may assume has a last token to seed from. The loop's module doc lists both among
what a request runs and names the entry as where they run.

## What chunk 2 landed

Seven notes. Two close deviations chunk 1 recorded, one records a new deviation
from the proposal above with its reason, one records what item 6 still owes, and
three are values the migration had to preserve or move.

- **`RoundOutcome` is a type now, and it carries the committed count rather
  than the `RoundEmit`.** The sidecar advances its drafting position by what the
  round committed, which is the second reader chunk 1 said the struct would
  arrive with. It carries `committed: usize` and not `emit: RoundEmit`, which is
  a deviation from the proposal: the loop breaks on `hit_eos` before it reaches
  either `rollback` or `condition`, so the flag is structurally `false` at the
  only sites a drafter could read it — the same objection that kept the struct
  out of chunk 1, one field down.
- **The verifier offset has the two-valued basis of item 6.**
  `RoundDrafter::VERIFIER_OFFSET_BASIS` is the declaration and the loop reads
  the offset before and after the forward. The sidecar counts its rollback back
  from the tail and reports the post-forward read; the assistant counts forward
  and reports the pre-forward one. They name the same position and are two
  numbers on the round line, and both are pinned. The rollback target is
  computed from the head spelling for both, which is what the target was on
  `main` for each.
- **No head-against-tail refusal, still.** Item 6 proposes one and owes it two
  controls. Chunk 2 is the chunk that makes it expressible — the loop now holds
  both reads — and it does not add it: the controls are a planted off-by-one and
  a full 36-cell capture with the refusal armed, and neither is this chunk's
  evidence.
- **`rollback` keeps its default and the assistant keeps answering it.** The
  assistant's only change is `condition`'s signature and its `Ok(None)`; it
  produces neither paired struct, which is what the reviewer checks against its
  claim to carry no cache.
- **The sidecar's dispositions are preserved as the spec states them**: the seed
  exit returns the resolved block and skips the resident-KV report
  (`KV_REPORT_SKIPPED_BY::TheSeedExit`); the request record's
  `conditioned_rows` stays `None`, because the sidecar slices a verifier row and
  projects nothing; its round line keeps `condition_rows: Some(1)` beside
  `projected_rows: None` and its `d_offset_before` / `d_target` pair; it charges
  its phases; and its recurrent refold runs through `rollback_round` with the
  recurrent stack, `refolded` read off the return.
- **Two figures moved and one log field went, and nothing reads any of them.**
  The sidecar's `rollback_ms` used to close before the conditioning slice and
  now closes after it, because the shared loop times its rollback across both
  drafter calls. Its `total_ms` now covers the drafter's cache reset, which used
  to happen before the request's own clock started and is now the first
  statement of `prefill`. The pinned stream drops every `*_ms` field and the
  request record's spans have no bound to fail. And the sidecar's starting
  `info!` is the loop's, so the capture layer it named is no longer on that
  line; it is a property of the verifier, not of the request.
- **The shared loop answers for two rows of the disposition table.** Its one
  empty-chain refusal stands for both, which is the `rows.min(1)` arm of
  `every_loop_refuses_a_drafter_that_proposed_nothing_by_its_declared_measure`
  taking its first second row. The sidecar's own refusal text is gone with its
  body — item 11's cost, paid a second time.

## What chunk 3 landed

Six notes. One is a field the interface gained, one records a deviation with its
reason, and four are values the migration had to preserve or move.

- **`Prefilled::projects_conditioning` is the declaration, exactly as item 9
  proposes, and the rows are the loop's alone.** The drafter answers a `bool`;
  the loop opens the count at zero for a `true` and adds each round's
  `Conditioning::projected` to it under the `.max(0)` clamp the body it replaced
  applied. An `Option<usize>` here would let a drafter supply an opening value,
  and a non-zero one inside a block passes `RoundStats::conditioning_violation`
  silently — the misreport class item 9 exists to prevent, re-admitted through
  the field meant to close it. A drafter answering `false` accumulates nothing,
  so the sidecar's and the assistant's records do not move.
- **The declaration got a second reader, in the loop.** A drafter that projects
  rows and declares it does not would leave the record reporting `None`, and
  `conditioning_violation` opens by returning on that `None` — so the bound over
  `emitted_in_rounds` is off for the whole request while every other observable
  reads clean. `run_rounds` refuses a round whose `Conditioning` carries a
  projection the request declared it would not make. It holds on this tree by
  construction: the sidecar reports `projected: None`, the assistant reports no
  `Conditioning` at all, and DFlash 2 reports `Some` under a `true`.
- **`RoundOutcome` carries the round's index.** `guard_round_conditioning` names
  the round in its refusal, and the alternative — each drafter counting its own
  rounds — is a second counter beside the loop's that can drift, with the only
  consequence being a reader sent to the wrong round and nothing to catch it. It
  is the loop's own half of the report, so it is one compile error at one site,
  which is the property the pairing exists to keep. DFlash 1 calls the same guard
  and reads the same field.
- **`CaptureTail` stayed drafter-internal and the loop learned nothing about
  it.** The bounded prompt-window capture is a parameter of
  `forward_verify_capture_chunked`, passed from inside `BlockRound::prefill` as
  the drafter's own `conditioning_rows`; no fact about it reaches `run_rounds`.
- **DFlash 2's dispositions are preserved as the spec states them**: the seed
  exit returns the resolved block and skips the resident-KV report
  (`KV_REPORT_SKIPPED_BY::TheSeedExit`); it counts its rollback back from the
  tail and reports the post-forward read
  (`VERIFIER_OFFSET_BASIS::AfterTheForward`) while the target is computed from
  the head spelling, which is the number `main` computed; the request record's
  `conditioned_rows` is `Some(0)` on the seed exit and `Some(sum)` on the tail;
  its round line keeps `condition_rows: Some(..)` beside `projected_rows:
  Some(..)` and no `d_offset_before` / `d_target`, the drafter keeping no cache
  and answering `rollback`'s default; and it charges its phases, the third loop
  to do so.
- **One figure moved and one log field went, and nothing reads either.** Its
  `rollback_ms` used to close before the conditioning slide and now closes after
  it, because the shared loop times its rollback across both drafter calls. And
  its starting `info!` is the loop's, so the capture's target layers and the
  opening conditioning row count are no longer on that line. The pinned stream
  drops every `*_ms` field and the request record's spans have no bound to fail.
- **A cell disposition the next chunk inherits, recorded rather than
  declared.** `run_rounds` always reports `RoundPhases`, and the four loops that
  still carry a body all report `phases: None`. So each of them gains five
  `*_ms` fields on its round line when it migrates, and reaches `log_round`'s
  overrun `error!` arm — the one that fires when a round's phase timers claim
  more time than the round has — for the first time. The pinned stream drops
  every `*_ms` field, so the cells do not move for it; what can move is that
  arm's own line, which carries no emitted total and so stays out of the stream
  by the same rule that keeps it out today. DFlash 1 is the first to meet it.
- **The shared loop answers for three rows of the disposition table**, which is
  the `rows.min(1)` arm of
  `every_loop_refuses_a_drafter_that_proposed_nothing_by_its_declared_measure`
  taking its third row. DFlash 2's own refusal text is gone with its body —
  item 11's cost, paid a third time.

## What chunk 4 landed

Six notes. One is the cell disposition chunk 3 recorded for this chunk, one
records a per-drafter fact that needed no interface growth, and four are values
the migration had to preserve or move.

- **The adaptive block goes through `block` and nothing else.** The schedule is
  `dflash_next_block_size`, unchanged and still pinned by
  `the_round_block_is_one_function_of_the_block_and_the_budget`; what moved is
  its caller. `AdaptiveRound::block` is the one override of the trait's default,
  the six other drafters keep `round_block`, and `run_rounds` carries no arm for
  it. The history the schedule reads — `(accepted, drafted)` per round — is the
  drafter's own field, written at the end of `verify` where the round's
  acceptance is known and read at the head of the next round, which is the same
  order the body it replaced wrote and read it in. No fact reaches `block` that
  the interface did not already hand it.
- **The cell disposition chunk 3 recorded, met.** DFlash 1 reported
  `phases: None` and now reports `RoundPhases`, so its round line gains five
  `*_ms` fields and reaches `log_round`'s overrun `error!` arm for the first
  time. The pinned stream drops every `*_ms` field, so its cells do not move;
  the overrun line carries no emitted total and stays out of the stream by the
  rule that keeps it out for the three loops already there. The charge decision
  is still DFlash 1's own and still `false` — the entry writes it into
  `RoundCfg` and the census reads `charge_phases:3 false:4` over seven sites.
- **DFlash 1's dispositions are preserved as the spec states them**: the seed
  exit returns the resolved block and skips the resident-KV report
  (`KV_REPORT_SKIPPED_BY::TheSeedExit`); it counts its rollback back from the
  tail and reports the post-forward read
  (`VERIFIER_OFFSET_BASIS::AfterTheForward`) while the target is computed from
  the head spelling, which is the number the body it replaced computed; the
  request record's `conditioned_rows` is `Some(0)` on the seed exit and
  `Some(sum)` on the tail, under `projects_conditioning: true`; its round line
  keeps `condition_rows: Some(..)` beside `projected_rows: Some(..)` and no
  `d_offset_before` / `d_target`, the drafter keeping no cache of its own and
  answering `rollback`'s default; and it charges no phase.
- **The conditioning buffer still grows and is still never trimmed.** DFlash 2
  slides a bounded window; this drafter's layers are full-attention and its
  block queries read the whole context unmasked, so `grow_conditioning` appends
  and drops nothing. The two are one method of the interface and two bodies of
  the drafter, which is the split item 2 predicted.
- **One span arrived, one log field went, and nothing reads either.** This loop
  reported `phases: None`, so it had no `rollback_ms` for the migration to move;
  what it has now is the shared loop's, which closes *after* the conditioning
  growth, so the growth is billed to the rollback span rather than left for the
  next round's drafter to pay for. Nothing charges it either way — this request
  is uncharged — and the pinned stream drops every `*_ms` field. And its
  starting `info!` is the loop's, so the capture's target layers are no longer
  on that line.
- **The two block drafters' shared statements are one item each, not a copy
  per drafter.** Migrating DFlash 1 onto the same interface DFlash 2 uses put
  two pairs of near-identical bodies in the tree, which is what the twin rule
  refuses: the residual probe each takes once per request, and the refusal a
  round that read its conditioning early raises. Both are now one fn in
  `crates/rmlx-models/src/speculative/mod.rs` —
  `report_conditioning_residual`, which slices the carried tail, measures it
  against a fresh projection and writes the line, and `missing_conditioning`,
  which builds the refusal. Each takes the loop as an argument, which was the
  whole of what differed. The MTP sidecar's own copy of the refusal went with
  them, so its message is the shared one; it is built on a path no passing
  request takes.
- **The shared loop answers for four rows of the disposition table**, which is
  the `rows.min(1)` arm of
  `every_loop_refuses_a_drafter_that_proposed_nothing_by_its_declared_measure`
  taking its fourth row. DFlash 1's own refusal text is gone with its body —
  item 11's cost, paid a fourth time. The drafter still returns an empty chain
  for a block of one, and the loop is what refuses it.

## What chunk 5 landed

Eight notes. Two are the interface growing by one field each, one records the
cell disposition chunk 3 recorded for every remaining loop, one records a
per-drafter fact the migration had to keep somewhere, and four are values the
migration had to preserve or move.

- **The attribution buffer is the loop's, and the prefix length is the
  round's.** `run_rounds` takes `Option<&mut Vec<DecidedBy>>` and hands it to
  `emit_round_tokens` with `Verdict::restricted` as the prefix, exactly as item
  3 proposes. Neither half could sit anywhere else: the buffer is written, and
  the configuration a loop is handed is `&RoundCfg` and read-only by the charge
  gate's own structural rule; the prefix is one entry per token the request
  *emitted*, and the request's budget can cut a round's commit below its
  acceptance, so only the emission knows how many entries to write. The seed's
  `DecidedBy::FullVocab` is now the loop's general rule rather than EAGLE-3's
  statement, and `DecidedBy` itself moved to `round_loop.rs` beside its only
  producer — `eagle3` re-exports it, because it is in `eagle3_generate`'s public
  signature. The four drafters that score every position over the verifier's
  whole vocabulary write `restricted: 0` and their entries pass `None`, which is
  the pairing property working: a field added to the drafter's half is a compile
  error at every drafter.
- **`RoundCtx` carries the context ceiling.** `verifier_cache_stack` already
  resolved it and the loop dropped it; EAGLE-3 sizes its own KV cache from it, so
  its drafter cannot overflow before the verifier does. The alternative is the
  entry resolving the ceiling a second time through
  `crate::context::resolve_context`, which is the same refusal raised twice on
  one request and a second producer of a number that has one.
- **`RoundCtx` did not need widening on `rollback`.** `accept_and_reseed` runs a
  drafter forward and advances the *drafter's* cache, which the drafter owns; it
  reads the verifier immutably and touches neither the verifier's caches nor the
  draw. So `rollback` and `condition` still take `&RoundCtx<'_>`, and the one
  signature widening chunk 4 budgeted for was not spent.
- **The read-back is decided once per request, in `prefill`, and the branch
  that runs yields the length.** The reduced argmax is over a subset of the
  verifier's row and a distribution needs the whole row's normalising constant,
  so a sampled request cannot take it: the decision is
  `self.drafter.hot_path_active() && !ctx.draw.sampling()`. It is taken in
  `prefill` rather than in `verify` because it is a *request*-level fact — both
  terms are constant for the run — and because the loop's own per-request line
  has to report it (below). The drafter still reads the context's draw and never
  builds one or reads `sampler_cfg`, which is the rule the sampling gate holds.
  The branch in `verify` then returns its own flag beside the tokens, which
  closes the *swap*: reporting the flag beside the branch instead left an
  exchange of the two arms invisible to every observable at temperature 0 while
  the boundary waived what it must refuse. It does not close the **flip** — the
  two arm literals are free constants, so a `false` written `true` reports a
  read-back its own arm did not take, and `restricted == accept` passes the
  acceptance bound while every text reading of the branch still holds. What
  closes that is the request's own declaration, one frame up: the loop hands
  `Prefilled::restricted_read_back` to `guard_restricted_prefix` beside the
  round's prefix, and a request that declared no reduced read-back is refused
  the moment a round reports one. The decision, the branch and the two literals
  are all read by text in
  `crates/rmlx-models/src/speculative/eagle3/round_tests.rs` as well, because no
  pair in the tree executes the full-vocabulary arm — every EAGLE-3 pair runs at
  temperature 0 with the reduced ids present — so the runtime guard has nothing
  here to fire on and the text is what sees the flip in `make ci`.
- **Whether a request took a reduced read-back is a field of the loop's own
  line.** `Prefilled::restricted_read_back` is the declaration and the shared
  `info!` carries it, so the fact survives as a structured field of the run's
  `.jsonl` for every loop rather than for the one that happens to log it. It was
  a field of EAGLE-3's own starting `info!` before the migration and the shared
  line had nothing for it, which left it recoverable from no log at all — a
  traceability regression the migration would have shipped silently. A `debug!`
  from `prefill` would have covered the same fact and re-introduced the
  per-drafter starting line chunks 2 to 4 deleted.
- **The reduced prefix is bounded, and the bound is the acceptance.**
  `guard_restricted_prefix` refuses a round whose `Verdict::restricted` reaches
  past its acceptance: the correction is the verifier's own token over its whole
  vocabulary, and attributing it to a reduced one makes the declared boundary
  waive the position it exists to judge. The bound is not the committed count —
  a round the budget cut commits fewer tokens than it accepted and its prefix
  stays where it was. `round_skeleton_tests.rs` reads it on the CPU.
- **An obligation for chunk 6.** The seed's `DecidedBy::FullVocab` is pushed on
  every request that carries a buffer, and item 3's rule is "a loop with an
  attribution buffer **and a seed**". The two are the same thing only while
  `Prefilled::seed` is a `u32`; when chunk 6 makes it an `Option` for the
  two-model pair, the push moves inside the `Some` arm beside
  `emit_seed_token`, or a pair that emits no seed attributes one it never
  emitted.
- **The cell disposition chunk 3 recorded, met a second time.** EAGLE-3 reported
  `phases: None` and now reports `RoundPhases`, so its round line gains five
  `*_ms` fields and reaches `log_round`'s overrun `error!` arm for the first
  time. The pinned stream drops every `*_ms` field, so its cells do not move; the
  overrun line carries no emitted total and stays out of the stream by the rule
  that keeps it out for the four loops already there. The charge decision is
  still EAGLE-3's own and still `false` — the entry writes it into `RoundCfg` and
  the census reads `charge_phases:3 false:4` over seven sites.
- **EAGLE-3's dispositions are preserved as the spec states them**: the seed
  exit returns the resolved block and skips the resident-KV report
  (`KV_REPORT_SKIPPED_BY::TheSeedExit`); it counts its rollback back from the
  tail and reports the post-forward read
  (`VERIFIER_OFFSET_BASIS::AfterTheForward`) while the target is computed from
  the head spelling, which is the number the body it replaced computed; the
  request record's `conditioned_rows` stays `None` under
  `projects_conditioning: false`, because the drafter holds a KV cache and one
  hidden row and projects nothing; its round line keeps `condition_rows: None`
  beside `projected_rows: None` and its `d_offset_before` / `d_target` pair,
  where the target is read back off the drafter's cache after the re-run rather
  than computed — the cross-check EAGLE-3's migration was chosen to land; and it
  charges no phase.
- **Two figures moved and one log field went, and nothing reads any of them.**
  The drafter cache offset the round opens at used to be read before the draft
  span opened and is now the first statement of `propose`, so a getter is inside
  that span; and `rollback_ms` is the shared loop's, closing after
  `accept_and_reseed` rather than before it — which is where the re-run was
  billed before, since this loop reported no phases at all. And its starting
  `info!` is the loop's, so the aux layer ids, the draft vocabulary size and
  whether this request took the restricted read-back are no longer on that line;
  the last of those is on the round's own `Verdict` instead. The pinned stream
  drops every `*_ms` field and the request record's spans have no bound to fail.
- **The step trace keeps its own three counters, and they are the one thing this
  chunk duplicated.** The per-position trace names the round index and a
  cumulative accept rate, and it is written in `verify`, where the loop's
  counters are not in scope. The drafter therefore counts its own rounds,
  proposals and acceptances for it. They cannot drift — `verify` runs once per
  round and after `propose`, so a round the loop counted is a round `verify`
  counted — and nothing but the trace reads them. The alternative was moving the
  trace into `rollback`, which a round that stopped on an EOS never reaches, so
  the last round of a request would stop tracing.
- **The shared loop answers for five rows of the disposition table**, which is
  the `rows.min(1)` arm of
  `every_loop_refuses_a_drafter_that_proposed_nothing_by_its_declared_measure`
  taking its fifth row. EAGLE-3's own refusal text is gone with its body — item
  11's cost, paid a fifth time.

## What chunk 6 landed

Nine notes. Two are the interface changing shape, two close deviations earlier
chunks recorded, one is a declaration that moved and three are values the
migration had to preserve or move — and one is a gate that lost its only name.

- **`Prefilled::seed` is a two-valued `Seed` and not an `Option<u32>`.** A round
  always opens on a token; what differs is whether the request is entitled to
  emit it. `Seed::Emitted(t)` is a token the prefill forward drew past the whole
  prompt, `Seed::Carried(t)` is the prompt's own last token that the pair's
  prefill stopped short of. An `Option` beside a separate carry would let a
  drafter state the two of different tokens, and the loop would have no way to
  tell. The loop reads the carry out of either arm and the `Emitted` arm holds
  all three of the seed's statements: the `DecidedBy::FullVocab` push chunk 5
  named as this chunk's obligation, the `emit_seed_token` call, and the seed-EOS
  return that reads `KV_REPORT_SKIPPED_BY`. The five migrated drafters re-signed
  one field each; `run_rounds` carries no drafter-specific branch for it.
- **`RoundCtx` carries the KV codec.** `verifier_cache_stack` already resolved
  it and the loop only logged it; the two-model drafter builds a second cache
  stack for the draft model and it must be the verifier's codec at the
  verifier's ceiling, because the verifier owns a pair's KV geometry. The
  alternative is the drafter resolving `kv_quant_override` against
  `DEFAULT_KV_QUANT` a second time, which is a second producer of a number that
  has one — the same argument that put `max_seq` there in chunk 5.
- **The block figure keeps its unit, and the subtraction is the entry's.** The
  loop returns the widest block that ran, as it does for the other six; the
  two-model entry returns `drafts_per_round` of it, and
  `spec_generate_greedy`'s one `map` adds the verifier's own token back. The
  value the dispatcher receives is unchanged, which is what the equivalence
  harness asserts on every arm and every prompt. `drafts_per_round` is the
  existing one producer of that boundary, not a second spelling of it.
- **`rollback` did not need `&mut RoundCtx`, for the second chunk running.** The
  draft model, its cache stack and its recurrent state are the drafter's own
  fields; the rollback reads the verifier not at all and the round's context only
  for the device and the charge decision. So `rollback` and `condition` still
  take `&RoundCtx<'_>`, and the one signature widening chunk 4 budgeted for is
  still unspent.
- **The full-accept resync is in `rollback`, and the drafter-side target is read
  back.** The resync — the last proposal fed ahead of the correction on a round
  that accepted every proposal — is a function of *this* round's acceptance, so
  `propose`, which runs before it, cannot state it without keeping a second copy
  of a fact the round already has. The span's `target` is read off the draft
  cache after the rollback rather than computed from the retention, which is the
  cross-check the round line calls for, and the same shape EAGLE-3 landed in
  chunk 5.
- **The empty-chain refusal moved spelling, and the row moved with it.** The
  loop refuses `draft_tokens.is_empty()`; this pair's own body refused `v_k < 2`
  one statement later, on the verifier input the empty chain produced. Item 11
  says the two are the same test — a two-model round's carry is always one token
  — so the `DISPOSITIONS` row is now `ChainRefusedBy::TheProposalChain` and the
  `rows.min(1)` arm takes its sixth row. No request stops anywhere it did not
  stop before; what goes is the sixth of the seven texts, item 11's cost paid a
  sixth time.
- **The two-model dispositions are preserved as the spec states them**: the
  in-round EOS exit writes its own record and returns before the resident-KV
  report (`KV_REPORT_SKIPPED_BY::TheInRoundExit`, the first producer of that arm
  — its `dead_code` allowance is gone); it counts its rollback back from the tail
  and reports the post-forward read
  (`VERIFIER_OFFSET_BASIS::AfterTheForward`) while the target is computed from
  the head spelling, which is the number the body it replaced computed; the
  request record's `conditioned_rows` stays `None` under
  `projects_conditioning: false`; its round line keeps `condition_rows: None`
  beside `projected_rows: None` and its `d_offset_before` / `d_target` pair; and
  it charges no phase — the entry writes `charged: false` into `RoundCfg` and the
  census reads `charge_phases:3 false:4` over seven sites, now `1 classic, 1
  forwarded, 6 entries`.
- **Two spans moved and one log field went, and nothing reads any of them.** The
  request's own clock used to start before the verifier's cache stack was built
  and now starts after it, because the shared loop opens `t_total` there; and the
  draft model's cache stack is allocated inside `prefill`, before that drafter's
  own `prefill_ns` opens, which is where the body it replaced allocated it
  relative to its own span. Its starting `info!` is the loop's, so `max_seq` and
  the draft count `k` are no longer on that line — the block is, and it is `k`
  plus the verifier's own token. And this loop reported `phases: None`, so its
  round line gains five `*_ms` fields and reaches `log_round`'s overrun `error!`
  arm for the first time: the chunk-3 cell disposition, met a third time. The
  pinned stream drops every `*_ms` field.
- **The round's carry has one producer, and it is the loop.** `RoundOutcome`
  carries it. The loop takes the verifier's own token at the accepted position
  off the commit, and a drafter that opens its next round on that token reads
  `outcome.carry` rather than reading the same `Verdict` for itself. Two
  producers agree until one changes, and then the drafting pass opens on a token
  the verify input does not carry — which moves the accept rate and nothing
  else, so no equivalence pair and no pinned cell can see it. The two-model
  drafter's `propose` therefore ignores the `carry` it is handed: the token
  reaches its draft model inside the seed the last round's resync built, which is
  two tokens on a full acceptance and one otherwise. EAGLE-3 held the same second
  producer — the identical arithmetic off the identical `Verdict`, for the
  correction its re-run conditions on — and reads `outcome.carry` as of this
  chunk; its `carry_tok` stays, because the per-position trace names the token
  the round *opened* on and that is `propose`'s parameter, not the next round's
  carry. Both drafters pin the read by text and refuse any line of their own that
  reads the round's commit.
- **`draft_ns` widened, as it did for the five before it.** The span is the
  whole `propose` call now — the tape arming, the drafting forwards, the fed
  buffer and the two scalars the round keeps — where the body it replaced timed
  `draft_decode_n` alone. Nothing bounds it: the pinned stream drops every `*_ms`
  field and the request record's spans have no bound to fail.
- **`make check-spec-sampling` names no fn any more.** Its one recorded
  exception was `spec_generate_greedy_cached` taking no sampler at all; that path
  is now an entry that carries the request's sampler in the `RoundCfg` it builds,
  and the gate reads seven of seven without it. The exemption, the `exempt` arm
  of its call scan and the clause on its success line are deleted. Its recall
  test loses the three-direction exemption reading — the name passes, the body
  renamed does not, a copy with the name struck out refuses the clean tree — and
  gains two readings of that same path held to both of RULE 1's conditions like
  every other, taken against a third synthetic root that carries the shape the
  tree now has — that path as an *entry* — rather than against a loop shape it
  no longer wears: the sampler parameter stripped, and the sampler taken and
  left out of the `RoundCfg` it hands over. 27 cases become 29 over four roots —
  the fourth being the tree's own census, one shared loop and one acceptance rule
  with a body of its own beside six entries, which the suite could not state
  before — and no case edits the gate any more: the suite's one self-mutation was
  of the exemption, and it went with it.
- **The shared loop answers for six rows of the disposition table**, which is the
  `rows.min(1)` arm of
  `every_loop_refuses_a_drafter_that_proposed_nothing_by_its_declared_measure`
  taking its sixth row, and the first time that arm folds two *spellings* rather
  than two rows of one spelling.

## What chunk 7 landed

Nine notes. Two are the interface reached rather than grown — the last migration
added no field and no method — one is a rule that moved into a drafter, one is a
helper the migration orphaned, one records what the campaign leaves owed, and
four are values the migration had to preserve or move.

- **The acceptance rule is a field of the one two-model drafter, not a second
  drafter.** `two_model::Acceptance` is two-valued: `Prefix` is the greedy walk
  and `Stochastic(q)` is Leviathan's test, carrying the distributions this
  round's proposals were drawn from — the state only that rule has, in the arm
  that has it. `propose` and `verify` are the two methods that read it and
  nothing else in the drafter branches: the prefill, the tape, the rollback, the
  resync and the round's carry are the same statements either way, which is what
  item 1 predicted when it said the rule is a second implementation of one
  method rather than a second drafter. `run_rounds` gained no arm.
- **`Verdict` expressed the stochastic round without growing.** "The accepted
  prefix plus one resampled correction, or a bonus" is `accept` and `commit`
  exactly as the greedy walk fills them: the correction and the bonus are both
  "the one token the verifier stands behind", which is the last element of
  `commit` under either rule. The one growth the chunk was allowed was not
  spent.
- **One RNG stream per request, and `VerifierDraw` is where it lives.** The
  deleted body seeded a `Pcg32` from `sampler_cfg.seed_or_default()` and drew
  its proposals, its acceptance coins and its corrections from it;
  `VerifierDraw` seeds one from the same value for every request on the shared
  loop. A drafter seeding a second from the same seed would be two correlated
  streams, each reading as reproducible on its own — so the draw is where the
  stream stays and the drafter draws through it: `block_distributions` for the
  verifier's own distributions, `proposal` for a drafted token and the
  distribution it came from, and `rng` for the coins the rule tosses itself. The
  rule is the drafter's because it is the drafter's; the stream is the
  request's. `block_tokens` and `block_distributions` share one row slice, so
  the two read-backs cannot drift into slicing differently.
- **The draw order did not move, and it was read rather than argued.** The
  stream advances in the same order it did in the body — `n` proposal draws,
  then one coin per proposal, then one residual draw on the round that rejected
  or one bonus draw on the round that did not — off a generator seeded from the
  same value. That is the argument. It is not evidence, and nothing already in
  the tree could be: `crates/rmlx-models/tests/two_model_stochastic.rs` asserts
  self-consistency *within* a build, which a reordered stream satisfies exactly
  as well; the pinned round stream and the equivalence pairs run at temperature
  0, where the verifier's tokens come off an argmax that reaches no draw; and
  `crates/rmlx-models/tests/spec_sampled_distribution.rs` drives a sidecar pair.

  **The reading is the same seed at two commits.** `origin/main` and this
  chunk's head, each built in its own worktree with its own `CARGO_TARGET_DIR`,
  and one harness copied unchanged into both. Eight cells per side: two prompts
  — one that stops on an EOS near 50 tokens, one that runs its whole budget —
  two temperatures, 0.7 and 1.0, and two seeds, 7 and 8, at 256 tokens on the
  `gemma-4-e4b` / `gemma-4-e2b` pair. **Eight of eight identical, id for id and
  text for text**, under one sha256 over each side's cell lines — the same
  digest on both. Re-read at every later revision of the chunk, the
  `std::mem::take` in `verify` included, with the same result.

  The control is now
  `print_the_seeded_stochastic_streams_for_a_cross_commit_diff`, beside the gate
  it complements, and that file's doc carries the recipe — so the next change to
  this stream has a control to run rather than an argument to make. Nothing in
  `two_model_stochastic.rs` was re-blessed: it pins no literal sequence.
- **`rollback_target_from_tail` is deleted.** Its last caller was the deleted
  body; every drafter on the shared loop has its target computed from the head
  spelling. The equality the two spellings had — the post-forward read names the
  same position — is the invariant a round line reporting
  `VERIFIER_OFFSET_BASIS::AfterTheForward` rests on, and it is stated on
  `rollback_target_from_head` and driven by
  `the_rollback_target_retains_the_carry_and_the_accepted_prefix`, which writes
  the tail arithmetic out rather than calling a helper kept for one test to
  compare against itself. `mod.rs`'s re-export of `RoundTotals` went the same
  way.
- **The two-model dispositions are preserved for the second path as the spec
  states them**: the in-round EOS exit writes its own record and returns before
  the resident-KV report (`KV_REPORT_SKIPPED_BY::TheInRoundExit`); the round line
  reports the post-forward read (`VERIFIER_OFFSET_BASIS::AfterTheForward`) while
  the target is computed from the head spelling, which is the number the body
  computed from the tail; `conditioned_rows` stays `None` under
  `projects_conditioning: false`; the round line keeps `condition_rows: None`
  beside `projected_rows: None` and its `d_offset_before` / `d_target` pair; the
  block figure goes back through `drafts_per_round` at the entry; the empty
  chain is refused by `ChainRefusedBy::TheProposalChain`; and it charges no
  phase — the entry writes `charged: false` into `RoundCfg` and the census reads
  `charge_phases:3 false:4` over seven sites, now `0 classic, 1 forwarded, 7
  entries`.
- **Two figures moved and one log field went, and nothing reads any of them.**
  The request's own clock used to start before the verifier's cache stack was
  built and now starts after it; the draft model's cache stack is allocated
  inside `prefill`. Its `verify_ns` used to cover the forward and the whole
  block of post-sampling distributions built off it and still does — the
  read-back is inside each rule's own arm, so the span means the same thing
  under both — where the acceptance test, which the body billed to nothing, is
  now `walk_ns`. Its starting `info!` is the loop's, so `k`, `max_seq`, the
  temperature's three filters and the request's seed are no longer on that line;
  the temperature still is. And this loop reported `phases: None`, so its round
  line gains five `*_ms` fields and reaches `log_round`'s overrun `error!` arm
  for the first time: the chunk-3 cell disposition, met a fourth time. The
  pinned stream drops every `*_ms` field.
- **The two entries are the closest thing this campaign leaves to a twin, and
  the gate is why.** `spec_generate_greedy_cached` and
  `spec_generate_stochastic_cached` are fifteen shared lines around three
  statements each makes alone: its `SpecLoop`, its `Acceptance`, and its own
  refusal text. Folding them into one entry with a parameter is what the twin
  rule would ask for, and it is refused here for a stated reason: the charge
  census and the sampling gate both pin **seven drafter paths**, one entry per
  path, and a fold would leave six entries and a wrapper in no population at
  all. `each_two_model_entry_names_its_own_rule_its_own_loop_kind_and_its_own_block`
  in `crates/rmlx-models/src/speculative/tests.rs` is what holds the three
  statements apart, which is the risk a near-copy carries.
- **The shared loop answers for all seven rows of the disposition table**, which
  is the `rows.min(1)` arm of
  `every_loop_refuses_a_drafter_that_proposed_nothing_by_its_declared_measure`
  taking its seventh row and the last. `LOOP_SOURCES` holds `round_loop.rs`
  alone, `ChainRefusedBy` has one arm — the `v_k < 2` spelling went with the
  body that stated it, and `TheVerifierInput` with the spelling — and
  `expected_pattern` went with the last row that named a file of its own. The
  seventh `Error::Model` text is gone: item 11's cost, paid the seventh and last
  time.

**What the campaign leaves owed.** Item 5's change — the seed exit returning the
widest block that ran rather than the resolved block — is **not** taken here and
is owed as its own change, on its own evidence. It applies to exactly one site:
the `return Ok((emitted, cfg.block_size))` inside `run_rounds`'s emitted-seed
arm. The mutation table's row for it still says nothing at runtime catches the
difference, which is why it is not folded into a refactor whose contract is that
nothing moves.

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

   **Landed in chunk 7, one step short of the proposal.** The rule is a
   `verify` arm, as proposed, but it is an arm of the *same drafter* rather than
   a seventh one: `two_model::Acceptance` is a two-valued field of
   `TwoModelRound`, because everything either rule does outside `propose` and
   `verify` — the prefill, the tape, the rollback, the resync, the carry — is
   the same statement. `accept_prefix` is the shared body six drafters delegate
   to, `stochastic_prefix` is the seventh path's own, and `run_rounds` carries
   no arm for either.
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

   **Landed in chunk 5, as proposed.** The buffer is a `run_rounds` parameter
   rather than a `RoundCfg` field, because the configuration is handed over by
   `&RoundCfg` and the buffer is written; and the prefix length is a `Verdict`
   field rather than something `verify` maps for itself, because the attribution
   is one entry per token the request *emitted* and the request's budget can cut
   a round's commit — so the length has to reach the emission, which is the
   loop's. The entry clears the buffer, not the loop, and it clears it before
   its own refusals — which is what its public signature already promised, and
   what the body it replaced did.
4. **DFlash 1's adaptive block.** `dflash_next_block_size` opens with
   `round_block` and then moves the result by the accept rate of the recent
   rounds. Proposed: the `block` method, six defaults and one override.

   **Landed in chunk 4, as proposed and with no new argument.** The schedule's
   history — `(accepted, drafted)` per round — is the drafter's own state,
   written in `verify` where the round's acceptance is known and read by
   `block` at the head of the next round, so the two arguments the method
   already takes are the two the schedule needs.
5. **The block figure the caller reads back, and the seed exit.** The five
   sidecar loops return the widest block any round ran; the two two-model loops
   return the widest *proposal count*, one less, and
   `spec_generate_greedy` adds one back on the way out — the unit boundary is
   that one `map` at the end of the dispatcher, and the two-model per-loop entry
   is where the subtraction goes so the value the dispatcher receives is
   unchanged.

   **Landed in chunk 6, as stated.** The loop returns the widest block that ran
   for all six drafters on it; the two-model entry returns `drafts_per_round` of
   that figure, and the dispatcher's one `map` adds the verifier's own token
   back, so the value it receives is byte-identical.

   **This figure is gated, on every arm and every prompt.** The equivalence
   harness asserts the driver's returned block against the block the pair runs
   at, for all six pairs, so it is not a figure the collapse may quietly move:
   it must be byte-identical per loop, and no pair reading is re-blessed for it.
   The server discards it, and
   `crates/rmlx-models/tests/qwen3_5_mtp_drafter_alignment.rs` reads it too, but
   the pairs are what pin it.

   What is *not* gated is the **seed** exit. There is one of them now, in
   `run_rounds`, and it serves the five drafters that draw a seed out of their
   prefill forward; it returns the resolved block rather than the widest that
   ran, because no round has run there and `widest_bs` is zero. No gate prompt
   stops on its seed, so nothing sees the difference. The in-round EOS exit is
   one exit too, shared now by all seven drafters, and it already returns the
   widest that ran. Proposed: the loop returns the widest block that ran on
   every exit, which changes that one value on those five drafters alone.

   **Still owed at the end of the campaign, and now a one-site change.** The
   value is the `return Ok((emitted, cfg.block_size))` in `run_rounds`'s
   emitted-seed arm. Whoever takes it takes it on its own evidence.

   **Not adopted in chunk 1, and it is the eighth deviation above.** The shared
   loop returns the resolved block on its seed exit, exactly as the assistant's
   own body did. Taking the proposal instead would have this refactor move a
   value nothing catches, in a chunk whose contract is that nothing moves; and
   the value it would move to is zero, since `widest_bs` counts rounds that ran
   and none has. Whoever takes it takes it on its own evidence, and the row in
   the mutation table is what says there is none to be had at runtime.
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

   **The basis landed in chunk 2 and the refusal did not.** The loop reads the
   offset before and after the forward, computes the target from the head
   spelling and reports whichever read the drafter declares in
   `VERIFIER_OFFSET_BASIS`. What is still owed is the refusal below.

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
   exits. It was the only per-drafter flag the loop branches on until chunk 2
   added item 6's `VERIFIER_OFFSET_BASIS`, and the two are the whole set: both
   are here rather than hidden because the alternative in each case is a
   behaviour change the campaign is not for. A third needs the same argument
   made for it from scratch.

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

   **A declaration needs two readers, and chunk 1 built the second.** That each
   drafter declares an exit is one fact; that the loop honours what it declared
   is another, and a drafter can declare the exit the loop ignores. The second
   reader is the same marker reading, over `round_loop.rs`: its two exits, and
   the two guards the constant is read at. So the source scan gains that file in
   the very commit that drops the first migrated one, and its declared pattern is
   the loop's own, not a drafter's — `SPEWGRIP`, where `S` and `I` are the two
   guards.

   **Each of those two markers is the whole guard, negation included.** The arm
   name alone is not a reader of the declaration: it is present whether the guard
   says `!matches!(…)` or `matches!(…)`, so dropping either `!` inverts which
   exit reports while every marker stays exactly where the table says it should
   be, and the constant assertion, the round stream, the pairs and both text
   gates all stay green. Reading the condition catches that and the arm swap
   alike; the cost is two long needles, and a rename that makes rustfmt wrap one
   of them fails the test rather than quietly passing it.
8. **What a round conditions on.** DFlash 1 conditions on the round's *committed*
   count and DFlash 2 on `accept + 1`, at the same two calls — the row count
   handed to `committed_rows` and the bound handed to `guard_round_conditioning`.
   The two agree except when the request's budget cuts the block, and neither
   derives from the other. The MTP sidecar advances its drafting position by the
   committed count for the same reason. The committed count is known only after
   the emission, which happens between `verify` and `condition`. Proposed: the
   loop's order is verify, emit, verifier rollback, drafter rollback, condition,
   and the `RoundEmit` the emission returned is passed to the last two.

   **Both halves are landed as of chunk 4, and neither needed an argument the
   other did not.** The order is the loop's and `RoundOutcome` is what the last
   two receive: DFlash 1's `condition` reads `outcome.committed`, DFlash 2's
   reads `verdict.accept + 1`, and both hand their figure to `committed_rows`
   and to `guard_round_conditioning` in the same two calls.
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

   **Landed in chunk 3, with the opening value beside the declaration.**
   `Prefilled::conditioned_rows` is the count before any round has run —
   `Some(0)` for DFlash 2, `None` for a drafter that projects nothing — and the
   loop adds each round's `Conditioning::projected` to it under the clamp. One
   reading serves both records and no drafter is branched on.
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

    **All seven are gone as of chunk 7.** Chunk 6 was the first to fold the two
    *spellings* rather than two loops of one, moving the two-model greedy pair's
    row in `DISPOSITIONS` from `ChainRefusedBy::TheVerifierInput` to
    `ChainRefusedBy::TheProposalChain`; chunk 7 moved the last row the same way
    and the `v_k < 2` spelling left the tree with the body that stated it. The
    enum has one arm, which is the claim that no request lost its refusal: the
    shared loop states one, before the verify forward, where both spellings
    always fired on the same round.

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
  answer. They are blind to every sampled arm, blind to the stochastic path
  entirely — which has no temperature-0 comparand and so no pair at all — and
  blind wherever a pair's drafter is not resolvable on the machine running them.
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
| a drafter's `CacheSpan::before` reported from the wrong read | round stream `d_offset_before` only — the truncation is driven by the span's `target`, computed on the drafter's own path, so a wrong `before` changes what the line says and not what the round does | yes, the pinned cells alone |
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
| **a seed attributed to the restricted vocabulary instead of the whole one** | nothing: the seed is the verifier's own argmax on both arms, so no pair parts at it and `Restriction` never reads `decided_by[0]`; the buffer's length is unmoved and the round stream carries no attribution. Measured on the restricted-vocabulary pair — it passes and 6 of 6 cells are identical | **new, none** |
| **`projects_conditioning` declared by a drafter that projects nothing** | nothing: the record's `conditioned_rows` moves from `None` to `Some(0)`, `conditioning_violation` passes on both, the loop's own refusal fires only the other way, and the round stream carries no request-level field. Measured the same way, same result. The inverse of the `Some(0)` row above | **new, none** |

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

**Chunk 2 closed one residual rather than declaring it.** Arming the recurrent
round tape was a statement inside each loop's own body, and the first migration
carried it into `SidecarRound::verify` — where moving it anywhere before the
forward is invisible to every observable, and where four more drafters would
have copied the line. It is now the loop's: `run_rounds` arms the tape
immediately before the `verify` call, the loop owning the recurrent stack and
being the tape's only consumer through its own rollback. A stack with no
recurrent layer arms nothing, so the drafters on a full-attention verifier are
unaffected.

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
  out. All four are **exit 2**. But the spellings are unbounded — a method on
  the parameter, a helper it is passed to, an assignment rustfmt wrapped — so
  the rule is on the parameter and not on the lines: **the configuration is
  handed over by `&RoundCfg`, and a loop taking it by `&mut` or by value is
  exit 2.** A loop that cannot write it cannot move it, whatever the spelling.
  The four line shapes stay as defence in depth. In each the decision the entry
  made is no longer the decision the loop applies, and no reading downstream can
  see it.

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
twenty-three are twenty-six runs of the harness. The suite is 76 runs: those
twenty-six, the thirty-eight the gate already had, and twelve more that review
and the first migration found — a token whose sites name it and whose binding
the scan never saw, three further ways a forwarded loop can move its
configuration rather than read it, a
configuration handed over by value, a write still read on an immutable
parameter, a rollback that charges off the wrong parameter, a comment in a
parameter list, which is prose, a low-level rollback behind a comment carrying a
brace, run against its own control, a trait of bodiless declarations above the
loop, which must not swallow it, and an entry with no forwarded loop to enter.

Cases 1 and 12 ask for a reason the old success line did not carry: it printed
the loop count and the census and nothing about populations. The line now names
all three — `N classic, M forwarded, K entries; census … (N sites).` and the
forwarded loops by name — so a tree that passes says which shape it passed as,
and a fixture asserting that a loop is forwarded has a line to assert against.
On the tree the campaign ends at it reads `0 classic, 1 forwarded, 7 entries;
census charge_phases:3 false:4 (7 sites).`

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
two-model entry guard that routes a request to one of two entries by reading
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
that the charge field's short name is. And the two-model stochastic loop built
no `VerifierDraw` at all while it had a body — its acceptance rule is its own,
and it seeded the whole draw stream with
`Pcg32::new(sampler_cfg.seed_or_default())` — so the re-key read that
construction as a second needle any loop could satisfy rather than as a name the
gate exempted.

**Chunk 7 deleted the second needle with the body.** That rule draws its coins
and its proposals through the round's `VerifierDraw` now, which seeds one
generator from the same value, so the tree has one draw construction and the
`Pcg32::new(` needle matched nothing. A needle that matches nothing cannot fire,
and what this one would admit is exactly what the migration removed: a loop
seeding a second generator from the request's seed, which is a second stream
correlated with the first and reads as reproducible on its own. Case 29 of the
recall suite asserts that shape is now refused by RULE 1(b).

**And a second generator beside a correct draw is not RULE 1(b)'s to catch.** A
loop that builds `VerifierDraw::new(sampler_cfg)` *and* seeds a `Pcg32` of its
own passes this gate, because the draw it must construct is there. What reads
that is `the_requests_draw_stream_has_one_generator` in
`crates/rmlx-models/src/speculative/round_skeleton_tests.rs`, a CPU scan of every
non-test source under `crates/rmlx-models/src/speculative/` for a `Pcg32::new(`
outside `VerifierDraw::new` — one producer of the request's stream, read over the
whole module family rather than over the one drafter that happens to own a rule.

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
does — `emit_round_tokens` calls `emit_step`, which is in no population either.

`spec_generate_greedy_cached` was the one real exception while it was the
two-model greedy loop, took no sampler at all and ran only at temperature 0 where
the verifier's argmax is the draw. **It is gone as of chunk 6**, with the loop
that justified it: that path is an entry now, it carries the request's sampler
in the `RoundCfg` it builds, and the gate reads seven of seven. The gate names no
fn at all, which is the state to keep it in — a name-shaped hole is somewhere to
hide. The recall test's three-direction reading of the exemption went with it,
and two readings of that same path took its place, one per condition of RULE 1.

### `make debt-report`

Its driver group is discovered by the `step_fn` signature alone, so it lists
`emit_step`, `emit_round_tokens` and `emit_seed_token` beside the loops and the
entries. Narrowing it to the charge gate's population (a) is an open item in its
own right, and it is the one gate change the campaign leaves unmade: at the end
of the campaign the group should list one driver plus the per-drafter `propose`
/ `condition` / `rollback` implementations, whose pairwise similarity is what
their real differences warrant, and it still lists every fn carrying the
signature.

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

Chunk 1 deleted the first body and the figure fell for the first time. Chunk 2
deleted the second, and the population falls with it: the bodies the campaign
measures are the loops that still carry one plus `run_rounds`, so the pair count
goes 21 → 15 and six pairs disappear with the sidecar rather than shrinking.
Measured with the same extractor over the same convention at both ends — summed
`difflib` matching-block lines over digit-folded bodies, `autojunk=False` —
`origin/main` reads 1841 matched lines over 2253 body lines and 21 pairs, and
this chunk's head reads 1180 over 1965 and 15. Each chunk's own convention has
to be stated with its two readings: the absolute figure is not comparable across
executors, only its endpoints against each other.

Chunk 3 deleted the third, and the pair count goes 15 → 10. Measured with that
same extractor and convention at both ends of this chunk, `origin/main` reads
1182 matched lines over 1969 body lines and 15 pairs, and this chunk's head
reads 718 over 1635 and 10.

Chunk 4 deleted the fourth, and the pair count goes 10 → 6. Measured with that
same extractor at both ends of this chunk, over the four remaining bodies plus
`run_rounds` and then the three plus it, `origin/main` reads 719 matched lines
over 1650 body lines and 10 pairs, and this chunk's head reads 421 over 1355
and 6. This executor's convention differs from chunk 3's by a constant — the
counts here are taken over the bodies `debt_report.py` extracts for fns
carrying both the driver signature and a `RoundTotals`, which is the charge
gate's population (a) — and the endpoints are what the chunk is judged on.

That population holds no `RoundDrafter` impl, so it cannot see duplication the
migration moves *into* a drafter. A second figure covers that, over the four
`impl RoundDrafter` bodies and by the same measure: before the twin removal 435
matched lines over 675 body lines and 6 pairs, after it 431 over 670. The
refusal is outside both figures — it sits in each drafter's inherent impl — and
is held by review.

Chunk 5 deleted the fifth body, and the pair count goes 6 → 3. Measured with
that same extractor at both ends of this chunk, over the three remaining bodies
plus `run_rounds` and then the two plus it, `origin/main` reads 416 matched
lines over 1355 body lines and 6 pairs, and this chunk's head reads 257 over 931
and 3. This executor's convention differs from chunk 4's by a constant — 416
against its 421 at the same tree — and the endpoints are what the chunk is
judged on.

The second figure rises, and the rise is accounted for line by line rather than
waived. Over the `impl RoundDrafter` bodies it reads 430 matched lines over 670
body lines and 6 pairs at `origin/main`, and 653 over 863 and 10 at this chunk's
head. A fifth impl enters the population and brings four new pairs with it —
EAGLE-3 against each of the four, 52, 53, 51 and 55 matched lines, the lowest
ratios in the table at 27% to 32%, so there is no twin here to fold out. The six
pairs that exist at both ends read 430 → 442, and the twelve lines are
`Verdict::restricted` and its one comment, stated by each of the four drafters
that score every position over the whole vocabulary: two lines per body, six
pairs. That is the mechanism "Why the fields are paired" above states — a field
every drafter must state is what makes a new field a compile error at every
drafter, and it costs an identical line in each — and not duplication this
migration moved.

Chunk 6 deleted the sixth body, and the pair count goes 3 → 1. Measured with
that same extractor at both ends of this chunk, over the two remaining bodies
plus `run_rounds` and then the one plus it, `origin/main` reads 262 matched
lines over 940 body lines and 3 pairs, and this chunk's head reads 10 over 615
and 1. This executor's convention differs from chunk 5's by a constant — 262
against its 257 at the same tree — and the endpoints are what the chunk is
judged on. The one pair left is `run_rounds` against the stochastic loop, at
3.3%: the acceptance rule item 1 says cannot share a body.

The second figure rises, and the rise is the sixth impl and nothing else. Over
the `impl RoundDrafter` bodies it reads 639 matched lines over 862 body lines
and 10 pairs at `origin/main`, and 932 over 1030 and 15 at this chunk's head. The
ten pairs that exist at both ends read **639 → 639**, unchanged line for line, so
the whole of the 293-line rise is the five new pairs the two-model impl brings:
against DFlash 1, DFlash 2, EAGLE-3, the assistant and the sidecar at 59, 53, 66,
49 and 66 matched lines, 38.3%, 28.5%, 37.1%, 29.5% and 39.5% — the middle of a
table whose top is DFlash 1 against the sidecar at 54.9%, a pair that predates
this chunk. No twin entered, and none of the six drafters that were already there
gained a line.

Chunk 7 deleted the last body, and the pair count goes 1 → 0. Measured with that
same extractor at both ends of this chunk, over the one remaining body plus
`run_rounds` and then `run_rounds` alone, `origin/main` reads 10 matched lines
over 616 body lines and 1 pair, and this chunk's head reads **0 over 282 and 0**.
There is one body and no pair, which is what the campaign was for. The figure
cannot fall further and this measurement retires with it: from here the
population is one fn, and what a future chunk moves is measured over the
drafters instead.

The second figure **falls**, for the first time in the campaign, and the fall is
the same 37 lines the two-model impl grew by. Over the six `impl RoundDrafter`
bodies it reads 956 matched lines over 1045 body lines and 15 pairs at
`origin/main`, and 919 over 1082 and 15 at this chunk's head. The ten pairs that
do not involve the two-model drafter read **663 → 663**, unchanged line for
line; its own five pairs read 293 → 256, each falling — against DFlash 1,
DFlash 2, EAGLE-3, the assistant and the sidecar at 42, 49, 60, 47 and 58
matched lines where they were 61, 55, 63, 51 and 63, and 23.9%, 23.6%, 30.0%,
25.0% and 30.7% where they were 38.7%, 29.0%, 34.7%, 30.1% and 37.0%. That is
what a rule no other drafter has looks like when it lands inside one that
already existed: the body grows and its similarity to every other body falls.
