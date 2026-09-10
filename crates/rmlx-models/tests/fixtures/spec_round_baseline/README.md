# The speculative per-round event stream, pinned

`MANIFEST.sha256` is one line per (pair, prompt) cell of
`crates/rmlx-models/tests/spec_greedy_equivalence.rs`:

```
<sha256>  <rounds>  <test>.<prompt>.stable.jsonl
```

## What it pins, and what it does not

Each round loop closes every round with one `tracing` event. The manifest pins
the **whole sequence of those events** for one run of each of the six pairs
over each of the six prompts — 36 cells, 3862 rounds — in the timing-free form
the run writes beside the full stream: the same JSON objects with every
wall-clock `*_ms` field dropped, because those move between two runs of one
engine and a digest over them compares two machines' load.

A digest pins bytes and says nothing about what they are, so the script reads
every line before it hashes one: a round event carries the round's index, what
it accepted, how many proposals it accepted them from and how many tokens the
request has emitted by the end of it, and carries no wall-clock field. A file of
well-formed JSON that is not a round stream is refused rather than digested.

The emitted total is the fourth for a reason worth stating: a round whose phase
timers do not partition it emits an `error!` beside its own line, and that
report names the round's index, its accepted count and its proposals so a reader
can act on it. Three fields would let that report into the stream, moving the
cell and doubling one round for everything that counts them. It carries no
running emitted total, and that is what keeps it out.

This is the observable the equivalence pairs cannot supply. They read the
answer, and greedy verification emits the verifier's own argmax at every
position whatever the drafter proposed — so a rollback off by one, a block
narrowed one token short or a conditioning row projected twice can change the
round stream and leave the answer alone. The accept counters read the aggregate
and are blind to which round moved. This is what sees it.

**It does not pin the phase schedule.** Every capture runs with `charged=false`,
because a capture that enabled `rmlx::spec::phase` at TRACE would be measuring a
different, slower run. `make check-spec-charge` is what covers that instead.

**It is one run, not a distribution.** Two runs of one engine agree here
exactly — measured — so a cell that moved is a change, not noise. A cell that
moved for a legitimate reason is re-blessed by regenerating the line and saying
in the commit message which loop changed and why.

## Reproducing a cell

One pair at a time, with the machine to itself. Snapshots resolve by slug from
`RMLX_O_MODELS_ROOT` (see `docs/TESTING.md`); every pair but the assistant one
also needs `RMLX_DRAFT_TEST_MODEL` set, which is what selects it.

```sh
RMLX_HOME=<a scratch dir> \
RMLX_O_MODELS_ROOT=<models root> \
RMLX_DRAFT_TEST_MODEL=<models root>/<the pair's drafter slug> \
  cargo test -p rmlx-models --test spec_greedy_equivalence -- \
    --ignored --exact <test name> --nocapture --test-threads=1
```

The run writes both files per cell under `<RMLX_HOME>/tmp/` and prints each
path with its round count and the timing-free file's byte length. Then:

```sh
python3 scripts/spec_round_stream_compare.py verify <RMLX_HOME>/tmp
```

A capture holding fewer than 36 cells — one pair is six — reports the cells that
did not run and exits **3**, `INCOMPLETE`: a cell that could not run is not a
cell that agreed, and 0 would say it was. A cell that ran and disagrees is exit
1, naming the cell and both digests. Exit 2 is a comparison that could not be
made at all — a missing directory, or a file still carrying a wall-clock field.

To compare two captures field by field rather than by digest — which is what
says *what* moved:

```sh
python3 scripts/spec_round_stream_compare.py compare <dir-a> <dir-b>
```

A change that renames, adds or drops a field makes two captures incomparable
line for line while the round's arithmetic is still comparable. `--fields` names
what to hold them to; an entry spelled `a|b` reads the first of the two a line
carries, which is how a field that was renamed is still compared:

```sh
python3 scripts/spec_round_stream_compare.py compare <dir-a> <dir-b> \
  --fields round,accept,num_draft,n_committed|emitted_round
```

A field present on one side and not the other is a difference; a field on
neither is not a common field of that cell and is skipped.

What `--fields` costs is the key-set check. A digest and a plain `compare` read
the whole line, so a field added, dropped or moved is a difference; under
`--fields` anything outside the list is not read at all — `target` and `message`
included, which is how a loop that moved its round event to another target reads
as identical. Every run therefore prints the fields it did not cover, and a list
naming a field no line in either capture carries is exit 2 rather than a pass
over one fewer field than was asked for. The cell sets, the round counts and the
completeness statement are read the same way.

## Regenerating

```sh
python3 scripts/spec_round_stream_compare.py manifest <RMLX_HOME>/tmp \
  > crates/rmlx-models/tests/fixtures/spec_round_baseline/MANIFEST.sha256
```

Only from a capture of all six pairs, one pair at a time, on an idle machine —
and `manifest` refuses anything less. The six pair names, the six prompt slugs
and the total round count are literals in that script, because a regeneration
cannot supply them for itself: a shrunken capture would otherwise render as a
complete manifest and read as one. A change that legitimately moves any of them
moves the literal in the same commit, with the reason in the message.
