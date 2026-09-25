# Reference docs: what they keep, what reads them, how a cut is proved

This doc is the contract for cutting the large reference docs down to current
truth. It names the readers of doc text, the oracle that proves a cut broke no
reader, the size cap, the mutations the oracle must catch, and the plan. When
the last cut lands, this doc keeps only the contract and the readers.

## The rule

A reference doc under `docs/` states only what is true now: the contract, the
defaults, the dispositions, the current anchor numbers. It keeps no history.
A dated measurement, a superseded anchor row, an overturned claim, the story
of a retired variant and a "why we did not do X" narrative are deleted. They
do not move to another doc or to `CHANGELOG.md`. Git holds them.

A cut also corrects. Text that is wrong against the code today is fixed in the
same cut, not noted for later.

## The premise, measured

`make debt-report` prints the docs over the cap (section "docs over 40 KB").
It is the one producer of the doc-size figure. At the start, 18 of the 28
top-level docs are over 40 KB (KiB, as the report counts them). The largest
is `docs/KV_QUANT.md` at 357 KB.

Three statements in the original plan are false:

- `scripts/regression_gate.sh` and `scripts/perf_canary.sh` do not read
  `docs/PERF_BASELINE.md`. The gate takes the baseline as arguments. The
  canary reads `runs.db`. No program reads an anchor number from that doc.
  The anchors that `CLAUDE.md` quotes are a second copy for human readers.
- `docs/reports/` does not exist. Source comments, one script and the
  `Makefile` name files under it or under `docs/research/`; none of those
  files is in the tree.
- `docs/superpowers/` is not tracked (`.gitignore`). It is not part of the
  public docs. A citation into it resolves to nothing on a clean checkout.

## Readers of doc text

Every reader below was found in the tree, not recalled. `python3
scripts/check_doc_refs.py --list` prints the full reference inventory.

| Reader | What it reads | What breaks if the text moves |
|---|---|---|
| `scripts/check_kv_codec_disposition.sh` | Every `> **INERT on this build**` banner in `docs/KV_QUANT.md`, and that each opens within 3 lines of a `### ` heading | The gate fails (RULE 3 or placement). Banners and their headings stay in `docs/KV_QUANT.md`. |
| `scripts/check_kv_boundary_default_parity.sh` | The `` `--kv-boundary-layers` \| `HEAD,TAIL` \| `2,8` `` flag rows in `docs/CLI.md` | The gate fails when no row states the default. At least one row stays in `docs/CLI.md`. |
| `make check-published-table` | All of `docs/PUBLISHED_PROTOCOL.md` | It is generated. A hand edit fails the gate. |
| `scripts/lib/published_table.py` | Writes the text `` `docs/PROFILING.md` §9 `` into the generated table | Section 9 of `docs/PROFILING.md` keeps its number, or the emitter, its fixture and the table change together. |
| `scripts/check_doc_source_citations.sh` | Every backticked `crates/...` path in every tracked doc | A cut that keeps a citation keeps it resolvable. |
| `scripts/lib/debt_report.py` | The size of every `docs/*.md` | Advisory. It reports; it does not fail. |
| `CLAUDE.md` documentation map | One row per top-level doc | A deleted or new doc changes the map in the same cut. |
| Section citations (a doc name, then `§` and a number or a word) | Mostly into `METRICS_DB.md` (59), `KV_CACHE.md` (28), `KV_QUANT.md` (19), `PROFILING.md` (17) | A cited heading keeps its number and title, or every citation changes with it. |
| Quoted-phrase citations (a doc name, then a phrase in double quotes) | Mostly into `KV_QUANT.md` (18) | The phrase stays in the doc, or the citation changes. |
| Line citations (a doc name, a colon and a line number) | Four, all in `scripts/perf_ceiling.py`, all into `docs/PERF_BASELINE.md` | Any edit above the line moves it. Replace them with a quoted phrase in the cut that touches the doc. |
| Links and anchors (a Markdown link, a rustdoc link definition, or a doc name with `#anchor`) | From `README.md`, `CLAUDE.md`, `CHANGELOG.md`, the docs and rustdoc | The heading keeps its slug, or the link changes. |
| Path mentions (a bare doc path) | The most common reference, including user-facing text in `crates/rmlx-cli/src/startup.rs` and `crates/rmlx-mlx/src/pin.rs` | The doc keeps its path. |

What cannot move:

- The INERT banners and their `### ` headings in `docs/KV_QUANT.md`.
- One `--kv-boundary-layers` row with its default in `docs/CLI.md`.
- `docs/PUBLISHED_PROTOCOL.md`, byte for byte, unless the emitter changes.
- The number of section 9 in `docs/PROFILING.md`.
- The path of every doc that a program prints to a user.

## The oracle

A cut deletes text. The oracle proves that the deletion broke no reader. The
unit is the reference: one citation, link, anchor, banner or row that a
reader resolves. The command is:

```
make check-doc-consumers
```

It runs five checks. `check-doc-refs` runs `scripts/check_doc_refs.py --base
<merge-base with origin/main>`. That script fails when a reference resolved at
the base and resolves to nothing now, or when it now resolves to a different
target: a numbered section whose title changed, a cited line whose text
changed. A reference that was already broken at the base is printed as
carried and does not fail. Carried references are counted per target, so
moving one with its code passes and copying one fails. The other four are
the readers that exist as gates: `check-doc-source-citations`,
`check-kv-codec-disposition`, `check-kv-boundary-default-parity` and
`check-published-table`.

`scripts/check_doc_refs_selftest.sh` holds the script to 26 cases. Each case
asserts the exit code and the reason. The green cases include a deleted
section that nothing reads, and a cited section moved to another doc with its
citations.

What the oracle cannot see:

- Whether the kept text is true. That is the reviewer's check in every cut.
- A reference in a form it does not parse: a heading named in prose with no
  doc name beside it, or a doc name and its section split across a
  code-comment line break.
- A section cited by name is matched on the first word after `§`, against
  every word of every heading. A cut that deletes the cited heading and keeps
  another heading with that word passes.
- A partial deletion of the `--kv-boundary-layers` rows passes while one row
  keeps the default.
- A map row that names the right file under the wrong topic.
- The size cap. The oracle does not read sizes; `make debt-report` does.

## The size cap

The cap is 40 KB. `scripts/lib/debt_report.py` holds it
(`DOC_SIZE_THRESHOLD_KB`) and is its one producer. There is no second size
scan.

The cap is advisory until the last cut. A failing gate now fails every commit
on the 18 docs over the cap, whatever the commit changes. The last cut adds a
failing mode to `debt_report.py`, the way `--matched-lines` forwards its exit
code, and wires it into `make ci`. A doc that must stay over the cap carries
a `size-exempt: <reason>` line at its top, the way a source file carries
`LOC-exempt`, and the report prints the reason.

## Mutations and the assertion that catches each

| Mutation of a cut | Caught by |
|---|---|
| An INERT banner deleted | `check-kv-codec-disposition`: RULE 3 names the codec (fixture `rule3_inert_without_a_banner`; real-tree run: `'planar'`) |
| An INERT banner moved away from its heading | `check-kv-codec-disposition`: "does not open within 3 lines of a '### ' heading" |
| Every `--kv-boundary-layers` default row deleted | `check-kv-boundary-default-parity`: "rows that state no default" |
| The generated table edited by hand | `check-published-table`: "is not what the emitter renders" |
| A `crates/...` citation broken | `check-doc-source-citations`: "docs cite source paths that do not exist" |
| A heading cited by number deleted or renumbered | `check-doc-refs`: `SECTION … resolves to nothing` (case `numbered_section_renumbered`) |
| A heading cited by number retitled | `check-doc-refs`: `re-pointed` (case `numbered_section_retitled`) |
| A heading cited by name deleted | `check-doc-refs`: `SECTION … resolves to nothing` (case `named_section_deleted`) |
| A heading an anchor link names deleted | `check-doc-refs`: `LINK … resolves to nothing` (cases `anchor_heading_deleted`, `bare_anchor_deleted`, `heading_fenced`) |
| A quoted phrase deleted | `check-doc-refs`: `QUOTE … resolves to nothing` (case `quoted_phrase_deleted`) |
| Text deleted above a line citation | `check-doc-refs`: `LINE … re-pointed` (case `cited_line_shifted`) |
| A linked doc deleted | `check-doc-refs`: `LINK … resolves to nothing` (case `linked_doc_deleted`) |
| A relative link broken | `check-doc-refs`: `LINK … resolves to nothing` (case `relative_link_broken`) |
| A new doc with no map row | `check-doc-refs`: `MAP … resolves to nothing` (case `new_doc_unmapped`) |
| A map row whose link names another file | `check-doc-refs`: `MAPROW` (case `map_row_wrong_target`) |
| A doc left over the cap | Nothing fails. `make debt-report` lists it. The last cut adds the failing mode. |
| A map row with the right file under the wrong topic | Nothing. Review only. |
| Kept text that is false | Nothing. Review only. |

## Text that is wrong today

`python3 scripts/check_doc_refs.py` without `--base` lists 44 references that
resolve to nothing. The cut that owns the target doc, or the doc that holds
the citation, fixes them:

- `docs/CLI.md` cites a `TurboFlash` section of the KV quant doc (there is
  none) and a `Presets` section (the heading is "Preset interface").
  `docs/models/bonsai/8B/rMLX.md` cites the `TurboFlash` section too.
- `docs/SPECULATIVE.md` cites a `Profiles` section of the CLI doc; there is
  none.
- `docs/MODELS.md` links to two anchors of its own that no heading makes, and
  to a `WEIGHT_QUANTS.md` anchor that lost its section number.
- `docs/models/bonsai/8B/SIBLINGS.md` links to a `gemma4` doc by a wrong
  relative path.
- `docs/METRICS_DB.md` names an overview doc (`00-overview.md`) that does not
  exist.
- `docs/AUDIO.md` and `docs/E2E_TEST_PLAN.md` have no `CLAUDE.md` map row.
- `scripts/perf_ceiling.py` cites two line numbers of `docs/PERF_BASELINE.md`.
  Neither line holds the quoted text now.
- `docs/KV_ISO_TWINS.md` states a 200 KB advisory ceiling. The cap is 40 KB.
- Source comments name files under `docs/reports/`, `docs/research/`,
  `docs/superpowers/`, three numbered overview docs and a Jina recon doc. None is
  in the tree. Each goes in the cut of the doc that now holds its topic, or
  is deleted.

## The cut plan

One cut is one commit, 2 hours or less. Each cut: deletes history, corrects
what is wrong against the code, keeps every item in "What cannot move", runs
`make check-doc-consumers`, and states the doc's size before and after. The
reviewer reads the kept text against the code.

| Cut | Doc and range | Notes |
|---|---|---|
| 1 | `KV_QUANT.md`: Overview to Per-variant deep dive | Keep every INERT banner and its heading. The deep dive keeps the current disposition per spelling, one short entry each. |
| 2 | `KV_QUANT.md`: Layer-adaptive overrides, CLI flags | Keep the heading "Layer-adaptive overrides" (cited by name). The flag text that `docs/CLI.md` already holds goes; one link stays. |
| 3 | `KV_QUANT.md`: The auto default to the break-even condition | Keep "The auto default", "Codec disposition" and "Memory truth" phrases, or change their citations. |
| 4 | `KV_QUANT.md`: the four fused flash-decode sections, fused-QK storage, sparse attention | "Sparse attention" is cited by name. |
| 5 | `KV_QUANT.md`: Retired per-arch table, Codec fidelity; fix the inbound `TurboFlash` and `Presets` citations | The retired table goes whole. |
| 6 | `PERF_BASELINE.md`: caveats, current anchors, canary anchors | Keep one current anchor row set. Replace the four line citations in `perf_ceiling.py`. |
| 7 | `PERF_BASELINE.md`: the dated sections from "Codec cells across context" to the end | Almost every section is a dated record. Keep a current anchor only where it is the one live figure. |
| 8 | `METRICS_DB.md` §3 and §4 | 59 section citations point in. Keep numbers and titles, or move every citation with them. |
| 9 | `METRICS_DB.md` §5 to §15 | §7 "Migration plan" is history. §3.2, §4, §8.5.1 stay. |
| 10 | `TESTING.md`: the Metal-context `#[ignore]` section (99 KB) | Keep the rule and what each gate enforces; delete the narrative. |
| 11 | `TESTING.md`: the rest | |
| 12 | `CLI.md`: Subcommands | The flag table stays, the per-flag essays go. Keep one `--kv-boundary-layers` default row. |
| 13 | `SPECULATIVE.md` | |
| 14 | `SPEC_ROUND_SKELETON.md`, `KV_UPDATE_SPLIT.md`, `KV_ROTOR_TWINS.md`, `KV_ISO_TWINS.md`, `KV_TURBO_TWINS.md` | Closed campaign records. Each keeps only what a gate or test still names, or is deleted with its map row and its inbound links. The owner decides which. |
| 15 | `KV_CACHE.md`, `FFI.md` | 28 section citations into `KV_CACHE.md`. |
| 16 | `MODELS.md`, `SSD_TIER.md` | Fix the `MODELS.md` anchors. |
| 17 | `SERVER.md`, `SAMPLING.md`, `PROFILING.md` | `PROFILING.md` section 9 keeps its number. |
| 18 | The failing size mode; `AUDIO.md` and `E2E_TEST_PLAN.md` map rows; this doc cut to the contract and the readers | Every doc under the cap or `size-exempt` with a reason. |

A cut whose doc is still over the cap after its range adds a next cut for the
same doc. It does not raise the cap.
