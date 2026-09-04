# Answer equivalence for speculative decoding — state of the gate

`crates/rmlx-models/tests/spec_greedy_equivalence.rs`. This branch carries the
gate; it was split out of the metrics-recording branch after five review rounds,
not because it failed but because every finding left in it is a **calibration**
problem, and the numbers it is calibrated against are readings from an engine
with a known correctness defect. It should land with the fix for that defect,
where those numbers become measurable instead of assumed.

## What the gate is for

Greedy speculative decoding emits the verifier's own argmax at every position,
so at temperature 0 a speculative run and a plain run of the same verifier are
two ways of computing one answer. Nothing in a throughput number says whether
that still holds. Before this gate, the only checks were four alignment suites
comparing a **48-token** shared prefix, and one Gemma4-assistant test that
asserted only that the accept rate was high.

## What is proven

- **It found a real, pre-existing correctness defect** that had been invisible
  to the 48-token prefix checks for as long as they have existed. Driven from
  `prompts/longctx_4k.json`, the Gemma4-assistant speculative arm does not
  reproduce plain greedy. It reproduces identically with
  `crates/rmlx-models/src/speculative/` checked out at `8ccc0593`, so it is not
  this work's. Filed as **issue #506**, with the Qwen3.8-27B MTP sidecar's 0.520
  reading as a second case.
- **The defect has two manifestations and the prompt decides which appears.**
  This matters for whoever fixes it and is the single most useful thing recorded
  here:

  | prompt | speculative arm | whole-stream LCS | cycle reading |
  |---|---|---|---|
  | structured ("summarise section by section") | period-8 repetition loop, `x86 is:66 is x86 is:66 is …` | 0.1100 | **1.0000** at period 8 from token 128 |
  | prose ("summarise … in continuous prose") | a different, degraded summary that runs to the token budget while plain greedy stops at 218 | 0.2798 | **0.0926** — indistinguishable from healthy |

  A fix validated only against the loop shape may leave the second untouched.
- **Bit-identity is not the contract, and that is measured.** The verify pass
  scores a whole block in one forward where plain decode steps one token at a
  time — a different reduction order. On the most favourable pair there is (a
  full-attention verifier whose rollback is an exact KV truncation) the two arms
  share 91 leading tokens, differ by a word, and then continue the same answer.
- **The subsequence oracle separates the two rollback regimes**, measured on
  `gemma-4-e2b-it-mxfp8` + `gemma-4-E2B-it-assistant-bf16` at 256 tokens:

  | rollback | whole-stream LCS | weakest tail window | first divergence |
  |---|---|---|---|
  | as shipped | 0.9180 | 0.6875 @192 | 91 |
  | one rejected draft key left in the cache | 0.4062 | 0.2031 @192 | 8 |

- **Greedy decoding compounds, so a longer horizon separates the regimes less,
  not more**: the same shipped pair reads 0.9180 at 256 tokens, 0.8438 at 512
  and 0.6846 at its natural stop near 800 — the last below any floor that still
  refuses a broken rollback.
- **The pair resolves by slug** from `RMLX_O_MODELS_ROOT`, so `make gpu-test`
  runs it on a machine holding the snapshots. `-e2b-` and `-e4b-` assistants
  declare the same architecture, so the harness's arch stand-down cannot
  separate them; the drafter's `backbone_hidden_size` is checked against the
  verifier's width before the drafter is loaded and a mismatched pair skips with
  both widths named.
- Under Metal shader validation the pair produces **zero** hits, so it needs no
  entry in `scripts/gpu_validation_census.txt`.

## What is open

Ten findings from the fourth and fifth reviews. The first two are the ones that
block it.

1. **The "no gap" claim is false, and the fixture samples around the hole.**
   `two_arms_in_the_same_ragged_loop_are_not_a_pass` uses the noise list
   `[0, 12, 16, 25, 30, 40]` and steps over 34–38. At **36%** raggedness, two
   arms sharing a real 128-token prefix and locked in the same period-8 loop are
   returned as agreement, reproduced across seven seed pairs; the committed 40%
   case is decided by an LCS margin of 0.0086 and passes with other seeds. Fix:
   sweep the parameter (`(0..=60).step_by(2)` over several seed pairs) and
   assert the property, or delete the "no gap" sentences and state the measured
   hole (34–42%) as a declared blind spot.
2. **`two_arms_collapsing_over_their_last_quarter_are_not_a_pass` did not fail
   before the fix** it was committed to justify — it reads 0.8854 / 0.9271
   pre-fix and was refused either way — and it uses period 16, not the period-40
   tail-quarter shape the quarter-bound removal was argued from. (Pin 1 is
   sound: it genuinely failed pre-fix at lcs 0.8770, cycles 0.8523 / 0.7981.)
3. **Both pins assert only that something refused, not which rule.** Per the
   readings, the control refuses only at noise 0–32; at 34 the tail-LCS window
   does and at 35–40 the whole-stream LCS does. Deleting the cycle control
   entirely leaves both pins green.
4. **The prose calibration has an unenforced precondition.** The ceiling is only
   meaningful for prose; nothing checks the arms came back as prose, and the
   short prompt is enumerative. When a model answers with a list the gate does
   not report an unclassifiable input — it accuses the **plain greedy** arm of a
   repetition loop. The ceiling is also only safe for i.i.d.-word prose:
   evenly-spaced parallel-structure prose with a stock frame reads 0.52–0.55
   with no list markers at all.
5. **The short-prompt length floor equals the token budget** (`MIN_ANSWER_TOKENS`
   256 under `N_TOKENS` 256), so the gate passes only if neither arm ever emits a
   stop id — and `N_TOKENS`'s own doc says the opposite.
6. **The 200-token long-context floor is pinned by a tautology.** It sits 8.3%
   under one measurement of one arm on one prompt and one model pair, and
   `MEASURED_LONG_CONTEXT_PLAIN_ARM` has no producer that reads the engine, so
   nothing notices if the real arm moves to 195.
7. `// gpu-test-gate: exempt` on `the_false_positive_rate_on_healthy_output`
   exempts nothing — the test never names `Device::Gpu`. Delete the marker.
8. "1800 synthetic healthy streams across six lengths peaked at 0.1212" matches
   no committed test: the assertion runs 5 × 64 = 320 and peaks at 0.1212, the
   measurement runs 6 × 1000 = 6000 and peaks at 0.1351.
9. The LCS range "0.70 to 0.93 across the 12–40% ragged range" was measured at
   the retired 512-token horizon; at 256 the same construction reads 0.8594 at
   12% and 0.6914 at 40%.
10. A missing blank line between two items near the end of the pair pins.

## Why these should be re-derived rather than defended

Every constant in this file — `MIN_LCS_RATIO`, `MIN_TAIL_LCS_RATIO`,
`MAX_CYCLE_FRACTION`, `MIN_ANSWER_TOKENS`, `MIN_LONG_CONTEXT_ANSWER_TOKENS`,
`MEASURED_LONG_CONTEXT_PLAIN_ARM`, and the choice of `N_TOKENS` — is calibrated
against readings taken from an engine that does not reproduce plain greedy on
the long prompt. Several are guesses dressed as measurements for that reason:

- The long-context floor is set under a plain-arm length of 218 measured *while
  the speculative arm is broken*. After the fix both arms answer that prompt and
  the pair's real length distribution is observable — the floor should come from
  that, and `run_gate` should fail with "the measured plain arm moved from 218 to
  N" rather than a generic early-stop message.
- The horizon was chosen as the length where two regimes separate, one of which
  is a deliberately mutated rollback rather than any real defect. Post-fix, the
  benign divergence profile of a *correct* pair is measurable directly and the
  horizon can be set from it.
- The repetition control's ceiling is calibrated against prose from prompts
  chosen partly because they keep the gate green. Once the long prompt passes,
  the gate should move onto it — that was always the intent — and the control
  re-calibrated against what that prompt actually produces.

Two structural observations worth carrying forward, because they caused four
rounds of fix-plants-defect:

- **Fixtures must judge the pair, not an arm.** Every fixture written before the
  fifth round tested one arm in isolation, or a pair built from independently
  seeded heads. The gate's question is about two arms sharing a real prefix, and
  the interaction between the two oracles — which is the whole safety argument —
  went untested for that reason.
- **Do not characterise a population from points you chose.** Both the "healthy
  output ≤ 0.69" claim and the "no gap across the ragged range" claim were drawn
  from a handful of self-authored samples and both were false. Sweep the
  parameter; assert the property; let the sweep pick the points.

`Rng::prose` is an i.i.d. word model with zero autocorrelation. Real prose is
not, which is why finding 4 exists. A generator with realistic autocorrelation,
or a corpus of real arms captured from runs, would make the false-positive
argument mean something it does not currently mean.
