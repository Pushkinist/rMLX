# Releasing rMLX

The version lives in one place: `[workspace.package].version` in the root
`Cargo.toml`. Member crates inherit it. Internal path dependencies carry no
version: the crates are `publish = false`, and `deny.toml` sets
`allow-wildcard-paths = true`.

Hosted CI (`.github/workflows/ci.yml`) runs formatting, the source gates,
the MSL compile, clippy and a release build. It runs the scripts' self-tests
but no `cargo test`: the hosted macOS runners have no usable Metal device.
Release artifacts are built on a local Apple Silicon machine with
`brew install mlx-c`.

## Setup

- `brew install mlx-c` (pulls `mlx`) for the dylibs the binary links.
- The tap repo `Pushkinist/homebrew-rmlx`.
- `cosign` for signing, `gh` for the release.

## What the artifacts link

The linked dylibs' install names are the Homebrew `opt` symlinks
(`crates/rmlx-mlx/build.rs`). So the tarball and the formula build both load
the MLX that Homebrew links on the user's machine at run time.

- The formula's `mlx-c` dependency carries no version. The reason is in
  `packaging/homebrew/rmlx.rb`.
- The binary reads `events.mlx_nax` from the metallib it loaded at run time
  (`crates/rmlx-mlx/src/nax.rs`), so the value describes the user's MLX.
- Run `make mlx-preflight` on the release machine before building.

## Branch model

- **`main` holds the released state.** Its history is linear and tags live
  on it. It takes the release and hotfix fast-forwards, and PRs merged by
  rebase: the formula PR of step 9 and every Dependabot PR target `main`.
- **`next/<name>` accumulates the next release.** Each issue lands as one
  squash-merged PR, one commit. The day-to-day flow is in `CONTRIBUTING.md`
  §Workflow.
- **A release fast-forwards `next/<name>` onto `main`**; a hotfix
  fast-forwards `hotfix/<issue>` onto `main`.

The GitHub rulesets enforce this:

| Ruleset | Rules |
|---|---|
| `main` | Pull request, rebase merge only; checks `rustfmt` and `build + clippy`, branch up to date; linear history; no force-push, no deletion, no direct update |
| `next/*` | Pull request, squash merge only; the same two checks, branch up to date; no force-push, no deletion |

So `main` accepts a PR merged by rebase, or a fast-forward push by a
repository admin or maintainer. Admins and maintainers bypass both rulesets;
that bypass is what lets the fast-forward push below reach `main`.

## Cut a release

1. **Bump** `version` in `Cargo.toml` through a normal issue PR into
   `next/<name>`, once everything else for the release is in.
2. **Changelog.** Add a `## [<version>] - <date>` section to `CHANGELOG.md`
   (Keep a Changelog format) and the matching `[<version>]:` link at the
   bottom. It is the source of the release body. `README.md` carries no
   version and is not edited for a release. The section can end with an
   optional `### Retrospective` heading that lists links to the issues the
   previous release's retrospective filed (§Retrospective). Nothing else
   goes under it: the section is the public release body. Released
   sections are never edited.
3. **Gate.** `make ci` and the whole `make ci-perf` green on `next/<name>`,
   plus the real-model regression smoke. Every merge to `main` runs the whole
   `make ci-perf`.
4. **Release PR and fast-forward.** Open a PR `next/<name>` → `main` for the
   checks, then push:
   ```sh
   git fetch origin
   git merge-base --is-ancestor origin/main origin/next/<name>
   git push origin origin/next/<name>:main
   ```
   The ancestor check fails if `main` moved, for example after a hotfix.
   Then rebase `next/<name>` onto `main` and gate again. GitHub marks the
   PR merged once `main` reaches its head.
5. **Tag.** `make tag` creates the annotated `v<version>` tag from
   `Cargo.toml`. Push it: `git push origin v<version>`.
6. **Package.** `make release-package` builds
   `dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz` and its `.sha256`.
   The tarball holds `rmlx`, both licenses and `README.md`.
7. **GitHub Release**, with the changelog section as the body:
   ```sh
   gh release create v<version> \
     --title "rMLX <version>" \
     --notes-file <(scripts/release/changelog_section.sh <version>)
   gh release upload v<version> \
     dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz \
     dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz.sha256
   ```
8. **Sign.** `make release-sign` writes
   `dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz.cosign.bundle`.
   It is a keyless cosign signature and opens a browser OIDC login.
   Upload it with `gh release upload`. The `.sha256` alone is self-attested;
   the bundle ties the tarball to the signer's identity and the Rekor log.
9. **Formula.** `make release-sha` prints the sha256 of the `v<version>`
   source archive. `bash scripts/release/source_sha256.sh --write` also
   patches `url` and `sha256` in `packaging/homebrew/rmlx.rb`.
   - GitHub builds the archive on first access. Fetch it two or three more
     times and check that the digest is stable.
   - Check that `url` and `sha256` both name the new version, then open a
     PR with the formula change against `main`.
10. **Tap.** `make tap-sync` copies the formula into
    `Pushkinist/homebrew-rmlx` as `Formula/rmlx.rb` and pushes it.
11. **Verify** the install paths (§Verify the install paths).
12. **Retrospective** over the diff from the previous tag to the new one
    (§Retrospective).

### No Homebrew bottle

`brew install rmlx` builds from source. The formula has no `bottle do` block.
A bottle links the builder's `mlx-c`, while the formula's `mlx-c` is
unversioned. A mismatched mlx / mlx-c pair fails at load with a dyld
`Symbol not found` (`crates/rmlx-mlx/mlx-pin.txt`). `make bottle`
(`scripts/release/build_bottle.sh`) builds a bottle; no release step
runs it.

## Hotfix

A hotfix fixes a bug already released on `main`.

1. Branch `hotfix/<issue>` from `main`.
2. Squash the fix to one commit.
3. Open a PR `hotfix/<issue>` → `main` for the checks, and run `make ci`
   and the whole `make ci-perf` on it.
4. Fast-forward `main`:
   ```sh
   git fetch origin
   git merge-base --is-ancestor origin/main origin/hotfix/<issue>
   git push origin origin/hotfix/<issue>:main
   ```
5. A maintainer rebases `next/<name>` onto the new `main` and force-pushes
   it, through the ruleset bypass.

## Dependabot PRs

`.github/dependabot.yml` sets no target branch, so Dependabot opens its PRs
against `main`. Hosted CI runs no `cargo test`, so a green check does not
prove the bump. Gate each bump locally with `make ci` and, since it merges
into `main`, the whole `make ci-perf`. A runtime dependency, such as the
allocator or the tokenizer, also needs a real-model smoke.

Dependabot edits only the manifest. A major bump that needs a source change
stays red until that change is pushed to the PR branch:

```sh
gh pr checkout <PR>
# migrate the source, run make ci
git push origin HEAD:dependabot/cargo/<branch>
```

## Verify the install paths

Prebuilt binary:

```sh
brew install mlx-c
gh release download v<version> -p '*aarch64-apple-darwin.tar.gz*'
shasum -a 256 -c rmlx-v<version>-aarch64-apple-darwin.tar.gz.sha256
tar xzf rmlx-v<version>-aarch64-apple-darwin.tar.gz
./rmlx-v<version>-aarch64-apple-darwin/rmlx --version
```

Signature:

```sh
gh release download v<version> -p '*.cosign.bundle'
cosign verify-blob \
  --bundle rmlx-v<version>-aarch64-apple-darwin.tar.gz.cosign.bundle \
  --certificate-identity <maintainer-oidc-email> \
  --certificate-oidc-issuer <issuer-url> \
  rmlx-v<version>-aarch64-apple-darwin.tar.gz
# issuer: GitHub https://github.com/login/oauth · Google https://accounts.google.com
```

Homebrew. Homebrew refuses a formula from an untrusted third-party tap, so
`brew trust` runs once:

```sh
brew tap Pushkinist/rmlx
brew trust Pushkinist/rmlx
brew install rmlx
brew test rmlx
rmlx --version
```

Local formula check before publishing. Homebrew installs a formula only
from a tap, so copy it into the local tap clone first:

```sh
cp packaging/homebrew/rmlx.rb \
  "$(brew --repository)/Library/Taps/pushkinist/homebrew-rmlx/Formula/rmlx.rb"
HOMEBREW_NO_INSTALL_FROM_API=1 \
  brew install --build-from-source pushkinist/rmlx/rmlx
brew audit --strict pushkinist/rmlx/rmlx
```

## Retrospective

Per-PR review sees one diff. It cannot see that a new file repeats a file
beside it, or that a new gate does the work of an old one. The retrospective
reads the whole diff from the previous tag to the new tag and files issues
for what the release repeated, left inert, grew past a limit or made
deletable.

Rules:

- Run it after step 11, in one working session.
- It changes no code and makes no commit. Its output is issues.
- Search the open and closed issues before you file. When an issue already
  tracks a finding, name that issue and file nothing.
- File one issue per finding group, not one per site. Give it exactly one of
  the labels `documentation`, `bug`, `feature`, `enhancement` or `test`. Add
  `premise-unverified` when the finding comes from code reading only.
- It lists flags and codecs. It never retires, renames or deletes one:
  retirement is a separate decision with its own proof.
- Answer every checklist item, and write "none" when an item finds nothing.
  An empty answer is evidence too. A measurement that could not run is "not
  measured", never "none". Post the answers as one comment on the release
  PR of step 4.

Set the range and check out both tags beside the repo. Run every script
from the current checkout, so one version of each tool measures both trees:

```sh
NEW=v<version>
PREV=$(git describe --tags --abbrev=0 "$NEW^")
git worktree add --detach ../rmlx-retro-prev "$PREV"
git worktree add --detach ../rmlx-retro-new "$NEW"
mkdir -p .rmlx/tmp
```

The checklist:

1. **Twins.** Run the debt report and every `--matched-lines` population
   over both trees, and compare. A new file or fn pair in the report is a
   finding. It gets a const-generic or trait issue.
   ```sh
   bash scripts/debt_report.sh --root ../rmlx-retro-prev > .rmlx/tmp/retro-prev.txt
   bash scripts/debt_report.sh --root ../rmlx-retro-new --since "$PREV" > .rmlx/tmp/retro-new.txt
   diff .rmlx/tmp/retro-prev.txt .rmlx/tmp/retro-new.txt
   pops=$(python3 -c 'import sys; sys.path.insert(0, "scripts/lib"); import debt_report; print(*sorted(debt_report.MATCHED_LINES_POPULATIONS))')
   for tree in prev new; do
     for p in $pops; do
       printf '%s: ' "$p"
       bash scripts/debt_report.sh --root "../rmlx-retro-$tree" --matched-lines "$p" 2>&1 | tail -1
     done > ".rmlx/tmp/retro-lines-$tree.txt"
   done
   diff .rmlx/tmp/retro-lines-prev.txt .rmlx/tmp/retro-lines-new.txt
   ```
   Read a `--matched-lines` figure against its item count. Each new item
   adds pairs, so the figure grows with the count alone. Compare figures
   only at an equal item count. When the count changed, find the new items
   in the release diff and compare each one with its closest sibling. A
   population that reads `unavailable` in either tree is "not measured".
   The report reads `.rs` files only. Compare each `.metal` file that the
   release changed with its siblings in the same directory. The same edit
   made in more than one file is also a twin signal.
2. **Size.** A non-test source file that crossed 1000 lines gets a split
   proposal or a `// LOC-exempt:` marker with its reason.
   ```sh
   git diff --name-only --no-renames --diff-filter=AM "$PREV" "$NEW" -- 'crates/*.rs' |
     grep -v -E '/tests/|(^|/)tests\.rs$|_tests\.rs$' |
     while read -r f; do
       new=$(git show "$NEW:$f" | wc -l)
       old=$(git show "$PREV:$f" 2>/dev/null | wc -l)
       [ "$new" -gt 1000 ] && [ "$old" -le 1000 ] && echo "$old -> $new $f"
     done
   ```
3. **Allows.** Each new `#[allow(` or `#![allow(` site gets a fix-or-keep
   verdict.
   ```sh
   git diff -U0 "$PREV" "$NEW" -- '*.rs' |
     awk '/^\+\+\+ /{f=$2; next} /^\+.*#!?\[allow\(/{print f": "$0}'
   ```
4. **Dead paths.** A new comment that marks a path as inert, dormant,
   deferred or kept gets a delete-or-schedule verdict. The command reads
   `//` comments and `#` comments followed by a space, so URLs and Rust
   attributes do not match. It still matches prose that uses these words:
   "inert" is also the name of a codec disposition (`docs/KV_QUANT.md`).
   Read each hit.
   ```sh
   git diff -U0 "$PREV" "$NEW" -- '*.rs' '*.metal' '*.sh' '*.py' |
     awk '/^\+\+\+ /{f=$2; next}
          tolower($0) ~ /^\+([ \t]*|.*[ \t])(\/\/|#[ \t]).*(inert|dormant|deferred|kept for|future-reference|no longer)/ {print f": "$0}'
   ```
5. **Gates.** For each new gate, name the gate it makes unnecessary, or the
   structural change that would make it unnecessary. A gate is a new
   `check-*` target or a new line in the `ci:` recipe, which also runs the
   `*-selftest` and `*-fixtures` scripts.
   ```sh
   git diff -U0 "$PREV" "$NEW" -- Makefile | grep -E '^\+check-[a-z0-9-]*:'
   ci_recipe() { git show "$1:Makefile" | awk '/^ci:/{on=1; print; next} on && /^\t/{print; next} on{exit}'; }
   diff <(ci_recipe "$PREV") <(ci_recipe "$NEW")
   ```
6. **Docs.** A doc that grew by more than 20%, or a new doc, gets a
   proposal to cut it to current truth.
   ```sh
   git diff --name-only --no-renames --diff-filter=AM "$PREV" "$NEW" -- 'docs/*.md' |
     while read -r f; do
       old=$(git cat-file -s "$PREV:$f" 2>/dev/null || echo 0)
       new=$(git cat-file -s "$NEW:$f")
       [ "$new" -gt $((old * 6 / 5)) ] && echo "$old -> $new $f"
     done
   ```
7. **Flags and codecs.** A new CLI flag, `--kv-quant` spelling or
   `--kv-preset` name with no measured win: list it with the measurement
   that would settle it. A digest oracle needs a positive control in the
   same run.
   ```sh
   git diff -U0 "$PREV" "$NEW" -- 'crates/rmlx-cli/*.rs' | grep -E '^\+.*#\[arg\('
   git diff -U0 "$PREV" "$NEW" -- crates/rmlx-kv-quant/src/quant.rs \
       crates/rmlx-kv-quant/src/quant_descriptor.rs \
       crates/rmlx-cli/src/commands/preset_table.rs |
     grep -E '^\+[[:space:]]*(\|[[:space:]]*|\(|spelling: Fixed\()?"[a-z0-9_]+"'
   ```
   The second command finds fixed spellings (the `spelling: Fixed(..)` rows
   of the codec descriptor), aliases (`KV_QUANT_ALIASES` in `quant.rs`) and
   preset names. The parser also accepts parametric families such as
   `mixed_*` and `rot_k_v*`, so read its diff when it changed.
8. **One paragraph.** Name the pattern that this release repeated and the
   generalisation that would have prevented it. It goes in the retrospective
   comment on the release PR, not in `CHANGELOG.md`.

Remove the two checkouts when the comment is posted:

```sh
git worktree remove ../rmlx-retro-prev
git worktree remove ../rmlx-retro-new
```

## Files

| Path | Role |
|---|---|
| `CHANGELOG.md` | Release notes (Keep a Changelog); the release body |
| `packaging/homebrew/rmlx.rb` | The formula; the tap holds a copy |
| `scripts/release/package_binary.sh` | `make release-package` |
| `scripts/release/sign_artifact.sh` | `make release-sign` |
| `scripts/release/source_sha256.sh` | `make release-sha`; `--write` patches the formula |
| `scripts/release/sync_tap.sh` | `make tap-sync` |
| `scripts/release/changelog_section.sh` | Prints one version's changelog section |
| `scripts/release/build_bottle.sh` | `make bottle`; not part of the release flow |
