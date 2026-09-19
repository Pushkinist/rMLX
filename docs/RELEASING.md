# Releasing rMLX

The version lives in **one** place: `[workspace.package].version` in the root
`Cargo.toml`. Member crates inherit it (`version.workspace = true`); internal
path deps omit a version (`deny.toml` sets `allow-wildcard-paths = true`, and
crates are `publish = false`). There is no separate `VERSION` file.

> **Hosted CI cannot build rMLX.** GitHub's macOS runners are VMs without a
> usable Metal device, and the build links the Metal MLX libraries. CI
> (`.github/workflows/ci.yml`) runs fmt + clippy + a best-effort release build
> only. **All release artifacts are built locally** on an Apple-Silicon machine
> with `brew install mlx-c` present.

## One-time setup

- `brew install mlx-c` (pulls `mlx`) — provides the `libmlxc` / `libmlx` dylibs
  the binary links.
- A published tap repo `Pushkinist/homebrew-rmlx` for `brew tap`.

> **What the shipped artifacts resolve MLX to.** The release tarball links
> `libmlx.dylib` / `libmlxc.dylib` through the moving
> `/opt/homebrew/opt/...` symlinks, so **neither carries the MLX it was built
> against** — each runs against whatever the installing user has, which
> `depends_on "mlx-c"` resolves to the current release. Two consequences worth
> holding onto when reading a user report:
>
>   - `events.mlx_nax` is read at run time from the metallib the user's
>     machine actually loaded (`crates/rmlx-mlx/src/nax.rs`), so a row from a
>     distributed binary is the *user's* answer, not the builder's. Nothing
>     about the nax capability is baked into the artifact.
> - Run `make mlx-preflight` on the release machine before building the
>   release binary.
>   It does not change what users get, but it keeps the release machine's own
>   published prefill numbers honest.
>
> The formula deliberately does **not** pin an MLX version — that would force a
> downgrade on M1–M4 users for a benefit their hardware has no use for. The
> rationale is in `packaging/homebrew/rmlx.rb`; do not "fix" it by adding a
> version constraint.

## Branch model

- **`main` holds released state only.** History on `main` is linear: a
  commit reaches it only through a release fast-forward or a hotfix
  fast-forward, never a merge commit. Tags live on `main`. A GitHub ruleset
  on `main` requires a pull request, the `rustfmt` and `build + clippy`
  checks, and linear history, and forbids force-push, deletion, and any push
  or merge from a non-maintainer.
- **`next/<name>` is the accumulation branch** open between releases (for
  example `next/kv-quant-cleanup`). A GitHub ruleset on `next/*` requires
  a pull request and the same checks, allows only the squash merge method,
  and restricts force-push, deletion, and merges to maintainers.
- **One issue = one branch = one PR = one commit on `next/<name>`.** See
  `CONTRIBUTING.md` §Workflow for the day-to-day flow: branch from
  `next/<name>`, rebase onto it to stay current, one PR per issue, squash-merged
  by a maintainer.
- **A release moves `next/<name>` onto `main` by fast-forward** — see step 4
  below. A hotfix moves a `hotfix/<issue>` branch onto `main` the same way —
  see Hotfix below.

## Cut a release

1. **Bump** `version` in `Cargo.toml` `[workspace.package]`, through a normal
   issue branch merged into `next/<name>` like any other change — do this
   last, once everything else for the release is in.
2. **Changelog:** add a `## [<version>] - <date>` section to `CHANGELOG.md`
   (Keep a Changelog format) and the matching `[<version>]:` link at the
   bottom. This is the durable, in-repo record and the source of the Release
   body.

   > **Do not version-bump `README.md`.** The README is intentionally decoupled
   > from the release steps: its version badge (`shields.io/github/v/release`)
   > tracks GitHub Releases automatically, and the Status line carries no version
   > number. `CHANGELOG.md` is the per-release doc edit; the README changes only
   > when capabilities materially change (a new modality, architecture family, or
   > endpoint) — never for a routine version bump.
3. **Gate:** `make ci` green on `next/<name>` (fmt + clippy + test + deny +
   audit). This is also where the real-model regression smoke and, for a
   codec-layer or `.metal`-touching release, `make ci-perf` run — once, on
   `next/<name>`, fixed before the release PR (see `CONTRIBUTING.md`
   §Workflow).
4. **Release PR + fast-forward:** open a PR `next/<name>` → `main` for review
   and the required checks. Merge it by fast-forward push, not with a GitHub
   merge button — the ruleset leaves only rebase-merge enabled on `main` as a
   fallback that also keeps history linear, but the push below is the normal
   path:
   ```sh
   git fetch origin
   git merge-base --is-ancestor origin/main origin/next/<name>
   git push origin origin/next/<name>:main
   ```
   This pushes the accumulated issue commits onto `main` one by one, with no
   merge commit and no rewritten SHAs. GitHub marks the PR merged once `main`
   reaches the same commit. `git merge-base --is-ancestor` fails the command
   before the push if `main` moved out from under `next/<name>` (a hotfix
   landed, for example) — rebase `next/<name>` onto `main` and re-gate before
   retrying.
5. **Tag:** `make tag` → creates annotated `v<version>` from `Cargo.toml`.
   Push it: `git push origin v<version>`.
6. **Build the binary artifact:** `make release-package` →
   `dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz` (+ `.sha256`).
7. **GitHub Release** (body = the CHANGELOG section for this version):
   ```sh
   gh release create v<version> \
     --title "rMLX <version>" \
     --notes-file <(scripts/release/changelog_section.sh <version>)
   gh release upload v<version> \
     dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz \
     dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz.sha256
   ```
8. **Sign the artifact (keyless cosign):** `make release-sign` →
   `dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz.cosign.bundle` (needs
   `brew install cosign`; opens a browser OIDC prompt). Upload it:
   ```sh
   gh release upload v<version> \
     dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz.cosign.bundle
   ```
   The release binary is built locally (hosted CI has no Metal), so the
   `.sha256` alone is self-attested. The cosign bundle binds the tarball to
   your authenticated identity + the public Rekor log — real provenance for
   the prebuilt binary. Consumer-side verification is in "Verify both install
   paths".
9. **No Homebrew bottle is published.** `brew install rmlx` builds from source,
   which is what `depends_on "rust" => :build` in the formula has always
   described. The formula carries **no `bottle do` block**, and adding one back
   needs the two problems below solved first — not just a fresh sha.

   > **Why the block was removed (0.4.1).** A `bottle do` block names a
   > `root_url` pinned to one release, but Homebrew derives the bottle
   > *filename* from the formula's current version. So a block left behind after
   > a version bump sends `brew install` looking for a bottle that was never
   > built, under the previous release's URL, and it 404s. The block shipped
   > pinned to `v0.3.0` through both `v0.3.0` and `v0.4.0` while the only bottle
   > asset in existence was `rmlx-0.3.0.arm64_tahoe.bottle.tar.gz` — `brew info`
   > reported `(bottled)` the whole time. Nothing in the release flow regenerated
   > it, and nothing failed when it went stale.
   >
   > **The deeper reason not to reinstate it casually.** A bottle is a binary
   > linked against the `mlx-c` present on the build machine, but `mlx-c` is
   > deliberately an unversioned dependency (see the rationale in
   > `packaging/homebrew/rmlx.rb`), and `crates/rmlx-mlx/mlx-pin.txt` records
   > that a mismatched mlx / mlx-c pair aborts at load with a dyld
   > `Symbol not found`. A bottle poured onto a user whose mlx-c revision differs
   > from the builder's can therefore fail at load, where a source build against
   > that user's own mlx-c cannot. Building one on this machine also requires
   > `brew unpin mlx mlx-c`, which upgrades the pinned pair the project's own
   > measurements depend on.
   >
   > `scripts/release/build_bottle.sh` and `make bottle` are kept for a future
   > bottle channel that solves the ABI coupling — an mlx-c version constraint,
   > or a static link. They are **not** part of the release flow. Do not run them
   > and paste the output into the formula without also arranging for the next
   > release to rebuild or remove the block.

10. **Formula url + sha256.** Note `make release-sha` only **prints** the
    sha of the `v<version>` GitHub source tarball; to patch the `url` +
    `sha256` in `packaging/homebrew/rmlx.rb` in place, run the script with
    `--write`:
    ```sh
    bash scripts/release/source_sha256.sh --write
    ```
    > GitHub generates the source archive on first access, so its sha256 can
    > shift on the very first fetch right after a tag push. The
    > `source_sha256.sh`-written value is usually the correct stable one — but
    > re-fetch the archive 2-3× (`curl -fsSL .../archive/refs/tags/v<version>.tar.gz
    > | shasum -a 256`) and confirm the digest is stable before trusting it.
    Commit the url+sha change as its own formula PR; `main` is
    ruleset-protected. Verify **both** lines read the new version before opening
    it — `--write` patches `url` and `sha256` together, but a mismatch between
    them makes `brew install` fail the checksum.
11. **Publish the tap:** `make tap-sync` (copies the formula into
    `Pushkinist/homebrew-rmlx` as `Formula/rmlx.rb` and pushes).

## Hotfix

A hotfix fixes a bug already released on `main`, without waiting for the
current `next/<name>` to finish accumulating.

1. Branch `hotfix/<issue>` from `main` directly.
2. Squash the fix to a single commit before requesting review.
3. Open a PR `hotfix/<issue>` → `main` for review and the required checks.
4. Merge by fast-forward push, the same mechanics as a release:
   ```sh
   git fetch origin
   git merge-base --is-ancestor origin/main origin/hotfix/<issue>
   git push origin origin/hotfix/<issue>:main
   ```
5. **Carry the fix forward.** A maintainer rebases `next/<name>` onto the
   new `main` so the hotfix is not lost when the next release ships. This
   needs a force-push on `next/<name>`, which the ruleset allows only as a
   maintainer bypass.

## Dependency-bump PRs (Dependabot)

Hosted CI runs only fmt + clippy + a best-effort build (no Metal, no tests), so
a green Dependabot check does **not** prove the bump builds with Metal or passes
the suite. **Gate every bump locally with `make ci`** (and a real-model smoke for
runtime-affecting deps — allocator, tokenizer) before merging.

A **major** bump that needs source migration is the trap: Dependabot only edits
the manifest, so its branch stays RED until the migration commit is **pushed to
the remote PR branch**:

```sh
gh pr checkout <PR>            # work on the dependabot branch
# … migrate source, `make ci`, prove …
git push origin HEAD:dependabot/cargo/<branch>   # REQUIRED — push the fix
```

`main`'s ruleset requires the `rustfmt` and `build + clippy` checks with **no
bypass** (not even admin), so the PR cannot merge until the *remote* branch is
green. A migration that lives only in your local checkout will not unblock the
merge. Do **not** `git branch -D` the local branch before the remote has the
fix (the migration commit is otherwise only reachable via reflog).

## Verify both install paths

**Prebuilt binary (from the Release):**
```sh
brew install mlx-c
gh release download v<version> -p '*aarch64-apple-darwin.tar.gz*'
shasum -a 256 -c rmlx-v<version>-aarch64-apple-darwin.tar.gz.sha256
tar xzf rmlx-v<version>-aarch64-apple-darwin.tar.gz
./rmlx-v<version>-aarch64-apple-darwin/rmlx --version
```

**Provenance (cosign bundle, if the release ships one):**
```sh
gh release download v<version> -p '*.cosign.bundle'
cosign verify-blob \
  --bundle rmlx-v<version>-aarch64-apple-darwin.tar.gz.cosign.bundle \
  --certificate-identity <maintainer-oidc-email> \
  --certificate-oidc-issuer <issuer-url> \
  rmlx-v<version>-aarch64-apple-darwin.tar.gz
# issuer: GitHub https://github.com/login/oauth · Google https://accounts.google.com
```

**Homebrew (build from source):**
```sh
brew tap Pushkinist/rmlx
brew trust Pushkinist/rmlx   # one-time: Homebrew refuses to load formulae from untrusted third-party taps
brew install rmlx
brew test rmlx
rmlx --version
```
> Recent Homebrew versions block third-party taps until trusted — a fresh
> `brew install rmlx` fails with `Refusing to load formula … from untrusted tap`
> until `brew trust Pushkinist/rmlx` is run once.

Local formula check before publishing:
```sh
brew install --build-from-source ./packaging/homebrew/rmlx.rb
brew audit --strict --new rmlx
```

## Files

| Path | Role |
|---|---|
| `CHANGELOG.md` | Durable release notes (Keep a Changelog); body source |
| `packaging/homebrew/rmlx.rb` | Canonical formula (source of truth) |
| `scripts/release/package_binary.sh` | Build + bundle the binary tarball |
| `scripts/release/build_bottle.sh` | Retired from the flow (step 9). Kept for a future bottle channel; not run at release time |
| `scripts/release/source_sha256.sh` | Compute / patch the formula source sha256 |
| `scripts/release/sync_tap.sh` | Push the formula to the tap repo |
| `scripts/release/changelog_section.sh` | Print one version's CHANGELOG section |
