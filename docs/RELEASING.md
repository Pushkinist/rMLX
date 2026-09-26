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
   version and is not edited for a release.
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
