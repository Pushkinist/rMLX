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

## Cut a release

1. **Bump** `version` in `Cargo.toml` `[workspace.package]`.
2. **Changelog:** add a `## [<version>] - <date>` section to `CHANGELOG.md`
   (Keep a Changelog format) and the matching `[<version>]:` link at the
   bottom. This is the durable, in-repo record and the source of the Release
   body.
3. **Gate:** `make ci` green (fmt + clippy + test + deny + audit).
4. **Tag:** `make tag` → creates annotated `v<version>` from `Cargo.toml`.
   Push it: `git push origin v<version>`.
5. **Build the binary artifact:** `make release-package` →
   `dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz` (+ `.sha256`).
6. **GitHub Release** (body = the CHANGELOG section for this version):
   ```sh
   gh release create v<version> \
     --title "rMLX <version>" \
     --notes-file <(scripts/release/changelog_section.sh <version>)
   gh release upload v<version> \
     dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz \
     dist/rmlx-v<version>-aarch64-apple-darwin.tar.gz.sha256
   ```
7. **Sign the artifact (keyless cosign):** `make release-sign` →
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
8. **Build + publish the Homebrew bottle** (binary install channel — no Rust
   toolchain required for users):

   a. **Build the keg from source** on the release machine (the same
      Apple-Silicon Mac used for the binary artifact above).

      > **Do step 9 (formula `url` + `sha256`) FIRST.** The bottle is a build of
      > the *formula's* source `url`, so the formula must already point at the
      > new `v<version>` tag tarball before you build the keg — otherwise you
      > bottle the previous release. (The 0.2.x releases shipped the formula-url
      > bump and the bottle as two separate PRs for this reason.)

      > **Homebrew ≥6 refuses a loose `.rb` path** (`Error: Homebrew requires
      > formulae to be in a tap`). The documented `brew install --build-bottle
      > packaging/homebrew/rmlx.rb` no longer works — build from the tap formula
      > instead. Mirror the url+sha-bumped formula into the tap checkout, then
      > install from the tap reference:
      ```sh
      cp packaging/homebrew/rmlx.rb \
        "$(brew --repository)/Library/Taps/pushkinist/homebrew-rmlx/Formula/rmlx.rb"
      HOMEBREW_NO_INSTALL_FROM_API=1 brew install --build-bottle pushkinist/rmlx/rmlx
      ```
      This is a full source build (a few minutes). `--build-bottle` forces a
      source build (a plain `brew install` may reuse a cached bottle). The
      temporary tap-checkout copy is overwritten by `make tap-sync` (step 10).
      If the tap is not yet tapped locally:
      `brew tap pushkinist/rmlx && brew trust pushkinist/rmlx`. Note your `cp`
      may be aliased to `cp -i` — use `/bin/cp -f` if it prompts.

   b. **Build the bottle and upload it:**
      ```sh
      make bottle          # runs scripts/release/build_bottle.sh
      ```
      The script:
      - Runs `brew bottle --json --root-url=https://github.com/Pushkinist/rMLX/releases/download/v<ver>`.
      - Renames the local `rmlx--<ver>.<tag>.bottle.tar.gz` to the remote
        single-dash name `rmlx-<ver>.<tag>.bottle.tar.gz` (Homebrew's
        intentional local/remote naming split — the remote asset must use
        single-dash or `brew install` gets a 404).
      - Prints the `bottle do … end` block to paste into the formula.
      - Prints the exact `gh release upload` command.

      Run the printed upload command:
      ```sh
      gh release upload v<version> dist/rmlx-<ver>.<tag>.bottle.tar.gz
      ```

   c. **Paste the `bottle do` block into `packaging/homebrew/rmlx.rb`.**
      The script prints the block with the correct `root_url`, `cellar`, OS
      tag, and sha256.  Insert it immediately after the `head` line, before
      the `depends_on` lines:
      ```ruby
      bottle do
        root_url "https://github.com/Pushkinist/rMLX/releases/download/v<ver>"
        sha256 cellar: :any_skip_relocation, arm64_tahoe: "<sha256>"
      end
      ```
      The `depends_on "rust" => :build` and `depends_on "mlx-c"` lines remain
      unchanged.  When a `bottle do` block is present, `brew install` fetches
      the binary directly; if the bottle is unavailable for the user's macOS
      version Homebrew falls back to the source build automatically.

   d. **Commit the formula update** via a PR (main is ruleset-protected). By now
      the formula carries the new `url` + source `sha256` (step 9 — done first,
      see the ordering note in 8a) **and** the `bottle do` block (8c). Commit
      them together as one formula PR:
      ```sh
      git add packaging/homebrew/rmlx.rb
      git commit -m "chore(release): bump Homebrew formula to v<version>"
      ```
      Open the PR, let CI pass, merge it, then run `make tap-sync` (step 10).

   > **Clean-machine verification (optional but recommended):**
   > On a machine without a local rmlx keg:
   > - `brew tap Pushkinist/rmlx && brew trust Pushkinist/rmlx && brew install rmlx` —
   >   should download the prebuilt bottle, not compile from source (confirm with
   >   `brew install --verbose rmlx` — the word "Bottled" appears in output).
   > - `brew uninstall mlx-c && brew install rmlx` — should fail cleanly with a
   >   dependency error before attempting any download or compile.
   >
   > **Verifying the pour ON the release machine is misleading.** The release
   > machine's keg was built with `--build-bottle` (source), and `brew reinstall
   > rmlx` *repeats the install options* — so it rebuilds from source even when
   > the bottle is valid. To confirm the bottle pours here, either
   > `brew uninstall rmlx && brew install rmlx` (a fresh install pours) or
   > `brew reinstall --force-bottle rmlx` (pours or fails loudly with the real
   > reason). Check `INSTALL_RECEIPT.json` → `poured_from_bottle: true`, or
   > `brew info rmlx` → `(bottled)`.
   >
   > **The uploaded bottle asset has the same CDN-transient sha as the source
   > archive** (see step 9): right after `gh release upload`, GitHub may serve
   > stale bytes for ~1 min, so a verify run in that window sha-mismatches the
   > bottle and silently falls back to source. Re-fetch the bottle URL
   > (`curl -fsSL .../v<ver>/rmlx-<ver>.<tag>.bottle.tar.gz | shasum -a 256`)
   > until it equals the formula's bottle `sha256` before trusting a non-pour.
   > Do **not** re-upload with `--clobber` to "fix" it — that just restarts the
   > propagation window.

9. **Formula url + sha256** — do this **before** building the bottle (step 8):
   the bottle is a build of the formula's source `url`, so the formula must
   point at the new tag first. Note `make release-sha` only **prints** the
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
   The url+sha change is committed **together with the bottle block** as one
   formula PR (see 8d); `main` is ruleset-protected.
10. **Publish the tap:** `make tap-sync` (copies the formula into
    `Pushkinist/homebrew-rmlx` as `Formula/rmlx.rb` and pushes).

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

**Homebrew bottle (binary — no compile):**
```sh
brew tap Pushkinist/rmlx
brew trust Pushkinist/rmlx
brew install rmlx               # fetches bottle, does NOT invoke cargo
brew test rmlx
rmlx --version
```
Confirm no source build occurred: `brew install --verbose rmlx` shows "Bottled"
in the output. To confirm the clean mlx-c dependency failure:
```sh
brew uninstall mlx-c
brew install rmlx               # fails with dependency error, not a dyld crash
brew install mlx-c              # reinstate
```

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
| `scripts/release/build_bottle.sh` | Build the Homebrew bottle from an installed keg, rename, and print upload + formula instructions |
| `scripts/release/source_sha256.sh` | Compute / patch the formula source sha256 |
| `scripts/release/sync_tap.sh` | Push the formula to the tap repo |
| `scripts/release/changelog_section.sh` | Print one version's CHANGELOG section |
