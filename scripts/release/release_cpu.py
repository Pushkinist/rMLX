#!/usr/bin/env python3
"""The CPU target of the release tarball binary.

`.cargo/config.toml` builds with `target-cpu=native`: right for a from-source
build, wrong for a binary built on one Mac and run on another. The release
build pins BASELINE_CPU instead, keeping every other flag config.toml sets.

  rustflags [--root DIR]
      Print the release build's rustflags in CARGO_ENCODED_RUSTFLAGS form
      (0x1f-separated): config.toml's flags with its target CPU replaced by
      BASELINE_CPU. That variable replaces build.rustflags outright, so this is
      the one place the other flags are carried over rather than restated.
      Exit 1 when config.toml sets a target feature.

  check <tarball> [--root DIR]
      Exit 0 when the `rmlx` binary inside <tarball> was compiled with exactly
      one target CPU, BASELINE_CPU, no target feature, and otherwise with
      config.toml's flags in config.toml's order. It reads rustflags only: a
      rustc wrapper's own arguments are not recorded, which is why
      package_binary.sh builds with every wrapper cleared.
      The binary carries no record of its target CPU, so the chain is: the
      packaged bytes -> the byte-identical `target/release/deps/rmlx-<hash>`
      cargo linked -> `target/release/.fingerprint/rmlx-cli-<hash>/bin-rmlx.json`,
      where cargo records the rustflags that unit was compiled with.
      Exit 1 = built for another CPU or with other flags. Exit 2 = the chain
      cannot be followed (the verdict is unknown, not a pass).

--root is the repo root (default: this checkout); it is where config.toml and
target/ are read.
"""

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

# The oldest chip the project supports (M1-M5); also rustc's default CPU for
# aarch64-apple-darwin, so the pin changes nothing on an M1 build machine.
BASELINE_CPU = "apple-m1"
PACKAGE = "rmlx-cli"
BIN = "rmlx"


class Unreadable(Exception):
    pass


class Refused(Exception):
    pass


def codegen(opt):
    """`-C<opt>` with the option name spelled with dashes: rustc reads `_` and `-` alike."""
    name, eq, value = opt.partition("=")
    return "-C" + name.replace("_", "-") + eq + value


def normalize(flags):
    """rustc codegen flags as single `-C<opt>` tokens, whichever spelling was used."""
    out = []
    it = iter(flags)
    for tok in it:
        if tok in ("-C", "--codegen"):
            nxt = next(it, None)
            if nxt is None:
                raise Unreadable(f"flag list ends in a bare {tok}")
            out.append(codegen(nxt))
        elif tok.startswith("--codegen="):
            out.append(codegen(tok[len("--codegen="):]))
        elif tok.startswith("-C"):
            out.append(codegen(tok[2:]))
        else:
            out.append(tok)
    return out


def split_cpu(flags):
    """(target-cpu values, every other flag), both in order."""
    cpus, rest = [], []
    for tok in normalize(flags):
        if tok.startswith("-Ctarget-cpu="):
            cpus.append(tok[len("-Ctarget-cpu="):])
        else:
            rest.append(tok)
    return cpus, rest


def config_rustflags(root):
    path = root / ".cargo" / "config.toml"
    if not path.is_file():
        raise Unreadable(f"{path} not found")
    lines = [l for l in path.read_text().splitlines() if re.match(r"\s*rustflags\s*=", l)]
    if len(lines) != 1:
        raise Unreadable(f"{path}: expected one `rustflags = [...]` line, found {len(lines)}")
    m = re.fullmatch(r'\s*rustflags\s*=\s*(\[[^\]]*\])\s*', lines[0])
    if not m:
        raise Unreadable(f"{path}: the rustflags array is not on one line: {lines[0].strip()}")
    try:
        flags = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise Unreadable(f"{path}: cannot read the rustflags array: {e}")
    if not all(isinstance(f, str) for f in flags):
        raise Unreadable(f"{path}: rustflags holds a non-string entry")
    return flags


def packaged_binary(tarball):
    if not tarball.is_file():
        raise Unreadable(f"{tarball} not found")
    with tarfile.open(tarball, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).name == BIN]
        if len(members) != 1:
            raise Unreadable(f"{tarball}: expected one `{BIN}` binary, found {len(members)}")
        return tf.extractfile(members[0]).read()


def recorded_rustflags(root, binary):
    release = root / "target" / "release"
    digest = hashlib.sha256(binary).hexdigest()
    matches = [
        p for p in sorted((release / "deps").glob(f"{BIN}-*"))
        if re.fullmatch(rf"{re.escape(BIN)}-[0-9a-f]{{16}}", p.name) and p.is_file()
        and hashlib.sha256(p.read_bytes()).hexdigest() == digest
    ]
    if len(matches) != 1:
        raise Unreadable(
            f"expected one {release / 'deps'}/{BIN}-<hash> byte-identical to the packaged "
            f"binary, found {len(matches)}"
        )
    unit_hash = matches[0].name[len(BIN) + 1:]
    record = release / ".fingerprint" / f"{PACKAGE}-{unit_hash}" / f"bin-{BIN}.json"
    if not record.is_file():
        raise Unreadable(f"{record} not found: cargo left no record of how {matches[0].name} was built")
    try:
        flags = json.loads(record.read_text()).get("rustflags")
    except json.JSONDecodeError as e:
        raise Unreadable(f"{record}: not JSON: {e}")
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        raise Unreadable(f"{record}: no rustflags list")
    return record, flags


def target_features(flags):
    return [f for f in flags if f.startswith("-Ctarget-feature=")]


def release_rustflags(root):
    rest = split_cpu(config_rustflags(root))[1]
    if target_features(rest):
        raise Refused(f".cargo/config.toml sets {target_features(rest)}: the release binary "
                      f"enables no feature beyond {BASELINE_CPU}'s")
    return ["-Ctarget-cpu=" + BASELINE_CPU] + rest


def check(tarball, root):
    record, flags = recorded_rustflags(root, packaged_binary(tarball))
    cpus, rest = split_cpu(flags)
    want_rest = split_cpu(config_rustflags(root))[1]
    if cpus != [BASELINE_CPU]:
        print(f"release-cpu: FAIL {tarball.name} was built with target-cpu {cpus}, "
              f"expected exactly ['{BASELINE_CPU}'] (per {record.relative_to(root)})")
        return 1
    if target_features(rest):
        print(f"release-cpu: FAIL {tarball.name} was built with {target_features(rest)}: "
              f"no feature beyond {BASELINE_CPU}'s (per {record.relative_to(root)})")
        return 1
    if rest != want_rest:
        print(f"release-cpu: FAIL {tarball.name} was built with flags {rest} besides the "
              f"target CPU; .cargo/config.toml sets {want_rest} (per {record.relative_to(root)})")
        return 1
    print(f"release-cpu: ok {tarball.name} built with target-cpu={BASELINE_CPU} and {rest}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rustflags", parents=[common])
    c = sub.add_parser("check", parents=[common])
    c.add_argument("tarball", type=Path)
    args = ap.parse_args()
    try:
        if args.cmd == "rustflags":
            print("\x1f".join(release_rustflags(args.root)), end="")
            return 0
        return check(args.tarball, args.root)
    except Refused as e:
        print(f"release-cpu: FAIL {e}", file=sys.stderr)
        return 1
    except (Unreadable, OSError, tarfile.TarError) as e:
        print(f"release-cpu: unavailable: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
