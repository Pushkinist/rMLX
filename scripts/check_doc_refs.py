#!/usr/bin/env python3
"""Every reference into docs/ resolves, and a doc edit re-points none of them.

A reference is text in a tracked file that names a doc under `docs/`, and
maybe something inside it:

  PATH     a doc path (`docs/<path>.md`) or a bare top-level doc name
           (`KV_QUANT.md`); the file must exist
  LINK     a Markdown link or link definition whose target is under docs/, or
           any relative link written in a docs/ file; a `#anchor` must be a
           heading slug or an explicit id in the target
  ANCHOR   `<DOC>.md#anchor`; the anchor must exist
  SECTION  `<DOC>.md §N` must name a heading numbered N. `<DOC>.md` followed by
           `§ "Phrase"`, § and a backticked identifier, `section "Phrase"`,
           `under "Phrase"` or `§ Some Words` must name a heading that starts
           with the phrase; an unquoted run of words may also run on past the
           heading title
  QUOTE    `<DOC>.md "Phrase"`: a phrase that names a heading must keep naming
           a heading; any other phrase must occur in the doc
           Inside a doc, a `§` with no doc name before it cites that doc.
  LINE     `<DOC>.md:NNN` must be a line of the doc; a quoted phrase after it
           is checked as a QUOTE too
  MAP      every top-level docs/*.md has a `CLAUDE.md` documentation-map row
           whose label and link name the same file (MAPROW: a row whose label
           and link differ)

Phrases compare with backticks, asterisks and runs of whitespace removed, and
section phrases compare without case.

Readers it does parse, and so holds a cut to: every citation above in code,
scripts, docs, `README.md` and `CLAUDE.md`; the section names in CLI help
text (`crates/rmlx-cli/src/main.rs`, `serve.rs`, `preset_table.rs`); the
emitter `scripts/lib/published_table.py`, which writes `PROFILING.md` §9 into
a generated doc; and the tests that pin a figure the docs record, whose
failure messages cite the doc and the phrase that states the figure
(`isoquant_msl_tests.rs`, `isoquant_msl_v4_tests.rs`, `rate_distortion_tests.rs`,
`rot_k_msl_tests.rs`, `metal_kernel_tests.rs`). A new pinned test cites its
doc the same way, or no gate sees a cut that deletes its figure.

Readers of doc text that this script does not parse have their own gates:
the INERT banners of docs/KV_QUANT.md (check_kv_codec_disposition.sh), the
`--kv-boundary-layers` rows of docs/CLI.md (check_kv_boundary_default_parity.sh),
the generated docs/PUBLISHED_PROTOCOL.md (make check-published-table), and the
`crates/...` citations in every doc (check_doc_source_citations.sh).
`make check-doc-consumers` runs all five.

Without --base, every reference in the tree is checked and any broken one
fails. With --base REF, the working tree is compared with the merge-base of
HEAD and REF, "the base". A reference fails if it resolved at the base and
does not resolve now, or if it now resolves to a different target: a heading
with another title, a cited line with other text, a heading phrase that is
now only body text. A reference already broken at the base is carried: it is
printed and does not fail. Carried references are counted per target, so
moving a broken citation with its code passes and a second copy of one fails.
`--base auto` picks the nearest of origin/main and origin/next/*: the ref
whose merge-base leaves the fewest commits on HEAD, among refs that do not
already contain HEAD; with none, HEAD itself. Every run prints its base.

`CHANGELOG.md` is released history and is never edited for a doc cut. Its
references are scanned and a broken one is printed, but it never fails.

A file named `scripts/<name>_selftest.sh` or `scripts/<name>_fixtures.sh`
whose first ten lines carry a comment line `doc-refs: fixture` is not
scanned: its doc paths belong to a synthetic tree it builds. The marker counts
nowhere else. With --base, a file that existed at REF and gains the marker
fails. Every run names the files it skipped.

This reads text, not meaning. It cannot tell whether the text a reference
lands on still says what the citing sentence claims, and it cannot see a
reference spelled in a form it does not parse.

Exit 0 = pass, 1 = a broken or re-pointed reference, 2 = could not run.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TEXT_SUFFIXES = {".md", ".rs", ".sh", ".py", ".toml", ".txt", ".yml", ".yaml", ".sql", ".metal"}
TEXT_NAMES = {"Makefile"}
RELEASED_HISTORY = "CHANGELOG.md"

DOC_MENTION = re.compile(r"(?<![\w/.-])((?:\.\./)*docs/[\w./-]+?\.md|[A-Z][A-Z0-9_]+\.md)")
LEAD = r"`?\)?,? ?"
# A phrase in double quotes; the quotes may be escaped, as inside a string literal.
QUOTED = r'\\?"([A-Za-z`](?:[^"\\]|\\\n){2,240})\\?"'
AFTER_ANCHOR = re.compile(r"#([A-Za-z0-9_-]+)")
AFTER_LINE = re.compile(r":(\d+)\b")
QUOTE_AFTER_LINE = re.compile(r"\s*\(?\s*" + QUOTED)
SECTION_MARK = r"§[ \t]*(?:\n[ \t]*(?://[/!]?|#|>|--)?[ \t]*)?"
AFTER_SECTION_NUMBER = re.compile(LEAD + SECTION_MARK + r"([0-9]+(?:\.[0-9]+)*)(?![\w-])")
AFTER_SECTION_PHRASE = re.compile(
    LEAD + r"(?:" + SECTION_MARK + r"|,? ?(?:in )?sections? |,? ?under )"
    r"(?:" + QUOTED + r"|`([^`\n]+)`|\*([A-Za-z`][^*]{2,240})\*)"
)
AFTER_SECTION_WORDS = re.compile(LEAD + SECTION_MARK + r"([A-Za-z][\w:-]*(?: [A-Za-z0-9][\w:-]*)*)")
AFTER_QUOTE = re.compile(LEAD + QUOTED)
INLINE_LINK = re.compile(r"\]\(\s*<?([^()\s<>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
LINK_DEF = re.compile(r"^\s*(?://[/!]?\s*|#\s*)?\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.M)
MAP_ROW = re.compile(r"^\| \[`([^`]+)`\]\(([^)]+)\)", re.M)
COMMENT_LEADER = re.compile(r"\\?\n\s*(?://[/!]?|#|\*|>|--)?\s*")
FIXTURE_PATH = re.compile(r"^scripts/[^/]+_(?:selftest|fixtures)\.sh$")
FIXTURE_MARKER = re.compile(r"^\s*(?:#|//)\s*doc-refs: fixture\b", re.M)
EXPLICIT_ID = re.compile(r"""<a\s+(?:id|name)=["']([^"']+)["']""")


class CannotRun(Exception):
    pass


def git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True, check=False)
    if proc.returncode != 0:
        raise CannotRun(f"git {' '.join(args)}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def is_text(path: str) -> bool:
    p = PurePosixPath(path)
    return p.suffix in TEXT_SUFFIXES or p.name in TEXT_NAMES


def worktree_files(root: Path) -> tuple[dict[str, str], set[str]]:
    listed = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    names = {n for n in listed.decode().split("\0") if n and (root / n).is_file()}
    files = {n: (root / n).read_text(encoding="utf-8", errors="replace") for n in names if is_text(n)}
    return files, names


def ref_files(root: Path, ref: str) -> tuple[dict[str, str], set[str]]:
    all_names = {n for n in git(root, "ls-tree", "-r", "-z", "--name-only", ref).decode().split("\0") if n}
    names = sorted(n for n in all_names if is_text(n))
    if not names:
        return {}, all_names
    out = git(root, "cat-file", "--batch", stdin="".join(f"{ref}:{n}\n" for n in names).encode())
    files, pos = {}, 0
    for name in names:
        header_end = out.index(b"\n", pos)
        header = out[pos:header_end].split()
        if len(header) < 3 or header[1] != b"blob":
            raise CannotRun(f"cannot read {ref}:{name}")
        size = int(header[2])
        start = header_end + 1
        files[name] = out[start : start + size].decode("utf-8", errors="replace")
        pos = start + size + 1
    return files, all_names


def nearest_base(root: Path) -> tuple[str, str]:
    """(ref, merge-base sha) for `--base auto`; see the module docstring."""
    head = git(root, "rev-parse", "HEAD").decode().strip()
    refs = git(root, "for-each-ref", "--format=%(refname)", "refs/remotes/origin/main", "refs/remotes/origin/next/")
    candidates = refs.decode().split()
    if not candidates:
        raise CannotRun("--base auto: no origin/main or origin/next/* ref; fetch, or pass a base")
    best = None
    for ref in sorted(candidates):
        merge_base = git(root, "merge-base", "HEAD", ref).decode().strip()
        if merge_base == head:
            continue
        ahead = int(git(root, "rev-list", "--count", f"{merge_base}..HEAD").decode())
        if best is None or ahead < best[0]:
            best = (ahead, ref.removeprefix("refs/remotes/"), merge_base)
    return (best[1], best[2]) if best else ("HEAD", head)


def slug(heading: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.replace("`", "").replace("*", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def squash(text: str) -> str:
    return " ".join(text.split())


def plain(text: str) -> str:
    """Text as a phrase compares: comment leaders, backticks and asterisks gone."""
    text = COMMENT_LEADER.sub(" ", text).replace("`", "").replace("*", "")
    return squash(text).rstrip(".,;:")


def unfenced(text: str) -> str:
    """The text with fenced code blocks blanked, offsets kept."""
    out, fence = [], None
    for line in text.split("\n"):
        marker = line.lstrip()[:3]
        if marker in ("```", "~~~"):
            fence = None if fence == marker else (fence or marker)
            out.append(" " * len(line))
        else:
            out.append(" " * len(line) if fence else line)
    return "\n".join(out)


def heading_words(title: str) -> tuple[str, str]:
    """(leading section number, plain title with the number removed)."""
    text = plain(title).lstrip("§").strip()
    m = re.match(r"([0-9]+(?:\.[0-9]+)*)\.?\s+(.*)", text)
    return (m.group(1), m.group(2)) if m else ("", text)


@dataclass
class Doc:
    lines: list[str]
    headings: list[str]
    anchors: set[str]
    flat: str

    def heading_named(self, phrase: str, runs_on: bool = False) -> str | None:
        """The first heading that starts with the phrase. With `runs_on` (an
        unquoted name, which may run on into the sentence), else the longest
        heading the phrase starts with."""
        want = phrase.lower()
        titles = [(heading_words(t)[1].lower(), t) for t in self.headings]
        for name, title in titles:
            if name.startswith(want):
                return title
        if not runs_on:
            return None
        inside = [(len(name), title) for name, title in titles if name and (want + " ").startswith(name + " ")]
        return max(inside)[1] if inside else None


def parse_doc(text: str) -> Doc:
    headings, anchors, seen = [], set(), Counter()
    for line in unfenced(text).split("\n"):
        m = re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            title = m.group(2)
            headings.append(title)
            base = slug(title)
            anchors.add(base if seen[base] == 0 else f"{base}-{seen[base]}")
            seen[base] += 1
        anchors.update(EXPLICIT_ID.findall(line))
    return Doc(text.split("\n"), headings, anchors, plain(text))


class Tree:
    def __init__(self, listing: tuple[dict[str, str], set[str]]):
        self.files, self.names = listing
        self.docs = {n: parse_doc(t) for n, t in self.files.items() if n.startswith("docs/") and n.endswith(".md")}
        self.fixtures = sorted(
            n
            for n, t in self.files.items()
            if FIXTURE_PATH.match(n) and FIXTURE_MARKER.search("\n".join(t.split("\n")[:10]))
        )


@dataclass(frozen=True)
class Ref:
    citing: str
    kind: str
    doc: str
    key: str
    runs_on: bool = False

    @property
    def target(self) -> tuple[str, str, str]:
        return (self.kind, self.doc, self.key)


def resolve(ref: Ref, tree: Tree) -> str | None:
    """The identity of what the reference lands on, or None when it lands on nothing."""
    if ref.kind == "MAPROW":
        return None
    if ref.kind == "MAP":
        text = tree.files.get("CLAUDE.md", "")
        mapped = {label for label, target in MAP_ROW.findall(text) if label == target}
        return ref.doc if ref.doc in mapped else None
    if ref.kind == "LINK" and not (ref.doc.startswith("docs/") and ref.doc.endswith(".md")):
        target = ref.doc.rstrip("/")
        return ref.doc if target in tree.names or any(n.startswith(target + "/") for n in tree.names) else None
    doc = tree.docs.get(ref.doc)
    if doc is None:
        return None
    if ref.kind == "PATH" or (ref.kind == "LINK" and not ref.key):
        return ref.doc
    if ref.kind in ("LINK", "ANCHOR"):
        return ref.key if ref.key in doc.anchors else None
    if ref.kind == "SECTION":
        if ref.key[0].isdigit():
            return next((t for t in doc.headings if heading_words(t)[0] == ref.key), None)
        return doc.heading_named(ref.key, ref.runs_on)
    if ref.kind == "QUOTE":
        title = doc.heading_named(ref.key)
        if title is not None:
            return f"heading: {title}"
        return "body text" if ref.key in doc.flat else None
    if ref.kind == "LINE":
        n = int(ref.key)
        return doc.lines[n - 1] if 0 < n <= len(doc.lines) else None
    raise AssertionError(ref.kind)


def doc_path_of(name: str, doc_names: set[str]) -> str | None:
    while name.startswith("../"):
        name = name[3:]
    if name.startswith("docs/"):
        return name
    candidate = f"docs/{name}"
    return candidate if candidate in doc_names else None


def normalise(target: str, citing: str) -> str:
    parts: list[str] = []
    for part in (PurePosixPath(citing).parent / target).parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def quoted(m: re.Match) -> str | None:
    """The phrase of a QUOTED match, or None when it runs over more than three lines."""
    phrase = next(g for g in m.groups() if g is not None)
    return plain(phrase) if phrase.count("\n") <= 2 else None


def refs_after(tail: str) -> list[tuple[str, str, bool]]:
    """(kind, key, runs_on) for what follows a doc name."""
    if a := AFTER_ANCHOR.match(tail):
        return [("ANCHOR", a.group(1), False)]
    if a := AFTER_LINE.match(tail):
        found = [("LINE", a.group(1), False)]
        if (q := QUOTE_AFTER_LINE.match(tail, a.end())) and (phrase := quoted(q)):
            found.append(("QUOTE", phrase, False))
        return found
    if a := AFTER_SECTION_NUMBER.match(tail):
        return [("SECTION", a.group(1), False)]
    for pattern, kind, runs_on in (
        (AFTER_SECTION_PHRASE, "SECTION", False),
        (AFTER_SECTION_WORDS, "SECTION", True),
        (AFTER_QUOTE, "QUOTE", False),
    ):
        if (a := pattern.match(tail)) and (phrase := quoted(a)):
            return [(kind, phrase, runs_on)]
    return []


def collect(tree: Tree, doc_names: set[str]) -> list[tuple[Ref, int]]:
    """Every reference in `tree`; a bare doc name resolves against `doc_names`. MAP refs carry line 0."""
    refs: list[tuple[Ref, int]] = []
    for citing, text in tree.files.items():
        if citing in tree.fixtures:
            continue
        in_docs = citing.startswith("docs/")
        link_text = unfenced(text) if citing.endswith(".md") else text
        links = [(m.group(1), m.start()) for m in INLINE_LINK.finditer(link_text)]
        links += [(m.group(1), m.start(1)) for m in LINK_DEF.finditer(link_text)]
        for target, pos in links:
            if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("/"):
                continue
            path, _, anchor = target.partition("#")
            resolved = normalise(path, citing) if path else citing
            if not (in_docs or resolved.startswith("docs/")):
                continue
            if anchor and not resolved.endswith(".md"):
                anchor = ""
            refs.append((Ref(citing, "LINK", resolved, anchor), line_of(text, pos)))
        if in_docs and citing.endswith(".md"):
            for m in re.finditer("§", link_text):
                if ".md" in link_text[max(0, m.start() - 8) : m.start()]:
                    continue
                for kind, key, runs_on in refs_after(link_text[m.start() : m.start() + 400]):
                    refs.append((Ref(citing, kind, citing, key, runs_on), line_of(text, m.start())))
        for m in DOC_MENTION.finditer(text):
            doc = doc_path_of(m.group(1), doc_names)
            if doc is None:
                continue
            line = line_of(text, m.start())
            refs.append((Ref(citing, "PATH", doc, ""), line))
            for kind, key, runs_on in refs_after(text[m.end() : m.end() + 400]):
                refs.append((Ref(citing, kind, doc, key, runs_on), line))
    if "CLAUDE.md" in tree.files:
        text = tree.files["CLAUDE.md"]
        for m in MAP_ROW.finditer(text):
            if m.group(1) != m.group(2):
                refs.append((Ref("CLAUDE.md", "MAPROW", m.group(2), m.group(1)), line_of(text, m.start())))
        for doc in sorted(tree.docs):
            if doc.count("/") == 1:
                refs.append((Ref("CLAUDE.md", "MAP", doc, ""), 0))
    return refs


def describe(ref: Ref, line: int) -> str:
    where = f"{ref.citing}:{line}" if line else f"{ref.citing} (documentation map)"
    key = f" {ref.key!r}" if ref.key else ""
    return f"{where}: {ref.kind} {ref.doc}{key}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="git work tree to check")
    parser.add_argument("--base", help="git ref to compare the working tree with, or 'auto'")
    parser.add_argument("--list", action="store_true", help="print every reference and what it resolves to")
    args = parser.parse_args(argv)
    root = Path(args.root)

    try:
        head = Tree(worktree_files(root))
        base_ref = args.base
        if args.base == "auto":
            name, base_ref = nearest_base(root)
            print(f"base: {name} (merge-base {base_ref[:12]})")
        elif args.base:
            git(root, "rev-parse", "--verify", args.base + "^{commit}")
            base_ref = git(root, "merge-base", "HEAD", args.base).decode().strip()
            print(f"base: {args.base} (merge-base {base_ref[:12]})")
        base = Tree(ref_files(root, base_ref)) if base_ref else None
    except CannotRun as err:
        print(f"check_doc_refs: could not run: {err}", file=sys.stderr)
        return 2
    if not head.docs:
        print(f"check_doc_refs: could not run: no tracked docs/*.md under {root}", file=sys.stderr)
        return 2

    names = set(head.docs) | (set(base.docs) if base else set())
    head_refs = collect(head, names)

    if args.list:
        for ref, line in head_refs:
            got = resolve(ref, head)
            shown = "BROKEN" if got is None else squash(got)[:80]
            print(f"{ref.kind}\t{ref.citing}:{line}\t{ref.doc}\t{ref.key}\t{shown}")
        return 0

    base_broken: Counter[tuple[str, str, str]] = Counter()
    if base:
        for ref, _ in collect(base, names):
            if resolve(ref, base) is None:
                base_broken[ref.target] += 1

    failures, carried, history = [], [], []
    if base:
        for name in head.fixtures:
            if name in base.files and name not in base.fixtures:
                failures.append(f"  {name}: gained the 'doc-refs: fixture' marker, which hides its references")
    seen: Counter[tuple[str, str, str]] = Counter()
    for ref, line in head_refs:
        got = resolve(ref, head)
        before = resolve(ref, base) if base else None
        if got is None:
            problem = "resolves to nothing"
        elif before is not None and before != got:
            problem = f"re-pointed: was {squash(before)[:70]!r}, now {squash(got)[:70]!r}"
        else:
            continue
        if ref.citing == RELEASED_HISTORY:
            history.append(f"  {describe(ref, line)} — {problem}")
            continue
        if got is None:
            seen[ref.target] += 1
            if seen[ref.target] <= base_broken[ref.target]:
                carried.append(f"  {describe(ref, line)}")
                continue
        failures.append(f"  {describe(ref, line)} — {problem}")

    if carried:
        print(f"note: {len(carried)} reference(s) already broken at the base, carried:")
        print("\n".join(carried))
    if history:
        print(f"note: {len(history)} broken reference(s) in {RELEASED_HISTORY} (released history; never fails):")
        print("\n".join(history))
    if head.fixtures:
        print(f"not scanned ({len(head.fixtures)} file(s) marked 'doc-refs: fixture'): {', '.join(head.fixtures)}")
    if failures:
        print("ERROR: references into docs/ that a doc edit broke:" if base else "ERROR: broken references into docs/:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    scope = "none broken or re-pointed since the base" if base else "all resolve"
    print(f"OK: {len(head_refs)} references into docs/ — {scope}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
