#!/usr/bin/env python3
"""Every reference into docs/ resolves, and a doc edit re-points none of them.

A reference is anything in a tracked file that names a doc under `docs/` and
something inside it:

  LINK     a Markdown link or link definition whose target is under docs/, or
           any relative link written in a docs/ file; with a `#anchor`, the
           anchor must be a heading slug or an explicit id in the target
  PATH     a bare `docs/<path>.md` mention; the file must exist
  ANCHOR   a bare `<DOC>.md#anchor`; the anchor must exist
  SECTION  `<DOC>.md §N` or `<DOC>.md §Name`; a heading must carry that number,
           or start with that word
  QUOTE    `<DOC>.md "Some phrase"`; the phrase must occur in the doc
  LINE     `<DOC>.md:NNN`; the line must exist
  MAP      every top-level docs/*.md has a `CLAUDE.md` documentation-map row
           whose label and link name the same file (MAPROW: a row whose label
           and link differ)

Without --base, every reference in the tree is checked and any broken one
fails. With --base REF, the working tree is compared with REF: a reference
fails if it resolved at REF and does not resolve now, or if it now resolves to
a different target (a numbered section whose title changed, a cited line whose
text changed). A reference that was already broken at REF is reported as
carried and does not fail. Carried references are counted per target, not per
citing file: moving a broken citation with the code around it passes, and a
second copy of one fails.

A file whose first ten lines carry a comment line `doc-refs: fixture` is not
scanned: its doc paths belong to a synthetic tree it builds, not to this one.
Every run prints the files it skipped.

This reads text, not meaning. It cannot tell whether the text a reference
lands on still says what the citing sentence claims, and it cannot see a
reference spelled in a form it does not parse. The consumer gates that read
doc text (banners, flag rows, the generated table) are separate scripts.

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

DOC_MENTION = re.compile(r"(?<![\w/.-])((?:\.\./)*docs/[\w./-]+?\.md|[A-Z][A-Z0-9_]+\.md)")
AFTER_ANCHOR = re.compile(r"#([A-Za-z0-9_-]+)")
AFTER_LINE = re.compile(r":(\d+)\b")
AFTER_SECTION = re.compile(r"`?\)?,? ?§ ?([0-9]+(?:\.[0-9]+)*|[A-Za-z][\w-]*)")
AFTER_QUOTE = re.compile(r'`?\)?,? ?"([A-Za-z`][^"\n]{3,}(?:\n[^"\n]*){0,2}?)"')
INLINE_LINK = re.compile(r"\]\(\s*<?([^()\s<>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
LINK_DEF = re.compile(r"^\s*(?://[/!]?\s*|#\s*)?\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.M)
MAP_ROW = re.compile(r"^\| \[`([^`]+)`\]\(([^)]+)\)", re.M)
COMMENT_LEADER = re.compile(r"\n\s*(?://[/!]?|#|\*|>)?\s*")
FIXTURE_MARKER = re.compile(r"^\s*(?:#|//)\s*doc-refs: fixture\b", re.M)
EXPLICIT_ID = re.compile(r"""<a\s+(?:id|name)=["']([^"']+)["']""")


class CannotRun(Exception):
    pass


def git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(root), *args], input=stdin, capture_output=True, check=False
    )
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


def slug(heading: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.replace("`", "").replace("*", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def squash(text: str) -> str:
    return " ".join(text.split())


@dataclass
class Doc:
    lines: list[str]
    headings: list[str]
    anchors: set[str]
    flat: str


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
    return Doc(text.split("\n"), headings, anchors, squash(text))


def heading_words(title: str) -> tuple[str, str]:
    """(leading section number, title with the number removed)."""
    text = title.replace("`", "").replace("*", "").strip().lstrip("§").strip()
    m = re.match(r"([0-9]+(?:\.[0-9]+)*)\.?\s+(.*)", text)
    return (m.group(1), m.group(2)) if m else ("", text)


class Tree:
    def __init__(self, listing: tuple[dict[str, str], set[str]]):
        self.files, self.names = listing
        self.fixtures = sorted(
            n for n, t in self.files.items() if FIXTURE_MARKER.search("\n".join(t.split("\n")[:10]))
        )
        self.docs = {n: parse_doc(t) for n, t in self.files.items() if n.startswith("docs/") and n.endswith(".md")}


@dataclass(frozen=True)
class Ref:
    citing: str
    kind: str
    doc: str
    key: str

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
        want = ref.key.rstrip(".")
        for title in doc.headings:
            number, rest = heading_words(title)
            if want[0].isdigit():
                if number == want:
                    return title
            elif want.lower() in (w.lower() for w in re.split(r"[^\w-]+", rest)):
                return title
        return None
    if ref.kind == "QUOTE":
        return ref.key if ref.key in doc.flat else None
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


def collect(tree: Tree, doc_names: set[str]) -> list[tuple[Ref, int]]:
    """Every reference in `tree`; a bare doc name resolves against `doc_names`."""
    files = tree.files
    refs: list[tuple[Ref, int]] = []
    for citing, text in files.items():
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
        for m in DOC_MENTION.finditer(text):
            doc = doc_path_of(m.group(1), doc_names)
            if doc is None:
                continue
            pos, tail = m.start(), text[m.end() : m.end() + 400]
            if m.group(1).lstrip("./").startswith("docs/"):
                refs.append((Ref(citing, "PATH", doc, ""), line_of(text, pos)))
            if a := AFTER_ANCHOR.match(tail):
                refs.append((Ref(citing, "ANCHOR", doc, a.group(1)), line_of(text, pos)))
            elif a := AFTER_LINE.match(tail):
                refs.append((Ref(citing, "LINE", doc, a.group(1)), line_of(text, pos)))
            elif a := AFTER_SECTION.match(tail):
                refs.append((Ref(citing, "SECTION", doc, a.group(1)), line_of(text, pos)))
            elif a := AFTER_QUOTE.match(tail):
                phrase = squash(COMMENT_LEADER.sub(" ", a.group(1)))
                refs.append((Ref(citing, "QUOTE", doc, phrase), line_of(text, pos)))
    if "CLAUDE.md" in files:
        text = files["CLAUDE.md"]
        for m in MAP_ROW.finditer(text):
            if m.group(1) != m.group(2):
                refs.append((Ref("CLAUDE.md", "MAPROW", m.group(2), m.group(1)), line_of(text, m.start())))
        for doc in sorted(tree.docs):
            if doc.count("/") == 1:
                refs.append((Ref("CLAUDE.md", "MAP", doc, ""), 0))
    return refs


def describe(ref: Ref) -> str:
    key = f" {ref.key!r}" if ref.key else ""
    return f"{ref.kind} {ref.doc}{key}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="git work tree to check")
    parser.add_argument("--base", help="git ref to compare the working tree with")
    parser.add_argument("--list", action="store_true", help="print every reference and what it resolves to")
    args = parser.parse_args(argv)
    root = Path(args.root)

    try:
        head = Tree(worktree_files(root))
        base = Tree(ref_files(root, args.base)) if args.base else None
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

    failures, carried = [], []
    seen: Counter[tuple[str, str, str]] = Counter()
    for ref, line in head_refs:
        got = resolve(ref, head)
        where = f"{ref.citing}:{line}"
        if got is None:
            seen[ref.target] += 1
            if seen[ref.target] <= base_broken[ref.target]:
                carried.append(f"  {where}: {describe(ref)}")
            else:
                failures.append(f"  {where}: {describe(ref)} — resolves to nothing")
        elif base:
            before = resolve(ref, base)
            if before is not None and before != got:
                failures.append(
                    f"  {where}: {describe(ref)} — re-pointed: was {squash(before)[:70]!r}, now {squash(got)[:70]!r}"
                )

    if carried:
        print(f"note: {len(carried)} reference(s) already broken at {args.base}, carried:")
        print("\n".join(carried))
    if failures:
        print("ERROR: references into docs/ that a doc edit broke:" if base else "ERROR: broken references into docs/:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    scope = f"none broken or re-pointed since {args.base}" if base else "all resolve"
    print(f"OK: {len(head_refs)} references into docs/ — {scope}.")
    if head.fixtures:
        print(f"not scanned ({len(head.fixtures)} file(s) marked 'doc-refs: fixture'): {', '.join(head.fixtures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
