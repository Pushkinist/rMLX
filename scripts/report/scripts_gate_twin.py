"""The twin rule of the scripts gate: two files of one language that say the same lines.

Module of: report/debt_report.py

A line is significant when it is not blank, not a comment and not a bare
closer (`fi`, `done`, `}`). The figure of a pair is `matched_lines` over the
significant lines of both files, digit-folded, in the larger direction. A pair
is a twin at `TWIN_LINES` matched lines, or at half of the smaller file's
significant lines when that file has at least `TWIN_RATIO_FLOOR` of them: below
that, a header and a few lines of boilerplate would make every small pair one.

`difflib` costs too much to run over every pair. The size of the multiset
intersection of two files' significant lines bounds `matched_lines` from above
(a matching block pairs equal lines, each used once), so a pair under the bound
is not measured.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator

from line_match import matched_lines, normalize
from scripts_gate_files import Script, Violation
from scripts_gate_rules import Context

TWIN_LINES = 40
TWIN_RATIO_FLOOR = 20
LANGUAGES = (".sh", ".py")
CLOSER_RE = re.compile(r'(?:fi|done|esac|else|then|do|\{|\}|\(|\)|;;|EOF|pass|"""|\'\'\')')


def significant_lines(f: Script) -> list[str]:
    lines = (raw.strip() for raw in f.text.split("\n"))
    return [normalize(s) for s in lines if s and not s.startswith("#") and not CLOSER_RE.fullmatch(s)]


def twin_threshold(smaller: int) -> int:
    """Matched lines at which a pair with a smaller file of `smaller`
    significant lines is a twin."""
    if smaller < TWIN_RATIO_FLOOR:
        return TWIN_LINES
    return min(TWIN_LINES, (smaller + 1) // 2)


def rule_twin(ctx: Context) -> Iterator[Violation]:
    for language in LANGUAGES:
        group = sorted((f for f in ctx.files if f.rel.endswith(language)), key=lambda f: f.rel)
        lines = {f.rel: significant_lines(f) for f in group}
        counts = {rel: Counter(sig) for rel, sig in lines.items()}
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                need = twin_threshold(min(len(lines[a.rel]), len(lines[b.rel])))
                if sum((counts[a.rel] & counts[b.rel]).values()) < need:
                    continue
                matched = matched_lines("\n".join(lines[a.rel]), "\n".join(lines[b.rel]))
                if matched >= need:
                    detail = (
                        f"{matched} matched significant lines with {b.rel} "
                        f"({len(lines[a.rel])} and {len(lines[b.rel])} significant lines)"
                    )
                    yield Violation("twin", a.rel, 1, detail, figure=matched, pin_path=f"{a.rel}+{b.rel}")
