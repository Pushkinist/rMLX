"""Line-level overlap of two texts: digit-folded lines and the matched-line count.

Module of: report/debt_report.py

The one producer of the figure `debt_report.py --matched-lines` prints and the
scripts gate's twin rule reads.
"""

from __future__ import annotations

import difflib
import re

# A digit run is folded to a single 'N' whenever it touches an identifier
# character on either side — `rotor3` / `rotor4` -> `rotorN`, `K3` -> `KN`,
# `3u8` -> `Nu8`, `4-bit` -> `N-bit`. A bare number with no adjacent letter
# (a line count in a comment, say) is left alone.
_DIGIT_RUN = re.compile(r"(?<=[A-Za-z_])\d+|\d+(?=[A-Za-z_-])")


def normalize(text: str) -> str:
    return _DIGIT_RUN.sub("N", text)


def lines_of(text: str) -> list[str]:
    """The lines of a text. Every reader of the figure splits one way, so a
    count taken over these lines bounds the figure from above."""
    return text.splitlines()


def matched_lines(a: str, b: str) -> int:
    """Summed size of every matching block `difflib.SequenceMatcher` finds
    between `a` and `b`, over the same digit-folded, `autojunk=False` lines
    `debt_report.similarity()` compares — a line count, not a ratio. This is
    the figure the round-loop migration chunks reported by hand;
    `--matched-lines` is its one producer now.

    Measured both ways round and reported as the larger. `SequenceMatcher` is
    not symmetric: it anchors on the longest match it finds in `a` and can
    reach a smaller total with the arguments swapped. A one-directional
    measure would therefore move when a body changed file, or was renamed, or
    when a population gained an item that re-ordered it — with no line of any
    body changing. The larger of the two is the count of lines the pair
    genuinely shares, and it depends on the pair alone."""
    a_lines = lines_of(normalize(a))
    b_lines = lines_of(normalize(b))
    forward = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    reverse = difflib.SequenceMatcher(None, b_lines, a_lines, autojunk=False)
    return max(
        sum(block.size for block in forward.get_matching_blocks()),
        sum(block.size for block in reverse.get_matching_blocks()),
    )
