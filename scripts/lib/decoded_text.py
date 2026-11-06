"""Read the generated text out of a `rmlx baseline` stderr log.

`baseline` prints the text once, as the tracing field `decoded="..."`. The field is a Rust Debug
string, so `"`, `\\`, newline and tab arrive escaped, and the subscriber may wrap it in ANSI colour
codes. A log with several such fields yields the last.
"""

from __future__ import annotations

import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")
DECODED = re.compile(r'decoded="((?:[^"\\]|\\.)*)"')
DEBUG_ESCAPE = re.compile(r'\\(["\\nt])')
UNESCAPED = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}


def decoded_text(log: str) -> str | None:
    """The text of the last `decoded` field of a run's stderr log, or None when it has none."""
    fields = DECODED.findall(ANSI.sub("", log))
    if not fields:
        return None
    return DEBUG_ESCAPE.sub(lambda m: UNESCAPED[m[1]], fields[-1])
