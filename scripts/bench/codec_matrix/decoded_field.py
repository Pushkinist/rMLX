"""Print the generated text of a `rmlx baseline` stderr log read on stdin; exit 1 when it has none.

Module of: bench/codec_matrix/runner.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from decoded_text import decoded_text  # noqa: E402


def main() -> int:
    text = decoded_text(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
    if text is None:
        print("no decoded field in the log", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
