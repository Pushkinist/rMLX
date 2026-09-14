#!/usr/bin/env python3
"""Fit the published protocol's fixed-length prompt against a live server.

The protocol reports autoregressive output speed, input speed and resident
memory on ONE prompt of a stated token length. That length is a property of the
prompt AND of the tokenizer that reads it: a body of bytes is 1355 tokens to one
checkpoint's tokenizer and something else to the next, and this repository ships
no tokenizer to settle it offline. Checking in a body and calling it "the
1355-token prompt" would therefore be a claim nothing verified — so the body is
not checked in. It is derived, here, from three things that are:

  the corpus   a prompt file already in `prompts/`, pinned by the digest of the
               text this fit read out of it.
  the rule     the longest prefix of that corpus, cut at a word boundary and
               otherwise byte-for-byte, that the server counts at or below the
               target, plus as many copies of a one-word filler as it takes to
               land exactly on it.
  the target   the token count the protocol names.

The count is the server's own, off the `usage` block of a `max_tokens=1`
request, so it is the count of the fully templated prompt — the system message,
the role markers and the thinking tags included — which is what the model
actually prefills.

The fitted body is content-addressed the way the checked-in samples are, by the
digest `rmlx_metrics` gives a prompt body, so the row recorded for it joins to
the body that produced it. The digest changes with the tokenizer; the corpus
digest, the rule and the target do not, which is what makes the fit
reproducible rather than merely recorded.

WHY IT CAN FAIL RATHER THAN ROUND. A prefix one word longer can add more than
one token, so not every target is reachable from every corpus. Landing near the
target and calling it the target is the failure this whole module exists to
avoid, so a fit that cannot land exactly says so with the counts it reached.

Exit codes: 0 — fitted exactly; 1 — the target is not reachable, or the server
would not answer; 2 — an input could not be read.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from published_samples import body_sha256  # noqa: E402

# Pinned so the fit is a function of the corpus, the target and the tokenizer,
# and of nothing an operator typed.
INSTRUCTION = "Summarise the following document in a few sentences.\n\n"
FILLER_WORD = "the"
WORD = re.compile(r"\S+")


class FitError(Exception):
    """The target is not reachable from this corpus on this tokenizer."""


class Unreadable(Exception):
    """An input could not be read at all."""


def corpus_text(path):
    """The user message of a prompt file, as the text the prefix is cut from."""
    try:
        doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Unreadable(f"cannot read {path}: {exc}") from exc
    users = [m for m in doc.get("messages", []) if m.get("role") == "user"]
    if len(users) != 1:
        raise Unreadable(
            f"{path} holds {len(users)} user messages; the fixed prompt is cut "
            "from exactly one"
        )
    return users[0]["content"]


def body_for(text, ends, words, filler_reps):
    """The messages array for a prefix of `words` words plus `filler_reps` fillers."""
    content = INSTRUCTION + text[: ends[words - 1]]
    if filler_reps:
        content += (" " + FILLER_WORD) * filler_reps
    return [{"role": "user", "content": content}]


class Counter:
    """The server's own token count of a body, and how many it was asked for."""

    def __init__(self, url, model_id, timeout):
        self.url = url
        self.model_id = model_id
        self.timeout = timeout
        self.probes = 0
        self._seen = {}

    def __call__(self, messages):
        key = messages[0]["content"]
        if key in self._seen:
            return self._seen[key]
        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": 1,
                "enable_thinking": True,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                doc = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise FitError(f"the server would not answer a probe: {exc}") from exc
        self.probes += 1
        count = (doc.get("usage") or {}).get("prompt_tokens")
        if not isinstance(count, int) or count <= 0:
            raise FitError(
                f"the server answered a probe with prompt_tokens={count!r}; the "
                "fit has nothing to search on"
            )
        self._seen[key] = count
        return count


def fit(text, target, count_of):
    """`(words, filler_reps, prompt_tokens)` landing exactly on `target`."""
    ends = [m.end() for m in WORD.finditer(text)]
    if not ends:
        raise Unreadable("the corpus holds no word to cut a prefix from")

    full = count_of(body_for(text, ends, len(ends), 0))
    if full < target:
        raise FitError(
            f"the whole corpus counts {full} tokens on this tokenizer, below the "
            f"{target} the protocol names; the fixed prompt cannot be cut from it"
        )
    if count_of(body_for(text, ends, 1, 0)) > target:
        raise FitError(
            f"a one-word prefix already counts more than {target} tokens on this "
            "tokenizer"
        )

    # The largest prefix at or below the target. Token count is non-decreasing
    # in the prefix length, which is what makes this a search rather than a scan.
    lo, hi = 1, len(ends)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_of(body_for(text, ends, mid, 0)) <= target:
            lo = mid
        else:
            hi = mid - 1
    words = lo
    base = count_of(body_for(text, ends, words, 0))
    residual = target - base
    if residual == 0:
        return words, 0, base

    # One filler word costs whatever this tokenizer charges for it. Measured
    # rather than assumed: " the" is one token in most BPE vocabularies and the
    # fit must not be silently wrong on the one where it is not.
    per_filler = count_of(body_for(text, ends, words, 1)) - base
    if per_filler <= 0 or residual % per_filler != 0:
        raise FitError(
            f"the longest prefix at or below the target counts {base} tokens, "
            f"{residual} short of {target}, and one filler word costs "
            f"{per_filler} on this tokenizer, so no number of them lands on it. "
            "The target is not reachable from this corpus; choose another corpus "
            "or another target rather than publishing a near miss."
        )
    # Joining the filler onto the prefix can cost a token the marginal price of
    # a second filler does not, so the first count is corrected rather than
    # trusted. Bounded: a correction that does not converge is a tokenizer whose
    # filler cost is not constant, which is a refusal and not something to
    # iterate at.
    reps = residual // per_filler
    landed = None
    for _ in range(3):
        landed = count_of(body_for(text, ends, words, reps))
        if landed == target:
            return words, reps, landed
        step = (target - landed) // per_filler
        if step == 0 or reps + step < 1:
            break
        reps += step
    raise FitError(
        f"{words} corpus words plus {reps} filler words count {landed} tokens, "
        f"not {target}: the filler's cost is not constant on this tokenizer, so "
        "the fit cannot be completed by adding more of it"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, help="a prompt file under prompts/")
    ap.add_argument("--target", type=int, required=True, help="the token count to hit")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", required=True, help="where to write the fit record")
    ap.add_argument("--payload", required=True, help="where to write the request")
    ap.add_argument("--max-tokens", type=int, required=True, help="the measured budget")
    args = ap.parse_args()

    try:
        text = corpus_text(args.corpus)
    except Unreadable as exc:
        print(f"published_fixed_prompt: {exc}", file=sys.stderr)
        return 2

    counter = Counter(
        f"http://127.0.0.1:{args.port}/v1/chat/completions", args.model_id, args.timeout
    )
    try:
        words, reps, prompt_tokens = fit(text, args.target, counter)
    except FitError as exc:
        print(f"published_fixed_prompt: {exc}", file=sys.stderr)
        return 1
    except Unreadable as exc:
        print(f"published_fixed_prompt: {exc}", file=sys.stderr)
        return 2

    ends = [m.end() for m in WORD.finditer(text)]
    messages = body_for(text, ends, words, reps)
    record = {
        "target_tokens": args.target,
        "prompt_tokens": prompt_tokens,
        "body_sha256": body_sha256(messages),
        "corpus": os.path.basename(args.corpus),
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "corpus_words": len(ends),
        "instruction": INSTRUCTION,
        "words": words,
        "filler_word": FILLER_WORD,
        "filler_reps": reps,
        "probes": counter.probes,
        "max_tokens": args.max_tokens,
        "messages": messages,
    }
    pathlib.Path(args.out).write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )
    pathlib.Path(args.payload).write_text(
        json.dumps(
            {
                "model": args.model_id,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "enable_thinking": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"fitted {prompt_tokens} tokens: {words} corpus words + {reps} filler, "
        f"{counter.probes} probes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
