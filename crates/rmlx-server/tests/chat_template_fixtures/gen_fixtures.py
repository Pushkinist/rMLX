#!/usr/bin/env python3
"""
Chat-template fixture generator — Part 1 of A9.

Uses the mlx-lm venv transformers to produce HF apply_chat_template golden
strings for each local snapshot.  Run with:

    <mlx-lm>/.venv/bin/python gen_fixtures.py

Each produced fixture file (JSON) is the authoritative oracle for the
corresponding rMLX round-trip test in chat_template_roundtrip.rs.

Deterministic and re-runnable: overwrites existing fixtures.
Snapshots absent on the current machine are skipped with a warning (do not
fail generation — the round-trip test also skips gracefully when absent).

Mistral / Llama3 fixtures are not generated: no local snapshot.
"""

import json
import os
import sys
from pathlib import Path

# ── Bootstrap ──────────────────────────────────────────────────────────────────

FIXTURE_DIR = Path(__file__).parent
O_MODELS = Path(os.environ.get("RMLX_O_MODELS_ROOT", str(FIXTURE_DIR / ".." / ".." / ".." / ".." / "open-models")))

# ── Helper ──────────────────────────────────────────────────────────────────────

def bos_eos_from_config(snap_dir: Path):
    """Return (bos_token, eos_token) strings from tokenizer_config.json.

    The token field may be:
      - null  → return ""
      - str   → return as-is
      - dict  { "content": "<str>", ... } → return content
    """
    cfg_path = snap_dir / "tokenizer_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    def extract(val):
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return val.get("content", "")
        return str(val)

    return extract(cfg.get("bos_token")), extract(cfg.get("eos_token"))


def render(snap_dir: Path, messages: list, tools=None, add_generation_prompt=True,
           template_kwargs=None) -> str:
    """Apply HF chat template and return the rendered string."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snap_dir), trust_remote_code=True)
    kwargs = {
        "conversation": messages,
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
        **(template_kwargs or {}),
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tok.apply_chat_template(**kwargs)


def write_fixture(name: str, snap_dir: Path, arch: str, messages: list,
                  tools=None, add_generation_prompt=True, template_kwargs=None,
                  refused=False, same_template: tuple = ()):
    """Render with HF oracle and write the fixture JSON file.

    The oracle renders `snap_dir`. `same_template` lists snapshots that ship the
    byte-identical template: the fixture names them too, and the round-trip test
    asserts the render on each one present.

    `template_kwargs` are the extra template variables the request sets
    (`reasoning_effort`, `enable_thinking`). With `refused`, the template must
    raise: the fixture records its message as `expected_error`.
    """
    bos, eos = bos_eos_from_config(snap_dir)
    outcome = {}
    try:
        outcome["expected"] = render(snap_dir, messages, tools=tools,
                                     add_generation_prompt=add_generation_prompt,
                                     template_kwargs=template_kwargs)
    except Exception as exc:
        if not refused:
            print(f"  ERROR rendering {name}: {exc}", file=sys.stderr)
            return False
        outcome["expected_error"] = str(exc)
    if refused and "expected" in outcome:
        print(f"  ERROR {name}: the template did not refuse", file=sys.stderr)
        return False

    fixture = {
        "snapshots": [snap_dir.name, *(p.name for p in same_template)],
        "arch": arch,
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": bos,
        "eos_token": eos,
    }
    if template_kwargs:
        fixture["template_kwargs"] = template_kwargs
    fixture.update(outcome)
    out_path = FIXTURE_DIR / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"  wrote {out_path.name}")
    return True


# ── Snapshot catalogue ─────────────────────────────────────────────────────────

SNAPSHOTS = {
    "qwen3_5moe": O_MODELS / "mlx-community__Qwen3.6-35B-A3B-8bit",
    "gemma4":     O_MODELS / "mlx-community__gemma-4-e4b-it-mxfp8",
    "gemma3":     O_MODELS / "mlx-community__medgemma-1.5-4b-it-8bit",
    "qwen3":      O_MODELS / "prism-ml__Ternary-Bonsai-8B-mlx-2bit",
    "qwen2":      O_MODELS / "mlx-community__jinaai-ReaderLM-v2",
    "bonsai2":    O_MODELS / "prism-ml__Ternary-Bonsai-2-27B-mlx-2bit",
}

# Snapshots that ship the same chat_template.jinja as the SNAPSHOTS entry of the
# same key. `require_same_template` checks the bytes before a fixture names them.
SAME_TEMPLATE = {
    "bonsai2": [
        O_MODELS / "mlx-community__Qwen3.8-27B-4bit",
        O_MODELS / "mlx-community__Qwen3.8-27B-mxfp8",
    ],
}

ARCHES = {
    "qwen3_5moe": "Qwen3_5MoeForConditionalGeneration",
    "gemma4":     "Gemma4ForConditionalGeneration",
    "gemma3":     "Gemma3ForConditionalGeneration",
    "qwen3":      "Qwen3ForCausalLM",
    "qwen2":      "Qwen2ForCausalLM",
    "bonsai2":    "Qwen3_5ForConditionalGeneration",
}

# ── Tool spec used for tool-calling fixtures ───────────────────────────────────

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city, e.g. Paris",
                },
            },
            "required": ["location"],
        },
    },
}


def require_same_template(snap_dir: Path, others: list) -> None:
    """Exit unless every snapshot in `others` ships snap_dir's template bytes."""
    want = (snap_dir / "chat_template.jinja").read_bytes()
    for other in others:
        path = other / "chat_template.jinja"
        if not path.exists() or path.read_bytes() != want:
            sys.exit(f"{other.name}: chat_template.jinja is absent or differs from {snap_dir.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    generated = 0
    skipped = 0

    for key, snap_dir in SNAPSHOTS.items():
        arch = ARCHES[key]
        if not snap_dir.exists():
            print(f"[SKIP] {key}: {snap_dir} absent")
            skipped += 1
            continue

        print(f"[{key}] {snap_dir.name}")
        same_template = SAME_TEMPLATE.get(key, [])
        require_same_template(snap_dir, same_template)

        if key == "qwen3_5moe":
            # simple user
            if write_fixture(
                "qwen3_5moe_basic", snap_dir, arch,
                [{"role": "user", "content": "What is the capital of France?"}],
            ):
                generated += 1

            # user + tools (tools-injection path)
            if write_fixture(
                "qwen3_5moe_tools", snap_dir, arch,
                [{"role": "user", "content": "What is the weather in Paris?"}],
                tools=[GET_WEATHER_TOOL],
            ):
                generated += 1

            # system + user
            if write_fixture(
                "qwen3_5moe_system", snap_dir, arch,
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hi"},
                ],
            ):
                generated += 1

        elif key == "gemma4":
            # user only (Gemma has no system role in its template)
            if write_fixture(
                "gemma4_no_system", snap_dir, arch,
                [{"role": "user", "content": "What is 2+2?"}],
            ):
                generated += 1

            # multi-turn
            if write_fixture(
                "gemma4_multi_turn", snap_dir, arch,
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "model", "content": "Hi there! How can I help?"},
                    {"role": "user", "content": "Tell me a joke."},
                ],
            ):
                generated += 1

        elif key == "gemma3":
            # basic user msg
            if write_fixture(
                "gemma3_basic", snap_dir, arch,
                [{"role": "user", "content": "What is the capital of France?"}],
            ):
                generated += 1

        elif key == "qwen3":
            # basic user msg
            if write_fixture(
                "qwen3_basic", snap_dir, arch,
                [{"role": "user", "content": "What is the capital of France?"}],
            ):
                generated += 1

            # user + tool
            if write_fixture(
                "qwen3_tools", snap_dir, arch,
                [{"role": "user", "content": "What is the weather in Paris?"}],
                tools=[GET_WEATHER_TOOL],
            ):
                generated += 1

        elif key == "bonsai2":
            # The template's reasoning_effort (default xhigh) and
            # enable_thinking variables, and its refusal of another effort.
            question = [{"role": "user", "content": "What is 2+2?"}]
            for name, kwargs, tools in [
                ("bonsai2_default", None, None),
                ("bonsai2_effort_medium", {"reasoning_effort": "medium"}, None),
                ("bonsai2_effort_low", {"reasoning_effort": "low"}, None),
                ("bonsai2_no_thinking", {"enable_thinking": False}, None),
                ("bonsai2_tools_effort_low", {"reasoning_effort": "low"},
                 [GET_WEATHER_TOOL]),
            ]:
                if write_fixture(name, snap_dir, arch, question, tools=tools,
                                 template_kwargs=kwargs,
                                 same_template=same_template):
                    generated += 1
            if write_fixture("bonsai2_effort_high_refused", snap_dir, arch,
                             question,
                             template_kwargs={"reasoning_effort": "high"},
                             refused=True, same_template=same_template):
                generated += 1

        elif key == "qwen2":
            # basic user msg
            if write_fixture(
                "qwen2_basic", snap_dir, arch,
                [{"role": "user", "content": "What is the capital of France?"}],
            ):
                generated += 1

    print(f"\nDone: {generated} fixtures written, {skipped} snapshots skipped.")
    if generated < 15:
        print(
            f"WARNING: only {generated} fixtures generated (expected ≥15). "
            "Check that all 6 local snapshots are accessible.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
