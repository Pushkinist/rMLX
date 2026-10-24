#!/usr/bin/env python3
"""jina-embeddings-v4 numeric parity: rMLX vs Python reference.

Text: rMLX `/v1/embeddings` text outputs vs the snapshot's
bundled `transformers.AutoModel.encode_text`, >=3 texts x 3 tasks,
single-vector + multi-vector + one matryoshka dim.

Image: same, but vs `model.encode_image` for >=2
deterministic synthetic images x 3 tasks (single + multi + matryoshka). The
image gate is the keystone — `--collect-rmlx`/`--compare` cover both modes.

Single-MLX-process discipline: rMLX (Metal) and this script (CPU torch) MUST
NOT run on the GPU simultaneously. Flow:

  1. rMLX serve (Metal) up.
  2. `--collect-rmlx`: curl every (input,task,mode) -> writes rmlx vectors.
  3. kill rMLX, free the Metal claim.
  4. `--compare`: load Python ref on CPU, recompute, print cosine table.

DoD gate (each of text + image): single-vector cosine >= 0.999 every
(input,task); multi-vector per-token cosine >= 0.999 + shape match;
matryoshka(512) >= 0.999.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

_ROOT = Path(
    os.environ.get("RMLX_ROOT")
    or Path(__file__).resolve().parents[2]
)
O_MODELS = Path(
    os.environ.get("RMLX_O_MODELS_ROOT")
    or _ROOT.parents[1] / "open-models"
)
MODEL_PATH = str(O_MODELS / "jinaai__jina-embeddings-v4")
MODEL_ID = "jinaai__jina-embeddings-v4"

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Apple Silicon unifies CPU and GPU memory for ML inference.",
    "def softmax(x): return np.exp(x) / np.exp(x).sum()",
]
TASKS = ["retrieval", "text-matching", "code"]
MATRYOSHKA_DIM = 512


def synth_images() -> list:
    """Two deterministic synthetic PIL images (no external asset):

    - `gradient`: a smooth-ish 64x48 RGB gradient (a real-photo-like
      continuous field — exercises the smooth-content path).
    - `graphic`: a 96x72 hard-edged checkerboard + bars (a simple graphic —
      exercises high-frequency edges, where resample/quant deltas peak).
    """
    from PIL import Image

    g = np.zeros((48, 64, 3), dtype=np.uint8)
    for y in range(48):
        for x in range(64):
            g[y, x] = [(x * 251) % 256, (y * 193) % 256, ((x ^ y)) % 256]
    gradient = Image.fromarray(g, "RGB")

    c = np.zeros((72, 96, 3), dtype=np.uint8)
    for y in range(72):
        for x in range(96):
            on = ((x // 12) + (y // 12)) % 2 == 0
            c[y, x] = [240, 30, 30] if on else [20, 20, 200]
            if 30 <= y < 42:
                c[y, x] = [10, 220, 10]
    graphic = Image.fromarray(c, "RGB")
    return [("gradient", gradient), ("graphic", graphic)]


def image_b64_data_uri(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(
        buf.getvalue()
    ).decode()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── rMLX collection (curl the live server) ────────────────────────────────────


def _post(port: int, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/embeddings",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def _collect_modes(port: int, base_payload: dict) -> dict:
    """single / multi / matryoshka vectors for one input payload."""
    r = _post(port, {**base_payload})
    single = r["data"][0]["embedding"]
    r = _post(port, {**base_payload, "return_multivector": True})
    multi = r["data"][0]["embedding"]
    r = _post(port, {**base_payload, "dimensions": MATRYOSHKA_DIM})
    matr = r["data"][0]["embedding"]
    return {"single": single, "multi": multi, "matryoshka": matr}


def collect_rmlx(port: int, out_path: str) -> None:
    out: dict = {
        "text": {"single": {}, "multi": {}, "matryoshka": {}},
        "image": {"single": {}, "multi": {}, "matryoshka": {}},
    }
    # ---- text ----
    for task in TASKS:
        for m in ("single", "multi", "matryoshka"):
            out["text"][m][task] = []
        for text in TEXTS:
            v = _collect_modes(
                port, {"model": MODEL_ID, "input": text, "task": task}
            )
            for m in ("single", "multi", "matryoshka"):
                out["text"][m][task].append(v[m])
    # ---- image ----
    imgs = synth_images()
    out["image_labels"] = [name for name, _ in imgs]
    uris = [image_b64_data_uri(img) for _, img in imgs]
    for task in TASKS:
        for m in ("single", "multi", "matryoshka"):
            out["image"][m][task] = []
        for uri in uris:
            v = _collect_modes(
                port,
                {"model": MODEL_ID, "input": {"image": uri}, "task": task},
            )
            for m in ("single", "multi", "matryoshka"):
                out["image"][m][task].append(v[m])
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote rMLX vectors -> {out_path}")


# ── Python reference (CPU torch) ──────────────────────────────────────────────


def python_reference() -> dict:
    import contextlib

    import torch
    from transformers import AutoModel

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    # jina's `_process_batches` wraps the forward in
    # `torch.autocast(dtype=torch.bfloat16)`, so the projector + final
    # L2-normalize run in bf16 and the reference multi-vectors come back NOT
    # unit-norm (|v| ~ 0.997..1.002), purely a bf16-rounding artifact. rMLX
    # computes the final normalize cleanly. To get a faithful float32
    # reference (the actual ground truth, independent of the demo wrapper's
    # autocast), neutralise autocast for the reference compute.
    @contextlib.contextmanager
    def _no_autocast(*_a, **_k):
        yield

    torch.autocast = _no_autocast  # type: ignore[assignment]

    def to_np(t) -> np.ndarray:
        # jina's _process_batches autocasts to bf16; numpy() rejects bf16 ->
        # cast up to float32 first. Works for torch tensors and array-likes.
        if hasattr(t, "detach"):
            return t.detach().cpu().float().numpy().astype(np.float32)
        return np.asarray(t, dtype=np.float32)

    ref: dict = {
        "text": {"single": {}, "multi": {}, "matryoshka": {}},
        "image": {"single": {}, "multi": {}, "matryoshka": {}},
    }
    for task in TASKS:
        for m in ("single", "multi", "matryoshka"):
            ref["text"][m][task] = []
        for text in TEXTS:
            sv = model.encode_text(
                texts=[text], task=task, prompt_name="query"
            )[0]
            ref["text"]["single"][task].append(to_np(sv))
            mv = model.encode_text(
                texts=[text],
                task=task,
                prompt_name="query",
                return_multivector=True,
            )[0]
            ref["text"]["multi"][task].append(to_np(mv))
            mt = model.encode_text(
                texts=[text],
                task=task,
                prompt_name="query",
                truncate_dim=MATRYOSHKA_DIM,
            )[0]
            ref["text"]["matryoshka"][task].append(to_np(mt))

    imgs = [img for _, img in synth_images()]
    for task in TASKS:
        for m in ("single", "multi", "matryoshka"):
            ref["image"][m][task] = []
        for img in imgs:
            sv = model.encode_image(images=[img], task=task)[0]
            ref["image"]["single"][task].append(to_np(sv))
            mv = model.encode_image(
                images=[img], task=task, return_multivector=True
            )[0]
            ref["image"]["multi"][task].append(to_np(mv))
            mt = model.encode_image(
                images=[img], task=task, truncate_dim=MATRYOSHKA_DIM
            )[0]
            ref["image"]["matryoshka"][task].append(to_np(mt))
    return ref


# ── Compare ───────────────────────────────────────────────────────────────────


def _table(rmlx_mode: dict, ref_mode: dict, labels: list, gate: float):
    """Build (rows, failures) for one modality (text or image)."""
    rows = []
    failures = 0
    for task in TASKS:
        for i, lab in enumerate(labels):
            tlabel = lab[:34] + ("…" if len(lab) > 34 else "")

            r_sv = np.asarray(rmlx_mode["single"][task][i], dtype=np.float32)
            p_sv = ref_mode["single"][task][i]
            c_sv = cosine(r_sv, p_sv)
            ok_sv = c_sv >= gate and r_sv.shape == p_sv.shape
            failures += not ok_sv

            r_mt = np.asarray(
                rmlx_mode["matryoshka"][task][i], dtype=np.float32
            )
            p_mt = ref_mode["matryoshka"][task][i]
            c_mt = cosine(r_mt, p_mt)
            ok_mt = (
                c_mt >= gate
                and r_mt.shape == p_mt.shape
                and r_mt.shape[-1] == MATRYOSHKA_DIM
            )
            failures += not ok_mt

            r_mv = np.asarray(rmlx_mode["multi"][task][i], dtype=np.float32)
            p_mv = ref_mode["multi"][task][i]
            shape_ok = r_mv.shape == p_mv.shape
            if shape_ok and r_mv.size:
                per_tok = [
                    cosine(r_mv[t], p_mv[t]) for t in range(r_mv.shape[0])
                ]
                c_mv = float(np.min(per_tok))
            else:
                c_mv = 0.0
            ok_mv = shape_ok and c_mv >= gate
            failures += not ok_mv

            rows.append(
                (
                    task,
                    tlabel,
                    c_sv,
                    c_mt,
                    c_mv,
                    f"{r_mv.shape}=={p_mv.shape}"
                    if shape_ok
                    else f"{r_mv.shape}!={p_mv.shape}",
                )
            )
    return rows, failures


def _print_table(title: str, rows: list) -> None:
    w = max(len(r[1]) for r in rows)
    print()
    print(f"=== {title} ===")
    print(
        f"{'task':<14} {'input':<{w}} {'single':>9} {'matr512':>9} "
        f"{'multi(min)':>11}  mv_shape"
    )
    print("-" * (14 + w + 9 + 9 + 11 + 12))
    for task, tlabel, c_sv, c_mt, c_mv, mvs in rows:
        print(
            f"{task:<14} {tlabel:<{w}} {c_sv:>9.6f} {c_mt:>9.6f} "
            f"{c_mv:>11.6f}  {mvs}"
        )


def compare(rmlx_path: str) -> int:
    with open(rmlx_path) as f:
        rmlx = json.load(f)
    ref = python_reference()

    gate = 0.999
    img_labels = rmlx.get("image_labels", ["img0", "img1"])

    t_rows, t_fail = _table(rmlx["text"], ref["text"], TEXTS, gate)
    i_rows, i_fail = _table(rmlx["image"], ref["image"], img_labels, gate)

    _print_table("TEXT parity (must NOT regress)", t_rows)
    _print_table("IMAGE parity (gate)", i_rows)
    print()

    rc = 0
    if t_fail:
        print(f"TEXT PARITY FAILED: {t_fail} cell(s) below {gate}")
        rc = 1
    else:
        print(f"TEXT PARITY PASSED: all cells >= {gate}")
    if i_fail:
        print(f"IMAGE PARITY FAILED: {i_fail} cell(s) below {gate}")
        rc = 1
    else:
        print(f"IMAGE PARITY PASSED: all cells >= {gate}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-rmlx", metavar="OUT_JSON")
    ap.add_argument("--port", type=int, default=62265)
    ap.add_argument("--compare", metavar="RMLX_JSON")
    args = ap.parse_args()

    if args.collect_rmlx:
        collect_rmlx(args.port, args.collect_rmlx)
        return 0
    if args.compare:
        return compare(args.compare)
    ap.error("one of --collect-rmlx or --compare is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
