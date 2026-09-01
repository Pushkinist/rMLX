#!/usr/bin/env python3
"""Summarize turbo_probe.py jsonl output.

Prints, per (arm, mode): n, median / min / max decode TPS, the spread as a
percentage of the median, KV bytes true vs claimed, and the ratio of each mode
to a chosen reference mode. Ratios are computed from medians of cells measured
inside the same process, which is the only comparison the host's cross-run
drift permits.
"""
import json
import statistics
import sys
from collections import defaultdict


def load(paths):
    cells, errs = [], []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("record") == "cell":
                    (errs if "error" in d else cells).append(d)
                elif d.get("record") == "error":
                    errs.append(d)
    return cells, errs


def fmt_gb(b):
    return f"{b / 1024**3:7.3f}"


def main():
    ref = "fp16"
    paths = []
    for a in sys.argv[1:]:
        if a.startswith("--ref="):
            ref = a.split("=", 1)[1]
        else:
            paths.append(a)
    cells, errs = load(paths)
    by = defaultdict(list)
    for c in cells:
        by[(c["arm"], c["mode"])].append(c)

    print(f"{'arm':<8} {'mode':<10} {'n':>2} {'tps_med':>9} {'tps_min':>8} "
          f"{'tps_max':>8} {'spread%':>8} {'kvGB_true':>10} {'kvGB_claim':>11} "
          f"{'ttft_med':>9}")
    print("-" * 96)
    med = {}
    for (arm, mode), rows in sorted(by.items()):
        t = sorted(r["decode_tps"] for r in rows)
        m = statistics.median(t)
        med[(arm, mode)] = m
        kt = statistics.median(r["kv_bytes_true"] for r in rows)
        kc = [r["kv_bytes_claimed"] for r in rows if r["kv_bytes_claimed"] is not None]
        kcs = fmt_gb(statistics.median(kc)) if kc else "      n/a"
        tt = statistics.median(r["ttft_s"] for r in rows)
        print(f"{arm:<8} {mode:<10} {len(t):>2} {m:9.2f} {t[0]:8.2f} {t[-1]:8.2f} "
              f"{100 * (t[-1] - t[0]) / m:8.1f} {fmt_gb(kt)} {kcs:>11} {tt:9.2f}")

    print()
    print(f"ratios vs {ref} (same process, same arm):")
    for (arm, mode), m in sorted(med.items()):
        r = med.get((arm, ref))
        if not r:
            continue
        rows = by[(arm, mode)]
        refrows = by[(arm, ref)]
        kt = statistics.median(x["kv_bytes_true"] for x in rows)
        ktr = statistics.median(x["kv_bytes_true"] for x in refrows)
        print(f"  {arm:<8} {mode:<10} tps {m / r:6.3f}x   kv_true {kt / ktr:6.3f}x")

    if errs:
        print("\nerrors:")
        seen = set()
        for e in errs:
            k = (e.get("arm"), e.get("mode"), e.get("error", "")[:90])
            if k in seen:
                continue
            seen.add(k)
            print(f"  {k[0]} {k[1]}: {k[2]}")


if __name__ == "__main__":
    main()
