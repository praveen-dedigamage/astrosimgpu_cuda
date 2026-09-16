#!/usr/bin/env python3
"""Summarise a --dump-edges CSV (type,source,target) into degree statistics.

Works on the output of EITHER astrosimgpu's --dump-edges or
astrosim_fused's --dump-edges -- they write the identical format (same
column names, same "A"-prefix convention for astrocyte ids, same source
offset for inh_primary), specifically so the two can be compared directly
without any reformatting.

Usage:
    python3 degree_stats.py edges.csv
"""
import csv
import sys
from collections import defaultdict


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} edges.csv", file=sys.stderr)
        sys.exit(1)

    out_degree = defaultdict(lambda: defaultdict(int))  # type -> source -> count
    in_degree = defaultdict(lambda: defaultdict(int))   # type -> target -> count
    total = defaultdict(int)

    with open(sys.argv[1], newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["type", "source", "target"], f"unexpected header: {header}"
        for row in reader:
            t, src, dst = row
            out_degree[t][src] += 1
            in_degree[t][dst] += 1
            total[t] += 1

    def stats(counts):
        vals = sorted(counts.values())
        n = len(vals)
        if n == 0:
            return "n=0"
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return (f"n={n:>7d}  mean={mean:8.3f}  std={var**0.5:7.3f}  "
                f"min={vals[0]:>6d}  median={vals[n // 2]:>6d}  max={vals[-1]:>6d}")

    for t in sorted(total):
        print(f"== {t}: {total[t]} edges ==")
        print(f"  out-degree (per source): {stats(out_degree[t])}")
        print(f"  in-degree  (per target): {stats(in_degree[t])}")


if __name__ == "__main__":
    main()
