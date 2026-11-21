"""N4 step 3 -- leave-one-out on the CKA-vs-headroom Spearman correlation (CPU, stdlib).

    python3 codification/n4/loo_cka.py codification/results/iter1/numbers/analyze_test500.txt

Parses analyze.py's committed text output (one block per AR selector: CKA and headroom
per model) and recomputes Spearman's rho over the models that have an AR base, first on
all of them, then with each one dropped. Ties get average ranks, as in analyze.ranks.
With 4 models a leave-one-out rho is over 3 points: descriptive only.
"""
from __future__ import annotations

import re
import sys


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def parse(path):
    blocks, cur = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"== selector (\S+)", line)
            if m:
                cur = blocks.setdefault(m.group(1), {}); continue
            m = re.match(r"\s+(\S+)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([+-][0-9.]+)%", line)
            if cur is not None and m:                       # rows with "--" CKA (LLaDA, the control) do not match
                cur[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return blocks


def main():
    blocks = parse(sys.argv[1])
    assert blocks, "no selector blocks parsed"
    models = sorted(next(iter(blocks.values())))
    print(f"models with an AR base: {', '.join(models)}")
    print(f"{'selector':<22}{'all':>7}" + "".join(f"{'-' + m:>13}" for m in models))
    for sel, rows in blocks.items():
        assert sorted(rows) == models, f"{sel}: models {sorted(rows)} != {models}"
        rho = lambda ms: spearman([rows[m][0] for m in ms], [rows[m][1] for m in ms])
        print(f"{sel:<22}{rho(models):>+7.2f}" + "".join(f"{rho([x for x in models if x != m]):>+13.2f}" for m in models))


if __name__ == "__main__":
    main()
