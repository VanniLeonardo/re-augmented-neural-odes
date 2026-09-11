"""Audit the logs behind Chen et al. Fig. 3c (backward against forward NFE).

The original authors shared their code and logs privately (rtqichen/ode-nets). The code is not
redistributed. The per-batch evaluation counts behind Fig. 3c are, with their permission, in
results/chen_fig3c/fig3c_counts.csv, extracted by `make chen-logs LOGS=.../nfe_logs`.

In check_model.py the forward count is logged, the shared counter is reset, and the backward
pass then increments it. So a logged "forward" count L_k is the true forward count F_k plus the
previous batch's backward count B_{k-1}, plus one evaluation the adjoint makes outside the
reverse solve. The first batch of each tolerance runs the forward twice: L_0 = 2 F_0.
"""
from __future__ import annotations
import argparse
import csv
import re
import statistics as st
from pathlib import Path


def blocks(path: str):
    """Yield (tol, logged forward counts, backward counts) per tolerance, from the raw
    `nfe_logs` or from the extracted CSV."""
    if path.endswith(".csv"):
        rows = list(csv.DictReader(open(path)))
        for tol in dict.fromkeys(r["tol"] for r in rows):
            rs = [r for r in rows if r["tol"] == tol]
            yield tol, [int(r["logged_forward"]) for r in rs], [int(r["backward"]) for r in rs]
        return
    lines = [x.strip() for x in open(path).read().splitlines()]
    start = next(i for i, s in enumerate(lines) if s.startswith("Namespace(")) + 1
    cur = []
    for s in lines[start:]:
        if s.startswith("Tol "):
            L = [int(x) for x in cur[1:] if x.isdigit()]  # cur[0] is the tolerance header
            B = [int(m.group(1)) for x in cur if (m := re.fullmatch(r"Number of NFE in backward:\s*(\d+)", x))]
            assert len(L) == len(B), (s, len(L), len(B))
            yield s.split("|")[0][4:].strip(), L, B
            cur = []
        else:
            cur.append(s)


def inferred_forward(L, B):
    return [L[0] / 2] + [L[k] - B[k - 1] - 1 for k in range(1, len(L))]


def export(data, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["tol", "batch", "logged_forward", "backward"])
        for tol, L, B in data:
            w.writerows([tol, k, l, b] for k, (l, b) in enumerate(zip(L, B)))


def figure(data, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    n = len(data)
    for i, (tol, L, B) in enumerate(data):
        c = plt.get_cmap("rainbow_r")((n - 1 - i) / (n - 1))  # the original's colours: 1e-5 purple
        ax[0].scatter(L, B, s=8, color=c, label=tol)
        ax[1].scatter(inferred_forward(L, B), B, s=8, color=c)
    for a, title in zip(ax, ("as plotted: logged forward count", "corrected: previous backward removed")):
        a.plot([0, 150], [0, 150], "--", color="grey", lw=1)
        a.set(xlim=(0, 150), ylim=(0, 150), xlabel="NFE forward")
        a.set_title(title, fontsize=9)
    ax[0].set_ylabel("NFE backward")
    ax[0].legend(title="tolerance", fontsize=7, title_fontsize=7)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", help="the original nfe_logs, or results/chen_fig3c/fig3c_counts.csv")
    p.add_argument("--export", help="write the per-batch counts to this CSV")
    p.add_argument("--figure", help="write panel (c), as plotted and corrected, to this PNG")
    args = p.parse_args()
    data = list(blocks(args.source))
    print(f"{'tol':6} {'n':>3} | {'r(L,B_prev)':>11} {'r(L,B_same)':>11} | {'F':>5} {'B':>4} | "
          f"{'B/F':>5} {'B/L plotted':>11} | sd L -> sd F")
    for tol, L, B in data:
        F = inferred_forward(L, B)
        print(f"{tol:6} {len(L):3d} | {corr(L[1:], B[:-1]):11.2f} {corr(L[1:], B[1:]):11.2f} | "
              f"{st.median(F):5.0f} {st.median(B):4.0f} | {st.median(b / f for b, f in zip(B, F)):5.2f} "
              f"{st.median(b / l for b, l in zip(B[1:], L[1:])):11.2f} | "
              f"{st.pstdev(L[1:]):5.2f} -> {st.pstdev(F[1:]):5.2f}")
    if args.export:
        export(data, args.export)
        print(f"Wrote {args.export}")
    if args.figure:
        figure(data, args.figure)


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    sa, sb = sum((x - ma) ** 2 for x in a) ** .5, sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float("nan")


if __name__ == "__main__":
    main()
