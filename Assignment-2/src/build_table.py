"""Render the per-algorithm summary table into a LaTeX include file."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"


def main() -> None:
    df = pd.read_csv(RESULTS / "summary.csv")
    # Build a pivot:  rows = algorithm, columns multi-indexed by (size, metric).
    sizes = sorted(df["n_vehicles"].unique())
    cols_time = df.pivot(index="algorithm", columns="n_vehicles", values="time_s")
    cols_nodes = df.pivot(index="algorithm", columns="n_vehicles", values="nodes")
    cols_solved = df.pivot(index="algorithm", columns="n_vehicles", values="solved_rate")

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\caption{Mean wall-clock time (s) per solver per problem size — 3 seeds, 10\\,s budget.}",
        "\\label{tab:time}",
        "\\begin{tabular}{l" + "r" * len(sizes) + "}",
        "\\toprule",
        "Algorithm & " + " & ".join("$N{=}" + str(s) + "$" for s in sizes) + " \\\\",
        "\\midrule",
    ]
    for alg in cols_time.index:
        cells = [f"{cols_time.loc[alg, s]:.3f}" for s in sizes]
        lines.append(f"{alg.replace('+','{+}')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    lines += [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\caption{Mean nodes expanded per solver (a search-effort proxy).}",
        "\\label{tab:nodes}",
        "\\begin{tabular}{l" + "r" * len(sizes) + "}",
        "\\toprule",
        "Algorithm & " + " & ".join("$N{=}" + str(s) + "$" for s in sizes) + " \\\\",
        "\\midrule",
    ]
    for alg in cols_nodes.index:
        cells = [f"{cols_nodes.loc[alg, s]:,.0f}" for s in sizes]
        lines.append(f"{alg.replace('+','{+}')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    lines += [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\caption{Success rate (fraction of seeds in which a zero-conflict assignment was returned).}",
        "\\label{tab:solved}",
        "\\begin{tabular}{l" + "r" * len(sizes) + "}",
        "\\toprule",
        "Algorithm & " + " & ".join("$N{=}" + str(s) + "$" for s in sizes) + " \\\\",
        "\\midrule",
    ]
    for alg in cols_solved.index:
        cells = [f"{cols_solved.loc[alg, s]:.2f}" for s in sizes]
        lines.append(f"{alg.replace('+','{+}')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    (REPORT / "results_table.tex").write_text("\n".join(lines) + "\n")
    print(f"[save] {REPORT / 'results_table.tex'}")


if __name__ == "__main__":
    main()
