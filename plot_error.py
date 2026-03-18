# plot_error.py
# Usage examples:
#   python3 plot_error.py --files accuracy_trap.json accuracy_simp.json --labels trap simp --true 2.0 --title "Error vs h (sin on [0, pi])" --out error_vs_h.png
#   python3 plot_error.py --csv accuracy_trap.csv accuracy_simp.csv --labels trap simp --true 2.0
#
# The script expects line-delimited JSON produced by your program (one JSON per line),
# or CSV with headers matching the JSON keys: rule,func,a,b,n,threads,workK,best_s,median_s,result
# It will compute h=(b-a)/n and error=|result - TRUE|, do a log-log linear fit, plot, and annotate slopes.

import argparse
import json
import math
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_json_lines(paths: List[str]) -> List[pd.DataFrame]:
    dfs = []
    for p in paths:
        rows = []
        with open(p, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
                rows.append(obj)
        if not rows:
            raise ValueError(f"No valid JSON lines found in {p}")
        df = pd.DataFrame(rows)
        dfs.append(df)
    return dfs


def load_csv(paths: List[str]) -> List[pd.DataFrame]:
    return [pd.read_csv(p) for p in paths]


def compute_error_df(df: pd.DataFrame, true_val: float) -> pd.DataFrame:
    if not {'a','b','n','result'}.issubset(df.columns):
        raise ValueError("Input data must have columns a,b,n,result")
    out = df.copy()
    out['h'] = (out['b'] - out['a']) / out['n']
    out['error'] = (out['result'] - true_val).abs()
    cols = ['rule','func','n','h','error','threads','workK']
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols]


def fit_slope_loglog(h: pd.Series, err: pd.Series):
    x = h.astype(float)
    y = err.astype(float)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float('nan'), float('nan')
    lx = x.apply(lambda v: math.log10(v))
    ly = y.apply(lambda v: math.log10(v))
    import numpy as np
    A = np.vstack([lx.values, np.ones_like(lx.values)]).T
    m, c = np.linalg.lstsq(A, ly.values, rcond=None)[0]
    return m, c


def main():
    parser = argparse.ArgumentParser(description="Plot Error vs h (log-log) for trap/simp and annotate slopes")
    parser.add_argument('--files', nargs='*', default=[], help='Line-delimited JSON files from program output')
    parser.add_argument('--csv', nargs='*', default=[], help='CSV files with columns like the JSON keys')
    parser.add_argument('--labels', nargs='*', default=[], help='Labels for each series (e.g., trap simp)')
    parser.add_argument('--true', type=float, required=True, help='True integral value (e.g., 2.0 for sin on [0,pi])')
    parser.add_argument('--title', type=str, default='Error vs h (log-log)')
    parser.add_argument('--out', type=str, default='error_vs_h.png')
    parser.add_argument('--xlim', nargs=2, type=float, default=None, help='x-axis limits in h (e.g., 1e-7 1e-1)')
    parser.add_argument('--ylim', nargs=2, type=float, default=None, help='y-axis limits in error')
    args = parser.parse_args()

    series_dfs = []

    if args.files:
        series_dfs.extend(load_json_lines(args.files))
    if args.csv:
        series_dfs.extend(load_csv(args.csv))

    if not series_dfs:
        print("No input data provided. Use --files or --csv.")
        sys.exit(2)

    processed = [compute_error_df(df, args.true) for df in series_dfs]

    labels = args.labels if args.labels else [f"series{i+1}" for i in range(len(processed))]
    if len(labels) != len(processed):
        print("Number of labels must match number of input series.")
        sys.exit(2)

    plt.figure(figsize=(7,5))

    annotations = []
    for df, lab in zip(processed, labels):
        d = df.sort_values('h')
        m, c = fit_slope_loglog(d['h'], d['error'])
        d = d[(d['h']>0) & (d['error']>0)]
        plt.loglog(d['h'], d['error'], marker='o', linestyle='-', label=f"{lab} (slope≈{m:.2f})")
        annotations.append((lab, m))

    plt.xlabel('h (step size)')
    plt.ylabel('absolute error |I_true - I_n|')
    plt.title(args.title)
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.legend()

    if args.xlim:
        plt.xlim(args.xlim)
    if args.ylim:
        plt.ylim(args.ylim)

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"[ok] Saved plot to {args.out}")
    for lab, m in annotations:
        print(f"{lab}: fitted slope ≈ {m:.3f}")

if __name__ == '__main__':
    main()
