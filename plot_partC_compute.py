#!/usr/bin/env python3
"""
plot_partC_compute.py

Usage example:
  python3 plot_partC_compute.py --files partC_compute.json \
      --metric median_s --title "Runtime vs K (compute-bound sweep)" \
      --out partC_runtime_vs_K.png

Input: line-delimited JSON from your program.
It groups by workK and plots the chosen metric (median_s or best_s).

Tip: For a clean figure, keep threads, rule, schedule, and n fixed 
while varying workK.
"""
import argparse, json
import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(paths):
    rows = []
    for p in paths:
        with open(p, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # skip malformed/unparseable lines
                    pass
    if not rows:
        raise SystemExit("No data loaded. Ensure you passed the right files.")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--metric', default='median_s',
                    choices=['median_s','best_s'])
    ap.add_argument('--title', default='Runtime vs K')
    ap.add_argument('--out', default='partC_runtime_vs_K.png')
    args = ap.parse_args()

    df = load_jsonl(args.files)

    # basic sanity columns
    for c in ['workK','n','threads','rule']:
        if c not in df.columns:
            df[c] = None

    # aggregate by K (median across repeats)
    g = df.groupby('workK', as_index=False)[args.metric].median() \
          .sort_values('workK')

    plt.figure(figsize=(7,4.5))
    plt.plot(g['workK'], g[args.metric], marker='o')
    plt.xlabel('K (work amplifier)')
    plt.ylabel(f'runtime (s) [{args.metric}]')
    plt.title(args.title)
    plt.grid(True, ls='--', alpha=0.3)

    # if K spans large ranges, use log scale
    if g['workK'].min() > 0 and g['workK'].max() / max(1, g['workK'].min()) >= 100:
        plt.xscale('log')

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"[ok] saved {args.out}")


if __name__ == '__main__':
    main()