#!/usr/bin/env python3
"""
plot_partC_accum.py

Compares accumulation methods (reduction vs padded) for Part C.

Usage (single JSON file containing both accum modes):
  python3 plot_partC_accum.py \
    --files partC/false_sharing.json \
    --metric median_s \
    --title "Reduction vs Padded (memory-bound, K=0)" \
    --out partC/accum_compare.png

If your JSON does NOT contain an "accum" field, you can pass
two files and label them manually:

  python3 plot_partC_accum.py \
    --files reduction.json padded.json \
    --labels reduction padded \
    --metric median_s \
    --out partC/accum_compare.png
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(paths, labels=None):
    rows = []
    for i, p in enumerate(paths):
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # If "accum" is missing, inject label manually
                if "accum" not in obj and labels is not None:
                    obj["accum"] = labels[i]

                rows.append(obj)

    if not rows:
        raise SystemExit("No data loaded. Check input files.")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="JSON log files")
    ap.add_argument("--labels", nargs="*",
                    help="Optional labels if JSON lacks 'accum' field")
    ap.add_argument("--metric", default="median_s",
                    choices=["median_s", "best_s"],
                    help="Timing metric to plot")
    ap.add_argument("--title", default="Reduction vs Padded")
    ap.add_argument("--out", default="partC_accum_compare.png")
    args = ap.parse_args()

    df = load_jsonl(args.files, args.labels)

    if "accum" not in df.columns:
        raise SystemExit("No 'accum' field found and no labels provided.")

    # Optional: filter to one configuration if logs contain multiple
    # Example (uncomment if needed):
    # df = df[(df["workK"] == 0) & (df["threads"] == 4)]

    # Median across repeats
    grouped = df.groupby("accum", as_index=False)[args.metric].median()

    plt.figure(figsize=(6, 4))
    plt.bar(grouped["accum"], grouped[args.metric],
            color=["#4e79a7", "#f28e2b"])

    plt.ylabel(f"Runtime (s) [{args.metric}]")
    plt.title(args.title)

    # Label bars
    for i, v in enumerate(grouped[args.metric]):
        plt.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"[ok] saved {args.out}")


if __name__ == "__main__":
    main()