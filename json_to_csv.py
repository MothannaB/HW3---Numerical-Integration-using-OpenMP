#!/usr/bin/env python3
"""
json_to_csv.py

Convert line-delimited JSON logs (one JSON object per line)
emitted by myprogram into a CSV, and PREPEND a sysinfo
header (commented with '#') from a text file.

Usage:
  python3 json_to_csv.py \
      --json partD/strong_k0.json \
      --sysinfo logs/sysinfo.txt \
      --out partD/strong_k0.csv

The script:
  - Reads each JSON line (ignoring blanks and malformed lines)
  - Flattens expected fields into CSV columns
  - If --sysinfo is provided, writes those lines at the top
    (as comments starting with '#')
"""
import argparse, json, sys
from pathlib import Path
import csv

# Default columns we expect from myprogram's JSON
COLUMNS = [
    "rule","func","a","b","n","threads","workK",
    "accum","schedule","best_s","median_s","result"
]

def parse_line(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="input JSONL file (one JSON per line)")
    ap.add_argument("--sysinfo", help="sysinfo file to prepend as commented lines")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    in_path = Path(args.json)
    if not in_path.exists():
        print(f"[error] input JSON not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = parse_line(line)
            if not obj:
                continue
            # normalize fields into the canonical column order, fill missing
            row = {}
            for col in COLUMNS:
                if col in obj:
                    row[col] = obj[col]
                else:
                    # try to recover schedule if present in a different shape
                    row[col] = obj.get(col, "")
            # some runs may not include 'schedule' or 'accum' in JSON; keep blank
            rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as outf:
        # Write sysinfo at the top (commented)
        if args.sysinfo:
            s = Path(args.sysinfo)
            if s.exists():
                with s.open("r", encoding="utf-8") as sf:
                    for ln in sf:
                        if not ln.startswith("#"):
                            outf.write("# " + ln.rstrip() + "\n")
                        else:
                            outf.write(ln.rstrip() + "\n")
            # blank line after sysinfo block
            outf.write("#\n")

        # Write CSV header + rows
        writer = csv.DictWriter(outf, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[ok] wrote {out_path} ({len(rows)} rows)")

if __name__ == "__main__":
    main()