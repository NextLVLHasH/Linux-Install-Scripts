#!/usr/bin/env python3
"""Convert a CSV (e.g. NSL-KDD, CIC-IDS2017, phishing URL feeds) into the
trainer's instruction-tuning JSONL format.

Pattern: each row becomes one Alpaca-style record:
  instruction: <user-supplied template>
  input:       <selected feature columns, joined as "k=v">
  output:      <value of --label-col>

Example (NSL-KDD style):
  python csv_to_jsonl.py KDDTrain+.csv \\
      --out ./data/nsl_kdd.jsonl \\
      --label-col attack_type \\
      --feature-cols protocol_type,service,flag,src_bytes,dst_bytes \\
      --instruction "Classify this network connection as benign or the attack family it belongs to."
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--out", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument(
        "--feature-cols",
        default="",
        help="Comma-separated columns to include as input. Empty = all non-label cols.",
    )
    p.add_argument(
        "--instruction",
        default="Analyze this record and produce the appropriate label.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--delimiter", default=",")
    args = p.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    written = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, delimiter=args.delimiter)
        if args.label_col not in (reader.fieldnames or []):
            print(f"ERROR: label-col {args.label_col!r} not in {reader.fieldnames}", file=sys.stderr)
            return 2
        cols = feature_cols or [c for c in (reader.fieldnames or []) if c != args.label_col]
        for row in reader:
            if args.limit and written >= args.limit:
                break
            features = " | ".join(f"{c}={row.get(c, '')}" for c in cols)
            rec = {
                "instruction": args.instruction,
                "input": features,
                "output": str(row.get(args.label_col, "")).strip(),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"[csv_to_jsonl] wrote {written} records -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
