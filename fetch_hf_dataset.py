#!/usr/bin/env python3
"""Pull a Hugging Face dataset and write it as a trainer-ready JSONL.

Auto-detects which fields to map onto the trainer's expected shapes:
  - {"messages": [...]}                      -> chat
  - {"instruction","output"} (+optional input) -> Alpaca
  - {"prompt","completion"}                   -> prompt/completion
  - else: pass --text-field NAME to use a single text column

Examples:
  python fetch_hf_dataset.py Canstralian/pentesting_dataset \\
      --out ./data/pentest.jsonl --split train

  python fetch_hf_dataset.py tatsu-lab/alpaca --out ./data/alpaca.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


KNOWN_TEXT_FIELDS = ["text", "content", "body", "completion", "response", "answer"]


def to_record(row: dict, text_field: str | None) -> dict | None:
    if "messages" in row and isinstance(row["messages"], list):
        return {"messages": row["messages"]}
    if "instruction" in row and ("output" in row or "response" in row):
        return {
            "instruction": row["instruction"],
            "input": row.get("input", ""),
            "output": row.get("output") or row.get("response", ""),
        }
    if "prompt" in row and ("completion" in row or "response" in row):
        return {
            "prompt": row["prompt"],
            "completion": row.get("completion") or row.get("response", ""),
        }
    if text_field and text_field in row and row[text_field]:
        return {"text": str(row[text_field])}
    for cand in KNOWN_TEXT_FIELDS:
        if cand in row and row[cand]:
            return {"text": str(row[cand])}
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--limit", type=int, default=None, help="Only export first N rows")
    args = p.parse_args()

    print(f"[fetch_hf_dataset] loading {args.repo_id} split={args.split}", flush=True)
    ds = load_dataset(args.repo_id, split=args.split, token=args.token)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.limit and written >= args.limit:
                break
            rec = to_record(row, args.text_field)
            if rec is None:
                skipped += 1
                if i < 3:
                    print(f"[fetch_hf_dataset] no mapping for row keys={list(row.keys())}", flush=True)
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"[fetch_hf_dataset] wrote {written} records (skipped {skipped}) -> {out_path}", flush=True)
    if written == 0:
        print(
            "[fetch_hf_dataset] WARNING: 0 records written. "
            "Inspect the dataset on huggingface.co and re-run with --text-field <name>.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
