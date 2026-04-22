#!/usr/bin/env python3
"""Download a Hugging Face model snapshot into ./models/<repo>.

Used for fine-tuning (LM Studio's GGUF files are inference-only and can't be
trained directly, so we pull the HF-format source weights).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id", help="e.g. meta-llama/Llama-3.2-1B or Qwen/Qwen2.5-0.5B")
    p.add_argument("--dest", default="./models", help="Root directory for downloaded models")
    p.add_argument("--revision", default=None)
    p.add_argument("--token", default=None, help="HF token for gated models")
    args = p.parse_args()

    out_dir = Path(args.dest) / args.repo_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_model] {args.repo_id} -> {out_dir}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(out_dir),
        token=args.token,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.py"],
    )
    print(f"[download_model] done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
