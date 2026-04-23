#!/usr/bin/env python3
"""Download a Hugging Face model snapshot into ./models/<repo>.

Supports two modes:
  - Full repo snapshot (HF safetensors format, for fine-tuning)
  - Single GGUF file (for inference with llama.cpp / ollama)

Usage examples:
  python download_model.py meta-llama/Llama-3.2-1B
  python download_model.py BlossomsAI/Qwen2.5-Coder-7B-Instruct-Uncensored-GGUF --file q4_k_m.gguf
  # Also accepts URLs pasted from the HF website:
  python download_model.py 'BlossomsAI/Qwen2.5-Coder-7B-Instruct-Uncensored-GGUF?show_file_info=q4_k_m.gguf'
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs


def _ensure_requirements() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[download_model] huggingface_hub not found — installing...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])


_ensure_requirements()

from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files  # noqa: E402


def _parse_repo_arg(raw: str) -> tuple[str, str | None]:
    """Return (clean_repo_id, filename_or_None).

    Handles HF website URLs like:
      BlossomsAI/Foo?show_file_info=q4_k_m.gguf
    """
    if "?" in raw:
        repo_part, query_part = raw.split("?", 1)
        params = parse_qs(query_part)
        filename = params.get("show_file_info", [None])[0]
        return repo_part.strip(), filename
    return raw.strip(), None


def _resolve_gguf_filename(repo_id: str, hint: str, token: str | None) -> str:
    """Find the actual filename in the repo that matches the user's hint.

    The hint is case-insensitive and matched as a substring, so 'q4_k_m'
    matches 'Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf'.
    """
    files = list(list_repo_files(repo_id, token=token))
    gguf_files = [f for f in files if f.lower().endswith(".gguf")]
    hint_lower = hint.lower().replace(".gguf", "")
    matches = [f for f in gguf_files if hint_lower in f.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[download_model] Multiple matches for '{hint}':", file=sys.stderr)
        for m in matches:
            print(f"  {m}", file=sys.stderr)
        print(f"[download_model] Using first match: {matches[0]}", file=sys.stderr)
        return matches[0]
    # No substring match — try exact
    if hint in gguf_files:
        return hint
    print(f"[download_model] Available GGUF files in {repo_id}:", file=sys.stderr)
    for f in gguf_files:
        print(f"  {f}", file=sys.stderr)
    raise SystemExit(f"No GGUF file matching '{hint}' found in {repo_id}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "repo_id",
        help="HF repo id, e.g. meta-llama/Llama-3.2-1B or "
             "BlossomsAI/Qwen2.5-Coder-7B-Instruct-Uncensored-GGUF?show_file_info=q4_k_m.gguf",
    )
    p.add_argument("--dest", default="./models", help="Root directory for downloaded models")
    p.add_argument("--revision", default=None)
    p.add_argument("--token", default=None, help="HF token for gated models")
    p.add_argument("--file", default=None, help="Download a single GGUF file (name or partial match)")
    args = p.parse_args()

    repo_id, file_from_url = _parse_repo_arg(args.repo_id)
    target_file = args.file or file_from_url

    out_dir = Path(args.dest) / repo_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_file:
        # Single-file GGUF download
        resolved = _resolve_gguf_filename(repo_id, target_file, args.token)
        print(f"[download_model] {repo_id}/{resolved} -> {out_dir}", flush=True)
        dest_path = hf_hub_download(
            repo_id=repo_id,
            filename=resolved,
            revision=args.revision,
            local_dir=str(out_dir),
            token=args.token,
        )
        print(f"[download_model] done: {dest_path}", flush=True)
    else:
        # Full snapshot (safetensors weights for fine-tuning)
        print(f"[download_model] {repo_id} -> {out_dir}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            local_dir=str(out_dir),
            token=args.token,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.py", "*.gguf"],
        )
        print(f"[download_model] done: {out_dir}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
