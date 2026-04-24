"""RunPod Serverless handler that wraps llama-server.

Cold-start sequence (runs once per worker boot, before SDK takes jobs):
  1. Resolve MODEL_REPO / MODEL_FILE → a GGUF on disk under
     /runpod-volume/models (network volume) or /cache/models (fallback).
  2. Spawn start-llama-server.sh as a subprocess. The launcher does the
     VRAM probe, KV-cache budgeting, and ctx auto-sizing — same logic
     as the local-install path, no fork.
  3. Poll http://127.0.0.1:8080/health until it returns 200.

Per-job (handler is a generator → /stream returns chunks, /runsync
returns the aggregated stream because return_aggregate_stream=True):
  Forward the OpenAI-shaped chat-completion payload to llama-server in
  streaming mode, yield each parsed SSE chunk back to the SDK.

Job input shape (subset of OpenAI Chat Completions):
  {
    "messages":          [{"role": "...", "content": "..."}],
    "temperature":       0.7,
    "max_tokens":        512,
    "top_p":             0.95,
    "top_k":             40,
    "stop":              ["..."],
    "presence_penalty":  0.0,
    "frequency_penalty": 0.0,
    "seed":              null
  }

Endpoint env vars:
  MODEL_REPO   HF repo id, e.g. "HauhauCS/Qwen3.5-9B-Uncensored-..."  [required]
  MODEL_FILE   GGUF filename or substring hint                         [optional]
  HF_TOKEN     For gated/private repos                                 [optional]
  LLAMA_*      Anything start-llama-server.sh accepts (ctx override,
               flash-attn, batch sizes, tensor-split, ...)             [optional]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import requests
import runpod


# ── paths ────────────────────────────────────────────────────────────────
# /runpod-volume is the network-volume mount inside Serverless workers.
# Without an attached volume, fall back to an in-container path so the
# image still boots for local docker-run testing (model re-downloads
# every cold start in that mode — fine for smoke tests).
_VOLUME = Path("/runpod-volume")
VOLUME_ROOT = _VOLUME if _VOLUME.is_dir() else Path("/cache")
MODELS_DIR = VOLUME_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_HOST = os.environ.get("LLAMA_BIND", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
LLAMA_URL  = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

# Generous load timeout — a 70B Q5 from cold disk through PCIe can take
# 3-4 min to map into VRAM. 10 min covers that with headroom.
LOAD_TIMEOUT_S = int(os.environ.get("LLAMA_LOAD_TIMEOUT", "600"))

_server_proc: subprocess.Popen | None = None


# ── model resolution ─────────────────────────────────────────────────────
def _find_local_gguf(repo_dir: Path, hint: str | None) -> Path | None:
    if not repo_dir.is_dir():
        return None
    candidates = sorted(repo_dir.glob("**/*.gguf"))
    if not candidates:
        return None
    if not hint:
        return candidates[0]
    needle = hint.lower().replace(".gguf", "")
    for c in candidates:
        if needle in c.name.lower():
            return c
    return None


def _resolve_model() -> Path:
    repo = os.environ.get("MODEL_REPO")
    if not repo:
        raise RuntimeError("MODEL_REPO env var is required")
    file_hint = os.environ.get("MODEL_FILE")

    repo_dir = MODELS_DIR / repo.replace("/", "__")
    local = _find_local_gguf(repo_dir, file_hint)
    if local:
        print(f"[handler] using cached model: {local}", flush=True)
        return local

    print(f"[handler] downloading {repo}" + (f" / {file_hint}" if file_hint else ""), flush=True)
    cmd = [sys.executable, "/app/download_model.py", repo, "--dest", str(MODELS_DIR)]
    if file_hint:
        cmd += ["--file", file_hint]
    if os.environ.get("HF_TOKEN"):
        cmd += ["--token", os.environ["HF_TOKEN"]]
    subprocess.check_call(cmd)

    local = _find_local_gguf(repo_dir, file_hint)
    if not local:
        raise RuntimeError(f"download finished but no GGUF found under {repo_dir}")
    return local


# ── llama-server lifecycle ───────────────────────────────────────────────
def _start_llama_server(model_path: Path) -> None:
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        return
    env = os.environ.copy()
    env["MODEL"]      = str(model_path)
    env["LLAMA_PORT"] = str(LLAMA_PORT)
    env["LLAMA_BIND"] = LLAMA_HOST
    print(f"[handler] launching start-llama-server.sh with MODEL={model_path}", flush=True)
    _server_proc = subprocess.Popen(
        ["/bin/bash", "/app/start-llama-server.sh"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    deadline = time.time() + LOAD_TIMEOUT_S
    while time.time() < deadline:
        if _server_proc.poll() is not None:
            raise RuntimeError(f"llama-server exited during load (code={_server_proc.returncode})")
        try:
            r = requests.get(f"{LLAMA_URL}/health", timeout=2)
            if r.status_code == 200:
                print("[handler] llama-server ready", flush=True)
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"llama-server did not bind within {LOAD_TIMEOUT_S}s")


# ── request handling ─────────────────────────────────────────────────────
_PASSTHROUGH_KEYS = (
    "temperature", "max_tokens", "top_p", "top_k", "min_p",
    "stop", "presence_penalty", "frequency_penalty", "seed",
    "repeat_penalty", "tfs_z", "typical_p", "mirostat",
    "response_format", "tools", "tool_choice", "logit_bias",
)


def _build_payload(inp: dict[str, Any]) -> dict[str, Any]:
    if "messages" not in inp:
        raise ValueError("input.messages is required")
    payload: dict[str, Any] = {
        "messages": inp["messages"],
        "stream":   True,  # always stream from llama-server; SDK aggregates if needed
    }
    for k in _PASSTHROUGH_KEYS:
        if k in inp:
            payload[k] = inp[k]
    return payload


def handler(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    inp = job.get("input") or {}
    try:
        payload = _build_payload(inp)
    except ValueError as e:
        yield {"error": str(e)}
        return

    try:
        with requests.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, 600),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                chunk = line[6:].decode("utf-8", errors="replace").strip()
                if chunk == "[DONE]":
                    return
                try:
                    yield json.loads(chunk)
                except json.JSONDecodeError:
                    # Pass through malformed chunks rather than 500ing — they
                    # are usually keep-alive artifacts from upstream proxies.
                    continue
    except requests.exceptions.RequestException as e:
        yield {"error": f"llama-server request failed: {e}"}


# ── boot ─────────────────────────────────────────────────────────────────
print("[handler] cold start: resolving model + booting llama-server", flush=True)
_start_llama_server(_resolve_model())
print("[handler] entering RunPod SDK loop", flush=True)

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
