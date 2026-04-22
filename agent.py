#!/usr/bin/env python3
"""Agent runtime: load a base model + (optional) LoRA adapter, generate a
response, parse <tool_call>{...}</tool_call> blocks, and (optionally) execute
shell commands — feeding the output back to the model and looping.

SAFETY MODEL
============
This thing runs *model-generated commands on your host*. That is dangerous.
Three execution modes (--mode), default is the safe one:

  1. dry-run  (default): never executes, just prints what the model wanted
                         to run. Use this to evaluate a freshly trained model.
  2. approve : prompts y/N before each command (or, in API mode, requires an
               explicit /api/agent/approve POST).
  3. allow   : runs commands automatically if they match an allowlist regex
               from --allowlist (one regex per line). Anything else still
               prompts.

Other guardrails:
  - --timeout caps each shell call (default 30s).
  - --max-iters caps the agent loop (default 6).
  - Commands run with the dashboard's current user — NOT root. Run inside a
    Docker container or a dedicated unprivileged user for real isolation.
  - Hard-blocked patterns (rm -rf /, mkfs, dd of=/dev, :() {:|:&};:) refuse
    even in --mode allow.

CLI:
  python agent.py --model ./models/X --adapter ./runs/r1 \\
      --goal "list listening tcp services" --mode dry-run

Library use (FastAPI imports AgentSession):
  from agent import AgentSession
  s = AgentSession(model_path=..., adapter_path=..., mode="approve")
  for ev in s.run(goal="..."): print(ev)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# Always-blocked patterns (regex). Matched against the full command string.
HARD_BLOCKED = [
    r"\brm\s+-[rRf]+[a-zA-Z]*\s+/(\s|$)",      # rm -rf /
    r"\bmkfs(\.|\s)",                            # filesystem creation
    r"\bdd\s+.*\bof=/dev/",                     # dd to a raw device
    r":\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # fork bomb
    r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b",
    r">\s*/dev/sd[a-z]\b",                      # writing to a raw disk
    r"\bchmod\s+(-R\s+)?0?777\s+/(\s|$)",       # chmod 777 /
]


# ---------- events ----------

@dataclass
class Event:
    kind: str                      # "model" | "tool_call" | "tool_result" | "blocked" | "skipped" | "done" | "error"
    text: str = ""
    cmd: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: Optional[int] = None
    iteration: int = 0


# ---------- safety ----------

def is_hard_blocked(cmd: str) -> Optional[str]:
    for pat in HARD_BLOCKED:
        if re.search(pat, cmd):
            return pat
    return None


def matches_allowlist(cmd: str, allowlist: list[re.Pattern]) -> bool:
    return any(p.search(cmd) for p in allowlist)


def load_allowlist(path: Optional[str]) -> list[re.Pattern]:
    if not path:
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(re.compile(line))
    return out


# ---------- shell tool ----------

def run_shell(cmd: str, timeout: int = 30, cwd: Optional[str] = None,
              max_output_bytes: int = 16 * 1024) -> tuple[int, str, str]:
    """Run cmd in a subprocess. Truncates output. Never uses shell=True with
    interpolation we don't control — but we do pass through `bash -c` to keep
    pipes/redirects working. The hard-block list and allowlist are upstream."""
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
        return proc.returncode, proc.stdout[:max_output_bytes], proc.stderr[:max_output_bytes]
    except subprocess.TimeoutExpired:
        return 124, "", f"[timeout after {timeout}s]"
    except FileNotFoundError as e:
        return 127, "", str(e)


# ---------- session ----------

@dataclass
class AgentSession:
    model_path: str
    adapter_path: Optional[str] = None
    mode: str = "dry-run"            # dry-run | approve | allow
    allowlist_path: Optional[str] = None
    timeout: int = 30
    max_iters: int = 6
    max_new_tokens: int = 512
    temperature: float = 0.2

    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)
    _allowlist: list = field(default_factory=list, init=False, repr=False)
    # Pending approval (used in API mode): cmd waiting for /agent/approve
    _approve_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _pending_cmd: Optional[str] = field(default=None, init=False, repr=False)
    _approval_decision: Optional[bool] = field(default=None, init=False, repr=False)
    _approval_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def __post_init__(self) -> None:
        self._allowlist = load_allowlist(self.allowlist_path)

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    def _generate(self, messages: list[dict]) -> str:
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(0.01, self.temperature),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ---- approval workflow (API mode) ----

    def request_approval(self, cmd: str, timeout: float = 300.0) -> bool:
        """Used by the dashboard /agent runtime. Blocks until /agent/approve POSTs."""
        with self._approve_lock:
            self._pending_cmd = cmd
            self._approval_decision = None
            self._approval_event.clear()
        ok = self._approval_event.wait(timeout=timeout)
        with self._approve_lock:
            decision = bool(self._approval_decision) if ok else False
            self._pending_cmd = None
            self._approval_decision = None
        return decision

    def submit_approval(self, decision: bool) -> bool:
        with self._approve_lock:
            if self._pending_cmd is None:
                return False
            self._approval_decision = decision
        self._approval_event.set()
        return True

    def pending_command(self) -> Optional[str]:
        with self._approve_lock:
            return self._pending_cmd

    # ---- main loop ----

    def run(self, goal: str, system: Optional[str] = None) -> Iterator[Event]:
        self._lazy_load()
        sys_msg = system or (
            "You are a security analyst with a shell tool. "
            "To run a command, emit exactly: "
            "<tool_call>{\"name\":\"shell\",\"args\":{\"cmd\":\"<command>\"}}</tool_call>. "
            "After you receive a tool result, summarize findings. "
            "Never run destructive commands without confirming."
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": goal},
        ]
        for it in range(1, self.max_iters + 1):
            try:
                reply = self._generate(messages)
            except Exception as e:
                yield Event(kind="error", text=f"generation failed: {e}", iteration=it)
                return
            yield Event(kind="model", text=reply, iteration=it)
            messages.append({"role": "assistant", "content": reply})

            m = TOOL_CALL_RE.search(reply)
            if not m:
                yield Event(kind="done", text="no tool call — stopping", iteration=it)
                return

            try:
                payload = json.loads(m.group(1))
                cmd = payload["args"]["cmd"]
            except Exception as e:
                yield Event(kind="error", text=f"malformed tool call: {e}", iteration=it)
                return

            yield Event(kind="tool_call", cmd=cmd, iteration=it)

            blocked = is_hard_blocked(cmd)
            if blocked:
                yield Event(kind="blocked", cmd=cmd, text=f"hard-blocked by pattern: {blocked}", iteration=it)
                messages.append({"role": "tool", "content": "[blocked: command refused for safety]"})
                continue

            should_run = False
            if self.mode == "dry-run":
                should_run = False
            elif self.mode == "allow":
                if matches_allowlist(cmd, self._allowlist):
                    should_run = True
                else:
                    should_run = self.request_approval(cmd)
            elif self.mode == "approve":
                should_run = self.request_approval(cmd)

            if not should_run:
                yield Event(kind="skipped", cmd=cmd, text="not approved / dry-run", iteration=it)
                messages.append({"role": "tool", "content": "[command not executed]"})
                continue

            rc, out, err = run_shell(cmd, timeout=self.timeout)
            yield Event(kind="tool_result", cmd=cmd, stdout=out, stderr=err, return_code=rc, iteration=it)
            tool_payload = (out or "") + (("\n[stderr]\n" + err) if err else "")
            messages.append({"role": "tool", "content": tool_payload[:8000] or f"[exit {rc}]"})

        yield Event(kind="done", text=f"max_iters reached ({self.max_iters})", iteration=self.max_iters)


# ---------- CLI ----------

def _cli_approval(cmd: str) -> bool:
    print(f"\n[agent] proposes: {cmd}")
    try:
        ans = input("execute? [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--goal", required=True)
    p.add_argument("--mode", choices=("dry-run", "approve", "allow"), default="dry-run")
    p.add_argument("--allowlist", default=None, help="path to allowlist regex file (one per line)")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    sess = AgentSession(
        model_path=args.model,
        adapter_path=args.adapter,
        mode=args.mode,
        allowlist_path=args.allowlist,
        timeout=args.timeout,
        max_iters=args.max_iters,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # In CLI mode, hijack the approval workflow to use stdin.
    sess.request_approval = lambda cmd, timeout=300.0: _cli_approval(cmd)  # type: ignore[assignment]

    for ev in sess.run(args.goal):
        line = json.dumps(asdict(ev), ensure_ascii=False)
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
