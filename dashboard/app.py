"""FastAPI backend for the LM Studio + Custom Trainer dashboard.

Endpoints:
  GET  /api/state            current job + listings
  GET  /api/models           list downloaded HF models
  GET  /api/datasets         list uploaded datasets
  GET  /api/runs             list training run output dirs
  POST /api/download         {repo_id, token?} -> kicks off model download
  POST /api/upload-dataset   multipart file upload (.jsonl)
  POST /api/train            {model, dataset, output, epochs, batch_size, lr, lora_r}
  POST /api/cancel           cancel current job
  POST /api/launch-lmstudio  launches LM Studio AppImage
  GET  /api/logs/stream      SSE log stream of current/last job
  GET  /                     dashboard UI
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import gpu_profile  # noqa: E402
MODELS_DIR = ROOT / "models"
DATASETS_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
for d in (MODELS_DIR, DATASETS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
LMSTUDIO_PATH = Path(os.environ.get("LMSTUDIO_DIR", str(Path.home() / "LMStudio"))) / "LMStudio.AppImage"


class Job:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.kind: Optional[str] = None  # "download" | "train"
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.return_code: Optional[int] = None
        self.label: str = ""
        self.log: deque[str] = deque(maxlen=4000)
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> dict[str, Any]:
        return {
            "running": self.is_running(),
            "kind": self.kind,
            "label": self.label,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "return_code": self.return_code,
            "log_tail": list(self.log)[-50:],
        }

    def start(self, argv: list[str], kind: str, label: str, cwd: Optional[Path] = None) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError(f"A {self.kind} job is already running.")
            self.kind = kind
            self.label = label
            self.started_at = time.time()
            self.ended_at = None
            self.return_code = None
            self.log.clear()
            self.log.append(f"$ {' '.join(shlex.quote(a) for a in argv)}")
            self.proc = subprocess.Popen(
                argv,
                cwd=str(cwd or ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._reader_thread = threading.Thread(target=self._pump, daemon=True)
            self._reader_thread.start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.log.append(line.rstrip("\n"))
        self.proc.wait()
        self.return_code = self.proc.returncode
        self.ended_at = time.time()
        self.log.append(f"[exit {self.return_code}]")

    def cancel(self) -> bool:
        with self._lock:
            if not self.is_running():
                return False
            assert self.proc is not None
            try:
                self.proc.send_signal(signal.SIGTERM)
            except Exception:
                self.proc.kill()
            return True


JOB = Job()


# ---------- agent session (model + LoRA + shell tool) ----------

class AgentManager:
    """Wraps a single AgentSession at a time. Streams events to the UI and
    handles the approval handshake (UI POSTs /api/agent/approve)."""

    def __init__(self) -> None:
        self.session = None              # type: ignore[assignment]
        self.events: list[dict] = []
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.running = False
        self.last_goal: Optional[str] = None

    def start(self, model_path: str, adapter_path: Optional[str], goal: str,
              mode: str, max_iters: int, timeout: int, system: Optional[str]) -> None:
        # Lazy-imported here so the dashboard boots fast even without torch.
        from agent import AgentSession
        with self.lock:
            if self.running:
                raise RuntimeError("agent is already running")
            self.session = AgentSession(
                model_path=model_path,
                adapter_path=adapter_path,
                mode=mode,
                max_iters=max_iters,
                timeout=timeout,
            )
            self.events = [{"kind": "start", "text": f"goal: {goal}", "iteration": 0}]
            self.last_goal = goal
            self.running = True

        def _run():
            from dataclasses import asdict
            try:
                for ev in self.session.run(goal=goal, system=system):  # type: ignore[union-attr]
                    self.events.append(asdict(ev))
            except Exception as e:
                self.events.append({"kind": "error", "text": str(e), "iteration": -1})
            finally:
                self.running = False

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def status(self) -> dict[str, Any]:
        pending = self.session.pending_command() if self.session else None
        return {
            "running": self.running,
            "goal": self.last_goal,
            "pending_command": pending,
            "events": self.events[-200:],
        }

    def approve(self, decision: bool) -> bool:
        if not self.session:
            return False
        return self.session.submit_approval(decision)


AGENT = AgentManager()


app = FastAPI(title="LM Studio Dashboard")


def _auto_launch_lmstudio() -> None:
    """Launch LM Studio alongside the dashboard if installed and a display is available.
    Disable by setting AUTO_LAUNCH_LMSTUDIO=0. Headless boxes (ZimaOS/CasaOS containers)
    are auto-detected via missing $DISPLAY and skipped."""
    if os.environ.get("AUTO_LAUNCH_LMSTUDIO", "1") == "0":
        return
    if not LMSTUDIO_PATH.exists():
        print(f"[startup] LM Studio not found at {LMSTUDIO_PATH}, skipping auto-launch.")
        return
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("[startup] No display server detected, skipping LM Studio auto-launch.")
        return
    try:
        subprocess.Popen(
            [str(LMSTUDIO_PATH)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"[startup] Launched LM Studio: {LMSTUDIO_PATH}")
    except Exception as e:
        print(f"[startup] Failed to launch LM Studio: {e}")


@app.on_event("startup")
def _on_startup() -> None:
    _auto_launch_lmstudio()


# ---------- listings ----------

def list_models() -> list[dict[str, Any]]:
    out = []
    for p in sorted(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else []:
        if p.is_dir():
            cfg = p / "config.json"
            out.append({"name": p.name, "path": str(p), "has_config": cfg.exists()})
    return out


def list_datasets() -> list[dict[str, Any]]:
    out = []
    for p in sorted(DATASETS_DIR.glob("*.jsonl")):
        try:
            n = sum(1 for _ in p.open("r", encoding="utf-8"))
        except Exception:
            n = -1
        out.append({"name": p.name, "path": str(p), "lines": n, "size": p.stat().st_size})
    return out


def list_runs() -> list[dict[str, Any]]:
    out = []
    for p in sorted(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []:
        if p.is_dir():
            out.append({
                "name": p.name,
                "path": str(p),
                "modified": p.stat().st_mtime,
                "has_adapter": (p / "adapter_model.safetensors").exists()
                                or any(p.glob("adapter_*.bin")),
            })
    return out


# ---------- models ----------

class DownloadReq(BaseModel):
    repo_id: str
    token: Optional[str] = None
    revision: Optional[str] = None


class DatasetDownloadReq(BaseModel):
    repo_id: str                    # e.g. Canstralian/pentesting_dataset
    split: Optional[str] = "train"
    text_field: Optional[str] = None  # auto-detect if None
    token: Optional[str] = None
    out_name: Optional[str] = None    # filename under ./data (defaults to <repo>.jsonl)


class TrainReq(BaseModel):
    model: str            # path under ./models or hub id
    dataset: str          # path to jsonl
    output: str           # subdir name under ./runs
    epochs: float = 3.0
    # Scaling controls. None ⇒ let auto-tune pick (or fallback default).
    auto_tune: bool = True
    num_gpus: Optional[int] = None              # None -> use all visible GPUs
    gpu_ids: Optional[list[int]] = None         # explicit selection beats num_gpus
    exclude_smallest: bool = False              # auto-drop heterogeneous outliers
    strategy: str = "auto"                       # auto|single|ddp|fsdp|zero3
    mixed_precision: str = "bf16"
    batch_size: Optional[int] = None
    grad_accum: Optional[int] = None
    lr: float = 2e-4
    max_length: Optional[int] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    no_4bit: bool = False
    gradient_checkpointing: Optional[bool] = None
    target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    # Speed knobs (defaults match finetune.py defaults).
    attn_impl: Optional[str] = None              # flash_attention_2 | sdpa | eager
    no_packing: bool = False
    no_group_by_length: bool = False
    compile: bool = False
    num_workers: int = 4


# ---------- routes ----------

def _gpu_summary() -> dict[str, Any]:
    """Cached-on-call detection summary used by the UI to populate scaling controls."""
    try:
        return gpu_profile._detect_summary()
    except Exception as e:
        return {"gpus": [], "num_gpus": 0, "error": str(e),
                "profile": None, "recommended_strategy": "single"}


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return {
        "job": JOB.status(),
        "models": list_models(),
        "datasets": list_datasets(),
        "runs": list_runs(),
        "lmstudio_installed": LMSTUDIO_PATH.exists(),
        "lmstudio_path": str(LMSTUDIO_PATH),
        "gpu": _gpu_summary(),
    }


@app.get("/api/gpus")
def get_gpus() -> dict[str, Any]:
    return _gpu_summary()


@app.post("/api/download")
def download(req: DownloadReq) -> dict[str, Any]:
    if not req.repo_id.strip():
        raise HTTPException(400, "repo_id is required")
    argv = [PYTHON, str(ROOT / "download_model.py"), req.repo_id, "--dest", str(MODELS_DIR)]
    if req.revision:
        argv += ["--revision", req.revision]
    if req.token:
        argv += ["--token", req.token]
    try:
        JOB.start(argv, kind="download", label=f"download {req.repo_id}")
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.post("/api/download-dataset")
def download_dataset(req: DatasetDownloadReq) -> dict[str, Any]:
    if not req.repo_id.strip():
        raise HTTPException(400, "repo_id is required")
    out_name = req.out_name or (req.repo_id.replace("/", "__") + ".jsonl")
    argv = [
        PYTHON, str(ROOT / "fetch_hf_dataset.py"),
        req.repo_id,
        "--out", str(DATASETS_DIR / out_name),
        "--split", req.split or "train",
    ]
    if req.text_field:
        argv += ["--text-field", req.text_field]
    if req.token:
        argv += ["--token", req.token]
    try:
        JOB.start(argv, kind="download", label=f"dataset {req.repo_id}")
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True, "out_name": out_name}


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.endswith(".jsonl"):
        raise HTTPException(400, "Only .jsonl files are accepted.")
    safe = Path(file.filename).name
    dest = DATASETS_DIR / safe
    content = await file.read()
    dest.write_bytes(content)
    return {"ok": True, "name": safe, "size": len(content)}


@app.post("/api/train")
def train(req: TrainReq) -> dict[str, Any]:
    model_path = req.model
    if not model_path.startswith("/") and not model_path.startswith("./"):
        candidate = MODELS_DIR / model_path
        if candidate.exists():
            model_path = str(candidate)

    dataset_path = req.dataset
    if not dataset_path.startswith("/") and not dataset_path.startswith("./"):
        candidate = DATASETS_DIR / dataset_path
        if candidate.exists():
            dataset_path = str(candidate)

    if not Path(dataset_path).exists():
        raise HTTPException(400, f"dataset not found: {dataset_path}")
    if req.strategy not in gpu_profile.VALID_STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {gpu_profile.VALID_STRATEGIES}")

    out_dir = RUNS_DIR / req.output

    # Resolve GPU selection.
    detected = gpu_profile.detect_gpus()
    n_avail = len(detected)
    selected = gpu_profile.filter_gpus(
        detected,
        gpu_ids=req.gpu_ids,
        exclude_smallest=req.exclude_smallest,
    )
    if req.gpu_ids is not None or req.exclude_smallest:
        num_gpus = len(selected) or 1
    else:
        num_gpus = req.num_gpus if req.num_gpus is not None else max(1, n_avail)
        num_gpus = max(1, min(num_gpus, max(1, n_avail) if n_avail else 1))

    strategy = req.strategy
    if strategy == "auto":
        strategy = "ddp" if num_gpus > 1 else "single"

    # train.sh handles single-vs-multi launch and accelerate config generation.
    argv: list[str] = [
        str(ROOT / "train.sh"),
        "--num-gpus", str(num_gpus),
        "--strategy", strategy,
        "--mixed-precision", req.mixed_precision,
    ]
    if req.gpu_ids:
        argv += ["--gpu-ids", ",".join(str(i) for i in req.gpu_ids)]
    elif req.exclude_smallest:
        argv += ["--exclude-smallest"]

    argv += [
        "--model", model_path,
        "--data", dataset_path,
        "--output", str(out_dir),
        "--epochs", str(req.epochs),
        "--lr", str(req.lr),
        "--target-modules", req.target_modules,
        "--num-workers", str(req.num_workers),
    ]
    if req.auto_tune:
        argv.append("--auto-tune")
    if req.batch_size is not None:
        argv += ["--batch-size", str(req.batch_size)]
    if req.grad_accum is not None:
        argv += ["--grad-accum", str(req.grad_accum)]
    if req.max_length is not None:
        argv += ["--max-length", str(req.max_length)]
    if req.lora_r is not None:
        argv += ["--lora-r", str(req.lora_r)]
    if req.lora_alpha is not None:
        argv += ["--lora-alpha", str(req.lora_alpha)]
    if req.no_4bit:
        argv.append("--no-4bit")
    if req.gradient_checkpointing is True:
        argv.append("--gradient-checkpointing")
    elif req.gradient_checkpointing is False:
        argv.append("--no-gradient-checkpointing")
    if req.attn_impl:
        argv += ["--attn-impl", req.attn_impl]
    if req.no_packing:
        argv.append("--no-packing")
    if req.no_group_by_length:
        argv.append("--no-group-by-length")
    if req.compile:
        argv.append("--compile")

    label = f"train {req.output} ({num_gpus}× {strategy})"
    try:
        JOB.start(argv, kind="train", label=label)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True, "num_gpus": num_gpus, "strategy": strategy}


@app.post("/api/cancel")
def cancel() -> dict[str, Any]:
    return {"ok": JOB.cancel()}


class AgentRunReq(BaseModel):
    model: str
    adapter: Optional[str] = None
    goal: str
    mode: str = "approve"            # dry-run | approve | allow
    max_iters: int = 6
    timeout: int = 30
    system: Optional[str] = None


class AgentApproveReq(BaseModel):
    approve: bool


@app.post("/api/agent/run")
def agent_run(req: AgentRunReq) -> dict[str, Any]:
    if req.mode not in ("dry-run", "approve", "allow"):
        raise HTTPException(400, "mode must be dry-run|approve|allow")

    model_path = req.model
    if not model_path.startswith("/") and not model_path.startswith("./"):
        candidate = MODELS_DIR / model_path
        if candidate.exists():
            model_path = str(candidate)

    adapter_path = req.adapter
    if adapter_path and not adapter_path.startswith("/") and not adapter_path.startswith("./"):
        candidate = RUNS_DIR / adapter_path
        if candidate.exists():
            adapter_path = str(candidate)

    try:
        AGENT.start(
            model_path=model_path,
            adapter_path=adapter_path,
            goal=req.goal,
            mode=req.mode,
            max_iters=req.max_iters,
            timeout=req.timeout,
            system=req.system,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.get("/api/agent/state")
def agent_state() -> dict[str, Any]:
    return AGENT.status()


@app.post("/api/agent/approve")
def agent_approve(req: AgentApproveReq) -> dict[str, Any]:
    ok = AGENT.approve(req.approve)
    return {"ok": ok}


@app.get("/api/agent/stream")
async def agent_stream():
    async def gen():
        last = 0
        while True:
            evs = list(AGENT.events)
            if len(evs) > last:
                for ev in evs[last:]:
                    yield {"event": "agent", "data": json.dumps(ev)}
                last = len(evs)
            yield {"event": "status", "data": json.dumps({
                "running": AGENT.running,
                "pending_command": AGENT.session.pending_command() if AGENT.session else None,
            })}
            await asyncio.sleep(0.4)
    return EventSourceResponse(gen())


@app.post("/api/launch-lmstudio")
def launch_lmstudio() -> dict[str, Any]:
    if not LMSTUDIO_PATH.exists():
        raise HTTPException(404, f"LM Studio not found at {LMSTUDIO_PATH}")
    subprocess.Popen(
        [str(LMSTUDIO_PATH)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"ok": True}


@app.get("/api/logs/stream")
async def logs_stream():
    async def gen():
        last = 0
        while True:
            log_list = list(JOB.log)
            if len(log_list) > last:
                for line in log_list[last:]:
                    yield {"event": "log", "data": line}
                last = len(log_list)
            yield {"event": "status", "data": json.dumps(JOB.status())}
            await asyncio.sleep(0.5)
    return EventSourceResponse(gen())


# ---------- static UI ----------

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8765, reload=False)
