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
  GET  /api/logs/stream      SSE log stream of current/last job
  GET  /                     dashboard UI

LM Studio is controlled headlessly through /api/lms/* (uses the `lms` CLI
and the LM Studio HTTP server on :1234). The dashboard does not launch the
LM Studio GUI — if you want the GUI, run ./start-lmstudio.sh separately.
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
import urllib.request
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
LMS_BIN       = Path.home() / ".lmstudio" / "bin" / "lms"
LMS_MODELS_DIR = Path(os.environ.get("LMS_MODELS_DIR", str(Path.home() / ".lmstudio" / "models")))
LMS_API_PORT  = int(os.environ.get("LMS_API_PORT", "1234"))
LMS_API_BASE  = f"http://127.0.0.1:{LMS_API_PORT}"

# Extra GGUF store used by llama-server (the file baked into the systemd unit
# lives here). Scanned alongside LM Studio's own models dir.
EXTRA_MODELS_DIR = Path(os.environ.get("EXTRA_MODELS_DIR", str(Path.home() / "models")))
LLAMA_SERVER_UNIT = "llama-server.service"


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


# ─────────────────────────────────────────────────────────────
# LM Studio management
# ─────────────────────────────────────────────────────────────

_lms_server_proc: Optional[subprocess.Popen] = None
_lms_server_lock = threading.Lock()


def _lms_argv() -> list[str]:
    """Best available prefix for lms commands."""
    if LMS_BIN.exists():
        return [str(LMS_BIN)]
    if LMSTUDIO_PATH.exists():
        return [str(LMSTUDIO_PATH), "--no-sandbox", "--"]
    return []


def _lms_healthy() -> bool:
    try:
        urllib.request.urlopen(f"{LMS_API_BASE}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def _lms_loaded_model() -> Optional[str]:
    try:
        with urllib.request.urlopen(f"{LMS_API_BASE}/v1/models", timeout=2) as r:
            data = json.loads(r.read())
            items = data.get("data", [])
            return items[0]["id"] if items else None
    except Exception:
        return None


def _lms_scan_models() -> list[dict[str, Any]]:
    """Scan both the LM Studio catalog AND the extra ~/models dir (where
    llama-server's GGUFs live). Dedup by absolute path."""
    seen: set[str] = set()
    models: list[dict[str, Any]] = []
    for root_label, root in (("lmstudio", LMS_MODELS_DIR), ("extra", EXTRA_MODELS_DIR)):
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.gguf")):
            if p.name.endswith(".incomplete"):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                rel = p.name
            models.append({
                "name": p.stem,
                "rel_path": rel,
                "abs_path": key,
                "source": root_label,
                "size_gb": round(p.stat().st_size / (1024 ** 3), 2),
            })
    return models


def _llama_server_status() -> dict[str, Any]:
    """systemctl status for the llama-server unit, when it exists."""
    try:
        out = subprocess.check_output(
            ["systemctl", "show", LLAMA_SERVER_UNIT,
             "--property=ActiveState,SubState,MainPID,Environment,LoadState"],
            text=True, timeout=3,
        )
    except Exception:
        return {"present": False}
    info: dict[str, Any] = {"present": True}
    for line in out.strip().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        info[k] = v
    info["present"] = info.get("LoadState") not in (None, "not-found", "masked")
    info["running"] = info.get("ActiveState") == "active"
    # Pull MODEL= out of the unit's Environment= for display.
    env = info.get("Environment", "") or ""
    for tok in env.split():
        if tok.startswith("MODEL="):
            info["model_path"] = tok[len("MODEL="):]
            break
    return info


def _detect_backend() -> str:
    """Which model-serving backend is currently answering on LMS_API_PORT."""
    try:
        with urllib.request.urlopen(f"{LMS_API_BASE}/v1/models", timeout=2) as r:
            body = r.read().decode("utf-8", errors="replace")
    except Exception:
        return "none"
    srv = _llama_server_status()
    if srv.get("running"):
        return "llama-server"
    if LMSTUDIO_PATH.exists():
        return "lm-studio"
    # API is up but we don't know which; trust fingerprints in the body.
    if "b8893" in body or "llama.cpp" in body.lower():
        return "llama-server"
    return "unknown"


def _vram_usage() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=4,
        )
        result = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 4:
                used, total = float(p[2]), float(p[3])
                result.append({
                    "index": int(p[0]),
                    "name": p[1],
                    "used_mb": used,
                    "total_mb": total,
                    "pct": round(100 * used / total, 1) if total else 0,
                })
        return result
    except Exception:
        return []


@app.get("/api/lms/status")
def lms_status() -> dict[str, Any]:
    running = _lms_healthy()
    llama = _llama_server_status()
    return {
        "installed": LMSTUDIO_PATH.exists(),
        "lms_cli": LMS_BIN.exists(),
        "server_running": running,
        "loaded_model": _lms_loaded_model() if running else None,
        "models": _lms_scan_models(),
        "vram": _vram_usage(),
        "api_port": LMS_API_PORT,
        # Backend reflection: which engine is actually answering, plus the
        # systemd state of the llama-server unit when we're using it.
        "backend": _detect_backend() if running else "none",
        "llama_server": llama,
        # Hot-swap is only possible with LM Studio; llama-server bakes the
        # model into the systemd unit's Environment=MODEL=... and needs a
        # restart (not a /api/lms/models/load call) to change models.
        "supports_hot_swap": running and _detect_backend() != "llama-server",
    }


@app.get("/api/lms/vram")
def lms_vram() -> list[dict[str, Any]]:
    return _vram_usage()


class LmsServerStartReq(BaseModel):
    port: int = 1234
    gpu_layers: int = -1
    context_length: int = 4096


@app.post("/api/lms/server/start")
def lms_server_start(req: LmsServerStartReq) -> dict[str, Any]:
    global _lms_server_proc
    with _lms_server_lock:
        if _lms_healthy():
            return {"ok": True, "already_running": True}
        argv = _lms_argv()
        if not argv:
            raise HTTPException(503, "LM Studio not installed.")
        cmd = argv + ["server", "start", "--port", str(req.port)]
        _lms_server_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"ok": True, "pid": _lms_server_proc.pid, "port": req.port}


@app.post("/api/lms/server/stop")
def lms_server_stop() -> dict[str, Any]:
    global _lms_server_proc
    with _lms_server_lock:
        argv = _lms_argv()
        if argv:
            subprocess.run(argv + ["server", "stop"],
                           capture_output=True, timeout=10, check=False)
        if _lms_server_proc:
            _lms_server_proc.terminate()
            try:
                _lms_server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _lms_server_proc.kill()
            _lms_server_proc = None
        return {"ok": True}


class LmsLoadReq(BaseModel):
    rel_path: str
    gpu_layers: int = -1
    context_length: int = 4096


@app.post("/api/lms/models/load")
def lms_models_load(req: LmsLoadReq) -> dict[str, Any]:
    if not _lms_healthy():
        raise HTTPException(409, "Server not running — start it first.")
    argv = _lms_argv()
    if not argv:
        raise HTTPException(503, "LM Studio not installed.")
    cmd = argv + ["load", req.rel_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise HTTPException(500, result.stderr or "lms load failed")
    return {"ok": True}


@app.post("/api/lms/models/unload")
def lms_models_unload() -> dict[str, Any]:
    argv = _lms_argv()
    if argv:
        subprocess.run(argv + ["unload"], capture_output=True, timeout=10, check=False)
    return {"ok": True}


class LmsDownloadReq(BaseModel):
    repo_id: str
    filename: str = ""
    token: str = ""


@app.post("/api/lms/models/download")
async def lms_models_download(req: LmsDownloadReq):
    LMS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    async def _stream():
        yield {"event": "log", "data": f"Downloading {req.repo_id} → {LMS_MODELS_DIR}"}
        cmd = [PYTHON, str(ROOT / "download_model.py"),
               req.repo_id, "--dest", str(LMS_MODELS_DIR)]
        if req.filename:
            cmd += ["--file", req.filename]
        if req.token:
            cmd += ["--token", req.token]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            t = line.decode(errors="replace").rstrip()
            if t:
                yield {"event": "log", "data": t}
        await proc.wait()
        if proc.returncode == 0:
            yield {"event": "done", "data": "ok"}
        else:
            yield {"event": "error", "data": f"exit {proc.returncode}"}

    return EventSourceResponse(_stream())


@app.get("/api/lms/logs/stream")
async def lms_logs_stream():
    """SSE: tails the LM Studio server log while it runs."""
    lms_log = Path.home() / ".lmstudio" / "logs" / "lms-server.log"

    async def _stream():
        pos = lms_log.stat().st_size if lms_log.exists() else 0
        while True:
            if lms_log.exists():
                with lms_log.open("r", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read(8192)
                    if chunk:
                        for line in chunk.splitlines():
                            yield {"event": "log", "data": line}
                        pos += len(chunk.encode())
            await asyncio.sleep(1)

    return EventSourceResponse(_stream())


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
