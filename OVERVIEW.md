# Project Overview

A self-contained toolkit for going from **bare Ubuntu (or ZimaOS / CasaOS)** to
**a locally-trained, locally-running LLM that can also execute shell commands
on your behalf** — driven by a Material 3 web dashboard.

This document is the *narrative* tour. For the file-by-file reference, see
[README.md](README.md).

---

## 1. What this is (and isn't)

**Is:**
- A reproducible install path for the full local-LLM stack (driver → CUDA →
  PyTorch → Hugging Face → llama.cpp `llama-server`).
- A LoRA fine-tuner that auto-scales from a single 6 GB consumer GPU up to
  a 9 × 96 GB rig (864 GB sharded).
- A web dashboard you can use without ever touching the CLI.
- A tool-using agent runtime with explicit safety layers.
- A working cybersecurity-training example end-to-end.

**Isn't:**
- A multi-tenant service. There's no auth — keep it on a trusted LAN or
  inside a container.
- A GGUF converter. Trained adapters stay in HF format; conversion to GGUF
  for llama-server is a separate step you run yourself.
- A production training cluster. It's tuned for one host (1-9 GPUs).
  Multi-node would need an accelerate `--num_machines >1` config and shared
  storage.

---

## 2. Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Material 3 Dashboard                         │
│   (Material Web components via ESM, served by FastAPI on :8765)      │
│                                                                       │
│   Hardware │ Models │ Datasets │ Fine-tune │ Job log │ Runs │ Agent  │
└──────────────────────────────────────────────────────────────────────┘
                  ↑              ↑                ↑
                  │ HTTP/SSE     │ HTTP/SSE       │ HTTP/SSE
┌─────────────────┴──────────────┴────────────────┴───────────────────┐
│                       FastAPI backend (app.py)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │ Job runner  │  │ AgentManager │  │ Listings + GPU detection   │  │
│  │ (subprocess)│  │ (in-process) │  │ (gpu_profile)              │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────────────────────┘  │
└─────────┼────────────────┼──────────────────────────────────────────┘
          │                │
          │ shells out     │ imports
          ▼                ▼
┌─────────────────┐   ┌──────────────────────────────────────────────┐
│   train.sh      │   │   agent.py (AgentSession)                    │
│   ↓ decides     │   │   ┌─────────────┐  ┌──────────────────┐     │
│ single │ multi  │   │   │ generate()  │→ │ parse <tool_call>│     │
│ python │ accel. │   │   └─────────────┘  └────────┬─────────┘     │
│        │ launch │   │   ┌──────────────┐          ▼                │
│        ▼        │   │   │ run_shell()  │ ← approve / deny / block  │
│  finetune.py    │   │   └──────────────┘                           │
│  (auto-tunes    │   └──────────────────────────────────────────────┘
│   from          │
│   gpu_profile)  │
└─────────────────┘

   PERSISTENT STATE
   ./models/  ./data/  ./runs/  (volumes in Docker; bind mounts on bare metal)
```

### Component layering

| Layer | Files | Concern |
| --- | --- | --- |
| **OS provisioning** | `01-` … `05-`, `install-casaos.sh` | apt, NVIDIA, Python venv, headless llama-server, optional CasaOS |
| **ML stack** | `06-install-training-deps.sh`, `04-install-pytorch.sh` | PyTorch (CUDA 12.1), HF transformers/peft/trl/accelerate/bitsandbytes/deepspeed |
| **Hardware abstraction** | `gpu_profile.py` | VRAM detection, training profiles (6→96 GB), strategy + accelerate-config rendering |
| **Trainer** | `finetune.py`, `train.sh` | LoRA SFT, auto-tune, multi-GPU launch (DDP/FSDP/ZeRO-3) |
| **Data ingest** | `download_model.py`, `fetch_hf_dataset.py`, `csv_to_jsonl.py`, `11-fetch-cybersec-datasets.sh` | HF model snapshots, HF dataset → JSONL, CSV → JSONL, cybersec catalog |
| **Inference / agent** | `agent.py` | Load model + LoRA, parse tool calls, sandboxed shell execution |
| **Dashboard backend** | `dashboard/app.py` | FastAPI; /api/state, /api/train, /api/agent/*, SSE streams |
| **Dashboard UI** | `dashboard/static/*` | Material 3 web components, single-page app |
| **Deployment** | `09-start-dashboard.sh`, `10-install-systemd.sh`, `systemd/*`, `dashboard/Dockerfile`, `docker-compose.yml` | Foreground launch, boot service, container with multi-GPU NCCL tuning |

---

## 3. End-to-end lifecycle

The five scenarios this project is shaped around. Each one starts from the
same dashboard.

### A. Install everything on a fresh Ubuntu / ZimaOS host
```
./install-all.sh
   → 01 update OS
   → 02 prereqs (build tools, Python, optional GUI libs only if requested)
   → 03a create Python venv (/workspace/venv)
   → 08 FastAPI/uvicorn/sse-starlette
   → 03 NVIDIA driver + CUDA (skipped on CPU boxes; may reboot/resume)
   → 05 llama.cpp llama-server → ~/llama.cpp-bin/current
   → 04 PyTorch (CUDA 12.1 or CPU wheels)
   → 06 transformers/peft/trl/datasets/bitsandbytes/deepspeed
   → 11 cybersec datasets (Canstralian/pentesting + gfek catalog clone)
   → 10 systemd unit → dashboard auto-starts on boot
```
Reboot if the NVIDIA driver was new. Visit `http://<host>:8765`.

### B. Deploy headless on ZimaOS / CasaOS
```
./install-casaos.sh                          # if CasaOS isn't installed yet
docker compose up -d --build                 # GPU passthrough, NCCL tuning
                                             # http://<host>:8765
```
The container includes `llama-server` and exposes the OpenAI-compatible API on
`:1234`, with no display server required. You still get the trainer, dataset
pipeline, and agent.

### C. Train a model from the dashboard
```
1. Hardware card  → confirm GPUs detected, profile picked
2. Section 1      → "Download base model" (HF repo id)
3. Section 2      → upload .jsonl  OR  fetch from HF (defaults to
                    Canstralian/pentesting_dataset)
4. Section 3      → pick model + dataset, leave Auto-tune ON,
                    confirm GPUs and strategy, click Start training
5. Job log        → SSE-streamed live; completed runs appear in section 5
```
Behind the scenes the dashboard shells out to `train.sh`, which decides
whether to run `python finetune.py …` (single GPU) or
`accelerate launch --config_file <generated.yaml> finetune.py …`
(multi-GPU), generating the accelerate config from `gpu_profile.py`.

### D. Run the agent against a trained adapter
```
1. Section 4 (Agent) → pick base model + LoRA adapter (from your runs)
2. Mode = "approve each command"
3. Goal = "list listening tcp services on this host"
4. Click Run
5. Each <tool_call> the model emits surfaces as an approval bar:
   { check } Approve & run     {  } Deny
6. Output flows back into the model; it produces a final summary.
```

### E. Use llama-server for inference (parallel to training)
The dashboard controls llama.cpp `llama-server` on :1234. It loads GGUF
models for chat / OpenAI-compatible API access without a GUI, virtual display,
or VNC. The trainer and server share nothing: train in HF format, serve in GGUF.

---

## 4. Scaling architecture

Two orthogonal axes: **per-GPU VRAM** (memory pressure → batch / quant /
checkpointing decisions) and **GPU count + sharding** (parallelism /
bandwidth decisions).

### 4.1 The profile decision tree

```
nvidia-smi available?
├── no  → CPU profile (batch=1, max_len=512, no 4-bit, very slow)
└── yes → smallest per-GPU VRAM:
          ≥ 6 GB   → xs-6gb    (4-bit, batch=1, grad_accum=32, ≤1.5B params)
          ≥ 10 GB  → s-12gb    (4-bit, batch=1, grad_accum=16, ≤3B params)
          ≥ 16 GB  → m-24gb    (4-bit, batch=2, grad_accum=8,  ≤8B params)
          ≥ 32 GB  → l-48gb    (bf16, batch=4, grad_accum=4, gc=on, ≤14B)
          ≥ 64 GB  → xl-80gb   (bf16, batch=8, grad_accum=2, ≤34B)
          ≥ 88 GB  → xxl-96gb  (bf16, batch=16, max_len=8192, ≤70B)
```

User-facing knobs that override the profile:
- `--auto-tune` on the CLI (or the dashboard switch). Off ⇒ static defaults.
- Any explicit flag (`--batch-size`, `--max-length`, etc.) wins over the profile.

### 4.2 The strategy decision tree

```
num_gpus = 1?
├── yes → single (just python finetune.py …)
└── no  → strategy = ?
          auto  → DDP (model fits per-GPU)
          ddp   → Distributed Data Parallel (one full copy per GPU)
          fsdp  → PyTorch Fully Sharded Data Parallel (FULL_SHARD)
          zero3 → DeepSpeed ZeRO-3 (sharding + optional CPU offload)
```

**Memory math at 9 × 96 GB = 864 GB total:**

| Strategy | Effective VRAM for the model | When to use |
| --- | --- | --- |
| DDP | 96 GB | Model fits per-GPU; you want max throughput |
| FSDP | ~864 GB | Model > 96 GB; pure-PyTorch, simpler ops story |
| ZeRO-3 | 864 GB + host RAM/NVMe | Going beyond 864 GB or needing CPU offload |

**Compatibility caveats** (the trainer enforces these silently):
- 4-bit (`bitsandbytes`) is incompatible with FSDP/ZeRO-3 sharding of base
  weights. If you pick `--strategy fsdp` or `zero3`, the trainer turns
  4-bit off and falls back to bf16 weights.
- `gradient_checkpointing` requires `model.config.use_cache = False`. The
  trainer flips it for you when checkpointing is on.

### 4.3 Heterogeneous rigs (mixed VRAM tiers)

Out of the box, the profile picker uses the **smallest** GPU's VRAM as the
baseline. A 9-GPU rig of 8×H100 80GB + 1×A100 40GB clamps to the 32GB tier.
Three ways to fix that without writing code:

| Method | Where | Effect |
| --- | --- | --- |
| `--gpu-ids 0,1,2,3,4,5,6,7` | `train.sh` flag, dashboard checkbox grid | Use only those CUDA indices. Sets `CUDA_VISIBLE_DEVICES`. |
| `--exclude-smallest` | `train.sh` flag, dashboard "Exclude smallest" button | Auto-drops GPUs whose VRAM is >1 GB from the median. |
| Per-knob overrides | `--batch-size`, `--max-length`, etc. | Bypass auto-tune for the run. |

The dashboard's Hardware card shows a checkbox per GPU and flags outliers
visually; heterogeneity also raises an inline warning with a suggestion.

### 4.4 Speed knobs (minimize what crosses the interconnect)

Throughput-shaping settings, all on by default when CUDA is available:

| Knob | Default | What it does |
| --- | --- | --- |
| Flash Attention 2 | auto | 2-3× faster + ~2× lower attention memory at long seq lengths. Detected at import; falls back to SDPA, then eager. |
| TF32 matmul | on | ~2× speedup on Ampere+ with negligible accuracy loss in bf16 LoRA. |
| `cudnn.benchmark` | on | Picks the fastest conv/matmul kernels for your shape. |
| Sequence packing | on | Concatenates short samples to fill `max_length` — 2-5× useful tokens/batch on instruction data. |
| `group_by_length` | on | Bucket batches by length → less padding waste (~10-30%). |
| `dataloader_num_workers` | 4 | Parallel data prep so GPUs aren't input-starved. |
| `dataloader_pin_memory` | on | Faster H2D transfers via pinned host memory. |
| `ddp_bucket_cap_mb` | 25 | Coalesces gradient all-reduces into 25 MB buckets — fewer NCCL calls. |
| `torch.compile` | off | Big steady-state speedup but a multi-minute first step. Toggle when training >1k steps. |

All have explicit dashboard switches; CLI equivalents on `finetune.py` are
`--attn-impl`, `--no-packing`, `--no-group-by-length`, `--compile`,
`--num-workers`.

### 4.5 Memory math at 9 × 96 GB = 864 GB total

| Strategy | Effective VRAM for the model | Comm pattern (per step) |
| --- | --- | --- |
| DDP | 96 GB | Gradient all-reduce, ~2× model size on the wire |
| FSDP `FULL_SHARD` | ~864 GB | All-gather params (forward) + reduce-scatter grads |
| ZeRO-3 | 864 GB + host RAM/NVMe | All-gather params + reduce-scatter grads + optional offload |

**Interconnect dominates wall time** above 1 GPU. Rough ratios on the same
hardware:

| Interconnect | DDP slowdown vs single GPU | FSDP/ZeRO slowdown |
| --- | --- | --- |
| NVLink/NVSwitch | ~5-15% | ~15-30% |
| PCIe Gen5 x16 | ~10-25% | ~30-60% |
| PCIe Gen4 x16 | ~20-40% | ~60-150% |
| PCIe Gen4 x8 / no peer | ~40-80% | unusable for ZeRO-3 |

If you can't change the fabric: prefer DDP, raise `--grad-accum` (fewer
all-reduces per epoch), keep `--num-workers` high so I/O isn't on the
critical path.

---

## 5. Data pipeline

Everything funnels into the same **JSONL** format that `finetune.py`
auto-detects from the first record. Four shapes:

| Shape | Trigger keys | When |
| --- | --- | --- |
| Chat | `messages` | Multi-turn, system prompts, tool use |
| Alpaca | `instruction` + `output` (+ optional `input`) | Classic instruction-tuning |
| Prompt/completion | `prompt` + `completion` | Continuation-style |
| Raw | `text` | Unconditional language modeling |

### Three on-ramps

1. **Upload** a local `.jsonl` from the dashboard's section 2.
2. **Fetch from Hugging Face** (`fetch_hf_dataset.py`). Auto-maps common
   field names; pass `--text-field NAME` for unusual datasets. Default
   target is `Canstralian/pentesting_dataset` (instruction-tuning ready).
3. **Convert CSVs** (`csv_to_jsonl.py`). Each row → an Alpaca record:
   `instruction` from your template, `input` from selected feature
   columns joined as `k=v`, `output` from the label column. Designed for
   the public security CSVs cataloged in
   [gfek/Real-CyberSecurity-Datasets](https://github.com/gfek/Real-CyberSecurity-Datasets).

### Worked example — agent training data

`data/sample_agent_commands.jsonl` is the canonical chat-format example:
each record is a 3-5 turn conversation with a system prompt, user goal,
assistant emitting `<tool_call>{…}</tool_call>`, a `tool` turn with
the (synthetic) shell output, and a final assistant summary. Including
a refusal example (`Delete /var/log` → assistant declines) teaches the
model to follow the safety rule in the system prompt.

---

## 6. Agent execution model

The agent is the riskiest component because it runs **model-generated
commands on your host**. The runtime ([agent.py](agent.py)) layers
defenses in this order:

```
model output
    │
    ▼
parse <tool_call>{…}</tool_call>      ← regex; malformed JSON → error event
    │
    ▼
HARD_BLOCKED regex match?             ← rm -rf /, mkfs, dd of=/dev/*,
    │                                    fork bombs, shutdown, chmod 777 /
    │ yes → REFUSE, feed "[blocked]" back to the model
    │ no
    ▼
mode dispatch
    │
    ├── dry-run  → never executes; emits the proposed command for review
    ├── allow    → matches allowlist regex? auto-run; else ask
    └── approve  → ask the operator (CLI prompt OR dashboard approval bar)
    │
    ▼
run_shell(cmd, timeout=30)            ← bash -c; capture stdout+stderr;
    │                                    truncate to 16 KB; LC_ALL=C
    ▼
feed (stdout + [stderr]) back into the chat as a "tool" turn
    │
    ▼
loop until the model stops emitting tool calls OR max_iters reached
```

### The approval handshake (dashboard mode)

```
agent thread                              UI (browser, SSE-driven)
─────────────                            ──────────────────────────
emit "tool_call" event                   shows the command in the
emit "status" with pending_command  ──→  Approval Bar; renders
                                         { Deny }   { Approve & run }
request_approval(cmd, timeout=300)
[blocks on threading.Event]
                                         user clicks Approve →
                                         POST /api/agent/approve { approve: true }
session.submit_approval(True)       ←───
[event.set() unblocks the thread]
run_shell(cmd) → emit "tool_result"  ──→ appended to the agent log pane
loop continues
```

If the operator never clicks, `request_approval` times out (default 5 min)
and the command is treated as denied.

### What this does NOT protect against

- **Information disclosure.** The model can read any file the dashboard
  user can read. `cat /etc/shadow` won't work as non-root, but
  `cat ~/.aws/credentials` will.
- **Network egress.** Allowed commands can curl out. Use a network
  policy or run in the container with `--network=none` if that matters.
- **Tampering with the model.** A malicious model that has been trained
  to disguise destructive commands (e.g., `python -c "..."` doing rm)
  will bypass the regex blocklist. The blocklist is a guardrail against
  *plausible* mistakes, not a sandbox.
- **The dashboard itself.** No auth, no CSRF protection. Anyone on the
  network reachable on :8765 can trigger jobs and approve commands.

For real isolation: container, dedicated unprivileged user, or a VM.

---

## 7. Deployment topologies

### Bare-metal Ubuntu / ZimaOS (the default)
- venv at `/workspace/venv`, llama.cpp at `~/llama.cpp-bin/current/`,
  dashboard at `:8765`, and systemd units auto-start on boot.
- `llama-server.service` serves the pinned GGUF model on `:1234`.

### Headless Ubuntu / ZimaOS server
- Same install scripts. No GUI, virtual display, VNC, or LM Studio AppImage is
  required for the default local provider.

### Docker container (recommended for ZimaOS / CasaOS)
- `dashboard/Dockerfile` builds CUDA 12.1 + the full ML + DeepSpeed stack.
- `docker-compose.yml` exposes :8765, mounts named volumes for
  `models / data / runs / hf-cache`, and sets `shm_size=8gb`,
  `ipc=host`, and `ulimits.memlock=-1` for stable NCCL collectives
  across multiple GPUs.
- CasaOS labels in the compose file make the container appear in the
  CasaOS dashboard with a name + icon.

### CasaOS / ZimaOS app store flow
1. `./install-casaos.sh` (wraps `curl -fsSL https://get.casaos.io | sudo bash`).
2. CasaOS UI → Custom Install → paste the `docker-compose.yml`.
3. Container starts, dashboard at `http://<host>:8765`.

### Multi-node (not currently scripted)
The accelerate config generator is single-machine only
(`num_machines: 1`). Multi-node would need a per-host config with
matching `machine_rank`, a shared `models/` mount, and an explicit
`--main_process_ip`. Out of scope here.

---

## 8. Extensibility

### Add a new GPU profile
Append to `PROFILES` in [gpu_profile.py:88](gpu_profile.py#L88), keep
the list sorted ascending by `per_gpu_vram_min_gb`. The picker (line
~155) walks the list and returns the highest profile whose threshold
is met. The dashboard surfaces it automatically via `/api/state`.

### Add a new dataset format
Add a branch to `detect_format()` in [finetune.py](finetune.py) and the
matching mapper in `to_text()`. Keep the auto-detect heuristic robust
against older formats.

### Add a new tool to the agent
The tool-call format is currently `{"name": "shell", "args": {"cmd": "…"}}`.
To add (say) an HTTP tool, extend the regex parse in `AgentSession.run`
to dispatch on `payload["name"]` and add a `run_http` function alongside
`run_shell`. Update the system prompt accordingly. Update the training
data with examples using the new tool.

### Add a new training strategy
Add a case to `render_accelerate_config()` in
[gpu_profile.py:165](gpu_profile.py#L165), append the strategy name to
`VALID_STRATEGIES`, and add a `<md-select-option>` in the dashboard UI
([index.html](dashboard/static/index.html) — strategy dropdown).

---

## 9. State and storage

| Path | What lives there | Lifetime |
| --- | --- | --- |
| `./models/<repo>/` | HF model snapshots (one dir per `repo_id`) | Persistent. Bind mount or named volume. |
| `./data/*.jsonl` | Uploaded or fetched datasets | Persistent. |
| `./data/cybersec-catalog/` | Cloned gfek catalog | Re-pullable; safe to delete. |
| `./runs/<name>/` | LoRA adapters, tokenizer, `training_meta.json`, tensorboard logs | Persistent. The agent loads adapters from here. |
| `/workspace/venv/` | Python venv | Re-creatable via step 4. |
| `~/llama.cpp-bin/current/llama-server` | Headless inference server | Rebuildable via step 5. |
| `/etc/systemd/system/lmstudio-dashboard.service` | Boot unit | Created by step 10; `systemctl disable` to remove. |
| Container volumes | `models / data / runs / hf-cache` | Survive container restarts/rebuilds. |

---

## 10. Common gotchas

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `bitsandbytes` import fails | CUDA not visible, or `LD_LIBRARY_PATH` wrong | Check `nvidia-smi`. In Docker, ensure `--gpus all`. |
| Multi-GPU run hangs at startup | NCCL can't pick a device or shm too small | In Docker, set `shm_size: 8gb` and `ipc: host` (already in compose). |
| FSDP with 4-bit OOM-loops | Incompatible combination | Trainer disables 4-bit under FSDP/ZeRO; if you forced it, drop `--force-4bit`. |
| Trained adapter doesn't show in agent dropdown | No `adapter_model.safetensors` in the run dir | Training crashed before save; re-run, watch the log pane for the `[exit N]` line. |
| `llama-server` does not start | Missing GGUF model or binary | Run `./05-install-llama-server.sh`, download/select a GGUF in the Server tab, then run `./10-install-systemd.sh`. |
| `accelerate launch` says "no module named yaml" | Step 6 was skipped on a CPU box | `pip install PyYAML` into the venv. |
| Agent says "no tool call — stopping" immediately | Model not trained on tool format | Train on `data/sample_agent_commands.jsonl` (or your own), or pick a base model that already speaks tool-use. |
| `nvidia-smi` shows GPUs but `gpu_profile detect` returns empty | nvidia-smi binary missing in `$PATH` | The Docker image expects NVIDIA Container Toolkit on the host. |

---

## 11. Glossary

- **LoRA** — Low-Rank Adaptation. Fine-tunes a small set of injected
  matrices instead of the full model. ~0.1-1% of total params.
- **PEFT** — Parameter-Efficient Fine-Tuning. The HF library that
  implements LoRA, prefix tuning, etc.
- **SFT** — Supervised Fine-Tuning. The straightforward "train on
  prompt + completion" recipe used here. The TRL library provides
  `SFTTrainer`.
- **DDP** — Distributed Data Parallel. Each GPU has the full model,
  processes a different shard of the batch, gradients all-reduced.
- **FSDP** — Fully Sharded Data Parallel. PyTorch-native sharding of
  weights/grads/optimizer state across GPUs.
- **ZeRO-3** — DeepSpeed's stage-3 optimizer that shards everything
  and supports CPU/NVMe offload.
- **GGUF** — The model file format llama.cpp `llama-server` consumes.
  Quantized, inference-only, single file per model.
- **bf16 / nf4** — Mixed-precision training in 16-bit brain-float;
  4-bit normal-float for `bitsandbytes` quantization.
- **AppImage** — Self-contained Linux app bundle. Requires `libfuse2`.
- **NCCL** — NVIDIA's collectives library. Multi-GPU training depends on it.
- **CasaOS / ZimaOS** — Debian-based home-server OS with a Docker-app
  dashboard. ZimaOS is built on CasaOS. The `docker-compose.yml` in this
  repo includes CasaOS labels for app-store integration.
