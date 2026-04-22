# LM Studio + Custom Trainer + Material 3 Dashboard

End-to-end install scripts for Ubuntu (and ZimaOS / CasaOS) that set up:

- **LM Studio** for local LLM inference
- **PyTorch + Hugging Face stack** for fine-tuning
- A **custom LoRA trainer** that auto-scales from **6 GB → 96 GB per-GPU VRAM** and **1 → 9 GPUs** (DDP / FSDP / DeepSpeed ZeRO-3)
- A **Material 3 web dashboard** that drives downloads, training, and an agent loop, and auto-launches LM Studio
- A **command-execution agent** that runs shell commands the trained model emits — with hard-blocks, allowlists, and per-command approval
- **systemd** auto-start on boot, or **Docker Compose** deployment for ZimaOS/CasaOS
- A **cybersecurity dataset pipeline** (Canstralian/pentesting_dataset + the gfek/Real-CyberSecurity-Datasets catalog)

---

## Get the code (Linux)

```bash
sudo apt update && sudo apt install -y git           # if git isn't already installed
git clone https://github.com/NextLVLHasH/Linux-Install-Scripts.git
cd Linux-Install-Scripts
chmod +x *.sh                                        # make the installers executable
```

To grab updates later:

```bash
git pull
```

---

## Quick start (Ubuntu desktop)

```bash
./install-all.sh                      # installs everything; reboot if NVIDIA driver was new
./09-start-dashboard.sh               # or just visit http://<host>:8765 — systemd already started it
```

Open the dashboard, click **Launch LM Studio**, download a base model, upload or fetch a dataset, set LoRA params, hit **Start training**.

## Quick start (headless ZimaOS / CasaOS)

```bash
./install-casaos.sh                   # wraps: curl -fsSL https://get.casaos.io | sudo bash
docker compose up -d --build          # http://<host>:8765
```

Compose is GPU-passthrough by default (`--gpus all` equivalent) and sets
`shm_size=8gb` + `ipc=host` for stable NCCL collectives across multiple GPUs.

---

## Scaling: 6 GB → 96 GB · 1 GPU → 9 GPUs

The trainer detects each GPU's VRAM via `nvidia-smi` and picks a profile:

| Profile | per-GPU VRAM | batch | grad_accum | max_len | LoRA r | 4-bit | grad_ckpt | Recommended max model |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `xs-6gb`   | ≥ 6 GB  | 1  | 32 | 512  | 8   | yes | yes | ~1.5B |
| `s-12gb`   | ≥ 10 GB | 1  | 16 | 1024 | 16  | yes | yes | ~3B |
| `m-24gb`   | ≥ 16 GB | 2  | 8  | 2048 | 32  | yes | yes | ~7-8B (4-bit) |
| `l-48gb`   | ≥ 32 GB | 4  | 4  | 4096 | 64  | no  | yes | ~13-14B (bf16) |
| `xl-80gb`  | ≥ 64 GB | 8  | 2  | 4096 | 64  | no  | no  | ~30-34B |
| `xxl-96gb` | ≥ 88 GB | 16 | 1  | 8192 | 128 | no  | no  | ~65-70B |

Profiles live in [gpu_profile.py](gpu_profile.py) — tune them to your hardware.

### Multi-GPU strategies

| Strategy | Use when |
| --- | --- |
| `single` | One GPU, or single-process model parallelism (`device_map=auto`). |
| `ddp` | Multiple GPUs, model fits in one GPU. Default for >1 GPU. |
| `fsdp` | Multiple GPUs, model doesn't fit per-GPU. PyTorch-native, no extra deps. |
| `zero3` | DeepSpeed ZeRO-3 sharding. Best for huge models and CPU offload. |
| `auto` | Pick `single` for 1 GPU, `ddp` for >1. |

The dashboard exposes all of this via dropdowns. From the CLI:

```bash
# Auto-tune on whatever GPU(s) are present:
./train.sh --auto-tune --model ./models/Qwen__Qwen2.5-0.5B \
    --data ./data/sample_cybersec.jsonl --output ./runs/r1

# 4× GPU DDP, auto-tuned:
./train.sh --num-gpus 4 --strategy ddp --auto-tune \
    --model ./models/Llama-3.1-8B \
    --data ./data/my.jsonl --output ./runs/llama8b-ddp

# 8× GPU ZeRO-3 for a 70B model:
./train.sh --num-gpus 8 --strategy zero3 --auto-tune \
    --model ./models/Llama-3.1-70B \
    --data ./data/my.jsonl --output ./runs/llama70b-zero3
```

`train.sh` decides between plain `python finetune.py` (single GPU) and
`accelerate launch --config_file <generated.yaml>` (multi-GPU), generating
the accelerate config at runtime via [gpu_profile.py](gpu_profile.py).

Inspect detection:

```bash
python gpu_profile.py detect          # JSON: gpus + recommended profile
python gpu_profile.py args            # CLI args the picked profile would set
python gpu_profile.py accelerate --num-gpus 4 --strategy fsdp
```

---

## Agent: train a model to run commands

Beyond classification, you can train the model to **emit and execute shell
commands** in a tool-use loop. The training format is the chat shape with a
`<tool_call>` payload — see [data/sample_agent_commands.jsonl](data/sample_agent_commands.jsonl)
for ten worked examples (interface lookup, port scan, log triage, refusal of
destructive actions, etc.).

Workflow:

1. Train on agent-style data: `--data ./data/sample_agent_commands.jsonl`.
   The trainer auto-detects the `messages`/chat format.
2. Open the dashboard's **Agent** card. Pick a base model, optionally a LoRA
   adapter from your run, set a goal, choose an approval mode, click **Run**.
3. The model writes `<tool_call>{...}</tool_call>` blocks. The runtime parses
   them, asks you to approve, runs the shell command, feeds the output back,
   and loops until the model stops emitting tool calls or hits `max_iters`.

### Safety layers (in order)

1. **Hard-blocked patterns** (always refused):
   `rm -rf /`, `mkfs.*`, `dd of=/dev/*`, fork bombs, `shutdown`/`reboot`/`poweroff`,
   raw-disk redirection, `chmod 777 /`. See [agent.py:HARD_BLOCKED](agent.py#L37).
2. **Approval mode** (`approve` is the dashboard default): every command needs
   a click before it runs.
3. **Allowlist** (`--allowlist regex.txt`, `--mode allow`): commands matching
   any regex auto-run; everything else still prompts.
4. **dry-run mode**: model proposes commands, runtime never executes — useful
   right after training to evaluate the model's behaviour.
5. **Per-command timeout** (default 30 s) and **iteration cap** (default 6).
6. **Output truncation** (16 KB) before feeding back to the model.

CLI:

```bash
source ~/pytorch-env/bin/activate
python agent.py \
    --model ./models/Qwen__Qwen2.5-0.5B \
    --adapter ./runs/agent-finetune \
    --goal "list listening tcp services" \
    --mode approve --max-iters 4
```

> ⚠ Run on a **trusted host** — ideally inside the Docker container, a VM, or
> a dedicated unprivileged user. The runtime executes commands as whoever
> launched it.

---

## File map

### Install scripts (run in order, or via `install-all.sh`)
| File | What it does |
| --- | --- |
| [01-update-system.sh](01-update-system.sh) | apt update / upgrade / dist-upgrade / autoremove |
| [02-install-prerequisites.sh](02-install-prerequisites.sh) | build tools, Python 3, AppImage/GUI libs |
| [03-install-nvidia-cuda.sh](03-install-nvidia-cuda.sh) | NVIDIA driver + CUDA toolkit (auto-skipped on CPU boxes) |
| [04-install-pytorch.sh](04-install-pytorch.sh) | venv + PyTorch (CUDA 12.1 or CPU wheels) |
| [05-install-lmstudio.sh](05-install-lmstudio.sh) | LM Studio AppImage + `.desktop` launcher |
| [06-install-training-deps.sh](06-install-training-deps.sh) | transformers, peft, trl, datasets, bitsandbytes, **deepspeed**, PyYAML |
| [07-download-model.sh](07-download-model.sh) | wrapper: pulls a HF model into `./models/` |
| [08-install-dashboard.sh](08-install-dashboard.sh) | FastAPI + uvicorn + SSE deps |
| [09-start-dashboard.sh](09-start-dashboard.sh) | runs the dashboard on `0.0.0.0:8765` |
| [10-install-systemd.sh](10-install-systemd.sh) | enables `lmstudio-dashboard.service` for boot auto-start |
| [11-fetch-cybersec-datasets.sh](11-fetch-cybersec-datasets.sh) | pulls Canstralian/pentesting_dataset + clones the gfek catalog |
| [install-casaos.sh](install-casaos.sh) | optional: installs CasaOS on the host |
| [install-all.sh](install-all.sh) | runs steps 1–11; honors `SKIP_*` env vars |
| [start.sh](start.sh) | activates the venv and launches LM Studio (CLI shortcut) |

### Python tools
| File | What it does |
| --- | --- |
| [gpu_profile.py](gpu_profile.py) | VRAM detection, 6 training profiles, accelerate-config generator |
| [download_model.py](download_model.py) | snapshot a HF model into `./models/<repo>` |
| [fetch_hf_dataset.py](fetch_hf_dataset.py) | pull a HF dataset and convert it to trainer JSONL |
| [csv_to_jsonl.py](csv_to_jsonl.py) | convert CSVs (NSL-KDD, CIC-IDS, phishing feeds) to instruction JSONL |
| [finetune.py](finetune.py) | LoRA SFT trainer; auto-tune from `gpu_profile`; multi-GPU via accelerate |
| [train.sh](train.sh) | smart launcher — single-process or `accelerate launch` |
| [agent.py](agent.py) | model + LoRA + shell tool runtime with hard-blocks/approval/allowlist |

### Dashboard
| File | What it does |
| --- | --- |
| [dashboard/app.py](dashboard/app.py) | FastAPI backend; SSE log + agent streams; auto-launches LM Studio on startup |
| [dashboard/static/index.html](dashboard/static/index.html) | Material 3 UI: hardware card, scaling controls, agent tab |
| [dashboard/static/app.js](dashboard/static/app.js) | dashboard client logic |
| [dashboard/static/styles.css](dashboard/static/styles.css) | Material 3 tokens + layout (light + dark) |
| [dashboard/Dockerfile](dashboard/Dockerfile) | CUDA 12.1 + trainer + DeepSpeed + dashboard, headless |
| [docker-compose.yml](docker-compose.yml) | ZimaOS/CasaOS-ready Compose with multi-GPU passthrough + NCCL tuning |
| [systemd/lmstudio-dashboard.service](systemd/lmstudio-dashboard.service) | unit template (filled in by step 10) |

### Data
| File | What it is |
| --- | --- |
| [data/sample_cybersec.jsonl](data/sample_cybersec.jsonl) | classification-style cybersecurity samples |
| [data/sample_agent_commands.jsonl](data/sample_agent_commands.jsonl) | chat-format tool-use samples for the agent |

---

## Custom training data

The trainer accepts JSONL with one of four record shapes (auto-detected):

```jsonl
{"prompt": "...", "completion": "..."}
{"instruction": "...", "input": "...", "output": "..."}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"text": "raw passage to language-model on"}
```

For tool-using agents, embed the tool call inside an assistant turn:

```jsonl
{"messages":[{"role":"system","content":"You have a shell tool. To run a command emit <tool_call>{\"name\":\"shell\",\"args\":{\"cmd\":\"...\"}}</tool_call>"},
              {"role":"user","content":"What's my IP?"},
              {"role":"assistant","content":"<tool_call>{\"name\":\"shell\",\"args\":{\"cmd\":\"ip -4 addr\"}}</tool_call>"},
              {"role":"tool","content":"<output>"},
              {"role":"assistant","content":"Your IP is..."}]}
```

Three ways to get data in:

1. **Upload** a local `.jsonl` from the dashboard.
2. **Fetch from Hugging Face** via the dashboard's HF input (defaults to
   `Canstralian/pentesting_dataset`). Falls back to `--text-field <name>`.
3. **Convert CSVs** with [csv_to_jsonl.py](csv_to_jsonl.py) — useful for
   the public security datasets indexed in
   [data/cybersec-catalog/](data/cybersec-catalog/) (cloned by step 11
   from [gfek/Real-CyberSecurity-Datasets](https://github.com/gfek/Real-CyberSecurity-Datasets)).

---

## Auto-start behavior

- **Bare-metal Ubuntu/ZimaOS desktop:** step 10 installs a systemd unit. The
  dashboard starts on every boot. With `DISPLAY=:0` set in the unit, the
  dashboard's startup hook also auto-launches the LM Studio AppImage.
- **Headless server / Docker:** the container sets `AUTO_LAUNCH_LMSTUDIO=0`
  and the dashboard skips the GUI launch (no display server present).
- Disable manually: `sudo systemctl disable --now lmstudio-dashboard` or set
  `AUTO_LAUNCH_LMSTUDIO=0` in the unit.

---

## Environment overrides

| Var | Default | Used by |
| --- | --- | --- |
| `VENV_DIR` | `~/pytorch-env` | all venv-using scripts |
| `LMSTUDIO_DIR` | `~/LMStudio` | step 5, dashboard, systemd unit |
| `LMSTUDIO_URL` | pinned to 0.3.9-6 | step 5 — set to current URL from https://lmstudio.ai/ if 404s |
| `DASHBOARD_HOST` | `0.0.0.0` | step 9 |
| `DASHBOARD_PORT` | `8765` | step 9 |
| `AUTO_LAUNCH_LMSTUDIO` | `1` | dashboard startup |
| `SKIP_*` | — | `install-all.sh` (e.g. `SKIP_NVIDIA=1 SKIP_CYBERSEC=1`) |

---

## Notes & caveats

- LM Studio distributes **GGUF (inference-only)** weights. Training requires
  the HF source weights — the dashboard's "Download base model" pulls those
  via `huggingface_hub`. You can't fine-tune a `.gguf` file directly.
- 4-bit quantized loading via `bitsandbytes` only works on **CUDA**, and is
  **not compatible with FSDP/ZeRO-3** sharding — pick `--strategy ddp` if you
  want 4-bit + multi-GPU. The trainer detects this and silently disables
  4-bit when FSDP/ZeRO are active.
- The CUDA wheel index in [04-install-pytorch.sh:19](04-install-pytorch.sh#L19)
  is hard-coded to `cu121`. Edit if you need cu118 or cu124.
- The cybersec catalog repo is an **index** of datasets, not the data itself.
  Most entries link to external CSVs/PCAPs — download them, then run
  [csv_to_jsonl.py](csv_to_jsonl.py) to shape them for the trainer.
- The agent runtime executes commands **as the dashboard's user**. For real
  isolation use the Docker container, a VM, or a dedicated unprivileged user
  — and keep the dashboard off the public internet (no auth is built in).
