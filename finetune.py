#!/usr/bin/env python3
"""Custom LoRA fine-tuner. Accepts your own data as JSONL and scales
across 6GB → 96GB per-GPU VRAM and 1 → 9 GPUs.

Auto-detected JSONL shapes:
  1. {"prompt": "...", "completion": "..."}
  2. {"instruction": "...", "input": "...", "output": "..."}   (Alpaca)
  3. {"messages": [{"role": "...", "content": "..."}, ...]}    (Chat)
  4. {"text": "..."}                                           (Raw)

Single-GPU example:
  python finetune.py --auto-tune \\
      --model ./models/Qwen__Qwen2.5-0.5B \\
      --data ./data/sample_cybersec.jsonl \\
      --output ./runs/run-1

Multi-GPU example (use the train.sh wrapper, or accelerate directly):
  ./train.sh --num-gpus 4 --strategy ddp --auto-tune \\
      --model ./models/Llama-3.1-8B \\
      --data ./data/my.jsonl --output ./runs/llama8b-ddp

  accelerate launch --config_file accelerate_configs/ddp_4gpu.yaml \\
      finetune.py --auto-tune --model ... --data ... --output ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

import gpu_profile


# ---------- distributed helpers ----------

def _is_main_process() -> bool:
    """True on rank 0 only — used to gate logging / saving side-effects."""
    return int(os.environ.get("RANK", "0")) == 0


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def log(msg: str) -> None:
    if _is_main_process():
        print(msg, flush=True)


# ---------- dataset loading ----------

def detect_format(sample: dict) -> str:
    if "messages" in sample:
        return "chat"
    if "instruction" in sample and "output" in sample:
        return "alpaca"
    if "prompt" in sample and "completion" in sample:
        return "prompt_completion"
    if "text" in sample:
        return "text"
    raise ValueError(
        f"Unrecognized JSONL record shape. Keys: {list(sample.keys())}. "
        "Use prompt/completion, instruction/input/output, messages, or text."
    )


def load_dataset(path: Path, tokenizer) -> Dataset:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records in {path}")

    fmt = detect_format(records[0])
    log(f"[finetune] dataset format: {fmt} ({len(records)} records)")

    def to_text(rec: dict) -> dict:
        if fmt == "chat":
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
        elif fmt == "alpaca":
            instr = rec["instruction"]
            inp = rec.get("input", "")
            out = rec["output"]
            if inp:
                text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
        elif fmt == "prompt_completion":
            text = f"{rec['prompt']}{rec['completion']}"
        else:
            text = rec["text"]
        return {"text": text}

    return Dataset.from_list([to_text(r) for r in records])


# ---------- model loading ----------

def _flash_attn_available() -> bool:
    """Flash Attention 2 cuts attention memory ~2x and runtime ~2-3x at long
    seq lengths. We try-import only — if missing, transformers falls back to
    SDPA which is also fast on Ampere+."""
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def build_model(model_path: str, use_4bit: bool, gradient_checkpointing: bool,
                attn_impl: Optional[str] = None):
    """Load base model. Honors FSDP/DeepSpeed by NOT setting device_map under them
    (the launcher places shards). Sets device_map={"": local_rank} for DDP and
    'auto' for single-process multi-GPU model parallelism (rare with LoRA)."""
    kwargs: dict = {"trust_remote_code": True}

    distributed_type = os.environ.get("ACCELERATE_DISTRIBUTED_TYPE", "")
    is_fsdp_or_ds = distributed_type in ("FSDP", "DEEPSPEED")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    # Pick the fastest attention backend the install supports.
    if attn_impl is None:
        if torch.cuda.is_available() and _flash_attn_available():
            attn_impl = "flash_attention_2"
        elif torch.cuda.is_available():
            attn_impl = "sdpa"  # PyTorch 2.x fused SDPA — solid fallback
    if attn_impl and attn_impl != "eager":
        kwargs["attn_implementation"] = attn_impl

    if use_4bit and not is_fsdp_or_ds:
        # bnb 4-bit is incompatible with FSDP/DS sharding of the base weights.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["torch_dtype"] = torch.bfloat16
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16

    if local_rank >= 0 and not is_fsdp_or_ds:
        # DDP: each rank gets a full copy on its own GPU.
        kwargs["device_map"] = {"": local_rank}
    elif _world_size() == 1 and torch.cuda.device_count() > 1 and not is_fsdp_or_ds:
        # Single process, multiple GPUs visible -> spread the model (model parallel).
        kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except (ValueError, ImportError) as e:
        # flash_attention_2 isn't supported by every model arch. Retry with sdpa.
        if kwargs.get("attn_implementation") == "flash_attention_2":
            log(f"[finetune] flash_attention_2 unsupported here ({e}); retrying with sdpa")
            kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        else:
            raise
    if use_4bit and not is_fsdp_or_ds:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # Cache must be off when gradient checkpointing is on.
        model.config.use_cache = False
    return model


# ---------- argv ----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)

    # If --auto-tune is set, the missing knobs get overridden from gpu_profile.
    # Explicit CLI values always win over auto-tune.
    p.add_argument("--auto-tune", action="store_true",
                   help="Pick batch/grad/lora/etc. defaults from per-GPU VRAM.")

    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantized loading (forces bf16/fp16).")
    p.add_argument("--force-4bit", action="store_true",
                   help="Force 4-bit even if auto-tune wouldn't pick it (incompatible with FSDP/ZeRO).")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Trade ~30% throughput for ~3-5x activation-memory savings.")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--optim", default=None,
                   help="HF optimizer name (auto-tune picks paged_adamw_8bit on small VRAM).")
    p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated linear layer names for LoRA. Use 'all-linear' to wrap every linear.",
    )

    # Speed knobs. All on by default when CUDA is available.
    p.add_argument("--attn-impl", default=None,
                   choices=("flash_attention_2", "sdpa", "eager"),
                   help="Override attention backend (default: flash_attention_2 if installed, else sdpa).")
    p.add_argument("--no-packing", action="store_true",
                   help="Disable sequence packing (packing concatenates short samples to fill max_length).")
    p.add_argument("--no-group-by-length", action="store_true",
                   help="Disable length-bucketed batching (default ON to cut padding waste).")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the model in torch.compile (extra speed, slow first step).")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker count (more = faster I/O on big datasets).")
    return p


def apply_auto_tune(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in any None/unset knobs from the GPU profile."""
    gpus = gpu_profile.detect_gpus()
    if gpus:
        per_gpu = min(g.vram_gb for g in gpus)
        prof = gpu_profile.pick_profile(per_gpu)
        log(f"[auto-tune] {len(gpus)}× GPU, smallest={per_gpu}GB, profile={prof.name}")
    else:
        prof = gpu_profile.cpu_profile()
        log("[auto-tune] no GPU detected, using CPU profile")

    if args.batch_size is None:
        args.batch_size = prof.batch_size
    if args.grad_accum is None:
        args.grad_accum = prof.grad_accum
    if args.max_length is None:
        args.max_length = prof.max_length
    if args.lora_r is None:
        args.lora_r = prof.lora_r
    if args.lora_alpha is None:
        args.lora_alpha = prof.lora_alpha
    if args.optim is None:
        args.optim = prof.optim
    if not args.no_4bit and not args.force_4bit:
        # Profile picks 4-bit on small VRAM; respect that unless user vetoed.
        args.no_4bit = not prof.use_4bit
    if not args.gradient_checkpointing and not args.no_gradient_checkpointing:
        args.gradient_checkpointing = prof.gradient_checkpointing

    return args


def fill_static_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """When --auto-tune is NOT used, supply any missing values from a sane fallback
    so users on any tier get a working run without specifying every flag."""
    defaults = {
        "batch_size": 2,
        "grad_accum": 4,
        "max_length": 1024,
        "lora_r": 16,
        "lora_alpha": 32,
        "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    }
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


# ---------- main ----------

def main() -> int:
    args = build_arg_parser().parse_args()
    if args.auto_tune:
        args = apply_auto_tune(args)
    args = fill_static_defaults(args)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset not found: {data_path}", file=sys.stderr)
        return 2

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    grad_ckpt = bool(args.gradient_checkpointing)

    # ---- speed-defaults that don't need a GPU profile to pick ----
    if torch.cuda.is_available():
        # TF32 matmul on Ampere+ (~2x speedup with negligible accuracy loss in bf16 LoRA).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log(
        f"[finetune] model={args.model} data={args.data}\n"
        f"           epochs={args.epochs} batch={args.batch_size} grad_accum={args.grad_accum} "
        f"max_len={args.max_length} lr={args.lr}\n"
        f"           lora_r={args.lora_r} alpha={args.lora_alpha} 4bit={use_4bit} "
        f"grad_ckpt={grad_ckpt} optim={args.optim} world_size={_world_size()}\n"
        f"           attn_impl={args.attn_impl or 'auto'} packing={not args.no_packing} "
        f"group_by_length={not args.no_group_by_length} compile={args.compile} "
        f"workers={args.num_workers}"
    )

    log(f"[finetune] loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(data_path, tokenizer)

    log(f"[finetune] loading model (4bit={use_4bit})")
    model = build_model(
        args.model, use_4bit=use_4bit, gradient_checkpointing=grad_ckpt,
        attn_impl=args.attn_impl,
    )

    target_modules = args.target_modules
    if target_modules.strip().lower() == "all-linear":
        target_modules_arg = "all-linear"
    else:
        target_modules_arg = [m.strip() for m in target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_arg,
    )
    model = get_peft_model(model, lora_cfg)
    if _is_main_process():
        model.print_trainable_parameters()

    out_dir = Path(args.output)
    if _is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16_supported,
        fp16=torch.cuda.is_available() and not bf16_supported,
        tf32=bf16_supported,
        report_to=["tensorboard"],
        optim=args.optim,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        # group_by_length packs same-length sequences -> ~10-30% less padding
        # waste in instruction-tuning workloads.
        group_by_length=not args.no_group_by_length,
        torch_compile=args.compile,
        # Reduce all-reduce overhead in DDP/FSDP: use bucket coalescing.
        ddp_bucket_cap_mb=25,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        # Packing concatenates short samples to fill the context window —
        # 2-5x throughput gain on small-text datasets. Doesn't help (and can
        # hurt) when samples are already near max_length.
        packing=not args.no_packing,
    )

    log("[finetune] starting training...")
    trainer.train()

    # Only rank 0 saves to avoid race conditions on shared filesystems.
    if _is_main_process():
        log(f"[finetune] saving adapter -> {out_dir}")
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        # Stash the resolved hyperparameters so future runs are reproducible.
        (out_dir / "training_meta.json").write_text(
            json.dumps(
                {
                    "args": {k: v for k, v in vars(args).items()},
                    "world_size": _world_size(),
                    "torch_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                },
                indent=2,
            )
        )
        log("[finetune] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
