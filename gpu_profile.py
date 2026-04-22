#!/usr/bin/env python3
"""GPU detection + hardware-aware training profile picker.

Scales training settings across the 6GB → 96GB per-GPU range and 1 → 9 GPUs.

Used by:
  - finetune.py     (--auto-tune flag pulls profile defaults)
  - dashboard/app.py (/api/gpus endpoint and train launcher)
  - train.sh        (decides single-process vs accelerate launch)

Invocation as a CLI (for the dashboard / shell):
  python gpu_profile.py detect          # JSON: gpus + recommended profile
  python gpu_profile.py args [N]        # prints CLI args for the picked profile
                                        # (N = optional override of detected per-GPU VRAM)
  python gpu_profile.py accelerate \\
        --num-gpus 4 --strategy ddp     # prints an accelerate config (yaml)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional


# ---------- detection ----------

@dataclass
class GPUInfo:
    index: int
    name: str
    vram_gb: float
    driver: Optional[str] = None
    compute_cap: Optional[str] = None


def detect_gpus() -> list[GPUInfo]:
    """Return one GPUInfo per visible NVIDIA GPU (empty list on CPU-only)."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            vram_mb = float(parts[2])
        except ValueError:
            continue
        gpus.append(
            GPUInfo(
                index=idx,
                name=name,
                vram_gb=round(vram_mb / 1024.0, 2),
                driver=parts[3] if len(parts) > 3 else None,
                compute_cap=parts[4] if len(parts) > 4 else None,
            )
        )
    return gpus


# ---------- profiles ----------

@dataclass
class TrainingProfile:
    """One row of recommended training knobs for a per-GPU VRAM tier.

    Memory math (LoRA fine-tuning, rough):
      - 4-bit weight: ~0.5 byte/param
      - LoRA adapter: ~r * 4 bytes/param touched (small)
      - bf16 activations + grads dominate at long sequences; gradient
        checkpointing trades ~30% throughput for ~3-5x activation memory
    """
    name: str
    per_gpu_vram_min_gb: float
    batch_size: int
    grad_accum: int
    max_length: int
    lora_r: int
    lora_alpha: int
    use_4bit: bool
    gradient_checkpointing: bool
    optim: str
    max_recommended_params_b: float    # base model size that fits comfortably (in 4-bit if use_4bit)
    notes: str


# Sorted ASCENDING by per_gpu_vram_min_gb. pick_profile walks until VRAM < next tier.
PROFILES: list[TrainingProfile] = [
    TrainingProfile(
        name="xs-6gb",
        per_gpu_vram_min_gb=6.0,
        batch_size=1, grad_accum=32,
        max_length=512,
        lora_r=8, lora_alpha=16,
        use_4bit=True, gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_recommended_params_b=1.5,
        notes="6GB tier (e.g. GTX 1660, RTX 2060). Stick to <=1.5B params in 4-bit.",
    ),
    TrainingProfile(
        name="s-12gb",
        per_gpu_vram_min_gb=10.0,
        batch_size=1, grad_accum=16,
        max_length=1024,
        lora_r=16, lora_alpha=32,
        use_4bit=True, gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_recommended_params_b=3.0,
        notes="10-12GB tier (RTX 3060/4060). Up to ~3B params in 4-bit.",
    ),
    TrainingProfile(
        name="m-24gb",
        per_gpu_vram_min_gb=16.0,
        batch_size=2, grad_accum=8,
        max_length=2048,
        lora_r=32, lora_alpha=64,
        use_4bit=True, gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_recommended_params_b=8.0,
        notes="16-24GB tier (RTX 3090/4090, A4000). 7-8B params in 4-bit comfortably.",
    ),
    TrainingProfile(
        name="l-48gb",
        per_gpu_vram_min_gb=32.0,
        batch_size=4, grad_accum=4,
        max_length=4096,
        lora_r=64, lora_alpha=128,
        use_4bit=False, gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_recommended_params_b=14.0,
        notes="32-48GB tier (A6000, L40, RTX 6000 Ada). 13B in bf16, larger if 4-bit.",
    ),
    TrainingProfile(
        name="xl-80gb",
        per_gpu_vram_min_gb=64.0,
        batch_size=8, grad_accum=2,
        max_length=4096,
        lora_r=64, lora_alpha=128,
        use_4bit=False, gradient_checkpointing=False,
        optim="adamw_torch_fused",
        max_recommended_params_b=34.0,
        notes="64-80GB tier (A100 80GB). 30-34B in bf16 with LoRA.",
    ),
    TrainingProfile(
        name="xxl-96gb",
        per_gpu_vram_min_gb=88.0,
        batch_size=16, grad_accum=1,
        max_length=8192,
        lora_r=128, lora_alpha=256,
        use_4bit=False, gradient_checkpointing=False,
        optim="adamw_torch_fused",
        max_recommended_params_b=70.0,
        notes="80-96GB tier (H100, MI300). 65-70B with LoRA, longer contexts.",
    ),
]


def pick_profile(per_gpu_vram_gb: float) -> TrainingProfile:
    """Pick the highest profile whose threshold is <= the GPU's VRAM."""
    chosen = PROFILES[0]
    for p in PROFILES:
        if per_gpu_vram_gb >= p.per_gpu_vram_min_gb:
            chosen = p
        else:
            break
    return chosen


def cpu_profile() -> TrainingProfile:
    """Fallback for CPU-only systems (slow, but the trainer still runs)."""
    return TrainingProfile(
        name="cpu",
        per_gpu_vram_min_gb=0.0,
        batch_size=1, grad_accum=8,
        max_length=512,
        lora_r=8, lora_alpha=16,
        use_4bit=False,             # bnb requires CUDA
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_recommended_params_b=0.5,
        notes="CPU only — expect very slow training; stick to <=500M params.",
    )


# ---------- multi-GPU strategy ----------

VALID_STRATEGIES = ("auto", "single", "ddp", "fsdp", "zero3")


def pick_strategy(num_gpus: int, model_params_b: float, profile: TrainingProfile) -> str:
    """Default strategy selector. `auto` -> ddp if model fits per-GPU else zero3."""
    if num_gpus <= 1:
        return "single"
    fits_per_gpu = model_params_b <= profile.max_recommended_params_b
    return "ddp" if fits_per_gpu else "zero3"


def render_accelerate_config(num_gpus: int, strategy: str, mixed_precision: str = "bf16") -> dict:
    """Return an accelerate config dict (serialize with yaml or json)."""
    base = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "downcast_bf16": "no",
        "gpu_ids": "all",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": mixed_precision,
        "num_machines": 1,
        "num_processes": max(1, num_gpus),
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
    }
    if strategy == "single" or num_gpus <= 1:
        base["distributed_type"] = "NO"
        base["num_processes"] = 1
        return base
    if strategy == "ddp":
        return base
    if strategy == "fsdp":
        base["distributed_type"] = "FSDP"
        base["fsdp_config"] = {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_forward_prefetch": False,
            "fsdp_offload_params": False,
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_sync_module_states": True,
            "fsdp_use_orig_params": True,
        }
        return base
    if strategy == "zero3":
        base["distributed_type"] = "DEEPSPEED"
        base["deepspeed_config"] = {
            "deepspeed_multinode_launcher": "standard",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "offload_optimizer_device": "none",
            "offload_param_device": "none",
            "zero3_init_flag": True,
            "zero3_save_16bit_model": True,
            "zero_stage": 3,
        }
        return base
    raise ValueError(f"unknown strategy: {strategy}")


# ---------- CLI helpers ----------

HETEROGENEITY_VRAM_TOLERANCE_GB = 1.0


def is_heterogeneous(gpus: list[GPUInfo]) -> bool:
    if len(gpus) <= 1:
        return False
    vrams = [g.vram_gb for g in gpus]
    return (max(vrams) - min(vrams)) > HETEROGENEITY_VRAM_TOLERANCE_GB


def majority_subset(gpus: list[GPUInfo]) -> list[GPUInfo]:
    """Return the largest subset of GPUs whose per-GPU VRAM is within
    HETEROGENEITY_VRAM_TOLERANCE_GB of the median. Used to suggest which
    laggards to drop from a mixed rig so the profile picker doesn't clamp
    to the smallest device's tier."""
    if not gpus:
        return []
    sorted_v = sorted(g.vram_gb for g in gpus)
    median = sorted_v[len(sorted_v) // 2]
    return [g for g in gpus if abs(g.vram_gb - median) <= HETEROGENEITY_VRAM_TOLERANCE_GB]


def filter_gpus(gpus: list[GPUInfo], gpu_ids: Optional[list[int]] = None,
                exclude_smallest: bool = False) -> list[GPUInfo]:
    """Apply user selection on top of the raw nvidia-smi listing."""
    if gpu_ids is not None:
        keep = set(gpu_ids)
        gpus = [g for g in gpus if g.index in keep]
    if exclude_smallest and len(gpus) > 1:
        gpus = majority_subset(gpus) or gpus
    return gpus


def _detect_summary() -> dict:
    gpus = detect_gpus()
    if not gpus:
        prof = cpu_profile()
        return {
            "gpus": [],
            "num_gpus": 0,
            "per_gpu_vram_gb": 0,
            "total_vram_gb": 0,
            "heterogeneous": False,
            "majority_subset_indices": [],
            "profile": asdict(prof),
            "recommended_strategy": "single",
        }
    per_gpu = min(g.vram_gb for g in gpus)  # smallest = safe baseline
    total = sum(g.vram_gb for g in gpus)
    prof = pick_profile(per_gpu)
    strategy = "ddp" if len(gpus) > 1 else "single"
    hetero = is_heterogeneous(gpus)
    majority = majority_subset(gpus) if hetero else gpus
    return {
        "gpus": [asdict(g) for g in gpus],
        "num_gpus": len(gpus),
        "per_gpu_vram_gb": per_gpu,
        "total_vram_gb": round(total, 2),
        "heterogeneous": hetero,
        "majority_subset_indices": [g.index for g in majority],
        "majority_subset_vram_gb": (round(min(g.vram_gb for g in majority), 2)
                                    if majority else per_gpu),
        "profile": asdict(prof),
        "recommended_strategy": strategy,
    }


def _print_args(override_vram: Optional[float]) -> None:
    if override_vram is not None:
        prof = pick_profile(override_vram) if override_vram > 0 else cpu_profile()
    else:
        gpus = detect_gpus()
        prof = pick_profile(min(g.vram_gb for g in gpus)) if gpus else cpu_profile()
    flags = [
        f"--batch-size {prof.batch_size}",
        f"--grad-accum {prof.grad_accum}",
        f"--max-length {prof.max_length}",
        f"--lora-r {prof.lora_r}",
        f"--lora-alpha {prof.lora_alpha}",
    ]
    if not prof.use_4bit:
        flags.append("--no-4bit")
    if prof.gradient_checkpointing:
        flags.append("--gradient-checkpointing")
    flags.append(f"--optim {prof.optim}")
    print(" ".join(flags))


def _print_accelerate(num_gpus: int, strategy: str, mixed_precision: str) -> None:
    try:
        import yaml  # type: ignore
        print(yaml.safe_dump(render_accelerate_config(num_gpus, strategy, mixed_precision), sort_keys=False))
    except ImportError:
        print(json.dumps(render_accelerate_config(num_gpus, strategy, mixed_precision), indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect")
    pa = sub.add_parser("args")
    pa.add_argument("vram_gb", nargs="?", type=float, default=None)
    pc = sub.add_parser("accelerate")
    pc.add_argument("--num-gpus", type=int, required=True)
    pc.add_argument("--strategy", default="ddp", choices=VALID_STRATEGIES)
    pc.add_argument("--mixed-precision", default="bf16")
    args = p.parse_args()

    if args.cmd == "detect":
        print(json.dumps(_detect_summary(), indent=2))
    elif args.cmd == "args":
        _print_args(args.vram_gb)
    elif args.cmd == "accelerate":
        _print_accelerate(args.num_gpus, args.strategy, args.mixed_precision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
