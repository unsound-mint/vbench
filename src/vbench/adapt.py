"""Fine-tuning method setup: Full FT, LoRA, BitFit; plus activation checkpointing."""

from functools import partial

from peft import AdaLoraConfig, LoraConfig, get_peft_model
from torch import nn
from torch.utils.checkpoint import checkpoint

from .models import ModelSpec


def enable_checkpointing(model: nn.Module, block_class: str) -> int:
    """Recompute each Transformer block's activations in the backward pass."""
    n = 0
    for module in model.modules():
        if type(module).__name__ == block_class:
            module.forward = partial(checkpoint, module.forward, use_reentrant=False)
            n += 1
    return n


def _head_module(model: nn.Module, spec: ModelSpec) -> nn.Module:
    return model.get_submodule("inner." + spec.head)


def apply_method(model: nn.Module, spec: ModelSpec, method: str,
                 total_steps: int = 100) -> nn.Module:
    """Freeze/adapt `model` for `method`; the classification head always trains.

    `method` is full, lora, qlora (LoRA over an NF4 base; the caller quantizes),
    adalora, or bitfit. `total_steps` only sets AdaLoRA's rank-pruning schedule.
    """
    if method == "full":
        return model
    head = _head_module(model, spec)
    if method == "adalora":
        cfg = AdaLoraConfig(
            init_r=12, target_r=8, lora_alpha=8, target_modules=list(spec.lora_targets),
            modules_to_save=["inner." + spec.head],
            total_step=total_steps, tinit=max(1, total_steps // 10),
            tfinal=max(1, total_steps // 5), deltaT=max(1, total_steps // 20),
        )
        return get_peft_model(model, cfg)
    if method in ("lora", "qlora"):
        cfg = LoraConfig(
            r=8, lora_alpha=8, target_modules=list(spec.lora_targets),
            modules_to_save=["inner." + spec.head],
        )
        # peft wraps the whole canonical module; the head is saved via modules_to_save
        return get_peft_model(model, cfg)
    if method == "bitfit":
        for name, p in model.named_parameters():
            p.requires_grad = name.endswith(".bias")
        for p in head.parameters():
            p.requires_grad = True
        return model
    raise ValueError(f"unknown method {method!r}")


def trainable_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
