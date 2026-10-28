"""Replace frozen base Linear layers with bitsandbytes 4-bit (NF4) or 8-bit layers."""

import logging

import bitsandbytes as bnb
import torch
from torch import nn

from .models import ModelSpec

# bitsandbytes logs one warning per int8 matmul about the input cast
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


class _FlatInt8(bnb.nn.Linear8bitLt):
    """bitsandbytes int8 matmul only accepts 2-3D input; Swin feeds 5D tensors."""

    def forward(self, x):
        lead = x.shape[:-1]
        # Linear8bitLt.forward casts its bias to the input dtype in place, which flips
        # between bf16 and fp32 under autocast and breaks optimizers training the bias (BitFit).
        bias = self.bias.data if self.bias is not None else None
        out = super().forward(x.reshape(-1, x.shape[-1]))
        if bias is not None and self.bias.data.dtype != bias.dtype:
            self.bias.data = bias
        return out.reshape(*lead, -1)


def _replace(parent: nn.Module, name: str, old: nn.Linear, mode: str, compute_dtype) -> None:
    has_bias = old.bias is not None
    if mode == "nf4":
        new = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=has_bias,
                                compute_dtype=compute_dtype, quant_type="nf4")
        new.weight = bnb.nn.Params4bit(old.weight.data, requires_grad=False, quant_type="nf4")
    elif mode == "int8":
        new = _FlatInt8(old.in_features, old.out_features, bias=has_bias,
                                  has_fp16_weights=False, threshold=6.0)
        new.weight = bnb.nn.Int8Params(old.weight.data, requires_grad=False, has_fp16_weights=False)
    else:
        raise ValueError(mode)
    if has_bias:
        new.bias = nn.Parameter(old.bias.data, requires_grad=False)
    setattr(parent, name, new)


def quantize_linears(model: nn.Module, spec: ModelSpec, mode: str,
                     compute_dtype=torch.bfloat16) -> int:
    """Quantize every Linear except the classifier and `spec.quant_skip`.

    Weights are quantized when the model is moved to the GPU, so call this on
    the CPU model before `.cuda()`. Returns the number of layers replaced.
    """
    head = "inner." + spec.head
    targets = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not n.startswith(head) and not n.endswith(spec.quant_skip)
    ]
    for name, mod in targets:
        parent_name, _, child = name.rpartition(".")
        _replace(model.get_submodule(parent_name), child, mod, mode, compute_dtype)
    return len(targets)
