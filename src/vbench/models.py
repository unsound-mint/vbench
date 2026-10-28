"""Registry of open-weight video Transformer candidates.

Every model is exposed through the same call, `model(x)` with
x of shape (B, T, C, H, W) returning logits, so measurement code never needs
to know the native layout. `pretrained=False` skips weight download for
memory/feasibility probes, which depend only on architecture.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

NUM_CLASSES = 25  # ERA; other datasets pass their own to ModelSpec.build


class _Canonical(nn.Module):
    """Adapts a model to (B, T, C, H, W) in and plain logits out."""

    def __init__(self, inner: nn.Module, channels_first: bool, hf: bool):
        super().__init__()
        self.inner = inner
        self.channels_first = channels_first
        self.hf = hf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_first:
            x = x.permute(0, 2, 1, 3, 4)
        out = self.inner(x)
        return out.logits if self.hf else out


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[..., nn.Module]  # (pretrained, num_classes=25) -> canonical-input module
    block_class: str  # class name of the repeated Transformer block
    lora_targets: tuple[str, ...]  # trailing module names LoRA adapts
    head: str  # dotted path of the classifier module, relative to `.inner`
    native_frames: int
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    quant_skip: tuple[str, ...] = ()  # Linear name suffixes that must stay unquantized
    weight_files: Callable[[], list[str]] = lambda: []  # pretrained checkpoint paths (for hashing)
    note: str = ""


def _videomae(pretrained: bool, num_classes: int = NUM_CLASSES) -> nn.Module:
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    if pretrained:
        m = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes, ignore_mismatched_sizes=True,
        )
    else:  # default config is the base size; no network needed
        m = VideoMAEForVideoClassification(VideoMAEConfig(num_labels=num_classes))
    return _Canonical(m, channels_first=False, hf=True)


def _timesformer(pretrained: bool, num_classes: int = NUM_CLASSES) -> nn.Module:
    from transformers import TimesformerConfig, TimesformerForVideoClassification

    if pretrained:
        m = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=num_classes, ignore_mismatched_sizes=True,
        )
    else:  # default config is the base size; no network needed
        m = TimesformerForVideoClassification(TimesformerConfig(num_labels=num_classes))
    return _Canonical(m, channels_first=False, hf=True)


def _effective_weight(linear: nn.Module) -> torch.Tensor:
    """Weight of a Linear, including any peft LoRA delta."""
    if hasattr(linear, "get_delta_weight"):
        delta = sum(linear.get_delta_weight(a) for a in linear.active_adapters)
        return linear.base_layer.weight + delta
    return linear.weight


def _patch_swin3d_lora_aware() -> None:
    """torchvision's Swin3D reads `qkv.weight` and calls F.linear itself, so a
    peft LoRA wrapper on `qkv` would never run and LoRA would silently train
    only the head. Build the weight through the wrapper instead."""
    from torchvision.models.video import swin_transformer as st

    attn = st.ShiftedWindowAttention3d
    if getattr(attn, "_vbench_patched", False):
        return

    def forward(self, x):
        _, t, h, w, _ = x.shape
        window_size, shift_size = st._get_window_and_shift_size(
            self.shift_size.copy(), [t, h, w], self.window_size.copy()
        )
        return st.shifted_window_attention_3d(
            x, _effective_weight(self.qkv), self.proj.weight,
            self.get_relative_position_bias(window_size), window_size, self.num_heads,
            shift_size=shift_size, attention_dropout=self.attention_dropout,
            dropout=self.dropout, qkv_bias=self.qkv.bias, proj_bias=self.proj.bias,
            training=self.training,
        )

    attn.forward = forward
    attn._vbench_patched = True


def _swin3d_b(pretrained: bool, num_classes: int = NUM_CLASSES) -> nn.Module:
    from torchvision.models.video import Swin3D_B_Weights, swin3d_b

    _patch_swin3d_lora_aware()
    m = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1 if pretrained else None)
    m.head = nn.Linear(m.head.in_features, num_classes)
    return _Canonical(m, channels_first=True, hf=False)


def _hiera_b(pretrained: bool, num_classes: int = NUM_CLASSES) -> nn.Module:
    import hiera

    m = hiera.hiera_base_16x224(
        pretrained=pretrained, checkpoint="mae_k400_ft_k400" if pretrained else None
    )
    m.head.projection = nn.Linear(m.head.projection.in_features, num_classes)
    return _Canonical(m, channels_first=True, hf=False)


_HALF = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)


def _hf_file(repo: str, filename: str = "model.safetensors") -> Callable[[], list[str]]:
    def files() -> list[str]:
        from huggingface_hub import hf_hub_download

        return [hf_hub_download(repo, filename)]

    return files


def _hub_file(url: str) -> Callable[[], list[str]]:
    def files() -> list[str]:
        import os

        return [os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(url))]

    return files


MODELS: dict[str, ModelSpec] = {
    s.name: s
    for s in [
        ModelSpec(
            "videomae-b", _videomae, "VideoMAELayer", ("query", "value"), "classifier", 16,
            weight_files=_hf_file("MCG-NJU/videomae-base-finetuned-kinetics"),
        ),
        ModelSpec(
            "timesformer-b", _timesformer, "TimesformerLayer", ("qkv",), "classifier", 8,
            mean=_HALF[0], std=_HALF[1],
            weight_files=_hf_file("facebook/timesformer-base-finetuned-k400", "pytorch_model.bin"),
            note="native 8 frames; fed 16 (time embeddings interpolate)",
        ),
        ModelSpec(
            "videoswin-b", _swin3d_b, "SwinTransformerBlock", ("qkv",), "head", 16,
            # torchvision's attention reads qkv/proj weights directly, so they cannot be quantized
            quant_skip=("attn.qkv", "attn.proj"),
            weight_files=_hub_file("https://download.pytorch.org/models/swin3d_b_1k-24f7c7c6.pth"),
            note="fused qkv: LoRA adapts q, k and v together",
        ),
        ModelSpec(
            "hiera-b", _hiera_b, "HieraBlock", ("qkv",), "head.projection", 16,
            mean=_HALF[0], std=_HALF[1],  # assumed: PySlowFast Kinetics default, not stated by the package
            weight_files=_hub_file("https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth"),
            note="fused qkv: LoRA adapts q, k and v together",
        ),
    ]
}
