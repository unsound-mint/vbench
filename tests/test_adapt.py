"""Every method must be wired into the forward pass: all trainable params get grads."""

import pytest
import torch

from vbench.adapt import apply_method, trainable_params_m
from vbench.models import MODELS
from vbench.training import assert_all_trainable_have_grad


@pytest.mark.parametrize("method", ["lora", "adalora", "bitfit"])
@pytest.mark.parametrize("name", list(MODELS))
def test_trainable_params_receive_grads(name, method):
    spec = MODELS[name]
    model = apply_method(spec.build(False), spec, method).train()
    frames = 16 if name != "timesformer-b" else 8
    out = model(torch.randn(1, frames, 3, 224, 224))
    out.sum().backward()
    assert_all_trainable_have_grad(model)
    full = spec.build(False)
    assert trainable_params_m(model) < 0.02 * trainable_params_m(full)


@pytest.mark.parametrize("method", ["lora", "adalora"])
@pytest.mark.parametrize("name", list(MODELS))
def test_adapter_is_identity_at_init(name, method):
    """Adapters start at zero, so the adapted model must equal the base model.
    This also proves the adapter sits on a layer the forward pass really uses."""
    spec = MODELS[name]
    torch.manual_seed(0)
    base = spec.build(False).eval()
    x = torch.randn(1, 16 if name != "timesformer-b" else 8, 3, 224, 224)
    with torch.no_grad():
        ref = base(x)
        adapted = apply_method(base, spec, method, total_steps=50).eval()
        out = adapted(x)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()
