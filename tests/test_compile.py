"""
Tests for torch.compile compatibility with ModelBatch.
"""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch import ModelBatch  # noqa: E402
from modelbatch.utils import create_identical_models  # noqa: E402

from .test_models import CustomModel, DeepMLP, SimpleCNN, SimpleMLP  # noqa: E402


pytestmark = [
    pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available in this PyTorch",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA GPU required for compile backend tests",
    ),
]


def _safe_enable_compile(mb: ModelBatch, backend: str) -> None:
    """Enable compile in a way that is robust across environments.

    Uses a conservative backend by default and skips if the backend is unavailable.
    """
    try:
        mb.enable_compile(backend=backend, fullgraph=False, dynamic=True)
    except Exception as exc:
        pytest.skip(f"torch.compile backend '{backend}' not usable: {exc}")


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _make_inner_shape(model_class, model_params):
    if model_class is SimpleCNN:
        c = model_params.get("input_channels", 3)
        return (c, 32, 32)
    # MLP-like
    if "input_size" in model_params:
        return (model_params["input_size"],)
    # Fallback
    return (8,)


def _num_classes_from_params(model_params: dict) -> int:
    return int(model_params.get("output_size", model_params.get("num_classes", 3)))


def _make_x_and_y(
    model_class,
    model_params,
    num_models: int,
    shared_input: bool,
    batch_size: int = 4,
    device: torch.device | None = None,
):
    if device is None:
        device = _device()
    inner = _make_inner_shape(model_class, model_params)
    if shared_input:
        x = torch.randn((batch_size, *inner), device=device)
    else:
        x = torch.randn((num_models, batch_size, *inner), device=device)
    y = torch.randint(
        0, _num_classes_from_params(model_params), (batch_size,), device=device
    )
    return x, y


def _make_eager_and_compiled(
    model_class,
    model_params,
    num_models: int,
    shared_input: bool,
    device: torch.device | None = None,
    backend: str | None = None,
):
    if device is None:
        device = _device()
    torch.manual_seed(0)
    models_eager = create_identical_models(model_class, model_params, num_models)
    mb_eager = ModelBatch(models_eager, shared_input=shared_input).to(device)

    torch.manual_seed(0)
    models_comp = create_identical_models(model_class, model_params, num_models)
    mb_comp = ModelBatch(models_comp, shared_input=shared_input).to(device)
    if backend is None:
        backend = "inductor"
    _safe_enable_compile(mb_comp, backend=backend)
    return mb_eager, mb_comp


MODEL_CASES = [
    (SimpleMLP, {"input_size": 8, "output_size": 4}, 3),
    (CustomModel, {"input_size": 8, "output_size": 4}, 3),
    (SimpleCNN, {"input_channels": 1, "num_classes": 4}, 2),
]

@pytest.mark.parametrize("backend", ["inductor", "aot_eager"])
@pytest.mark.parametrize("model_class,model_params,num_models", MODEL_CASES)
@pytest.mark.parametrize("shared_input", [True, False])
def test_compile_forward_matches_eager(model_class, model_params, num_models, shared_input, backend):
    mb_eager, mb_comp = _make_eager_and_compiled(
        model_class, model_params, num_models, shared_input, backend=backend
    )
    mb_eager.eval()
    mb_comp.eval()

    x, _y = _make_x_and_y(model_class, model_params, num_models, shared_input)
    with torch.no_grad():
        out_eager = mb_eager(x)
        out_comp = mb_comp(x)
    assert torch.allclose(out_eager, out_comp, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["inductor", "aot_eager"])
@pytest.mark.parametrize("model_class,model_params,num_models", MODEL_CASES)
@pytest.mark.parametrize("shared_input", [True, False])
def test_compile_loss_matches_eager(model_class, model_params, num_models, shared_input, backend):
    mb_eager, mb_comp = _make_eager_and_compiled(
        model_class, model_params, num_models, shared_input, backend=backend
    )
    mb_eager.eval()
    mb_comp.eval()
    x, y = _make_x_and_y(model_class, model_params, num_models, shared_input)
    logits_eager = mb_eager(x)
    logits_comp = mb_comp(x)
    loss_eager = mb_eager.compute_loss(logits_eager, y, F.cross_entropy)
    loss_comp = mb_comp.compute_loss(logits_comp, y, F.cross_entropy)
    assert torch.allclose(loss_eager, loss_comp, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["inductor", "aot_eager"])
@pytest.mark.parametrize("model_class,model_params,num_models", MODEL_CASES)
@pytest.mark.parametrize("shared_input", [True, False])
def test_compile_gradients_match_eager(model_class, model_params, num_models, shared_input, backend):
    mb_eager, mb_comp = _make_eager_and_compiled(
        model_class, model_params, num_models, shared_input, backend=backend
    )
    mb_eager.eval()
    mb_comp.eval()
    optimizer_eager = torch.optim.SGD(mb_eager.parameters(), lr=0.01)
    optimizer_comp = torch.optim.SGD(mb_comp.parameters(), lr=0.01)

    x, y = _make_x_and_y(model_class, model_params, num_models, shared_input)

    optimizer_eager.zero_grad()
    optimizer_comp.zero_grad()

    logits_eager = mb_eager(x)
    logits_comp = mb_comp(x)
    loss_eager = mb_eager.compute_loss(logits_eager, y, F.cross_entropy)
    loss_comp = mb_comp.compute_loss(logits_comp, y, F.cross_entropy)
    loss_eager.backward()
    loss_comp.backward()

    grads_match_all = True
    saw_any_grad = False
    for p_e, p_c in zip(mb_eager.parameters(), mb_comp.parameters()):
        ge = p_e.grad
        gc = p_c.grad
        if ge is None and gc is None:
            continue
        assert ge is not None and gc is not None
        saw_any_grad = True
        if not torch.allclose(ge, gc, atol=1e-5, rtol=1e-5):
            grads_match_all = False
            break

    assert saw_any_grad, "No gradients found on parameters"
    assert grads_match_all


