"""Test AMP (Automatic Mixed Precision) integration with ModelBatch optimizers."""

import numpy as np
import pytest
import torch
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modelbatch import ModelBatch
from modelbatch.optimizer import (
    OptimizerFactory,
    create_adam_configs,
)
from modelbatch.utils import create_identical_models, random_init_fn

from .test_models import ImageMLP, create_dummy_data


def is_amp_supported():
    """Check if AMP is supported on the current system."""
    if not torch.cuda.is_available():
        return False

    # Check if CUDA supports FP16
    try:
        device = torch.device("cuda")
        torch.tensor([1.0], dtype=torch.float16, device=device)
    except (RuntimeError, AssertionError):
        return False
    else:
        return True


# Skip all AMP tests if not supported
pytestmark = pytest.mark.skipif(
    not is_amp_supported(),
    reason="AMP not supported (CUDA unavailable or FP16 not supported)"
)


@pytest.fixture
def setup_amp_test():
    """Setup for AMP tests."""
    device = torch.device("cuda")
    num_models = 4
    batch_size = 32

    # Create dummy data
    dataset = create_dummy_data(num_samples=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create models
    model_config = {"input_size": 784, "hidden_size": 64, "num_classes": 10}
    models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)
    model_batch = ModelBatch(models, shared_input=True)
    model_batch.to(device)

    # Create optimizer
    learning_rates = [0.001, 0.002, 0.005, 0.01]
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer_configs = create_adam_configs(learning_rates)
    optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)

    return {
        "device": device,
        "model_batch": model_batch,
        "loader": loader,
        "optimizer": optimizer,
        "models": models,
    }


@pytest.mark.parametrize(
    ("model_config", "num_steps", "optimizer_class", "optimizer_kwargs"),
    [
        # Basic AMP functionality with different model configs
        ({"input_size": 784, "hidden_size": 64, "num_classes": 10}, 1, torch.optim.Adam, {"lr": 0.001}),
        ({"input_size": 256, "hidden_size": 128, "num_classes": 5}, 1, torch.optim.Adam, {"lr": 0.001}),
        ({"input_size": 512, "hidden_size": 32, "num_classes": 15}, 3, torch.optim.Adam, {"lr": 0.001}),
        # Test different optimizers
        ({"input_size": 784, "hidden_size": 64, "num_classes": 10}, 1, torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
        ({"input_size": 784, "hidden_size": 64, "num_classes": 10}, 1, torch.optim.AdamW, {"lr": 0.001, "weight_decay": 1e-4}),
        ({"input_size": 784, "hidden_size": 64, "num_classes": 10}, 1, torch.optim.RMSprop, {"lr": 0.001}),
    ]
)
def test_amp_training_comprehensive(model_config, num_steps, optimizer_class, optimizer_kwargs):
    """Test AMP training with various model configs, optimizers, and step counts."""
    device = torch.device("cuda")
    batch_size = 32
    num_models = 3

    # Create dummy data
    dataset = create_dummy_data(
        num_samples=128,
        input_size=model_config["input_size"],
        num_classes=model_config["num_classes"]
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create models
    models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)
    model_batch = ModelBatch(models, shared_input=True).to(device)

    # Create optimizer
    optimizer_factory = OptimizerFactory(optimizer_class, optimizer_kwargs)
    configs = [optimizer_kwargs] * num_models
    optimizer = optimizer_factory.create_optimizer(model_batch, configs)

    scaler = GradScaler(device="cuda")

    # Run training steps
    initial_params = [p.clone() for p in model_batch.parameters()]
    losses = []

    for idx, (batch_data, batch_target) in enumerate(loader):
        if idx >= num_steps:
            break
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)  # noqa: PLW2901

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model_batch(batch_data)
            loss = model_batch.compute_loss(
                outputs, batch_target, F.cross_entropy
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    # Verify results
    assert all(not (np.isnan(val) or np.isinf(val)) for val in losses)

    if num_steps > 1:
        # Check parameters changed
        params_changed = any(
            not torch.allclose(initial, current)
            for initial, current in zip(initial_params, model_batch.parameters())
        )
        assert params_changed, "Parameters didn't change during training"


def test_amp_fp32_consistency():
    """Test AMP vs FP32 training produce similar results."""
    device = torch.device("cuda")
    batch_size = 32
    model_config = {"input_size": 784, "hidden_size": 64, "num_classes": 10}
    num_models = 2

    # Create identical model batches
    models_amp = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)
    models_fp32 = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)

    # Copy weights to ensure identical start
    for m_amp, m_fp32 in zip(models_amp, models_fp32):
        m_fp32.load_state_dict(m_amp.state_dict())

    model_batch_amp = ModelBatch(models_amp, shared_input=True).to(device)
    model_batch_fp32 = ModelBatch(models_fp32, shared_input=True).to(device)

    # Create optimizers
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    configs = create_adam_configs([0.001, 0.001])
    optimizer_amp = optimizer_factory.create_optimizer(model_batch_amp, configs)
    optimizer_fp32 = optimizer_factory.create_optimizer(model_batch_fp32, configs)

    scaler = GradScaler(device="cuda")

    # Create dummy data
    dataset = create_dummy_data(num_samples=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Train for a few steps
    for _ in range(3):
        optimizer_amp.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs_amp = model_batch_amp(data)
            loss_amp = model_batch_amp.compute_loss(
                outputs_amp, target, F.cross_entropy
            )
        scaler.scale(loss_amp).backward()
        scaler.step(optimizer_amp)
        scaler.update()

        # FP32 step
        optimizer_fp32.zero_grad()
        outputs_fp32 = model_batch_fp32(data)
        loss_fp32 = model_batch_fp32.compute_loss(outputs_fp32, target, F.cross_entropy)
        loss_fp32.backward()
        optimizer_fp32.step()

    # Losses should be similar (within tolerance for FP16 precision)
    assert abs(loss_amp.item() - loss_fp32.item()) < 1.0


@pytest.mark.parametrize(("num_models", "input_size", "scaling_factors"), [
    (2, 784, [1.0, 2.0]),
    (3, 512, [0.5, 1.0, 2.0]),
    (4, 1024, [0.1, 1.0, 10.0, 5.0]),
])
def test_batched_vs_individual_consistency(num_models, input_size, scaling_factors):
    """Test batched AMP matches individual training for same and different scaling."""
    device = torch.device("cuda")
    batch_size = 32
    num_classes = 10

    # Create dummy data
    dataset = create_dummy_data(num_samples=128, input_size=input_size, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Create models
    model_config = {"input_size": input_size, "hidden_size": 64, "num_classes": num_classes}
    models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)

    # Apply scaling factors
    for model, scale in zip(models, scaling_factors):
        for param in model.parameters():
            param.data.mul_(scale)

    # Individual training setup
    individual_models = [ImageMLP(**model_config).to(device) for _ in range(num_models)]
    for individual_model, base_model in zip(individual_models, models):
        individual_model.load_state_dict(base_model.state_dict())

    individual_optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.001)
        for model in individual_models
    ]

    # Batched training setup
    model_batch = ModelBatch(models, shared_input=True).to(device)
    optimizer_factory = OptimizerFactory(torch.optim.Adam, {"lr": 0.001})
    batched_optimizer = optimizer_factory.create_optimizer(
        model_batch, [{"lr": 0.001}] * num_models
    )

    # Shared scaler for consistency
    scaler = GradScaler(device="cuda")

    # Batched training
    batched_optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        outputs_batched = model_batch(data)
        loss_batched = model_batch.compute_loss(
            outputs_batched, target, F.cross_entropy
        )
    scaler.scale(loss_batched).backward()
    scaler.step(batched_optimizer)
    scaler.update()

    # Individual training
    individual_losses = []
    for model, optimizer in zip(individual_models, individual_optimizers):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
        individual_losses.append(loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
    scaler.update()

    # Check consistency
    individual_loss_sum = sum(individual_losses)
    assert torch.allclose(individual_loss_sum, loss_batched, rtol=1e-3, atol=1e-3)

    # Check parameter consistency
    final_params_individual = [[p.clone() for p in model.parameters()] for model in individual_models]
    final_params_batched = [[p.clone() for p in model.parameters()] for model in model_batch.models]

    for individual_params, batched_params in zip(final_params_individual, final_params_batched):
        for ind_param, batch_param in zip(individual_params, batched_params):
            assert torch.allclose(ind_param, batch_param, rtol=1e-2, atol=1e-2)


@pytest.mark.skip("Currently skipping support for consistent single/batched AMP overflow handling")
def test_amp_overflow_handling(input_size):  # noqa: PLR0915
    """Test AMP handles gradient overflow correctly with both individual and batched training."""
    device = torch.device("cuda")
    batch_size = 32
    num_models = 4
    num_classes = 10

    # Create dummy data
    dataset = create_dummy_data(num_samples=128, input_size=input_size, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Create model configs
    model_config = {"input_size": input_size, "hidden_size": 64, "num_classes": num_classes}

    # Create base models
    base_models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)

    # Apply scaling factors to create overflow scenarios
    scaling_factors = [1.0, 50.0, 200.0, 1000.0]
    for model, scale in zip(base_models, scaling_factors):
        for param in model.parameters():
            param.data.mul_(scale)

    # Individual training
    individual_models = [ImageMLP(**model_config).to(device) for _ in range(num_models)]
    for individual_model, base_model in zip(individual_models, base_models):
        individual_model.load_state_dict(base_model.state_dict())

    individual_optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.001)
        for model in individual_models
    ]
    individual_scalers = [GradScaler(device="cuda") for _ in range(num_models)]

    individual_losses = []
    for model, optimizer, scaler in zip(individual_models, individual_optimizers, individual_scalers):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)

        individual_losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Batched training
    batched_models = [ImageMLP(**model_config).to(device) for _ in range(num_models)]
    for batched_model, base_model in zip(batched_models, base_models):
        batched_model.load_state_dict(base_model.state_dict())

    model_batch = ModelBatch(batched_models, shared_input=True).to(device)
    optimizer_factory = OptimizerFactory(torch.optim.Adam, {"lr": 0.001})
    batched_optimizer = optimizer_factory.create_optimizer(
        model_batch, [{"lr": 0.001}] * num_models
    )
    batched_scaler = GradScaler(device="cuda")

    # Run batched training step (we only need it for scaler updates, not the loss value)
    batched_optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        outputs = model_batch(data)
        loss_batched = model_batch.compute_loss(outputs, target, F.cross_entropy)
    batched_scaler.scale(loss_batched).backward()
    batched_scaler.step(batched_optimizer)
    batched_scaler.update()

    # Check overflow behavior consistency per model
    # Each model should have consistent NaN behavior between individual and batched training
    batched_outputs = model_batch(data)
    batched_individual_losses = []
    for i in range(num_models):
        with torch.amp.autocast("cuda"):
            loss = F.cross_entropy(batched_outputs[i], target)
        batched_individual_losses.append(loss.item())

    # Check that each model's NaN behavior matches between individual and batched training
    for i, (individual_loss, batched_individual_loss) in enumerate(zip(individual_losses, batched_individual_losses)):
        individual_has_nan = np.isnan(individual_loss) or np.isinf(individual_loss)
        batched_has_nan = np.isnan(batched_individual_loss) or np.isinf(batched_individual_loss)

        assert individual_has_nan == batched_has_nan, (
            f"Model {i}: Individual training has NaN={individual_has_nan}, "
            f"but batched training has NaN={batched_has_nan}. "
        )

    # Verify scaler is still valid
    assert batched_scaler.get_scale() > 0

    # Check that models without NaN have similar losses
    for i, (individual_loss, batched_individual_loss) in enumerate(zip(individual_losses, batched_individual_losses)):
        individual_has_nan = np.isnan(individual_loss) or np.isinf(individual_loss)
        batched_has_nan = np.isnan(batched_individual_loss) or np.isinf(batched_individual_loss)

        if not individual_has_nan and not batched_has_nan:
            relative_error = abs(individual_loss - batched_individual_loss) / max(abs(individual_loss), 1e-6)
            assert relative_error < 0.2, (
                f"Model {i}: Loss mismatch between individual ({individual_loss}) "
                f"and batched ({batched_individual_loss}) training"
            )
