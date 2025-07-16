"""
Test AMP (Automatic Mixed Precision) integration with ModelBatch optimizers.
"""

import pytest
import torch
import numpy as np
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modelbatch import ModelBatch
from modelbatch.optimizer import (
    OptimizerFactory,
    create_adam_configs,
    train_step_with_amp,
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
        # Test if we can create a half tensor
        torch.tensor([1.0], dtype=torch.float16, device=device)
        return True
    except (RuntimeError, AssertionError):
        return False


# Skip all AMP tests if not supported
pytestmark = pytest.mark.skipif(
    not is_amp_supported(),
    reason="AMP not supported (CUDA unavailable or FP16 not supported)"
)


@pytest.fixture
def setup_amp_test():
    """Setup for AMP tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for AMP testing")

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


@pytest.mark.parametrize("model_config", [
    {"input_size": 256, "hidden_size": 64, "num_classes": 15},
    {"input_size": 784, "hidden_size": 64, "num_classes": 10},
    {"input_size": 512, "hidden_size": 128, "num_classes": 5},
    {"input_size": 1024, "hidden_size": 32, "num_classes": 20},
])
def test_amp_training_step_with_configs(setup_amp_test, model_config):
    """Test AMP training step with different model configurations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda")
    batch_size = 32
    num_models = 4

    # Create dummy data with matching input size and num_classes
    input_size = model_config["input_size"]
    num_classes = model_config["num_classes"]
    dataset = create_dummy_data(num_samples=128, input_size=input_size, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Create models with specified configuration
    models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)
    model_batch = ModelBatch(models, shared_input=True).to(device)

    # Create optimizer
    learning_rates = [0.001, 0.002, 0.005, 0.01]
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer_configs = create_adam_configs(learning_rates)
    optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)

    # Create scaler
    scaler = GradScaler(device="cuda")

    # Test AMP training step
    loss = train_step_with_amp(
        model_batch, data, target, F.cross_entropy, optimizer, scaler
    )

    # Check that loss is computed
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Check that gradients are computed
    has_gradients = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model_batch.parameters()
    )
    assert has_gradients, "No gradients found after AMP step"


@pytest.mark.parametrize("num_steps", [3, 5, 10])
def test_amp_multiple_steps(setup_amp_test, num_steps):
    """Test multiple AMP training steps."""
    setup = setup_amp_test
    device = setup["device"]
    model_batch = setup["model_batch"]
    optimizer = setup["optimizer"]
    loader = setup["loader"]

    scaler = GradScaler(device="cuda")

    initial_params = [p.clone() for p in model_batch.parameters()]
    losses = []

    # Run training steps
    for idx, (data, target) in enumerate(loader):
        if idx >= num_steps:
            break
        data, target = data.to(device), target.to(device)

        loss = train_step_with_amp(
            model_batch, data, target, F.cross_entropy, optimizer, scaler
        )
        losses.append(loss.item())

    # Check that parameters have changed
    params_changed = any(
        not torch.allclose(initial, current)
        for initial, current in zip(initial_params, model_batch.parameters())
    )
    assert params_changed, "Parameters didn't change during training"

    # Check that we have expected number of loss values
    assert len(losses) == min(num_steps, len(loader)), f"Expected {min(num_steps, len(loader))} training steps, got {len(losses)}"
    assert all(
        not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l)))
        for l in losses
    )


def test_amp_vs_fp32_consistency(setup_amp_test):
    """Test that AMP and FP32 training produce similar results."""
    setup = setup_amp_test
    device = setup["device"]
    model_config = {"input_size": 784, "hidden_size": 64, "num_classes": 10}

    # Create two identical model batches
    models_amp = create_identical_models(ImageMLP, model_config, 2, random_init_fn)
    models_fp32 = create_identical_models(ImageMLP, model_config, 2, random_init_fn)

    # Copy weights to ensure they start identical
    for m_amp, m_fp32 in zip(models_amp, models_fp32):
        m_fp32.load_state_dict(m_amp.state_dict())

    model_batch_amp = ModelBatch(models_amp, shared_input=True).to(device)
    model_batch_fp32 = ModelBatch(models_fp32, shared_input=True).to(device)

    # Create optimizers
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    configs = create_adam_configs([0.001, 0.001])  # Same LR for both models

    optimizer_amp = optimizer_factory.create_optimizer(model_batch_amp, configs)
    optimizer_fp32 = optimizer_factory.create_optimizer(model_batch_fp32, configs)

    scaler = GradScaler(device="cuda")

    # Train for a few steps
    data, target = next(iter(setup["loader"]))
    data, target = data.to(device), target.to(device)

    for _ in range(3):
        # AMP step
        loss_amp = train_step_with_amp(
            model_batch_amp, data, target, F.cross_entropy, optimizer_amp, scaler
        )

        # FP32 step
        optimizer_fp32.zero_grad()
        outputs_fp32 = model_batch_fp32(data)
        loss_fp32 = model_batch_fp32.compute_loss(outputs_fp32, target, F.cross_entropy)
        loss_fp32.backward()
        optimizer_fp32.step()

    # Losses should be similar (within reasonable tolerance due to FP16 precision)
    assert abs(loss_amp.item() - loss_fp32.item()) < 1.0, (
        f"AMP and FP32 losses differ too much: {loss_amp.item():.4f} vs {loss_fp32.item():.4f}"
    )


@pytest.mark.parametrize("optimizer_class,optimizer_config", [
    (torch.optim.Adam, {"lr": 0.001}),
    (torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
    (torch.optim.AdamW, {"lr": 0.001, "weight_decay": 1e-4}),
    (torch.optim.RMSprop, {"lr": 0.001}),
])
def test_amp_with_different_optimizers(setup_amp_test, optimizer_class, optimizer_config):
    """Test AMP with different optimizer types."""
    setup = setup_amp_test
    device = setup["device"]

    # Create fresh models for each optimizer test
    model_config = {"input_size": 784, "hidden_size": 64, "num_classes": 10}
    models = create_identical_models(ImageMLP, model_config, 2, random_init_fn)
    mb = ModelBatch(models, shared_input=True).to(device)

    optimizer_factory = OptimizerFactory(optimizer_class, optimizer_config)
    configs = [optimizer_config, optimizer_config]
    optimizer = optimizer_factory.create_optimizer(mb, configs)

    scaler = GradScaler(device="cuda")

    # Test a single step
    data, target = next(iter(setup["loader"]))
    data, target = data.to(device), target.to(device)

    try:
        loss = train_step_with_amp(
            mb, data, target, F.cross_entropy, optimizer, scaler
        )
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
    except Exception as e:
        pytest.fail(f"AMP failed with {optimizer_class.__name__}: {e}")


def test_individual_vs_batched_amp_same_scaling():
    """Test that individual AMP training matches batched AMP training with same scaling."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 32
    num_models = 3

    # Create dummy data
    dataset = create_dummy_data(num_samples=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Create identical models
    model_config = {"input_size": 784, "hidden_size": 64, "num_classes": 10}
    models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)

    # Individual training setup
    individual_models = [ImageMLP(**model_config).to(device) for _ in range(num_models)]
    for i, model in enumerate(individual_models):
        model.load_state_dict(models[i].state_dict())

    # Batched training setup
    model_batch = ModelBatch(models, shared_input=True).to(device)

    # Optimizers
    optimizer_factory = OptimizerFactory(torch.optim.Adam, {"lr": 0.001})
    batched_optimizer = optimizer_factory.create_optimizer(
        model_batch, [{"lr": 0.001}] * num_models
    )

    individual_optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.001)
        for model in individual_models
    ]

    # Shared scaler (same scaling for all)
    scaler = GradScaler(device="cuda")

    # Training step
    # Batched training
    loss_batched = train_step_with_amp(
        model_batch, data, target, F.cross_entropy, batched_optimizer, scaler
    )

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

    # Check losses match - compare aggregated loss
    individual_loss_sum = sum(individual_losses)
    assert torch.allclose(
        individual_loss_sum,
        loss_batched,
        rtol=1e-3,
        atol=1e-3
    ), f"Loss mismatch: {individual_loss_sum.item()} vs {loss_batched.item()}"


def test_individual_vs_batched_amp_different_scaling():
    """Test that batched AMP handles different scaling requirements correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 32
    num_models = 2

    # Create dummy data
    dataset = create_dummy_data(num_samples=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Create identical models
    model_config = {"input_size": 784, "hidden_size": 64, "num_classes": 10}
    models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)

    # Individual training setup with identical scalers (same scaling)
    individual_models = [ImageMLP(**model_config).to(device) for _ in range(num_models)]
    for idx, model in enumerate(individual_models):
        model.load_state_dict(models[idx].state_dict())

    # Optimizers
    individual_optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.001)
        for model in individual_models
    ]

    # Same scaler for individual training to ensure consistency
    individual_scaler = GradScaler(device="cuda")

    # Individual training
    final_params_individual = []
    for model, optimizer in zip(individual_models, individual_optimizers):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
        individual_scaler.scale(loss).backward()
        individual_scaler.step(optimizer)
        individual_scaler.update()
        final_params_individual.append([p.clone() for p in model.parameters()])

    # Reset models for batched training with identical initial state
    models_reset = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)
    for idx, (model, original_model) in enumerate(zip(models_reset, models)):
        model.load_state_dict(original_model.state_dict())

    model_batch = ModelBatch(models_reset, shared_input=True).to(device)

    optimizer_factory = OptimizerFactory(torch.optim.Adam, {"lr": 0.001})
    batched_optimizer = optimizer_factory.create_optimizer(
        model_batch, [{"lr": 0.001}] * num_models
    )

    # Batched training
    batched_scaler = GradScaler(device="cuda")
    train_step_with_amp(
        model_batch, data, target, F.cross_entropy, batched_optimizer, batched_scaler
    )

    # Check parameter consistency - allow for numerical differences due to batching
    final_params_batched = [[p.clone() for p in model.parameters()] for model in model_batch.models]

    for individual_params, batched_params in zip(
        final_params_individual, final_params_batched
    ):
        for ind_param, batch_param in zip(individual_params, batched_params):
            # Allow for reasonable numerical tolerance
            assert torch.allclose(
                ind_param, batch_param, rtol=1e-2, atol=1e-2
            ), "Parameter mismatch between individual and batched training"


@pytest.mark.parametrize("num_models,scaling_factors,input_size", [
    (2, [1.0, 2.0], 784),
    (3, [0.5, 1.0, 2.0], 512),
    (4, [0.1, 1.0, 10.0, 5.0], 1024),
])
def test_batched_amp_different_loss_scales(num_models, scaling_factors, input_size):
    """Test batched AMP with models that have different loss scales."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 32
    num_classes = 10

    # Create dummy data with matching input size
    dataset = create_dummy_data(num_samples=128, input_size=input_size, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Create models with specified input size
    model_config = {"input_size": input_size, "hidden_size": 64, "num_classes": num_classes}
    models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)

    # Apply different scaling to initial weights
    for idx, (model, scale) in enumerate(zip(models, scaling_factors)):
        for param in model.parameters():
            param.data.mul_(scale)

    # Test batched training
    model_batch = ModelBatch(models, shared_input=True).to(device)

    optimizer_factory = OptimizerFactory(torch.optim.Adam, {"lr": 0.001})
    optimizer = optimizer_factory.create_optimizer(
        model_batch, [{"lr": 0.001}] * num_models
    )

    scaler = GradScaler(device="cuda")

    # Run training step
    loss = train_step_with_amp(
        model_batch, data, target, F.cross_entropy, optimizer, scaler
    )

    # Verify loss is valid
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.dim() == 0  # Scalar


@pytest.mark.parametrize("input_size", [784, 512, 1024])
def test_amp_scaling_with_gradient_overflow(input_size):
    """Test AMP scaling behavior when gradients overflow."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 32
    num_models = 2
    num_classes = 10

    # Create dummy data with matching input size
    dataset = create_dummy_data(num_samples=128, input_size=input_size, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Create models with specified input size
    model_config = {"input_size": input_size, "hidden_size": 64, "num_classes": num_classes}
    models = create_identical_models(ImageMLP, model_config, num_models, random_init_fn)

    # Make one model prone to overflow with moderate scaling
    for param in models[0].parameters():
        param.data.mul_(50.0)  # Moderate scaling to avoid complete NaN

    model_batch = ModelBatch(models, shared_input=True).to(device)

    optimizer_factory = OptimizerFactory(torch.optim.Adam, {"lr": 0.001})
    optimizer = optimizer_factory.create_optimizer(
        model_batch, [{"lr": 0.001}] * num_models
    )

    scaler = GradScaler(device="cuda")

    # Run multiple steps to test scaling behavior
    valid_steps = 0
    for _ in range(3):
        loss = train_step_with_amp(
            model_batch, data, target, F.cross_entropy, optimizer, scaler
        )

        # Verify loss is valid
        if not torch.isnan(loss) and not torch.isinf(loss):
            valid_steps += 1

    # Verify scaling factor has been adjusted appropriately
    assert scaler.get_scale() > 0, "Scaler became invalid"
    assert valid_steps > 0, "No valid training steps completed"


@pytest.mark.parametrize("input_size", [256, 512, 784, 1024])
def test_mixed_overflow_handling(input_size):
    """Test that ModelBatch handles actual overflow scenarios correctly with relative tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 32
    num_models = 4  # Normal, overflow-prone, very overflow-prone, extreme
    num_classes = 10

    # Create dummy data with matching dimensions
    dataset = create_dummy_data(num_samples=128, input_size=input_size, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    # Create model configs
    model_config = {"input_size": input_size, "hidden_size": 64, "num_classes": num_classes}
    
    # Create identical models for both individual and batched testing
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

    # Batched training with identical models
    batched_models = [ImageMLP(**model_config).to(device) for _ in range(num_models)]
    for batched_model, base_model in zip(batched_models, base_models):
        batched_model.load_state_dict(base_model.state_dict())

    model_batch = ModelBatch(batched_models, shared_input=True).to(device)
    optimizer_factory = OptimizerFactory(torch.optim.Adam, {"lr": 0.001})
    batched_optimizer = optimizer_factory.create_optimizer(
        model_batch, [{"lr": 0.001}] * num_models
    )
    batched_scaler = GradScaler(device="cuda")
    
    batched_loss = train_step_with_amp(
        model_batch, data, target, F.cross_entropy, batched_optimizer, batched_scaler
    )

    # Check that overflow behavior is consistent between individual and batched training
    individual_has_nan = any(np.isnan(loss) or np.isinf(loss) for loss in individual_losses)
    batched_has_nan = np.isnan(batched_loss.item()) or np.isinf(batched_loss.item())
    
    # Both should NaN/inf consistently - this is the key test
    assert individual_has_nan == batched_has_nan, f"Inconsistent NaN/inf behavior: individual={individual_has_nan}, batched={batched_has_nan}"

    # Check that valid losses are comparable (skip NaN cases)
    valid_individual = [loss for loss in individual_losses if not (np.isnan(loss) or np.isinf(loss))]
    if not batched_has_nan and len(valid_individual) == num_models:
        # All losses are valid, compare them
        individual_sum = sum(valid_individual)
        assert abs(individual_sum - batched_loss.item()) / max(abs(individual_sum), 1e-6) < 0.2
