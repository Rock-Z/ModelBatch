"""
Test AMP (Automatic Mixed Precision) integration with ModelBatch optimizers.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp.grad_scaler import GradScaler

from modelbatch import ModelBatch
from modelbatch.optimizer import OptimizerFactory, create_adam_configs, train_step_with_amp
from modelbatch.utils import create_identical_models, random_init_fn
from test_models import ImageMLP, create_dummy_data


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
        'device': device,
        'model_batch': model_batch,
        'loader': loader,
        'optimizer': optimizer,
        'models': models
    }


def test_amp_training_step(setup_amp_test):
    """Test that a single AMP training step works correctly."""
    setup = setup_amp_test
    device = setup['device']
    model_batch = setup['model_batch']
    optimizer = setup['optimizer']
    
    # Get a batch of data
    data, target = next(iter(setup['loader']))
    data, target = data.to(device), target.to(device)
    
    # Create scaler
    scaler = GradScaler(device='cuda')
    
    # Test AMP training step
    loss = train_step_with_amp(model_batch, data, target, F.cross_entropy, optimizer, scaler)
    
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


def test_amp_multiple_steps(setup_amp_test):
    """Test multiple AMP training steps."""
    setup = setup_amp_test
    device = setup['device']
    model_batch = setup['model_batch']
    optimizer = setup['optimizer']
    loader = setup['loader']
    
    scaler = GradScaler(device='cuda')
    
    initial_params = [p.clone() for p in model_batch.parameters()]
    losses = []
    
    # Run several training steps
    for i, (data, target) in enumerate(loader):
        if i >= 5:  # Only run a few steps
            break
        data, target = data.to(device), target.to(device)
        
        loss = train_step_with_amp(model_batch, data, target, F.cross_entropy, optimizer, scaler)
        losses.append(loss.item())
    
    # Check that parameters have changed
    params_changed = any(
        not torch.allclose(initial, current)
        for initial, current in zip(initial_params, model_batch.parameters())
    )
    assert params_changed, "Parameters didn't change during training"
    
    # Check that we have multiple loss values (at least 3 steps)
    assert len(losses) >= 3, f"Expected at least 3 training steps, got {len(losses)}"
    assert all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in losses)


def test_amp_vs_fp32_consistency(setup_amp_test):
    """Test that AMP and FP32 training produce similar results."""
    setup = setup_amp_test
    device = setup['device']
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
    
    scaler = GradScaler(device='cuda')
    
    # Train for a few steps
    data, target = next(iter(setup['loader']))
    data, target = data.to(device), target.to(device)
    
    for _ in range(3):
        # AMP step
        loss_amp = train_step_with_amp(model_batch_amp, data, target, F.cross_entropy, optimizer_amp, scaler)
        
        # FP32 step
        optimizer_fp32.zero_grad()
        outputs_fp32 = model_batch_fp32(data)
        loss_fp32 = model_batch_fp32.compute_loss(outputs_fp32, target, F.cross_entropy)
        loss_fp32.backward()
        optimizer_fp32.step()
    
    # Losses should be similar (within reasonable tolerance due to FP16 precision)
    assert abs(loss_amp.item() - loss_fp32.item()) < 1.0, f"AMP and FP32 losses differ too much: {loss_amp.item():.4f} vs {loss_fp32.item():.4f}"


def test_amp_with_different_optimizers(setup_amp_test):
    """Test AMP with different optimizer types."""
    setup = setup_amp_test
    device = setup['device']
    model_batch = setup['model_batch']
    
    optimizers_to_test = [
        (torch.optim.Adam, {'lr': 0.001}),
        (torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
        (torch.optim.AdamW, {'lr': 0.001, 'weight_decay': 1e-4}),
    ]
    
    for opt_cls, base_config in optimizers_to_test:
        # Create fresh models for each optimizer test
        model_config = {"input_size": 784, "hidden_size": 64, "num_classes": 10}
        models = create_identical_models(ImageMLP, model_config, 2, random_init_fn)
        mb = ModelBatch(models, shared_input=True).to(device)
        
        optimizer_factory = OptimizerFactory(opt_cls, base_config)
        configs = [base_config, base_config]
        optimizer = optimizer_factory.create_optimizer(mb, configs)
        
        scaler = GradScaler(device='cuda')
        
        # Test a single step
        data, target = next(iter(setup['loader']))
        data, target = data.to(device), target.to(device)
        
        try:
            loss = train_step_with_amp(mb, data, target, F.cross_entropy, optimizer, scaler)
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
        except Exception as e:
            pytest.fail(f"AMP failed with {opt_cls.__name__}: {e}")
