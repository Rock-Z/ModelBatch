#!/usr/bin/env python3
"""
Unit tests for ModelBatch core functionality.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modelbatch import ModelBatch
from modelbatch.utils import create_identical_models


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TestModelBatch:
    """Test cases for ModelBatch class."""
    
    def test_init_empty_models(self):
        """Test that empty model list raises error."""
        with pytest.raises(ValueError):
            ModelBatch([])
    
    def test_init_single_model(self):
        """Test initialization with single model."""
        model = SimpleMLP()
        mb = ModelBatch([model])
        assert mb.num_models == 1
        assert len(mb.models) == 1
    
    def test_init_multiple_models(self):
        """Test initialization with multiple models."""
        models = create_identical_models(SimpleMLP, {}, 4)
        mb = ModelBatch(models)
        assert mb.num_models == 4
        assert len(mb.models) == 4
    
    def test_incompatible_models(self):
        """Test that incompatible models raise error."""
        model1 = SimpleMLP(input_size=10)
        model2 = SimpleMLP(input_size=20)  # Different input size
        
        with pytest.raises(ValueError):
            ModelBatch([model1, model2])
    
    def test_forward_shared_input(self):
        """Test forward pass with shared input."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 3)
        mb = ModelBatch(models, shared_input=True)
        
        # Create input
        batch_size = 5
        input_tensor = torch.randn(batch_size, 10)
        
        # Forward pass
        outputs = mb(input_tensor)
        
        # Check output shape
        assert outputs.shape == (3, batch_size, 3)  # (num_models, batch_size, output_size)
    
    def test_forward_different_input(self):
        """Test forward pass with different inputs per model."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 3)
        mb = ModelBatch(models, shared_input=False)
        
        # Create input for each model
        batch_size = 5
        input_tensor = torch.randn(3, batch_size, 10)  # (num_models, batch_size, input_size)
        
        # Forward pass
        outputs = mb(input_tensor)
        
        # Check output shape
        assert outputs.shape == (3, batch_size, 3)
    
    def test_forward_wrong_input_shape(self):
        """Test that wrong input shape raises error."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 3)
        mb = ModelBatch(models, shared_input=False)
        
        # Wrong number of models in input
        input_tensor = torch.randn(2, 5, 10)  # Should be (3, 5, 10)
        
        with pytest.raises(ValueError):
            mb(input_tensor)
    
    def test_compute_loss(self):
        """Test loss computation."""
        models = create_identical_models(SimpleMLP, {"input_size": 10, "output_size": 3}, 2)
        mb = ModelBatch(models)
        
        # Create dummy data
        outputs = torch.randn(2, 5, 3)  # (num_models, batch_size, num_classes)
        targets = torch.randint(0, 3, (5,))  # (batch_size,)
        
        # Compute loss
        loss = mb.compute_loss(outputs, targets, F.cross_entropy)
        
        # Check that loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert mb.latest_losses is not None
        assert mb.latest_losses.shape == (2,)  # Per-model losses
    
    def test_get_set_model_states(self):
        """Test getting and setting model states."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 2)
        mb = ModelBatch(models)
        
        # Get initial states
        states = mb.get_model_states()
        assert len(states) == 2
        
        # Modify parameters
        with torch.no_grad():
            for param in mb.parameters():
                param.add_(1.0)
        
        # Load original states back
        mb.load_model_states(states)
        
        # Check that parameters are restored
        new_states = mb.get_model_states()
        for old_state, new_state in zip(states, new_states):
            for key in old_state:
                assert torch.allclose(old_state[key], new_state[key])
    
    def test_save_load_all(self, tmp_path):
        """Test saving and loading all models."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 2)
        mb = ModelBatch(models)
        
        # Save models
        save_dir = str(tmp_path / "test_models")
        mb.save_all(save_dir)
        
        # Create new ModelBatch and load
        new_models = create_identical_models(SimpleMLP, {"input_size": 10}, 2)
        new_mb = ModelBatch(new_models)
        new_mb.load_all(save_dir)
        
        # Check that states match
        old_states = mb.get_model_states()
        new_states = new_mb.get_model_states()
        
        for old_state, new_state in zip(old_states, new_states):
            for key in old_state:
                assert torch.allclose(old_state[key], new_state[key])
    
    def test_metrics(self):
        """Test metrics generation."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 3)
        mb = ModelBatch(models)
        
        # Initially no metrics
        metrics = mb.metrics()
        assert len(metrics) == 0
        
        # After computing loss
        outputs = torch.randn(3, 5, 3)
        targets = torch.randint(0, 3, (5,))
        mb.compute_loss(outputs, targets, F.cross_entropy)
        
        metrics = mb.metrics()
        assert len(metrics) == 3
        assert all(key.startswith("loss_model_") for key in metrics.keys())


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    import traceback
    
    test_class = TestModelBatch()
    test_methods = [method for method in dir(test_class) if method.startswith("test_")]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            if method_name == "test_save_load_all":
                # Mock tmp_path for manual testing
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    class MockPath:
                        def __init__(self, path):
                            self.path = path
                        def __truediv__(self, other):
                            return os.path.join(self.path, other)
                    method(MockPath(tmp_dir))
            else:
                method()
            print(f"✅ {method_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed") 