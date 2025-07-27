"""
Unit tests for ModelBatch core functionality.
"""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch import ModelBatch
from modelbatch.utils import create_identical_models

from .test_models import SimpleMLP


class TestModelBatch:
    """Test cases for ModelBatch class."""

    def test_init_empty_models(self):
        """Test that empty model list raises error."""
        with pytest.raises(ValueError, match="At least one model must be provided"):
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

        with pytest.raises(ValueError, match="different.*shape"):
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
        assert outputs.shape == (
            3,
            batch_size,
            3,
        )  # (num_models, batch_size, output_size)

    def test_forward_different_input(self):
        """Test forward pass with different inputs per model."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 3)
        mb = ModelBatch(models, shared_input=False)

        # Create input for each model
        batch_size = 5
        input_tensor = torch.randn(
            3, batch_size, 10
        )  # (num_models, batch_size, input_size)

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

        with pytest.raises(ValueError, match="Expected 3 inputs, got 2"):
            mb(input_tensor)

    def test_compute_loss(self):
        """Test loss computation."""
        models = create_identical_models(
            SimpleMLP, {"input_size": 10, "output_size": 3}, 2
        )
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
        assert all(key.startswith("loss_model_") for key in metrics)
