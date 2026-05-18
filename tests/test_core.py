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

    def test_module_state_exposes_only_stacked_state(self):
        """Test that normal module APIs expose only live stacked tensors."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 2)
        mb = ModelBatch(models)

        param_names = [name for name, _param in mb.named_parameters()]
        state_names = list(mb.state_dict())

        assert param_names
        assert all(name.startswith("stacked_param_") for name in param_names)
        assert not any(
            name.startswith(("models.", "func_model.")) for name in param_names
        )
        assert not any(
            name.startswith(("models.", "func_model.")) for name in state_names
        )

    def test_incompatible_models(self):
        """Test that incompatible models raise error."""
        model1 = SimpleMLP(input_size=10)
        model2 = SimpleMLP(input_size=20)  # Different input size

        with pytest.raises(ValueError, match="different.*shape"):
            ModelBatch([model1, model2])

    def test_duplicate_model_instances_raise_error(self):
        """Test that models in a batch must be independent module instances."""
        model = SimpleMLP(input_size=10)

        with pytest.raises(ValueError, match="distinct instance"):
            ModelBatch([model, model])

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

    def test_model_views_and_materialization_use_live_stacked_state(self):
        """Test model views and explicit copies use live batched state."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 2)
        mb = ModelBatch(models)

        with torch.no_grad():
            mb.stacked_params["network.0.weight"][1].add_(1.0)

        materialized = mb.materialize_model(1)

        live_state = mb.get_model_states()[1]
        for key, tensor in live_state.items():
            assert torch.allclose(mb.models[1].state_dict()[key], tensor)
            assert torch.allclose(materialized.state_dict()[key], tensor)
        assert torch.allclose(
            models[1].state_dict()["network.0.weight"],
            live_state["network.0.weight"],
        )

        with torch.no_grad():
            mb.models[1].network[0].weight.add_(2.0)
        assert torch.allclose(
            mb.stacked_params["network.0.weight"][1],
            mb.models[1].network[0].weight,
        )

    def test_model_views_refresh_after_dtype_change(self):
        """Test that .to() refreshes model views to the current stacked storage."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 2)
        mb = ModelBatch(models).to(dtype=torch.float64)

        assert mb.models[1].network[0].weight.dtype == torch.float64
        assert (
            mb.models[1].network[0].weight.data_ptr()
            == mb.stacked_params["network.0.weight"][1].data_ptr()
        )

    def test_single_model_view_forward_matches_batched_forward(self):
        """Test that indexing mb.models returns a usable live model view."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 2)
        mb = ModelBatch(models)
        mb.eval()
        input_tensor = torch.randn(4, 10)

        batched_outputs = mb(input_tensor)
        single_output = mb.models[1](input_tensor)

        assert torch.allclose(single_output, batched_outputs[1])

    def test_single_model_view_optimizer_updates_stacked_storage(self):
        """Test that optimizing a model view updates the stacked storage."""
        models = create_identical_models(SimpleMLP, {"input_size": 10}, 2)
        mb = ModelBatch(models)
        input_tensor = torch.randn(4, 10)
        target = torch.randint(0, 3, (4,))
        optimizer = torch.optim.SGD(mb.models[1].parameters(), lr=0.1)

        before = mb.stacked_params["network.0.weight"][1].clone()
        optimizer.zero_grad()
        loss = F.cross_entropy(mb.models[1](input_tensor), target)
        loss.backward()
        optimizer.step()

        assert not torch.allclose(before, mb.stacked_params["network.0.weight"][1])
        assert torch.allclose(
            mb.models[1].network[0].weight,
            mb.stacked_params["network.0.weight"][1],
        )

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
