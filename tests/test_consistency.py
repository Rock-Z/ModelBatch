from pathlib import Path
import sys
from typing import ClassVar

import pytest
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch import ModelBatch
from modelbatch.utils import create_identical_models

from .test_models import CustomModel, DeepMLP, SimpleCNN, SimpleMLP


class TestModelBatchConsistency:
    """Test class for ModelBatch consistency with individual models."""

    MODEL_CONFIGS: ClassVar = [
        (SimpleMLP, {"input_size": 8, "output_size": 4}, 3, (6, 8), (6,)),
        (SimpleMLP, {"input_size": 6, "output_size": 3}, 2, (3, 6), (3,)),
        (CustomModel, {"input_size": 8, "output_size": 4}, 3, (6, 8), (6,)),
        (CustomModel, {"input_size": 6, "output_size": 3}, 2, (3, 6), (3,)),
        (DeepMLP, {"input_size": 8, "output_size": 4}, 3, (6, 8), (6,)),
        (DeepMLP, {"input_size": 6, "output_size": 3}, 2, (3, 6), (3,)),
        (
            SimpleCNN,
            {"input_channels": 1, "num_classes": 4},
            3,
            (6, 1, 32, 32),
            (6,),
        ),
        (
            SimpleCNN,
            {"input_channels": 3, "num_classes": 3},
            2,
            (3, 3, 32, 32),
            (3,),
        ),
    ]

    @pytest.mark.parametrize(
        (
            "model_class",
            "model_params",
            "num_models",
            "input_shape",
        ),
        [
            (SimpleMLP, {"input_size": 8, "output_size": 4}, 3, (6, 8)),
            (SimpleMLP, {"input_size": 6, "output_size": 3}, 2, (3, 6)),
            (CustomModel, {"input_size": 8, "output_size": 4}, 3, (6, 8)),
            (CustomModel, {"input_size": 6, "output_size": 3}, 2, (3, 6)),
            (DeepMLP, {"input_size": 8, "output_size": 4}, 3, (6, 8)),
            (DeepMLP, {"input_size": 6, "output_size": 3}, 2, (3, 6)),
            (
                SimpleCNN,
                {"input_channels": 1, "num_classes": 4},
                3,
                (6, 1, 32, 32),
            ),
            (
                SimpleCNN,
                {"input_channels": 3, "num_classes": 3},
                2,
                (3, 3, 32, 32),
            ),
        ],
    )
    def test_output_consistency(
        self,
        model_class,
        model_params,
        num_models,
        input_shape,
    ):
        """Test that ModelBatch outputs are consistent with individual model outputs."""
        models = create_identical_models(model_class, model_params, num_models)
        mb_shared = ModelBatch(models, shared_input=True)
        mb_nonshared = ModelBatch(models, shared_input=False)

        # Create input tensor
        input_tensor = torch.randn(input_shape)

        # Test shared input
        mb_out = mb_shared(input_tensor)
        for _i, model in enumerate(models):
            ref_out = model(input_tensor)
            assert torch.allclose(mb_out[_i], ref_out, atol=1e-6)

        # Test non-shared input
        input_tensor_ns = torch.randn(num_models, *input_shape)
        mb_out_ns = mb_nonshared(input_tensor_ns)
        for _i, model in enumerate(models):
            ref_out = model(input_tensor_ns[_i])
            assert torch.allclose(mb_out_ns[_i], ref_out, atol=1e-6)

    @pytest.mark.parametrize(
        (
            "model_class",
            "model_params",
            "num_models",
            "input_shape",
            "target_shape",
        ),
        MODEL_CONFIGS,
    )
    def test_loss_consistency(
        self,
        model_class,
        model_params,
        num_models,
        input_shape,
        target_shape,
    ):
        """Test that ModelBatch loss computation is consistent with individual model losses."""
        models = create_identical_models(model_class, model_params, num_models)
        mb = ModelBatch(models)

        # Create input and targets
        input_tensor = torch.randn(input_shape)
        num_classes = model_params.get(
            "output_size", model_params.get("num_classes", 5)
        )
        targets = torch.randint(0, num_classes, target_shape)

        # Compute outputs and loss
        outputs = mb(input_tensor)
        loss_fn = F.cross_entropy
        mb_losses = mb.compute_loss(outputs, targets, loss_fn, reduction="none")

        # Compute individual losses
        ref_losses = []
        for model in models:
            out = model(input_tensor)
            ref_losses.append(loss_fn(out, targets))
        ref_losses = torch.stack(ref_losses)

        # Assertions
        for i, loss in enumerate(mb_losses):
            assert torch.allclose(loss, ref_losses[i])

    @pytest.mark.parametrize(
        (
            "model_class",
            "model_params",
            "num_models",
            "input_shape",
            "target_shape",
        ),
        MODEL_CONFIGS,
    )
    def test_gradient_consistency(
        self,
        model_class,
        model_params,
        num_models,
        input_shape,
        target_shape,
    ):
        """Test that ModelBatch gradients are consistent with individual model gradients."""
        models = create_identical_models(model_class, model_params, num_models)
        mb = ModelBatch(models)

        # Create input and targets
        input_tensor = torch.randn(input_shape, requires_grad=True)
        num_classes = model_params.get(
            "output_size", model_params.get("num_classes", 5)
        )
        targets = torch.randint(0, num_classes, target_shape)

        # Compute outputs and loss, then backward
        outputs = mb(input_tensor)
        loss = mb.compute_loss(outputs, targets, F.cross_entropy)
        loss.backward()

        # Compute individual gradients
        for _i, model in enumerate(models):
            model.zero_grad()
            out = model(input_tensor)
            loss = F.cross_entropy(out, targets)
            loss.backward()

        # Compare gradients directly
        for i, model in enumerate(models):
            for p_mb, p_ind in zip(mb.models[i].parameters(), model.parameters()):
                g_mb = p_mb.grad
                g_ind = p_ind.grad
                if g_mb is not None and g_ind is not None:
                    assert isinstance(g_mb, torch.Tensor)
                    assert isinstance(g_ind, torch.Tensor)
                    assert torch.allclose(g_mb, g_ind)

    def test_specific_model_combinations(self):
        """Test specific combinations that might have edge cases."""
        # Test with different model counts
        for num_models in [1, 2, 5]:
            models = create_identical_models(
                SimpleMLP, {"input_size": 4, "output_size": 2}, num_models
            )
            mb = ModelBatch(models)
            input_tensor = torch.randn(3, 4)
            outputs = mb(input_tensor)
            assert outputs.shape[0] == num_models

        # Test with very small inputs
        models = create_identical_models(
            SimpleMLP, {"input_size": 2, "output_size": 1}, 2
        )
        mb = ModelBatch(models)
        input_tensor = torch.randn(1, 2)
        outputs = mb(input_tensor)
        assert outputs.shape == (2, 1, 1)

    def test_empty_model_list(self):
        """Test that ModelBatch raises ValueError with empty model list."""
        with pytest.raises(ValueError, match="At least one model must be provided"):
            ModelBatch([])

    def test_mismatched_model_types(self):
        """Test that ModelBatch works with different model types."""
        models = [
            SimpleMLP(input_size=4, output_size=2),
            CustomModel(input_size=4, output_size=2),
        ]
        with pytest.raises(ValueError, match="different parameters"):
            ModelBatch(models)
