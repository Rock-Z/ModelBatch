"""
Tests for OptimizerFactory - catching learning rate and gradient bugs.
"""

from pathlib import Path
import sys
from typing import cast

import pytest
import torch
from torch import nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch import ModelBatch, OptimizerFactory
from modelbatch.optimizer import create_adam_configs
from modelbatch.utils import create_identical_models

from .test_models import SimpleMLP


class TestOptimizerFactory:
    """Test OptimizerFactory per-model learning rates and behavior."""

    def test_optimizer_factory_basic(self):
        """Test basic OptimizerFactory creation."""
        factory = OptimizerFactory(torch.optim.Adam)
        assert factory.optimizer_cls == torch.optim.Adam
        assert factory.base_config == {}

        # With base config
        factory = OptimizerFactory(torch.optim.SGD, {"momentum": 0.9})
        assert factory.base_config == {"momentum": 0.9}

    def test_optimizer_creation_with_lr_sweep(self):
        """Test optimizer creation with lr sweep."""
        models = create_identical_models(
            SimpleMLP, {"input_size": 4, "output_size": 2}, 3
        )
        mb = ModelBatch(models)
        factory = OptimizerFactory(torch.optim.Adam)
        configs = create_adam_configs([0.001, 0.01, 0.1])
        optimizer = factory.create_optimizer(mb, configs)
        assert len(optimizer.param_groups) == 3
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[1]["lr"] == 0.01
        assert optimizer.param_groups[2]["lr"] == 0.1

    def test_optimizer_creation_wrong_config_count(self):
        """Test that wrong number of configs raises error."""
        models = create_identical_models(
            SimpleMLP, {"input_size": 4, "output_size": 2}, 3
        )
        mb = ModelBatch(models)

        factory = OptimizerFactory(torch.optim.Adam)
        configs = create_adam_configs([0.001, 0.002])  # Only 2 configs for 3 models

        with pytest.raises(ValueError, match="Expected 3 configs, got 2"):
            factory.create_optimizer(mb, configs)

    def test_individual_learning_rates(self):
        """Test that per-model learning rates work correctly."""
        models = create_identical_models(
            SimpleMLP, {"input_size": 4, "output_size": 2}, 3
        )
        mb = ModelBatch(models)

        # Create different learning rates
        learning_rates = [0.001, 0.01, 0.1]  # Very different rates
        configs = create_adam_configs(learning_rates)

        factory = OptimizerFactory(torch.optim.Adam)
        optimizer = factory.create_optimizer(mb, configs)

        # FIXED: Now we should have per-model parameter groups with different LRs
        param_groups = optimizer.param_groups

        # Verify we have the correct number of parameter groups (one per model)
        assert len(param_groups) == len(learning_rates), (
            f"Expected {len(learning_rates)} param groups, got {len(param_groups)}"
        )

        # Verify each model gets its intended learning rate
        for i, group in enumerate(param_groups):
            assert group["lr"] == learning_rates[i], (
                f"Model {i} should have LR {learning_rates[i]}, got {group['lr']}"
            )


class TestTrainingEquivalence:
    """Test training equivalence between ModelBatch and sequential training."""

    def test_gradient_equivalence_mean_vs_sum_reduction(self):
        """Test that sum reduction makes gradients equivalent."""
        models = create_identical_models(
            SimpleMLP, {"input_size": 4, "output_size": 3}, 2
        )

        # Test data
        inputs = torch.randn(5, 4)
        targets = torch.randint(0, 3, (5,))

        # ModelBatch with mean reduction
        mb_mean = ModelBatch([SimpleMLP(input_size=4, output_size=3) for _ in range(2)])
        mb_mean.load_model_states([m.state_dict() for m in models])

        outputs_mean = mb_mean(inputs)
        loss_mean = mb_mean.compute_loss(
            outputs_mean, targets, F.cross_entropy, reduction="mean"
        )
        loss_mean.backward()

        # ModelBatch with sum reduction
        mb_sum = ModelBatch([SimpleMLP(input_size=4, output_size=3) for _ in range(2)])
        mb_sum.load_model_states([m.state_dict() for m in models])

        outputs_sum = mb_sum(inputs)
        loss_sum = mb_sum.compute_loss(
            outputs_sum, targets, F.cross_entropy, reduction="sum"
        )
        loss_sum.backward()

        # Individual model gradients
        ref_gradients = []
        for model in models:
            model.zero_grad()
            out = model(inputs)
            loss = F.cross_entropy(out, targets)
            loss.backward()
            ref_gradients.append(
                [
                    p.grad.clone() if p.grad is not None else None
                    for p in model.parameters()
                ]
            )

        # Check: sum reduction should match individual gradients exactly
        for i in range(2):
            model_params = list(mb_sum.stacked_params.values())
            for j, ref_grad in enumerate(ref_gradients[i]):
                mb_grad = model_params[j].grad[i]  # Extract model i's gradient
                assert torch.allclose(mb_grad, ref_grad, atol=1e-6), (
                    f"Sum reduction gradient mismatch for model {i}, param {j}"
                )

        # Check: mean reduction gradients are scaled by 1/num_models
        for i in range(2):
            model_params = list(mb_mean.stacked_params.values())
            for j, ref_grad in enumerate(ref_gradients[i]):
                mb_grad = model_params[j].grad[i]  # Extract model i's gradient
                expected_grad = ref_grad / 2  # Scaled by 1/num_models
                assert torch.allclose(mb_grad, expected_grad, atol=1e-6), (
                    f"Mean reduction gradient scaling incorrect for model {i}, param {j}"
                )

    @pytest.mark.parametrize("num_models", [1, 2, 4, 8])
    def test_single_step_training_equivalence(self, num_models):
        """Test single training step equivalence with different learning rates per model."""
        # Create models
        torch.manual_seed(6235)
        models = [SimpleMLP(input_size=4, output_size=3) for _ in range(num_models)]

        # Test data
        inputs = torch.randn(5, 4)
        targets = torch.randint(0, 3, (5,))

        # Automatically generate different learning rates for each model
        # Use a range from 0.0001 to 0.1 with exponential spacing
        base_lr = 0.0001
        max_lr = 0.1

        if num_models == 1:
            learning_rates = [base_lr]
        else:
            learning_rates = [
                base_lr * (max_lr / base_lr) ** (i / (num_models - 1))
                for i in range(num_models)
            ]

        # Train individual models with different learning rates
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=learning_rates[i])
            for i, model in enumerate(models)
        ]

        for _i, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad()
            output = model(inputs)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()

        # Train with ModelBatch using different learning rates
        torch.manual_seed(6235)
        mb_models = [SimpleMLP(input_size=4, output_size=3) for _ in range(num_models)]

        mb = ModelBatch(cast(list[nn.Module], mb_models))

        # Test OptimizerFactory with different learning rates per model
        factory = OptimizerFactory(torch.optim.Adam)
        configs = create_adam_configs(learning_rates)
        mb_optimizer = factory.create_optimizer(mb, configs)

        # Verify each model gets its intended learning rate
        for i, expected_lr in enumerate(learning_rates):
            actual_lr = mb_optimizer.param_groups[i]["lr"]
            assert actual_lr == expected_lr, (
                f"Model {i} expected LR {expected_lr}, got {actual_lr}"
            )

        mb_optimizer.zero_grad()
        mb_outputs = mb(inputs)
        mb_loss = mb.compute_loss(mb_outputs, targets, F.cross_entropy)
        mb_loss.backward()
        mb_optimizer.step()

        # Compare final parameters - should be identical when using same LR per model
        mb_states = mb.get_model_states()
        for i, (model, mb_state) in enumerate(zip(models, mb_states)):
            for (name, param), (mb_name, mb_param) in zip(
                model.state_dict().items(), mb_state.items()
            ):
                assert name == mb_name
                # Parameters should match after training step
                assert torch.allclose(param, mb_param, atol=1e-6), (
                    f"Parameter mismatch in model {i}, param {name}"
                )
