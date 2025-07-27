"""
Optimizer factory for creating optimizers with per-model parameter groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.optim import Optimizer

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .core import ModelBatch


class OptimizerFactory:
    """
    Factory for creating optimizers with per-model parameter groups.

    Enables different learning rates, weight decay, etc. for each model
    while using a single optimizer instance for efficiency.
    """

    def __init__(
        self,
        optimizer_cls: type[Optimizer],
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the optimizer factory.

        Args:
            optimizer_cls: PyTorch optimizer class (e.g., torch.optim.Adam)
            base_config: Base configuration applied to all parameter groups
        """
        self.optimizer_cls = optimizer_cls
        self.base_config = base_config or {}

    def create_optimizer(
        self,
        model_batch: ModelBatch,  # Forward reference to avoid circular imports
        configs: list[dict[str, Any]],
    ) -> Optimizer:
        """
        Create optimizer for the stacked parameters with per-model parameter groups.

        Args:
            model_batch: The ModelBatch instance
            configs: List of config dicts, one per model

        Returns:
            Configured optimizer with separate parameter groups for each model
        """
        if len(configs) != model_batch.num_models:
            raise ValueError(
                f"Expected {model_batch.num_models} configs, got {len(configs)}",
            )

        # Create parameter groups for each model
        param_groups = []

        # Create Parameter views for each model. These share storage with the
        # stacked parameters so no data copying is required.
        model_parameters = {}  # model_idx -> {param_name -> Parameter}

        for model_idx in range(model_batch.num_models):
            # Merge base config with model-specific config
            model_config = {**self.base_config, **configs[model_idx]}

            # Create separate Parameter objects for this model
            model_params = []
            model_parameters[model_idx] = {}

            for param_name, stacked_param in model_batch.stacked_params.items():
                # Create a Parameter that is a view into the stacked tensor.
                # This keeps the optimizer API happy while avoiding extra
                # tensor copies. Gradients will be assigned manually.
                model_param = torch.nn.Parameter(stacked_param[model_idx])

                # Store the mapping for gradient synchronization
                model_param._stacked_parent = stacked_param  # type: ignore[attr-defined]
                model_param._model_index = model_idx  # type: ignore[attr-defined]
                model_param._param_name = param_name  # type: ignore[attr-defined]

                model_params.append(model_param)
                model_parameters[model_idx][param_name] = model_param

            # Create parameter group for this model
            param_group = {
                "params": model_params,
                **model_config,
            }
            param_groups.append(param_group)

        factory = self

        class BatchOptimizer(self.optimizer_cls):
            """Optimizer subclass that syncs gradients for ModelBatch."""

            def __init__(self) -> None:  # type: ignore[override]
                super().__init__(param_groups, **factory.base_config)
                self._model_batch = model_batch
                self._model_parameters = model_parameters

                for name, stacked in model_batch.stacked_params.items():
                    views = [model_parameters[i][name] for i in range(model_batch.num_models)]

                    def hook(grad, views=views):
                        for idx, view in enumerate(views):
                            view.grad = grad[idx]

                    stacked.register_hook(hook)

            def zero_grad(self, *, set_to_none: bool = True) -> None:  # type: ignore[override]
                super().zero_grad(set_to_none=set_to_none)
                model_batch.zero_grad(set_to_none=set_to_none)

        return BatchOptimizer()

    def create_lr_scheduler(
        self,
        optimizer: Optimizer,
        scheduler_cls: type,
        configs: list[dict[str, Any]],
    ) -> list:
        """
        Create per-model learning rate schedulers.

        Args:
            optimizer: The optimizer with parameter groups
            scheduler_cls: Scheduler class (e.g., torch.optim.lr_scheduler.StepLR)
            configs: List of scheduler configs, one per model

        Returns:
            List of schedulers, one per model
        """
        if len(configs) != len(optimizer.param_groups):
            raise ValueError(
                f"Expected {len(optimizer.param_groups)} configs, got {len(configs)}",
            )

        schedulers = []
        for config in configs:
            # Create a scheduler for each parameter group
            # Note: This is a simplification - real implementation might need
            # custom scheduler that handles multiple param groups
            scheduler = scheduler_cls(optimizer, **config)
            schedulers.append(scheduler)

        return schedulers



def create_sgd_configs(
    learning_rates: list[float],
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
) -> list[dict[str, Any]]:
    """
    Utility to create SGD configs with different learning rates.

    Args:
        learning_rates: List of learning rates for each model
        momentum: Momentum parameter (shared)
        weight_decay: Weight decay parameter (shared)

    Returns:
        List of optimizer configs
    """
    return [
        {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        for lr in learning_rates
    ]


def create_adam_configs(
    learning_rates: list[float],
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Utility to create Adam configs with different learning rates.

    Args:
        learning_rates: List of learning rates for each model
        betas: Adam beta parameters (shared)
        eps: Adam epsilon parameter (shared)
        weight_decay: Weight decay parameter (shared)

    Returns:
        List of optimizer configs
    """
    return [
        {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        for lr in learning_rates
    ]


def create_lr_sweep_configs(
    min_lr: float,
    max_lr: float,
    num_models: int,
    scale: str = "log",
) -> list[dict[str, float]]:
    """
    Create learning rate sweep configurations.

    Args:
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        num_models: Number of models/learning rates to generate
        scale: "log" or "linear" spacing

    Returns:
        List of configs with different learning rates
    """
    if scale == "log":
        lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_models)
    elif scale == "linear":
        lrs = np.linspace(min_lr, max_lr, num_models)
    else:
        raise ValueError(f"Unknown scale: {scale}")

    return [{"lr": float(lr)} for lr in lrs]


