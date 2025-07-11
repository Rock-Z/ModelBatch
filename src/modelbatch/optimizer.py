"""
Optimizer factory for creating optimizers with per-model parameter groups.
"""

from typing import Any, Dict, List, Optional, Type

import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer


class OptimizerFactory:
    """
    Factory for creating optimizers with per-model parameter groups.

    Enables different learning rates, weight decay, etc. for each model
    while using a single optimizer instance for efficiency.
    """

    def __init__(
        self,
        optimizer_cls: Type[Optimizer],
        base_config: Optional[Dict[str, Any]] = None,
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
        model_batch: "ModelBatch",  # Forward reference to avoid circular imports
        configs: List[Dict[str, Any]],
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

        # We need to create separate leaf Parameter objects for each model
        # These will be copies of the stacked parameter slices
        model_parameters = {}  # model_idx -> {param_name -> Parameter}

        for model_idx in range(model_batch.num_models):
            # Merge base config with model-specific config
            model_config = {**self.base_config, **configs[model_idx]}

            # Create separate Parameter objects for this model
            model_params = []
            model_parameters[model_idx] = {}

            for param_name, stacked_param in model_batch.stacked_params.items():
                # Create a new Parameter object that's a copy of the slice
                # This will be a leaf tensor that the optimizer can work with
                param_data = stacked_param[model_idx].clone().detach()
                model_param = torch.nn.Parameter(param_data)

                # Store the mapping for gradient synchronization
                model_param._stacked_parent = stacked_param  # type: ignore
                model_param._model_index = model_idx  # type: ignore
                model_param._param_name = param_name  # type: ignore

                model_params.append(model_param)
                model_parameters[model_idx][param_name] = model_param

            # Create parameter group for this model
            param_group = {
                "params": model_params,
                **model_config,
            }
            param_groups.append(param_group)

        # Create the optimizer
        optimizer = self.optimizer_cls(param_groups, **self.base_config)

        # Store references for custom step logic
        optimizer._model_batch = model_batch  # type: ignore
        optimizer._model_parameters = model_parameters  # type: ignore

        # Replace the step method with our custom implementation
        original_step = optimizer.step
        original_zero_grad = optimizer.zero_grad

        def custom_step(closure=None):
            # First, sync gradients from stacked parameters to individual parameters
            self._sync_gradients_to_individual(model_batch, model_parameters)

            # Perform the normal optimizer step
            result = original_step(closure)

            # Sync the updated parameters back to stacked parameters
            self._sync_parameters_to_stacked(model_batch, model_parameters)

            return result

        optimizer.step = custom_step  # type: ignore

        def custom_zero_grad(set_to_none: bool = True):
            """Zero gradients for both individual and stacked parameters."""
            original_zero_grad(set_to_none=set_to_none)
            model_batch.zero_grad(set_to_none=set_to_none)

        optimizer.zero_grad = custom_zero_grad  # type: ignore

        return optimizer

    def _sync_gradients_to_individual(
        self,
        model_batch: "ModelBatch",
        model_parameters: Dict[int, Dict[str, torch.nn.Parameter]],
    ) -> None:
        """Sync gradients from stacked parameters to individual model parameters."""
        for model_idx in range(model_batch.num_models):
            for param_name, stacked_param in model_batch.stacked_params.items():
                individual_param = model_parameters[model_idx][param_name]

                if stacked_param.grad is not None:
                    # Copy the gradient slice to the individual parameter
                    individual_param.grad = stacked_param.grad[model_idx].clone()
                else:
                    individual_param.grad = None

    def _sync_parameters_to_stacked(
        self,
        model_batch: "ModelBatch",
        model_parameters: Dict[int, Dict[str, torch.nn.Parameter]],
    ) -> None:
        """Sync updated individual parameters back to stacked parameters."""
        for model_idx in range(model_batch.num_models):
            for param_name, stacked_param in model_batch.stacked_params.items():
                individual_param = model_parameters[model_idx][param_name]

                # Copy the updated parameter data back to the stacked parameter
                with torch.no_grad():
                    stacked_param[model_idx].copy_(individual_param)

    def create_lr_scheduler(
        self,
        optimizer: Optimizer,
        scheduler_cls: Type,
        configs: List[Dict[str, Any]],
    ) -> List:
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
        for i, config in enumerate(configs):
            # Create a scheduler for each parameter group
            # Note: This is a simplification - real implementation might need
            # custom scheduler that handles multiple param groups
            scheduler = scheduler_cls(optimizer, **config)
            schedulers.append(scheduler)

        return schedulers


class AMPOptimizerFactory(OptimizerFactory):
    """
    Optimizer factory with Automatic Mixed Precision (AMP) support.
    """

    def __init__(
        self,
        optimizer_cls: Type[Optimizer],
        base_config: Optional[Dict[str, Any]] = None,
        scaler_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(optimizer_cls, base_config)
        self.scaler_config = scaler_config or {}
        self.grad_scaler = GradScaler(**self.scaler_config)

    def create_optimizer(
        self,
        model_batch: "ModelBatch",
        configs: List[Dict[str, Any]],
    ) -> Optimizer:
        """Create optimizer with AMP support."""
        return super().create_optimizer(model_batch, configs)

    def step(self, optimizer: Optimizer, closure=None) -> None:
        """
        Perform optimizer step with gradient scaling.

        Args:
            optimizer: The optimizer to step
            closure: Optional closure function
        """
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        return self.grad_scaler.scale(loss)

    def unscale_gradients(self, optimizer: Optimizer) -> None:
        """Unscale gradients before gradient clipping."""
        self.grad_scaler.unscale_(optimizer)


def create_sgd_configs(
    learning_rates: List[float],
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
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
    learning_rates: List[float],
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> List[Dict[str, Any]]:
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
) -> List[Dict[str, float]]:
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
    import numpy as np

    if scale == "log":
        lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_models)
    elif scale == "linear":
        lrs = np.linspace(min_lr, max_lr, num_models)
    else:
        raise ValueError(f"Unknown scale: {scale}")

    return [{"lr": float(lr)} for lr in lrs]
