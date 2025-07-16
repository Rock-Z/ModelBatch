"""
Optimizer factory for creating optimizers with per-model parameter groups.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .core import ModelBatch

import torch
from torch.optim import Optimizer
from torch.amp.autocast_mode import autocast

class OptimizerFactory:
    """
    Factory for creating optimizers with per-model parameter groups.

    Enables different learning rates, weight decay, etc. for each model
    while using a single optimizer instance for efficiency.
    """

    def __init__(
        self,
        optimizer_cls: type[Optimizer],
        base_config: Optional[dict[str, Any]] = None,
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
            # Assign gradient views from the stacked parameters to the
            # per-model parameters before stepping.
            self._sync_gradients_to_individual(model_batch, model_parameters)

            # Perform the normal optimizer step which updates the parameter
            # views in-place. This automatically updates the stacked
            # parameters since they share storage.
            result = original_step(closure)

            return result

        def custom_zero_grad(set_to_none: bool = True):
            """Zero gradients for both individual and stacked parameters."""
            original_zero_grad(set_to_none=set_to_none)
            model_batch.zero_grad(set_to_none=set_to_none)

        # Store original methods for potential AMP usage
        optimizer._original_step = original_step  # type: ignore
        optimizer._original_zero_grad = original_zero_grad  # type: ignore
        optimizer._sync_gradients_fn = lambda: self._sync_gradients_to_individual(model_batch, model_parameters)  # type: ignore
        
        optimizer.step = custom_step  # type: ignore
        optimizer.zero_grad = custom_zero_grad  # type: ignore

        return optimizer

    def _sync_gradients_to_individual(
        self,
        model_batch: "ModelBatch",
        model_parameters: dict[int, dict[str, torch.nn.Parameter]],
    ) -> None:
        """Sync gradients from stacked parameters to individual model parameters."""
        for model_idx in range(model_batch.num_models):
            for param_name, stacked_param in model_batch.stacked_params.items():
                individual_param = model_parameters[model_idx][param_name]

                if stacked_param.grad is not None:
                    # Assign the gradient slice directly as a view to avoid
                    # extra memory allocations.
                    individual_param.grad = stacked_param.grad[model_idx]
                else:
                    individual_param.grad = None

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
        for i, config in enumerate(configs):
            # Create a scheduler for each parameter group
            # Note: This is a simplification - real implementation might need
            # custom scheduler that handles multiple param groups
            scheduler = scheduler_cls(optimizer, **config)
            schedulers.append(scheduler)

        return schedulers

    def create_amp_optimizer(
        self,
        model_batch: "ModelBatch",
        configs: list[dict[str, Any]],
    ) -> "AMPCompatibleOptimizer":
        """
        Create AMP-compatible optimizer for use with torch.amp.GradScaler.
        
        Args:
            model_batch: The ModelBatch instance
            configs: List of config dicts, one per model
            
        Returns:
            AMP-compatible optimizer wrapper
        """
        # Create the regular optimizer
        optimizer = self.create_optimizer(model_batch, configs)
        
        # Wrap it for AMP compatibility
        return AMPCompatibleOptimizer(optimizer)


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
    import numpy as np

    if scale == "log":
        lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_models)
    elif scale == "linear":
        lrs = np.linspace(min_lr, max_lr, num_models)
    else:
        raise ValueError(f"Unknown scale: {scale}")

    return [{"lr": float(lr)} for lr in lrs]


class AMPCompatibleOptimizer:
    """
    AMP-compatible wrapper for ModelBatch optimizers.
    
    This wrapper ensures that the optimizer works correctly with torch.amp.GradScaler
    by providing the proper interface and state management that GradScaler expects.
    """
    
    def __init__(self, optimizer: Optimizer):
        """
        Initialize the AMP-compatible wrapper.
        
        Args:
            optimizer: The ModelBatch optimizer to wrap
        """
        self.optimizer = optimizer
        self._model_batch = getattr(optimizer, '_model_batch', None)
        self._model_parameters = getattr(optimizer, '_model_parameters', None)
        
        # Store the custom step method from ModelBatch optimizer
        self._custom_step = optimizer.step
        
        # Restore the original step method for GradScaler compatibility
        if hasattr(optimizer, '_original_step'):
            optimizer.step = optimizer._original_step
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for both individual and stacked parameters."""
        return self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        """
        Perform optimizer step with proper gradient synchronization.
        
        This method ensures gradients are properly synced from stacked parameters
        to individual model parameters before calling the original optimizer step.
        """
        # Sync gradients from stacked to individual parameters
        if self._model_batch is not None and self._model_parameters is not None:
            self._sync_gradients_to_individual()
        
        # Call the original optimizer step
        return self.optimizer.step(closure)
    
    def _sync_gradients_to_individual(self):
        """Sync gradients from stacked parameters to individual model parameters."""
        for model_idx in range(self._model_batch.num_models):
            for param_name, stacked_param in self._model_batch.stacked_params.items():
                individual_param = self._model_parameters[model_idx][param_name]
                
                if stacked_param.grad is not None:
                    # Assign the gradient slice directly as a view to avoid
                    # extra memory allocations.
                    individual_param.grad = stacked_param.grad[model_idx]
                else:
                    individual_param.grad = None
    
    @property
    def param_groups(self):
        """Expose parameter groups for GradScaler compatibility."""
        return self.optimizer.param_groups
    
    @property 
    def state(self):
        """Expose optimizer state for GradScaler compatibility."""
        return self.optimizer.state
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped optimizer."""
        return getattr(self.optimizer, name)


def train_step_with_amp(model_batch, data, target, loss_fn, optimizer, scaler, device='cuda'):
    """
    Convenience function for performing a single training step with AMP.
    
    Args:
        model_batch: ModelBatch instance
        data: Input data tensor
        target: Target tensor  
        loss_fn: Loss function (e.g., F.cross_entropy)
        optimizer: ModelBatch optimizer
        scaler: torch.amp.GradScaler instance
        device: Device for autocast (default: 'cuda'). Can be string or torch.device
        
    Returns:
        loss: Computed loss value
    """
    optimizer.zero_grad()
    
    # Ensure device is a string for autocast
    device_str = device if isinstance(device, str) else device.type
    
    with autocast(device_str):
        outputs = model_batch(data)
        loss = model_batch.compute_loss(outputs, target, loss_fn)
    
    # Scale the loss and perform backward pass
    scaler.scale(loss).backward()
    
    # Manually sync gradients before unscaling (required for ModelBatch)
    if hasattr(optimizer, '_sync_gradients_fn'):
        optimizer._sync_gradients_fn()
    
    # Unscale gradients before calling optimizer.step()
    scaler.unscale_(optimizer)
    
    # Step the optimizer and update the scaler
    scaler.step(optimizer)
    scaler.update()
    
    return loss
