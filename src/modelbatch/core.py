"""
Core ModelBatch implementation using torch.vmap for vectorized training.
"""

from typing import Callable, Dict, List, Optional

import torch
from torch import nn
from torch.func import functional_call, stack_module_state


class ModelBatch(nn.Module):
    """
    Vectorized batch of independent PyTorch models.

    Stacks parameters from multiple structurally identical models and uses
    torch.vmap to execute forward/backward passes in parallel.

    Args:
        models: List of PyTorch models (must have identical structure)
        shared_input: If True, all models receive the same input data
    """

    def __init__(self, models: List[nn.Module], shared_input: bool = True):
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided")

        # Verify all models have the same structure
        self._verify_model_compatibility(models)

        self.num_models = len(models)
        self.shared_input = shared_input

        # Store reference models for metadata/inspection
        self.models = nn.ModuleList(models)

        # Stack parameters and buffers from all models
        stacked_params, stacked_buffers = stack_module_state(models)

        # Register stacked parameters as individual PyTorch parameters so they move with .to(device)
        self.stacked_params = {}
        for name, param in stacked_params.items():
            # Replace dots with underscores for PyTorch parameter names
            safe_name = name.replace(".", "_")
            param_tensor = nn.Parameter(param)
            setattr(self, f"stacked_param_{safe_name}", param_tensor)
            self.stacked_params[name] = param_tensor

        # Register stacked buffers as PyTorch buffers
        self.stacked_buffers = {}
        for name, buffer in stacked_buffers.items():
            safe_name = name.replace(".", "_")
            self.register_buffer(f"stacked_buffer_{safe_name}", buffer)
            self.stacked_buffers[name] = getattr(self, f"stacked_buffer_{safe_name}")

        # Store the functional form of the first model for vmap
        self.func_model = models[0]

        # Track latest losses for monitoring
        self.latest_losses: Optional[torch.Tensor] = None

        # Enable/disable compilation
        self._compiled = False

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients for all stacked parameters."""
        for param in self.stacked_params.values():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def _verify_model_compatibility(self, models: List[nn.Module]) -> None:
        """Verify all models have identical structure."""
        if len(models) < 2:
            return

        reference = models[0]
        ref_state = reference.state_dict()

        for i, model in enumerate(models[1:], 1):
            model_state = model.state_dict()

            if set(ref_state.keys()) != set(model_state.keys()):
                raise ValueError(f"Model {i} has different parameters than model 0")

            for key in ref_state.keys():
                if ref_state[key].shape != model_state[key].shape:
                    raise ValueError(
                        f"Parameter '{key}' has different shape in model {i}: "
                        f"{model_state[key].shape} vs {ref_state[key].shape}",
                    )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward pass through all models.

        Args:
            inputs: Input tensor. If shared_input=True, shape [batch_size, ...].
                   If shared_input=False, shape [num_models, batch_size, ...].

        Returns:
            Output tensor of shape [num_models, batch_size, ...]
        """
        if self.shared_input:
            # Broadcast input to all models
            if inputs.dim() == 0:
                raise ValueError("Input tensor must have at least 1 dimension")

            # Create a wrapper function that applies one model with given params/buffers
            def apply_model_shared(params, buffers):
                # Combine parameters and buffers into single state dict
                combined_state = {**params, **buffers}
                return functional_call(self.func_model, combined_state, inputs)

            # Use vmap to vectorize over parameter/buffer dimensions
            vectorized_func = torch.vmap(
                apply_model_shared,
                in_dims=(0, 0),
                out_dims=0,
                randomness="different",
            )

            with torch.random.fork_rng():
                outputs = vectorized_func(
                    self.stacked_params,
                    self.stacked_buffers,
                )
        else:
            # Each model gets different input
            if inputs.shape[0] != self.num_models:
                raise ValueError(
                    f"Expected {self.num_models} inputs, got {inputs.shape[0]}",
                )

            # Create a wrapper function for different inputs per model
            def apply_model_separate(params, buffers, input_):
                # Combine parameters and buffers into single state dict
                combined_state = {**params, **buffers}
                return functional_call(self.func_model, combined_state, input_)

            # Use vmap to vectorize over all dimensions
            vectorized_func = torch.vmap(
                apply_model_separate,
                in_dims=(0, 0, 0),
                out_dims=0,
                randomness="different",
            )

            with torch.random.fork_rng():
                outputs = vectorized_func(
                    self.stacked_params,
                    self.stacked_buffers,
                    inputs,
                )

        return outputs

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reduction: str = "sum",
    ) -> torch.Tensor:
        """
        Compute loss for all models.

        Args:
            outputs: Model outputs [num_models, batch_size, ...]
            targets: Target values [batch_size, ...] (shared) or [num_models, batch_size, ...]
            loss_fn: Loss function
            reduction: How to combine per-model losses ("mean", "sum", "none")

        Returns:
            Combined loss (scalar) or per-model losses [num_models]
        """
        # Check if targets need to be broadcast to all models
        expected_batch_size = (
            outputs.shape[1] if outputs.dim() >= 2 else outputs.shape[0]
        )

        # If targets is 1D or 2D but doesn't have the right shape for per-model targets
        if targets.dim() == 1:
            # Always broadcast 1D targets to all models
            if targets.shape[0] == expected_batch_size:
                targets = targets.unsqueeze(0).expand(self.num_models, -1)
            else:
                raise ValueError(
                    f"1D target shape {targets.shape} doesn't match batch size {expected_batch_size}",
                )
        elif targets.dim() >= 2 and not (
            targets.shape[0] == self.num_models
            and targets.shape[1] == expected_batch_size
        ):
            # For multi-dimensional targets, only skip broadcasting if they're already in [num_models, batch_size, ...] format
            if targets.shape[0] == expected_batch_size:
                targets = targets.unsqueeze(0).expand(
                    self.num_models,
                    -1,
                    *[-1] * (targets.dim() - 1),
                )
            else:
                raise ValueError(
                    f"Target shape {targets.shape} doesn't match expected format. "
                    f"Expected either [batch_size={expected_batch_size}, ...] for shared targets "
                    f"or [num_models={self.num_models}, batch_size={expected_batch_size}, ...] for per-model targets.",
                )

        # Compute loss for each model
        per_model_losses = torch.vmap(loss_fn)(outputs, targets)

        # Store for monitoring
        self.latest_losses = per_model_losses.detach()

        if reduction == "mean":
            return per_model_losses.mean()
        if reduction == "sum":
            return per_model_losses.sum()
        if reduction == "none":
            return per_model_losses
        raise ValueError(f"Unknown reduction: {reduction}")

    def get_model_states(self) -> List[Dict[str, torch.Tensor]]:
        """Extract individual model state dicts."""
        states = []

        for i in range(self.num_models):
            state_dict = {}
            # Extract parameters
            for name, stacked_param in self.stacked_params.items():
                state_dict[name] = stacked_param[i].clone()
            # Extract buffers
            for name, stacked_buffer in self.stacked_buffers.items():
                state_dict[name] = stacked_buffer[i].clone()
            states.append(state_dict)

        return states

    def load_model_states(self, states: List[Dict[str, torch.Tensor]]) -> None:
        """Load individual model states back into the batch."""
        if len(states) != self.num_models:
            raise ValueError(f"Expected {self.num_models} states, got {len(states)}")

        for i, state_dict in enumerate(states):
            for key, param in state_dict.items():
                # Check if it's a parameter
                if key in self.stacked_params:
                    with torch.no_grad():
                        self.stacked_params[key][i].copy_(param)
                # Check if it's a buffer
                elif key in self.stacked_buffers:
                    with torch.no_grad():
                        self.stacked_buffers[key][i].copy_(param)

    def save_all(self, path: str) -> None:
        """Save all model states to directory."""
        import os

        os.makedirs(path, exist_ok=True)

        states = self.get_model_states()
        for i, state in enumerate(states):
            torch.save(state, os.path.join(path, f"model_{i}.pt"))

    def load_all(self, path: str) -> None:
        """Load all model states from directory."""
        import os

        states = []
        for i in range(self.num_models):
            state_path = os.path.join(path, f"model_{i}.pt")
            if not os.path.exists(state_path):
                raise FileNotFoundError(f"Model state not found: {state_path}")
            states.append(torch.load(state_path))

        self.load_model_states(states)

    def enable_compile(self, **kwargs) -> None:
        """Enable torch.compile for the vectorized function."""
        if hasattr(torch, "compile"):
            self.func_model = torch.compile(self.func_model, **kwargs)
            self._compiled = True

    def metrics(self) -> Dict[str, float]:
        """Get per-model metrics (losses)."""
        if self.latest_losses is None:
            return {}

        return {
            f"loss_model_{i}": float(loss) for i, loss in enumerate(self.latest_losses)
        }
