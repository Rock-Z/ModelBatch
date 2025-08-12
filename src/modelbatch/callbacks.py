"""
Callback system for ModelBatch monitoring and logging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
import warnings

import torch

from .core import ModelBatch


class Callback(ABC):
    """Base class for ModelBatch callbacks."""

    @abstractmethod
    def on_train_step(
        self,
        model_batch: ModelBatch,
        step: int,
        _metrics: dict[str, float],
    ) -> None:
        """Called after each training step."""

    @abstractmethod
    def on_validation_step(
        self,
        model_batch: ModelBatch,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Called after each validation step."""

    @abstractmethod
    def on_epoch_end(
        self,
        model_batch: ModelBatch,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of each epoch."""

class CallbackPack:
    """
    Collection of callbacks for ModelBatch monitoring and management.

    Handles logging per-model metrics, detecting divergent models,
    and applying corrective actions.
    """

    def __init__(self, callbacks: list[Callback] | None = None):
        """
        Initialize callback pack.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
        self.frozen_models: set = set()  # Track frozen/disabled models

    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the pack."""
        self.callbacks.append(callback)

    def on_train_step(
        self,
        model_batch: ModelBatch,
        step: int,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Execute all callbacks on training step."""
        if metrics is None:
            metrics = model_batch.metrics()

        for callback in self.callbacks:
            callback.on_train_step(model_batch, step, metrics)

    def on_validation_step(
        self,
        model_batch: ModelBatch,
        step: int,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Execute all callbacks on validation step."""
        if metrics is None:
            metrics = model_batch.metrics()

        for callback in self.callbacks:
            callback.on_validation_step(model_batch, step, metrics)

    def on_epoch_end(
        self,
        model_batch: ModelBatch,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Execute all callbacks on epoch end."""
        if metrics is None:
            metrics = model_batch.metrics()

        for callback in self.callbacks:
            callback.on_epoch_end(model_batch, epoch, metrics)


class NaNCallback(Callback):
    """
    Callback to detect and handle NaN losses in models.

    Freezes models with NaN losses to prevent corruption of other models.
    """

    def __init__(self, action: str = "freeze"):
        """
        Initialize NaN callback.

        Args:
            action: Action to take - "freeze", "reset", or "warn"
        """
        if action not in ["freeze", "reset", "warn"]:
            raise ValueError(f"Unknown action: {action}")

        self.action = action
        self.nan_counts = {}

    def on_train_step(
        self,
        model_batch: ModelBatch,
        step: int,
        _metrics: dict[str, float],
    ) -> None:
        """Check for NaN losses and take action."""
        if model_batch.latest_losses is None:
            return

        for i, loss in enumerate(model_batch.latest_losses):
            if torch.isnan(loss):
                self._handle_nan_model(model_batch, i, step)

    def _handle_nan_model(
        self, model_batch: ModelBatch, model_idx: int, step: int
    ) -> None:
        """Handle a model with NaN loss."""
        model_key = f"model_{model_idx}"

        # Track NaN occurrences
        self.nan_counts[model_key] = self.nan_counts.get(model_key, 0) + 1

        if self.action == "warn":
            warnings.warn(f"NaN detected in {model_key} at step {step}", stacklevel=2)

        elif self.action == "freeze":
            self._freeze_model(model_batch, model_idx)

        elif self.action == "reset":
            self._reset_model(model_batch, model_idx)

    def _freeze_model(self, model_batch: ModelBatch, model_idx: int) -> None:
        """Freeze a model by zeroing its gradients."""
        for stacked_param in model_batch.stacked_params.values():
            if stacked_param[model_idx].grad is not None:
                stacked_param[model_idx].grad.zero_()
            stacked_param[model_idx].requires_grad_(requires_grad=False)

    def _reset_model(self, model_batch: ModelBatch, model_idx: int) -> None:
        """Reset a model to its initial state using its own method if available."""
        model = model_batch.models[model_idx]
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        else:
            warnings.warn(
                f"Model at index {model_idx} does not implement reset_parameters(). "
                "Falling back to Normal(0, 0.01).",
                stacklevel=2,
            )
            for stacked_param in model_batch.stacked_params.values():
                with torch.no_grad():
                    stacked_param[model_idx].normal_(0, 0.01)


class MetricsLogger(Callback):
    """
    Callback for logging per-model metrics.

    Supports console output and integration with logging frameworks.
    """

    def __init__(
        self,
        log_frequency: int = 100,
        *,
        log_to_console: bool = True,
        logger: Callable | None = None,
    ):
        """
        Initialize metrics logger.

        Args:
            log_frequency: Log every N steps
            log_to_console: Whether to print to console
            logger: Optional external logger function
        """
        self.log_frequency = log_frequency
        self.log_to_console = log_to_console
        self.logger = logger
        self.step_count = 0

    def on_train_step(
        self,
        model_batch: ModelBatch, # noqa: ARG002
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Log training metrics."""
        self.step_count += 1

        if self.step_count % self.log_frequency == 0:
            self._log_metrics(metrics, step, "train")

    def on_validation_step(
        self,
        model_batch: ModelBatch, # noqa: ARG002
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Log validation metrics."""
        self._log_metrics(metrics, step, "val")

    def _log_metrics(self, metrics: dict[str, float], step: int, prefix: str) -> None:
        """Log metrics with given prefix."""
        if self.log_to_console:
            # Compute summary statistics
            losses = [v for k, v in metrics.items() if k.startswith("loss_model_")]
            if losses:
                sum(losses) / len(losses)
                min(losses)
                max(losses)

        if self.logger:
            # Log to external logger
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            self.logger(prefixed_metrics, step)


def create_default_callbacks(
    log_frequency: int = 100,
    *,
    handle_nan: bool = True,
    log_to_console: bool = True,
) -> CallbackPack:
    """
    Create a default set of callbacks.

    Args:
        log_frequency: How often to log metrics
        handle_nan: Whether to include NaN detection
        log_to_console: Whether to log to console

    Returns:
        CallbackPack with default callbacks
    """
    callbacks = []

    if log_to_console:
        callbacks.append(MetricsLogger(log_frequency=log_frequency))

    if handle_nan:
        callbacks.append(NaNCallback(action="freeze"))

    return CallbackPack(callbacks)
