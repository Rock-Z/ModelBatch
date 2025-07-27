"""
HuggingFace integration for ModelBatch hyperparameter optimization.

Provides integration with HuggingFace transformers and datasets while
maintaining ModelBatch's batching efficiency and constraint system.
"""

from __future__ import annotations

from typing import Any

# Import transformers first - this is the main requirement
try:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers.utils.generic import ModelOutput

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    # Don't use Any for isinstance checks - create a dummy class instead
    class _DummyPreTrainedModel:
        pass
    PreTrainedModel = _DummyPreTrainedModel  # type: ignore[assignment]
    PreTrainedTokenizer = Any  # type: ignore[assignment]
    TrainingArguments = Any  # type: ignore[assignment]
    Trainer = Any  # type: ignore[assignment]
    ModelOutput = Any  # type: ignore[assignment]

# Import datasets separately - this is optional for some functionality
try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = Any  # type: ignore[assignment]

import importlib
import json
from pathlib import Path

import torch
from torch import nn

from .core import ModelBatch
from .optimizer import OptimizerFactory


class HFModelBatch(ModelBatch):
    """Lightweight ModelBatch adapter for HuggingFace models."""

    compute_loss_inside_forward: bool = False

    def __init__(
        self,
        models: list[PreTrainedModel],
        shared_input: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        # Add explicit check for transformers availability
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for HFModelBatch")

        for m in models:
            if not isinstance(m, PreTrainedModel):
                raise TypeError(
                    "All models must be HuggingFace PreTrainedModel instances"
                )
        super().__init__(models, shared_input=shared_input)
        self._verify_config_compatibility()

    def _verify_config_compatibility(self) -> None:
        if len(self.models) < 2:
            return
        attrs = [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
        ]
        ref_cfg = self.models[0].config
        for i, model in enumerate(self.models[1:], 1):
            for attr in attrs:
                if getattr(ref_cfg, attr, None) != getattr(model.config, attr, None):
                    raise ValueError(f"Model {i} differs in config field '{attr}'")

    def forward(self, **kwargs) -> ModelOutput:  # type: ignore[name-defined]
        outputs = [model(**kwargs) for model in self.models]
        logits = torch.stack([out.logits for out in outputs])
        losses = None
        if (
            self.compute_loss_inside_forward
            and hasattr(outputs[0], "loss")
            and outputs[0].loss is not None
        ):
            # Check that all outputs have non-None losses before stacking
            all_losses = [out.loss for out in outputs]
            if all(loss is not None for loss in all_losses):
                losses = torch.stack(all_losses)
        if losses is not None:
            return ModelOutput(logits=logits, loss=losses.mean())
        return ModelOutput(logits=logits)

    def apply_to_submodels(
        self, attr: str, *args, stack: bool = True, **kwargs
    ) -> list[Any] | torch.Tensor:
        results = []
        for model in self.models:
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part)
            val = obj(*args, **kwargs) if callable(obj) else obj
            results.append(val)

        if stack and results and isinstance(results[0], torch.Tensor):
            return torch.stack(results)
        return results

    def gradient_checkpointing_enable(self) -> None:
        for model in self.models:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        for model in self.models:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

    def save_pretrained(self, path: str, **_kw: Any) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "num_models": self.num_models,
            "model_cls": f"{self.models[0].__class__.__module__}."
            f"{self.models[0].__class__.__name__}",
        }
        for i, model in enumerate(self.models):
            model.save_pretrained(p / f"model_{i}")
        with (p / "hf_batch.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh)

    @classmethod
    def from_pretrained(cls, path: str, **_kw: Any) -> HFModelBatch:
        p = Path(path)
        with (p / "hf_batch.json").open(encoding="utf-8") as fh:
            meta = json.load(fh)
        module, name = meta["model_cls"].rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module), name)
        models = [
            model_cls.from_pretrained(p / f"model_{i}")
            for i in range(meta["num_models"])
        ]
        return cls(models)


class HFTrainerMixin:
    """Mixin providing optimizer logic for HF Trainer subclasses."""

    optimizer_factory_cls = OptimizerFactory

    def create_optimizer(self) -> torch.optim.Optimizer:  # type: ignore[override]
        if getattr(self, "optimizer", None) is not None:
            return self.optimizer
        factory = self.optimizer_factory_cls(torch.optim.AdamW)
        self.optimizer = factory.create_optimizer(
            self.model_batch, self.optimizer_configs
        )
        return self.optimizer


class ModelBatchTrainer(HFTrainerMixin, Trainer):
    """Minimal Trainer wrapper that builds optimizer with OptimizerFactory."""

    def __init__(
        self,
        models: list[nn.Module],
        optimizer_configs: list[dict[str, Any]],
        *,
        _lr_scheduler_configs: list[dict[str, Any]] | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for ModelBatchTrainer")
        self.optimizer_configs = optimizer_configs
        self.model_batch = HFModelBatch(models)
        super().__init__(model=self.model_batch, **trainer_kwargs)
        self.optimizer = self.create_optimizer()
