"""
HuggingFace integration for ModelBatch hyperparameter optimization.

Provides integration with HuggingFace transformers and datasets while
maintaining ModelBatch's batching efficiency and constraint system.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from datasets import Dataset
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
    PreTrainedModel = Any  # type: ignore[assignment]
    PreTrainedTokenizer = Any  # type: ignore[assignment]
    TrainingArguments = Any  # type: ignore[assignment]
    Trainer = Any  # type: ignore[assignment]
    Dataset = Any  # type: ignore[assignment]
    ModelOutput = Any  # type: ignore[assignment]

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
            losses = torch.stack([out.loss for out in outputs])
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

        if stack and isinstance(results[0], torch.Tensor):
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


class ModelBatchHFTrainer:
    """
    HuggingFace Trainer wrapper for ModelBatch training.

    Provides compatibility with HuggingFace's training ecosystem while
    leveraging ModelBatch's vectorized training capabilities.
    """

    def __init__(
        self,
        model_batch: ModelBatch,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        data_collator: Callable | None = None,
        compute_metrics: Callable | None = None,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for ModelBatchHFTrainer. "
                "Install with: pip install transformers datasets"
            )

        self.model_batch = model_batch
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics

        # Create underlying trainer for dataloaders and schedulers
        self.trainer = Trainer(
            model=self.model_batch.models[0],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    def train(self) -> list[dict[str, float]]:
        """Train models using ModelBatch approach."""
        # Override training loop for ModelBatch
        return self._train_with_modelbatch()

    def _train_with_modelbatch(self) -> list[dict[str, float]]:
        """Custom training loop using ModelBatch."""
        device = torch.device(self.args.device)

        # Setup data loader
        train_dataloader = self.trainer.get_train_dataloader()

        # Setup optimizer
        optimizer = self._create_optimizer()

        # Training loop
        metrics = []
        for _epoch in range(self.args.num_train_epochs):
            epoch_metrics = self._train_epoch(train_dataloader, optimizer, device)
            metrics.append(epoch_metrics)

        return metrics

    def _create_optimizer(self):
        """Create optimizer compatible with ModelBatch."""
        # Use OptimizerFactory for per-model configurations
        factory = OptimizerFactory(torch.optim.AdamW)

        # Create configs for each model
        configs = []
        for _ in range(self.model_batch.num_models):
            config = {
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            configs.append(config)

        return factory.create_optimizer(self.model_batch, configs)

    def _train_epoch(self, dataloader, optimizer, device):
        """Train for one epoch using ModelBatch."""
        self.model_batch.train()
        total_loss = 0.0
        num_steps = 0

        for batch in dataloader:
            batch_dict = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass
            outputs = self.model_batch(**batch_dict)

            # Compute loss (custom loss function needed)
            loss = self._compute_loss(outputs, batch_dict)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

        return {"loss": total_loss / num_steps}

    def _compute_loss(self, outputs, _batch):
        """Compute loss for ModelBatch outputs."""
        labels = _batch.get("labels")
        if labels is None and hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if labels is None:
            return outputs.logits.mean()
        logits = outputs.logits
        target = labels.unsqueeze(0).expand(logits.size(0), -1, -1)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )


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
