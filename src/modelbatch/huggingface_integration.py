"""
HuggingFace integration for ModelBatch hyperparameter optimization.

Provides integration with HuggingFace transformers and datasets while
maintaining ModelBatch's batching efficiency and constraint system.
"""

from __future__ import annotations

from typing import Any

try:
    from transformers import PreTrainedModel, Trainer
    from transformers.utils.generic import ModelOutput

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

    # Don't use Any for isinstance checks - create a dummy class instead
    class _DummyPreTrainedModel:
        pass

    PreTrainedModel = _DummyPreTrainedModel  # type: ignore[assignment]
    Trainer = Any  # type: ignore[assignment]
    ModelOutput = Any  # type: ignore[assignment]

import importlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call

from .core import ModelBatch
from .optimizer import OptimizerFactory


class HFModelBatch(ModelBatch):
    """Lightweight ModelBatch adapter for HuggingFace models."""

    def __init__(
        self,
        models: list[PreTrainedModel],
        shared_input: bool = True,
    ) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for HFModelBatch")

        for m in models:
            if not isinstance(m, PreTrainedModel):
                raise TypeError(
                    "All models must be HuggingFace PreTrainedModel instances"
                )
        super().__init__(models, shared_input=shared_input)
        self._verify_model_compatibility(models)

    def forward(self, **kwargs) -> ModelOutput:  # type: ignore[name-defined]
        kwargs.pop("num_items_in_batch", None)

        outputs = []
        losses = []
        for i in range(self.num_models):
            params = {name: param[i] for name, param in self.stacked_params.items()}
            buffers = {name: buffer[i] for name, buffer in self.stacked_buffers.items()}
            output = functional_call(
                self.func_model,
                {**params, **buffers},
                (),
                kwargs,
            )
            outputs.append(output)
            if hasattr(output, "loss") and output.loss is not None:
                losses.append(output.loss)

        result = {}
        output_items = (
            outputs[0].items()
            if hasattr(outputs[0], "items")
            else (("logits", outputs[0].logits),)
            if hasattr(outputs[0], "logits")
            else ()
        )
        for key, value in output_items:
            if key == "loss" or not isinstance(value, torch.Tensor):
                continue
            values = [
                out[key] if hasattr(out, "items") else getattr(out, key)
                for out in outputs
            ]
            result[key] = torch.stack(values)

        if losses:
            if len(losses) != self.num_models:
                raise TypeError(
                    "HFModelBatch requires all models to return loss when any model does"
                )
            result["loss"] = torch.stack(losses).mean()
        if not result:
            raise TypeError(
                "HFModelBatch requires HuggingFace models to return tensor outputs"
            )
        return ModelOutput(**result)

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

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Greedy decoder-only generation through the batched forward path."""
        if do_sample:
            raise ValueError(
                "HFModelBatch.generate currently supports greedy decoding only"
            )
        if kwargs:
            raise ValueError(
                "HFModelBatch.generate only supports input_ids, attention_mask, "
                "max_new_tokens, max_length, pad_token_id, and eos_token_id"
            )
        if max_new_tokens is None:
            if max_length is None:
                raise ValueError("Either max_new_tokens or max_length must be set")
            max_new_tokens = max_length - input_ids.shape[1]
        if max_new_tokens < 0:
            raise ValueError("Generation length is shorter than the input prompt")

        device = next(self.parameters()).device
        generated = input_ids.to(device).unsqueeze(0).expand(self.num_models, -1, -1)
        if attention_mask is None:
            attention_mask = torch.ones_like(generated)
        else:
            attention_mask = (
                attention_mask.to(device).unsqueeze(0).expand(self.num_models, -1, -1)
            )

        finished = torch.zeros(
            self.num_models,
            input_ids.shape[0],
            dtype=torch.bool,
            device=device,
        )
        for _ in range(max_new_tokens):
            next_tokens = []
            for i in range(self.num_models):
                params = {name: param[i] for name, param in self.stacked_params.items()}
                buffers = {
                    name: buffer[i] for name, buffer in self.stacked_buffers.items()
                }
                output = functional_call(
                    self.func_model,
                    {**params, **buffers},
                    (),
                    {
                        "input_ids": generated[i],
                        "attention_mask": attention_mask[i],
                    },
                )
                next_tokens.append(output.logits[:, -1].argmax(dim=-1))
            next_tokens = torch.stack(next_tokens)
            if eos_token_id is not None:
                was_finished = finished
                finished = finished | (next_tokens == eos_token_id)
                if pad_token_id is not None:
                    next_tokens = torch.where(
                        was_finished,
                        torch.full_like(next_tokens, pad_token_id),
                        next_tokens,
                    )

            generated = torch.cat(
                [
                    generated,
                    next_tokens.unsqueeze(-1),
                ],
                dim=-1,
            )
            attention_mask = torch.ones_like(generated)
            if eos_token_id is not None and finished.all():
                break

        return generated

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
        for i in range(self.num_models):
            self.materialize_model(i).save_pretrained(p / f"model_{i}")
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
        **trainer_kwargs: Any,
    ) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for ModelBatchTrainer")
        self.optimizer_configs = optimizer_configs
        self.model_batch = HFModelBatch(models)
        super().__init__(model=self.model_batch, **trainer_kwargs)
        self.optimizer = self.create_optimizer()

    # Avoid saving checkpoints since ModelBatch shares tensor storage across
    # modules, which safetensors refuses to serialize. This keeps demos/tests
    # simple and non-interactive.
    def save_model(
        self, output_dir: str | None = None, _internal_call: bool = False
    ) -> None:  # type: ignore[override]
        return
