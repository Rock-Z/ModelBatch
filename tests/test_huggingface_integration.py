# ruff: noqa: E402
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from modelbatch.optimizer import create_adam_configs

transformers = pytest.importorskip("transformers")
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments

import modelbatch


class TestHFModelBatch:
    def setup_method(self):
        config = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
        self.models = [BertForSequenceClassification(config) for _ in range(2)]
        self.batch = modelbatch.huggingface_integration.HFModelBatch(self.models)
        self.config = config

    def test_forward(self):
        inputs = {
            "input_ids": torch.randint(0, self.config.vocab_size, (4, 8)),
            "attention_mask": torch.ones(4, 8),
        }
        outputs = self.batch(**inputs)
        assert outputs.logits.shape[0] == len(self.models)

    def test_apply_to_submodels(self):
        hs = self.batch.apply_to_submodels("config.hidden_size")
        assert hs == [self.config.hidden_size] * len(self.models)
        param_counts = self.batch.apply_to_submodels("num_parameters", stack=False)
        assert isinstance(param_counts, list)
        assert len(param_counts) == len(self.models)

    def test_checkpoint_roundtrip(self, tmp_path):
        path = tmp_path / "hf_pack"
        self.batch.save_pretrained(str(path))
        loaded = modelbatch.huggingface_integration.HFModelBatch.from_pretrained(
            str(path)
        )
        assert len(loaded.models) == len(self.models)

    def test_gradient_checkpointing_toggle(self):
        self.batch.gradient_checkpointing_enable()
        assert all(m.is_gradient_checkpointing for m in self.batch.models)
        self.batch.gradient_checkpointing_disable()
        assert not any(m.is_gradient_checkpointing for m in self.batch.models)


class TestModelBatchTrainer:
    def test_optimizer_param_groups(self, tmp_path):
        config = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
        models = [BertForSequenceClassification(config) for _ in range(3)]
        trainer_cls = modelbatch.huggingface_integration.ModelBatchTrainer
        optimizer_cfgs = create_adam_configs([1e-4, 2e-4, 3e-4])
        args = TrainingArguments(
            output_dir=str(tmp_path), per_device_train_batch_size=2
        )
        trainer = trainer_cls(
            models=models, optimizer_configs=optimizer_cfgs, args=args
        )
        optimizer = trainer.optimizer
        assert len(optimizer.param_groups) == len(models)
        lrs = [g["lr"] for g in optimizer.param_groups]
        assert lrs == [1e-4, 2e-4, 3e-4]
