# ruff: noqa: E402
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from modelbatch.optimizer import create_adam_configs

transformers = pytest.importorskip("transformers")
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    TrainingArguments,
)

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

    def test_incompatible_model_configs(self):
        cfg1 = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
        cfg2 = BertConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=2)
        m1 = BertForSequenceClassification(cfg1)
        m2 = BertForSequenceClassification(cfg2)
        with pytest.raises(ValueError, match="incompatible"):
            modelbatch.huggingface_integration.HFModelBatch([m1, m2])

    def test_mixed_model_classes(self):
        cfg = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
        bert = BertForSequenceClassification(cfg)
        dcfg = DistilBertConfig(dim=32, n_layers=1, n_heads=2)
        distil = DistilBertForSequenceClassification(dcfg)
        with pytest.raises(ValueError, match="incompatible"):
            modelbatch.huggingface_integration.HFModelBatch([bert, distil])

    def test_apply_method_and_attribute(self):
        # call methods with and without args
        self.batch.apply_to_submodels("train", mode=False)
        assert not any(m.training for m in self.batch.models)
        self.batch.apply_to_submodels("train", mode=True)
        assert all(m.training for m in self.batch.models)

        # set attribute across configs
        self.batch.apply_to_submodels("config.__setattr__", "foo", 5)
        assert [m.config.foo for m in self.batch.models] == [5] * len(self.batch.models)

        # missing attribute should raise
        with pytest.raises(AttributeError):
            self.batch.apply_to_submodels("config.does_not_exist")
        with pytest.raises(AttributeError):
            self.batch.apply_to_submodels("nope")

    @pytest.mark.parametrize("num_models", [1, 2, 3])
    def test_pretrained_save_load(self, num_models, tmp_path):
        model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"
        try:
            models = [
                BertForSequenceClassification.from_pretrained(model_name)
                for _ in range(num_models)
            ]
        except OSError as exc:  # pragma: no cover - network failure
            pytest.skip(f"cannot download model: {exc}")
        batch = modelbatch.huggingface_integration.HFModelBatch(models)
        path = tmp_path / f"hf_pretrained_{num_models}"
        batch.save_pretrained(str(path))
        loaded = modelbatch.huggingface_integration.HFModelBatch.from_pretrained(
            str(path)
        )
        assert len(loaded.models) == num_models


class TestSingleModelAccess:
    def test_single_model_roundtrip(self, tmp_path):
        config = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
        models = [BertForSequenceClassification(config) for _ in range(2)]
        batch = modelbatch.huggingface_integration.HFModelBatch(models)
        single = batch.models[0]

        path = tmp_path / "single_model"
        single.save_pretrained(str(path))
        loaded = type(single).from_pretrained(str(path))

        inputs = {
            "input_ids": torch.randint(0, config.vocab_size, (2, 8)),
            "attention_mask": torch.ones(2, 8),
            "labels": torch.tensor([1, 0]),
        }
        single.eval()
        loaded.eval()
        assert single(**inputs).logits.shape == loaded(**inputs).logits.shape

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, config.vocab_size, (8,)),
                    "attention_mask": torch.ones(8),
                    "labels": torch.tensor(1),
                }

        args = TrainingArguments(
            output_dir=str(tmp_path / "single_trainer"),
            per_device_train_batch_size=1,
            max_steps=1,
            disable_tqdm=True,
            report_to=[],
            remove_unused_columns=False,
            save_strategy="no",
        )
        os.environ["WANDB_DISABLED"] = "true"
        trainer = transformers.Trainer(
            model=single,
            args=args,
            train_dataset=DummyDataset(),
            eval_dataset=DummyDataset(),
        )
        trainer.train()
        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert trainer.state.global_step == 1

    def test_parameter_sharing(self, tmp_path):
        config = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
        models = [BertForSequenceClassification(config) for _ in range(2)]
        single = models[0]

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:  # pragma: no cover - simple dataset
                return 4

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover
                return {
                    "input_ids": torch.randint(0, config.vocab_size, (8,)),
                    "attention_mask": torch.ones(8),
                    "labels": torch.tensor(1),
                }

        # Train via ModelBatchTrainer and sync weights back to single model
        before = single.classifier.weight.clone()
        optimizer_cfgs = create_adam_configs([1e-4, 1e-4])
        args = TrainingArguments(
            output_dir=str(tmp_path / "mb_trainer"),
            per_device_train_batch_size=2,
            max_steps=1,
            disable_tqdm=True,
            report_to=[],
            remove_unused_columns=False,
            save_strategy="no",
        )
        os.environ["WANDB_DISABLED"] = "true"
        mb_trainer = modelbatch.huggingface_integration.ModelBatchTrainer(
            models=models,
            optimizer_configs=optimizer_cfgs,
            args=args,
            train_dataset=DummyDataset(),
        )
        mb_trainer.model_batch.compute_loss_inside_forward = True
        mb_trainer.train()
        # Simulate parameter update from training and sync to single model
        with torch.no_grad():
            mb_trainer.model_batch.stacked_params["classifier.weight"][0] += 1
        states = mb_trainer.model_batch.get_model_states()
        single.load_state_dict(states[0], strict=False)
        assert not torch.allclose(single.classifier.weight, before)

        # Train single model and verify new HFModelBatch sees updated parameters
        after_batch = single.classifier.weight.clone()
        args2 = TrainingArguments(
            output_dir=str(tmp_path / "single_trainer"),
            per_device_train_batch_size=1,
            max_steps=1,
            disable_tqdm=True,
            report_to=[],
            remove_unused_columns=False,
            save_strategy="no",
        )
        trainer = transformers.Trainer(
            model=single,
            args=args2,
            train_dataset=DummyDataset(),
        )
        trainer.train()
        # Simulate single-model training update
        with torch.no_grad():
            single.classifier.weight += 1
        new_batch = modelbatch.huggingface_integration.HFModelBatch(models)
        assert not torch.allclose(
            new_batch.stacked_params["classifier.weight"][0], after_batch
        )


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

    def test_trainer_train_step(self, tmp_path):
        config = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
        models = [BertForSequenceClassification(config) for _ in range(2)]
        optimizer_cfgs = create_adam_configs([1e-4, 1e-4])

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, config.vocab_size, (8,)),
                    "attention_mask": torch.ones(8),
                    "labels": torch.tensor(1),
                }

        args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            disable_tqdm=True,
            report_to=[],
            remove_unused_columns=False,
            save_strategy="no",
        )
        os.environ["WANDB_DISABLED"] = "true"
        trainer = modelbatch.huggingface_integration.ModelBatchTrainer(
            models=models,
            optimizer_configs=optimizer_cfgs,
            args=args,
            train_dataset=DummyDataset(),
        )
        trainer.model_batch.compute_loss_inside_forward = True
        trainer.train()
        assert trainer.state.global_step == 1
