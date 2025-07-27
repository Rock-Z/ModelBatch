"""Quick demo of training custom ``PreTrainedModel`` instances in ``HFModelBatch``."""
# ruff: noqa: INP001

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch.huggingface_integration import HFModelBatch, ModelBatchHFTrainer


class TinyConfig(PretrainedConfig):
    model_type = "tiny"

    def __init__(
        self, input_size: int = 10, hidden_size: int = 8, num_labels: int = 2, **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels


class TinyModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config: TinyConfig):
        super().__init__(config)
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, labels=None):
        x = F.relu(self.fc1(input_ids.float()))
        logits = self.fc2(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return type("Output", (), {"logits": logits, "loss": loss})()


class TinyDataset(Dataset):
    def __init__(self, n_samples: int = 100, config: TinyConfig | None = None):
        self.config = config or TinyConfig()
        self.inputs = torch.randn(n_samples, self.config.input_size)
        self.labels = torch.randint(0, self.config.num_labels, (n_samples,))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}


def main() -> None:
    config = TinyConfig()
    models = [TinyModel(config) for _ in range(2)]
    batch = HFModelBatch(models)
    batch.compute_loss_inside_forward = True

    args = TrainingArguments(
        output_dir="/tmp/hf_custom",  # noqa: S108
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=1,
    )
    trainer = ModelBatchHFTrainer(
        model_batch=batch,
        args=args,
        train_dataset=TinyDataset(64, config),
    )
    trainer.train()
    print("done")


if __name__ == "__main__":
    main()
