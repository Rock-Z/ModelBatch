"""Demo: training small GPT2 language models in a ``HFModelBatch``."""

from __future__ import annotations

from pathlib import Path
import os
import sys

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.training_args import TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch.huggingface_integration import ModelBatchTrainer
from modelbatch.optimizer import create_adam_configs


class RandomTextDataset(Dataset):
    def __init__(self, n_samples: int = 64, seq_len: int = 8, vocab_size: int = 100):
        self.input_ids = torch.randint(0, vocab_size, (n_samples, seq_len))
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        ids = self.input_ids[idx]
        return {
            "input_ids": ids,
            "attention_mask": self.attention_mask[idx],
            "labels": ids,
        }


def main() -> None:
    config = GPT2Config(n_layer=1, n_head=2, n_embd=32, n_positions=32, vocab_size=100)
    models = [GPT2LMHeadModel(config) for _ in range(2)]

    # Disable external reporting (e.g., wandb) for a clean demo run
    os.environ.setdefault("WANDB_DISABLED", "true")

    args = TrainingArguments(
        output_dir="/tmp/hf_pretrained",  # noqa: S108
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        save_strategy="no",
    )
    optimizer_cfgs = create_adam_configs([1e-4, 2e-4])
    trainer = ModelBatchTrainer(
        models=models,
        optimizer_configs=optimizer_cfgs,
        args=args,
        train_dataset=RandomTextDataset(1024, 8, config.vocab_size),
    )
    trainer.model_batch.compute_loss_inside_forward = True
    trainer.train()
    print("done")


if __name__ == "__main__":
    main()
