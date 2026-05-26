"""Grokking Benchmark: Train multiple small Transformers on modular multiplication."""

from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sized, cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Make local src importable when running this example directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    set_random_seeds,
    evaluate_accuracy,
    create_adam_configs,
    train_single_model,
    train_modelbatch,
)


@dataclass
class ModularMultiplicationConfig:
    modulus: int = 97  # prime modulus commonly used in grokking demos
    sequence_length: int = 2  # two operands as tokens
    test_fraction: float = 0.50


class ModularMultiplicationDataset(Dataset):
    """Generate pairs (a, b) with label (a * b) % p for a small algorithmic task."""

    def __init__(self, *, config: ModularMultiplicationConfig, split: str = "train"):
        assert split in {"train", "test"}
        self.p = config.modulus
        self.seq_len = config.sequence_length
        values = torch.arange(self.p)
        a, b = torch.meshgrid(values, values, indexing="ij")
        all_inputs = torch.stack([a.reshape(-1), b.reshape(-1)], dim=1)

        rng = torch.Generator().manual_seed(3471)
        permutation = torch.randperm(all_inputs.shape[0], generator=rng)
        num_test = round(config.test_fraction * all_inputs.shape[0])
        test_indices = permutation[:num_test]
        train_indices = permutation[num_test:]
        indices = train_indices if split == "train" else test_indices

        # Inputs are token ids with shape [num_examples, seq_len]
        self.inputs = all_inputs[indices]
        # Labels are integers in [0, p)
        self.labels = (self.inputs[:, 0] * self.inputs[:, 1]) % self.p

    def __len__(self) -> int:  # type: ignore[override]
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.inputs[idx], self.labels[idx]


class SmallTransformerClassifier(nn.Module):
    """Minimal GPT-like classifier in a single class (nanoGPT-style attention).

    - Token and positional embeddings
    - num_layers blocks of: LayerNorm -> Causal MHA -> residual, LayerNorm -> MLP -> residual
    - Final classification head from last token's representation

    Attention and MLP are implemented inline per block (no auxiliary classes),
    following a simplified version of nanoGPT's design.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        mlp_hidden_mult: int = 4,
        dropout_rate: float = 0.1,
        max_seq_len: int = 2,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.num_layers = num_layers

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout_rate)

        # Per-block layer norms
        self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # Per-block attention projections (q, k, v, out)
        self.q_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.o_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.attn_drop = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_layers)]
        )
        self.resid_drop = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_layers)]
        )

        # Per-block MLP
        hidden_dim = mlp_hidden_mult * d_model
        self.fc1 = nn.ModuleList(
            [nn.Linear(d_model, hidden_dim) for _ in range(num_layers)]
        )
        self.fc2 = nn.ModuleList(
            [nn.Linear(hidden_dim, d_model) for _ in range(num_layers)]
        )
        self.mlp_drop = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_layers)]
        )

        # Final layer norm and classifier
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Buffers
        self.position_ids: torch.Tensor
        self.register_buffer(
            "position_ids", torch.arange(0, max_seq_len).unsqueeze(0), persistent=False
        )

    def _attend(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # x: [batch, seq, d_model]
        batch_size, seq_len, _ = x.size()
        q = self.q_proj[layer_idx](x)
        k = self.k_proj[layer_idx](x)
        v = self.v_proj[layer_idx](x)

        # reshape to [batch, nhead, seq, head_dim]
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(
                1, 2
            )

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scale = 1.0 / (self.head_dim**0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]
        # Causal mask
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        attn_scores = attn_scores.masked_fill(
            ~causal_mask.view(1, 1, seq_len, seq_len), float("-inf")
        )
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_drop[layer_idx](attn)
        y = torch.matmul(attn, v)  # [B, H, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        y = self.o_proj[layer_idx](y)
        y = self.resid_drop[layer_idx](y)
        return y

    def _mlp(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        x = self.fc1[layer_idx](x)
        x = F.gelu(x)
        x = self.fc2[layer_idx](x)
        x = self.mlp_drop[layer_idx](x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq_len]
        batch_size, seq_len = input_ids.shape
        position_ids = self.position_ids[:, :seq_len].to(input_ids.device)

        x = self.token_emb(input_ids) + self.pos_emb(position_ids)
        x = self.drop(x)

        # Transformer blocks (pre-norm)
        for i in range(self.num_layers):
            x = x + self._attend(self.ln1[i](x), i)
            x = x + self._mlp(self.ln2[i](x), i)

        x = self.ln_f(x)
        # Use last token representation
        x = x[:, -1, :]
        logits = self.head(x)
        return logits


def load_grokking_data(
    *,
    batch_size: int = 256,
    modulus: int = 97,
    test_fraction: float = 0.50,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders for the modular multiplication task."""

    config = ModularMultiplicationConfig(
        modulus=modulus, sequence_length=2, test_fraction=test_fraction
    )
    train_ds = ModularMultiplicationDataset(config=config, split="train")
    test_ds = ModularMultiplicationDataset(config=config, split="test")

    def seed_worker(_worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(6325)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        generator=g,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, test_loader


if __name__ == "__main__":
    print("ModelBatch Grokking Transformer Benchmark")

    print(f"\n{'=' * 60}")
    print("SCALABILITY STUDY")
    print("=" * 60)

    configs = [
        {"num_models": 4},
        {"num_models": 8},
        {"num_models": 16},
        {"num_models": 32},
    ]
    num_epochs = 1000
    batch_size = 256
    modulus = 97
    test_fraction = 0.50
    max_num_models = max(config["num_models"] for config in configs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Data
    trainloader, testloader = load_grokking_data(
        batch_size=batch_size,
        modulus=modulus,
        test_fraction=test_fraction,
    )
    train_ds = cast(Sized, trainloader.dataset)
    test_ds = cast(Sized, testloader.dataset)
    print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Stable grokking recipe for this GPT-like modular multiplication task.
    dropout_choices = [0.10, 0.00, 0.05, 0.15]
    dropout_rates = [
        dropout_choices[i % len(dropout_choices)] for i in range(max_num_models)
    ]
    learning_rate = 1e-3
    print(f"Dropout range: {min(dropout_rates):.3f}-{max(dropout_rates):.3f}")
    print(f"Learning rate: {learning_rate:.6f}")

    # Models
    set_random_seeds()
    models = [
        SmallTransformerClassifier(
            vocab_size=modulus,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout_rate=dropout_rates[i],
            max_seq_len=2,
        )
        for i in range(max_num_models)
    ]
    sample_params = sum(p.numel() for p in models[0].parameters())
    print(f"Parameters per model: {sample_params:,}")

    # Sequential baseline, trained once.
    print("\n" + "=" * 60)
    sequential_model = copy.deepcopy(models[0])
    sequential_time_per_model = train_single_model(
        sequential_model,
        trainloader,
        num_epochs,
        learning_rate,
        device,
    )
    sequential_train_accuracy = evaluate_accuracy(
        [sequential_model], trainloader, device, is_batch=False
    )[0]
    sequential_test_accuracy = evaluate_accuracy(
        [sequential_model], testloader, device, is_batch=False
    )[0]
    print(f"Sequential train accuracy: {sequential_train_accuracy:.1f}%")
    print(f"Sequential test accuracy: {sequential_test_accuracy:.1f}%")

    results = []
    for config in configs:
        num_models = config["num_models"]
        print(f"\nTesting {num_models} models...")
        print("\n" + "=" * 60)

        batch_models = [copy.deepcopy(models[i]) for i in range(num_models)]
        batch_time, model_batch = train_modelbatch(
            batch_models,
            trainloader,
            num_epochs,
            device,
            optimizer_configs=create_adam_configs([learning_rate] * num_models),
        )

        sequential_time = sequential_time_per_model * num_models
        speedup = sequential_time / max(batch_time, 1e-8)
        print("\nRESULTS")
        print("-" * 30)
        print(
            f"Sequential: {sequential_time:.2f}s "
            f"({sequential_time_per_model:.2f}s/model x {num_models})"
        )
        print(f"ModelBatch: {batch_time:.2f}s")
        print(f"Speedup: {speedup:.1f}x")
        print(f"Sequential test accuracy: {sequential_test_accuracy:.1f}%")

        # Check the trained batched models.
        batch_accuracies = evaluate_accuracy(
            model_batch, testloader, device, is_batch=True
        )
        best_accuracy = max(batch_accuracies)
        print(
            "ModelBatch accuracy: "
            f"best={best_accuracy:.1f}%, "
            f"mean={np.mean(batch_accuracies):.1f}%, "
            f"range={min(batch_accuracies):.1f}-{max(batch_accuracies):.1f}%"
        )

        result = {
            "num_models": num_models,
            "sequential_time": sequential_time,
            "batch_time": batch_time,
            "speedup": speedup,
            "sequential_test_accuracy": sequential_test_accuracy,
            "best_batch_accuracy": best_accuracy,
        }
        results.append(result)
        print(f"{speedup:.1f}x speedup")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("-" * 60)
    print(f"{'Models':<8} {'Speedup':<10} {'Seq acc':<10} {'Best acc':<10}")
    print("-" * 30)
    for r in results:
        print(
            f"{r['num_models']:<8} {r['speedup']:<10.1f} "
            f"{r['sequential_test_accuracy']:<10.1f} {r['best_batch_accuracy']:<10.1f}"
        )
    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE!")
