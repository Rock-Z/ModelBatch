"""Grokking Benchmark: Train multiple small Transformers on modular addition."""

from __future__ import annotations

import copy
import math
import random
import sys
import time
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
    train_sequential,
    train_modelbatch,
)


# set_random_seeds is imported from benchmark_utils


@dataclass
class ModularAdditionConfig:
    modulus: int = 97  # prime modulus commonly used in grokking demos
    sequence_length: int = 2  # two operands as tokens
    num_train: int = 20_000
    num_test: int = 5_000


class ModularAdditionDataset(Dataset):
    """Generate pairs (a, b) with label (a + b) % p for a small algorithmic task."""

    def __init__(self, *, config: ModularAdditionConfig, split: str = "train"):
        assert split in {"train", "test"}
        self.p = config.modulus
        self.seq_len = config.sequence_length
        size = config.num_train if split == "train" else config.num_test

        # Generate samples uniformly at random
        rng = torch.Generator().manual_seed(3471 if split == "train" else 6235)
        a = torch.randint(low=0, high=self.p, size=(size,), generator=rng)
        b = torch.randint(low=0, high=self.p, size=(size,), generator=rng)

        # Inputs are token ids with shape [size, seq_len]
        self.inputs = torch.stack([a, b], dim=1)
        # Labels are integers in [0, p)
        self.labels = (a + b) % self.p

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
        self.q_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.o_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.attn_drop = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])
        self.resid_drop = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])

        # Per-block MLP
        hidden_dim = mlp_hidden_mult * d_model
        self.fc1 = nn.ModuleList([nn.Linear(d_model, hidden_dim) for _ in range(num_layers)])
        self.fc2 = nn.ModuleList([nn.Linear(hidden_dim, d_model) for _ in range(num_layers)])
        self.mlp_drop = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])

        # Final layer norm and classifier
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Buffers
        self.position_ids: torch.Tensor
        self.register_buffer("position_ids", torch.arange(0, max_seq_len).unsqueeze(0), persistent=False)

    def _attend(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # x: [batch, seq, d_model]
        batch_size, seq_len, _ = x.size()
        q = self.q_proj[layer_idx](x)
        k = self.k_proj[layer_idx](x)
        v = self.v_proj[layer_idx](x)

        # reshape to [batch, nhead, seq, head_dim]
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))
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
    num_train: int = 20_000,
    num_test: int = 5_000,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders for the modular addition task."""

    config = ModularAdditionConfig(modulus=modulus, sequence_length=2, num_train=num_train, num_test=num_test)
    train_ds = ModularAdditionDataset(config=config, split="train")
    test_ds = ModularAdditionDataset(config=config, split="test")

    def seed_worker(_worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(6325)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, generator=g, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


def run_benchmark(
    *,
    num_models: int = 16,
    num_epochs: int = 5,
    batch_size: int = 256,
    modulus: int = 97,
    num_train: int = 20_000,
    num_test: int = 5_000,
) -> dict[str, float]:
    """Run modular addition (grokking) benchmark with sequential vs ModelBatch training."""

    print(f"Grokking Transformer Benchmark: {num_models} Models")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Data
    trainloader, testloader = load_grokking_data(
        batch_size=batch_size,
        modulus=modulus,
        num_train=num_train,
        num_test=num_test,
    )
    train_ds = cast(Sized, trainloader.dataset)
    test_ds = cast(Sized, testloader.dataset)
    print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Hyperparameter variations across models
    dropout_rates = [0.1 + 0.02 * i for i in range(num_models)]
    learning_rates = [0.001 * (0.5**i) for i in range(num_models)]
    print(f"Dropout range: {min(dropout_rates):.3f}-{max(dropout_rates):.3f}")
    print(f"Learning rate range: {min(learning_rates):.6f}-{max(learning_rates):.6f}")

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
        for i in range(num_models)
    ]
    sample_params = sum(p.numel() for p in models[0].parameters())
    print(f"Parameters per model: {sample_params:,}")

    # Sequential baseline
    print("\n" + "=" * 60)
    sequential_models = [copy.deepcopy(models[i]) for i in range(num_models)]
    sequential_time = train_sequential(
        sequential_models, trainloader, num_epochs, learning_rates, device
    )

    # ModelBatch training
    print("\n" + "=" * 60)
    batch_models = [copy.deepcopy(models[i]) for i in range(num_models)]
    batch_time, model_batch = train_modelbatch(
        batch_models, trainloader, num_epochs, learning_rates, device
    )

    # Results
    speedup = sequential_time / max(batch_time, 1e-8)
    print("\nRESULTS")
    print("-" * 30)
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"ModelBatch: {batch_time:.2f}s")
    print(f"Speedup: {speedup:.1f}x")

    # Verify accuracy computation
    batch_accuracies = evaluate_accuracy(model_batch, testloader, device, is_batch=True)
    sequential_accuracies = evaluate_accuracy(
        sequential_models, testloader, device, is_batch=False
    )
    _ = (batch_accuracies, sequential_accuracies)  # kept for parity and potential debugging

    # GPU memory (if any)
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(
            f"GPU Memory: {memory_used:.2f}GB / {memory_total:.1f}GB ({memory_used / memory_total * 100:.1f}%)"
        )
        torch.cuda.reset_peak_memory_stats()

    return {
        "num_models": num_models,
        "sequential_time": sequential_time,
        "batch_time": batch_time,
        "speedup": speedup,
    }


if __name__ == "__main__":
    print("ModelBatch Grokking Transformer Benchmark")

    print(f"\n{'=' * 60}")
    print("SCALABILITY STUDY")
    print("=" * 60)

    configs = [
        {"num_models": 4, "num_epochs": 1},
        {"num_models": 8, "num_epochs": 1},
        {"num_models": 16, "num_epochs": 1},
        {"num_models": 32, "num_epochs": 1},
    ]

    results = []
    for config in configs:
        print(f"\nTesting {config['num_models']} models...")
        result = run_benchmark(**config)
        results.append(result)
        print(f"{result['speedup']:.1f}x speedup")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("-" * 60)
    print(f"{'Models':<8} {'Speedup':<10}")
    print("-" * 30)
    for r in results:
        print(f"{r['num_models']:<8} {r['speedup']:<10.1f}")
    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE!")


