#!/usr/bin/env python3
"""
Demo: ModelBatch training with Automatic Mixed Precision (AMP).
Trains multiple MLPs simultaneously, comparing standard and AMP training.
"""

from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add src to path so we can import modelbatch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch import ModelBatch
from modelbatch.optimizer import (
    OptimizerFactory,
    create_adam_configs,
)
from modelbatch.utils import create_identical_models, random_init_fn


def create_dummy_data(
    num_samples: int = 1000, input_size: int = 784, num_classes: int = 10
):
    """Create dummy classification data for testing."""
    x = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)


class SimpleMLP(nn.Module):
    """Simple MLP for demonstration."""

    def __init__(
        self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def train_modelbatch_amp(
    model_batch, train_loader, num_epochs, device, optimizer, *, use_amp: bool = False
):
    """Train ModelBatch with or without AMP, using a provided optimizer (per-model)."""
    scaler = GradScaler(device="cuda") if use_amp else None
    model_batch.train()
    start_time = time.time()
    final_loss = None
    for _epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # noqa: PLW2901

            if use_amp:
                optimizer.zero_grad()
                with torch.amp.autocast(device.type):
                    outputs = model_batch(data)
                    loss = model_batch.compute_loss(
                        outputs, target, F.cross_entropy
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                outputs = model_batch(data)
                loss = model_batch.compute_loss(outputs, target, F.cross_entropy)
                loss.backward()
                optimizer.step()
            final_loss = loss.item()
    elapsed = time.time() - start_time
    return elapsed, final_loss


def main():
    num_models = 8
    num_epochs = 3
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nAMP Demo: Training {num_models} MLPs on {device}")
    print("=" * 50)
    train_dataset = create_dummy_data(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model_config = {"input_size": 784, "hidden_size": 128, "num_classes": 10}

    # Per-model optimizer configs
    learning_rates = [0.001 * (0.5**i) for i in range(num_models)]
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer_configs = create_adam_configs(learning_rates)

    print("\nStandard ModelBatch Training (FP32, per-model optimizers)")
    models = create_identical_models(
        SimpleMLP, model_config, num_models, random_init_fn
    )
    model_batch = ModelBatch(models, shared_input=True)
    model_batch.to(device)
    optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)
    t_fp32, loss_fp32 = train_modelbatch_amp(
        model_batch, train_loader, num_epochs, device, optimizer, use_amp=False
    )
    print(f"Time: {t_fp32:.2f}s, Final loss: {loss_fp32:.4f}")

    if device.type == "cuda":
        print(
            "\nModelBatch Training with AMP (autocast + GradScaler, per-model optimizers)"
        )
        models = create_identical_models(
            SimpleMLP, model_config, num_models, random_init_fn
        )
        model_batch = ModelBatch(models, shared_input=True)
        model_batch.to(device)
        optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)
        t_amp, loss_amp = train_modelbatch_amp(
            model_batch, train_loader, num_epochs, device, optimizer, use_amp=True
        )
        print(f"Time: {t_amp:.2f}s, Final loss: {loss_amp:.4f}")
        print(f"\nSpeedup with AMP: {t_fp32 / t_amp:.2f}x")
    else:
        print("\nAMP is only available on CUDA devices.")


if __name__ == "__main__":
    main()
