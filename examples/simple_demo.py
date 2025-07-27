"""
Simple demo of ModelBatch training multiple MLPs simultaneously.
This demonstrates the core functionality with a clear speed comparison.
"""

from pathlib import Path
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add src to path so we can import modelbatch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch import ModelBatch
from modelbatch.optimizer import OptimizerFactory
from modelbatch.utils import create_identical_models, random_init_fn


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


def create_dummy_data(
    num_samples: int = 1000, input_size: int = 784, num_classes: int = 10
):
    """Create dummy classification data for testing."""
    x = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)


def train_sequential(models, train_loader, num_epochs, device):
    """Train models sequentially (baseline)."""
    start_time = time.time()

    for _epoch in range(num_epochs):
        for _model_idx, model in enumerate(models):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()

            for batch_data, batch_target in train_loader:
                data = batch_data.to(device)
                target = batch_target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

    return time.time() - start_time


def train_modelbatch(
    model_batch,
    train_loader,
    num_epochs,
    device,
    *,
    use_optimizer_factory=False,
):
    """Train models using ModelBatch with either same or different optimizers."""
    start_time = time.time()

    if use_optimizer_factory:
        optimizer_factory = OptimizerFactory(torch.optim.Adam)
        optimizer = optimizer_factory.create_optimizer(
            model_batch, [{"lr": 1 * 10**-i} for i in range(model_batch.num_models)]
        )
    else:
        optimizer = torch.optim.Adam(model_batch.parameters(), lr=0.001)

    for _epoch in range(num_epochs):
        model_batch.train()

        for batch_data, batch_target in train_loader:
            data = batch_data.to(device)
            target = batch_target.to(device)

            optimizer.zero_grad()
            outputs = model_batch(data)
            loss = model_batch.compute_loss(outputs, target, F.cross_entropy)
            loss.backward()
            optimizer.step()

    return time.time() - start_time


def benchmark_training(num_models: int = 32, num_epochs: int = 3):
    """Compare sequential vs ModelBatch training performance."""

    print(f"🚀 ModelBatch Demo: Training {num_models} MLPs")
    print("=" * 50)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = create_dummy_data(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(f"Dataset: {len(train_dataset)} samples, batch size: 64")

    # Create models
    model_config = {"input_size": 784, "hidden_size": 128, "num_classes": 10}

    # Sequential training
    print("\n📊 Sequential Training (Baseline)")
    print("-" * 50)
    models = create_identical_models(
        SimpleMLP, model_config, num_models, random_init_fn
    )
    for model in models:
        model.to(device)

    sequential_time = train_sequential(models, train_loader, num_epochs, device)
    print(f"Sequential time: {sequential_time:.2f}s")

    # ModelBatch training with same optimizer
    print("\n⚡ ModelBatch Training (Same Optimizer)")
    print("-" * 50)
    models = create_identical_models(
        SimpleMLP, model_config, num_models, random_init_fn
    )
    model_batch = ModelBatch(models, shared_input=True)
    model_batch.to(device)

    batch_same_time = train_modelbatch(
        model_batch, train_loader, num_epochs, device, use_optimizer_factory=False
    )
    print(f"ModelBatch (same optimizer) time: {batch_same_time:.2f}s")

    # ModelBatch training with different optimizers
    print("\n🔧 ModelBatch Training (Different Optimizers)")
    print("-" * 50)
    models = create_identical_models(
        SimpleMLP, model_config, num_models, random_init_fn
    )
    model_batch = ModelBatch(models, shared_input=True)
    model_batch.to(device)

    batch_diff_time = train_modelbatch(
        model_batch, train_loader, num_epochs, device, use_optimizer_factory=True
    )
    print(f"ModelBatch (different optimizers) time: {batch_diff_time:.2f}s")

    # Results
    print("\n🏆 Performance Comparison")
    print("=" * 50)
    speedup_same = sequential_time / batch_same_time
    speedup_diff = sequential_time / batch_diff_time
    print("Training time:\n" + "-" * 50)
    print(f"Sequential time:                    {sequential_time:.2f}s")
    print(f"ModelBatch (same optimizer):        {batch_same_time:.2f}s")
    print(f"ModelBatch (different optimizers):  {batch_diff_time:.2f}s")
    print("Speedup:\n" + "-" * 50)
    print(f"Speedup (same optimizer):           {speedup_same:.1f}x")
    print(f"Speedup (different optimizers):     {speedup_diff:.1f}x")

    return speedup_same, speedup_diff


if __name__ == "__main__":
    # Test configurations
    test_configs = [
        {"num_models": 8, "num_epochs": 2},
        {"num_models": 32, "num_epochs": 2},
    ]

    speedups_same = []
    speedups_diff = []

    for config in test_configs:
        print(f"\n{'=' * 50}")
        print(f"Testing: {config['num_models']} models, {config['num_epochs']} epochs")
        print(f"{'=' * 50}")

        speedup_same, speedup_diff = benchmark_training(**config)
        speedups_same.append(speedup_same)
        speedups_diff.append(speedup_diff)

    # Summary
    print(f"\n{'=' * 50}")
    print("📋 FINAL SUMMARY")
    print(f"{'=' * 50}")

    for i, (config, speedup_same, speedup_diff) in enumerate(
        zip(test_configs, speedups_same, speedups_diff)
    ):
        print(f"Test {i + 1}: {config['num_models']} models")
        print(f"  Same optimizer:     {speedup_same:.1f}x speedup")
        print(f"  Different optimizers: {speedup_diff:.1f}x speedup")
