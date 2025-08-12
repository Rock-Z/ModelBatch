"""CIFAR10 LeNet Benchmark: Train multiple LeNet models simultaneously."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sized, cast
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark_utils import (
    set_random_seeds,
    evaluate_accuracy,
    train_sequential,
    train_modelbatch,
)
# set_random_seeds is imported from benchmark_utils
class LeNet5CIFAR(nn.Module):
    """LeNet-5 adapted for CIFAR10."""

    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def load_cifar10_data(
    batch_size: int = 128, num_samples: int | None = None
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR10 with standard preprocessing."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    if num_samples is not None:
        # Use fixed indices for reproducibility
        torch.manual_seed(42)
        indices = torch.randperm(len(trainset))[:num_samples].tolist()
        trainset = Subset(trainset, indices)
        test_indices = torch.randperm(len(testset))[: num_samples // 5].tolist()
        testset = Subset(testset, test_indices)

    def seed_worker(_worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(6325)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=g,
        worker_init_fn=seed_worker,
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainloader, testloader


# evaluate_accuracy is imported from benchmark_utils


# train_sequential is imported from benchmark_utils


# train_modelbatch is imported from benchmark_utils


def run_benchmark(
    num_models: int = 16,
    num_epochs: int = 5,
    batch_size: int = 128,
    num_samples: int = 60000,
) -> dict[str, float]:
    """
    Run CIFAR10 LeNet benchmark. Compare sequential training and ModelBatch training.
    """
    print(f"CIFAR10 LeNet Benchmark: {num_models} Models")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load data
    trainloader, testloader = load_cifar10_data(
        batch_size=batch_size, num_samples=num_samples
    )
    train_ds = cast(Sized, trainloader.dataset)
    test_ds = cast(Sized, testloader.dataset)
    print(
        f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}"
    )

    # Create hyperparameter variations
    dropout_rates = [0.1 + 0.02 * i for i in range(num_models)]
    learning_rates = [0.01 * (0.5**i) for i in range(num_models)]
    print(f"Dropout range: {min(dropout_rates):.3f}-{max(dropout_rates):.3f}")
    print(f"Learning rate range: {min(learning_rates):.6f}-{max(learning_rates):.6f}")

    # Create models with deterministic initialization
    set_random_seeds()
    models = [LeNet5CIFAR(dropout_rate=dropout_rates[i]) for i in range(num_models)]
    sample_params = sum(p.numel() for p in models[0].parameters())
    print(f"Parameters per model: {sample_params:,}")

    # Sequential training
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

    # Performance comparison
    speedup = sequential_time / batch_time

    print("\nRESULTS")
    print("-" * 30)
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"ModelBatch: {batch_time:.2f}s")
    print(f"Speedup: {speedup:.1f}x")

    # Verify equivalence with seeded training
    batch_accuracies = evaluate_accuracy(model_batch, testloader, device, is_batch=True)
    sequential_accuracies = evaluate_accuracy(
        sequential_models, testloader, device, is_batch=False
    )

    # Memory usage
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
    print("ModelBatch CIFAR10 LeNet Benchmark")

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

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("-" * 60)
    print(f"{'Models':<8} {'Speedup':<10}")
    print("-" * 30)

    for r in results:
        print(f"{r['num_models']:<8} {r['speedup']:<10.1f}")

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE!")
