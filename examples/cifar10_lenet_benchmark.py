#!/usr/bin/env python3
"""CIFAR10 LeNet Benchmark: Train multiple LeNet models simultaneously."""

from __future__ import annotations

import copy
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch import ModelBatch, OptimizerFactory
from modelbatch.optimizer import create_adam_configs
from modelbatch.utils import count_parameters


def set_random_seeds(seed: int = 6235):
    """Set random seeds for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.default_rng(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LeNet5_CIFAR(nn.Module):
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
        np.random.default_rng(worker_seed)
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


def evaluate_accuracy(
    models, dataloader: DataLoader, device: torch.device, *, is_batch: bool = False
) -> list[float]:
    """Evaluate model accuracy - works for both single models and ModelBatch."""
    if is_batch:
        models.eval()
        correct = torch.zeros(models.num_models).to(device)
        total = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)  # noqa: PLW2901
                outputs = models(data)
                _, predicted = torch.max(outputs, 2)
                target_expanded = target.unsqueeze(0).expand(models.num_models, -1)
                correct += (predicted == target_expanded).sum(dim=1).float()
                total += target.size(0)

        return (100 * correct / total).cpu().tolist()
    accuracies = []
    for model in models:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)  # noqa: PLW2901
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracies.append(100 * correct / total)
    return accuracies


def verify_equivalence(
    sequential_accuracies: list[float],
    batch_accuracies: list[float],
    learning_rates: list[float],
    dropout_rates: list[float],
) -> bool:
    """Verify that models with same parameters achieve nearly identical performance."""
    print("\n🔍 EQUIVALENCE VERIFICATION")
    print("-" * 50)

    # Calculate all differences
    differences = []
    for i in range(len(sequential_accuracies)):
        diff = sequential_accuracies[i] - batch_accuracies[i]
        differences.append(diff)

        print(
            f"Model {i}: Sequential={sequential_accuracies[i]:.2f}%, "
            f"ModelBatch={batch_accuracies[i]:.2f}%, Diff={diff:.2f}%"
        )

    # Calculate quartiles
    differences_sorted = sorted(differences)
    n = len(differences_sorted)
    q1_idx = n // 4
    q2_idx = n // 2
    q3_idx = 3 * n // 4

    q1 = differences_sorted[q1_idx]
    q2 = differences_sorted[q2_idx]
    q3 = differences_sorted[q3_idx]
    min_diff = differences_sorted[0]
    max_diff = differences_sorted[-1]

    print("Difference quartiles:")
    print(f" {'Min':>8} | {'Q1':>8} | {'Q2':>8} | {'Q3':>8} | {'Max':>8}")
    print("-" * 56)
    print(f"{min_diff:8.2f}% |{q1:8.2f}% |{q2:8.2f}% |{q3:8.2f}% |{max_diff:8.2f}%")

    # With fixed seeds, expect very close results
    tolerance = 0.5  # 0.5% tolerance for seeded training
    return abs(max_diff) < tolerance and abs(min_diff) < tolerance


def train_sequential(
    models,
    trainloader: DataLoader,
    num_epochs: int,
    learning_rates: list[float],
    device: torch.device,
) -> float:
    """Train models sequentially - one model completely, then the next."""
    print("📊 Sequential Training")
    start_time = time.time()

    for _model_idx, (model, lr) in enumerate(zip(models, learning_rates)):
        set_random_seeds()
        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _epoch in range(num_epochs):
            for data, target in trainloader:
                data, target = data.to(device), target.to(device)  # noqa: PLW2901
                optimizer.zero_grad()
                loss = F.cross_entropy(model(data), target)
                loss.backward()
                optimizer.step()

    total_time = time.time() - start_time
    print(f"Sequential time: {total_time:.2f}s")
    return total_time


def train_modelbatch(
    models,
    trainloader: DataLoader,
    num_epochs: int,
    learning_rates: list[float],
    device: torch.device,
) -> Tuple[float, ModelBatch]:
    """Train models using ModelBatch."""
    print("⚡ ModelBatch Training")
    set_random_seeds()

    model_batch = ModelBatch(models, shared_input=True).to(device)
    param_info = count_parameters(model_batch)
    print(
        f"Total parameters: {param_info['total_params']:,} ({model_batch.num_models} models)"
    )

    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer_configs = create_adam_configs(learning_rates)
    optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)

    start_time = time.time()

    for _epoch in range(num_epochs):
        model_batch.train()
        for _batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model_batch(data)
            loss = model_batch.compute_loss(outputs, target, F.cross_entropy)
            loss.backward()
            optimizer.step()

    total_time = time.time() - start_time
    print(f"ModelBatch time: {total_time:.2f}s")
    return total_time, model_batch


def run_benchmark(
    num_models: int = 16,
    num_epochs: int = 5,
    batch_size: int = 128,
    num_samples: int = 60000,
) -> Dict[str, float]:
    """Run CIFAR10 LeNet benchmark."""
    print(f"🚀 CIFAR10 LeNet Benchmark: {num_models} Models")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load data
    trainloader, testloader = load_cifar10_data(
        batch_size=batch_size, num_samples=num_samples
    )
    print(
        f"Training samples: {len(trainloader.dataset)}, Test samples: {len(testloader.dataset)}"
    )  # type: ignore

    # Create hyperparameter variations
    dropout_rates = [0.1 + 0.02 * i for i in range(num_models)]
    learning_rates = [0.01 * (0.5**i) for i in range(num_models)]
    print(f"Dropout range: {min(dropout_rates):.3f}-{max(dropout_rates):.3f}")
    print(f"Learning rate range: {min(learning_rates):.6f}-{max(learning_rates):.6f}")

    # Create models with deterministic initialization
    set_random_seeds()
    models = [LeNet5_CIFAR(dropout_rate=dropout_rates[i]) for i in range(num_models)]
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

    print("\n🏆 RESULTS")
    print("-" * 30)
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"ModelBatch: {batch_time:.2f}s")
    print(f"Speedup: {speedup:.1f}x")

    # Verify equivalence with seeded training
    batch_accuracies = evaluate_accuracy(model_batch, testloader, device, is_batch=True)
    sequential_accuracies = evaluate_accuracy(
        sequential_models, testloader, device, is_batch=False
    )

    equivalent = verify_equivalence(
        sequential_accuracies, batch_accuracies, learning_rates, dropout_rates
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
        "equivalent": equivalent,
    }


def scalability_study():
    """Run scalability study with different numbers of models."""
    print(f"\n{'=' * 60}")
    print("🔬 SCALABILITY STUDY")
    print("=" * 60)

    configs = [
        {"num_models": 4, "num_epochs": 5},
        {"num_models": 8, "num_epochs": 5},
        {"num_models": 16, "num_epochs": 5},
        {"num_models": 32, "num_epochs": 5},
    ]

    results = []
    for config in configs:
        print(f"\nTesting {config['num_models']} models...")
        try:
            result = run_benchmark(**config)
            results.append(result)
            equiv_status = "✅ EQUIVALENT" if result["equivalent"] else "⚠️ DIVERGENT"
            print(f"✅ {result['speedup']:.1f}x speedup, {equiv_status}")
        except Exception as e:
            print(f"❌ Failed: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("📋 SUMMARY")
    print("-" * 60)
    print(f"{'Models':<8} {'Speedup':<10} {'Equivalent':<12}")
    print("-" * 30)

    for r in results:
        equiv_mark = "✅" if r["equivalent"] else "⚠️"
        print(f"{r['num_models']:<8} {r['speedup']:<10.1f}x {equiv_mark:<12}")

    if results:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        equivalent_count = sum(1 for r in results if r["equivalent"])
        print(f"\nAverage speedup: {avg_speedup:.1f}x")
        print(f"Equivalent results: {equivalent_count}/{len(results)}")


if __name__ == "__main__":
    print("🧠 ModelBatch CIFAR10 LeNet Benchmark")

    # Scalability study
    scalability_study()

    print(f"\n{'=' * 60}")
    print("🎉 BENCHMARK COMPLETE!")
