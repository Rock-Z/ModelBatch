"""CIFAR10 LeNet Benchmark: Train multiple LeNet models simultaneously."""

from __future__ import annotations

import copy
from pathlib import Path
import random
import sys
from typing import Sized, cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    set_random_seeds,
    evaluate_accuracy,
    train_single_model,
    train_modelbatch,
)


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
    batch_size: int = 256,
    num_samples: int | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
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
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    trainloader = DataLoader(
        trainset,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        **loader_kwargs,
    )
    testloader = DataLoader(testset, shuffle=False, **loader_kwargs)
    return trainloader, testloader


if __name__ == "__main__":
    print("ModelBatch CIFAR10 LeNet Benchmark")

    print(f"\n{'=' * 60}")
    print("SCALABILITY STUDY")
    print("=" * 60)

    configs = [
        {"num_models": 4},
        {"num_models": 8},
        {"num_models": 16},
        {"num_models": 32},
    ]
    num_epochs = 30
    batch_size = 256
    num_samples = 60000
    max_num_models = max(config["num_models"] for config in configs)

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
    print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Tutorial-style CIFAR recipe: crop/flip/normalize + SGD momentum + weight decay.
    dropout_choices = [0.05, 0.10, 0.00, 0.15]
    learning_rate = 0.05
    dropout_rates = [
        dropout_choices[i % len(dropout_choices)] for i in range(max_num_models)
    ]
    print(f"Dropout range: {min(dropout_rates):.3f}-{max(dropout_rates):.3f}")
    print(f"Learning rate: {learning_rate:.6f}")
    print("SGD: momentum=0.9, weight_decay=5e-4, milestones=60%/83%")

    # Create models with deterministic initialization
    set_random_seeds()
    models = [
        LeNet5CIFAR(dropout_rate=dropout_rates[i]) for i in range(max_num_models)
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
        optimizer_cls=torch.optim.SGD,
        optimizer_config={"momentum": 0.9, "weight_decay": 5e-4, "nesterov": True},
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[num_epochs * 3 // 5, num_epochs * 5 // 6],
            gamma=0.1,
        ),
    )
    sequential_accuracy = evaluate_accuracy(
        [sequential_model], testloader, device, is_batch=False
    )[0]
    print(f"Sequential accuracy: {sequential_accuracy:.1f}%")

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
            optimizer_cls=torch.optim.SGD,
            optimizer_configs=[
                {
                    "lr": learning_rate,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                    "nesterov": True,
                }
                for _ in range(num_models)
            ],
            scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[num_epochs * 3 // 5, num_epochs * 5 // 6],
                gamma=0.1,
            ),
        )

        sequential_time = sequential_time_per_model * num_models
        speedup = sequential_time / batch_time

        print("\nRESULTS")
        print("-" * 30)
        print(
            f"Sequential: {sequential_time:.2f}s "
            f"({sequential_time_per_model:.2f}s/model x {num_models})"
        )
        print(f"ModelBatch: {batch_time:.2f}s")
        print(f"Speedup: {speedup:.1f}x")
        print(f"Sequential accuracy: {sequential_accuracy:.1f}%")

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
            "sequential_accuracy": sequential_accuracy,
            "best_batch_accuracy": best_accuracy,
        }
        results.append(result)
        print(f"{speedup:.1f}x speedup")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("-" * 60)
    print(f"{'Models':<8} {'Speedup':<10} {'Best acc':<10}")
    print("-" * 30)

    for r in results:
        print(
            f"{r['num_models']:<8} {r['speedup']:<10.1f} {r['best_batch_accuracy']:<10.1f}"
        )

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE!")
