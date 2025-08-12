"""Shared utilities for benchmark scripts.

Provides common helpers for seeding, evaluation, and training both
sequentially and with ModelBatch.
"""

from __future__ import annotations

import random
import time
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modelbatch import ModelBatch, OptimizerFactory
from modelbatch.optimizer import create_adam_configs
from modelbatch.utils import count_parameters


def set_random_seeds(seed: int = 6235) -> None:
    """Set random seeds for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_accuracy(
    models_or_batch,
    dataloader: DataLoader,
    device: torch.device,
    *,
    is_batch: bool = False,
) -> list[float]:
    """Evaluate accuracy for either a `ModelBatch` or a list of models.

    Assumes classification with logits over classes and integer labels.
    """

    if is_batch:
        models_or_batch.eval()
        correct = torch.zeros(models_or_batch.num_models, device=device)
        total = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in dataloader:
                inputs = batch_inputs.to(device)
                labels = batch_labels.to(device)
                logits = models_or_batch(inputs)  # [num_models, batch, num_classes]
                preds = logits.argmax(dim=2)
                labels_expanded = labels.unsqueeze(0).expand(models_or_batch.num_models, -1)
                correct += (preds == labels_expanded).sum(dim=1).float()
                total += labels.size(0)
        return (100 * correct / max(total, 1)).detach().cpu().tolist()

    accuracies: list[float] = []
    for model in models_or_batch:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in dataloader:
                inputs = batch_inputs.to(device)
                labels = batch_labels.to(device)
                logits = model(inputs)  # [batch, num_classes]
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracies.append(100.0 * correct / max(total, 1))
    return accuracies


def train_sequential(
    models: Iterable[torch.nn.Module],
    trainloader: DataLoader,
    num_epochs: int,
    learning_rates: Sequence[float],
    device: torch.device,
) -> float:
    """Train models sequentially (baseline)."""
    print("Sequential Training")
    start_time = time.time()

    for model, lr in zip(models, learning_rates):
        set_random_seeds()
        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _epoch in range(num_epochs):
            for batch_inputs, batch_labels in trainloader:
                inputs = batch_inputs.to(device)
                labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

    total_time = time.time() - start_time
    print(f"Sequential time: {total_time:.2f}s")
    return total_time


def train_modelbatch(
    models: Sequence[torch.nn.Module],
    trainloader: DataLoader,
    num_epochs: int,
    learning_rates: Sequence[float],
    device: torch.device,
) -> tuple[float, ModelBatch]:
    """Train models using ModelBatch (vectorized)."""
    print("ModelBatch Training")
    set_random_seeds()

    model_batch = ModelBatch(models, shared_input=True).to(device)
    param_info = count_parameters(model_batch)
    print(f"Total parameters: {param_info['total_params']:,} ({model_batch.num_models} models)")

    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer_configs = create_adam_configs(learning_rates)
    optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)

    start_time = time.time()
    for _epoch in range(num_epochs):
        model_batch.train()
        for batch_inputs, batch_labels in trainloader:
            inputs = batch_inputs.to(device)
            labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model_batch(inputs)
            loss = model_batch.compute_loss(logits, labels, F.cross_entropy)
            loss.backward()
            optimizer.step()

    total_time = time.time() - start_time
    print(f"ModelBatch time: {total_time:.2f}s")
    return total_time, model_batch


