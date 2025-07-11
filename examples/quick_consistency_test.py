#!/usr/bin/env python3
"""
Quick consistency test - runs extremely fast for rapid debugging.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modelbatch import ModelBatch, OptimizerFactory
from modelbatch.optimizer import create_adam_configs
from modelbatch.utils import create_identical_models, random_init_fn


class TinyMLP(nn.Module):
    """Tiny MLP for quick testing."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 4, num_classes: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x):
        return self.layers(x)


def quick_test():
    """Run a really quick consistency test."""
    print("⚡ Quick Consistency Test")
    print("=" * 50)
    
    # Minimal configuration
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_models = 3
    num_epochs = 3
    learning_rates = [0.001, 0.01, 0.1]
    input_size = 8
    hidden_size = 4
    num_classes = 2
    num_samples = 100
    batch_size = 20
    
    print(f"Models: {num_models}, Epochs: {num_epochs}")
    print(f"Learning rates: {learning_rates}")
    print(f"Architecture: {input_size}→{hidden_size}→{num_classes}")
    print(f"Data: {num_samples} samples")
    
    # Create simple synthetic data
    X = torch.randn(num_samples, input_size)
    # Create clear patterns for each class
    X[:num_samples//2, 0] = 2.0  # Class 0 pattern
    X[num_samples//2:, 1] = 2.0  # Class 1 pattern
    y = torch.cat([
        torch.zeros(num_samples//2, dtype=torch.long),
        torch.ones(num_samples//2, dtype=torch.long)
    ])
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Sequential training
    print("\n📊 Sequential Training")
    models_seq = create_identical_models(TinyMLP, {}, num_models, random_init_fn)
    seq_accuracies = []
    
    for i, (model, lr) in enumerate(zip(models_seq, learning_rates)):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        
        correct = 0
        total = 0
        
        for epoch in range(num_epochs):
            for data, target in dataloader:
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        seq_accuracies.append(accuracy)
        print(f"  Model {i}: {accuracy:.1f}%")
    
    # ModelBatch training
    print("\n⚡ ModelBatch Training")
    models_batch = create_identical_models(TinyMLP, {}, num_models, random_init_fn)
    model_batch = ModelBatch(models_batch, shared_input=True)
    
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer_configs = create_adam_configs(learning_rates)
    optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)
    
    model_batch.train()
    correct_per_model = torch.zeros(num_models)
    total = 0
    
    for epoch in range(num_epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            outputs = model_batch(data)
            loss = model_batch.compute_loss(outputs, target, F.cross_entropy)
            loss.backward()
            optimizer.step()
            
            pred = outputs.argmax(dim=2)
            for model_idx in range(num_models):
                correct_per_model[model_idx] += pred[model_idx].eq(target).sum().item()
            total += target.size(0)
    
    mb_accuracies = [100.0 * correct / total for correct in correct_per_model]
    for i, acc in enumerate(mb_accuracies):
        print(f"  Model {i}: {acc:.1f}%")
    
    # Compare
    print("\n🔍 Comparison")
    max_diff = 0.0
    for i, (seq_acc, mb_acc) in enumerate(zip(seq_accuracies, mb_accuracies)):
        diff = abs(seq_acc - mb_acc)
        print(f"  Model {i}: Seq={seq_acc:.1f}%, MB={mb_acc:.1f}%, Diff={diff:.1f}%")
        max_diff = max(max_diff, diff)
    
    print(f"\nMax difference: {max_diff:.1f}%")
    
    if max_diff > 2.0:
        print("❌ DIVERGENT: Significant difference detected!")
        return False
    else:
        print("✅ CONSISTENT: Results match within tolerance")
        return True


if __name__ == "__main__":
    quick_test() 