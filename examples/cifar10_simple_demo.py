#!/usr/bin/env python3
"""
Simple CIFAR-10 ModelBatch-Optuna Demo.

A working version that demonstrates hyperparameter optimization on CIFAR-10
with realistic search spaces for learning rates and model architectures.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import optuna
except ImportError:
    print("Optuna not available. Install with: pip install optuna")
    sys.exit(1)

from modelbatch import ModelBatch
from modelbatch.optuna_integration import ModelBatchStudy, ConstraintSpec


class SimpleCIFARNet(nn.Module):
    """Simple configurable CNN for CIFAR-10."""
    
    def __init__(
        self,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc1_units: int = 128,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(conv2_channels * 8 * 8, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 10)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_model(params: Dict[str, Any]) -> nn.Module:
    """Factory function to create models based on hyperparameters."""
    model = SimpleCIFARNet(
        conv1_channels=params.get('model.conv1_channels', 32),
        conv2_channels=params.get('model.conv2_channels', 64),
        fc1_units=params.get('model.fc1_units', 128),
        dropout_rate=params.get('model.dropout_rate', 0.5),
    )
    return model


def load_cifar10_data(batch_size: int = 32, num_samples: Optional[int] = None):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    if num_samples is not None:
        indices = torch.randperm(len(trainset))[:num_samples].tolist()
        trainset = torch.utils.data.Subset(trainset, indices)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader


class CIFAR10SimpleStudy(ModelBatchStudy):
    """Simple CIFAR-10 study with hyperparameter search."""
    
    def suggest_parameters(self, trial):
        """Suggest hyperparameters for CIFAR-10 models."""
        # Use fixed architecture for compatibility, vary hyperparameters
        return {
            'model.conv1_channels': 32,  # Fixed
            'model.conv2_channels': 64,  # Fixed
            'model.fc1_units': 128,      # Fixed
            'batch_size': 32,            # Fixed
            'model.dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1),
            'optimizer.lr': trial.suggest_float('lr', 1e-3, 1e-1, log=True),
            'optimizer.weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
        }


def run_simple_cifar10_demo():
    """Run a simple CIFAR-10 hyperparameter optimization demo."""
    print("🚀 Simple CIFAR-10 Hyperparameter Optimization")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load smaller subset for quick demo
    trainloader, testloader = load_cifar10_data(batch_size=32, num_samples=1000)
    print(f"Loaded CIFAR-10: {len(trainloader.dataset)} train, {len(testloader.dataset)} test")
    
    # Define constraints - all models compatible, vary hyperparameters
    constraints = ConstraintSpec(
        fixed_params=[
            'model.conv1_channels',
            'model.conv2_channels',
            'model.fc1_units',
            'batch_size',
        ],  # All models have same architecture
        variable_params=[
            'model.dropout_rate',
            'optimizer.lr',
            'optimizer.weight_decay',
        ]
    )
    
    study = optuna.create_study(direction="maximize")
    
    mb_study = CIFAR10SimpleStudy(
        study=study,
        model_factory=create_model,
        constraint_spec=constraints,
        min_models_per_batch=2,
        max_models_per_batch=3,
    )
    
    # Training function
    def objective_fn(model_batch, configs):
        from modelbatch.optimizer import OptimizerFactory
        
        # Create optimizer
        optimizer_factory = OptimizerFactory(torch.optim.Adam)
        optimizer = optimizer_factory.create_optimizer(model_batch, configs)
        
        # Training loop
        criterion = nn.CrossEntropyLoss()
        model_batch.train()
        
        for epoch in range(2):  # Quick training
            for batch_idx, (data, target) in enumerate(trainloader):
                if batch_idx >= 10:  # Limit batches
                    break
                
                optimizer.zero_grad()
                outputs = model_batch(data)
                loss = model_batch.compute_loss(outputs, target, criterion)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model_batch.eval()
        accuracies = []
        
        with torch.no_grad():
            correct = torch.zeros(model_batch.num_models)
            total = 0
            
            for batch_idx, (data, target) in enumerate(testloader):
                if batch_idx >= 5:  # Limit batches
                    break
                    
                outputs = model_batch(data)
                _, predicted = torch.max(outputs, 2)
                target_expanded = target.unsqueeze(0).expand(model_batch.num_models, -1)
                correct += (predicted == target_expanded).sum(dim=1).float()
                total += target.size(0)
            
            accuracies = (100 * correct / total).tolist()
        
        return accuracies
    
    # Run optimization
    mb_study.optimize(
        objective_fn=objective_fn,
        n_trials=4,
        show_progress_bar=True
    )
    
    # Results
    print(f"\n📊 Results:")
    print(f"Total trials: {len(study.trials)}")
    
    if study.best_trial:
        print(f"Best accuracy: {study.best_trial.value:.2f}%")
        print("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
    
    return study, mb_study


if __name__ == "__main__":
    print("🧠 Simple CIFAR-10 ModelBatch-Optuna Demo")
    print("Searches for optimal CNN architectures and learning rates")
    print()
    
    run_simple_cifar10_demo()
    print("\n🎉 Demo completed!")