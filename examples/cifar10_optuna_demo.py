#!/usr/bin/env python3
"""
CIFAR-10 ModelBatch-Optuna Hyperparameter Optimization Demo.

This demo shows how to use ModelBatch with Optuna for comprehensive
hyperparameter search on CIFAR-10, including model architecture,
learning rates, and training hyperparameters.
"""

import os
import sys
import time
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

from modelbatch import ModelBatch, OptimizerFactory
from modelbatch.optuna_integration import ModelBatchStudy, ConstraintSpec


class CIFAR10CNN(nn.Module):
    """Configurable CNN for CIFAR-10 with variable architecture."""
    
    def __init__(
        self,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc1_units: int = 128,
        dropout_rate: float = 0.5,
        num_conv_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.fc1_units = fc1_units
        self.dropout_rate = dropout_rate
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        
        # Calculate feature map sizes
        conv_output_size = 32
        for _ in range(num_conv_layers):
            conv_output_size = conv_output_size // 2
        
        final_conv_channels = conv2_channels if num_conv_layers >= 2 else conv1_channels
        
        # Build convolutional layers
        layers = []
        in_channels = 3
        
        # First conv layer
        layers.append(nn.Conv2d(in_channels, conv1_channels, kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        
        # Additional conv layers
        current_channels = conv1_channels
        for i in range(1, num_conv_layers):
            next_channels = conv2_channels if i == 1 else min(conv2_channels * 2, 256)
            layers.append(nn.Conv2d(current_channels, next_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            current_channels = next_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Fully connected layers
        self.fc_input_size = current_channels * conv_output_size * conv_output_size
        self.fc1 = nn.Linear(self.fc_input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 10)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNetBlock(nn.Module):
    """Simple ResNet-like block for CIFAR-10."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFAR10ResNet(nn.Module):
    """Configurable ResNet-like architecture for CIFAR-10."""
    
    def __init__(
        self,
        num_blocks: int = 2,
        base_channels: int = 16,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(3, base_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Build blocks
        layers = []
        in_channels = base_channels
        
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            stride = 2 if i > 0 else 1
            layers.append(ResNetBlock(in_channels, out_channels, stride))
            layers.append(ResNetBlock(out_channels, out_channels, 1))
            in_channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, 10)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def create_model(params: Dict[str, Any]) -> nn.Module:
    """Factory function to create models based on hyperparameters."""
    model_type = params.get('model.type', 'cnn')
    
    if model_type == 'cnn':
        model = CIFAR10CNN(
            conv1_channels=params.get('model.conv1_channels', 32),
            conv2_channels=params.get('model.conv2_channels', 64),
            fc1_units=params.get('model.fc1_units', 128),
            dropout_rate=params.get('model.dropout_rate', 0.5),
            num_conv_layers=params.get('model.num_conv_layers', 2),
            kernel_size=params.get('model.kernel_size', 3),
        )
    elif model_type == 'resnet':
        model = CIFAR10ResNet(
            num_blocks=params.get('model.num_blocks', 2),
            base_channels=params.get('model.base_channels', 16),
            dropout_rate=params.get('model.dropout_rate', 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def load_cifar10_data(batch_size: int = 128, num_samples: Optional[int] = None):
    """Load CIFAR-10 dataset with data augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    if num_samples is not None:
        # Use subset for faster experimentation
        indices = torch.randperm(len(trainset))[:num_samples].tolist()
        trainset = torch.utils.data.Subset(trainset, indices)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def train_objective(
    model_batch: ModelBatch,
    configs: List[Dict[str, Any]],
    trainloader: DataLoader,
    testloader: DataLoader,
    num_epochs: int = 3,
    device: torch.device = torch.device('cpu'),
) -> List[float]:
    """Training objective function for ModelBatch."""
    model_batch.train()
    
    # Create optimizer with individual configurations
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer = optimizer_factory.create_optimizer(model_batch, configs)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model_batch(data)
            loss = model_batch.compute_loss(outputs, target, criterion)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Limit batches for faster demo
            if batch_idx >= 50:  # ~1 epoch on subset
                break
    
    # Evaluate on test set
    model_batch.eval()
    accuracies = []
    
    with torch.no_grad():
        correct = torch.zeros(model_batch.num_models).to(device)
        total = 0
        
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model_batch(data)
            _, predicted = torch.max(outputs, 2)
            target_expanded = target.unsqueeze(0).expand(model_batch.num_models, -1)
            correct += (predicted == target_expanded).sum(dim=1).float()
            total += target.size(0)
            
            # Limit test batches
            if total >= 1000:
                break
    
    accuracies = (100 * correct / total).cpu().tolist()
    return accuracies


class CIFAR10OptunaStudy(ModelBatchStudy):
    """Optuna study for CIFAR-10 with comprehensive hyperparameter search."""
    
    def suggest_parameters(self, trial):
        """Suggest comprehensive hyperparameters for CIFAR-10 models."""
        
        # Model architecture choice
        model_type = trial.suggest_categorical('model.type', ['cnn', 'resnet'])
        
        params = {'model.type': model_type}
        
        if model_type == 'cnn':
            # CNN-specific parameters
            params.update({
                'model.conv1_channels': trial.suggest_int('conv1_channels', 16, 128, step=16),
                'model.conv2_channels': trial.suggest_int('conv2_channels', 32, 256, step=32),
                'model.fc1_units': trial.suggest_int('fc1_units', 64, 512, step=64),
                'model.num_conv_layers': trial.suggest_int('num_conv_layers', 2, 4),
                'model.kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
            })
        elif model_type == 'resnet':
            # ResNet-specific parameters
            params.update({
                'model.num_blocks': trial.suggest_int('num_blocks', 1, 3),
                'model.base_channels': trial.suggest_int('base_channels', 8, 64, step=8),
            })
        
        # Common parameters
        params.update({
            'model.dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.7, step=0.1),
            'optimizer.lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'optimizer.weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'data.batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        })
        
        return params


def run_cifar10_optuna_demo():
    """Run CIFAR-10 hyperparameter optimization demo."""
    print("🚀 CIFAR-10 ModelBatch-Optuna Hyperparameter Optimization")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data with smaller subset for demo
    trainloader, testloader = load_cifar10_data(batch_size=64, num_samples=5000)
    print(f"Loaded CIFAR-10: {len(trainloader.dataset)} train, {len(testloader.dataset)} test")
    
    # Define constraints for batching
    # Group by model type and architecture parameters
    constraints = ConstraintSpec(
        fixed_params=[
            'model.type',           # CNN vs ResNet
            'model.conv1_channels', # Architecture compatibility
            'model.conv2_channels',
            'model.fc1_units',
            'model.num_conv_layers',
            'model.kernel_size',
            'model.num_blocks',
            'model.base_channels',
            'data.batch_size',      # Batch size for efficiency
        ],
        variable_params=[
            'model.dropout_rate',
            'optimizer.lr',
            'optimizer.weight_decay',
        ]
    )
    
    # Create study
    study = optuna.create_study(
        study_name="cifar10_hyperopt",
        direction="maximize",  # Maximize accuracy
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    # Create ModelBatch study
    mb_study = CIFAR10OptunaStudy(
        study=study,
        model_factory=create_model,
        constraint_spec=constraints,
        min_models_per_batch=2,
        max_models_per_batch=4,
    )
    
    # Run optimization with smaller subset for faster demo
    small_trainloader, small_testloader = load_cifar10_data(batch_size=32, num_samples=2000)
    print("\n🔍 Running hyperparameter optimization...")
    
    def objective_fn(model_batch, configs):
        return train_objective(
            model_batch=model_batch,
            configs=configs,
            trainloader=small_trainloader,
            testloader=small_testloader,
            num_epochs=2,
            device=device,
        )
    
    start_time = time.time()
    mb_study.optimize(
        objective_fn=objective_fn,
        n_trials=2,  # Very small number for quick demo
        show_progress_bar=True
    )
    optimization_time = time.time() - start_time
    
    # Results
    print(f"\n📊 Optimization Results")
    print("-" * 50)
    print(f"Total time: {optimization_time:.2f}s")
    
    summary = mb_study.get_optimization_summary()
    print(f"Total trials: {summary['total_trials']}")
    print(f"Completed trials: {summary['completed_trials']}")
    print(f"Failed trials: {summary['failed_trials']}")
    print(f"Batch groups: {summary['total_groups']}")
    
    # Best parameters
    print(f"\n🏆 Best Configuration:")
    best_trial = study.best_trial
    print(f"Best accuracy: {best_trial.value:.2f}%")
    
    print("\n📋 Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Top 5 trials
    print(f"\n📈 Top 5 Configurations:")
    for i, trial in enumerate(study.trials[:5]):
        acc_str = f"{trial.value:.2f}%" if trial.value is not None else "None"
        print(f"  {i+1}. Accuracy: {acc_str}")
        print(f"     Model: {trial.params.get('model.type', 'unknown')}")
        print(f"     LR: {trial.params.get('lr', 'unknown'):.4f}")
        print(f"     Dropout: {trial.params.get('dropout_rate', 'unknown')}")
    
    return study, mb_study


def run_advanced_search():
    """Run advanced search with more complex constraints."""
    print(f"\n{'='*70}")
    print("🔬 Advanced CIFAR-10 Search")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, testloader = load_cifar10_data(batch_size=128, num_samples=10000)
    
    # More complex constraints allowing architecture variation
    constraints = ConstraintSpec(
        fixed_params=[
            'model.type',
            'data.batch_size',  # Still group by batch size
        ],
        variable_params=[
            'model.conv1_channels',
            'model.conv2_channels',
            'model.fc1_units',
            'model.num_conv_layers',
            'model.kernel_size',
            'model.num_blocks',
            'model.base_channels',
            'model.dropout_rate',
            'optimizer.lr',
            'optimizer.weight_decay',
        ]
    )
    
    study = optuna.create_study(direction="maximize")
    
    class AdvancedStudy(ModelBatchStudy):
        def suggest_parameters(self, trial):
            model_type = trial.suggest_categorical('model.type', ['cnn', 'resnet'])
            
            params = {'model.type': model_type, 'data.batch_size': 64}
            
            if model_type == 'cnn':
                params.update({
                    'model.conv1_channels': trial.suggest_int('conv1_channels', 32, 128, step=32),
                    'model.conv2_channels': trial.suggest_int('conv2_channels', 64, 256, step=64),
                    'model.fc1_units': trial.suggest_int('fc1_units', 128, 512, step=128),
                    'model.num_conv_layers': trial.suggest_int('num_conv_layers', 2, 3),
                    'model.kernel_size': trial.suggest_int('kernel_size', 3, 5, step=2),
                })
            else:
                params.update({
                    'model.num_blocks': trial.suggest_int('num_blocks', 2, 4),
                    'model.base_channels': trial.suggest_int('base_channels', 16, 64, step=16),
                })
            
            params.update({
                'model.dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.6, step=0.1),
                'optimizer.lr': trial.suggest_loguniform('lr', 5e-4, 5e-2),
                'optimizer.weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-2),
            })
            
            return params
    
    mb_study = AdvancedStudy(
        study=study,
        model_factory=create_model,
        constraint_spec=constraints,
        min_models_per_batch=2,
        max_models_per_batch=3,
    )
    
    def objective_fn(model_batch, configs):
        return train_objective(
            model_batch=model_batch,
            configs=configs,
            trainloader=trainloader,
            testloader=testloader,
            num_epochs=2,
            device=device,
        )
    
    mb_study.optimize(
        objective_fn=objective_fn,
        n_trials=6,
        show_progress_bar=True
    )
    
    print(f"\n📊 Advanced Results:")
    print(f"Total trials: {len(study.trials)}")
    
    if study.best_trial:
        print(f"Best accuracy: {study.best_trial.value:.2f}%")
        print("Best parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
    
    return study


if __name__ == "__main__":
    print("🧠 CIFAR-10 ModelBatch-Optuna Hyperparameter Optimization")
    print("This demo searches for optimal:")
    print("• Learning rates and weight decay")
    print("• Model architectures (CNN vs ResNet)")
    print("• Network sizes and layer configurations")
    print("• Dropout rates and other regularization")
    print("• Batch sizes for training efficiency")
    print()
    
    # Run basic search
    study, mb_study = run_cifar10_optuna_demo()
    
    # Run advanced search
    advanced_study = run_advanced_search()
    
    print(f"\n🎉 Demo completed!")
    print("Check the results above for optimal CIFAR-10 configurations.")