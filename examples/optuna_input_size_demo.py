#!/usr/bin/env python3
"""
Extended Optuna integration demo with different model input sizes.

This example demonstrates how ModelBatch-Optuna integration can handle
hyperparameter optimization across different model input sizes by automatically
grouping compatible configurations (same input size) together.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import optuna
except ImportError:
    print("Optuna not available. Install with: pip install optuna")
    sys.exit(1)

from modelbatch import ModelBatch
from modelbatch.optuna_integration import ModelBatchStudy, ConstraintSpec


class FlexibleMLP(nn.Module):
    """MLP that can handle different input sizes."""
    
    def __init__(
        self, 
        input_size: int = 10, 
        hidden_size: int = 20, 
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        output_size: int = 1
    ):
        super().__init__()
        self.input_size = input_size
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def create_model(params: Dict[str, Any]) -> nn.Module:
    """Factory function for creating models based on parameters."""
    return FlexibleMLP(
        input_size=params.get('model.input_size', 10),
        hidden_size=params.get('model.hidden_size', 20),
        num_layers=params.get('model.num_layers', 2),
        dropout_rate=params.get('model.dropout_rate', 0.1),
        output_size=params.get('model.output_size', 1),
    )


def generate_dummy_data_for_input_size(
    input_size: int, 
    batch_size: int = 32, 
    num_samples: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy regression data for a specific input size."""
    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, 1)
    return X, y


def train_objective_with_input_size(
    model_batch: ModelBatch,
    configs: List[Dict[str, Any]],
    input_size: int,
    num_epochs: int = 5
) -> List[float]:
    """
    Training objective function for ModelBatch with input size awareness.
    
    Returns list of validation losses for each model/trial.
    """
    from modelbatch.optimizer import OptimizerFactory
    
    # Generate data for this specific input size
    X, y = generate_dummy_data_for_input_size(input_size)
    
    device = next(model_batch.parameters()).device
    X, y = X.to(device), y.to(device)
    
    # Create optimizer with variable configurations
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer = optimizer_factory.create_optimizer(model_batch, configs)
    
    # Simple training loop
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model_batch.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model_batch(X)
        
        # Compute loss for each model
        losses = []
        for i in range(model_batch.num_models):
            loss = criterion(outputs[i], y)
            losses.append(loss)
        
        # Combine losses (sum for gradient computation)
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        optimizer.step()
    
    # Return validation losses
    model_batch.eval()
    with torch.no_grad():
        outputs = model_batch(X)
        losses = []
        for i in range(model_batch.num_models):
            loss = criterion(outputs[i], y)
            losses.append(loss.item())
    
    return losses


class InputSizeStudy(ModelBatchStudy):
    """Study that searches across different input sizes."""
    
    def suggest_parameters(self, trial):
        """Suggest hyperparameters including different input sizes."""
        return {
            'model.input_size': trial.suggest_categorical('input_size', [5, 10, 15, 20]),
            'model.hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64]),
            'model.num_layers': trial.suggest_int('num_layers', 1, 3),
            'model.dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1),
            'model.output_size': 1,
            'optimizer.lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'optimizer.weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        }


def run_input_size_demo():
    """Run the input size optimization demo."""
    print("🚀 ModelBatch-Optuna Input Size Demo")
    print("=" * 50)
    print("This demo searches across different model input sizes")
    print("while automatically grouping compatible configurations.")
    print()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define constraints - group by all model architecture parameters for ModelBatch compatibility
    print("\n📋 Setting up constraints...")
    constraints = ConstraintSpec(
        fixed_params=[
            'model.input_size',    # Must be same for all models in batch
            'model.hidden_size',   # Must be same for all models in batch
            'model.num_layers',    # Must be same for all models in batch
            'model.dropout_rate',  # Must be same for all models in batch
            'model.output_size',   # Must be same for all models in batch
        ],
        variable_params=[
            'optimizer.lr',
            'optimizer.weight_decay',
        ],
        batch_aware_params=[
            'model.input_size',
            'model.hidden_size',
            'model.num_layers',
            'model.dropout_rate',
            'model.output_size',
        ],
    )
    
    # Create study
    print("\n🔍 Creating Optuna study...")
    study = optuna.create_study(
        study_name="input_size_optimization",
        direction="minimize",  # Minimize validation loss
    )
    
    # Create ModelBatch study
    print("\n⚡ Setting up ModelBatch study...")
    mb_study = InputSizeStudy(
        study=study,
        model_factory=create_model,
        constraint_spec=constraints,
        min_models_per_batch=2,
        max_models_per_batch=4,
        batch_timeout=30.0,
    )
    
    # Run optimization
    print("\n🏃 Running input size optimization...")
    
    def objective_fn(model_batch, configs, constraint_context=None):
        # Extract input size from constraint parameters
        if constraint_context is not None:
            input_size = constraint_context['constraint_params'].get('model.input_size', 10)
        else:
            # Fallback for backward compatibility
            input_size = 10
        
        return train_objective_with_input_size(
            model_batch=model_batch,
            configs=configs,
            input_size=input_size,
            num_epochs=3
        )
    
    mb_study.optimize(
        objective_fn=objective_fn,
        n_trials=12,
        show_progress_bar=True
    )
    
    # Print results
    print("\n📊 Optimization Results")
    print("-" * 30)
    
    summary = mb_study.get_optimization_summary()
    print(f"Total trials: {summary['total_trials']}")
    print(f"Completed trials: {summary['completed_trials']}")
    print(f"Batch groups: {summary['total_groups']}")
    
    # Analyze results by input size
    print("\n📈 Results by Input Size:")
    input_size_results = {}
    for trial in study.trials:
        if trial.value is not None:
            input_size = trial.params.get('input_size', 10)
            if input_size not in input_size_results:
                input_size_results[input_size] = []
            input_size_results[input_size].append(trial.value)
    
    for input_size, losses in sorted(input_size_results.items()):
        avg_loss = np.mean(losses)
        min_loss = min(losses)
        count = len(losses)
        print(f"  Input size {input_size}: {count} trials, avg loss={avg_loss:.4f}, best={min_loss:.4f}")
    
    # Best parameters
    print("\n🏆 Best Parameters:")
    best_trial = study.best_trial
    print(f"Best validation loss: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    return study


def run_advanced_input_size_demo():
    """Run advanced demo with more complex input size handling."""
    print("\n" + "=" * 50)
    print("🔧 Advanced Input Size Demo")
    print("=" * 50)
    print("Advanced demo with dynamic data generation per input size")
    
    # More sophisticated constraint handling
    constraints = ConstraintSpec(
        fixed_params=[
            'model.input_size',    # Same input size for batch compatibility
            'model.hidden_size',   # Same hidden size for batch compatibility
            'model.num_layers',    # Same number of layers for batch compatibility
            'model.dropout_rate',  # Same dropout rate for batch compatibility
            'model.output_size',   # Same output size for batch compatibility
        ],
        variable_params=[
            'optimizer.lr',
            'optimizer.weight_decay',
            'optimizer.betas_0',
            'optimizer.betas_1',
        ],
        batch_aware_params=[
            'model.input_size',
            'model.hidden_size',
            'model.num_layers',
            'model.dropout_rate',
            'model.output_size',
        ],
    )
    
    # Create study
    study = optuna.create_study(direction="minimize")
    
    class AdvancedInputSizeStudy(ModelBatchStudy):
        def suggest_parameters(self, trial):
            return {
                'model.input_size': trial.suggest_categorical('input_size', [3, 5, 8, 13, 21]),  # Fibonacci sizes
                'model.hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128]),
                'model.num_layers': trial.suggest_int('num_layers', 1, 4),
                'model.dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.6, step=0.1),
                'model.output_size': 1,
                'optimizer.lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
                'optimizer.weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'optimizer.betas_0': trial.suggest_float('beta1', 0.8, 0.99),
                'optimizer.betas_1': trial.suggest_float('beta2', 0.9, 0.999),
            }
    
    mb_study = AdvancedInputSizeStudy(
        study=study,
        model_factory=create_model,
        constraint_spec=constraints,
        min_models_per_batch=3,
        max_models_per_batch=5,
        batch_timeout=45.0,
    )
    
    # Enhanced objective function with input size tracking
    def objective_fn(model_batch, configs, constraint_context=None):
        # Extract input size from constraint parameters
        if constraint_context is not None:
            input_size = constraint_context['constraint_params'].get('model.input_size', 10)
        else:
            # Fallback for backward compatibility
            input_size = 10
        
        return train_objective_with_input_size(
            model_batch=model_batch,
            configs=configs,
            input_size=input_size,
            num_epochs=2  # Shorter for demo
        )
    
    # Run optimization
    mb_study.optimize(
        objective_fn=objective_fn,
        n_trials=15,
        show_progress_bar=True
    )
    
    # Advanced analysis
    print("\n📊 Advanced Results:")
    print(f"Total trials: {len(study.trials)}")
    
    # Group by input size and show performance
    results_by_input_size = {}
    for trial in study.trials:
        if trial.value is not None:
            params = trial.params
            input_size = params.get('input_size', 10)
            
            if input_size not in results_by_input_size:
                results_by_input_size[input_size] = []
            
            results_by_input_size[input_size].append({
                'loss': trial.value,
                'hidden_size': params.get('hidden_size'),
                'num_layers': params.get('num_layers'),
                'lr': params.get('lr'),
            })
    
    print("\n📊 Performance by Input Size:")
    for input_size, trials in sorted(results_by_input_size.items()):
        losses = [t['loss'] for t in trials]
        best_idx = np.argmin(losses)
        best_trial = trials[best_idx]
        
        print(f"\n  Input Size {input_size} ({len(trials)} trials):")
        print(f"    Best: loss={best_trial['loss']:.4f}")
        print(f"    Hidden: {best_trial['hidden_size']}, Layers: {best_trial['num_layers']}")
        print(f"    LR: {best_trial['lr']:.2e}")
        print(f"    Avg Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    
    return study


if __name__ == "__main__":
    print("🧠 ModelBatch-Optuna Input Size Integration Demo")
    print("This demonstrates hyperparameter optimization across different")
    print("input sizes while automatically grouping compatible configurations.")
    print()
    
    # Run basic input size demo
    basic_study = run_input_size_demo()
    
    # Run advanced demo
    advanced_study = run_advanced_input_size_demo()
    
    print("\n🎉 Input size optimization completed!")
    print("The demos show how ModelBatch efficiently handles different")
    print("input sizes by automatically grouping compatible models.")