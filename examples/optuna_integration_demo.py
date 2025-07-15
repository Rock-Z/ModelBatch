#!/usr/bin/env python3
"""
Demonstration of ModelBatch-Optuna integration for hyperparameter optimization.

This example shows how to use ModelBatch with Optuna for efficient
hyperparameter search while maintaining batching constraints.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import optuna
except ImportError:
    print("Optuna not available. Install with: pip install optuna")
    sys.exit(1)

from modelbatch import ModelBatch
from modelbatch.optuna_integration import ModelBatchStudy, ConstraintSpec


class SimpleMLP(nn.Module):
    """Simple MLP for demonstration."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def create_model(params: Dict[str, Any]) -> nn.Module:
    """Factory function for creating models based on parameters."""
    return SimpleMLP(
        input_size=params.get('model.input_size', 10),
        hidden_size=params.get('model.hidden_size', 20),
        dropout_rate=params.get('model.dropout_rate', 0.1),
    )


def generate_dummy_data(batch_size: int = 32, num_samples: int = 1000):
    """Generate dummy regression data."""
    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    return X, y


def train_objective(
    model_batch: ModelBatch,
    configs: List[Dict[str, Any]],
    X: torch.Tensor,
    y: torch.Tensor,
    num_epochs: int = 5
) -> List[float]:
    """
    Training objective function for ModelBatch.
    
    Returns list of validation losses for each model/trial.
    """
    from modelbatch.optimizer import OptimizerFactory
    
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


class SimpleOptunaStudy(ModelBatchStudy):
    """Simple Optuna study for demonstration."""
    
    def suggest_parameters(self, trial):
        """Suggest hyperparameters for each trial, including different model architectures."""
        return {
            'model.input_size': 10,  # Fixed by data
            'model.hidden_size': trial.suggest_int('hidden_size', 16, 64, step=16),  # Different architectures
            'model.dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'optimizer.lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        }


def run_demo():
    """Run the complete demonstration."""
    print("🚀 ModelBatch-Optuna Integration Demo")
    print("=" * 50)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dummy data
    X, y = generate_dummy_data()
    print(f"Generated dummy data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define constraints - allow different model configs but group by architecture
    print("\n📋 Setting up constraints...")
    constraints = ConstraintSpec(
        fixed_params=['model.input_size', 'model.hidden_size'],  # Models with same architecture batched together
        variable_params=[
            'model.dropout_rate',
            'optimizer.lr',
        ],
    )
    
    # Create study
    print("\n🔍 Creating Optuna study...")
    study = optuna.create_study(
        study_name="modelbatch_demo",
        direction="minimize",  # Minimize validation loss
    )
    
    # Create ModelBatch study
    print("\n⚡ Setting up ModelBatch study...")
    mb_study = SimpleOptunaStudy(
        study=study,
        model_factory=create_model,
        constraint_spec=constraints,
        min_models_per_batch=2,
        max_models_per_batch=4,
    )
    
    # Run optimization
    print("\n🏃 Running hyperparameter optimization...")
    
    def objective_fn(model_batch, configs):
        return train_objective(
            model_batch=model_batch,
            configs=configs,
            X=X,
            y=y,
            num_epochs=3
        )
    
    mb_study.optimize(
        objective_fn=objective_fn,
        n_trials=8,
        show_progress_bar=True
    )
    
    # Print results
    print("\n📊 Optimization Results")
    print("-" * 30)
    
    summary = mb_study.get_optimization_summary()
    print(f"Total trials: {summary['total_trials']}")
    print(f"Completed trials: {summary['completed_trials']}")
    print(f"Batch groups: {summary['total_groups']}")
    
    # Best parameters
    print("\n🏆 Best Parameters:")
    best_trial = study.best_trial
    print(f"Best validation loss: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Show all trials
    print("\n📈 All Trials:")
    for i, trial in enumerate(study.trials):
        loss_str = f"{trial.value:.4f}" if trial.value is not None else "None"
        print(f"Trial {i+1}: loss={loss_str}, params={trial.params}")
    
    return study


def run_advanced_demo():
    """Run advanced demo with custom constraints."""
    print("\n" + "=" * 50)
    print("🔧 Advanced Constraint Demo")
    print("=" * 50)
    
    # Custom constraint spec with more complex rules
    constraints = ConstraintSpec(
        fixed_params=[
            'model.input_size',
            'model.num_layers',  # Architecture must be same
            'model.hidden_size',  # Must be same for ModelBatch compatibility
        ],
        variable_params=[
            'model.dropout_rate',
            'optimizer.lr',
            'optimizer.weight_decay',
        ],
        batch_aware_params=[
            'data.batch_size',
        ],
    )
    
    # Create study with custom constraints
    study = optuna.create_study(direction="minimize")
    
    class AdvancedStudy(ModelBatchStudy):
        def suggest_parameters(self, trial):
            return {
                'model.input_size': 10,  # Fixed
                'model.num_layers': 2,   # Fixed
                'model.hidden_size': 64,  # Fixed for ModelBatch compatibility
                'model.dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.3),
                'optimizer.lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'optimizer.weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
                'data.batch_size': 32,   # Fixed for this demo
            }
    
    mb_study = AdvancedStudy(
        study=study,
        model_factory=create_model,
        constraint_spec=constraints,
        min_models_per_batch=3,
        max_models_per_batch=3,
    )
    
    # Generate data
    X, y = generate_dummy_data()
    
    def objective_fn(model_batch, configs):
        return train_objective(
            model_batch=model_batch,
            configs=configs,
            X=X,
            y=y,
            num_epochs=2
        )
    
    # Run optimization
    mb_study.optimize(
        objective_fn=objective_fn,
        n_trials=6,  # Exactly 2 batches of 3
        show_progress_bar=True
    )
    
    print("\n📊 Advanced Results:")
    print(f"Total trials: {len(study.trials)}")
    print(f"Batch groups used: {len(set(t.user_attrs.get('model_batch_group', 'unknown') for t in study.trials))}")
    
    return study


if __name__ == "__main__":
    print("🧠 ModelBatch-Optuna Integration Demo")
    print("This demonstrates hyperparameter optimization with ModelBatch")
    print("while maintaining batching efficiency through constraints.")
    print()
    
    # Run basic demo
    basic_study = run_demo()
    
    # Run advanced demo
    advanced_study = run_advanced_demo()
    
    print("\n🎉 Demo completed!")
    print("Check the results above to see how ModelBatch efficiently")
    print("searches hyperparameters while respecting batching constraints.")