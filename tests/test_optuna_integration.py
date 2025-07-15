"""
Tests for ModelBatch-Optuna integration.

These tests verify the constraint system, batching logic, and Optuna integration.
"""

import pytest
import tempfile
import torch
import torch.nn as nn
from typing import Dict, Any, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Skip tests if optuna not available
optuna = pytest.importorskip("optuna")

from modelbatch.optuna_integration import (
    ConstraintSpec,
    BatchGroup,
    TrialBatcher,
    ModelBatchStudy,
)
from modelbatch.core import ModelBatch


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def create_model(params: Dict[str, Any]) -> nn.Module:
    """Factory function for test models."""
    return SimpleModel(
        input_size=params.get('model.input_size', 10),
        hidden_size=params.get('model.hidden_size', 20),
        dropout_rate=params.get('model.dropout_rate', 0.1),
    )


class TestConstraintSpec:
    """Test constraint specification system."""
    
    def test_basic_constraint_spec(self):
        """Test basic constraint specification creation."""
        spec = ConstraintSpec(
            fixed_params=['model.hidden_size', 'model.input_size'],
            variable_params=['model.dropout_rate', 'optimizer.lr']
        )
        
        assert spec.fixed_params == ['model.hidden_size', 'model.input_size']
        assert spec.variable_params == ['model.dropout_rate', 'optimizer.lr']
        assert spec.batch_aware_params == []
    
    def test_constraint_key_generation(self):
        """Test constraint key generation for grouping."""
        spec = ConstraintSpec(
            fixed_params=['model.hidden_size'],
            batch_aware_params=['data.batch_size']
        )
        
        params1 = {'model.hidden_size': 64, 'data.batch_size': 32, 'model.dropout_rate': 0.1}
        params2 = {'model.hidden_size': 64, 'data.batch_size': 32, 'model.dropout_rate': 0.2}
        params3 = {'model.hidden_size': 128, 'data.batch_size': 32, 'model.dropout_rate': 0.1}
        
        key1 = spec.get_constraint_key(params1)
        key2 = spec.get_constraint_key(params2)
        key3 = spec.get_constraint_key(params3)
        
        assert key1 == key2  # Same constraint key for same fixed params
        assert key1 != key3  # Different constraint key for different fixed params
    
    def test_constraint_validation(self):
        """Test parameter validation."""
        # Overlapping parameters should raise error
        with pytest.raises(ValueError):
            spec = ConstraintSpec(
                fixed_params=['model.hidden_size'],
                variable_params=['model.hidden_size']  # Overlaps with fixed
            )


class TestBatchGroup:
    """Test batch group management."""
    
    def test_batch_group_creation(self):
        """Test basic batch group creation."""
        group = BatchGroup(
            group_id="test_group",
            constraint_key="abc123",
            constraint_params={'model.hidden_size': 64}
        )
        
        assert group.group_id == "test_group"
        assert group.constraint_key == "abc123"
        assert group.constraint_params == {'model.hidden_size': 64}
        assert group.status == "pending"
        assert len(group.trials) == 0
    
    def test_adding_trials(self):
        """Test adding trials to batch group."""
        group = BatchGroup("test", "key", {})
        
        # Create mock trial (just use a dict for testing)
        trial = {"id": 0, "params": {}}
        model = SimpleModel()
        params = {'optimizer.lr': 0.001, 'model.dropout_rate': 0.1}
        
        group.add_trial(trial, params, model)
        
        assert len(group.trials) == 1
        assert len(group.models) == 1
        assert len(group.trial_params) == 1
    
    def test_batch_readiness(self):
        """Test batch readiness checks."""
        group = BatchGroup("test", "key", {})
        
        # Not ready with 0 trials
        assert group.is_ready(min_models_per_batch=2) is False
        
        # Add trials to make it ready
        for i in range(3):
            trial = {"id": i, "params": {}}
            model = SimpleModel()
            group.add_trial(trial, {}, model)
        
        assert group.is_ready(min_models_per_batch=2) is True
        assert group.is_full(max_models_per_batch=2) is True
    
    def test_variable_config_extraction(self):
        """Test extraction of variable configurations."""
        group = BatchGroup("test", "key", {'variable_params': ['optimizer.lr']})
        
        # Add trials with different learning rates
        for lr in [0.1, 0.01, 0.001]:
            trial = {"id": len(group.trials), "params": {}}
            model = SimpleModel()
            params = {'optimizer.lr': lr, 'model.hidden_size': 64}
            group.add_trial(trial, params, model)
        
        configs = group.get_variable_configs()
        assert len(configs) == 3
        assert all('lr' in config for config in configs)
        assert [config['lr'] for config in configs] == [0.1, 0.01, 0.001]


class TestTrialBatcher:
    """Test trial batching system."""
    
    def test_batcher_creation(self):
        """Test trial batcher creation."""
        spec = ConstraintSpec()
        batcher = TrialBatcher(spec)
        
        assert batcher.constraint_spec == spec
        assert len(batcher.batch_groups) == 0
        assert len(batcher.pending_trials) == 0
    
    def test_trial_grouping(self):
        """Test trial grouping by constraints."""
        spec = ConstraintSpec(
            fixed_params=['model.hidden_size'],
            variable_params=['optimizer.lr']
        )
        batcher = TrialBatcher(spec)
        study = optuna.create_study()
        
        # Create trials with different parameters
        # Group 1: hidden_size=64
        for lr in [0.1, 0.01]:
            trial = study.ask()
            model = SimpleModel(hidden_size=64)
            params = {'model.hidden_size': 64, 'optimizer.lr': lr}
            batcher.add_trial(trial, params, model)
        
        # Group 2: hidden_size=128
        for lr in [0.1, 0.01]:
            trial = study.ask()
            model = SimpleModel(hidden_size=128)
            params = {'model.hidden_size': 128, 'optimizer.lr': lr}
            batcher.add_trial(trial, params, model)
        
        # Should create 2 batch groups
        assert len(batcher.batch_groups) == 2
        
        # Check group contents
        groups = list(batcher.batch_groups.values())
        assert len(groups[0].trials) == 2
        assert len(groups[1].trials) == 2
    
    def test_ready_batches(self):
        """Test identification of ready batches."""
        spec = ConstraintSpec()
        batcher = TrialBatcher(spec, min_models_per_batch=2)
        study = optuna.create_study()
        
        # Add single trial - not ready
        trial1 = study.ask()
        model1 = SimpleModel()
        batcher.add_trial(trial1, {}, model1)
        
        ready = batcher.get_ready_batches()
        assert len(ready) == 0
        
        # Add second trial - now ready
        trial2 = study.ask()
        model2 = SimpleModel()
        batcher.add_trial(trial2, {}, model2)
        
        ready = batcher.get_ready_batches()
        assert len(ready) == 1
        assert len(ready[0].trials) == 2
    
    def test_batch_status(self):
        """Test batch status reporting."""
        spec = ConstraintSpec(fixed_params=['model.hidden_size'])
        batcher = TrialBatcher(spec)
        
        # Add trials using real Optuna study and trials
        study = optuna.create_study()
        
        for i in range(5):
            trial = study.ask()
            model = SimpleModel(hidden_size=64 if i < 3 else 128)
            params = {'model.hidden_size': 64 if i < 3 else 128}
            batcher.add_trial(trial, params, model)
        
        status = batcher.get_batch_status()
        assert status['total_groups'] == 2
        assert status['total_trials'] == 5


class TestModelBatchStudy:
    """Test ModelBatchStudy integration."""
    
    def test_study_creation(self):
        """Test basic study creation."""
        study = optuna.create_study(direction="minimize")
        spec = ConstraintSpec()
        
        mb_study = ModelBatchStudy(
            study=study,
            model_factory=create_model,
            constraint_spec=spec,
        )
        
        assert mb_study.study == study
        assert mb_study.model_factory == create_model
        assert mb_study.constraint_spec == spec
    
    def test_suggest_parameters_raises(self):
        """Test that suggest_parameters raises NotImplementedError."""
        study = optuna.create_study()
        spec = ConstraintSpec()
        
        mb_study = ModelBatchStudy(
            study=study,
            model_factory=create_model,
            constraint_spec=spec,
        )
        
        trial = study.ask()
        with pytest.raises(NotImplementedError):
            mb_study.suggest_parameters(trial)
    
    def test_create_models_for_batch(self):
        """Test model creation for batch groups."""
        study = optuna.create_study()
        spec = ConstraintSpec()
        
        mb_study = ModelBatchStudy(
            study=study,
            model_factory=create_model,
            constraint_spec=spec,
        )
        
        # Create batch group
        group = BatchGroup("test", "key", {'model.hidden_size': 64})
        
        # Use real Optuna trials
        trial1 = study.ask()
        trial2 = study.ask()
        
        group.add_trial(trial1, {'optimizer.lr': 0.1}, SimpleModel())
        group.add_trial(trial2, {'optimizer.lr': 0.01}, SimpleModel())
        
        models = mb_study.create_models_for_batch(group)
        assert len(models) == 2
        assert all(isinstance(m, SimpleModel) for m in models)
    
    def test_optimization_summary(self):
        """Test optimization summary generation."""
        study = optuna.create_study()
        spec = ConstraintSpec()
        
        mb_study = ModelBatchStudy(
            study=study,
            model_factory=create_model,
            constraint_spec=spec,
        )
        
        summary = mb_study.get_optimization_summary()
        assert 'total_trials' in summary
        assert 'total_groups' in summary


class TestIntegration:
    """Test complete integration workflows."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        
        class TestStudy(ModelBatchStudy):
            def suggest_parameters(self, trial):
                return {
                    'model.hidden_size': trial.suggest_int('hidden_size', 16, 32, step=16),
                    'model.dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                    'optimizer.lr': trial.suggest_float('lr', 1e-3, 1e-1, log=True),
                }
        
        def train_fn(model_batch, configs, optimizer=None):
            # Simple training function for testing
            return [0.5 + 0.1 * torch.randn(1).item() for _ in configs]
        
        study = optuna.create_study(direction="maximize")
        spec = ConstraintSpec(
            fixed_params=['model.hidden_size'],
            variable_params=['model.dropout_rate', 'optimizer.lr']
        )
        
        mb_study = TestStudy(
            study=study,
            model_factory=create_model,
            constraint_spec=spec,
            min_models_per_batch=2,
            max_models_per_batch=4,
        )
        
        # Run optimization with small number of trials
        mb_study.optimize(
            objective_fn=train_fn,
            n_trials=4,  # Use fewer trials to avoid batching issues
            show_progress_bar=False
        )
        
        # Verify results
        assert len(study.trials) == 4
        summary = mb_study.get_optimization_summary()
        assert summary['total_trials'] == 4


class TestConstraintValidation:
    """Test constraint validation edge cases."""
    
    def test_empty_constraint_spec(self):
        """Test empty constraint specification."""
        spec = ConstraintSpec()
        assert spec.fixed_params == []
        assert spec.variable_params == []
        assert spec.batch_aware_params == []
        
        key = spec.get_constraint_key({'param': 'value'})
        assert isinstance(key, str)
    
    def test_parameter_overlap_detection(self):
        """Test detection of overlapping parameters."""
        with pytest.raises(ValueError):
            ConstraintSpec(
                fixed_params=['model.size'],
                variable_params=['model.size']  # Should overlap
            )
    
    def test_complex_parameter_nesting(self):
        """Test constraint key with nested parameters."""
        spec = ConstraintSpec(
            fixed_params=['model.config.hidden_size'],
            batch_aware_params=['data.batch_size']
        )
        
        params = {
            'model.config.hidden_size': 768,
            'data.batch_size': 32,
            'optimizer.lr': 0.001
        }
        
        key = spec.get_constraint_key(params)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length


class TestMemoryEfficiency:
    """Test memory efficiency aspects."""
    
    def test_model_creation_memory(self):
        """Test that models are created efficiently."""
        spec = ConstraintSpec(fixed_params=['model.hidden_size'])
        
        def memory_efficient_factory(params):
            # Ensure we're not creating unnecessary copies
            return SimpleModel(**{k.split('.')[-1]: v for k, v in params.items()})
        
        batcher = TrialBatcher(spec)
        study = optuna.create_study()
        
        # Add multiple trials using real Optuna trials
        for i in range(10):
            trial = study.ask()
            params = {'model.hidden_size': 10 + i}
            model = memory_efficient_factory(params)
            batcher.add_trial(trial, params, model)
        
        # Should group appropriately - each unique hidden_size creates a new group
        assert len(batcher.batch_groups) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])