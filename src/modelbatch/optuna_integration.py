"""
Optuna integration for ModelBatch hyperparameter optimization.

This module provides integration between ModelBatch and Optuna for efficient
hyperparameter search while maintaining batching constraints.
"""

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional
import warnings
try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import optuna
    from optuna import Study
    from optuna.trial import Trial, TrialState
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    Study = Any  # type: ignore
    Trial = Any  # type: ignore
    TrialState = Any  # type: ignore

import torch
import torch.nn as nn

from .core import ModelBatch
from .optimizer import OptimizerFactory


class ConstraintSpec:
    """
    Specification for hyperparameter constraints within ModelBatch runs.
    
    Defines which parameters must be identical across models in a batch
    versus which can vary between models.
    
    Attributes:
        fixed_params: Parameters that must be identical across all models
        variable_params: Parameters that can vary between models
        batch_aware_params: Parameters that affect batching decisions
    """
    
    def __init__(
        self,
        fixed_params: Optional[List[str]] = None,
        variable_params: Optional[List[str]] = None,
        batch_aware_params: Optional[List[str]] = None,
    ):
        self.fixed_params = fixed_params or []
        self.variable_params = variable_params or []
        self.batch_aware_params = batch_aware_params or []
        
        # Validate at initialization
        fixed_set = set(self.fixed_params)
        variable_set = set(self.variable_params)
        
        overlap = fixed_set.intersection(variable_set)
        if overlap:
            raise ValueError(f"Parameters cannot be both fixed and variable: {overlap}")
    
    def get_constraint_key(self, trial_params: Dict[str, Any]) -> str:
        """Generate constraint key for grouping trials."""
        constraint_params = {}
        
        # Include fixed parameters in constraint key
        for param in self.fixed_params:
            if param in trial_params:
                constraint_params[param] = trial_params[param]
        
        # Include batch-aware parameters
        for param in self.batch_aware_params:
            if param in trial_params:
                constraint_params[param] = trial_params[param]
        
        # Create deterministic hash
        key_data = str(sorted(constraint_params.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def validate_trial(self, trial_params: Dict[str, Any]) -> bool:
        """Validate that trial parameters meet constraint requirements."""
        # Ensure no parameter appears in both fixed and variable lists
        fixed_set = set(self.fixed_params)
        variable_set = set(self.variable_params)
        
        overlap = fixed_set.intersection(variable_set)
        if overlap:
            raise ValueError(f"Parameters cannot be both fixed and variable: {overlap}")
        
        return True


class BatchGroup:
    """
    A collection of trials that can be trained together in a single ModelBatch.
    
    Manages the lifecycle of trials within a constraint-compatible group.
    """
    
    def __init__(self, group_id: str, constraint_key: str, constraint_params: Dict[str, Any]):
        self.group_id = group_id
        self.constraint_key = constraint_key
        self.constraint_params = constraint_params
        self.trials: List[Trial] = []
        self.trial_params: List[Dict[str, Any]] = []
        self.models: List[nn.Module] = []
        self.model_batch: Optional[ModelBatch] = None
        self.metrics: List[float] = []
        self.start_time: Optional[float] = None
        self.status = "pending"  # pending, running, completed, failed
    
    def add_trial(self, trial: Trial, params: Dict[str, Any], model: nn.Module) -> None:
        """Add a trial to this batch group."""
        self.trials.append(trial)
        self.trial_params.append(params)
        self.models.append(model)
    
    def is_ready(self, min_models_per_batch: int = 1) -> bool:
        """Check if this group has enough trials to start training."""
        return len(self.trials) >= min_models_per_batch
    
    def is_full(self, max_models_per_batch: Optional[int] = None) -> bool:
        """Check if this group has reached maximum size."""
        if max_models_per_batch is None:
            return False
        return len(self.trials) >= max_models_per_batch
    
    def should_start(self, timeout: Optional[float] = None) -> bool:
        """Determine if batch should start based on timeout or other criteria."""
        if not self.start_time:
            return False
        
        if timeout and (time.time() - self.start_time) > timeout:
            return True
        
        return False
    
    def get_variable_configs(self) -> List[Dict[str, Any]]:
        """Extract variable parameter configurations for optimizer setup."""
        variable_configs = []
        
        for params in self.trial_params:
            config = {}
            for param_name in self.constraint_params.get('variable_params', []):
                if param_name in params:
                    # Convert parameter names to optimizer format
                    opt_name = param_name.replace('optimizer.', '')
                    config[opt_name] = params[param_name]
            variable_configs.append(config)
        
        return variable_configs


class TrialBatcher:
    """
    Groups Optuna trials into compatible batches based on constraints.
    
    Responsible for trial-to-batch mapping and batch lifecycle management.
    """
    
    def __init__(
        self,
        constraint_spec: ConstraintSpec,
        min_models_per_batch: int = 1,
        max_models_per_batch: Optional[int] = None,
        batch_timeout: Optional[float] = None,
    ):
        self.constraint_spec = constraint_spec
        self.min_models_per_batch = min_models_per_batch
        self.max_models_per_batch = max_models_per_batch
        self.batch_timeout = batch_timeout
        
        self.batch_groups: Dict[str, BatchGroup] = {}
        self.pending_trials: List[Tuple[Trial, Dict[str, Any], nn.Module]] = []
    
    def add_trial(
        self,
        trial: Trial,
        trial_params: Dict[str, Any],
        model: nn.Module
    ) -> str:
        """Add trial to appropriate batch group."""
        # Validate trial parameters
        self.constraint_spec.validate_trial(trial_params)
        
        # Get constraint key
        constraint_key = self.constraint_spec.get_constraint_key(trial_params)
        
        # Create or get batch group
        if constraint_key not in self.batch_groups:
            group_id = f"batch_{constraint_key}_{len(self.batch_groups)}"
            constraint_params = {
                k: v for k, v in trial_params.items()
                if k in self.constraint_spec.fixed_params + self.constraint_spec.batch_aware_params
            }
            self.batch_groups[constraint_key] = BatchGroup(
                group_id=group_id,
                constraint_key=constraint_key,
                constraint_params=constraint_params
            )
        
        group = self.batch_groups[constraint_key]
        group.add_trial(trial, trial_params, model)
        
        return group.group_id
    
    def get_ready_batches(self) -> List[BatchGroup]:
        """Return batch groups ready for training."""
        ready_batches = []
        
        for group in self.batch_groups.values():
            if group.status == "pending" and (
                group.is_ready(self.min_models_per_batch) or
                group.is_full(self.max_models_per_batch) or
                group.should_start(self.batch_timeout)
            ):
                ready_batches.append(group)
        
        return ready_batches
    
    def get_batch_status(self) -> Dict[str, int]:
        """Get summary of batch group status."""
        return {
            "total_groups": len(self.batch_groups),
            "ready_groups": len(self.get_ready_batches()),
            "total_trials": sum(len(g.trials) for g in self.batch_groups.values()),
        }


class ModelBatchStudy:
    """
    Optuna study wrapper that enforces batching constraints while enabling
    flexible hyperparameter search.
    
    This is the main integration point between ModelBatch and Optuna.
    """
    
    def __init__(
        self,
        study: Study,
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        constraint_spec: ConstraintSpec,
        optimizer_factory: Optional[OptimizerFactory] = None,
        batch_size: int = 32,
        min_models_per_batch: int = 1,
        max_models_per_batch: Optional[int] = None,
        batch_timeout: Optional[float] = 60.0,
    ):
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for ModelBatchStudy. Install with: pip install optuna")
        
        self.study = study
        self.model_factory = model_factory
        self.constraint_spec = constraint_spec
        self.optimizer_factory = optimizer_factory or OptimizerFactory(torch.optim.Adam)
        self.batch_size = batch_size
        
        self.trial_batcher = TrialBatcher(
            constraint_spec=constraint_spec,
            min_models_per_batch=min_models_per_batch,
            max_models_per_batch=max_models_per_batch,
            batch_timeout=batch_timeout,
        )
    
    def suggest_parameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest parameters for a trial using Optuna's suggest API."""
        # This should be overridden by user to define parameter space
        raise NotImplementedError(
            "Implement suggest_parameters to define your parameter space"
        )
    
    def create_models_for_batch(
        self,
        batch_group: BatchGroup
    ) -> List[nn.Module]:
        """Create models for a batch group using constraint parameters."""
        models = []
        
        for trial, params in zip(batch_group.trials, batch_group.trial_params):
            # Merge constraint parameters with trial parameters
            full_params = {**batch_group.constraint_params, **params}
            model = self.model_factory(full_params)
            models.append(model)
        
        return models
    
    def optimize(
        self,
        objective_fn: Callable[[ModelBatch, List[Dict[str, Any]]], List[float]],
        timeout: Optional[float] = None,
        n_trials: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Run hyperparameter optimization with ModelBatch integration.
        
        Args:
            objective_fn: Function that trains ModelBatch and returns per-trial metrics
            timeout: Maximum time for optimization
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs (currently only supports 1)
            show_progress_bar: Whether to show progress bar
        """
        if n_jobs != 1:
            warnings.warn("ModelBatchStudy currently only supports n_jobs=1")
        
        # Use ask-and-tell pattern to avoid premature trial completion
        trials_created = 0
        target_trials = n_trials or 100  # Default if n_trials is None
        
        if HAS_TQDM and show_progress_bar:
            pbar = tqdm.tqdm(total=target_trials)
        else:
            class DummyPbar:
                def update(self, *args, **kwargs): pass
                def close(self): pass
            pbar = DummyPbar()
        
        start_time = time.time()
        
        while True:
            # Check termination conditions
            if n_trials and trials_created >= n_trials:
                break
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Ask for new trial
            try:
                trial = self.study.ask()
            except:
                break  # Study is complete
            
            # Suggest parameters and create model
            trial_params = self.suggest_parameters(trial)
            model = self.model_factory(trial_params)
            
            # Add to batcher
            group_id = self.trial_batcher.add_trial(trial, trial_params, model)
            trials_created += 1
            pbar.update(1)
            
            # Check for ready batches and execute them
            ready_batches = self.trial_batcher.get_ready_batches()
            for batch_group in ready_batches:
                self._execute_batch(batch_group, objective_fn)
        
        pbar.close()
        
        # Execute any remaining batches
        self._execute_remaining_batches(objective_fn)
    
    def _execute_batch(
        self,
        batch_group: BatchGroup,
        objective_fn: Callable[[ModelBatch, List[Dict[str, Any]]], List[float]]
    ) -> None:
        """Execute training for a batch group."""
        try:
            batch_group.status = "running"
            batch_group.start_time = time.time()
            
            # Create models for this batch
            models = self.create_models_for_batch(batch_group)
            
            # Create ModelBatch
            model_batch = ModelBatch(models).to(next(models[0].parameters()).device)
            batch_group.model_batch = model_batch
            
            # Create optimizer with variable configurations
            variable_configs = batch_group.get_variable_configs()
            optimizer = self.optimizer_factory.create_optimizer(
                model_batch, variable_configs
            )
            
            # Execute objective function with constraint parameters
            constraint_context = {
                'constraint_params': batch_group.constraint_params,
                'group_id': batch_group.group_id,
                'num_models': len(models)
            }
            
            # Handle both old and new objective function signatures
            import inspect
            sig = inspect.signature(objective_fn)
            if len(sig.parameters) == 3:
                metrics = objective_fn(model_batch, variable_configs, constraint_context)
            else:
                metrics = objective_fn(model_batch, variable_configs)
            
            # Update trial results - tell the study the actual values
            batch_group.metrics = metrics
            batch_group.status = "completed"
            
            # Report actual metrics to the study
            for trial, metric in zip(batch_group.trials, metrics):
                try:
                    self.study.tell(trial, metric)  # type: ignore
                except Exception as e:
                    # Handle cases where trial might already be finished
                    pass
            
        except Exception as e:
            batch_group.status = "failed"
            # Report failure for all trials in batch
            for trial in batch_group.trials:
                try:
                    self.study.tell(trial, state=TrialState.FAIL)  # type: ignore
                except:
                    pass
            raise
    
    def _execute_remaining_batches(
        self,
        objective_fn: Callable[[ModelBatch, List[Dict[str, Any]]], List[float]]
    ) -> None:
        """Execute any remaining pending batches."""
        remaining_batches = list(self.trial_batcher.batch_groups.values())
        
        for batch_group in remaining_batches:
            if batch_group.status == "pending" and len(batch_group.trials) > 0:
                self._execute_batch(batch_group, objective_fn)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization process."""
        batch_status = self.trial_batcher.get_batch_status()
        
        return {
            "total_trials": len(self.study.trials),
            "completed_trials": len([t for t in self.study.trials if t.state == TrialState.COMPLETE]),  # type: ignore
            "failed_trials": len([t for t in self.study.trials if t.state == TrialState.FAIL]),  # type: ignore
            **batch_status,
        }


class SimpleObjective:
    """
    Simple objective wrapper for common training scenarios.
    
    Handles basic training loops with ModelBatch and Optuna integration.
    """
    
    def __init__(
        self,
        train_fn: Callable[[nn.Module, Dict[str, Any]], float],
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        constraint_spec: ConstraintSpec,
    ):
        self.train_fn = train_fn
        self.model_factory = model_factory
        self.constraint_spec = constraint_spec
    
    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        **kwargs
    ) -> ModelBatchStudy:
        """Create a ModelBatchStudy with this objective."""
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            **kwargs
        )
        
        return ModelBatchStudy(
            study=study,
            model_factory=self.model_factory,
            constraint_spec=self.constraint_spec,
        )