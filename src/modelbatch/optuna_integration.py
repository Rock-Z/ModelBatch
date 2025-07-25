"""
Optuna integration for ModelBatch hyperparameter optimization.

This module provides integration between ModelBatch and Optuna for efficient
hyperparameter search while maintaining batching constraints.
"""

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import warnings
from collections import defaultdict
import tqdm
from enum import Enum

if TYPE_CHECKING:
    from optuna import Study
    from optuna.trial import Trial, TrialState

try:
    import optuna
    from optuna import Study
    from optuna.trial import Trial, TrialState
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None  # type: ignore
    if not TYPE_CHECKING:
        Study = Any  # type: ignore
        Trial = Any  # type: ignore
        TrialState = Any  # type: ignore

import torch
import torch.nn as nn

from .core import ModelBatch, _check_models_compatible
from .optimizer import OptimizerFactory


class BatchState(Enum):
    """Enum for batch group states to prevent race conditions."""
    PENDING = "pending"      # Accepting new trials, not yet ready to execute
    READY = "ready"         # Ready to execute, but still accepting trials until execution starts
    RUNNING = "running"     # Currently executing, no new trials allowed
    COMPLETED = "completed" # Execution finished successfully
    FAILED = "failed"       # Execution failed


def _get_model_structure_signature(model: nn.Module) -> str:
    """
    Get a deterministic signature representing the model's parameter structure.
    
    This signature captures the parameter names and shapes, which is exactly
    what ModelBatch uses to determine compatibility.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        String signature that uniquely identifies the model structure
    """
    state_dict = model.state_dict()
    structure_info = []
    
    for name, param in state_dict.items():
        # Include parameter name and shape
        structure_info.append(f"{name}:{tuple(param.shape)}")
    
    # Create deterministic signature by sorting parameter info
    structure_str = "|".join(sorted(structure_info))
    return hashlib.md5(structure_str.encode()).hexdigest()


class OptunaBatchProgressBar:
    """Simple progress bar for Optuna-ModelBatch integration with dual progress bars."""
    
    def __init__(self, total_trials: int, show_batch_details: bool = True):
        self.total_trials = total_trials
        self.show_batch_details = show_batch_details
        
        # Progress tracking
        self.trials_suggested = 0
        self.trials_executed = 0
        self.best_value = None
        self.active_batches = 0
        
        # Two progress bars: suggestion and execution
        self.suggest_bar = tqdm.tqdm(
            total=total_trials,
            desc="Suggesting",
            position=0,
            bar_format="{desc}: {n}/{total} {bar} {percentage:3.0f}% | {postfix}",
            leave=True
        )
        
        self.execute_bar = tqdm.tqdm(
            total=total_trials,
            desc="Executing",
            position=1,
            bar_format="{desc}: {n}/{total} {bar} {percentage:3.0f}% | {postfix}",
            leave=True
        )
    
    def update_suggestion(self, trial_id: str, trial_params: Dict[str, Any], best_value: Optional[float] = None) -> None:
        """Update progress when a new trial is suggested."""
        self.trials_suggested += 1
        
        if best_value is not None:
            self.best_value = best_value
        
        # Update suggestion bar
        best_str = f"{self.best_value:.4f}" if self.best_value is not None else "N/A"
        self.suggest_bar.set_postfix_str(f"Best: {best_str}")
        self.suggest_bar.update(1)
    
    def update_batch_group(self, group_id: str, constraint_params: Dict[str, Any], 
                          current_trials: int, max_trials: int, status: str,
                          timeout_remaining: Optional[float] = None) -> None:
        """Update batch group information."""
        # Simplified - just update execution bar display
        self._update_execution_bar()
    
    def start_batch_execution(self, group_id: str) -> None:
        """Mark a batch as starting execution."""
        self.active_batches += 1
        self._update_execution_bar()
    
    def complete_batch_execution(self, group_id: str, trial_count: int) -> None:
        """Mark a batch as completed."""
        self.active_batches = max(0, self.active_batches - 1)
        self.trials_executed += trial_count
        
        # Update execution bar progress
        self.execute_bar.update(trial_count)
        self._update_execution_bar()
    
    def _update_execution_bar(self) -> None:
        """Update the execution progress bar display."""
        status_parts = []
        
        if self.active_batches > 0:
            status_parts.append(f"Active: {self.active_batches}")
        
        # Show queue size (suggested but not yet executed)
        queue_size = self.trials_suggested - self.trials_executed
        if queue_size > 0:
            status_parts.append(f"Queue: {queue_size}")
        
        postfix = " | ".join(status_parts) if status_parts else "Ready"
        self.execute_bar.set_postfix_str(postfix)
    
    def close(self) -> None:
        """Close the progress display."""
        # Final update
        self.suggest_bar.refresh()
        self.execute_bar.refresh()
        
        # Close bars
        self.suggest_bar.close()
        self.execute_bar.close()
        
        # Final summary
        print(f"\nOptimization completed: {self.trials_executed} trials executed")
        if self.best_value is not None:
            print(f"Best value: {self.best_value:.4f}")





class ConstraintSpec:
    """
    Optional specification for hyperparameter constraints within ModelBatch runs.
    
    When provided, defines which parameters must be identical across models in a batch
    versus which can vary between models. When not provided, ModelBatch uses automatic
    compatibility detection based on model structure.
    
    Attributes:
        fixed_params: Parameters that must be identical across all models (optional)
        variable_params: Parameters that can vary between models (optional)
        batch_aware_params: Parameters that affect batching decisions (optional)
        use_auto_compatibility: If True, ignore constraints and use automatic detection
    """
    
    def __init__(
        self,
        fixed_params: Optional[List[str]] = None,
        variable_params: Optional[List[str]] = None,
        batch_aware_params: Optional[List[str]] = None,
        use_auto_compatibility: bool = False,
    ):
        self.fixed_params = fixed_params or []
        self.variable_params = variable_params or []
        self.batch_aware_params = batch_aware_params or []
        self.use_auto_compatibility = use_auto_compatibility
        
        # If any constraint params are provided, disable auto-compatibility by default
        if (fixed_params or variable_params or batch_aware_params) and use_auto_compatibility:
            warnings.warn(
                "Constraint parameters provided but use_auto_compatibility=True. "
                "Auto-compatibility will be used. Set use_auto_compatibility=False to use constraints."
            )
        
        # Validate constraint parameters if using them
        if not self.use_auto_compatibility:
            fixed_set = set(self.fixed_params)
            variable_set = set(self.variable_params)
            
            overlap = fixed_set.intersection(variable_set)
            if overlap:
                raise ValueError(f"Parameters cannot be both fixed and variable: {overlap}")
    
    def get_constraint_key(self, trial_params: Dict[str, Any], model: Optional[nn.Module] = None) -> str:
        """Generate constraint key for grouping trials."""
        if self.use_auto_compatibility and model is not None:
            # Use model structure for automatic compatibility
            return _get_model_structure_signature(model)
        
        # Fall back to parameter-based constraints
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
    
    def validate_trial(self, trial: Trial) -> bool:
        """Validate that trial parameters meet constraint requirements."""
        if self.use_auto_compatibility:
            # No validation needed for auto-compatibility
            return True
        
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
        self.creation_time: float = time.time()
        self.execution_start_time: Optional[float] = None
        self.state = BatchState.PENDING
    
    def add_trial(self, trial: Trial, params: Dict[str, Any], model: nn.Module) -> None:
        """Add a trial to this batch group."""
        # Prevent adding trials to non-pending batches
        if self.state != BatchState.PENDING:
            raise ValueError(
                f"Cannot add trial {trial.number} to batch group {self.group_id} "
                f"in state {self.state.value}. Only PENDING batches can accept new trials."
            )
        
        # Check compatibility with existing models in the group
        if self.models:
            if not _check_models_compatible(self.models[0], model)[0]:
                raise ValueError(
                    f"Model for trial {trial.number} is not compatible with existing models "
                    f"in batch group {self.group_id}. Model structures must match exactly for batching. "
                    f"This suggests an issue with the constraint specification or model factory."
                )
        
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
    
    def get_timeout_remaining(self, timeout: Optional[float] = None) -> Optional[float]:
        """Get remaining timeout for this batch group."""
        if not timeout:
            return None
        
        elapsed = time.time() - self.creation_time
        remaining = timeout - elapsed
        return max(0, remaining)
    
    def is_timed_out(self, timeout: Optional[float] = None) -> bool:
        """Check if this batch group has timed out."""
        if not timeout:
            return False
        
        elapsed = time.time() - self.creation_time
        return elapsed >= timeout
    
    def get_age_seconds(self) -> float:
        """Get the age of this batch group in seconds."""
        return time.time() - self.creation_time
    
    def should_start(self, timeout: Optional[float] = None) -> bool:
        """Determine if batch should start based on timeout or other criteria."""
        if self.state != BatchState.PENDING:
            return False
        
        if timeout and (time.time() - self.creation_time) > timeout:
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
        partial_batch_timeout_factor: float = 0.5,
        min_accumulation_time: float = 10.0,
        periodic_execution_interval: int = 10,
    ):
        self.constraint_spec = constraint_spec
        self.min_models_per_batch = min_models_per_batch
        self.max_models_per_batch = max_models_per_batch
        self.batch_timeout = batch_timeout
        self.partial_batch_timeout_factor = partial_batch_timeout_factor
        self.min_accumulation_time = min_accumulation_time
        self.periodic_execution_interval = periodic_execution_interval
        
        self.batch_groups: Dict[str, BatchGroup] = {}
        self.pending_trials: List[Tuple[Trial, Dict[str, Any], nn.Module]] = []
    
    def add_trial(
        self,
        trial: Trial,
        trial_params: Dict[str, Any],
        model: nn.Module
    ) -> str:
        """Add trial to appropriate batch group using automatic compatibility detection."""
        # Validate trial parameters
        self.constraint_spec.validate_trial(trial)
        
        if self.constraint_spec.use_auto_compatibility:
            # Use automatic compatibility detection
            return self._add_trial_auto_compatibility(trial, trial_params, model)
        else:
            # Use constraint-based grouping (legacy behavior)
            return self._add_trial_constraint_based(trial, trial_params, model)
    
    def _add_trial_auto_compatibility(
        self,
        trial: Trial,
        trial_params: Dict[str, Any],
        model: nn.Module
    ) -> str:
        """Add trial using automatic model structure compatibility detection."""
        print(f"\n🔍 DEBUG: Adding trial {trial.number} to batch using auto compatibility")
        print(f"  Current batch groups: {list(self.batch_groups.keys())}")
        
        # Try to find an existing compatible batch group that can accept new trials
        compatible_group = None
        
        for group_id, group in self.batch_groups.items():
            print(f"  Checking compatibility with group {group_id} (has {len(group.models)} models, state: {group.state.value})")
            
            # Skip groups that are not pending (the key fix for the race condition)
            if group.state != BatchState.PENDING:
                print(f"    Skipping group {group_id}: not pending (state: {group.state.value})")
                continue
            
            # Skip empty groups
            if not group.models:
                print(f"    Skipping group {group_id}: empty")
                continue
            
            # Skip groups that are at maximum capacity
            if self.max_models_per_batch and len(group.models) >= self.max_models_per_batch:
                print(f"    Skipping group {group_id}: at max capacity ({self.max_models_per_batch})")
                continue
            
            # Check if this model is compatible with the group
            is_compatible, reason = _check_models_compatible(model, group.models[0])
            print(f"    Compatibility check with group {group_id}: {is_compatible} ({reason})")
            
            if is_compatible:
                compatible_group = group
                print(f"    ✅ Found compatible group: {group_id}")
                break
        
        # If no compatible group found, create a new one
        if compatible_group is None:
            # Generate unique group ID
            group_id = f"auto_batch_{len(self.batch_groups)}"
            
            # Use model structure signature as constraint key
            constraint_key = _get_model_structure_signature(model)
            
            print(f"  ❌ No compatible group found, creating new group: {group_id}")
            print(f"    Model structure signature: {constraint_key}")
            
            # For auto-compatibility, constraint_params can be empty or contain metadata
            constraint_params = {
                '_auto_generated': True,
                '_structure_signature': constraint_key
            }
            
            compatible_group = BatchGroup(
                group_id=group_id,
                constraint_key=constraint_key,
                constraint_params=constraint_params
            )
            
            # Use group_id as the key in batch_groups dict
            self.batch_groups[group_id] = compatible_group
        
        # Add trial to the compatible group
        compatible_group.add_trial(trial, trial_params, model)
        print(f"  ✅ Added trial {trial.number} to group {compatible_group.group_id} (now has {len(compatible_group.trials)} trials)")
        
        return compatible_group.group_id
    
    def _add_trial_constraint_based(
        self,
        trial: Trial,
        trial_params: Dict[str, Any],
        model: nn.Module
    ) -> str:
        """Add trial using legacy constraint-based grouping."""
        # Get constraint key (without model for legacy behavior)
        constraint_key = self.constraint_spec.get_constraint_key(trial_params)
        
        # Create or get batch group
        if constraint_key not in self.batch_groups:
            group_id = f"constraint_batch_{constraint_key}_{len(self.batch_groups)}"
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
        group.add_trial(trial, trial_params, model)  # This will validate compatibility
        
        return group.group_id
    
    def get_ready_batches(self) -> List[BatchGroup]:
        """Return batch groups ready for training with smarter criteria."""
        ready_batches = []
        
        for group in self.batch_groups.values():
            if group.state == BatchState.PENDING and (
                group.is_full(self.max_models_per_batch) or
                group.should_start(self.batch_timeout) or
                (group.is_ready(self.min_models_per_batch) and self._should_execute_partial_batch(group))
            ):
                # Mark as ready to prevent new trials from being added
                group.state = BatchState.READY
                ready_batches.append(group)
        
        return ready_batches
    
    def _should_execute_partial_batch(self, group: BatchGroup) -> bool:
        """
        Determine if a partial batch should be executed now.
        
        This implements smarter batching logic that prioritizes larger batches
        and uses timeouts to prevent trials from waiting indefinitely.
        """
        # If there are other groups with more trials, wait a bit longer
        larger_groups = [
            g for g in self.batch_groups.values() 
            if g.state == BatchState.PENDING and len(g.trials) > len(group.trials)
        ]
        
        age = time.time() - group.creation_time
        
        if larger_groups:
            # Wait longer if there are larger groups that might be ready soon
            # Use configurable factor of batch_timeout for partial batches
            partial_timeout = (self.batch_timeout or 30.0) * self.partial_batch_timeout_factor
            return age > partial_timeout
        
        # No larger groups, execute if we've waited the minimum accumulation time
        return age > self.min_accumulation_time
    
    def get_batch_status(self) -> Dict[str, int]:
        """Get summary of batch group status."""
        return {
            "total_groups": len(self.batch_groups),
            "ready_groups": len(self.get_ready_batches()),
            "total_trials": sum(len(g.trials) for g in self.batch_groups.values()),
        }


class ModelBatchStudy:
    """
    Optuna study wrapper that automatically groups compatible models for efficient training.
    
    This is the main integration point between ModelBatch and Optuna. By default, it uses
    automatic compatibility detection to group models with identical structures together,
    eliminating the need to manually specify constraints.
    
    Models are automatically determined to be compatible if they have:
    - Identical parameter names
    - Identical parameter shapes
    
    This allows for flexible hyperparameter search where users can search over any
    parameters they want, and ModelBatch will automatically batch together models
    that can be trained simultaneously.
    
    Args:
        partial_batch_timeout_factor: Factor of batch_timeout to wait for partial batches 
                                     when larger batches exist (default: 0.5)
        min_accumulation_time: Minimum time to wait before executing partial batches 
                              when no larger batches exist (default: 10.0 seconds)
        periodic_execution_interval: Execute partial batches every N trials to prevent 
                                    starvation (default: 10)
    """
    
    def __init__(
        self,
        study: Study,
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        constraint_spec: Optional[ConstraintSpec] = None,
        min_models_per_batch: int = 1,
        max_models_per_batch: Optional[int] = None,
        batch_timeout: Optional[float] = 60.0,
        partial_batch_timeout_factor: float = 0.5,
        min_accumulation_time: float = 10.0,
        periodic_execution_interval: int = 10,
    ):
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for ModelBatchStudy. Install with: pip install optuna")
        
        self.study = study
        self.model_factory = model_factory
        
        # Use automatic compatibility detection by default
        if constraint_spec is None:
            self.constraint_spec = ConstraintSpec(use_auto_compatibility=True)
        else:
            self.constraint_spec = constraint_spec
        
        self.trial_batcher = TrialBatcher(
            constraint_spec=self.constraint_spec,
            min_models_per_batch=min_models_per_batch,
            max_models_per_batch=max_models_per_batch,
            batch_timeout=batch_timeout,
            partial_batch_timeout_factor=partial_batch_timeout_factor,
            min_accumulation_time=min_accumulation_time,
            periodic_execution_interval=periodic_execution_interval,
        )
    
    def suggest_parameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest parameters for a trial using Optuna's suggest API."""
        # This should be overridden by user to define parameter space
        raise NotImplementedError(
            "Implement suggest_parameters to define your parameter space"
        )
    
    def _find_batch_group(self, group_id: str) -> BatchGroup:
        """Find a batch group by its ID."""
        for batch_group in self.trial_batcher.batch_groups.values():
            if batch_group.group_id == group_id:
                return batch_group
        raise RuntimeError(f"Could not find batch group with id {group_id}")
    
    def _update_progress_bar(
        self, 
        progress_bar: OptunaBatchProgressBar, 
        trial: Trial, 
        trial_params: Dict[str, Any], 
        batch_group: BatchGroup
    ) -> None:
        """Update progress bar with trial and batch group information."""
        best_value = None
        if self.study.best_trials:
            best_value = self.study.best_value
        
        progress_bar.update_suggestion(
            trial_id=str(trial.number),
            trial_params=trial_params,
            best_value=best_value
        )
        
        # Update batch group information using integrated timeout tracking
        timeout_remaining = batch_group.get_timeout_remaining(self.trial_batcher.batch_timeout)
        progress_bar.update_batch_group(
            group_id=batch_group.group_id,
            constraint_params=batch_group.constraint_params,
            current_trials=len(batch_group.trials),
            max_trials=self.trial_batcher.max_models_per_batch or 4,  # Default max
            status='pending',
            timeout_remaining=timeout_remaining
        )
    
    def optimize(
        self,
        objective_fn: Callable[[ModelBatch, List[Dict[str, Any]]], List[float]],
        timeout: Optional[float] = None,
        n_trials: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Run hyperparameter optimization with ModelBatch integration.
        
        Args:
            objective_fn: Function that trains ModelBatch and returns per-trial metrics
            timeout: Maximum time for optimization
            n_trials: Number of trials to run
            show_progress_bar: Whether to show progress bar
        """
        
        # Use ask-and-tell pattern to avoid premature trial completion
        trials_created = 0
        target_trials = n_trials or 100  # Default if n_trials is None
        
        # Initialize progress bar
        progress_bar = None
        if show_progress_bar:
            progress_bar = OptunaBatchProgressBar(target_trials, show_batch_details=True)
        
        start_time = time.time()
        
        try:
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
                
                # Find the batch group and update progress bar
                batch_group = self._find_batch_group(group_id)
                
                # Update progress bar
                if progress_bar:
                    self._update_progress_bar(progress_bar, trial, trial_params, batch_group)
                
                # Check for ready batches and execute them, but use smarter timing
                self._execute_ready_batches_if_needed(objective_fn, progress_bar, trials_created)
            
            # Execute any remaining batches
            self._execute_remaining_batches(objective_fn, progress_bar)
            
        finally:
            if progress_bar:
                progress_bar.close()
    
    def _execute_ready_batches_if_needed(
        self,
        objective_fn: Callable[[ModelBatch, List[Dict[str, Any]]], List[float]],
        progress_bar: Optional[OptunaBatchProgressBar],
        trials_created: int
    ) -> None:
        """
        Execute ready batches using smarter coordination to prevent race conditions.
        
        This method implements better ask-and-tell coordination by:
        1. Only executing full batches immediately
        2. Delaying execution of partial batches to allow more trials to accumulate
        3. Executing partial batches periodically to prevent starvation
        """
        ready_batches = self.trial_batcher.get_ready_batches()
        
        # Separate full batches from partial batches
        full_batches = []
        partial_batches = []
        
        for batch_group in ready_batches:
            if (self.trial_batcher.max_models_per_batch and 
                len(batch_group.trials) >= self.trial_batcher.max_models_per_batch):
                full_batches.append(batch_group)
            else:
                partial_batches.append(batch_group)
        
        # Always execute full batches immediately
        for batch_group in full_batches:
            if progress_bar:
                progress_bar.start_batch_execution(batch_group.group_id)
            self._execute_batch(batch_group, objective_fn, progress_bar)
        
        # Execute partial batches only under certain conditions:
        # 1. Periodically based on configurable interval to prevent starvation
        # 2. When timeout has been reached
        # 3. When no more trials are being suggested (handled in _execute_remaining_batches)
        should_execute_partial = (
            trials_created % self.trial_batcher.periodic_execution_interval == 0 or  # Periodic execution
            any(batch_group.should_start(self.trial_batcher.batch_timeout) for batch_group in partial_batches)
        )
        
        if should_execute_partial:
            for batch_group in partial_batches:
                if progress_bar:
                    progress_bar.start_batch_execution(batch_group.group_id)
                self._execute_batch(batch_group, objective_fn, progress_bar)
    
    def _execute_batch(
        self,
        batch_group: BatchGroup,
        objective_fn: Callable[[ModelBatch, List[Dict[str, Any]]], List[float]],
        progress_bar: Optional[OptunaBatchProgressBar] = None
    ) -> None:
        """Execute training for a batch group."""
        try:
            batch_group.state = BatchState.RUNNING
            batch_group.execution_start_time = time.time()

            print(f"🔍 DEBUG: Executing batch group {batch_group.group_id} with {len(batch_group.trials)} trials")
            
            # Create models for this batch
            models = batch_group.models
            
            # Create ModelBatch
            model_batch = ModelBatch(models).to(next(models[0].parameters()).device)
            batch_group.model_batch = model_batch
            
            # Create optimizer with variable configurations
            variable_configs = batch_group.get_variable_configs()
            
            metrics = objective_fn(model_batch, variable_configs)
            
            # Update trial results - tell the study the actual values
            batch_group.metrics = metrics
            batch_group.state = BatchState.COMPLETED
            
            # Report actual metrics to the study
            for trial, metric in zip(batch_group.trials, metrics):
                try:
                    self.study.tell(trial, metric)  # type: ignore
                except Exception:
                    
                    # Handle cases where trial might already be finished
                    pass
            
            # Update progress bar
            if progress_bar:
                progress_bar.complete_batch_execution(batch_group.group_id, len(batch_group.trials))
            
        except Exception as e:
            batch_group.state = BatchState.FAILED
            # Report failure for all trials in batch
            for trial in batch_group.trials:
                try:
                    self.study.tell(trial, state=TrialState.FAIL)  # type: ignore
                except:
                    pass
            raise
    
    def _execute_remaining_batches(
        self,
        objective_fn: Callable[[ModelBatch, List[Dict[str, Any]]], List[float]],
        progress_bar: Optional[OptunaBatchProgressBar] = None
    ) -> None:
        """Execute any remaining pending batches."""
        remaining_batches = list(self.trial_batcher.batch_groups.values())
        
        for batch_group in remaining_batches:
            if batch_group.state == BatchState.PENDING and len(batch_group.trials) > 0:
                if progress_bar:
                    progress_bar.start_batch_execution(batch_group.group_id)
                self._execute_batch(batch_group, objective_fn, progress_bar)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization process."""
        batch_status = self.trial_batcher.get_batch_status()
        
        return {
            "total_trials": len(self.study.trials),
            "completed_trials": len([t for t in self.study.trials if t.state == TrialState.COMPLETE]),  # type: ignore
            "failed_trials": len([t for t in self.study.trials if t.state == TrialState.FAIL]),  # type: ignore
            **batch_status,
        }
