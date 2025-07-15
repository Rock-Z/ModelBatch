# ModelBatch-Optuna Integration Design

## Overview

This document outlines the design for integrating ModelBatch with Optuna hyperparameter optimization while maintaining batching efficiency. The key challenge is enabling flexible hyperparameter search while preserving the constraint that all models in a batch must have identical architecture and batch size.

## Core Design Philosophy

**Constraint-Driven Batching**: Rather than allowing completely free hyperparameter search, we create "batch groups" where models within each group share critical constraints (architecture, batch size) while allowing variation in other hyperparameters (learning rate, dropout, weight decay, etc.).

## Main Abstractions

### 1. ModelBatchStudy

The primary interface for Optuna integration that manages batch grouping and constraint enforcement.

```python
class ModelBatchStudy:
    """
    Optuna study wrapper that enforces batching constraints while enabling
    flexible hyperparameter search.
    
    Key responsibilities:
    - Group trials into compatible batches based on constraints
    - Manage trial lifecycle within batches
    - Handle early stopping and pruning at batch level
    - Coordinate with Optuna's trial system
    """
    
    def __init__(
        self,
        study: optuna.Study,
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        constraint_spec: ConstraintSpec,
        batch_size: int = 32,
        max_models_per_batch: int = None,
    ):
        self.study = study
        self.model_factory = model_factory
        self.constraint_spec = constraint_spec
        self.batch_size = batch_size
        self.max_models_per_batch = max_models_per_batch
```

### 2. ConstraintSpec

Defines which hyperparameters must be homogeneous within a batch vs. which can vary.

```python
class ConstraintSpec:
    """
    Specification for hyperparameter constraints within ModelBatch runs.
    
    Attributes:
        fixed_params: Parameters that must be identical across all models in batch
        variable_params: Parameters that can vary between models in batch
        batch_aware_params: Parameters that affect batching decisions
    """
    
    def __init__(
        self,
        fixed_params: List[str] = None,  # e.g., ['model.hidden_size', 'model.num_layers']
        variable_params: List[str] = None,  # e.g., ['optimizer.lr', 'model.dropout_rate']
        batch_aware_params: List[str] = None,  # e.g., ['data.batch_size']
    ):
        self.fixed_params = fixed_params or []
        self.variable_params = variable_params or []
        self.batch_aware_params = batch_aware_params or []
```

### 3. BatchGroup

Represents a group of trials that can be trained together in a single ModelBatch.

```python
class BatchGroup:
    """
    A collection of trials that share batching constraints and can be trained together.
    
    Attributes:
        group_id: Unique identifier for this batch group
        constraint_hash: Hash of fixed parameters for this group
        trials: List of Optuna trials in this group
        models: Instantiated models for this group
        model_batch: ModelBatch instance for training
    """
    
    def __init__(self, group_id: str, constraint_hash: str):
        self.group_id = group_id
        self.constraint_hash = constraint_hash
        self.trials: List[optuna.Trial] = []
        self.models: List[nn.Module] = []
        self.model_batch: Optional[ModelBatch] = None
```

### 4. TrialBatcher

Responsible for grouping new Optuna trials into compatible batches.

```python
class TrialBatcher:
    """
    Groups Optuna trials into compatible batches based on constraints.
    
    Key methods:
    - add_trial: Add a new trial to appropriate batch group
    - get_ready_batches: Return batches ready for training
    - should_start_batch: Determine if a batch group has enough trials
    """
    
    def add_trial(self, trial: optuna.Trial) -> Optional[str]:
        """Add trial to appropriate batch group, return group_id."""
        
    def get_ready_batches(self) -> List[BatchGroup]:
        """Return batch groups ready for training."""
```

### 5. ModelBatchObjective

Wrapper around Optuna objective function that handles batching.

```python
class ModelBatchObjective:
    """
    Wraps user-defined objective function to work with ModelBatch training.
    
    Responsibilities:
    - Convert trial parameters to model configurations
    - Handle batch-level vs. trial-level metrics
    - Manage trial lifecycle within batches
    """
    
    def __call__(
        self, 
        trial_group: BatchGroup,
        train_fn: Callable[[ModelBatch, List[Dict]], List[float]]
    ) -> List[float]:
        """Execute training for a batch group and return per-trial metrics."""
```

## Constraint Enforcement Strategies

### Strategy 1: Hash-based Grouping

Each trial's fixed parameters are hashed to create a constraint key. Trials with the same key are grouped together.

```python
def _get_constraint_hash(self, trial_params: Dict[str, Any]) -> str:
    """Generate hash for grouping based on fixed parameters."""
    constraint_params = {
        k: v for k, v in trial_params.items()
        if k in self.constraint_spec.fixed_params
    }
    return hashlib.md5(str(sorted(constraint_params.items())).encode()).hexdigest()
```

### Strategy 2: Dynamic Batch Size

Allow different batch sizes across groups but enforce homogeneity within groups.

```python
class BatchSizeManager:
    """Manages batch size constraints across trial groups."""
    
    def get_batch_size(self, trial_params: Dict[str, Any]) -> int:
        """Extract batch size from trial parameters."""
        return trial_params.get('data.batch_size', self.default_batch_size)
```

### Strategy 3: Progressive Filling

Fill batches progressively as trials complete, rather than waiting for full batches.

```python
def should_start_batch(self, group: BatchGroup) -> bool:
    """Determine if batch should start training."""
    return (
        len(group.trials) >= self.min_models_per_batch or
        len(group.trials) >= self.max_models_per_batch or
        self._has_timeout_elapsed(group)
    )
```

## Gradient Checkpointing for Heterogeneous Batch Sizes

### CheckpointingStrategy

Enable training models with different effective batch sizes within the same ModelBatch.

```python
class CheckpointingStrategy:
    """
    Strategy for handling different batch sizes within ModelBatch.
    
    Options:
    1. Micro-batching: Split large batches into micro-batches
    2. Gradient accumulation: Accumulate gradients over multiple steps
    3. Dynamic padding: Pad smaller batches to match largest
    """
    
    def __init__(self, strategy: str = "micro_batching"):
        self.strategy = strategy
    
    def adjust_batch_sizes(
        self, 
        batch_group: BatchGroup,
        target_batch_sizes: List[int]
    ) -> Tuple[ModelBatch, List[int]]:
        """Adjust batch sizes to enable mixed-size training."""
```

## HuggingFace Integration

### ModelBatchHFTrainer

Integration with HuggingFace Trainer while maintaining batching benefits.

```python
class ModelBatchHFTrainer:
    """
    HuggingFace Trainer wrapper for ModelBatch training.
    
    Handles:
    - Dataset preparation for batch training
    - Evaluation across multiple models
    - Checkpoint management for batch groups
    """
    
    def __init__(
        self,
        model_batch: ModelBatch,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        tokenizer = None,
    ):
        self.model_batch = model_batch
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
```

### HFModelFactory

Factory for creating HuggingFace models with constraint-aware parameterization.

```python
class HFModelFactory:
    """
    Factory for creating HuggingFace models with constraint enforcement.
    
    Handles:
    - Model architecture constraints
    - Tokenizer configuration
    - Dataset preprocessing based on constraints
    """
    
    def create_model(
        self,
        model_name: str,
        trial_params: Dict[str, Any],
        constraint_spec: ConstraintSpec
    ) -> nn.Module:
        """Create HuggingFace model with constraint validation."""
```

## API Design

### Basic Usage Pattern

```python
import optuna
from modelbatch.optuna import ModelBatchStudy, ConstraintSpec

# Define constraints
constraints = ConstraintSpec(
    fixed_params=['model.hidden_size', 'model.num_layers'],
    variable_params=['optimizer.lr', 'model.dropout_rate'],
    batch_aware_params=['data.batch_size']
)

# Create study
study = optuna.create_study(direction="maximize")

# Create ModelBatch wrapper
mb_study = ModelBatchStudy(
    study=study,
    model_factory=create_model,
    constraint_spec=constraints,
    batch_size=32
)

# Run optimization
mb_study.optimize(
    objective_fn=train_and_evaluate,
    n_trials=100,
    timeout=3600
)
```

### Advanced Usage with Custom Constraints

```python
class CustomConstraintSpec(ConstraintSpec):
    """Custom constraint specification with domain-specific rules."""
    
    def validate_compatibility(self, trial1: Dict, trial2: Dict) -> bool:
        """Custom validation for trial compatibility."""
        # Ensure models have compatible memory requirements
        return (
            trial1['model.hidden_size'] == trial2['model.hidden_size'] and
            trial1['model.num_layers'] == trial2['model.num_layers'] and
            abs(trial1['optimizer.lr'] - trial2['optimizer.lr']) < 0.1
        )
```

## Performance Considerations

### 1. Batch Utilization
- Monitor batch fill rates to minimize idle GPU time
- Implement timeout-based batch starting for sparse trials
- Use trial prioritization for high-value configurations

### 2. Memory Management
- Track per-model memory requirements
- Implement dynamic batch sizing based on GPU memory
- Use gradient checkpointing for memory-constrained scenarios

### 3. Early Stopping
- Implement batch-level early stopping
- Support trial-level pruning within batches
- Coordinate with Optuna's pruning system

## Testing Strategy

### Core Tests Needed

1. **Constraint Validation**
   - Test hash-based grouping accuracy
   - Verify parameter constraint enforcement
   - Test edge cases in constraint specification

2. **Batching Logic**
   - Test progressive batch filling
   - Verify timeout mechanisms
   - Test batch size handling

3. **Optuna Integration**
   - Test trial lifecycle management
   - Verify metric reporting accuracy
   - Test early stopping coordination

4. **HF Integration**
   - Test model creation with constraints
   - Verify dataset preprocessing
   - Test checkpoint management

5. **Performance Tests**
   - Benchmark batch utilization rates
   - Test memory usage with different strategies
   - Verify scaling behavior

### Test Architecture

```python
class TestModelBatchStudy(unittest.TestCase):
    def test_constraint_grouping(self):
        """Test that trials are grouped by constraints."""
        
    def test_batch_filling(self):
        """Test progressive batch filling behavior."""
        
    def trial_lifecycle(self):
        """Test complete trial lifecycle within batches."""
        
    def test_hf_integration(self):
        """Test HuggingFace model integration."""
```

## Migration Path

### Phase 1: Basic Integration
- Implement ModelBatchStudy with simple constraint system
- Add basic HF model support
- Focus on correctness over performance

### Phase 2: Advanced Features
- Add gradient checkpointing for heterogeneous batch sizes
- Implement sophisticated constraint systems
- Add performance optimizations

### Phase 3: Production Features
- Add comprehensive monitoring and logging
- Implement advanced early stopping
- Add distributed training support