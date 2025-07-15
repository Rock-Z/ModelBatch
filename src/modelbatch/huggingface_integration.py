"""
HuggingFace integration for ModelBatch hyperparameter optimization.

Provides integration with HuggingFace transformers and datasets while
maintaining ModelBatch's batching efficiency and constraint system.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import warnings

try:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        TrainingArguments,
        Trainer,
        AutoModel,
        AutoTokenizer,
        AutoConfig,
    )
    from transformers.modeling_utils import PreTrainedModel as HFModel
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedModel = Any  # type: ignore
    PreTrainedTokenizer = Any  # type: ignore
    TrainingArguments = Any  # type: ignore
    Trainer = Any  # type: ignore
    Dataset = Any  # type: ignore
    HFModel = Any  # type: ignore

try:
    from optuna import Study
    from optuna.trial import Trial
except ImportError:
    Study = Any  # type: ignore
    Trial = Any  # type: ignore

import torch
import torch.nn as nn

from .core import ModelBatch
from .optimizer import OptimizerFactory
from .optuna_integration import ConstraintSpec, ModelBatchStudy


class HFModelFactory:
    """
    Factory for creating HuggingFace models with constraint enforcement.
    
    Handles model creation with constraint validation and configuration management.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        tokenizer_name_or_path: Optional[str] = None,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for HFModelFactory. "
                "Install with: pip install transformers"
            )
        
        self.model_name_or_path = model_name_or_path
        self.config_overrides = config_overrides or {}
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        
        # Load base config and tokenizer for validation
        self.base_config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
    
    def create_model(
        self,
        trial_params: Dict[str, Any],
        constraint_spec: ConstraintSpec,
    ) -> HFModel:
        """Create HuggingFace model with constraint validation."""
        # Validate parameters against constraints
        self._validate_constraints(trial_params, constraint_spec)
        
        # Create configuration with overrides
        config = self._create_config(trial_params)
        
        # Create model
        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )
        
        return model
    
    def _validate_constraints(
        self,
        trial_params: Dict[str, Any],
        constraint_spec: ConstraintSpec,
    ) -> None:
        """Validate trial parameters against constraints."""
        for param_name in constraint_spec.fixed_params:
            if param_name.startswith('model.'):
                # Extract model parameter
                _, param_key = param_name.split('.', 1)
                if param_key not in trial_params:
                    raise ValueError(f"Missing required parameter: {param_name}")
    
    def _create_config(self, trial_params: Dict[str, Any]) -> Any:
        """Create model configuration from trial parameters."""
        config = AutoConfig.from_pretrained(self.model_name_or_path)
        
        # Apply overrides
        for key, value in self.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Apply trial-specific parameters
        for param_name, value in trial_params.items():
            if param_name.startswith('model.'):
                _, config_key = param_name.split('.', 1)
                if hasattr(config, config_key):
                    setattr(config, config_key, value)
        
        return config


class HFConstraintSpec(ConstraintSpec):
    """
    Constraint specification tailored for HuggingFace models.
    
    Provides sensible defaults for common HuggingFace model parameters.
    """
    
    def __init__(
        self,
        fixed_params: Optional[List[str]] = None,
        variable_params: Optional[List[str]] = None,
        batch_aware_params: Optional[List[str]] = None,
    ):
        # Default fixed parameters for HF models
        if fixed_params is None:
            fixed_params = [
                'model.hidden_size',
                'model.num_attention_heads',
                'model.num_hidden_layers',
                'model.intermediate_size',
                'model.vocab_size',
            ]
        
        # Default variable parameters
        if variable_params is None:
            variable_params = [
                'optimizer.lr',
                'optimizer.weight_decay',
                'optimizer.betas',
                'model.dropout_rate',
                'model.attention_dropout_rate',
                'data.batch_size',
            ]
        
        # Default batch-aware parameters
        if batch_aware_params is None:
            batch_aware_params = ['data.batch_size', 'data.max_length']
        
        super().__init__(fixed_params, variable_params, batch_aware_params)


class ModelBatchHFTrainer:
    """
    HuggingFace Trainer wrapper for ModelBatch training.
    
    Provides compatibility with HuggingFace's training ecosystem while
    leveraging ModelBatch's vectorized training capabilities.
    """
    
    def __init__(
        self,
        model_batch: ModelBatch,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for ModelBatchHFTrainer. "
                "Install with: pip install transformers datasets"
            )
        
        self.model_batch = model_batch
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        
        # Create underlying trainer for compatibility
        self.trainer = Trainer(
            model=None,  # Will use ModelBatch instead
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    
    def train(self) -> List[Dict[str, float]]:
        """Train models using ModelBatch approach."""
        # Override training loop for ModelBatch
        return self._train_with_modelbatch()
    
    def _train_with_modelbatch(self) -> List[Dict[str, float]]:
        """Custom training loop using ModelBatch."""
        device = torch.device(self.args.device)
        
        # Setup data loader
        train_dataloader = self.trainer.get_train_dataloader()
        
        # Setup optimizer
        optimizer = self._create_optimizer()
        
        # Training loop
        metrics = []
        for epoch in range(self.args.num_train_epochs):
            epoch_metrics = self._train_epoch(train_dataloader, optimizer, device)
            metrics.append(epoch_metrics)
        
        return metrics
    
    def _create_optimizer(self):
        """Create optimizer compatible with ModelBatch."""
        # Use OptimizerFactory for per-model configurations
        factory = OptimizerFactory(torch.optim.AdamW)
        
        # Create configs for each model
        configs = []
        for i in range(self.model_batch.num_models):
            config = {
                'lr': self.args.learning_rate,
                'weight_decay': self.args.weight_decay,
                'betas': (self.args.adam_beta1, self.args.adam_beta2),
                'eps': self.args.adam_epsilon,
            }
            configs.append(config)
        
        return factory.create_optimizer(self.model_batch, configs)
    
    def _train_epoch(self, dataloader, optimizer, device):
        """Train for one epoch using ModelBatch."""
        self.model_batch.train()
        total_loss = 0.0
        num_steps = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model_batch(batch['input_ids'])
            
            # Compute loss (custom loss function needed)
            loss = self._compute_loss(outputs, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_steps += 1
        
        return {"loss": total_loss / num_steps}
    
    def _compute_loss(self, outputs, batch):
        """Compute loss for ModelBatch outputs."""
        # This needs to be customized based on the task
        # For now, return a placeholder
        return outputs.mean()


class HFModelBatchStudy(ModelBatchStudy):
    """
    Specialized ModelBatchStudy for HuggingFace models.
    
    Provides HuggingFace-specific functionality and defaults.
    """
    
    def __init__(
        self,
        study: Study,
        model_name_or_path: str,
        constraint_spec: Optional[HFConstraintSpec] = None,
        tokenizer_name_or_path: Optional[str] = None,
        **kwargs
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for HFModelBatchStudy. "
                "Install with: pip install transformers"
            )
        
        # Create constraint spec with defaults
        if constraint_spec is None:
            constraint_spec = HFConstraintSpec()
        
        # Create model factory
        model_factory = HFModelFactory(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
        
        super().__init__(
            study=study,
            model_factory=model_factory.create_model,
            constraint_spec=constraint_spec,
            **kwargs
        )
    
    def suggest_hf_parameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest HuggingFace-specific parameters."""
        params = {}
        
        # Model parameters
        params['model.dropout_rate'] = trial.suggest_float(
            'model.dropout_rate', 0.0, 0.5, step=0.1
        )
        params['model.attention_dropout_rate'] = trial.suggest_float(
            'model.attention_dropout_rate', 0.0, 0.5, step=0.1
        )
        
        # Optimizer parameters
        params['optimizer.lr'] = trial.suggest_loguniform(
            'optimizer.lr', 1e-5, 1e-2
        )
        params['optimizer.weight_decay'] = trial.suggest_loguniform(
            'optimizer.weight_decay', 1e-5, 0.1
        )
        params['optimizer.betas'] = (
            trial.suggest_float('optimizer.beta1', 0.8, 0.99),
            trial.suggest_float('optimizer.beta2', 0.9, 0.999)
        )
        
        # Data parameters
        params['data.batch_size'] = trial.suggest_categorical(
            'data.batch_size', [8, 16, 32, 64]
        )
        params['data.max_length'] = trial.suggest_int(
            'data.max_length', 128, 512, step=64
        )
        
        return params


class HFObjective:
    """
    HF-compatible objective for ModelBatch optimization.
    
    Handles the training and evaluation of HuggingFace models within
    the ModelBatch framework.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        model_name_or_path: str = "bert-base-uncased",
        task_type: str = "classification",
        num_labels: Optional[int] = None,
    ):
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.model_name_or_path = model_name_or_path
        self.task_type = task_type
        self.num_labels = num_labels
    
    def train_and_evaluate(
        self,
        model_batch: ModelBatch,
        configs: List[Dict[str, Any]]
    ) -> List[float]:
        """Train models and return evaluation metrics."""
        # This would implement the actual training loop
        # For now, return dummy metrics
        return [0.5] * len(configs)
    
    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        **kwargs
    ) -> HFModelBatchStudy:
        """Create HFModelBatchStudy for this objective."""
        return HFModelBatchStudy(
            study=optuna.create_study(
                study_name=study_name,
                direction=direction
            ),
            model_name_or_path=self.model_name_or_path,
            **kwargs
        )