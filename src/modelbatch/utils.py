"""
Utility functions for ModelBatch training and evaluation.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .callbacks import CallbackPack
from .core import ModelBatch


def train_one_epoch(
    model_batch: ModelBatch,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    callbacks: Optional[CallbackPack] = None,
    clip_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """
    Train ModelBatch for one epoch.
    
    Args:
        model_batch: The ModelBatch instance
        train_loader: Training data loader
        optimizer: Optimizer instance
        loss_fn: Loss function
        device: Device to train on
        callbacks: Optional callback pack
        clip_grad_norm: Optional gradient clipping threshold
        
    Returns:
        Dictionary of training metrics
    """
    model_batch.train()
    total_loss = 0.0
    num_batches = 0
    
    for step, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model_batch(inputs)
        
        # Compute loss
        loss = model_batch.compute_loss(outputs, targets, loss_fn)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model_batch.parameters(), 
                clip_grad_norm,
            )
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Execute callbacks
        if callbacks is not None:
            callbacks.on_train_step(model_batch, step)
    
    # Return average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = {"train_loss": avg_loss}
    
    # Add per-model losses if available
    if model_batch.latest_losses is not None:
        for i, loss in enumerate(model_batch.latest_losses):
            metrics[f"train_loss_model_{i}"] = float(loss)
    
    return metrics


def evaluate_model_batch(
    model_batch: ModelBatch,
    val_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    callbacks: Optional[CallbackPack] = None,
) -> Dict[str, float]:
    """
    Evaluate ModelBatch on validation data.
    
    Args:
        model_batch: The ModelBatch instance
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to evaluate on
        callbacks: Optional callback pack
        
    Returns:
        Dictionary of validation metrics
    """
    model_batch.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model_batch(inputs)
            
            # Compute loss
            loss = model_batch.compute_loss(outputs, targets, loss_fn)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Execute callbacks
            if callbacks is not None:
                callbacks.on_validation_step(model_batch, step)
    
    # Return average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = {"val_loss": avg_loss}
    
    # Add per-model losses if available
    if model_batch.latest_losses is not None:
        for i, loss in enumerate(model_batch.latest_losses):
            metrics[f"val_loss_model_{i}"] = float(loss)
    
    return metrics


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy for each model.
    
    Args:
        outputs: Model outputs [num_models, batch_size, num_classes]
        targets: Target labels [batch_size] or [num_models, batch_size]
        
    Returns:
        Per-model accuracies [num_models]
    """
    # Get predictions
    if outputs.dim() == 3:  # Classification case
        predictions = outputs.argmax(dim=-1)  # [num_models, batch_size]
    else:
        predictions = outputs  # Assume already processed
    
    # Handle target broadcasting
    if targets.dim() == 1:
        # Broadcast targets to all models
        targets = targets.unsqueeze(0).expand(outputs.shape[0], -1)
    
    # Compute accuracy per model
    correct = (predictions == targets).float()
    accuracies = correct.mean(dim=1)  # Average over batch dimension
    
    return accuracies


def create_identical_models(
    model_class: type,
    model_kwargs: Dict[str, Any],
    num_models: int,
    init_fn: Optional[Callable[[nn.Module], None]] = None,
) -> List[nn.Module]:
    """
    Create a list of identical models with optional custom initialization.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model constructor
        num_models: Number of models to create
        init_fn: Optional function to initialize each model
        
    Returns:
        List of initialized models
    """
    models = []
    
    for i in range(num_models):
        # Create model
        model = model_class(**model_kwargs)
        
        # Apply custom initialization
        if init_fn is not None:
            init_fn(model)
        
        models.append(model)
    
    return models


def random_init_fn(model: nn.Module, std: float = 0.01) -> None:
    """
    Random initialization function for models.
    
    Args:
        model: Model to initialize
        std: Standard deviation for normal initialization
    """
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.normal_(param, mean=0.0, std=std)
        else:
            nn.init.zeros_(param)


def xavier_init_fn(model: nn.Module) -> None:
    """
    Xavier initialization function for models.
    
    Args:
        model: Model to initialize
    """
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_normal_(param)
        else:
            nn.init.zeros_(param)


def count_parameters(model_batch: ModelBatch) -> Dict[str, int]:
    """
    Count parameters in ModelBatch.
    
    Args:
        model_batch: The ModelBatch instance
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model_batch.parameters())
    trainable_params = sum(p.numel() for p in model_batch.parameters() if p.requires_grad)
    params_per_model = total_params // model_batch.num_models
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_per_model": params_per_model,
        "num_models": model_batch.num_models,
    }


def estimate_memory_usage(
    model_batch: ModelBatch,
    batch_size: int,
    input_shape: tuple,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Estimate memory usage for ModelBatch training.
    
    Args:
        model_batch: The ModelBatch instance
        batch_size: Training batch size
        input_shape: Shape of input tensors (excluding batch dimension)
        dtype: Data type for calculations
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Bytes per element
    if dtype == torch.float32:
        bytes_per_element = 4
    elif dtype == torch.float16:
        bytes_per_element = 2
    else:
        bytes_per_element = 4  # Default assumption
    
    # Parameter memory
    param_count = sum(p.numel() for p in model_batch.parameters())
    param_memory = param_count * bytes_per_element
    
    # Gradient memory (same as parameters)
    grad_memory = param_memory
    
    # Activation memory (rough estimate)
    input_elements = batch_size * torch.prod(torch.tensor(input_shape))
    # Assume activations are ~2x input size per model
    activation_memory = input_elements * model_batch.num_models * 2 * bytes_per_element
    
    # Convert to MB
    mb_factor = 1024 * 1024
    
    return {
        "parameters_mb": param_memory / mb_factor,
        "gradients_mb": grad_memory / mb_factor,
        "activations_mb": activation_memory / mb_factor,
        "total_estimated_mb": (param_memory + grad_memory + activation_memory) / mb_factor,
    } 