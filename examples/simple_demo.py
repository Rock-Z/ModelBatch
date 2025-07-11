#!/usr/bin/env python3
"""
Simple demo of ModelBatch training 32 MLPs simultaneously.
This demonstrates the core functionality as outlined in Milestone M1.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path so we can import modelbatch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modelbatch import ModelBatch, OptimizerFactory
from modelbatch.callbacks import create_default_callbacks
from modelbatch.optimizer import create_adam_configs
from modelbatch.utils import count_parameters, create_identical_models, random_init_fn


class SimpleMLP(nn.Module):
    """Simple MLP for demonstration."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x):
        return self.layers(x)


def create_dummy_data(num_samples: int = 1000, input_size: int = 784, num_classes: int = 10):
    """Create dummy classification data for testing."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def benchmark_sequential_vs_batch(num_models: int = 32, num_epochs: int = 3):
    """Compare sequential training vs ModelBatch training."""
    
    print(f"🚀 ModelBatch Demo: Training {num_models} MLPs")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data
    train_dataset = create_dummy_data(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print(f"Dataset: {len(train_dataset)} samples, batch size: 64")
    
    # 1. Sequential training (baseline)
    print("\n📊 Sequential Training (Baseline)")
    print("-" * 40)
    
    models = create_identical_models(
        SimpleMLP, 
        {"input_size": 784, "hidden_size": 128, "num_classes": 10},
        num_models,
        random_init_fn,
    )
    
    # Move models to device
    for model in models:
        model.to(device)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        for model_idx, model in enumerate(models):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
    
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f}s")
    
    # 2. ModelBatch training (vectorized)
    print("\n⚡ ModelBatch Training (Vectorized)")
    print("-" * 40)
    
    # Create fresh models for fair comparison
    models = create_identical_models(
        SimpleMLP, 
        {"input_size": 784, "hidden_size": 128, "num_classes": 10},
        num_models,
        random_init_fn,
    )
    
    # Create ModelBatch
    model_batch = ModelBatch(models, shared_input=True)
    model_batch.to(device)
    
    # Print model info
    param_info = count_parameters(model_batch)
    print(f"Total parameters: {param_info['total_params']:,}")
    print(f"Parameters per model: {param_info['params_per_model']:,}")
    
    # Create optimizer with different learning rates for each model
    learning_rates = [0.001 * (0.8 ** i) for i in range(num_models)]  # Exponential decay
    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer_configs = create_adam_configs(learning_rates)
    optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)
    
    # Create callbacks
    callbacks = create_default_callbacks(log_frequency=10)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        model_batch.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model_batch(data)  # [num_models, batch_size, num_classes]
            
            # Compute loss
            loss = model_batch.compute_loss(outputs, target, F.cross_entropy)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Execute callbacks
            callbacks.on_train_step(model_batch, batch_idx)
        
        avg_loss = total_loss / num_batches
        print(f"Average loss: {avg_loss:.4f}")
        
        # Print per-model losses
        if model_batch.latest_losses is not None:
            losses = model_batch.latest_losses.cpu().numpy()
            print(f"Per-model losses - Min: {losses.min():.4f}, Max: {losses.max():.4f}, Std: {losses.std():.4f}")
    
    batch_time = time.time() - start_time
    print(f"\nModelBatch time: {batch_time:.2f}s")
    
    # 3. Compare results
    print("\n🏆 Performance Comparison")
    print("=" * 40)
    speedup = sequential_time / batch_time
    print(f"Sequential time:  {sequential_time:.2f}s")
    print(f"ModelBatch time:  {batch_time:.2f}s")
    print(f"Speedup:          {speedup:.1f}x")
    
    if speedup > 5.0:
        print("🎉 Excellent speedup achieved!")
    elif speedup > 2.0:
        print("✅ Good speedup achieved!")
    else:
        print("⚠️  Lower than expected speedup - may need optimization")
    
    # 4. Test model saving/loading
    print("\n💾 Testing Save/Load Functionality")
    print("-" * 40)
    
    save_dir = "model_batch_checkpoint"
    print(f"Saving models to: {save_dir}")
    model_batch.save_all(save_dir)
    
    # Create new ModelBatch and load
    new_models = create_identical_models(
        SimpleMLP, 
        {"input_size": 784, "hidden_size": 128, "num_classes": 10},
        num_models,
    )
    new_model_batch = ModelBatch(new_models)
    new_model_batch.to(device)
    
    print("Loading saved models...")
    new_model_batch.load_all(save_dir)
    print("✅ Save/load test completed successfully")
    
    # Clean up
    import shutil
    shutil.rmtree(save_dir, ignore_errors=True)
    
    print("\n🎯 Demo completed successfully!")
    return speedup


if __name__ == "__main__":
    # Test with different numbers of models
    test_configs = [
        {"num_models": 8, "num_epochs": 2},
        {"num_models": 32, "num_epochs": 3},
    ]
    
    speedups = []
    
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"Running test with {config['num_models']} models, {config['num_epochs']} epochs")
        print(f"{'='*80}")
        
        speedup = benchmark_sequential_vs_batch(**config)
        speedups.append(speedup)
        
        print(f"\nTest completed with {speedup:.1f}x speedup")
    
    print(f"\n{'='*80}")
    print("📋 FINAL SUMMARY")
    print(f"{'='*80}")
    
    for i, (config, speedup) in enumerate(zip(test_configs, speedups)):
        print(f"Test {i+1}: {config['num_models']} models → {speedup:.1f}x speedup")
    
    avg_speedup = sum(speedups) / len(speedups)
    print(f"\nAverage speedup: {avg_speedup:.1f}x")
    
    if avg_speedup >= 10.0:
        print("🏆 MILESTONE M1 ACHIEVED: 10x+ speedup with 32 MLPs!")
    elif avg_speedup >= 5.0:
        print("🎯 Good progress: 5x+ speedup achieved!")
    else:
        print("📈 Basic functionality working, room for optimization") 