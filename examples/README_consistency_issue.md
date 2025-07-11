# ModelBatch Consistency Issue Demo

This demo originally reproduced an issue where ModelBatch training with different
learning rates diverged from sequential training. The root cause turned out to be
stacked parameter gradients accumulating between steps. ModelBatch now exposes a
`zero_grad()` method and the factory wired `optimizer.zero_grad()` to clear those
gradients automatically. With seeds reset before creating the second set of models
(`torch.manual_seed(42)` and `np.random.seed(42)`), the quick test shows identical
results.

## Problem Description

When training multiple models with different learning rates, ModelBatch produces different results compared to training the same models sequentially. This suggests there may be issues with:

1. **Optimizer state handling**: Per-model optimizer states might not be properly isolated
2. **Parameter updates**: Gradient updates might interfere between models
3. **Vectorized computation**: Subtle differences in how torch.vmap handles operations
4. **Memory sharing**: Parameters or gradients might be inadvertently shared

## Running the Demo

```bash
# From the project root
uv run examples/quick_consistency_test.py
```

## Expected Output

The demo will show significant differences between sequential and ModelBatch training:

```
⚡ Quick Consistency Test
==================================================
Models: 3, Epochs: 3
Learning rates: [0.001, 0.01, 0.1]
Architecture: 8→4→2
Data: 100 samples

📊 Sequential Training
  Model 0: 49.3%
  Model 1: 67.3%
  Model 2: 80.7%

⚡ ModelBatch Training
  Model 0: 50.7%
  Model 1: 76.7%
  Model 2: 80.7%

🔍 Comparison
  Model 0: Seq=49.3%, MB=50.7%, Diff=1.3%
  Model 1: Seq=67.3%, MB=76.7%, Diff=9.3%
  Model 2: Seq=80.7%, MB=80.7%, Diff=0.0%

Max difference: 9.3%
❌ DIVERGENT: Significant difference detected!
```

## Demo Features

- **Lightweight**: Uses synthetic data, runs on CPU
- **Reproducible**: Fixed random seeds for consistent results
- **Configurable**: Easy to modify learning rates, model sizes, epochs
- **Clear comparison**: Side-by-side results with difference calculations

## Configuration

The demo uses:
- 4 models with different learning rates: [0.001, 0.003, 0.01, 0.03]
- Simple MLP architecture: 20→16→16→4
- Synthetic classification data (500 samples, 4 classes)
- 10 training epochs

## Quick Test Version

For even faster testing, you can modify the configuration in `main()`:

```python
# Faster version
num_models = 3
num_epochs = 5
learning_rates = [0.001, 0.01, 0.1]
```

## Investigation Areas

To debug this issue, focus on:

1. **OptimizerFactory**: Check if per-model parameter groups are correctly isolated
2. **Parameter stacking**: Verify that parameter updates don't leak between models
3. **Gradient computation**: Look for differences in how gradients are computed in vmap
4. **Adam optimizer state**: Ensure momentum and RMSprop buffers are per-model

## Related Files

- `src/modelbatch/core.py`: Main ModelBatch implementation
- `src/modelbatch/optimizer.py`: OptimizerFactory and per-model optimization
- `examples/simple_demo.py`: Original performance benchmark
- `examples/cifar10_lenet_benchmark.py`: CIFAR-10 version (heavier)

## Notes

This demo isolates the consistency issue from performance concerns, making it easier to debug the root cause of the divergence between sequential and vectorized training. 