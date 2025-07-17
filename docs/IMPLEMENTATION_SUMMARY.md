# ModelBatch Implementation Summary

*Status as of July 13 2025*

## 🎯 Project Overview
ModelBatch is a library for training many independent PyTorch models simultaneously on a single GPU by grouping them into a single `ModelBatch` object. This enables efficient hyperparameter sweeps and maximizes GPU utilization.

## ⚠️ Current Status: Active Development

### Implementation Milestones

✅ **M1**: Core ModelBatch + demo (90% complete, failing more complex models)
  - ModelBatch class with parameter stacking
  - OptimizerFactory for per-model optimizer configs
  - DataRouter for data filtering
  - CallbackPack for monitoring
  - Working demo with performance benchmarks

✅ **M2**: OptimizerFactory + AMP (90% complete, consistency issues compared to sequential training)  
  - OptimizerFactory for per-model optimizer configs -- different lrs tested and passing
  - AMP support with GradScaler -- NOT TESTED

🔄 **M3**: HuggingFace integration  
🔄 **M4**: Lightning example + docs  
🔄 **M5**: Benchmark suite  
🔄 **M6**: v1.0 release

### Known Issues
**test suite does not pass**

1. **Training Equivalence**: Batched training now matches sequential training *unless* dropout is used. Dropout randomness remains different despite setting seeds.
   - See `examples/cifar10_lenet_benchmark.py`
2. ~~**LSTM Models**~~ (dropped for now): `RuntimeError: Batching rule not implemented for aten::lstm.input`. This is because LSTM module is not supported by `torch.vmap`.

## 🏗️ Project Structure

```
ModelBatch/
├── src/modelbatch/
│   ├── core.py          # ModelBatch class (main component)
│   ├── optimizer.py     # OptimizerFactory + AMP support
│   ├── data.py          # DataRouter for data filtering
│   ├── callbacks.py     # CallbackPack for monitoring
│   └── utils.py         # Training utilities
├── tests/               # Unit tests
├── examples/            # Demo scripts
└── docs/               # Design documentation
```
