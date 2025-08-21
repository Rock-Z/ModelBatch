# ModelBatch

**Train many independent PyTorch models simultaneously on a single GPU using vectorized operations.**

ModelBatch eliminates GPU waste by training multiple independent models in a single vectorized step using `torch.vmap`. Achieve near-linear speedup until VRAM or compute saturates.

## Key Features

- **Massive Speedups**: almost linear speedup with correct setup
- **Single GPU Efficiency**: Max out GPU utilization with many small models
- **Drop-in Replacement**: Minimal code changes to existing PyTorch workflows
- **Framework Integration**: (Hopes to) work with HuggingFace, PyTorch Lightning
- **Per-model Isolation**: Separate parameters, optimizers, and metrics

## Getting Started

New to ModelBatch? See the [getting started guide](getting_started.md) for installation, a basic example, and common development commands.

## API Reference

The [API reference](api.md) covers the core `ModelBatch` class, data routing helpers,
and optimizer utilities.

## Additional Resources

- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)**: Technical details & progress tracking
- **[Development Workflow](https://github.com/rock-z/ModelBatch/blob/main/AGENTS.md)**: Instructions for development, LLMs and humans alike
