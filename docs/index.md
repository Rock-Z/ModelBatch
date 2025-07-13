# ModelBatch

**Train tens to hundreds of independent PyTorch models simultaneously on a single GPU using vectorized operations.**

## Overview

ModelBatch eliminates GPU waste by training multiple independent models in a single vectorized step using `torch.vmap`. Achieve near-linear speedup until VRAM or compute saturates.

### Key Features

- **Massive Speedups**: almost linear speedup with correct setup
- **Single GPU Efficiency**: Max out GPU utilization with many small models
- **Drop-in Replacement**: Minimal code changes to existing PyTorch workflows
- **Framework Integration**: (Hopes to) work with HuggingFace, PyTorch Lightning
- **Per-model Isolation**: Separate parameters, optimizers, and metrics

## Quick Start

### Installation

This repo uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
uv sync --dev
uv pip install -e ".[dev]"
```

### Basic Usage

```python
import torch
from modelbatch import ModelBatch

# Create multiple models
models = [SimpleNet() for _ in range(32)]

# Wrap with ModelBatch
mb = ModelBatch(models, lr_list=[0.001] * 32, optimizer_cls=torch.optim.Adam)

# Train normally - but 32x faster!
for batch in dataloader:
    mb.zero_grad()
    outputs = mb(batch)
    loss = mb.compute_loss(outputs, targets)
    loss.backward()
    mb.step()
```

## Quick Commands

```bash
# Run demo
uv run examples/simple_demo.py

# Run tests  
uv run -m pytest

# Local docs
uv run mkdocs serve
```

## Documentation

- **[Core Design](design.md)**: Architecture and goals
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)**: Technical details & progress tracking
- **[Development Workflow](../AGENTS.md)**: Instructions for development, LLMs and humans alike