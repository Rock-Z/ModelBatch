# Getting Started

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
uv sync --dev
uv pip install -e ".[dev]"
```

## Basic Usage

```python
import torch
from modelbatch import ModelBatch

# Create multiple models
num_models = 4  # choose the number of models to batch
models = [SimpleNet() for _ in range(num_models)]

# Wrap with ModelBatch
mb = ModelBatch(models, lr_list=[0.001] * num_models, optimizer_cls=torch.optim.Adam)

# Train models together
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

# Build docs
uv run mkdocs build -s
```
