# ModelBatch

**Train many independent PyTorch models simultaneously on a single GPU using vectorized operations.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

## ⚠️ Current Status

**ModelBatch is still in active development. Core functionality is tested and working, but the API may be subject to change.**

## 🚀 Quick Start

### Installation

```bash
uv sync --dev
uv pip install -e ".[dev]"
```

### Basic Example

```python
import torch
from modelbatch import ModelBatch

# Create multiple models
num_models = 4  # choose the number of models to batch
models = [SimpleNet() for _ in range(num_models)]

# Wrap with ModelBatch - that's it!
mb = ModelBatch(models, lr_list=[0.001] * num_models, optimizer_cls=torch.optim.Adam)

# Train normally (but many times faster!), batched across models
for batch in dataloader:
    mb.zero_grad()
    outputs = mb(batch)
    loss = mb.compute_loss(outputs, targets)  
    loss.backward()
    mb.step()
```

See [here](examples) for more examples.

## 📚 Documentation

See [docs](https://rock-z.github.io/ModelBatch/).

## 🛠️ Development

### Environment Setup

```bash
uv sync --dev
```

### Commands

```bash
# Tests (currently showing failures)
uv run -m pytest

# Linting  
uv run ruff check --fix . && uv run ruff format .

# Documentation
uv run mkdocs serve
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
