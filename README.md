# ModelBatch

**Train tens to hundreds of independent PyTorch models simultaneously on a single GPU using vectorized operations.**

[![Tests](https://github.com/your-username/ModelBatch/workflows/tests/badge.svg)](https://github.com/your-username/ModelBatch/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

## ⚠️ Current Status

**ModelBatch is in active development with known issues:**

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
models = [SimpleNet() for _ in range(32)]

# Wrap with ModelBatch - that's it!
mb = ModelBatch(models, lr_list=[0.001] * 32, optimizer_cls=torch.optim.Adam)

# Train normally, but 32x faster
for batch in dataloader:
    mb.zero_grad()
    outputs = mb(batch)
    loss = mb.compute_loss(outputs, targets)  
    loss.backward()
    mb.step()
```

There are more examples in the [examples](examples) directory.

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

# Type checking
uv run mypy src tests

# Documentation
uv run mkdocs serve
```

## 📚 Documentation

- **[Full Documentation](https://rock-z.github.io/ModelBatch/)**
- **[Core Design](docs/design.md)**: Architecture and goals
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)**: Technical details and current issues
- **[Development Guide](AGENTS.md)**: Workflow and status

## 📄 License

This project is licensed under the [MIT License](LICENSE).
