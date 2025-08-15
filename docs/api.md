# API Reference

This reference covers the core `ModelBatch` container, data routing utilities, and
optimizer helpers. Shapes follow PyTorch conventions with batch dimensions shown
explicitly.

## Core

### `ModelBatch`

```python
class modelbatch.core.ModelBatch(models: list[nn.Module], shared_input: bool = True)
```

Stacks parameters from identical models and executes them in parallel with
`torch.vmap`.

#### Parameters

- **models** (`list[nn.Module]`): Models with identical structure.
- **shared_input** (`bool`, optional): If `True`, all models receive the same
  input tensor. Default is `True`.

#### `forward(inputs: torch.Tensor) -> torch.Tensor`

Vectorized forward pass.

- **inputs** (`Tensor`):
  - `[batch, ...]` when `shared_input=True`
  - `[num_models, batch, ...]` when `shared_input=False`
- **returns** (`Tensor`): `[num_models, batch, ...]`

```python
batch = torch.randn(64, 3, 32, 32)
mb = ModelBatch(models)
outputs = mb(batch)  # [num_models, 64, 10]
```

#### `zero_grad(set_to_none: bool = True) -> None`

Clear gradients for all parameters. Pass `set_to_none=False` to zero tensors
instead of setting to `None`.

#### `compute_loss(outputs: torch.Tensor, targets: torch.Tensor, loss_fn: Callable, reduction: str = "mean") -> torch.Tensor`

Compute per-model losses.

- **outputs** (`Tensor`): `[num_models, batch, ...]`
- **targets** (`Tensor`):
  - `[batch, ...]` for shared targets
  - `[num_models, batch, ...]` for per-model targets
- **returns** (`Tensor`):
  - scalar when `reduction="mean"` or `"sum"`
  - `[num_models]` when `reduction="none"`

```python
loss = mb.compute_loss(outputs, targets, torch.nn.functional.cross_entropy)
loss.backward()
```

#### `get_model_states() -> list[dict[str, torch.Tensor]]`
Return individual `state_dict` objects in `[num_models]` list form.

#### `load_model_states(states: list[dict[str, torch.Tensor]]) -> None`
Load per-model states produced by `get_model_states`.

#### `save_all(path: str) -> None`
Persist all model states under `path/model_{i}.pt`.

#### `load_all(path: str) -> None`
Load states saved by `save_all`.

#### `enable_compile(**kwargs) -> None`
Wrap the internal model with `torch.compile`.

#### `metrics() -> dict[str, float]`
Return latest per-model metrics such as `{"loss_model_0": 0.1}`.

## Data

### `DataRouter`

```python
class modelbatch.data.DataRouter(mode: str = "passthrough")
```

Routes a batch to specific models using masks or index tensors.

- **mode** (`str`): `"passthrough"`, `"mask"`, or `"indices"`.

#### `route_batch(batch: torch.Tensor, masks: list[torch.Tensor] | None = None, indices: list[torch.Tensor] | None = None) -> torch.Tensor`

- **batch** (`Tensor`): `[batch, ...]`
- **masks** (`list[Tensor]`, optional): one boolean mask per model, `[batch]`.
- **indices** (`list[Tensor]`, optional): one index tensor per model.
- **returns** (`Tensor`):
  - passthrough: original batch `[batch, ...]`
  - mask/indices: `[num_models, max_subset, ...]`

```python
router = DataRouter(mode="mask")
masked = router.route_batch(batch, masks=create_random_masks(len(batch), num_models))
```

### `StratifiedRouter`

```python
class modelbatch.data.StratifiedRouter(num_models: int, strategy: str = "round_robin")
```

Generates stratified index tensors to balance data across models.

- **num_models** (`int`): number of models.
- **strategy** (`str`): `"round_robin"`, `"random"`, or `"class_based"`.

#### `create_stratified_indices(labels: torch.Tensor, num_classes: int | None = None) -> list[torch.Tensor]`

Return index tensors per model ensuring each model sees a balanced subset of
`labels` (`[batch]`).

### `create_random_masks(batch_size: int, num_models: int, subset_ratio: float = 0.8) -> list[torch.Tensor]`

Generate boolean masks `[batch]` for each model selecting a random subset of the
batch.

## Optimizer

### `OptimizerFactory`

```python
class modelbatch.optimizer.OptimizerFactory(optimizer_cls: type[Optimizer], base_config: dict[str, Any] | None = None)
```

Build optimizers and schedulers with per-model parameter groups.

- **optimizer_cls** (`type[Optimizer]`): class like `torch.optim.SGD`.
- **base_config** (`dict[str, Any]`, optional): shared config for all groups.

#### `create_optimizer(model_batch: ModelBatch, configs: list[dict[str, Any]]) -> Optimizer`

Return optimizer with one parameter group per model. `configs` should have length
`num_models`.

#### `create_lr_scheduler(optimizer: Optimizer, scheduler_cls: type, configs: list[dict[str, Any]]) -> list`

Return a list of schedulers, one per parameter group.

### Helper Functions

- `create_sgd_configs(learning_rates: list[float], momentum: float = 0.9, weight_decay: float = 1e-4) -> list[dict[str, Any]]`
- `create_adam_configs(learning_rates: list[float], betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> list[dict[str, Any]]`
- `create_lr_sweep_configs(min_lr: float, max_lr: float, num_models: int, scale: str = "log") -> list[dict[str, float]]`

```python
factory = OptimizerFactory(torch.optim.SGD)
opt = factory.create_optimizer(mb, create_sgd_configs([1e-2] * mb.num_models))
```

