**Design Doc – ModelBatch**
*Version 0.3 • 10 Jul 2025*

---

### 1 · Objective (Expanded)

Modern GPUs idle when training sub-million-parameter networks: one tiny model rarely exceeds ≈5 TFLOP/s of a 300 TFLOP/s device and <5 % memory bandwidth. **ModelBatch** eliminates that waste by **training hundreds–thousands of *independent* PyTorch models in a single vectorised step**.

Primary goals

| Goal                                    | Detail                                                                                                                                              |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Max-out single-GPU throughput**       | Merge forward/backward of *N* structurally identical models with `torch.vmap`, achieving near-linear speedup until VRAM or compute saturates.       |
| **Hyper-parameter sweeps on one card**  | Treat each model as a “trial” (e.g., different LR, seed) yet avoid per-trial processes; supports rapid search when GPU count is limited.            |
| **Plug-and-play**                       | Drop-in replacement for `nn.Module` inside raw PyTorch loops, HuggingFace *Trainer*, or PyTorch Lightning with minimal code changes.                |
| **Data-loading consolidation**          | Copy each batch to GPU once; all models reuse it, slashing CPU/I/O overhead.                                                                        |
| **Per-model isolation**                 | Separate parameter tensors, optimisers, and metrics; failure or divergence in one model can be quarantined without crashing others (via callbacks). |
| **Transparent logging & checkpointing** | Per-model loss/metric streams to W\&B/TensorBoard; one-line save/load of all weights.                                                               |

Out-of-scope for v1.0: heterogeneous architectures within one batch, fully dynamic dataset routing, and automated multi-GPU sharding.

---

### 2 · Expected Behaviour

| Feature       | Behaviour                                                                             |
| ------------- | ------------------------------------------------------------------------------------- |
| Construction  | `mb = ModelBatch(models, lr_list, optimizer_cls, base_batch)`                         |
| Forward       | Single `torch.vmap` over stacked params → output `[N, …]`; shared input by default.   |
| Loss          | Returns `combined_loss` (mean of per-model losses) + exposes `mb.latest_losses`.      |
| Optimiser     | One optimiser with **N param-groups** (per-model LR/WD).                              |
| Data          | One DataLoader; optional mask for sample-level routing.                               |
| Logging hooks | `mb.metrics()` yields dict per model for callbacks.                                   |
| Checkpointing | `mb.save_all(dir)` / `mb.load_all(dir)` store per-model `state_dict`s + hyper-params. |

---

### 3 · Architecture & Key Components

1. **`ModelBatch` (nn.Module)**

   * Stacks params/buffers via `torch.func.stack_module_state`.
   * Wraps functional call with `torch.vmap` (`in_dims=(0,0,None)`).
   * Keeps `self.models` for extraction/eval.
2. **`OptimizerFactory`**

   * Creates one optimiser (`optimizer_cls`) with param-groups per model.
   * AMP support through shared `GradScaler`.
3. **`DataRouter` v1.1**

   * Optional mask/indices to filter batch per model; default passthrough.
4. **Framework Adapters**

   * **HFTrainerMixin** – overrides `create_optimizer`, `compute_loss`, `evaluate`.
   * **LightningModuleExample** – vectorised `training_step`/`validation_step`.
5. **CallbackPack**

   * Logs `{loss_model_i}`, `{val_acc_model_i}`; freezes NaN models.

---

### 4 · Integrations

| Tool                            | Integration Point                                                            |
| ------------------------------- | ---------------------------------------------------------------------------- |
| **Raw PyTorch**                 | `train_one_epoch(mb, loader, optim)` helper.                                 |
| **HuggingFace**                 | `BundledTrainer` subclass with mix-in.                                       |
| **PyTorch Lightning**           | Ready-to-use `LightningModule`.                                              |
| **W\&B / TensorBoard**          | Single run logs per-model metrics; supports W\&B *groups* if desired.        |
| **Optuna / Ray Tune**           | User supplies list of configs → one ModelBatch run; returns dict of results. |
| **torch.compile / CUDA Graphs** | `mb.enable_compile()` / `mb.capture_cuda_graph()` toggles.                   |

---

### 5 · Benchmarking Goals

| Metric          | Target (A100 40 GB, FP16, CIFAR-10, ResNet-8 @75 k params) |
| --------------- | ---------------------------------------------------------- |
| GPU utilisation | >70 % with 256 models (vs <5 % baseline)                   |
| Throughput      | ≥100× images/s vs sequential training                      |
| Scaling         | ≤20 % slowdown when doubling models until VRAM limit       |
| Memory          | ≤1.5 GB + 2 MB × #models (FP16)                            |

Benchmark script `bench.py --models 1 16 64 256 512` outputs CSV + plot for README.

---

### 6 · Milestones

| #  | Deliverable                         | Notes                    |
| -- | ----------------------------------- | ------------------------ |
| M1 | Prototype `ModelBatch` + raw demo   | 32 MLPs 10× faster.      |
| M2 | OptimiserFactory + AMP              | Per-model LR verified.   |
| M3 | HF `BundledTrainer` + W\&B callback | Classification notebook. |
| M4 | Lightning example + docs            | CI tests.                |
| M5 | Benchmark suite & results           | Publish in README.       |
| M6 | v1.0 release (PyPI)                 | Tutorial & video.        |

---

### 7 · Risks & Mitigations

| Risk                           | Mitigation                                         |
| ------------------------------ | -------------------------------------------------- |
| Missing `vmap` rules           | Detect, fallback loop, warn user.                  |
| Divergent model corrupts grads | Callback zeros grads, removes model from batch.    |
| Optimiser step bottleneck      | Explore TorchOpt fused/functional updates in v1.2. |

---

**Hand-off:** implement components in §3, hit benchmarks §5, follow roadmap §6.
