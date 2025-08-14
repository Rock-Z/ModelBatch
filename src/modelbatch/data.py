"""
Data routing utilities for ModelBatch.
"""

from __future__ import annotations

import torch


class DataRouter:
    """
    Routes data to specific models in a ModelBatch.

    Supports filtering batches per model using masks or indices.
    By default, all models receive the same data (passthrough).
    """

    def __init__(self, mode: str = "passthrough"):
        """
        Initialize data router.

        Args:
            mode: Routing mode - "passthrough", "mask", or "indices"
        """
        if mode not in ["passthrough", "mask", "indices"]:
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = mode

    def route_batch(
        self,
        batch: torch.Tensor,
        masks: list[torch.Tensor] | None = None,
        indices: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Route batch data to models.

        Args:
            batch: Input batch [batch_size, ...]
            masks: List of boolean masks, one per model
            indices: List of index tensors, one per model

        Returns:
            Routed data - format depends on mode:
            - passthrough: returns original batch
            - mask/indices: returns [num_models, max_subset_size, ...]
        """
        if self.mode == "passthrough":
            return batch

        if self.mode == "mask":
            if masks is None:
                raise ValueError("Masks required for mask mode")

            return self._route_by_mask(batch, masks)

        if self.mode == "indices":
            if indices is None:
                raise ValueError("Indices required for indices mode")

            return self._route_by_indices(batch, indices)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _route_by_mask(
        self,
        batch: torch.Tensor,
        masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Route batch using boolean masks."""
        batch_size = batch.shape[0]

        # Validate masks
        for i, mask in enumerate(masks):
            if mask.shape[0] != batch_size:
                raise ValueError(
                    f"Mask {i} has length {mask.shape[0]}, expected {batch_size}",
                )

        # Find maximum subset size
        max_subset_size = max(int(mask.sum().item()) for mask in masks)

        if max_subset_size == 0:
            raise ValueError("All masks are empty")

        # Create output tensor
        output_shape = (len(masks), max_subset_size, *batch.shape[1:])
        routed_batch = torch.zeros(output_shape, dtype=batch.dtype, device=batch.device)

        # Fill with masked data (pad with zeros if needed)
        for i, mask in enumerate(masks):
            subset = batch[mask]
            subset_size = subset.shape[0]
            if subset_size > 0:
                routed_batch[i, :subset_size] = subset

        return routed_batch

    def _route_by_indices(
        self,
        batch: torch.Tensor,
        indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """Route batch using index tensors."""
        batch_size = batch.shape[0]

        # Validate indices
        for i, idx in enumerate(indices):
            # Allow empty index tensors; they simply contribute no samples
            if idx.numel() == 0:
                continue
            # Ensure correct dtype for indexing
            if idx.dtype not in (torch.long, torch.int64):
                idx = idx.to(dtype=torch.long)
                indices[i] = idx
            # Bounds check
            if idx.max().item() >= batch_size or idx.min().item() < 0:
                raise ValueError(
                    f"Indices {i} out of range for batch size {batch_size}"
                )

        # Find maximum subset size
        max_subset_size = max(len(idx) for idx in indices)

        if max_subset_size == 0:
            raise ValueError("All index lists are empty")

        # Create output tensor
        output_shape = (len(indices), max_subset_size, *batch.shape[1:])
        routed_batch = torch.zeros(output_shape, dtype=batch.dtype, device=batch.device)

        # Fill with indexed data (pad with zeros if needed)
        for i, idx in enumerate(indices):
            subset = batch[idx]
            subset_size = subset.shape[0]
            if subset_size > 0:
                routed_batch[i, :subset_size] = subset

        return routed_batch


class StratifiedDataRouter(DataRouter):
    """
    Data router that stratifies data based on labels/classes.

    Useful for ensuring each model sees balanced data or specific
    class distributions.
    """

    def __init__(self, num_models: int, strategy: str = "round_robin"):
        """
        Initialize stratified router.

        Args:
            num_models: Number of models to route to
            strategy: Stratification strategy - "round_robin", "random", "class_based"
        """
        super().__init__(mode="indices")
        self.num_models = num_models
        self.strategy = strategy

    def create_stratified_indices(
        self,
        labels: torch.Tensor,
        num_classes: int | None = None,
    ) -> list[torch.Tensor]:
        """
        Create stratified indices for routing.

        Args:
            labels: Class labels [batch_size]
            num_classes: Number of classes (inferred if None)

        Returns:
            List of index tensors, one per model
        """
        if num_classes is None:
            num_classes = int(labels.max().item()) + 1

        if self.strategy == "round_robin":
            return self._round_robin_indices(labels)
        if self.strategy == "random":
            return self._random_indices(labels)
        if self.strategy == "class_based":
            return self._class_based_indices(labels, int(num_classes))
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _round_robin_indices(
        self,
        labels: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Distribute samples round-robin across models."""
        batch_size = labels.shape[0]
        indices_per_model = [[] for _ in range(self.num_models)]

        for i in range(batch_size):
            model_idx = i % self.num_models
            indices_per_model[model_idx].append(i)

        return [
            torch.tensor(indices, dtype=torch.long) for indices in indices_per_model
        ]

    def _random_indices(self, labels: torch.Tensor) -> list[torch.Tensor]:
        """Randomly distribute samples across models."""
        batch_size = labels.shape[0]
        perm = torch.randperm(batch_size)

        chunk_size = batch_size // self.num_models
        indices_per_model = []

        for i in range(self.num_models):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_models - 1 else batch_size
            indices_per_model.append(perm[start_idx:end_idx])

        return indices_per_model

    def _class_based_indices(
        self,
        labels: torch.Tensor,
        num_classes: int,
    ) -> list[torch.Tensor]:
        """Distribute samples ensuring each model sees all classes."""
        indices_per_model = [[] for _ in range(self.num_models)]

        # Group indices by class
        for class_id in range(num_classes):
            class_mask = labels == class_id
            class_indices = torch.where(class_mask)[0]

            # Distribute this class across models
            for i, idx in enumerate(class_indices):
                model_idx = i % self.num_models
                indices_per_model[model_idx].append(idx.item())

        return [
            torch.tensor(indices, dtype=torch.long) for indices in indices_per_model
        ]


def create_random_masks(
    batch_size: int,
    num_models: int,
    subset_ratio: float = 0.8,
) -> list[torch.Tensor]:
    """
    Create random boolean masks for data routing.

    Args:
        batch_size: Size of the batch
        num_models: Number of models
        subset_ratio: Fraction of batch each model should see

    Returns:
        List of boolean mask tensors
    """
    subset_size = int(batch_size * subset_ratio)
    masks = []

    for _ in range(num_models):
        # Create random mask
        indices = torch.randperm(batch_size)[:subset_size]
        mask = torch.zeros(batch_size, dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)

    return masks
