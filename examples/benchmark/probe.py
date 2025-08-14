"""BERT Token-Level POS Benchmark: Train multiple layer probes simultaneously."""

from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import set_random_seeds
from modelbatch import ModelBatch, OptimizerFactory, DataRouter
from modelbatch.data import StratifiedDataRouter
from modelbatch.optimizer import create_adam_configs
from modelbatch.utils import count_parameters


class BERTTokenLevelProbe(nn.Module):
    """Linear probe for token-level POS classification on BERT layer representations."""
    
    def __init__(self, target_layer: int, hidden_size: int = 768, num_pos_tags: int = 17, dropout_rate: float = 0.1):
        super().__init__()
        self.target_layer = target_layer
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_pos_tags)
    
    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        # token_features: [batch_size, seq_len, hidden_size]
        x = self.dropout(token_features)
        return self.classifier(x)  # [batch_size, seq_len, num_pos_tags]


class BERTFeatureExtractor(nn.Module):
    """Real BERT feature extractor that outputs representations from all layers."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.num_layers = len(self.bert.encoder.layer)  # BERT layers
        
        # Freeze BERT weights
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        """Extract features from all BERT layers."""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states  # tuple of (layer_0, layer_1, ..., layer_n)
            
            layer_outputs = {}
            # Skip embedding layer (index 0), use transformer layers 1-6
            for i in range(1, len(hidden_states)):
                layer_outputs[i] = hidden_states[i]
            
            return layer_outputs


class UDPOSDataset(Dataset):
    """Real dataset for token-level POS classification."""
    
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                 labels: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


def load_ud_pos_data(batch_size: int = 16, max_length: int = 128, 
                     model_name: str = "bert-base-uncased"):
    """Load batterydata/pos_tagging and create DataLoaders with proper tokenization."""
    
    print("Loading batterydata/pos_tagging dataset...")
    dataset = load_dataset("batterydata/pos_tagging")
    print("Successfully loaded batterydata/pos_tagging")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def process_split(split_data):
        """Process a dataset split and return tokenized inputs with aligned labels."""
        sentences = []
        pos_sequences = []
        all_pos_tags = set()
        
        for example in split_data:
            tokens = None
            pos_tags = None
            
            # Try different field names for tokens
            for field in ['tokens', 'words', 'text', 'sentence']:
                if field in example and example[field]:
                    tokens = example[field]
                    break
            
            # Try different field names for POS tags
            for field in ['pos_tags', 'upos', 'tags', 'pos', 'labels', 'ner_tags']:
                if field in example and example[field]:
                    pos_tags = example[field]
                    break
            
            if tokens and pos_tags and len(tokens) == len(pos_tags):
                sentences.append(tokens)
                pos_sequences.append(pos_tags)
                if pos_tags and isinstance(pos_tags[0], str):
                    all_pos_tags.update(pos_tags)
        
        if not sentences:
            raise ValueError("Could not extract valid sentences and POS tags from dataset")
        
        # Create tag mapping for string tags
        if all_pos_tags:
            tag_to_id = {tag: i for i, tag in enumerate(sorted(list(all_pos_tags)))}
            pos_sequences = [[tag_to_id.get(tag, -100) for tag in seq] for seq in pos_sequences]
            num_tags = len(tag_to_id)
        else:
            # Integer tags - find max value
            num_tags = max(max(seq) for seq in pos_sequences) + 1
        
        return sentences, pos_sequences, num_tags
    
    # Process train and test splits
    train_sentences, train_pos_sequences, num_pos_tags = process_split(dataset['train'])
    try:
        test_sentences, test_pos_sequences, _ = process_split(dataset['test'])
    except KeyError:
        # Use validation or split train data
        if 'validation' in dataset:
            test_sentences, test_pos_sequences, _ = process_split(dataset['validation'])
        else:
            split_idx = len(train_sentences) // 5
            test_sentences = train_sentences[-split_idx:]
            test_pos_sequences = train_pos_sequences[-split_idx:]
            train_sentences = train_sentences[:-split_idx]
            train_pos_sequences = train_pos_sequences[:-split_idx]
    
    print(f"Dataset: batterydata/pos_tagging, POS tags: {num_pos_tags}")
    
    def tokenize_and_align(sentences, pos_sequences):
        """Tokenize sentences and align labels."""
        tokenized = tokenizer(
            sentences, is_split_into_words=True, truncation=True,
            padding=True, max_length=max_length, return_tensors="pt"
        )
        
        aligned_labels = []
        for i, pos_seq in enumerate(pos_sequences):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                elif word_idx < len(pos_seq):
                    label_ids.append(pos_seq[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            # Pad to tokenized length
            max_len = tokenized['input_ids'].shape[1]
            padded = label_ids + [-100] * (max_len - len(label_ids))
            aligned_labels.append(padded[:max_len])
        
        return tokenized, torch.tensor(aligned_labels, dtype=torch.long)
    
    # Tokenize and create datasets
    train_tokenized, train_labels = tokenize_and_align(train_sentences, train_pos_sequences)
    test_tokenized, test_labels = tokenize_and_align(test_sentences, test_pos_sequences)
    
    train_dataset = UDPOSDataset(train_tokenized['input_ids'], train_tokenized['attention_mask'], train_labels)
    test_dataset = UDPOSDataset(test_tokenized['input_ids'], test_tokenized['attention_mask'], test_labels)
    
    # Create DataLoaders
    def seed_worker(_worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(6325)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    return train_loader, test_loader, num_pos_tags


def evaluate_token_accuracy(models_or_batch, bert_extractor: BERTFeatureExtractor, dataloader: DataLoader,
                            target_layers: list[int], device: torch.device, *, is_batch: bool = False) -> list[float]:
    """Evaluate token-level accuracy for layer probes."""
    if is_batch:
        models_or_batch.eval()
        correct = torch.zeros(models_or_batch.num_models, device=device)
        total = torch.zeros(models_or_batch.num_models, device=device)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                layer_outputs = bert_extractor(input_ids, attention_mask)

                # Create round-robin indices per model
                strat_router = StratifiedDataRouter(models_or_batch.num_models, strategy="round_robin")
                rr_indices = strat_router.create_stratified_indices(labels[:, 0])
                subset_sizes = [len(idx) for idx in rr_indices]
                max_subset = max(subset_sizes)

                # Build per-model inputs
                per_model_inputs = torch.zeros(
                    (models_or_batch.num_models, max_subset, layer_outputs[target_layers[0]].shape[1], layer_outputs[target_layers[0]].shape[2]),
                    dtype=layer_outputs[target_layers[0]].dtype, device=device
                )
                for i in range(models_or_batch.num_models):
                    li = target_layers[i]
                    if subset_sizes[i] > 0:
                        per_model_inputs[i, :subset_sizes[i]] = layer_outputs[li][rr_indices[i]]

                logits = models_or_batch(per_model_inputs)
                preds = logits.argmax(dim=3)

                # Route labels and compute accuracy
                base_router = DataRouter(mode="indices")
                labels_routed = base_router.route_batch(labels, indices=rr_indices)
                for i in range(models_or_batch.num_models):
                    if subset_sizes[i] < max_subset:
                        labels_routed[i, subset_sizes[i]:] = -100

                valid_mask = (labels_routed != -100)
                correct += ((preds == labels_routed) & valid_mask).sum(dim=[1, 2]).float()
                total += valid_mask.sum(dim=[1, 2]).float()

        return (100 * correct / torch.clamp(total, min=1)).detach().cpu().tolist()

    # Sequential evaluation
    accuracies: list[float] = []
    for model, layer_idx in zip(models_or_batch, target_layers):
        model.eval()
        correct_total = 0
        valid_total = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                features = bert_extractor(input_ids, attention_mask)[layer_idx]
                logits = model(features)
                preds = logits.argmax(dim=2)

                valid_mask = (labels != -100)
                correct_total += ((preds == labels) & valid_mask).sum().item()
                valid_total += valid_mask.sum().item()

        accuracies.append(100.0 * correct_total / max(valid_total, 1))
    return accuracies


def train_sequential_token_probes(models: list[BERTTokenLevelProbe], bert_extractor: BERTFeatureExtractor,
                                  train_loader: DataLoader, target_layers: list[int], learning_rates: list[float],
                                  num_epochs: int, device: torch.device) -> float:
    """Train token-level probes sequentially."""
    print("Sequential Token-Level Training")
    start_time = time.time()

    for model, layer_idx, lr in zip(models, target_layers, learning_rates):
        set_random_seeds()
        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _epoch in range(num_epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                with torch.no_grad():
                    features = bert_extractor(input_ids, attention_mask)[layer_idx]
                logits = model(features)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss.backward()
                optimizer.step()

    total_time = time.time() - start_time
    print(f"Sequential time: {total_time:.2f}s")
    return total_time


def train_modelbatch_token_probes(models: list[BERTTokenLevelProbe], bert_extractor: BERTFeatureExtractor,
                                  train_loader: DataLoader, target_layers: list[int], learning_rates: list[float],
                                  num_epochs: int, device: torch.device) -> tuple[float, ModelBatch]:
    """Train token-level probes with ModelBatch."""
    print("ModelBatch Token-Level Training")
    set_random_seeds()

    model_batch = ModelBatch(cast(list[nn.Module], models), shared_input=False).to(device)
    param_info = count_parameters(model_batch)
    print(f"Total parameters: {param_info['total_params']:,} ({model_batch.num_models} models)")

    optimizer_factory = OptimizerFactory(torch.optim.Adam)
    optimizer_configs = create_adam_configs(learning_rates)
    optimizer = optimizer_factory.create_optimizer(model_batch, optimizer_configs)

    start_time = time.time()
    router = DataRouter(mode="indices")
    strat_router = StratifiedDataRouter(model_batch.num_models, strategy="round_robin")
    
    for _epoch in range(num_epochs):
        model_batch.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                layer_outputs = bert_extractor(input_ids, attention_mask)

            # Create round-robin indices and build per-model inputs
            rr_indices = strat_router.create_stratified_indices(labels[:, 0])
            subset_sizes = [len(idx) for idx in rr_indices]
            max_subset = max(subset_sizes)

            _, seq_len, hidden = layer_outputs[target_layers[0]].shape
            per_model_inputs = torch.zeros((model_batch.num_models, max_subset, seq_len, hidden),
                                         dtype=layer_outputs[target_layers[0]].dtype, device=device)
            for i in range(model_batch.num_models):
                li = target_layers[i]
                if subset_sizes[i] > 0:
                    per_model_inputs[i, :subset_sizes[i]] = layer_outputs[li][rr_indices[i]]

            # Route labels and train
            labels_routed = router.route_batch(labels, indices=rr_indices)
            for i in range(model_batch.num_models):
                if subset_sizes[i] < max_subset:
                    labels_routed[i, subset_sizes[i]:] = -100

            logits = model_batch(per_model_inputs)

            def token_loss_fn(model_logits: torch.Tensor, model_labels: torch.Tensor) -> torch.Tensor:
                return F.cross_entropy(model_logits.view(-1, model_logits.size(-1)), 
                                     model_labels.view(-1), ignore_index=-100)

            loss = model_batch.compute_loss(logits, labels_routed, token_loss_fn, reduction="mean")
            loss.backward()
            optimizer.step()

    total_time = time.time() - start_time
    print(f"ModelBatch time: {total_time:.2f}s")
    return total_time, model_batch


def run_benchmark(num_layers: int = 6, num_epochs: int = 3, batch_size: int = 16,
                 max_length: int = 128) -> dict[str, float]:
    """Run BERT token-level POS probing benchmark."""
    print(f"BERT Token-Level POS Benchmark: {num_layers} Layer Probes")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    train_loader, test_loader, num_pos_tags = load_ud_pos_data(batch_size, max_length)
    
    # Create BERT feature extractor
    bert_extractor = BERTFeatureExtractor().to(device)
    
    # Create probe models for different layers
    target_layers = list(range(1, min(num_layers + 1, bert_extractor.num_layers + 1)))
    learning_rates = [1e-3 * (0.8 ** i) for i in range(len(target_layers))]
    dropout_rates = [0.1 + 0.02 * i for i in range(len(target_layers))]
    
    set_random_seeds()
    models = [
        BERTTokenLevelProbe(target_layer=layer, num_pos_tags=num_pos_tags, dropout_rate=dropout_rates[i])
        for i, layer in enumerate(target_layers)
    ]
    sample_params = sum(p.numel() for p in models[0].parameters())
    print(f"Parameters per probe: {sample_params:,}")
    
    # Sequential training
    print("\n" + "=" * 60)
    sequential_models = [copy.deepcopy(models[i]) for i in range(len(target_layers))]
    sequential_time = train_sequential_token_probes(
        sequential_models, bert_extractor, train_loader, target_layers,
        learning_rates, num_epochs, device
    )
    
    # ModelBatch training  
    print("\n" + "=" * 60)
    batch_models = [copy.deepcopy(models[i]) for i in range(len(target_layers))]
    batch_time, model_batch = train_modelbatch_token_probes(
        batch_models, bert_extractor, train_loader, target_layers,
        learning_rates, num_epochs, device
    )
    
    # Performance comparison
    speedup = sequential_time / batch_time
    
    print("\nRESULTS")
    print("-" * 30)
    print(f"Sequential: {sequential_time:.2f}s") 
    print(f"ModelBatch: {batch_time:.2f}s")
    print(f"Speedup: {speedup:.1f}x")
    
    # Verify equivalence
    batch_accuracies = evaluate_token_accuracy(
        model_batch, bert_extractor, test_loader, target_layers, device, is_batch=True
    )
    sequential_accuracies = evaluate_token_accuracy(
        sequential_models, bert_extractor, test_loader, target_layers, device, is_batch=False
    )
    
    print(f"\nLayer-wise Accuracies:")
    for i, layer in enumerate(target_layers):
        print(f"Layer {layer}: Sequential={sequential_accuracies[i]:.1f}%, "
              f"ModelBatch={batch_accuracies[i]:.1f}%")
    
    return {
        "num_layers": len(target_layers),
        "sequential_time": sequential_time,
        "batch_time": batch_time,
        "speedup": speedup,
    }


if __name__ == "__main__":
    print("ModelBatch BERT Token-Level POS Benchmark")
    
    print(f"\n{'=' * 60}")
    print("SCALABILITY STUDY")
    print("=" * 60)
    
    configs = [
        {"num_layers": 3, "num_epochs": 2},
        {"num_layers": 6, "num_epochs": 2}, 
        {"num_layers": 9, "num_epochs": 2},
        {"num_layers": 12, "num_epochs": 2},
    ]
    
    results = []
    for config in configs:
        print(f"\nTesting {config['num_layers']} layer probes...")
        result = run_benchmark(**config)
        results.append(result)
        print(f"{result['speedup']:.1f}x speedup")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("-" * 60)
    print(f"{'Layers':<8} {'Speedup':<10}")
    print("-" * 30)
    
    for r in results:
        print(f"{r['num_layers']:<8} {r['speedup']:<10.1f}")
    
    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE!")