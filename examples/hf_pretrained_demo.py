"""Lifecycle demo: train, checkpoint, and materialize HF decoder LMs on SCAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, TrainerCallback
from transformers.training_args import TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelbatch.huggingface_integration import ModelBatchTrainer

PAD = "<pad>"
EOS = "<eos>"
UNK = "<unk>"
SEP = "<sep>"
PRIMITIVES = {
    "walk": "WALK",
    "look": "LOOK",
    "run": "RUN",
    "jump": "JUMP",
}
TURNS = {"left": "LTURN", "right": "RTURN"}


def create_adam_configs(
    learning_rates: list[float],
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Create Adam configs for this demo's per-model learning rates."""
    return [
        {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        for lr in learning_rates
    ]


def scan_atom(command: str) -> list[str]:
    parts = command.split()
    if parts[0] == "turn":
        action = TURNS[parts[-1]]
        if len(parts) == 2:
            return [action]
        if parts[1] == "opposite":
            return [action, action]
        if parts[1] == "around":
            return [action] * 4

    action = PRIMITIVES[parts[0]]
    if len(parts) == 1:
        return [action]

    turn = TURNS[parts[-1]]
    if parts[1] == "opposite":
        return [turn, turn, action]
    if parts[1] == "around":
        return [turn, action] * 4
    return [turn, action]


def interpret_scan(command: str) -> str:
    if " and " in command:
        left, right = command.split(" and ", 1)
        return " ".join([interpret_scan(left), interpret_scan(right)])
    if " after " in command:
        left, right = command.split(" after ", 1)
        return " ".join([interpret_scan(right), interpret_scan(left)])

    parts = command.split()
    if parts[-1] == "twice":
        return " ".join(scan_atom(" ".join(parts[:-1])) * 2)
    if parts[-1] == "thrice":
        return " ".join(scan_atom(" ".join(parts[:-1])) * 3)
    return " ".join(scan_atom(command))


def build_scan_pairs(seed: int) -> list[tuple[str, str]]:
    atoms = list(PRIMITIVES) + [f"turn {direction}" for direction in TURNS]
    directed = [f"{verb} {direction}" for verb in PRIMITIVES for direction in TURNS] + [
        f"{word} {modifier} {direction}"
        for word in [*PRIMITIVES, "turn"]
        for modifier in ["opposite", "around"]
        for direction in TURNS
    ]
    repeated = [
        f"{command} {count}"
        for command in [*atoms, *directed]
        for count in ["twice", "thrice"]
    ]
    clauses = [*atoms, *directed, *repeated]
    combined = [
        f"{left} {joiner} {right}"
        for left in clauses[:36]
        for right in clauses[:36]
        for joiner in ["and", "after"]
        if left != right
    ]
    commands = sorted({*clauses, *combined})
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(commands), generator=generator).tolist()
    commands = [commands[idx] for idx in order]
    return [(command, interpret_scan(command)) for command in commands]


class WordTokenizer:
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        vocab = {PAD: 0, EOS: 1, UNK: 2, SEP: 3}
        for command, target in pairs:
            for token in [*command.split(), *target.split()]:
                vocab.setdefault(token, len(vocab))
        self.token_to_id = vocab
        self.id_to_token = {idx: token for token, idx in vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(token, self.token_to_id[UNK]) for token in tokens]

    def prompt_ids(self, command: str) -> list[int]:
        return self.encode_tokens([*command.split(), SEP])

    def target_ids(self, target: str) -> list[int]:
        return self.encode_tokens([*target.split(), EOS])

    def decode_action_ids(self, ids: torch.Tensor) -> str:
        tokens = []
        for idx in ids.tolist():
            token = self.id_to_token.get(int(idx), UNK)
            if token == EOS:
                break
            if token not in {PAD, SEP}:
                tokens.append(token)
        return " ".join(tokens)


class ScanLanguageModelingDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: WordTokenizer,
        max_length: int = 96,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        command, target = self.pairs[idx]
        prompt = self.tokenizer.prompt_ids(command)
        completion = self.tokenizer.target_ids(target)
        ids = [*prompt, *completion][: self.max_length]
        labels = [*[-100] * len(prompt), *completion][: self.max_length]

        pad_count = self.max_length - len(ids)
        input_ids = torch.tensor([*ids, *[self.tokenizer.pad_id] * pad_count])
        attention_mask = torch.tensor([*([1] * len(ids)), *([0] * pad_count)])
        label_ids = torch.tensor([*labels, *[-100] * pad_count])
        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "labels": label_ids.long(),
        }


def build_models(num_models: int, vocab_size: int) -> list[GPT2LMHeadModel]:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
    )
    return [GPT2LMHeadModel(config) for _ in range(num_models)]


def exact_match_accuracy(
    model: GPT2LMHeadModel,
    dataset: ScanLanguageModelingDataset,
    limit: int,
    device: torch.device,
) -> float:
    model.eval()
    model.to(device)
    correct = 0
    total = min(limit, len(dataset))
    for idx in range(total):
        command, expected = dataset.pairs[idx]
        prompt = dataset.tokenizer.prompt_ids(command)
        input_ids = torch.tensor([prompt], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=dataset.max_length - len(prompt),
                pad_token_id=dataset.tokenizer.pad_id,
                eos_token_id=dataset.tokenizer.eos_id,
                do_sample=False,
            )
        prediction = dataset.tokenizer.decode_action_ids(
            generated[0, len(prompt) :].cpu()
        )
        correct += prediction == expected
    return correct / total


def batched_exact_match_accuracy(
    model_batch,
    dataset: ScanLanguageModelingDataset,
    limit: int,
    device: torch.device,
) -> dict[str, float]:
    model_batch.eval()
    model_batch.to(device)
    total = min(limit, len(dataset))
    correct = [0] * model_batch.num_models
    for idx in range(total):
        command, expected = dataset.pairs[idx]
        prompt = dataset.tokenizer.prompt_ids(command)
        input_ids = torch.tensor([prompt], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            generated = model_batch.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=dataset.max_length - len(prompt),
                pad_token_id=dataset.tokenizer.pad_id,
                eos_token_id=dataset.tokenizer.eos_id,
            )
        for model_idx in range(model_batch.num_models):
            prediction = dataset.tokenizer.decode_action_ids(
                generated[model_idx, 0, len(prompt) :].cpu()
            )
            correct[model_idx] += prediction == expected
    return {
        f"model_{model_idx}": score / total for model_idx, score in enumerate(correct)
    }


def generation_consistency(
    model_batch,
    dataset: ScanLanguageModelingDataset,
    limit: int,
    device: torch.device,
) -> dict[str, bool]:
    model_batch.eval()
    model_batch.to(device)
    total = min(limit, len(dataset))
    materialized = [
        model_batch.materialize_model(model_idx).to(device).eval()
        for model_idx in range(model_batch.num_models)
    ]
    matches = [True] * model_batch.num_models
    for idx in range(total):
        command, _expected = dataset.pairs[idx]
        prompt = dataset.tokenizer.prompt_ids(command)
        input_ids = torch.tensor([prompt], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            batched = model_batch.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=dataset.max_length - len(prompt),
                pad_token_id=dataset.tokenizer.pad_id,
                eos_token_id=dataset.tokenizer.eos_id,
            )
            for model_idx, model in enumerate(materialized):
                single = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=dataset.max_length - len(prompt),
                    pad_token_id=dataset.tokenizer.pad_id,
                    eos_token_id=dataset.tokenizer.eos_id,
                    do_sample=False,
                )
                batched_text = dataset.tokenizer.decode_action_ids(
                    batched[model_idx, 0, len(prompt) :].cpu()
                )
                single_text = dataset.tokenizer.decode_action_ids(
                    single[0, len(prompt) :].cpu()
                )
                matches[model_idx] &= batched_text == single_text
    return {f"model_{model_idx}": matched for model_idx, matched in enumerate(matches)}


class LifecycleCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: Path,
        train_eval_dataset: ScanLanguageModelingDataset,
        heldout_dataset: ScanLanguageModelingDataset,
        eval_limit: int,
    ) -> None:
        self.output_dir = output_dir
        self.train_eval_dataset = train_eval_dataset
        self.heldout_dataset = heldout_dataset
        self.eval_limit = eval_limit

    def on_evaluate(self, _args, state, control, model=None, **_kwargs):
        if model is None:
            return control
        checkpoint_dir = self.output_dir / f"modelbatch_step_{state.global_step}"
        model.save_pretrained(str(checkpoint_dir))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scores = {
            "train_batched": batched_exact_match_accuracy(
                model, self.train_eval_dataset, self.eval_limit, device
            ),
            "heldout_batched": batched_exact_match_accuracy(
                model, self.heldout_dataset, self.eval_limit, device
            ),
            "batched_matches_materialized": generation_consistency(
                model, self.train_eval_dataset, self.eval_limit, device
            ),
        }
        with (checkpoint_dir / "exact_match.json").open("w") as handle:
            json.dump(scores, handle, indent=2)
        print(f"step={state.global_step} exact_match={scores}")
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/modelbatch_scan_hf")
    )
    parser.add_argument("--train-size", type=int, default=768)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.environ.setdefault("WANDB_DISABLED", "true")

    pairs = build_scan_pairs(args.seed)
    train_pairs = pairs[: args.train_size]
    train_eval_pairs = train_pairs[: args.eval_size]
    heldout_pairs = pairs[args.train_size : args.train_size + args.eval_size]
    tokenizer = WordTokenizer(train_pairs)
    train_dataset = ScanLanguageModelingDataset(train_pairs, tokenizer)
    train_eval_dataset = ScanLanguageModelingDataset(train_eval_pairs, tokenizer)
    heldout_dataset = ScanLanguageModelingDataset(heldout_pairs, tokenizer)
    models = build_models(num_models=3, vocab_size=tokenizer.vocab_size)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = ModelBatchTrainer(
        models=models,
        optimizer_configs=create_adam_configs([3e-3, 1e-3, 3e-4]),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=heldout_dataset,
        callbacks=[
            LifecycleCallback(
                args.output_dir,
                train_eval_dataset,
                heldout_dataset,
                args.eval_limit,
            ),
        ],
    )

    trainer.train()
    trainer.evaluate()

    final_dir = args.output_dir / "final_modelbatch"
    trainer.model_batch.save_pretrained(str(final_dir))
    reloaded_batch = type(trainer.model_batch).from_pretrained(str(final_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_scores = {
        "train_batched": batched_exact_match_accuracy(
            trainer.model_batch,
            train_eval_dataset,
            args.eval_limit,
            device,
        ),
        "heldout_batched": batched_exact_match_accuracy(
            trainer.model_batch,
            heldout_dataset,
            args.eval_limit,
            device,
        ),
        "reloaded_heldout_batched": batched_exact_match_accuracy(
            reloaded_batch,
            heldout_dataset,
            args.eval_limit,
            device,
        ),
        "batched_matches_materialized": generation_consistency(
            trainer.model_batch,
            train_eval_dataset,
            args.eval_limit,
            device,
        ),
        "materialized_train": {},
        "materialized_heldout": {},
        "saved_reloaded_heldout": {},
    }
    for idx in range(trainer.model_batch.num_models):
        model = trainer.model_batch.materialize_model(idx)
        model_dir = args.output_dir / f"final_model_{idx}"
        model.save_pretrained(str(model_dir))
        reloaded_model = type(model).from_pretrained(model_dir)
        final_scores["materialized_train"][f"model_{idx}"] = exact_match_accuracy(
            model,
            train_eval_dataset,
            args.eval_limit,
            device,
        )
        final_scores["materialized_heldout"][f"model_{idx}"] = exact_match_accuracy(
            model,
            heldout_dataset,
            args.eval_limit,
            device,
        )
        final_scores["saved_reloaded_heldout"][f"model_{idx}"] = exact_match_accuracy(
            reloaded_model,
            heldout_dataset,
            args.eval_limit,
            device,
        )

    with (args.output_dir / "vocab.json").open("w") as handle:
        json.dump(tokenizer.token_to_id, handle, indent=2)
    with (args.output_dir / "final_exact_match.json").open("w") as handle:
        json.dump(final_scores, handle, indent=2)
    print(f"final exact_match={final_scores}")


if __name__ == "__main__":
    main()
