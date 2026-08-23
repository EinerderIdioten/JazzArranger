"""Stage 1A v2 anti-collapse root-completion training."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.harmony_tokens import NEW_TOKENS
from src.experiment_manifest import (
    experiment_manifest_summary,
    load_experiment_manifest,
    manifest_allowed_qualities,
    manifest_dataset_names,
    resolve_experiment_manifest_path,
)
from src.train.masked_completion_dataset import MaskedCompletionCollator
from src.train.root_completion_v2_dataset import (
    MASK_RECIPE_WEIGHTS,
    Stage1RootCompletionV2Dataset,
    dataset_names_from_csv,
    describe_root_completion_v2_dataset,
    SIMPLE_ALLOWED_QUALITIES,
)
from src.train.sft_masked_completion import configure_trainable_parameters
from src.train.tokenizer_setup import configure_tokenizer, resize_model_for_tokenizer


def training_args_kwargs(args, *, has_eval: bool) -> dict:
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": args.optim,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    if "warmup_ratio" in signature.parameters:
        kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in signature.parameters:
        kwargs["warmup_steps"] = int(round(max(args.max_steps, 0) * args.warmup_ratio)) if args.max_steps > 0 else 0
    eval_strategy = "steps" if has_eval else "no"
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    if "save_strategy" in signature.parameters:
        kwargs["save_strategy"] = "steps"
    return kwargs


def make_trainer_class(args):
    from transformers import Trainer

    class _RootCompletionV2Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("example_weights").to(model.device)
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            token_mask = shift_labels.ne(-100)
            # Labels are only root tokens. This averages over masked positions
            # first, then averages examples, preventing multi-mask tasks from
            # receiving larger gradient share.
            per_example_loss = (token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
            loss = (per_example_loss * weights).sum() / weights.sum().clamp_min(1e-8)
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            decay = []
            no_decay = []
            token_params = []
            input_weight = self.model.get_input_embeddings().weight
            output_embeddings = self.model.get_output_embeddings()
            output_weight = output_embeddings.weight if output_embeddings is not None else None
            token_param_ids = {id(input_weight)}
            if output_weight is not None:
                token_param_ids.add(id(output_weight))
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if id(parameter) in token_param_ids:
                    token_params.append(parameter)
                elif parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
                    no_decay.append(parameter)
                else:
                    decay.append(parameter)
            groups = [
                {"params": decay, "weight_decay": args.weight_decay, "lr": args.learning_rate},
                {"params": no_decay, "weight_decay": 0.0, "lr": args.learning_rate},
                {"params": token_params, "weight_decay": 0.0, "lr": args.token_learning_rate},
            ]
            groups = [group for group in groups if group["params"]]
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
            return self.optimizer

    return _RootCompletionV2Trainer


def harmony_token_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    for token in NEW_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"not a single token after tokenizer setup: {token} -> {encoded}")
        ids.append(encoded[0])
    return sorted(set(ids))


def qualities_from_csv(text: str | None) -> set[str] | None:
    if not text:
        return None
    values = {item.strip() for item in text.split(",") if item.strip()}
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1a-v2-root"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--updates-per-epoch", type=int, default=400)
    parser.add_argument("--examples-per-root-per-update", type=int, default=1)
    parser.add_argument("--balanced-eval-examples-per-root", type=int, default=24)
    parser.add_argument("--natural-eval-limit", type=int, default=768)
    parser.add_argument("--max-chords", type=int)
    parser.add_argument("--diversity-candidate-pool", type=int, default=64)
    parser.add_argument("--transpose-to-sampled-root", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--token-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--include-zero-weight", action="store_true")
    parser.add_argument("--datasets", help="Comma-separated dataset names to include, e.g. EMOPIA+,HLSD,POP909")
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument(
        "--allowed-qualities",
        default=None,
        help="Comma-separated allowed quality families; defaults to the frozen manifest or the built-in simple set.",
    )
    parser.add_argument("--train-last-ratio", type=float, default=0.50)
    parser.add_argument("--train-old-token-rows", action="store_true")
    parser.add_argument("--skip-save-final", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("use either bf16 or fp16, not both")
    expected_accum = 12 * args.examples_per_root_per_update
    if args.gradient_accumulation_steps != expected_accum:
        print(
            json.dumps(
                {
                    "warning": "gradient_accumulation_steps should equal 12 * examples_per_root_per_update "
                    "to keep each optimizer update root-balanced",
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "expected": expected_accum,
                },
                indent=2,
            )
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    manifest_path = resolve_experiment_manifest_path(args.experiment_manifest)
    manifest = load_experiment_manifest(manifest_path)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    added = configure_tokenizer(tokenizer)
    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    resize_model_for_tokenizer(model, tokenizer)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    trainable_summary = configure_trainable_parameters(
        model,
        tokenizer=tokenizer,
        train_last_ratio=args.train_last_ratio,
        train_old_token_rows=args.train_old_token_rows,
    )

    dataset_filter = dataset_names_from_csv(args.datasets) or manifest_dataset_names(manifest)
    quality_filter = qualities_from_csv(args.allowed_qualities) or manifest_allowed_qualities(manifest)
    if quality_filter is None:
        quality_filter = set(SIMPLE_ALLOWED_QUALITIES)
    train_dataset = Stage1RootCompletionV2Dataset(
        data_dir=args.data_dir,
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        include_zero_weight=args.include_zero_weight,
        dataset_names=dataset_filter,
        allowed_qualities=quality_filter,
        mode="balanced_train",
        examples_per_root_per_update=args.examples_per_root_per_update,
        updates_per_epoch=args.updates_per_epoch,
        transpose_to_sampled_root=args.transpose_to_sampled_root,
        mask_recipe_weights=MASK_RECIPE_WEIGHTS,
        max_chords=args.max_chords,
        diversity_candidate_pool=args.diversity_candidate_pool,
    )
    eval_dataset = Stage1RootCompletionV2Dataset(
        data_dir=args.data_dir,
        split="val",
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed + 1000,
        include_zero_weight=args.include_zero_weight,
        dataset_names=dataset_filter,
        allowed_qualities=quality_filter,
        mode="balanced_eval",
        balanced_examples_per_root=args.balanced_eval_examples_per_root,
        mask_recipe_weights=MASK_RECIPE_WEIGHTS,
        max_chords=args.max_chords,
        diversity_candidate_pool=args.diversity_candidate_pool,
    )
    if not train_dataset:
        raise ValueError("no training examples after filtering")
    if not eval_dataset:
        eval_dataset = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "stage": "root_completion_v2",
        "model_name_or_path": args.model_name_or_path,
        "experiment_manifest": experiment_manifest_summary(manifest_path, manifest),
        "added_tokens": added,
        "vocab_size": len(tokenizer),
        "max_length": args.max_length,
        "max_chords": args.max_chords,
        "diversity_candidate_pool": args.diversity_candidate_pool,
        "updates_per_epoch": args.updates_per_epoch,
        "examples_per_root_per_update": args.examples_per_root_per_update,
        "transpose_to_sampled_root": args.transpose_to_sampled_root,
        "mask_recipe_weights": MASK_RECIPE_WEIGHTS,
        "effective_filters": {
            "datasets": sorted(dataset_filter) if dataset_filter else None,
            "allowed_qualities": sorted(quality_filter) if quality_filter else None,
        },
        "trainable": trainable_summary,
        "train": describe_root_completion_v2_dataset(train_dataset),
        "eval": describe_root_completion_v2_dataset(eval_dataset) if eval_dataset else None,
    }
    (args.output_dir / "run_data_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))

    trainer_cls = make_trainer_class(args)
    trainer = trainer_cls(
        model=model,
        args=TrainingArguments(**training_args_kwargs(args, has_eval=eval_dataset is not None)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MaskedCompletionCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if not args.skip_save_final:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
