"""Full-parameter SFT for ABC melody to tokenized harmony spans."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.train.stage1_dataset import (
    Stage1DataCollator,
    Stage1HarmonyDataset,
    dataset_names_from_csv,
    describe_dataset,
)
from src.train.tokenizer_setup import configure_tokenizer, resize_model_for_tokenizer


def training_args_kwargs(args, *, has_eval: bool) -> dict:
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
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
    signature = inspect.signature(TrainingArguments.__init__)
    eval_strategy = "steps" if has_eval else "no"
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    if "save_strategy" in signature.parameters:
        kwargs["save_strategy"] = "steps"
    return kwargs


def make_trainer_class():
    from transformers import Trainer

    class _WeightedCausalLMTrainer(Trainer):
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
            per_example_loss = (token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
            loss = (per_example_loss * weights).sum() / weights.sum().clamp_min(1e-8)
            return (loss, outputs) if return_outputs else loss

    return _WeightedCausalLMTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1-full-sft"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
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
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("use either bf16 or fp16, not both")

    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    dataset_filter = dataset_names_from_csv(args.datasets)
    train_dataset = Stage1HarmonyDataset(
        data_dir=args.data_dir,
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        include_zero_weight=args.include_zero_weight,
        dataset_names=dataset_filter,
    )
    eval_dataset = Stage1HarmonyDataset(
        data_dir=args.data_dir,
        split="val",
        tokenizer=tokenizer,
        max_length=args.max_length,
        include_zero_weight=args.include_zero_weight,
        dataset_names=dataset_filter,
    )
    if not train_dataset:
        raise ValueError("no training examples after filtering")
    if not eval_dataset:
        eval_dataset = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "model_name_or_path": args.model_name_or_path,
        "added_tokens": added,
        "max_length": args.max_length,
        "train": describe_dataset(train_dataset),
        "eval": describe_dataset(eval_dataset) if eval_dataset else None,
    }
    (args.output_dir / "run_data_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))

    trainer_cls = make_trainer_class()
    from transformers import TrainingArguments

    trainer = trainer_cls(
        model=model,
        args=TrainingArguments(**training_args_kwargs(args, has_eval=eval_dataset is not None)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Stage1DataCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
