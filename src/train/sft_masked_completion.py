"""Stage-one masked harmony completion SFT."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.harmony_tokens import NEW_TOKENS
from src.train.masked_completion_dataset import (
    MaskedCompletionCollator,
    Stage1MaskedCompletionDataset,
    dataset_names_from_csv,
    describe_masked_dataset,
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
        "max_steps": args.max_steps,
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


def harmony_token_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    for token in NEW_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"not a single token after tokenizer setup: {token} -> {encoded}")
        ids.append(encoded[0])
    return sorted(set(ids))


def _zero_old_token_rows_hook(trainable_token_ids: list[int]):
    trainable: torch.Tensor | None = None

    def hook(grad: torch.Tensor) -> torch.Tensor:
        nonlocal trainable
        if trainable is None or trainable.device != grad.device:
            trainable = torch.tensor(trainable_token_ids, dtype=torch.long, device=grad.device)
        masked = torch.zeros_like(grad)
        valid = trainable[trainable < grad.shape[0]]
        masked.index_copy_(0, valid, grad.index_select(0, valid))
        return masked

    return hook


def transformer_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise ValueError("could not find transformer layers on model")


def set_requires_grad(module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def configure_trainable_parameters(
    model,
    *,
    tokenizer,
    train_last_ratio: float,
    train_old_token_rows: bool,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad = False

    layers = transformer_layers(model)
    layer_count = len(layers)
    train_count = max(1, int(round(layer_count * train_last_ratio)))
    start = max(0, layer_count - train_count)
    for layer in layers[start:]:
        set_requires_grad(layer, True)

    for attr_path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        obj = model
        found = True
        for part in attr_path.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found:
            set_requires_grad(obj, True)

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_embeddings.weight.requires_grad = True
    if output_embeddings is not None:
        output_embeddings.weight.requires_grad = True

    trainable_token_ids = harmony_token_ids(tokenizer)
    if not train_old_token_rows:
        input_embeddings.weight.register_hook(_zero_old_token_rows_hook(trainable_token_ids))
        if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
            output_embeddings.weight.register_hook(_zero_old_token_rows_hook(trainable_token_ids))

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    return {
        "layer_count": layer_count,
        "trainable_layer_start": start,
        "trainable_layer_count": train_count,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_param_ratio": trainable_params / max(1, total_params),
        "train_old_token_rows": train_old_token_rows,
        "trainable_token_rows": len(trainable_token_ids),
    }


def make_trainer_class(args):
    from transformers import Trainer

    class _MaskedCompletionTrainer(Trainer):
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

    return _MaskedCompletionTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["root", "quality", "chord"], default="root")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1a-root-masked"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--examples-per-row", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
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
    parser.add_argument("--train-last-ratio", type=float, default=0.30)
    parser.add_argument("--train-old-token-rows", action="store_true")
    parser.add_argument("--skip-save-final", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("use either bf16 or fp16, not both")

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

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

    dataset_filter = dataset_names_from_csv(args.datasets)
    train_dataset = Stage1MaskedCompletionDataset(
        data_dir=args.data_dir,
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        stage=args.stage,
        seed=args.seed,
        examples_per_row=args.examples_per_row,
        include_zero_weight=args.include_zero_weight,
        dataset_names=dataset_filter,
    )
    eval_dataset = Stage1MaskedCompletionDataset(
        data_dir=args.data_dir,
        split="val",
        tokenizer=tokenizer,
        max_length=args.max_length,
        stage=args.stage,
        seed=args.seed + 1000,
        examples_per_row=1,
        include_zero_weight=args.include_zero_weight,
        dataset_names=dataset_filter,
    )
    if not train_dataset:
        raise ValueError("no training examples after filtering")
    if not eval_dataset:
        eval_dataset = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "stage": args.stage,
        "model_name_or_path": args.model_name_or_path,
        "added_tokens": added,
        "vocab_size": len(tokenizer),
        "max_length": args.max_length,
        "examples_per_row": args.examples_per_row,
        "trainable": trainable_summary,
        "train": describe_masked_dataset(train_dataset),
        "eval": describe_masked_dataset(eval_dataset) if eval_dataset else None,
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
