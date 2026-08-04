"""Utilities for adding stage-one harmony tokens to a HF tokenizer/model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.harmony_tokens import NEW_TOKENS


def harmony_added_tokens() -> list[object]:
    try:
        from tokenizers import AddedToken
    except ImportError:
        return list(NEW_TOKENS)
    return [
        AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=False)
        for token in NEW_TOKENS
    ]


def add_harmony_tokens(tokenizer) -> int:
    vocab = tokenizer.get_vocab()
    missing_strings = [token for token in NEW_TOKENS if token not in vocab]
    if not missing_strings:
        return 0
    try:
        from tokenizers import AddedToken
    except ImportError:
        missing = missing_strings
    else:
        missing = [
            AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=False)
            for token in missing_strings
        ]
    return tokenizer.add_tokens(missing, special_tokens=False)


def validate_harmony_tokenizer(tokenizer) -> None:
    bad: list[tuple[str, list[int]]] = []
    for token in NEW_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            bad.append((token, ids))
    if bad:
        preview = ", ".join(f"{token}:{ids}" for token, ids in bad[:10])
        raise ValueError(f"harmony tokens are not single tokens: {preview}")


def configure_tokenizer(tokenizer) -> int:
    added = add_harmony_tokens(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_harmony_tokenizer(tokenizer)
    return added


def resize_model_for_tokenizer(model, tokenizer) -> None:
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-resized-model", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    added = configure_tokenizer(tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_resized_model:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        resize_model_for_tokenizer(model, tokenizer)
        model.save_pretrained(args.output_dir)

    print(f"added_tokens={added} vocab_size={len(tokenizer)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
