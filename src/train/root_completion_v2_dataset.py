"""Task-bank datasets for stage 1A v2 root completion."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from src.data.common import ROOTS_SHARP, schema_key_parts, schema_key_to_abc_key, transpose_root
from src.data.harmony_tokens import ROOT_TOKENS, quality_to_token, root_to_token
from src.train.stage1_dataset import dataset_names_from_csv, split_files


ROOT_TOKEN_TO_PC = {token: idx for idx, token in enumerate(ROOT_TOKENS)}
PC_TO_ROOT_TOKEN = {idx: token for token, idx in ROOT_TOKEN_TO_PC.items()}

MASK_RECIPE_WEIGHTS = {
    "single_internal": 0.45,
    "cadential_center": 0.20,
    "adjacent_double": 0.20,
    "long_context_sparse": 0.15,
}

SIMPLE_ALLOWED_QUALITIES = {
    "maj",
    "min",
    "dom7",
    "maj7",
    "min7",
    "hdim7",
    "dim",
    "sus2",
    "sus4",
}

ABC_NOTE_RE = re.compile(r"(?P<acc>\^\^|__|\^|_|=)?(?P<letter>[A-Ga-g])(?P<oct>[,']*)(?P<dur>\d*)")
ABC_KEY_RE = re.compile(r"^K:(?P<key>\S+)(?P<rest>.*)$")


@dataclass(frozen=True)
class RootCompletionTask:
    row: dict
    primary_index: int
    mask_indices: tuple[int, ...]
    mask_type: str

    @property
    def primary_root(self) -> str:
        return self.row["chords"][self.primary_index]["root"]

    @property
    def primary_root_pc(self) -> int:
        return ROOTS_SHARP.index(self.primary_root)

    @property
    def primary_quality(self) -> str:
        return self.row["chords"][self.primary_index]["quality"]


def task_song_id(task: RootCompletionTask) -> str:
    metadata = task.row.get("metadata") or {}
    song_id = metadata.get("song_id") or task.row.get("id") or ""
    return f"{task.row.get('dataset') or ''}:{song_id}"


def task_key(task: RootCompletionTask) -> tuple[str, int, tuple[int, ...], str]:
    return (str(task.row.get("id") or ""), task.primary_index, task.mask_indices, task.mask_type)


@dataclass
class RootCompletionExample:
    input_ids: list[int]
    labels: list[int]
    weight: float
    row_id: str
    dataset: str
    mask_type: str
    primary_quality: str
    gold_roots: list[str]
    transpose_semitones: int


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _weighted_choice(rng: random.Random, weighted_items: dict[str, float]) -> str:
    total = sum(weighted_items.values())
    point = rng.random() * total
    cursor = 0.0
    for item, weight in weighted_items.items():
        cursor += weight
        if point <= cursor:
            return item
    return next(reversed(weighted_items))


def _root_pc(root: str) -> int:
    return ROOTS_SHARP.index(root)


def _find_251_like_indices(chords: list[dict]) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for idx in range(len(chords) - 2):
        a = _root_pc(chords[idx]["root"])
        b = _root_pc(chords[idx + 1]["root"])
        c = _root_pc(chords[idx + 2]["root"])
        if (b - a) % 12 == 5 and (c - b) % 12 == 5:
            triples.append((idx, idx + 1, idx + 2))
    return triples


def build_root_completion_task_bank(
    *,
    data_dir: Path,
    split: str,
    include_zero_weight: bool = False,
    dataset_names: set[str] | None = None,
    allowed_qualities: set[str] | None = None,
    max_chords: int | None = None,
) -> tuple[list[RootCompletionTask], Counter, list[Path]]:
    tasks: list[RootCompletionTask] = []
    skipped: Counter = Counter()
    source_files = split_files(data_dir, split)
    for path in source_files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                dataset = str(row.get("dataset") or "")
                if dataset_names and dataset not in dataset_names:
                    skipped["dataset_filter"] += 1
                    continue
                if float(row.get("dataset_weight", 1.0)) <= 0.0 and not include_zero_weight:
                    skipped["zero_weight"] += 1
                    continue
                chords = list(row.get("chords") or [])
                if not chords:
                    skipped["missing_chords"] += 1
                    continue
                if allowed_qualities is not None:
                    chords = [chord for chord in chords if str(chord.get("quality") or "") in allowed_qualities]
                    if not chords:
                        skipped["quality_filter"] += 1
                        continue
                    row = dict(row)
                    row["chords"] = chords
                if max_chords and len(chords) > max_chords:
                    skipped["too_many_chords"] += 1
                    continue
                n = len(chords)

                internal_indices = range(1, n - 1) if n >= 3 else range(n)
                for idx in internal_indices:
                    tasks.append(RootCompletionTask(row, idx, (idx,), "single_internal"))

                for left, middle, right in _find_251_like_indices(chords):
                    tasks.append(RootCompletionTask(row, middle, (middle,), "cadential_center"))
                    tasks.append(RootCompletionTask(row, right, (right,), "cadential_center"))

                for idx in range(n - 1):
                    tasks.append(RootCompletionTask(row, idx, (idx, idx + 1), "adjacent_double"))
                    tasks.append(RootCompletionTask(row, idx + 1, (idx, idx + 1), "adjacent_double"))

                if n >= 3:
                    for idx in internal_indices:
                        rng = random.Random(_stable_seed(f"{row.get('id', '')}:{idx}:sparse", 17))
                        candidates = [other for other in range(n) if other != idx]
                        extra_count = min(len(candidates), rng.randint(1, 3))
                        extra = rng.sample(candidates, extra_count) if extra_count else []
                        mask_indices = tuple(sorted([idx, *extra]))
                        tasks.append(RootCompletionTask(row, idx, mask_indices, "long_context_sparse"))
    return tasks, skipped, source_files


def _abc_key_to_schema(key_text: str) -> str:
    raw = key_text.strip()
    if raw.endswith("m"):
        return f"{raw[:-1]}:min"
    return f"{raw}:maj"


def transpose_abc_melody(abc: str, semitones: int) -> str:
    if semitones % 12 == 0:
        return abc
    out_lines = []
    in_body = False
    for line in abc.splitlines():
        key_match = ABC_KEY_RE.match(line)
        if key_match:
            schema_key = _abc_key_to_schema(key_match.group("key"))
            key_root, key_mode = schema_key_parts(schema_key)
            shifted_key = f"{transpose_root(key_root, semitones)}:{key_mode}"
            out_lines.append(f"K:{schema_key_to_abc_key(shifted_key)}{key_match.group('rest')}")
            in_body = True
            continue
        if in_body:
            out_lines.append(transpose_abc_line(line, semitones))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def transpose_abc_line(line: str, semitones: int) -> str:
    parts = []
    cursor = 0
    for match in ABC_NOTE_RE.finditer(line):
        start, end = match.span()
        if start > cursor:
            parts.append(line[cursor:start])
        token = match.group(0)
        # Avoid touching note letters that are part of control fields or words.
        before = line[start - 1] if start > 0 else " "
        after = line[end] if end < len(line) else " "
        if before.isalpha() or after.isalpha():
            parts.append(token)
        else:
            parts.append(transpose_abc_note_token(match, semitones))
        cursor = end
    parts.append(line[cursor:])
    return "".join(parts)


def transpose_abc_note_token(match: re.Match, semitones: int) -> str:
    acc = match.group("acc") or ""
    letter = match.group("letter")
    octaves = match.group("oct") or ""
    duration = match.group("dur") or ""
    base_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter.upper()]
    accidental_delta = {"^^": 2, "__": -2, "^": 1, "_": -1, "=": 0, "": 0}[acc]
    octave_offset = 1 if letter.islower() else 0
    octave_offset += octaves.count("'")
    octave_offset -= octaves.count(",")
    pitch = 60 + base_pc + accidental_delta + 12 * octave_offset

    from src.data.common import midi_pitch_to_abc

    return midi_pitch_to_abc(pitch + semitones) + duration


def transpose_root_token(token: str, semitones: int) -> str:
    pc = ROOT_TOKEN_TO_PC[token]
    return PC_TO_ROOT_TOKEN[(pc + semitones) % 12]


def shifted_root_to_token(root: str, semitones: int) -> str:
    return root_to_token(transpose_root(root, semitones))


def _choose_diverse_task(
    rng: random.Random,
    candidates: list[RootCompletionTask],
    *,
    desired_pc: int,
    usage: dict[str, Counter],
    candidate_pool_size: int,
    bad_task_keys: set[tuple[str, int, tuple[int, ...], str]] | None = None,
) -> RootCompletionTask | None:
    if len(candidates) <= 1:
        candidate = candidates[0]
        if bad_task_keys is not None and task_key(candidate) in bad_task_keys:
            return None
        return candidate
    pool_size = min(len(candidates), max(1, candidate_pool_size))
    pool: list[RootCompletionTask] = []
    attempts = 0
    max_attempts = max(pool_size * 8, 32)
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        candidate = candidates[rng.randrange(len(candidates))]
        if bad_task_keys is not None and task_key(candidate) in bad_task_keys:
            continue
        pool.append(candidate)
    if not pool:
        return None

    def score(task: RootCompletionTask) -> tuple[float, float]:
        row_id = str(task.row.get("id") or "")
        song_id = task_song_id(task)
        dataset = str(task.row.get("dataset") or "")
        quality = task.primary_quality
        value = (
            4.0 * usage["row"][row_id]
            + 2.0 * usage["song"][song_id]
            + 1.5 * usage["root_quality"][(desired_pc, quality)]
            + 0.5 * usage["dataset"][dataset]
        )
        return value, rng.random()

    return min(pool, key=score)


def _context_line(
    chord: dict,
    *,
    index: int,
    primary_index: int,
    mask_indices: set[int],
    semitones: int,
) -> str:
    start = int(chord["start"])
    end = int(chord["end"])
    quality_token = quality_to_token(chord["quality"])
    if index == primary_index:
        return f"<SPAN> @{start}-{end} <MASK_ROOT> {quality_token} </SPAN>"
    if index in mask_indices:
        return f"<SPAN> @{start}-{end} <MASK_CHORD> {quality_token} </SPAN>"
    return f"<SPAN> @{start}-{end} {shifted_root_to_token(chord['root'], semitones)} {quality_token} </SPAN>"


def build_root_completion_example(
    task: RootCompletionTask,
    *,
    tokenizer,
    max_length: int,
    semitones: int = 0,
) -> RootCompletionExample | None:
    row = task.row
    chords = list(row["chords"])
    mask_indices = set(task.mask_indices)
    abc = transpose_abc_melody(str(row["abc_melody"]).strip(), semitones)
    context_lines = [
        _context_line(
            chord,
            index=idx,
            primary_index=task.primary_index,
            mask_indices=mask_indices,
            semitones=semitones,
        )
        for idx, chord in enumerate(chords)
    ]
    prompt = "\n".join(
        [
            "<MELODY>",
            abc,
            "</MELODY>",
            "",
            "<HARMONY_CONTEXT>",
            *context_lines,
            "</HARMONY_CONTEXT>",
            "",
            "<ROOT_PREDICTION>",
            "",
        ]
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = list(prompt_ids)
    labels = [-100] * len(input_ids)
    token = shifted_root_to_token(chords[task.primary_index]["root"], semitones)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"root token is not single-token: {token} -> {token_ids}")
    input_ids.extend(token_ids)
    labels.extend(token_ids)
    gold_roots = [token]
    suffix_ids = tokenizer.encode("\n</ROOT_PREDICTION>", add_special_tokens=False)
    input_ids.extend(suffix_ids)
    labels.extend([-100] * len(suffix_ids))
    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(-100)
    if len(input_ids) > max_length:
        return None
    return RootCompletionExample(
        input_ids=input_ids,
        labels=labels,
        weight=float(row.get("dataset_weight", 1.0)),
        row_id=str(row.get("id") or ""),
        dataset=str(row.get("dataset") or ""),
        mask_type=task.mask_type,
        primary_quality=task.primary_quality,
        gold_roots=gold_roots,
        transpose_semitones=semitones,
    )


class Stage1RootCompletionV2Dataset(Dataset):
    def __init__(
        self,
        *,
        data_dir: Path,
        split: str,
        tokenizer,
        max_length: int,
        seed: int = 13,
        include_zero_weight: bool = False,
        dataset_names: set[str] | None = None,
        allowed_qualities: set[str] | None = None,
        mode: str = "balanced_train",
        examples_per_root_per_update: int = 1,
        updates_per_epoch: int = 400,
        balanced_examples_per_root: int = 24,
        natural_limit: int | None = None,
        natural_shuffle: bool = True,
        transpose_to_sampled_root: bool = False,
        mask_recipe_weights: dict[str, float] | None = None,
        max_chords: int | None = None,
        diversity_candidate_pool: int = 64,
    ) -> None:
        if mode not in {"balanced_train", "balanced_eval", "natural_eval"}:
            raise ValueError(f"unsupported root-completion v2 mode: {mode}")
        self.mode = mode
        self.examples: list[RootCompletionExample] = []
        self.skipped: Counter = Counter()
        self.tasks, task_skipped, self.source_files = build_root_completion_task_bank(
            data_dir=data_dir,
            split=split,
            include_zero_weight=include_zero_weight,
            dataset_names=dataset_names,
            allowed_qualities=allowed_qualities,
            max_chords=max_chords,
        )
        self.skipped.update(task_skipped)
        self.task_summary = summarize_task_bank(self.tasks)
        weights = mask_recipe_weights or MASK_RECIPE_WEIGHTS
        rng = random.Random(seed)
        if mode == "balanced_train":
            self._populate_balanced_train_examples(
                rng=rng,
                tokenizer=tokenizer,
                max_length=max_length,
                updates=updates_per_epoch,
                examples_per_root=examples_per_root_per_update,
                weights=weights,
                transpose_to_sampled_root=transpose_to_sampled_root,
                diversity_candidate_pool=diversity_candidate_pool,
            )
            self.example_summary = summarize_examples(self.examples)
            return
        elif mode == "balanced_eval":
            self._populate_balanced_eval_examples(
                rng=rng,
                tokenizer=tokenizer,
                max_length=max_length,
                examples_per_root=balanced_examples_per_root,
                weights=weights,
                diversity_candidate_pool=diversity_candidate_pool,
            )
            self.example_summary = summarize_examples(self.examples)
            return
        else:
            tasks = self._natural_eval_tasks(rng=rng, limit=natural_limit, shuffle=natural_shuffle)

        for task, semitones in tasks:
            example = build_root_completion_example(
                task,
                tokenizer=tokenizer,
                max_length=max_length,
                semitones=semitones,
            )
            if example is None:
                self.skipped["over_max_length"] += 1
                continue
            self.examples.append(example)
        self.example_summary = summarize_examples(self.examples)

    def _tasks_by_root_and_recipe(self) -> dict[int, dict[str, list[RootCompletionTask]]]:
        grouped: dict[int, dict[str, list[RootCompletionTask]]] = {
            pc: {recipe: [] for recipe in MASK_RECIPE_WEIGHTS} for pc in range(12)
        }
        for task in self.tasks:
            grouped[task.primary_root_pc].setdefault(task.mask_type, []).append(task)
        return grouped

    def _tasks_by_recipe(self) -> dict[str, list[RootCompletionTask]]:
        grouped: dict[str, list[RootCompletionTask]] = {recipe: [] for recipe in MASK_RECIPE_WEIGHTS}
        for task in self.tasks:
            grouped.setdefault(task.mask_type, []).append(task)
        return grouped

    def _balanced_training_tasks(
        self,
        *,
        rng: random.Random,
        updates: int,
        examples_per_root: int,
        weights: dict[str, float],
        transpose_to_sampled_root: bool,
    ) -> list[tuple[RootCompletionTask, int]]:
        selected: list[tuple[RootCompletionTask, int]] = []
        by_root_recipe = self._tasks_by_root_and_recipe()
        by_recipe = self._tasks_by_recipe()
        for _ in range(updates):
            cycle: list[tuple[RootCompletionTask, int]] = []
            for desired_pc in range(12):
                for _ in range(examples_per_root):
                    recipe = _weighted_choice(rng, weights)
                    if transpose_to_sampled_root:
                        candidates = by_recipe.get(recipe) or self.tasks
                    else:
                        candidates = by_root_recipe[desired_pc].get(recipe) or [
                            task for tasks in by_root_recipe[desired_pc].values() for task in tasks
                        ]
                    if not candidates:
                        self.skipped[f"missing_root_{desired_pc}"] += 1
                        continue
                    task = rng.choice(candidates)
                    semitones = (desired_pc - task.primary_root_pc) % 12 if transpose_to_sampled_root else 0
                    cycle.append((task, semitones))
            # Keep the update cycle root-balanced, but vary order inside it.
            rng.shuffle(cycle)
            selected.extend(cycle)
        return selected

    def _balanced_eval_tasks(
        self,
        *,
        rng: random.Random,
        examples_per_root: int,
        weights: dict[str, float],
    ) -> list[tuple[RootCompletionTask, int]]:
        by_root_recipe = self._tasks_by_root_and_recipe()
        selected: list[tuple[RootCompletionTask, int]] = []
        for pc in range(12):
            for _ in range(examples_per_root):
                recipe = _weighted_choice(rng, weights)
                candidates = by_root_recipe[pc].get(recipe) or [
                    task for tasks in by_root_recipe[pc].values() for task in tasks
                ]
                if candidates:
                    selected.append((rng.choice(candidates), 0))
        rng.shuffle(selected)
        return selected

    def _natural_eval_tasks(
        self,
        *,
        rng: random.Random,
        limit: int | None,
        shuffle: bool,
    ) -> list[tuple[RootCompletionTask, int]]:
        tasks = [task for task in self.tasks if task.mask_type == "single_internal"]
        if shuffle:
            rng.shuffle(tasks)
        if limit is not None:
            tasks = tasks[:limit]
        return [(task, 0) for task in tasks]

    def _try_add_example(
        self,
        task: RootCompletionTask,
        *,
        tokenizer,
        max_length: int,
        semitones: int,
    ) -> bool:
        example = build_root_completion_example(
            task,
            tokenizer=tokenizer,
            max_length=max_length,
            semitones=semitones,
        )
        if example is None:
            self.skipped["over_max_length"] += 1
            return False
        self.examples.append(example)
        return True

    def _populate_balanced_train_examples(
        self,
        *,
        rng: random.Random,
        tokenizer,
        max_length: int,
        updates: int,
        examples_per_root: int,
        weights: dict[str, float],
        transpose_to_sampled_root: bool,
        diversity_candidate_pool: int,
    ) -> None:
        by_root_recipe = self._tasks_by_root_and_recipe()
        by_recipe = self._tasks_by_recipe()
        usage: dict[str, Counter] = {
            "row": Counter(),
            "song": Counter(),
            "dataset": Counter(),
            "root_quality": Counter(),
        }
        bad_task_keys: set[tuple[str, int, tuple[int, ...], str]] = set()
        for _ in range(updates):
            cycle_examples: list[RootCompletionExample] = []
            for desired_pc in range(12):
                for _ in range(examples_per_root):
                    added = False
                    for _attempt in range(64):
                        recipe = _weighted_choice(rng, weights)
                        if transpose_to_sampled_root:
                            candidates = by_recipe.get(recipe) or self.tasks
                        else:
                            candidates = by_root_recipe[desired_pc].get(recipe) or [
                                task for tasks in by_root_recipe[desired_pc].values() for task in tasks
                            ]
                        if not candidates:
                            continue
                        task = _choose_diverse_task(
                            rng,
                            candidates,
                            desired_pc=desired_pc,
                            usage=usage,
                            candidate_pool_size=diversity_candidate_pool,
                            bad_task_keys=bad_task_keys,
                        )
                        if task is None:
                            continue
                        semitones = (desired_pc - task.primary_root_pc) % 12 if transpose_to_sampled_root else 0
                        example = build_root_completion_example(
                            task,
                            tokenizer=tokenizer,
                            max_length=max_length,
                            semitones=semitones,
                        )
                        if example is not None:
                            cycle_examples.append(example)
                            row_id = str(task.row.get("id") or "")
                            usage["row"][row_id] += 1
                            usage["song"][task_song_id(task)] += 1
                            usage["dataset"][str(task.row.get("dataset") or "")] += 1
                            usage["root_quality"][(desired_pc, task.primary_quality)] += 1
                            added = True
                            break
                        bad_task_keys.add(task_key(task))
                        self.skipped["over_max_length"] += 1
                    if not added:
                        self.skipped[f"unfilled_root_{desired_pc}"] += 1
            rng.shuffle(cycle_examples)
            self.examples.extend(cycle_examples)

    def _populate_balanced_eval_examples(
        self,
        *,
        rng: random.Random,
        tokenizer,
        max_length: int,
        examples_per_root: int,
        weights: dict[str, float],
        diversity_candidate_pool: int,
    ) -> None:
        by_root_recipe = self._tasks_by_root_and_recipe()
        usage: dict[str, Counter] = {
            "row": Counter(),
            "song": Counter(),
            "dataset": Counter(),
            "root_quality": Counter(),
        }
        bad_task_keys: set[tuple[str, int, tuple[int, ...], str]] = set()
        for pc in range(12):
            for _ in range(examples_per_root):
                added = False
                for _attempt in range(64):
                    recipe = _weighted_choice(rng, weights)
                    candidates = by_root_recipe[pc].get(recipe) or [
                        task for tasks in by_root_recipe[pc].values() for task in tasks
                    ]
                    if not candidates:
                        continue
                    task = _choose_diverse_task(
                        rng,
                        candidates,
                        desired_pc=pc,
                        usage=usage,
                        candidate_pool_size=diversity_candidate_pool,
                        bad_task_keys=bad_task_keys,
                    )
                    if task is None:
                        continue
                    if self._try_add_example(task, tokenizer=tokenizer, max_length=max_length, semitones=0):
                        row_id = str(task.row.get("id") or "")
                        usage["row"][row_id] += 1
                        usage["song"][task_song_id(task)] += 1
                        usage["dataset"][str(task.row.get("dataset") or "")] += 1
                        usage["root_quality"][(pc, task.primary_quality)] += 1
                        added = True
                        break
                    bad_task_keys.add(task_key(task))
                if not added:
                    self.skipped[f"unfilled_root_{pc}"] += 1
        rng.shuffle(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "labels": example.labels,
            "example_weights": example.weight,
        }


def summarize_task_bank(tasks: list[RootCompletionTask]) -> dict:
    by_root = Counter(PC_TO_ROOT_TOKEN[task.primary_root_pc] for task in tasks)
    by_recipe = Counter(task.mask_type for task in tasks)
    by_quality = Counter(task.primary_quality for task in tasks)
    by_dataset = Counter(str(task.row.get("dataset") or "") for task in tasks)
    by_dataset_song = Counter(str(task.row.get("dataset") or "") for task in _unique_song_tasks(tasks))
    return {
        "tasks": len(tasks),
        "unique_songs": len({task_song_id(task) for task in tasks}),
        "by_primary_root": dict(sorted(by_root.items())),
        "by_mask_type": dict(sorted(by_recipe.items())),
        "by_primary_quality": dict(sorted(by_quality.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "unique_songs_by_dataset": dict(sorted(by_dataset_song.items())),
    }


def _unique_song_tasks(tasks: list[RootCompletionTask]) -> list[RootCompletionTask]:
    seen: set[str] = set()
    unique: list[RootCompletionTask] = []
    for task in tasks:
        song_id = task_song_id(task)
        if song_id in seen:
            continue
        seen.add(song_id)
        unique.append(task)
    return unique


def summarize_examples(examples: list[RootCompletionExample]) -> dict:
    by_root = Counter(root for example in examples for root in example.gold_roots)
    by_recipe = Counter(example.mask_type for example in examples)
    by_quality = Counter(example.primary_quality for example in examples)
    by_dataset = Counter(example.dataset for example in examples)
    by_shift = Counter(str(example.transpose_semitones) for example in examples)
    by_row = Counter(example.row_id for example in examples)
    return {
        "examples": len(examples),
        "answer_roots": sum(len(example.gold_roots) for example in examples),
        "unique_rows": len(by_row),
        "max_examples_per_row": max(by_row.values(), default=0),
        "by_gold_root": dict(sorted(by_root.items())),
        "by_mask_type": dict(sorted(by_recipe.items())),
        "by_primary_quality": dict(sorted(by_quality.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_transpose": dict(sorted(by_shift.items(), key=lambda item: int(item[0]))),
        "avg_answer_roots_per_example": (
            sum(len(example.gold_roots) for example in examples) / len(examples) if examples else 0.0
        ),
        "max_length": max((len(example.input_ids) for example in examples), default=0),
        "mean_length": (
            math.fsum(len(example.input_ids) for example in examples) / len(examples) if examples else 0.0
        ),
    }


def describe_root_completion_v2_dataset(dataset: Stage1RootCompletionV2Dataset) -> dict:
    return {
        "mode": dataset.mode,
        "source_files": [str(path) for path in dataset.source_files],
        "task_bank": dataset.task_summary,
        "examples": dataset.example_summary,
        "skipped": dict(sorted(dataset.skipped.items())),
    }


__all__ = [
    "MASK_RECIPE_WEIGHTS",
    "PC_TO_ROOT_TOKEN",
    "ROOT_TOKEN_TO_PC",
    "RootCompletionExample",
    "RootCompletionTask",
    "Stage1RootCompletionV2Dataset",
    "dataset_names_from_csv",
    "describe_root_completion_v2_dataset",
]
