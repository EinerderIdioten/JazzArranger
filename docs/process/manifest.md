# Experiment Manifest

The manifest is a frozen JSON contract for one experiment line.

## What it controls

- Which datasets are in scope.
- Which chord qualities are allowed.
- Which task the run is supposed to solve.
- Which validator fields should be treated as reference or candidate.
- Which external assets and summaries are pinned by hash.

## How it is used

Training, evaluation, and validator scripts read the manifest through `src/experiment_manifest.py`.
If no explicit path is passed, the code falls back to `manifests/root_simple_v1.json` when that file exists.

## What to change

- Edit the manifest only when you are defining a new frozen contract.
- If the contract changes, bump the manifest version and write a new file.
- Do not reuse an old frozen manifest for a different experiment.

## Current manifest

- [manifests/root_simple_v1.json](../../manifests/root_simple_v1.json)
