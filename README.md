# Jazzarranger

Jazzarranger is the workspace for melody-to-harmony labeling, root completion experiments, and the validator/review loop around them.

## Entry points

- Workflow: [docs/process/project_workflow.md](docs/process/project_workflow.md)
- Manifest guide: [docs/process/manifest.md](docs/process/manifest.md)
- Codex to Perplexity handoff: [docs/process/codex_to_perplexity.md](docs/process/codex_to_perplexity.md)
- Frozen contract: [manifests/root_simple_v1.json](manifests/root_simple_v1.json)

## What the manifest is

The manifest is the frozen experiment contract. It pins the dataset set, quality filter, task name, and validator-facing fields so training, evaluation, and review all read the same rules.

If the experiment contract changes, create a new manifest version instead of silently editing the old one.
