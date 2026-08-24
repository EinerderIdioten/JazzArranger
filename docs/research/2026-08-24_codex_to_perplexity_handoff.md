# Handoff: Stage 1A root completion and validator loop

Date: 2026-08-24  
Source: Codex local observation  
Target: Perplexity project for search and planning

## Goal

Stabilize the root-completion experiment line and the validator/review loop so the next training iteration is based on evidence, not guesswork.

## Current state

- Frozen contract: [manifests/root_simple_v1.json](../../manifests/root_simple_v1.json)
- Workflow contract: [docs/process/project_workflow.md](../process/project_workflow.md)
- Current root-completion report: [docs/train/report_stage1a_v3_root_completion.md](../train/report_stage1a_v3_root_completion.md)
- Validator plan: [docs/train/2026-08-20_third_party_harmony_validator_plan.md](../train/2026-08-20_third_party_harmony_validator_plan.md)

Current local worktree already contains partial changes to:

- `src/train/sft_masked_completion.py`
- `src/validator/diagnostics.py`
- `src/validator/review.py`
- `src/validator/cli.py`

## Observed results

From the v3 report:

- Balanced root accuracy: 17.06%
- Natural root accuracy: 22.43%
- Balanced pred `<R_C>` share: 33.61%
- Natural pred `<R_C>` share: 44.75%
- Natural always-C baseline: 21.57%

Interpretation: the model is above random, but still strongly biased toward C-root shortcuts. Natural evaluation barely clears the always-C baseline.

The validator side now has stronger structural tags:

- `HARMONIC_RHYTHM_DISTORTION`
- `LOCAL_TONAL_DRIFT`
- `ROOT_MISMATCH`

The review renderer now groups windows by song title and limits how many windows appear per case.

## Difficulties

1. Root shortcut remains the main failure mode.
2. Visible harmony context still looks too easy for the model to exploit.
3. Review selection needs to avoid dumping too many windows from the same tune.
4. Validator thresholds are heuristic and need calibration against real cases.
5. The training script now has extra filtering switches, but we do not yet know whether they should become part of the default experiment contract.
6. We need a clean rule for when a manifest version should stay frozen versus when a new version should be minted.

## Search questions

Please search recent and relevant work on:

- Curriculum masking or context unmasking for causal LM masked completion.
- Techniques that reduce shortcut learning in sequence-to-sequence or infilling models.
- Evaluation-set construction for human review when multiple windows come from the same song.
- Practical calibration patterns for rule-based validators with severity tags.
- Any 2025-2026 work on melody-harmony completion, root recovery, or harmonic rhythm validation that would help choose the next experiment.

## Desired plan output

Return a plan with:

- Hypothesis.
- Proposed next experiment order.
- What to change in code, data, manifest, and review assets.
- Risks and failure modes.
- Acceptance criteria for the next run.
