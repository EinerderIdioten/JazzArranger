# Codex To Perplexity Handoff

This document defines the handoff loop from Codex to a Perplexity project.

## Purpose

When Codex has observed experiment results, bottlenecks, or ambiguous next steps, it should write a concise markdown handoff instead of carrying the whole discussion forward in chat.

Perplexity then uses that handoff as the research prompt: search, compare evidence, and return a plan.

## When to use it

- A training run exposed a failure mode.
- A validator change needs external confirmation or a better threshold.
- The current manifest or task contract may need a new version.
- You need a search-backed next-step plan before changing code.

## Handoff contents

Keep the handoff factual and short. Include:

- Current goal.
- What changed.
- Observed results.
- What is failing or unclear.
- Constraints, especially the active manifest.
- Questions that need search.
- The format you want back.

## Output from Perplexity

Ask for:

- A search-backed reading of the problem.
- A prioritized plan.
- Risks and failure modes.
- Any papers, tools, or prior art worth testing next.

## File flow

1. Codex writes `docs/research/YYYY-MM-DD_topic_handoff.md`.
2. Perplexity reads that file in its project.
3. Perplexity returns `docs/train/YYYY-MM-DD_topic_plan.md`.
4. Codex implements the plan and writes the run summary.
5. Review lives in `docs/reviews/`.

## Minimal template

```md
# Handoff: <topic>

## Goal

## Current state

## Observed results

## Difficulties

## Constraints

## Search questions

## Desired plan output
```
