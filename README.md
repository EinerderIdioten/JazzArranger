# AI Harmony Copilot

AI Harmony Copilot 是一个面向实际编曲场景的 AI 和声辅助系统。它的目标不是训练一个完全自主创作音乐的大模型，而是把 LLM、音乐知识库、规则引擎、候选搜索和音乐验证器组合起来，帮助音乐家完成 chord substitution、voicing 生成、候选排序和解释。

核心原则：

> 不让 LLM 直接生成最终音符；让 LLM 负责理解音乐语境、调用知识、比较候选方案和解释音乐选择。确定性的音高计算、规则约束、搜索和验证由 Harmony Engine 完成。

## Why This Architecture

直接让 LLM 输出和声或 voicing 容易出现这些问题：

- 八度错误
- 和弦音遗漏
- guide tone 缺失
- voice crossing
- avoid note 使用不当
- 不符合钢琴或目标乐器演奏范围
- 无法稳定处理大量组合搜索

和声编配本质上是一个多约束优化问题，需要同时考虑：

- 和弦功能
- 旋律音兼容性
- 风格特征
- 前后和声关系
- 声部进行
- 音域分布
- 演奏可行性
- tension 与 resolution
- 音乐家偏好

因此本项目采用混合架构：

```text
Knowledge Retrieval + Rule-based Generation + LLM Ranking + Music Validation
```

## System Architecture

```text
User Input
  |
  v
Music Analyzer
(Melody / Chord / Style / Context Analysis)
  |
  +----------------------------+
  |                            |
  v                            v
Harmony Knowledge Base         Harmony Engine
(RAG / Examples / Rules)       (Deterministic Search)
  |                            |
  +-------------+--------------+
                |
                v
Candidate Generator
(Substitutions + Voicings)
                |
                v
LLM Reranker
(Qwen / Llama / API Model)
                |
                v
Music Validator
(Harmony / Range / Voice Leading)
                |
                v
Harmony Recommendation
```

## Core Modules

### 1. Music Analyzer

Converts user input into structured musical context.

Example input sources:

- melody MIDI
- chord progression
- style preference
- previous measure context
- previous voicing

Example structured output:

```json
{
  "chord": "G7",
  "function": "dominant",
  "melody_notes": [
    {
      "pitch": "F4",
      "role": "minor_7"
    }
  ],
  "style": "jazz",
  "previous_voicing": ["F3", "C4", "E4", "A4"]
}
```

The analyzer does not compose. It only describes the current musical state.

### 2. Harmony Knowledge Base

The knowledge base should store musical decision cases, not only chord lookup tables.

A low-value record looks like this:

```text
Cmaj7 -> Em7
```

A high-value record looks like this:

```text
Context: Cmaj7
Melody: D
Style: Jazz Ballad
Substitution: Em7
Reason: Keeps the 9 color while creating smoother upward bass motion.
Suitable: Ending phrase
Avoid: Strong final cadence
```

The knowledge base should include:

- chord substitution knowledge
- rootless voicing examples
- Drop 2 / Drop 3 voicing examples
- jazz piano voicings
- pop, gospel, film scoring voicing patterns
- style-specific examples
- explanation-oriented musician decision cases
- user preference and feedback data

### 3. Harmony Engine

The Harmony Engine performs deterministic music calculation and constrained search.

Given:

```text
Chord: G7
Melody: F4
```

It can generate candidates such as:

```text
Candidate A: B3 E4 F4 A4
Candidate B: F3 B3 E4 A4
Candidate C: D3 F3 B3 E4
```

The engine is responsible for ensuring that candidates are musically plausible:

- notes belong to the chord or valid extensions
- required guide tones are included when needed
- avoid notes are controlled
- voicing register is reasonable
- instrument constraints are respected
- obvious conflicts are removed

The engine does not decide which candidate is the most musical. It creates a valid search space.

### 4. Candidate Generator

Combines retrieval results and rule-generated candidates.

It should produce multiple options for:

- chord substitutions
- reharmonization choices
- voicing choices
- style-specific variants
- different tension levels
- different register placements

The generator should keep candidates structured, auditable, and explainable.

### 5. LLM Reranker

The LLM acts as a musical judge, not as a raw note generator.

Example reranking prompt context:

```text
Previous voicing:
F3 C4 E4 A4

Current chord:
G7

Candidates:
A: B3 E4 F4 A4
B: F3 B3 E4 A4
C: D3 F3 B3 E4
```

Expected model behavior:

```text
Choice: B
Reason: It preserves the F as the minor seventh of G7, keeps guide-tone continuity, and produces smoother voice leading from the previous voicing.
```

The LLM is best used for:

- style judgment
- context interpretation
- musical preference comparison
- explaining decisions in musician language
- balancing competing constraints

It should not be trusted for:

- exact semitone calculation
- exhaustive note enumeration
- final music validation

### 6. Music Validator

The validator checks final candidates before returning them to the user.

Validation dimensions:

- harmonic correctness
- required chord tones
- guide tones
- avoid notes
- register limits
- hand span or instrument range
- voice crossing
- excessive leaps
- style consistency

Voice leading should normally be a scoring factor, not a strict binary rule. Some leaps are musically correct, such as a bass movement from C to F.

## MVP Scope

The first version should be a professional Harmony Assistant, not a fully autonomous arranger.

MVP functions:

1. Input melody plus chord progression.
2. Parse current chord, melody note, style and previous context.
3. Retrieve similar harmony cases from the knowledge base.
4. Recommend chord substitutions.
5. Generate 3 to 10 playable voicing candidates.
6. Use an LLM to rank candidates and explain the choice.
7. Validate final output with deterministic music rules.
8. Capture user feedback for future preference learning.

## Phase 1: Benchmark-first Implementation

Before building the full Copilot, Phase 1 should establish an evaluation framework.

Purpose:

> Measure how well raw LLMs understand jazz harmony before adding RAG, rule-based generation, or fine-tuning.

Phase 1 benchmark tasks:

- rootless Type A voicings
- rootless Type B voicings
- Drop 2 voicings
- guide-tone correctness
- avoid-note detection
- voice count and register checks
- basic voice-leading evaluation

Initial scoring dimensions:

| Category | Weight | Purpose |
| --- | ---: | --- |
| Voice count limit | 20% | Checks whether the model follows requested voicing size |
| Guide tones accuracy | 30% | Checks essential 3rd and 7th chord tones |
| Avoid-note violations | 30% | Penalizes conflicting notes |
| Voice-leading smoothness | 20% | Rewards compact and playable motion |

The benchmark gives a baseline score that future systems can compare against:

```text
Base LLM -> Base LLM + RAG -> Rule Engine + RAG + LLM Reranker -> Fine-tuned Ranker
```

## Planned Repository Structure

```text
.
├── benchmark/
│   ├── dataset.json          # Curated benchmark cases
│   └── run_eval.py           # Evaluation runner
├── src/
│   ├── analyzer.py           # Music context parser
│   ├── models.py             # Pydantic data models
│   ├── harmony_engine.py     # Rule-based candidate generation
│   ├── retriever.py          # Knowledge retrieval layer
│   ├── reranker.py           # LLM ranking layer
│   └── validator.py          # Music validation and scoring
├── tests/
│   └── ...                   # Unit tests
├── requirements.txt
├── README.md
└── ROADMAP.md
```

No implementation code has been added yet. This README defines the intended architecture and resource direction before development begins.

## Technical Stack

Recommended backend stack:

- Python 3.11 or 3.12
- FastAPI for API service, when the backend service starts
- Pydantic for structured music and API models
- music21 for symbolic music parsing
- NumPy for numeric pitch and interval operations
- pytest for tests
- Qdrant, Milvus, Weaviate, or pgvector for vector retrieval
- Neo4j or PostgreSQL for chord graph and structured harmony relationships
- vLLM or Ollama for self-hosted model serving
- OpenAI-compatible API format for model provider abstraction

## Recommended Environment

### Development Environment

For README planning, benchmark design, and rule-engine development:

- CPU: 4 to 8 cores
- RAM: 16 GB minimum, 32 GB recommended
- GPU: not required
- Storage: 20 to 50 GB
- OS: Ubuntu 22.04 or 24.04
- Python: 3.11 or 3.12
- Docker: recommended

### MVP Inference Environment

For running a 7B or 8B open-source reranker model:

- GPU: 1 x NVIDIA GPU with 16 GB VRAM minimum
- Recommended GPU: RTX 4090, L4, A10, A10G, A100, or H100
- RAM: 32 GB minimum
- Storage: 100 GB minimum
- CUDA: 12.1 or newer
- Serving: vLLM recommended
- Quantization: AWQ, GPTQ, or 4-bit quantization acceptable for early testing

A 7B model can usually run on 16 GB VRAM with quantization. For higher throughput or longer context, 24 GB VRAM is more comfortable.

### Knowledge Base Environment

For RAG and structured harmony search:

- Vector DB: Qdrant is recommended for the first version because it is simple to operate in Docker.
- Graph DB: Neo4j is useful later for chord-function relationships, but PostgreSQL may be enough for the MVP.
- Storage: start with 20 GB; increase as score examples, MIDI, MusicXML, and embeddings grow.

### Training / Fine-tuning Environment

Not needed for Phase 1 or MVP.

If later training a reward model or ranker:

- GPU: A100 40 GB, A100 80 GB, H100, or multiple 4090-class GPUs
- RAM: 64 GB or more
- Storage: 500 GB or more
- Framework: PyTorch + Hugging Face Transformers + PEFT / LoRA

## Recommended Docker Images

For Phase 1 and MVP development:

```text
python:3.11-slim
```

For GPU model serving with vLLM:

```text
vllm/vllm-openai:latest
```

For vector search:

```text
qdrant/qdrant:latest
```

Optional graph database:

```text
neo4j:5
```

A practical first deployment can use three services:

```text
backend: Python / FastAPI / music21
model-server: vLLM OpenAI-compatible server
vector-db: Qdrant
```

## Recommended Models

### First model to test

Use one of these as the first open-source reranker / reasoning model:

- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct, if more VRAM is available
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct

Recommendation:

```text
Qwen2.5-7B-Instruct through vLLM
```

Reason:

- strong instruction-following ability
- good structured output behavior
- can run on modest GPU resources
- suitable for candidate ranking and explanation

### Embedding model

For early RAG experiments:

- BAAI/bge-small-en-v1.5
- BAAI/bge-base-en-v1.5
- intfloat/e5-base-v2

If Chinese and English mixed knowledge is expected:

- BAAI/bge-m3

Recommendation:

```text
BAAI/bge-m3
```

because the project notes and music reasoning may mix Chinese and English.

## Implementation Roadmap

### Milestone 1: Base LLM Benchmark

- Create structured benchmark dataset.
- Test raw LLM ability on jazz voicing tasks.
- Build deterministic scoring functions.
- Produce baseline report.

### Milestone 2: Harmony Engine MVP

- Implement chord parsing.
- Generate rootless and Drop 2 voicings.
- Add range, avoid-note, guide-tone and voice-crossing checks.
- Return 3 to 10 valid candidates per chord context.

### Milestone 3: Knowledge Base and RAG

- Store chord substitution cases.
- Store voicing decision examples.
- Add embedding search.
- Retrieve context before candidate generation and ranking.

### Milestone 4: LLM Reranker

- Feed candidates, previous context and retrieved examples to the LLM.
- Ask for structured ranking and explanation.
- Validate final answer after ranking.

### Milestone 5: Feedback Loop

- Capture which candidate the user chooses.
- Build a preference dataset.
- Train or fine-tune a small ranker when enough feedback exists.

## Current Decision Points

Before implementation, confirm:

1. First target instrument: piano only, or piano plus guitar?
2. First style scope: jazz standard / ballad / bebop / gospel / pop?
3. First model endpoint: self-hosted vLLM or external API?
4. First database stack: Qdrant only, or Qdrant plus Neo4j/PostgreSQL?
5. Expected input format: chord chart text, MIDI, MusicXML, or all three?
6. Whether explanations should be in English, Chinese, or bilingual.

---

This README was drafted by an AI agent (OpenHands) on behalf of the repository owner.
