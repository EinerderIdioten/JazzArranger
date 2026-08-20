# 第三方和声验证工具实验计划

日期：2026-08-20（Asia/Shanghai）
阶段：Phase 1 后续验证 / Stage 1A-v4 之前的结构诊断
目标：先验证“形式化和声工具能不能当 oracle”，再决定是否把 PCFG/PACFG/JHT 线索接入生成 pipeline。

## 1. 结论先行

下一步最重要的不是马上训练新的生成器，而是先做一个第三方验证工具：

```text
melody + candidate harmony + reference harmony -> structured validation report
```

这个工具的角色不是替代模型生成和声，而是作为外部 critic / parser / reranker：

1. 自动复现 Phase 1 人工观察到的错误类型；
2. 判断 reference 和 candidate 在功能骨架、和声节奏、调性锚点、富和弦使用上是否可区分；
3. 为 Stage 1A-v4、检索增强和后续 rerank 提供可量化指标。

如果这个工具不能把明显错误的候选和 reference 拉开，就说明“形式化语法当 oracle”这个前提不成立。在这个前提没有验证之前，不应把 PACFG/JHT 直接变成生成前的强约束。

## 2. 为什么要先做验证器

当前项目最新记录显示，`1.7B` 小模型可以学到爵士和声表面词汇，但稳定性不足：

- 容易过度使用 `7` / `7b9` / altered dominant；
- 容易把真实功能链改写成泛化 `ii-V-I`；
- split-bar、终止式、转位、借和弦处理不稳；
- 小调和长段落存在 root drift。

这说明失败不是单一 token 分类错误，而是结构性错误。对应的工具优先级应该是：

| 问题 | 需要验证器检查什么 |
|---|---|
| 富和弦坍缩 | chord quality / extension / slash 的覆盖率和上下文合理性。 |
| 0% 小节内分割 | harmonic rhythm density、sub-bar onset、span 对齐。 |
| Stella / ATTYA 类离调曲目崩溃 | local key plan、tonicization、借用/离调链路。 |
| 平行乐句系统性 off-by-one | phrase-level anchor consistency、重复乐句的功能对齐。 |
| LLM 自评不稳 | 使用外部规则和 parser，不依赖 Self-Refine 式自我判断。 |

## 3. 工具边界

### 3.1 要做

验证器第一版只做离线评估，不参与训练。它不是一个单一函数，而是 5 个模块串起来：

```text
raw reference / candidate / melody
        |
        v
Normalizer -> ReferenceChecker -> HarmonyParser -> Comparator -> BatchReporter
        |
        v
JSON metrics + error tags + bar-level diagnostics
```

第一版必须支持三类用法：

| 用法 | 输入 | 输出 |
|---|---|---|
| reference sanity check | melody + reference harmony | reference 本身是否能被 parser / rules 解释。 |
| candidate vs reference | melody + candidate + reference | candidate 相对 reference 的错误类型和差距。 |
| batch regression | Phase 1 全量候选 | 自动统计每类错误命中率。 |

### 3.2 不做

第一版不做这些事：

| 暂不做 | 原因 |
|---|---|
| 不直接生成和弦 | 当前要先证明 validator 有区分力。 |
| 不做 LLM Self-Refine | Phase 1 已经显示模型自我识别错误不稳。 |
| 不把 JHT 树当唯一真值 | JHT tree 是专家分析 proposal，不是唯一合法解释。 |
| 不要求从 melody 直接预测完整语法树 | 这是鸡生蛋问题，推理时没有 gold chord sequence。 |
| 不做 voice-leading RL | root / quality / harmonic rhythm 尚未稳定，RL 先不上主线。 |

## 4. PCFG / PACFG / JHT 线索取舍

### 4.1 真正有价值的贡献

| 贡献 | 对本项目的实际价值 |
|---|---|
| latent harmonic rhythm | chord 数量和 onset 可以作为隐变量，而不是固定每小节一个和弦。直接对应当前 0% split-bar 问题。 |
| tree-structured phrase grammar | 和声不是逐拍独立分类，树结构能表达短语、终止式和功能链的层级关系。 |
| local key + function abstraction | PACFG/JHT 的核心价值是把 surface chord 映射到调性中心和功能角色，适合检查离调段落。 |
| parser as critic | 候选和弦生成后，可以用 parser 检查功能骨架是否自洽。这个角色比“直接生成器”更适合当前项目。 |
| tree-level diagnostics | 可以给出 bar / phrase / cadence 级错误标签，比单纯 root accuracy 更接近音乐错误。 |

### 4.2 不应照搬的部分

| 不照搬 | 原因 |
|---|---|
| Tsushima demo 生成结果 | 20 个样例里 PCFG-SS-HMM 仍有大量 root / rhythm 偏离，不能作为目标生成质量。 |
| triad-only chord vocabulary | 只覆盖 major/minor triads、other、NC，无法处理爵士标准曲的 7th、alteration、slash、half-dim 等。 |
| unsupervised nonterminal 语义 | PCFG-US 的 node label 难解释，不能直接当功能标签。 |
| 小样本 tree F1/accuracy | 只能说明语法抽象有研究价值，不能证明 parser 已经是可靠 oracle。 |
| 生成前强约束 | 没有 candidate chords 时，完整语法树无法解析；生成前最多提供弱骨架条件。 |

因此，本项目应学习的是：

```text
formal representation + parser scoring + diagnostic labels
```

而不是学习：

```text
PCFG demo harmonizer 的生成方式或窄词表
```

## 5. 输入输出规范

### 5.1 输入 JSONL

每行一个 tune 或片段。第一版推荐 schema：

```json
{
  "case_id": "phase1_001",
  "title": "Stella by Starlight",
  "key": "C:maj",
  "meter": "4/4",
  "form": {
    "sections": [
      {"label": "A1", "bar_start": 1, "bar_end": 8},
      {"label": "A2", "bar_start": 9, "bar_end": 16}
    ]
  },
  "melody": {
    "format": "abc",
    "text": "..."
  },
  "reference_harmony": [
    {"bar": 1, "start": 0.0, "end": 2.0, "chord": "Em7b5"},
    {"bar": 1, "start": 2.0, "end": 4.0, "chord": "A7b9"}
  ],
  "candidate_harmony": [
    {"bar": 1, "start": 0.0, "end": 4.0, "chord": "Cmaj7"}
  ],
  "candidate_source": "gpt55_phase1"
}
```

`form` 可以缺省。若没有 AABA / ABAC 标注，第一版仍可运行 bar-level 和 phrase-free 指标，但 phrase anchor 检查会降级。

### 5.2 输出 JSON

每个 case 输出：

```json
{
  "case_id": "phase1_001",
  "summary": {
    "validator_score": 0.42,
    "reference_score": 0.81,
    "candidate_score": 0.39,
    "error_tags": ["rich_quality_collapse", "missing_subbar_split", "local_key_drift"]
  },
  "metrics": {
    "root_match": 0.06,
    "quality_family_match": 0.12,
    "harmonic_rhythm_f1": 0.18,
    "subbar_split_recall": 0.0,
    "rich_chord_recall": 0.04,
    "functional_anchor_match": 0.22
  },
  "bar_diagnostics": [
    {
      "bar": 1,
      "tags": ["missing_subbar_split", "wrong_function"],
      "reference": ["Em7b5", "A7b9"],
      "candidate": ["Cmaj7"]
    }
  ]
}
```

第一版不要求 score 是绝对音乐质量分。它只需要能稳定排序：

```text
reference > strong candidate > weak candidate > collapsed baseline
```

## 6. 评分维度

### 6.1 Harmonic Rhythm Validator

目的：检查“什么时候换和弦”。

指标：

| 指标 | 说明 |
|---|---|
| chord-count MAE | 每 bar 候选 chord 数和 reference 的差距。 |
| onset F1 | 允许小容差的 chord onset 匹配。 |
| subbar split recall | reference 有小节内分割时，candidate 是否也分割。 |
| density class accuracy | 每 bar 分类为 1 / 2 / 3+ chords。 |

优先命中的错误：

```text
0% 小节内分割
Satin Doll 类高 split-bar 失败
```

### 6.2 Chord Token Validator

目的：检查 root / quality / extension / slash 的结构化覆盖。

建议 tokenization：

```text
root + joint_form_token
```

示例：

```text
G7b9      -> <R_G> <F_7b9>
Bm7b5     -> <R_B> <F_m7b5>
F/A       -> <R_F> <F_maj_slash_3>
G13sus    -> <R_G> <F_13sus>
```

第一版不要拆成 pitch-class 集合。原因是项目当前问题不是不会枚举音级，而是保留不了爵士风格里的 chord form 和长尾性质。

指标：

| 指标 | 说明 |
|---|---|
| root match | root 是否与 reference 对齐。 |
| form exact match | joint form token 是否精确匹配。 |
| form family match | 例如 dom / minor / half-dim / dim / sus / slash family。 |
| rich chord recall | reference 中 7th 以上、altered、slash、dim、m7b5 是否被候选覆盖。 |
| collapse index | 候选落入 maj7/min7/7 的比例相对 reference 的偏移。 |

### 6.3 Functional Skeleton Validator

目的：检查局部调性和功能骨架。

第一版不要求从 melody 预测骨架，而是从 chord sequence 解析候选骨架：

```text
candidate chord sequence -> possible local key/function parse -> score
reference chord sequence -> possible local key/function parse -> score
```

输出标签：

| 标签 | 含义 |
|---|---|
| `local_key_drift` | 候选解析出的局部调性中心与 reference 大幅偏离。 |
| `wrong_cadence_target` | 终止式落点错误。 |
| `dominant_chain_broken` | 属功能链中断或被错误替换成普通 diatonic chord。 |
| `borrowed_chord_missing` | reference 有借用/离调色彩，candidate 坍缩成普通调内和弦。 |
| `overgeneralized_ii_v_i` | 候选把 reference 的具体功能链泛化成模板化 ii-V-I。 |

实现优先级：

1. 先做规则式 romanization / local-key beam search；
2. 再接 PACFG/JHT-style parser score；
3. 最后才考虑把 parser score 用作 rerank feature。

### 6.4 Phrase Anchor Validator

目的：检查平行乐句的调性锚点是否一致。

输入依赖：

| 元信息 | 是否必须 |
|---|---|
| bar number | 必须。 |
| phrase / section label | 推荐，但可缺省。 |
| repeated melody motif detection | 后续补充。 |

第一版如果没有曲式标注，就先用 melody contour 相似度找候选平行乐句：

```text
phrase A melody contour ~= phrase A' melody contour
```

然后检查：

```text
reference anchors: Am7 -> Dm7 -> G7 -> Cmaj7
candidate anchors:  Cmaj7 -> Fmaj7 -> G7 -> Cmaj7
```

若相似错误在平行乐句中重复出现，打标签：

```text
parallel_phrase_anchor_shift
```

这个检查直接对应 Fly Me to the Moon 里 `Am7 -> Cmaj7`、`Dm7 -> Fmaj7` 的系统性 off-by-one。

### 6.5 Melody Support Validator

目的：检查 chord 是否至少能解释旋律重拍音。

第一版只做弱规则，不把它当最终裁判：

| 指标 | 说明 |
|---|---|
| strong-beat chord-tone support | 重拍长音是否落在 chord tones / accepted tensions。 |
| unsupported accent count | 重拍音与候选和弦冲突的次数。 |
| melody-cadence agreement | 乐句末尾 melody 与候选 cadence target 是否一致。 |

注意：爵士旋律中 tension、approach tone、enclosure 很常见，因此 melody support 只能作为辅助指标，不能单独否决富和弦。

## 7. 实验顺序

### V0：整理评估集

目标：把 Phase 1 的人工观察变成可跑的 JSONL。

产物：

```text
evals/phase1_harmony_validator_cases.jsonl
evals/phase1_harmony_validator_manual_tags.json
```

必须包含：

| 字段 | 说明 |
|---|---|
| melody | ABC melody。 |
| reference_harmony | reference chord spans。 |
| candidate_harmony | GPT-5.5 / 1.7B / 其他候选。 |
| manual_error_tags | 人工标注错误类型，用于验证自动工具召回率。 |
| case metadata | title、key、meter、bar range、candidate source。 |

### V1：无语法 baseline validator

先不接 PCFG/PACFG，只做强规则 validator 基础版。这里的 V1 就是：

- `Normalizer`
- `ReferenceChecker`
- `HarmonyParser` 的最小规则集
- `Comparator` 的 root / quality / rhythm / melody-fit 距离
- `BatchReporter`

先跑 deterministic metrics：

1. root / form / rich chord coverage；
2. chord-count 和 onset F1；
3. subbar split recall；
4. collapsed form distribution；
5. candidate vs reference chord edit distance。

验收：

```text
能自动复现 Phase 1 中 “6 种性质 vs reference 19 种”
能自动复现 “0% 小节内分割 vs reference 26.4%”
```

### V2：规则式功能骨架 parser

在候选和 reference 都已有 chord sequence 的前提下，做 local-key / function beam search。

第一版输出：

```text
bar -> top-k local key
bar -> top-k function labels: T / PD / D / prolongation / tonicization / borrowed
cadence candidates
dominant-chain candidates
```

验收：

```text
在 Stella / All The Things You Are 等离调密集曲目上，
reference 的 parser score 应显著高于 collapsed candidate。
```

如果 reference 自身也经常低分，说明规则太窄，不能进入 V3。

### V3：PACFG/JHT-style score

在 V2 有基本区分力后，再引入 PACFG/JHT 风格的 tree score：

1. 把 chord spans 转成局部调性 + roman/function token；
2. 用小型手写 grammar 或 JHT-derived grammar 解析；
3. 输出 tree likelihood / normalized parse score / parse failure tags；
4. 不要求生成树唯一，只要求 reference 的 top-k parse 更合理。

验收：

```text
reference_score > candidate_score
且该差距能覆盖人工标注的 high-level 错误：
local_key_drift
wrong_cadence_target
parallel_phrase_anchor_shift
overgeneralized_ii_v_i
```

### V4：作为 reranker 的离线实验

只有 V1-V3 通过后，才把 validator 接到候选排序：

```text
N candidate harmonies
        |
        v
validator features
        |
        v
rerank
        |
        v
best candidate
```

第一版 rerank 不训练新模型，使用线性加权：

```text
score =
  w_rhythm * harmonic_rhythm_score
+ w_form   * rich_form_score
+ w_func   * functional_parse_score
+ w_phrase * phrase_anchor_score
+ w_melody * melody_support_score
```

验收：

```text
rerank 后的 top-1 在人工盲测中明显少于原始 top-1 的结构性错误。
```

## 8. 验收标准

第一版工具要先满足工具自身的验收，而不是模型质量验收。

### 8.1 自动标签召回

用 Phase 1 人工标签做对照：

| 错误类型 | 自动工具最低目标 |
|---|---:|
| rich_quality_collapse | recall >= 0.80 |
| missing_subbar_split | recall >= 0.90 |
| local_key_drift | recall >= 0.50 起步 |
| wrong_cadence_target | recall >= 0.60 起步 |
| parallel_phrase_anchor_shift | 先做 case study，不设全局阈值 |

### 8.2 Reference / Candidate 区分力

对每个 case 同时跑 reference 和 candidate：

| 指标 | 目标 |
|---|---|
| reference parse failure rate | 尽量低；若高，说明规则太窄。 |
| reference_score - candidate_score | 在明显错误样本上应为正。 |
| collapsed baseline score | 应低于真实 candidate 和 reference。 |
| bar-level diagnostics precision | 人工抽查时，不应大量误报正常 jazz tension。 |

### 8.3 不通过时的决策

| 失败现象 | 决策 |
|---|---|
| reference 也频繁 parse fail | parser 太窄，先扩展规则，不进入 rerank。 |
| reference 和 candidate 分不开 | PACFG/JHT 不能直接当 oracle，只保留 rhythm/token metrics。 |
| melody support 大量误报 | 降低 melody support 权重，避免误伤 tension。 |
| phrase anchor 依赖曲式字段但数据缺失 | 先做人工标注小集或 contour-based 弱检测。 |

## 9. 和 Stage 1A-v4 的关系

这个验证器不替代 Stage 1A-v4，但会改变 v4 的诊断指标。

现有 v4 主要看：

```text
root accuracy
pred/gold entropy gap
pred <R_C> share
no-neighbor-root vs full-context accuracy gap
```

验证器补充看：

```text
harmonic rhythm density
rich form coverage
local key / function parse score
parallel phrase anchor consistency
cadence correctness
```

如果 v4 root accuracy 上升，但 validator 显示：

```text
rich chord collapse 仍在
subbar split recall 仍接近 0
local key drift 仍严重
```

则不应进入 Stage 2。说明模型只是学会了更好的局部 root 分类，还没有稳定 melody-to-harmony labeling。

## 10. 开放问题

| 问题 | 需要确认 |
|---|---|
| Phase 1 的 367 bars 是否已有结构化数据 | 若只有报告，需要先回填 JSONL。 |
| 是否有曲式/phrase 标注 | 没有则 phrase anchor validator 先降级。 |
| JHT 数据是否纳入仓库或只作为外部 benchmark | 涉及版权和数据来源管理。 |
| chord parser 的目标粒度 | 第一版建议 root + joint form，不拆 pitch-class。 |
| local key parser 是否允许多解释 | 必须允许 top-k，不要强行单一局部调性。 |

## 11. 推荐下一步

按低风险到高风险执行：

1. 新建 `evals/phase1_harmony_validator_cases.jsonl`，把 Phase 1 样本整理成统一输入。
2. 实现 V1 deterministic metrics，先复现富和弦坍缩和小节内分割缺失。
3. 用 V1 跑 reference / GPT-5.5 / 1.7B 候选，确认报告稳定。
4. 实现 V2 local-key / function beam parser，只在 chord sequence 已知后解析。
5. 用 Stella、All The Things You Are、Fly Me to the Moon 做重点 case study。
6. 只有 V2 能把 reference 和坏候选拉开时，再接 PACFG/JHT-style tree score。

一句话原则：

```text
先把形式化工具做成可验证的 critic，再考虑让它影响生成。
```

## 12. 具体构造（v1 实施版）

### 12.1 模块拆分

| 模块 | 输入 | 输出 | 职责 |
|---|---|---|---|
| `Normalizer` | raw chord strings、raw spans、meter、bar grid | normalized chord events | 规范化符号、别名、切分、拍号与小节对齐。 |
| `ReferenceChecker` | normalized reference | reference sanity report | 检查 reference 是否可作为稳定监督。 |
| `HarmonyParser` | normalized chord stream + optional melody + optional form | IR、top-k parse、confidence | 只做强规则/可回归解析，不做 learned judge。 |
| `Comparator` | ref IR + cand IR + melody | distances、tags、bar diagnostics | 在统一表示上计算距离并贴错误标签。 |
| `BatchReporter` | batch comparison outputs | aggregates、worst cases、diffs | 聚合一批样本，输出系统性分布与回归结果。 |

### 12.2 统一中间表示

一切比较都基于统一中间表示，不直接比原始 chord string。

#### `ChordEvent`

最小字段建议：

```json
{
  "root": "G",
  "quality_family": "dom",
  "extensions": ["9", "13"],
  "alterations": ["b9"],
  "bass": "D",
  "duration_span": [2.0, 4.0],
  "bar_index": 12,
  "beat_range": [1.0, 3.0],
  "function_role": "D",
  "local_key": "C:maj",
  "cadential_role": "cadence",
  "parse_confidence": 0.91
}
```

#### `Chunk`

`Chunk` 是局部进行单元，用来表达比单和弦更高一层的结构：

- `ii-V-I`
- `backdoor`
- `turnaround`
- `tonic_prolongation`
- `dominant_preparation`

#### `Tree`

`Tree` 只负责 phrase / section / cadence 级 skeleton：

- skeleton vs elaboration
- dominant-to-tonic 是否保留
- 平行乐句的功能锚点是否对齐

不把每个细节都树化。

### 12.3 Parser 的具体输出

`HarmonyParser` 第一版输出：

| 输出 | 含义 |
|---|---|
| normalized chord stream | 规范化后的 chord events。 |
| harmonic role | `tonic / predominant / dominant / other / ambiguous`。 |
| local key region | 当前局部调性中心，允许 top-k。 |
| cadence / turnaround / substitution pattern | 终止式、回转、替代模式。 |
| parse confidence | 解析置信度。 |

### 12.4 距离函数

validator 的核心不是 exact match，而是分层 harmony distance。

单和弦距离定义为：

```math
d_{event} = w_r d_r + w_q d_q + w_x d_x + w_b d_b + w_f d_f + w_m d_m
```

其中：

- `d_r`：root distance
- `d_q`：quality-family distance
- `d_x`：extension / alteration distance
- `d_b`：bass / inversion distance
- `d_f`：local functional distance
- `d_m`：melody compatibility delta

#### root distance

不是简单同根 / 不同根，而是结合：

- pitch-class 环距离
- 五度圈邻近性
- 功能邻近性
- 三全音替代可接受性

#### quality-family distance

quality 不做平铺 one-hot，而做 family：

| family | 例子 |
|---|---|
| `maj` | `maj`, `maj7`, `6` |
| `min` | `min`, `min7`, `min9` |
| `dom` | `7`, `9`, `11`, `13`, `7b9`, `7#9`, `7alt` |
| `hdim` | `m7b5` |
| `dim` | `dim`, `dim7` |
| `sus` | `sus2`, `sus4`, `7sus` |
| `slash` | `F/A`, `Dm/F` |
| `other` | `NC`, `other` |

同 family 低罚，不同 family 高罚。

#### extension / alteration distance

默认是小罚项，但如果 alteration 改变了核心功能，就放大惩罚。

#### bass / inversion distance

slash bass 不能默认忽略，因为它常常承载 voice-leading 和功能线索。

#### functional distance

这是最重要的一层之一。它依赖局部上下文，不是单个 chord label。

#### melody-fit distance

强拍长音与 chord tones / accepted tensions 的兼容关系单独计入。

### 12.5 序列和小节级距离

单和弦距离还不够，还要做 bar-level alignment：

```math
d_{bar} = \min_{\pi \in \text{alignments}} \sum_{(i,j)\in \pi} d_{event}(r_i, c_j) + \text{gap penalties}
```

gap penalty 分层：

- 漏掉关键切分：重罚
- 多出局部 embellishment：轻罚

### 12.6 轨道距离 / 允许变换系统

工程上不必强行追求严格群公理，更适合“带代价的允许变换系统”。

可先开放的动作：

- transposition normalization
- tritone substitution
- dominant-family variation
- contextual substitution

轨道距离写成：

```math
d_{orbit}(x,y) = \min_{a \in \langle \mathcal{A} \rangle} \mathrm{cost}(a : x \to y)
```

这里的重点是：

- 变换显式
- 代价可修订
- 只在乐理上有意义时开放

### 12.7 错误标签

错误标签应从距离函数导出，而不是与分数系统平行独立。

建议初版标签：

- `UNPARSABLE_SYMBOL`
- `BAR_DURATION_INCONSISTENT`
- `REF_PARSE_LOW_CONFIDENCE`
- `INTRA_BAR_SPLIT_MISSING`
- `INTRA_BAR_SPLIT_EXTRA`
- `ROOT_MISMATCH`
- `FUNCTION_NEIGHBOR_CONFUSION`
- `QUALITY_COLLAPSE_BASIC7`
- `EXTENSION_DROPPED`
- `ALTERATION_DROPPED`
- `SLASH_BASS_INFORMATION_LOST`
- `CADENTIAL_PATTERN_ERASED`
- `LOCAL_TONAL_DRIFT`
- `MELODY_STRONG_BEAT_CONFLICT`

导出逻辑示例：

- `d_q` 高，且 reference 属于 rich family、candidate 落入 safe family -> `QUALITY_COLLAPSE_BASIC7`
- gap penalty 高，且 candidate 事件数少于 reference -> `INTRA_BAR_SPLIT_MISSING`
- 连续若干小节 `d_f` 高 -> `LOCAL_TONAL_DRIFT`

### 12.8 报告 schema

建议 JSON 报告至少包含：

```json
{
  "reference_sanity": {
    "syntax": "pass",
    "temporal": "pass",
    "structural": "pass",
    "parse": "low_confidence"
  },
  "summary": {
    "score": 0.78,
    "subscores": {
      "event": 0.82,
      "bar": 0.74,
      "tree": 0.69,
      "orbit": 0.83,
      "melody_fit": 0.71,
      "rhythm_align": 0.76
    }
  },
  "error_tags": ["LOCAL_TONAL_DRIFT"],
  "bar_diagnostics": [],
  "parse_trace": [],
  "alignment_trace": []
}
```

### 12.9 v1 / v1.5 / v2 边界

#### v1

实现：

- `Normalizer`
- `ReferenceChecker`
- root / quality / gap / melody-fit 距离
- bar-level alignment
- 第一批错误标签
- batch regression report

#### v1.5

加入：

- extension / bass / local function 距离
- `Chunk`
- 弱树结构
- contextual substitution 表

#### v2

加入：

- tree edit distance
- 更正式的 orbit distance
- section / form-aware consistency
- reward adapter（仅在以后需要 RL 时启用）

## 13. 参考资料与使用方式

| 资料 | 本计划中的用途 |
|---|---|
| [PCFG harmonization demo](https://pcfgharmonization.github.io/) | 作为反例和结构参考：学习 parser / tree / harmonic rhythm，不学习 demo 生成质量。 |
| [Tsushima et al., Bayesian Melody Harmonization](https://kyushu-u.elsevierpure.com/en/publications/bayesian-melody-harmonization-based-on-a-tree-structured-generati/) | 参考 tree-structured generative model、metrical Markov harmonic rhythm 和 melody likelihood 的联合建模思路。 |
| [ISMIR 2017 PCFG paper](https://archives.ismir.net/ismir2017/paper/000002.pdf) | 参考 latent chord onset、split-merge sampling、tree-structured parsing 的工程目标。 |
| [Harasim et al., PACFG, ISMIR 2018](https://ismir2018.ircam.fr/doc/pdfs/258_Paper.pdf) | 参考 local key + harmonic function 的抽象语法表示。 |
| [Jazz Harmony Treebank, ISMIR 2020](https://archives.ismir.net/ismir2020/paper/000080.pdf) | 作为 jazz standards 上的 tree annotation 格式和 parser/critic benchmark。 |

这些资料只作为第三方验证工具的理论和评估来源；不作为当前生成器的直接实现模板。
