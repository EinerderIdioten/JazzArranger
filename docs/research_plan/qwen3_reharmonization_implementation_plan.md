# Qwen3 Jazz Reharmonization 实施计划（v1）

这份计划基于仓库里现有的三块基础能力：

1. **Canonical storage**：`docs/realbook_ingestion/canonical_storage.md`
2. **Temporal rendering**：`docs/realbook_ingestion/model_temporal_rendering.md`
3. **Chord-aware tokenizer / harmony training**：`docs/realbook_ingestion/chord_aware_tokenizer.md`、`docs/realbook_ingestion/harmony_rhythm_training_strategy.md`、`docs/realbook_ingestion/harmony_rhythm_token_classifier.md`

目标不是先把模型做成一个通用音乐大模型，而是先让 **Qwen3-Coder-1.7B** 在符号音乐里学会：

- 识别旋律与和声的对齐关系
- 学会 chord root / quality / bass / timing 这些基础和声因子
- 最终根据旋律生成可用的 jazz reharmonization

---

## 1. 现有仓库约束

仓库当前已经把数据和训练问题切成了三层：

### 1.1 Canonical storage

Canonical JSON 里的核心结构是：

- `context`
- `harmony_stream`
- `melody_stream`
- `review`

这层是**源数据**，不能为了训练方便而改成密集 note-level factor 表。

### 1.2 Model-facing rendering

仓库已经定义了把 canonical 数据渲染成模型可读视图的方式：

- `event_grid`
- `compact_text`
- `prompt_view`
- `REMI-like token stream`

这意味着模型看到的不是原始文件，而是**经过时序重排的视图**。

### 1.3 Tokenizer / harmony scope

当前 tokenizer 方案已经支持：

- root tokens
- quality tokens
- tension tokens
- bass tokens
- surface chord tokens（可选）

并且当前 active scope 是：

- **root + quality**

这很重要：第一阶段先不要把所有因子都开满，而是先把最关键的和声因子学稳。

---

## 2. 总体策略

### 2.1 Qwen backbone 可以全量继续预训练

你说得对，Qwen3-Coder-1.7B **不需要长期冻结**。

更合理的做法是：

1. 先在更容易学的、结构化的符号音乐表示上做继续预训练
2. 再切到监督任务
3. 最后做 reharmonization 端到端微调

也就是说，Qwen 既可以当 backbone，也可以在后期全量更新。

### 2.2 不把因子挂到每个音上

你前面提到的 concern 很关键：

- 不要把 derived factor 塞进每个 note
- 不要让 note-level 输入变成一大串噪音

所以训练上只保留两类东西：

- **瘦输入**：旋律、节拍、时值、局部和声上下文
- **稀疏监督**：chord change、root、quality、bass、cadence

派生因子如果要用，应该进入：

- context embedding
- span-level latent
- auxiliary supervision

而不是每个 note 的字段。

---

## 3. 数据集选择

这里把数据分成三层：主监督、辅助监督、后期扩展。

### 3.1 主监督数据：先学会“旋律 → 和声”

这一层必须是**显式 harmony** 数据。

优先顺序建议是：

1. **OpenBook / 本仓库 canonical jazz lead sheets**  
   这是仓库的主数据源，格式已经统一成 canonical lead-sheet JSON。

2. **HookTheory / HLSD**  
   适合学 melody-harmony 对齐，是最直接的 lead-sheet 监督源之一。

3. **POP909**  
   有 beat / chord / key 标注，适合训练 chord change、section boundary、pop harmony 结构。

4. **EMOPIA+**  
   这套数据把 melody / chord / key 和 functional representation 连起来，适合做 harmony + performance 的桥接。

5. **PDMX（只在可解析 MusicXML 的部分上使用）**  
   适合作为公有领域谱面补充源，但不是最核心的 chord label 数据集。

### 3.2 辅助监督数据：扩大语料，但不直接当 chord truth

这一层更偏大规模符号音乐语言建模。

建议顺序：

1. **MetaMIDI**
2. **GigaMIDI**
3. **Discover MIDI Dataset**
4. **Godzilla MIDI Dataset**
5. **Lakh MIDI Dataset**

这些数据可以提供：

- 风格多样性
- 旋律轮廓
- 伴奏纹理
- 和声统计先验

但默认不要把它们当成真值 chord supervision。

### 3.3 后期扩展数据：更偏表演、控制、文本化

如果主任务已经稳定，再引入：

- **MAESTRO**：表演 timing / velocity / pedal
- **Slakh**：多轨 voicing / orchestration
- **MetaScore**：文本标签 / caption 控制
- **XMIDI**、社区 anime MIDI：情绪或风格补充

这些不建议放在第一阶段主任务里。

---

## 4. 数据格式落地方式

### 4.1 统一到 canonical lead-sheet JSON

所有能进入主流程的数据，尽量都转换成：

```json
{
  "schema": "lead_sheet.v0",
  "context": {...},
  "harmony_stream": [[bar, beat, symbol, duration_beats], ...],
  "melody_stream": [[bar, beat, pitch, duration_beats, tag], ...],
  "review": {...}
}
```

### 4.2 不在存储里加密集 factor 字段

如果要做 salience / root affinity / harmonic tension 这类信息，建议：

- 作为单独的 derived analysis 层存储
- 不写进 raw melody_stream
- 不作为主输入展开到每个 note

### 4.3 模型视图只用三种

1. `compact_text`
2. `prompt_view`
3. `candidate row` 形式的训练序列

原则是：

- 存储层可以完整
- 训练层必须瘦

---

## 5. 训练阶段设计

## Phase 0：数据准备和标准化

目标：把所有训练数据变成统一的 canonical 和 rendered 视图。

要做的事：

- OpenBook / 其他 lead-sheet 数据统一进 canonical storage
- tune-level 去重和 split
- 12-key transposition augmentation
- 生成 compact_text / prompt_view / candidate rows
- 记录每条样本的 provenance 和版本号

产物：

- canonical JSON
- transposed canonical JSON
- rendered training views
- split manifest

---

## Phase 1：Qwen 结构化继续预训练

目标：让 Qwen3-Coder-1.7B 熟悉这个项目的符号语言，而不是直接学成一个“随机文本生成器”。

训练输入建议：

- `compact_text`
- `prompt_view`
- 少量 `event_grid`

训练内容建议：

- chord / melody 的顺序建模
- bar / beat 位置的概念
- harmony 与 melody 的同步关系
- key / meter / style 的基础对齐

这一阶段可以全量继续预训练，不需要长期冻结 backbone。

但建议先从**最结构化的渲染视图**开始，而不是直接上原始 JSON。

---

## Phase 2：和声时序任务

目标：先让模型学会“哪里换和弦”。

训练任务：

- harmony-rhythm / chord-change prediction
- candidate-based binary labeling

这一步可以直接用仓库里已有的：

- `run_qwen3_harmony_rhythm_baseline.py`
- `train_qwen3_harmony_rhythm_token_classifier.py`

作为原型。

输入：

- 旋律候选点
- bar / beat / type / pitch / duration

输出：

- 是否换和弦

这一步的价值是给后续 reharmonization 提供稳定的和声边界。

---

## Phase 3：基础和声因子学习

目标：让模型学会 chord root / quality / bass 这类最基本的和声因子。

训练任务：

- root prediction
- quality prediction
- bass / inversion prediction
- chord span prediction

这里可以利用 `prepare_qwen_chord_tokenizer.py` 生成的 token 体系：

- `<ROOT:...>`
- `<QUAL:...>`
- `<BASS:...>`
- `<TENS:...>`

但第一轮仍建议只把 root 和 quality 作为主激活范围。

这一步的意义是：

- 先学“和弦是什么”
- 再学“怎么 reharmonize”

---

## Phase 4：reharmonization 主任务

目标：输入旋律，输出可用的 jazz 和声编排。

训练输入：

- 旋律的 compact / prompt view
- key / meter / tempo / style
- 压缩后的和声上下文

训练输出：

- canonical-style `harmony_stream`
- 可选的 bass / voicing plan

这一阶段不再只做分类，而是做真正的生成。

建议先从：

- `melody -> chord skeleton`

开始，再扩展到：

- `melody -> chord + bass`
- `melody -> chord + voicing`
- `melody -> full reharmonization`

---

## Phase 5：全量联合微调

当上面的任务都稳定后，再做全量联合微调：

- backbone 不再只看单一任务
- chord timing、root、quality、reharmonization 一起优化
- 让模型在统一 latent 中整合这些能力

这个阶段才适合把 Qwen 真正当成最终 backbone。

---

## 6. 冻结策略

你说得对：**不要三个部分从一开始一起训练**。

### 推荐顺序

#### 6.1 初始阶段

- backbone 可以全量继续预训练
- 但任务先单一：只做 canonical rendering / harmony timing / chord factor

#### 6.2 中间阶段

- 逐步加入 reharmonization
- 保持任务分离
- 不要一上来把所有头一起开

#### 6.3 最终阶段

- 允许 backbone 全量更新
- 做端到端联合微调

### 不建议

- 一开始就把 note-level 因子、和声生成、文本控制一起训
- 一开始就用所有数据集混合大乱炖

---

## 7. 具体代码落地顺序

建议按下面顺序实现：

### Step 1
补齐数据 ingress：

- 把 OpenBook / 其他主监督源统一成 canonical lead-sheet JSON
- 确保 `harmony_stream` / `melody_stream` 结构一致

### Step 2
完善 rendered views：

- `event_grid`
- `compact_text`
- `prompt_view`
- `candidate rows`

### Step 3
扩展 tokenizer：

- 固定 root / quality tokens
- 先不把 tension / bass / surface token 全开

### Step 4
先跑 harmony timing baseline：

- 验证 candidate coverage
- 验证 token-classifier 的稳定性

### Step 5
做 chord factor 训练：

- root / quality / bass
- 先在 OpenBook + HookTheory / POP909 / EMOPIA+ 上跑通

### Step 6
进入 reharmonization：

- 用 canonical prompt 直接生成 chord stream
- 评估和旋律兼容性、功能走向和可读性

---

## 8. 这一版计划的核心原则

1. **canonical storage 保持完整，不为模型妥协**
2. **训练输入必须瘦，不能把 factor 塞进每个音**
3. **Qwen3-Coder-1.7B 可以全量继续预训练**
4. **先学 timing，再学 factor，再学 reharmonization**
5. **数据集以 OpenBook 为主，外部 lead-sheet 数据做补充，超大 MIDI 语料做后期扩展**

---

## 9. 当前最建议的实际起点

如果现在只做一个最小可行版本，我建议：

1. **OpenBook canonical + prompt_view**
2. **Qwen3-Coder-1.7B 全量继续预训练**
3. **harmony-rhythm candidate classification**
4. **root / quality factor token 学习**
5. **melody -> chord skeleton reharmonization**

这条路最短，也最贴合当前仓库已经实现的基础设施。

---

## 10. 下一步可以补的内容

如果你要，我下一步可以继续把这份计划细化成：

- 一个**里程碑表**
- 一个**数据集优先级清单**
- 一个**按脚本拆分的开发任务表**

也可以直接把这份计划进一步改成更正式的项目设计文档。