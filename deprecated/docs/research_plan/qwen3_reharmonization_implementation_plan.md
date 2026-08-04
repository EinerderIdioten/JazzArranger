# Qwen3-Coder-1.7B Jazz Reharmonization 研究计划

## 1. 思路判断

当前方案是合理的：第一阶段不直接训练最终 reharmonization，而是先训练一个能理解“旋律音对当前和弦的支撑作用”的模型。这个模型可以作为 teacher，为没有细粒度标注的 jazz 数据生成派生分析。后续做 reharmonization 时，可以保留 Qwen3-Coder-1.7B backbone，但更换输出头和训练目标。

核心原则是：

- 不把大量 derived 信息挂到每个音符的原始存储里。
- 先把不同数据集归一成同一种 lead-sheet 格式。
- 第一阶段训练 note-to-chord support，而不是 final chord generation。
- 第二阶段以后把 teacher 标注出的 latent / score 用于 reharmonization 数据构建。

## 2. 一阶段数据集

一阶段只使用有明确 melody-harmony 对齐价值的数据：

1. **OpenBook**  
   仓库已有 canonical lead-sheet JSON，是 jazz lead sheet 主数据源。

2. **HLSD / HookTheory**  
   有 melody、harmony、key、beat 对齐，适合训练旋律与和弦的局部关系。

3. **POP909**  
   有 melody、piano accompaniment、beat、key、chord 标注，适合补充流行语境下的旋律-和声关系。

4. **EMOPIA+**  
   有 melody、chord、key-relative functional representation，适合补充功能和声表示和 piano performance 相关信息。

第一阶段暂不使用 MetaMIDI、GigaMIDI、Discover、Godzilla、LMD 这类大规模 MIDI 仓库作为主监督数据。它们可以留到后续做弱监督扩展或风格补充。

## 3. 统一数据格式

四个数据集需要先归并到统一的 canonical lead-sheet 格式。源数据仍保留原始文件，训练只消费派生视图。

### 3.1 Canonical storage

```json
{
  "schema": "lead_sheet.v0",
  "context": {
    "title": "...",
    "key": "C major",
    "meter": "4/4",
    "tempo": 120,
    "style": "jazz_ballad",
    "source_dataset": "openbook|hlsd|pop909|emopia_plus"
  },
  "harmony_stream": [
    [1, 1.0, "Dm7", 4.0]
  ],
  "melody_stream": [
    [1, 1.0, "F4", 2.0, "unknown"]
  ],
  "review": {
    "status": "auto_converted",
    "notes": []
  }
}
```

### 3.2 Derived analysis 不进入 raw stream

第一阶段 teacher 产生的分析应放在单独 sidecar 或 `analysis` 层里，不写进 `melody_stream`：

```json
{
  "schema": "note_chord_support.v0",
  "source_id": "...",
  "teacher_version": "qwen3_support_teacher_v1",
  "note_support": [
    {
      "bar": 1,
      "beat": 1.0,
      "pitch": "F4",
      "active_chord": "Dm7",
      "support_score": 0.86,
      "role_distribution": {
        "root": 0.02,
        "third": 0.81,
        "fifth": 0.03,
        "seventh": 0.04,
        "tension": 0.08,
        "non_chord": 0.02
      },
      "confidence": 0.91
    }
  ]
}
```

训练输入不需要把这些字段全部展开给模型。它们主要用于监督 loss、teacher 输出缓存、后续数据过滤和再标注。

## 4. 一阶段训练目标

一阶段目标是训练一个 **note-to-current-chord support teacher**。

输入：

- melody window
- 当前 active chord
- key / meter / beat position
- note onset / duration / optional velocity
- 前后少量上下文 chord

输出：

- 当前音对 active chord 的支撑分数 `support_score`
- 当前音相对 active chord 的角色分布
- 可选的 confidence

这里不使用简单的 `structural / passing` 二分类，因为这个标签太粗。模型需要学习的是连续的、和声条件化的支撑关系。

推荐监督项：

- `role_distribution`：root / third / fifth / seventh / tension / non_chord
- `support_score`：连续回归
- `active_chord_root`：辅助预测
- `active_chord_quality`：辅助预测
- `chord_change_position`：辅助预测

## 5. 模型结构

第一阶段不需要单独做 music encoder 和 factor encoder 两个大模块。建议结构为：

```text
canonical/rendered lead-sheet view
  -> Qwen3-Coder-1.7B backbone
  -> support analysis head
```

### 5.1 Backbone

使用 Qwen3-Coder-1.7B。可以全量继续训练，不必长期冻结。

### 5.2 一阶段输出头

一阶段输出头是 support analysis head，用于每个候选 note 的监督：

- support score regression head
- role distribution classification head
- auxiliary root / quality / change heads

为了复用 Qwen 的 LM 能力，也可以先用 mask-token logits 实现一版离散分类头，再逐步替换成显式 projection head。

### 5.3 Reharmonization 阶段输出头

后续 reharmonization 不沿用 support analysis head，而是更换为生成式 harmony head：

```text
Qwen3-Coder-1.7B backbone
  -> reharmonization generation head
  -> canonical-style harmony_stream
```

输出目标包括：

- chord change timing
- root
- quality
- tension
- slash bass
- optional voicing / bass plan

因此，一阶段 teacher 和最终 reharmonizer 共享 backbone 思路，但输出头和 loss 不同。

## 6. 训练阶段

### Phase 0：数据转换

- OpenBook 保持现有 canonical 格式。
- HLSD 转成 `harmony_stream` + `melody_stream`。
- POP909 从 MIDI、`chord_audio/beat_audio.txt`、`key_audio.txt` 转成 canonical。
- EMOPIA+ 从 functional / lead-sheet 表示转成 canonical。
- 所有数据按 tune-level split，避免同一曲目的转调版本泄漏到不同 split。

### Phase 1：support teacher 训练

- 输入 melody + current chord context。
- 输出 note-to-chord support。
- 训练 Qwen3-Coder-1.7B backbone + support analysis head。
- 可以先冻结底层少量 embedding / tokenizer warmup，再全量训练 backbone。

### Phase 2：teacher 标注

- 用 Phase 1 teacher 标注 OpenBook 和后续 jazz 数据。
- 生成 `note_chord_support.v0` sidecar。
- 根据 confidence 过滤低质量样本。

### Phase 3：reharmonization 训练

- 替换 support analysis head 为 reharmonization generation head。
- 输入 melody + optional teacher sidecar summary。
- 输出 canonical-style `reharmonized_harmony_stream`。
- 先训练 chord skeleton，再扩展到 tension、slash bass、voicing。

## 7. 当前最小可行版本

最小可行实现建议：

1. 把 OpenBook、HLSD、POP909、EMOPIA+ 统一成 canonical lead-sheet JSON。
2. 生成 note-chord candidate rows。
3. 使用 Qwen3-Coder-1.7B 训练 support teacher。
4. teacher 输出 `note_chord_support.v0` sidecar。
5. 后续 reharmonization 阶段更换输出头，不复用第一阶段 support head。

这个版本足够短，也和仓库现有 tokenizer、canonical storage、candidate training 思路保持一致。
