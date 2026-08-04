# Stage 1 训练计划：Masked Harmony Completion

## 1. Stage 1 目标

Stage 1 不直接训练最终的 ABC-only full generation。它是一个短 curriculum，
目标是在 gold span 边界和部分 harmony context 已知的条件下，让模型先学会：

1. root recovery；
2. quality recovery（可选短阶段）；
3. root + quality joint chord completion。

这样做可以先建立旋律、span、quality、root、局部进行之间的关系，再进入
Stage 2 的完整任务：

```text
<MELODY> ABC </MELODY> -> <HARMONY> full span list </HARMONY>
```

Stage 1 仍然使用 causal LM，不加 LoRA、不加 adapter、不加 note-role head、
不加 change-point head、不做 DPO / GRPO / RL。

## 2. 当前可用数据

训练读取 `data/processed/*_train.jsonl`，验证读取 `data/processed/*_val.jsonl`。

主训练字段已经 C-centered：

```text
abc_melody
key
chords
harmony
harmony_tokens
```

原调备份字段：

```text
original_key
normalization.transpose_semitones
original
```

当前本地和远程都已验证：

```text
processed rows: 2075
datasets: EMOPIA+ 879, HLSD 11, POP909 1085, OpenBook 100
dataset weights: EMOPIA+/HLSD 1.0, POP909 0.7, OpenBook 0.0
training keys: C:maj 1323, C:min 752
harmony token parse errors: 0
```

OpenBook 默认不进训练，因为 `dataset_weight=0.0`。

## 3. Tokenizer 与新增 Token

现有 55 个 harmony token 继续保留：

```text
<MELODY> </MELODY>
<HARMONY> </HARMONY>
<SPAN> </SPAN>
<R_...>
<Q_...>
<EXT_...>
<ALT_...>
<B_...>
<OMIT_...>
```

Stage 1 masked completion 还需要新增控制 token：

```text
<HARMONY_CONTEXT>
</HARMONY_CONTEXT>
<ROOT_PREDICTION>
</ROOT_PREDICTION>
<QUALITY_PREDICTION>
</QUALITY_PREDICTION>
<CHORD_PREDICTION>
</CHORD_PREDICTION>
<MASK_ROOT>
<MASK_QUALITY>
<MASK_CHORD>
```

这些 token 应作为 regular added tokens 加入 tokenizer。扩词表后必须保存
tokenizer，并检查每个新增 token 都是 single token。

Qwen3-1.7B-Base 的 tokenizer / model matrix 状态已经验证：

```text
base tokenizer length: 151669
added harmony tokens: 55
current tokenizer length after 55 tokens: 151724
model input embedding rows: 151936
model LM head rows: 151936
```

注意：该 checkpoint 的 embedding / LM head 行数大于 tokenizer 长度。resize 逻辑
必须“只扩不缩”，否则会错误丢掉 checkpoint 里已有的 padded vocab rows。

## 4. 通用样本结构与 Mask

所有 Stage 1 样本都由三段组成：

```text
<MELODY>
... C-centered ABC ...
</MELODY>

<HARMONY_CONTEXT>
... masked or visible gold harmony spans ...
</HARMONY_CONTEXT>

<PREDICTION_BLOCK>
... answer tokens ...
</PREDICTION_BLOCK>
```

训练拼接：

```text
prompt_ids + answer_ids + eos
```

label mask：

```text
prompt labels: -100
answer labels: answer token ids
padding labels: -100
```

也就是说，ABC、context、wrapper、提示 token 都不直接计 loss；loss 只落在答案
区域，例如 `<R_G>` 或 `<R_G> <Q_DOM7>`。

attention mask 独立处理：

```text
real tokens: 1
padding tokens: 0
```

## 5. Stage 1A：Root Masked Completion

### 5.1 目标

在 gold span 边界、visible quality 和部分 chord context 条件下恢复 root。

样本形式：

```text
<MELODY>
... ABC ...
</MELODY>

<HARMONY_CONTEXT>
<SPAN> @0-4  <R_D>       <Q_MIN7> </SPAN>
<SPAN> @4-8  <MASK_ROOT> <Q_DOM7> </SPAN>
<SPAN> @8-12 <R_C>       <Q_MAJ7> </SPAN>
</HARMONY_CONTEXT>

<ROOT_PREDICTION>
<R_G>
</ROOT_PREDICTION>
```

### 5.2 Loss

设 \(X\) 是 ABC melody，\(B\) 是 gold span 边界，\(Q\) 是 visible quality，
\(R_{\neg i}\) 是未遮挡 root，\(\mathcal{M}_R\) 是 masked root 集合：

$$
\mathcal{L}_{R}
=
-\frac{1}{|\mathcal{M}_R|}
\sum_{i\in\mathcal{M}_R}
\log p_\theta(r_i\mid X,B,Q,R_{\neg i})
$$

工程实现仍是 answer 区间 token-level cross entropy。若一个样本只要求预测一个
root，答案区间通常只有一个 root token 加 eos。

### 5.3 Masking

| Mask 类型 | 占比 | 说明 |
|---|---:|---|
| 单 root，左右 root 都可见 | 40% | 最稳定的基础 root recovery。 |
| 单 root，只保留一侧上下文 | 20% | 减少对双侧答案泄漏的依赖。 |
| 连续遮 2 个 root | 20% | 学局部进行，不只补单点。 |
| phrase 内遮 40%-60% roots | 10% | 模拟更弱 harmony context。 |
| 真实 251-like 片段中遮一个 root | 10% | 强化常见局部进行，但不做 hard negative。 |

### 5.4 冻结策略

| 模块 | 策略 |
|---|---|
| 新 token embedding / LM head 行 | 训练 |
| 原 token embedding / LM head 行 | 冻结或极低 LR |
| Transformer 前 70% 层 | 冻结 |
| Transformer 后 30% 层 | 训练 |
| Final RMSNorm | 训练 |
| 新增分类 head | 无 |

## 6. Stage 1B：Quality Masked Completion（可选短阶段）

### 6.1 目标

在 root 已知时判断 quality。只有当 Stage 1A 后 quality confusion 明显高，才需要
单独跑这个短阶段；否则可以直接进入 Stage 1C。

样本形式：

```text
<MELODY>
... ABC ...
</MELODY>

<HARMONY_CONTEXT>
<SPAN> @0-4  <R_D> <Q_MIN7>       </SPAN>
<SPAN> @4-8  <R_G> <MASK_QUALITY> </SPAN>
<SPAN> @8-12 <R_C> <Q_MAJ7>       </SPAN>
</HARMONY_CONTEXT>

<QUALITY_PREDICTION>
<Q_DOM7>
</QUALITY_PREDICTION>
```

### 6.2 Loss

设 \(\mathcal{M}_Q\) 是 masked quality 集合：

$$
\mathcal{L}_{Q}
=
-\frac{1}{|\mathcal{M}_Q|}
\sum_{i\in\mathcal{M}_Q}
\log p_\theta(q_i\mid X,B,R,Q_{\neg i})
$$

### 6.3 Masking

| Mask 类型 | 占比 | 说明 |
|---|---:|---|
| 单 quality mask | 60% | 基础质量判断。 |
| 连续两 quality mask | 20% | 防止只靠相邻标签补全。 |
| 同 root 的 quality 区分样本过采样 | 15% | 强化 `C:maj` / `C:maj7` / `C:dom7` 等混淆点。 |
| 长尾 quality 过采样 | 5% | 避免 dim/hdim7/aug/sus 被完全压低。 |

### 6.4 冻结策略

| 模块 | 策略 |
|---|---|
| 新 token embedding / LM head 行 | 训练 |
| 原 token embedding / LM head 行 | 冻结或极低 LR |
| Transformer 前 60% 层 | 冻结 |
| Transformer 后 40% 层 | 训练 |
| Final RMSNorm | 训练 |
| 新增分类 head | 无 |

## 7. Stage 1C：Joint Chord Masked Completion

### 7.1 目标

同时恢复 root 和 quality，缩小 Stage 1 masked completion 与 Stage 2 full
generation 的任务落差。

样本形式：

```text
<MELODY>
... ABC ...
</MELODY>

<HARMONY_CONTEXT>
<SPAN> @0-4  <R_D> <Q_MIN7> </SPAN>
<SPAN> @4-8  <MASK_CHORD>   </SPAN>
<SPAN> @8-12 <R_C> <Q_MAJ7> </SPAN>
</HARMONY_CONTEXT>

<CHORD_PREDICTION>
<R_G> <Q_DOM7>
</CHORD_PREDICTION>
```

### 7.2 Loss

设 \(\mathcal{M}_{RQ}\) 是 masked chord 集合：

$$
\mathcal{L}_{RQ}
=
-\sum_{i\in\mathcal{M}_{RQ}}
\left[
\log p_\theta(r_i\mid\cdot)
+
\log p_\theta(q_i\mid X,B,r_i,\cdot)
\right]
$$

实现上继续使用答案区间 token-level cross entropy。答案序列固定为：

```text
<R_...> <Q_...>
```

不把 `root:quality` 展成完整 chord label token。

### 7.3 Masking

| Mask 类型 | 占比 | 说明 |
|---|---:|---|
| 单完整 chord mask | 50% | 核心 joint chord completion。 |
| 相邻两 chord mask | 25% | 学局部进行与连续 harmonic rhythm。 |
| phrase 中遮 40% chord | 15% | 减少对密集真值 context 的依赖。 |
| 全部 chord mask | 10% | 向 Stage 2 的 ABC-only generation 过渡。 |

### 7.4 冻结策略

| 模块 | 策略 |
|---|---|
| 新 token embedding / LM head 行 | 训练 |
| 原 token embedding / LM head 行 | 低 LR 或部分放开 |
| Transformer 前 50% 层 | 冻结 |
| Transformer 后 50% 层 | 训练 |
| Final RMSNorm | 训练 |
| 新增分类 head | 无 |

## 8. Stage 1 数据集加权

不同数据集的样本损失先在 answer token 上平均，再按数据集权重做 batch 聚合。

对样本 \(j\)，其 answer token 集合为 \(A_j\)：

$$
\mathcal{L}^{(j)}
=
\frac{1}{|A_j|}
\sum_{t\in A_j}
-\log p_\theta(y_t\mid y_{<t},X,C)
$$

其中 \(C\) 是 harmony context。

batch loss：

$$
\mathcal{L}_{batch}
=
\frac{\sum_j w_{d_j}\mathcal{L}^{(j)}}{\sum_j w_{d_j}}
$$

初始权重：

| Dataset | Weight | Stage 1 使用方式 |
|---|---:|---|
| EMOPIA+ | 1.0 | 主训练源 |
| HLSD | 1.0 | 主训练源，但当前本地只有 sample 规模 |
| POP909 | 0.7 | 使用，但降低噪声数据权重 |
| OpenBook | 0.0 | audit only，默认跳过 |

## 9. Stage 1 超参数建议

Stage 1 是短适配阶段，目标不是长时间重训全模型，而是让模型先掌握 harmony
token、root/quality 条件补全和局部进行。

| 参数 | Stage 1A | Stage 1B（可选） | Stage 1C |
|---|---:|---:|---:|
| max_length | 4096 | 4096 | 4096 |
| per_device_train_batch_size | 1 | 1 | 1 |
| gradient_accumulation_steps | 16 | 16 | 16 |
| effective batch size | 16 | 16 | 16 |
| epochs | 1 | 0.5-1 | 1 |
| learning_rate 后层 | 1e-5 | 1e-5 | 1e-5 |
| learning_rate 新 token 行 | 5e-5 | 5e-5 | 5e-5 |
| learning_rate 原 embedding | 0 或 1e-6 | 0 或 1e-6 | 1e-6 |
| weight_decay | 0.01 | 0.01 | 0.01 |
| warmup_ratio | 0.03 | 0.03 | 0.03 |
| precision | bf16 | bf16 | bf16 |
| gradient_checkpointing | true | true | true |
| optimizer | adamw_torch | adamw_torch | adamw_torch |

`max_length=4096` 是当前更合理的起点。tokenizer dry check 显示：

```text
max_length 2048: train kept 925, skipped 672 over length
max_length 3072: train kept 1213, skipped 384 over length
max_length 4096: train kept 1513, skipped 84 over length
max_length 6144: train kept 1594, skipped 3 over length
```

如果 24 GB 4090 上 full-parameter AdamW 仍然 OOM，则先降到 `3072` 或实现 chunking，
不要无提示地训练 2048 子集。

## 10. Stage 1 验收指标

Stage 1A：

```text
root accuracy
root accuracy by dataset
root accuracy by masked pattern
root confusion by interval distance
```

Stage 1B：

```text
quality accuracy
quality confusion matrix
major/minor/dom7/maj7/min7 confusion
long-tail quality recall
```

Stage 1C：

```text
root accuracy
quality accuracy
root+quality joint accuracy
joint accuracy by masked pattern
251-like masked chord recovery accuracy
```

所有阶段都要记录：

```text
parse success rate
answer token exact match
loss by dataset weight bucket
over_max_length skipped count
```

## 11. 当前还需要实现的工程任务

在开始 Stage 1A 训练前，需要补以下代码：

```text
masked-completion control tokens
Stage1MaskedCompletionDataset
root / quality / chord masking sampler
progression-aware masking index
parameter freezing and optimizer group utilities
Stage 1 eval script
```

现有 `src.train.sft_stage1` 是 ABC-only full harmony generation baseline，更接近
Stage 2。按照当前计划，它不应作为新的第一步直接启动。

## 12. 不做的事情

Stage 1 明确不做：

```text
LoRA / Adapter / Prefix Tuning
DPO / GRPO / RL
note-role supervision
passing-tone classification
independent change-point head
extension / alteration / bass 主训练
voicing generation
accompaniment pattern retrieval
```
