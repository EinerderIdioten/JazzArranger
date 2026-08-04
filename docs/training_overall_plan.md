# 总训练计划：ABC Melody-To-Harmony Labeling

## 1. 设计原则

本项目当前阶段先把 **melody-to-harmony labeling** 做稳：输入标准化后的
ABC 单旋律，输出与时间严格对齐的 chord span 序列。当前不做
accompaniment pattern、voicing、DPO、GRPO 或 RL。

能力设计遵循一条原则：

**有稳定金标、且对最终任务有直接贡献的能力，适合做显式 stage；缺少跨
数据集稳定标注、或者本质依赖最终和弦解释的能力，应该作为隐变量通过主任务
学习。**

功能和声、分段与和弦识别是强耦合问题。当前路线避免过早拆成
`边界 -> note role -> root -> quality` 的硬流水线，而是用短 curriculum
先建立 root / quality / chord identity，再过渡到最终的 ABC-only span 生成。

## 2. 能力分层

| 能力 | 是否有可靠监督 | 是否单独 stage | 是否单独 head | 建议位置 | 说明 |
|---|---|---:|---:|---|---|
| Root | 是，来自 chord label | 是 | 否，前期不用 | Stage 1A | 先学 root recovery，建立与旋律、quality、局部进行的关系。 |
| Quality | 是，来自 chord label | 可选，是 | 否，前期不用 | Stage 1B 或并入 1C | 若资源有限，可跳过独立 stage，直接进入联合 chord completion。 |
| Root+Quality 联合恢复 | 是 | 是 | 否 | Stage 1C | 让模型从“条件 root/quality 判断”过渡到完整 chord identity 恢复。 |
| 和弦何时变化 | 是，来自 onset/offset | 否，先不独立成主 stage | 可选，是 | Stage 3 辅助任务 | 若生成结果边界明显不稳，再加 boundary head。 |
| Span 时长 | 是 | 否 | 否 | 隐含在 Stage 2/3 | 通过 `@start-end` 与 span 输出共同学习。 |
| 251 / 局部进行 | 无需额外标注，但可从真值序列挖掘 | 否 | 否 | 贯穿 Stage 1A-2 的采样策略 | 不单独做 loss，用 progression-aware masking 与转调归一化来学。 |
| Key / local key | 只有部分数据可靠 | 否，当前版本不做 | 否，可后期加 | 未来版本 | 先不要伪造 key-function 标签。 |
| Note role | 通常无统一金标 | 否 | 否 | 隐变量 | 不做显式监督，靠最终和声目标学习。 |
| Passing / neighbor / suspension | 通常无统一金标 | 否 | 否 | 隐变量 | 本质依赖最终和弦解释，不适合前置硬标注。 |
| 音的重要性 / support weight | 无直接金标 | 否 | 否 | 隐变量 | 可通过 probing 诊断，不建议训练头。 |
| Cadence / phrase ending | 部分数据才有 | 否 | 否 | 未来版本 | 当前不是关键路径。 |
| Extension / alteration / bass | 部分有标签，但长尾且格式不统一 | 否，当前版本不做主训练 | 否，未来可做 | 未来版本 | 先 canonicalize 保存原值，但不进入当前主目标。 |

## 3. 训练顺序总览

推荐顺序如下：

1. **Stage 0：数据统一与表示审计**。先确认 ABC、grid、span、root、quality、C 调归一化都可靠，否则后续训练没有意义。
2. **Stage 1A：Root Masked Completion**。先让模型在已知 quality、span 和部分 chord context 时恢复 root，建立最核心的局部和声关系。
3. **Stage 1C：Joint Chord Masked Completion**。不建议把独立 Quality stage 做得太重；更好的做法是尽快进入 root+quality 联合恢复，减少训练-推理落差。
4. **Stage 2：ABC-only Harmony Generation**。这是当前项目真正的主任务：不给任何真值和声上下文，只给 ABC，让模型输出完整 span 序列。
5. **Stage 3：Boundary Auxiliary（可选）**。只有当 root/quality 已经基本正确、但“何时换和弦”持续不稳时，再加入轻量 boundary head。
6. **Stage 4：Span-Root-Quality Pair Scorer（可选）**。当需要更强的 span 表征质量、置信度或 reranking 时再加，不作为第一版主线。

最小可行路线：

```text
Stage 0 -> Stage 1A -> Stage 1C -> Stage 2 -> Stage 3（必要时）
```

扩展路线：

```text
Stage 0 -> Stage 1A -> Stage 1B -> Stage 1C -> Stage 2 -> Stage 3（可选）-> Stage 4（可选）
```

## 4. 为什么这个顺序最稳

先学 root 是合理的，因为 root 与旋律低层音高重心、局部进行和功能落点的关系
最直接。但如果长期只学 root，模型会过度依赖训练时给它的真值 quality 和相邻
chord，因此必须尽快进入联合 chord completion，再过渡到完全 ABC-only 生成。

“什么时候换和弦”不应在一开始就独立出来，因为边界错误会把后续和弦识别全部
带偏。分段与 chord labeling 更适合作为联合建模，或者作为后期辅助约束，而不是
早期硬流水线。

passing tone、support tone、note role 这些概念重要，但当前缺少跨数据集稳定
金标，更适合作为模型内部隐变量。后续如果模型明显被经过音或短时弱拍音误导，
再通过 probing 或轻量辅助任务诊断，而不是现在加硬标签。

## 5. 当前数据状态

当前处理后的训练字段已经做 C tonic 归一化：

```text
abc_melody
key
chords
harmony
harmony_tokens
```

原调版本保留在：

```text
original_key
normalization.transpose_semitones
original
```

若原调 tonic 的 pitch-class index 为 \(k\)，规约到 C 的平移量为：

$$
\tau = -k \pmod{12}
$$

对 melody pitch \(p\) 和 chord root pitch-class \(r\)，同步平移：

$$
p' = p + \tau
$$

$$
r' = (r + \tau) \bmod 12
$$

span 边界和 quality 不变。

当前验证结果：

```text
processed rows: 2075
EMOPIA+: 879, weight 1.0
HLSD: 11, weight 1.0
POP909: 1085, weight 0.7
OpenBook: 100, weight 0.0 audit only
training keys: C:maj 1323, C:min 752
harmony token parse errors: 0
```

## 6. 总体 Loss 设计

数据集加权采用：

$$
\mathcal{L}_{batch}
=
\frac{\sum_j w_{d_j}\mathcal{L}^{(j)}}{\sum_j w_{d_j}}
$$

其中 \(w_{d_j}\) 是样本 \(j\) 所属数据集的权重。

### Stage 1A：Root Loss

给定 ABC melody \(X\)、gold span 边界 \(B\)、visible quality \(Q\)、未遮挡 root
\(R_{\neg i}\)，对 masked root 集合 \(\mathcal{M}_R\)：

$$
\mathcal{L}_{R}
=
-\frac{1}{|\mathcal{M}_R|}
\sum_{i\in\mathcal{M}_R}
\log p_\theta(r_i\mid X,B,Q,R_{\neg i})
$$

### Stage 1B：Quality Loss（可选）

给定 \(X\)、\(B\)、root \(R\)、未遮挡 quality \(Q_{\neg i}\)，对 masked quality
集合 \(\mathcal{M}_Q\)：

$$
\mathcal{L}_{Q}
=
-\frac{1}{|\mathcal{M}_Q|}
\sum_{i\in\mathcal{M}_Q}
\log p_\theta(q_i\mid X,B,R,Q_{\neg i})
$$

### Stage 1C：Joint Root-Quality Loss

对 masked chord 集合 \(\mathcal{M}_{RQ}\)：

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

工程实现仍是 answer 区间的 token-level cross entropy：

```text
<R_...> <Q_...>
```

### Stage 2：ABC-Only Generation Loss

给定完整 harmony target \(Y=(y_1,\dots,y_T)\)：

$$
\mathcal{L}_{gen}
=
-\sum_{t\in\mathrm{harmony}}
\log p_\theta(y_t\mid X,y_{<t})
$$

ABC prompt 区间全部设为 `labels=-100`，只在 harmony 输出区间计 loss。

### Stage 3：Boundary Auxiliary Loss（可选）

令 \(b_t\in\{0,1\}\) 表示 grid \(t\) 是否为 chord onset，\(p_t\) 为 boundary
head 输出概率：

$$
\mathcal{L}_{boundary}
=
-\sum_t
\left[
w_1 b_t\log p_t
+
w_0(1-b_t)\log(1-p_t)
\right]
$$

总损失：

$$
\mathcal{L}_{stage3}
=
\mathcal{L}_{gen}
+
\lambda_B\mathcal{L}_{boundary}
$$

建议起始：

$$
\lambda_B = 0.1
$$

### Stage 4：Span-Root-Quality Pair Scorer（可选）

对 span 表征 \(h_i\) 和候选 \((r,q)\)：

$$
S(r,q\mid h_i)
=
a(r\mid h_i)
+
b(q\mid h_i)
+
c(r,q,h_i)
$$

分类损失：

$$
\mathcal{L}_{pair}
=
-\sum_i
\log
\frac{\exp S(r_i,q_i\mid h_i)}
{\sum_{r'}\sum_{q'}\exp S(r',q'\mid h_i)}
$$

如果和前面目标联合：

$$
\mathcal{L}_{stage4}
=
\mathcal{L}_{gen}
+
\lambda_B\mathcal{L}_{boundary}
+
\lambda_P\mathcal{L}_{pair}
$$

建议：

$$
\lambda_P \in [0.1, 0.3]
$$

## 7. 为什么不用 LoRA / Adapter

当前任务不仅要学习新 token（`<R_*>`、`<Q_*>`、`<SPAN>` 等）的输入输出
语义，还要让 backbone 重新组织 ABC 文本、和声上下文与 span 结构之间的关系。
低秩 adapter 未必适合这种“新输出语言 + 新结构任务”的早期建模。

因此推荐策略是：**不用 LoRA，不加 adapter，全程原模型参数训练，但采用逐步解冻
和分组学习率。**

## 8. 参数冻结总原则

| 阶段          | 原 token embedding | 新 token embedding | Transformer 前层 | Transformer 后层 | 新增 head          |
| ----------- | ----------------- | ----------------- | -------------- | -------------- | ---------------- |
| Stage 1A    | 冻结或只保留极低 LR       | 训练                | 冻结前 70%        | 训练后 30%        | 无                |
| Stage 1B    | 冻结或极低 LR          | 训练                | 冻结前 60%        | 训练后 40%        | 无                |
| Stage 1C    | 低 LR 或部分放开        | 训练                | 冻结前 50%        | 训练后 50%        | 无                |
| Stage 2 前期  | 低 LR              | 训练                | 冻结前 50%        | 训练后 50%        | 无                |
| Stage 2 中后期 | 训练                | 训练                | 全部训练           | 全部训练           | 无                |
| Stage 3     | 训练                | 训练                | 全部训练           | 全部训练           | boundary head 训练 |
| Stage 4     | 训练                | 训练                | 全部训练           | 全部训练           | pair scorer 训练   |

如果想严格保持旧词表 embedding 不动，可对 input embedding 和 LM head 的旧
vocab 行加 gradient mask，只允许新增 token 行更新。工程上想先简单可靠，则允许
整个 embedding 训练，但给 embedding 参数组设置更低学习率。

## 9. 当前工程含义

当前仓库已经具备：

```text
dataset conversion
C tonic normalization
tokenized harmony target
ABC-only causal LM SFT baseline
generation-time evaluation
```

在真正执行 Stage 1A / 1C 前，还需要补：

```text
masked-completion sample builder
masked-completion control tokens
stage-specific dataset mode
parameter freezing utilities
progression-aware masking index
```

现有 `src.train.sft_stage1` 是 ABC-only full generation baseline。按这个总计划，
它更接近 Stage 2，不应该作为新的第一步直接训练。
