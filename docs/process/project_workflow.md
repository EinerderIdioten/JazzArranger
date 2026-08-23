# 项目工作流总规范

日期：2026-08-24  
适用范围：当前 jazzarranger 项目全局  
目标：把 research、planning、executing、reviewing 固定成一条可重复的流水线，避免任务、数据、实验和结论互相污染。

## 1. 总原则

1. 一次只推进一个主目标。
2. 每个实验必须有冻结清单 `manifest`。
3. 每次实现必须对应一个可追踪 plan。
4. 每次运行必须产出可读的 summary。
5. 每次 review 必须对照 plan、diff 和运行结果。
6. 研究结论、实现结论、审阅结论不能混写。

## 2. 角色分工

| 角色 | 职责 | 输出 |
|---|---|---|
| Perplexity | 做 research，找论文、demo、数据集、先例和反例 | research brief，带来源与时间戳 |
| Codex | 做 planning 和 executing，把想法落成代码、文档、manifest、脚本 | plan，diff，run summary，commit |
| Hermes | 做 review gate，检查是否按计划执行，是否越界，是否需要返工 | review verdict，issue list，next questions |
| Human | 定方向、拍板、处理 tradeoff | 目标优先级和最终决策 |

说明：这里把 Hermes 视为第三方审阅代理。若你实际接入的是别的 review agent，只要它执行同样的输入输出协议即可。

## 3. 每日循环

### 3.1 Research

输入：

- 当前要解决的问题
- 过去一轮的失败点
- 需要验证的论文、demo、数据集、方法

要求：

- 只回答“事实和证据是什么”
- 不下最终实现结论
- 每条关键结论都要有来源
- 不确定的地方明确标注

建议落盘：

- `docs/research/YYYY-MM-DD_topic.md`

### 3.2 Planning

输入：

- research brief
- 当前仓库状态
- 现有 manifest / plan / eval 资产

要求：

- 把研究结论翻译成可执行任务
- 明确范围、非目标、验收标准
- 明确需要改的文件和不需要碰的文件
- 明确是否要新建或更新 manifest

建议落盘：

- `docs/train/YYYY-MM-DD_topic_plan.md`
- 或 `docs/process/YYYY-MM-DD_topic_plan.md`

### 3.3 Executing

输入：

- plan
- 当前 manifest
- 代码和数据现状

要求：

- 只做 plan 里列出的改动
- 先补最小闭环，再扩大
- 每次改动后跑语法检查或小型验证
- 运行结果必须写 summary

建议落盘：

- `src/` 代码
- `manifests/` 冻结清单
- `evals/` 固定样本
- `outputs/` 运行产物

### 3.4 Reviewing

输入：

- plan
- git diff
- run summary
- 必要时再看少量样本输出

要求：

- 只判断三件事：做没做对、有没有越界、还差什么
- review 先看失败点，再看整体评价
- review 结果要能驱动下一轮 research 或 planning

建议落盘：

- `docs/reviews/YYYY-MM-DD_topic_review.md`

## 4. 交付物顺序

每个正常循环都应按这个顺序结束：

```text
research brief
-> plan
-> code / doc / manifest change
-> run summary
-> review note
-> next plan
```

如果中途出现新发现，不要直接改结论，先把它放到下一轮 research。

## 5. 仓库职责边界

| 目录 | 职责 |
|---|---|
| `src/` | 生产代码 |
| `docs/train/` | 训练与验证计划 |
| `docs/data/` | 数据转换与口径说明 |
| `docs/process/` | 工作流规范、模板、操作协议 |
| `manifests/` | 冻结实验合同 |
| `evals/` | 固定评测样本、song list、人工标注输入 |
| `outputs/` | 运行产物、summary、检查结果 |
| `log/` | 转换和统计日志 |
| `scripts/` | 一次性工具 |
| `deprecated/` | 归档，只读 |

## 6. 防失控规则

1. 不允许没有 manifest 的实验。
2. 不允许没有 plan 的代码改动。
3. 不允许没有 summary 的长跑实验。
4. 不允许没有 review 的结论冻结。
5. 不允许把 research 结论直接当实现事实。
6. 不允许把 review 问题混进代码逻辑里不记录。
7. 不允许把临时脚本当长期接口。

## 7. 当前项目的三条主线

### 7.1 数据与训练

目标是稳定 melody-to-harmony / root completion / future generation 的数据合同。

当前对应资产：

- `docs/data/data_conversion.md`
- `docs/train/training_overall_plan.md`
- `docs/train/training_plan_v4.md`
- `manifests/root_simple_v1.json`

### 7.2 Validator

目标是把第三方和声验证器变成稳定的外部 oracle，而不是临时评分器。

当前对应资产：

- `docs/train/2026-08-20_third_party_harmony_validator_plan.md`
- `src/validator/`

### 7.3 研究输入

目标是把 Perplexity 的 research 结果变成可执行计划，而不是散落聊天记录。

建议做法：

- 每轮 research 都写清楚问题、来源、结论、反例、待验证点
- 只把经过筛选的结论推进到 planning

## 8. 推荐模板

### Research brief

```md
# Research Brief

- Question:
- Sources:
- Key findings:
- Contradictions:
- Open questions:
- Recommendation:
```

### Plan

```md
# Plan

- Goal:
- Scope:
- Non-goals:
- Inputs:
- Outputs:
- Acceptance criteria:
- Files to change:
- Risks:
```

### Review

```md
# Review

- Plan checked:
- Diff checked:
- Result:
- Issues:
- Follow-up:
```

## 9. 本项目默认节奏

1. 上午先做 research。
2. 中午把 research 收敛成 plan。
3. 下午用 Codex 执行最小改动。
4. 晚上让 Hermes 做 review。
5. 第二天只带着 review 里未解决的问题继续。

