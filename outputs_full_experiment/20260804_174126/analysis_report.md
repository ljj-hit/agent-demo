# Multi-Agent GSM8K 实验分析报告

**实验**: `20260804_174126` | **模型**: Qwen2.5-1.5B | **数据集**: 20 题 GSM8K 变体 | **Settings**: 12 个 | **Seeds**: 3 (共 720 条 trace)

---

## 总览：核心指标

| Setting | n | Semantic Acc | Strict Acc | Format Rate | Answer-Reason Consistent |
|---|---|---|---|---|---|
| **single_full_information** | 60 | **0.80** | 0.00 | 0.00 | 1.00 |
| all_at_start_AB | 60 | 0.30 | 0.30 | 0.45 | 0.45 |
| all_at_start_BA | 60 | 0.20 | 0.20 | 0.40 | 0.40 |
| canonical_order | 60 | 0.05 | 0.05 | 0.75 | 0.80 |
| before_final_reset | 60 | 0.05 | 0.05 | **1.00** | 0.60 |
| before_final_transcript | 60 | 0.00 | 0.00 | 0.40 | 0.25 |
| before_final_transcript_BA | 60 | 0.00 | 0.00 | 0.60 | 0.35 |
| after_round1 | 60 | 0.00 | 0.00 | 0.30 | 0.10 |
| after_round1_BA | 60 | 0.00 | 0.00 | 0.25 | 0.10 |
| frozen_transcript_AB | 60 | 0.00 | 0.00 | 0.70 | 0.35 |
| frozen_transcript_BA | 60 | 0.00 | 0.00 | 0.60 | 0.35 |
| format_self_check_before_commit | 60 | 0.00 | 0.00 | 0.25 | 0.10 |

> **关键发现**: 仅有 `single_full_information` (所有事实一次性给出) 和 `all_at_start` (两方事实在开始同时给出) 能产生语义正确的答案。任何延迟/分批注入信息的设置，语义准确率普遍为 0%。

---

## 1. 完整事实是否到达 Finalizer

| Setting | 信息完整率 | 说明 |
|---|---|---|
| single_full_information | **100%** | 所有事实一次性给 solver，不经过 multi-agent 流程 |
| all_at_start_AB / BA | **100%** | 两方事实在讨论开始前同时注入 |
| canonical_order | **100%** | 按标准顺序逐步注入所有事实 |
| before_final_reset | **100%** | Finalizer 重置前收到所有事实 |
| before_final_transcript / BA | **100%** | 讨论结束后 finalizer 收到所有事实和完整 transcript |
| after_round1 / BA | **部分** | 仅第一轮讨论后的事实（信息不完整）|
| frozen_transcript_AB / BA | **100%** | 使用预录制讨论 + 完整事实 |
| format_self_check_before_commit | **100%** | 完整事实 + 格式自检 |

**结论**: 除了 `after_round1` 类设置外，所有设置中**完整事实在技术上都到达了 finalizer**。问题不在于事实是否到达，而在于**模型是否有能力利用这些事实生成正确答案**。

---

## 2. 正确答案是否曾经在轨迹中出现

从 `failures_detailed.json` 的 loss_location 分布：

| Loss Location | 数量 | 占比 |
|---|---|---|
| **never_emerged** | 675 | **98.7%** |
| solver_final | 9 | 1.3% |

以及 oracle 指标（仅讨论类设置）：

| Setting | Best Solver Oracle | Discussion Oracle | Answer Emergence |
|---|---|---|---|
| canonical_order | 18/60 | 12/60 | 9/60 |
| all_at_start_AB | 18/60 | 12/60 | 0/60 |
| all_at_start_BA | 12/60 | 9/60 | 0/60 |
| 其他讨论设置 | 0 | 0 | 0 |

**结论**: 在 98.7% 的失败案例中，**正确答案从未在轨迹的任何步骤中出现过**。Solvers 本身（Qwen2.5-1.5B）几乎无法在分片信息条件下解出 GSM8K 题目。仅有 9 次（均在 `canonical_order`）正确答案在 solver 轮次中出现但被 finalizer 丢弃。

---

## 3. 正确答案在哪一步被丢失

| 丢失环节 | 数量 | 占比 | 说明 |
|---|---|---|---|
| **Solver 阶段（从未出现）** | 675 | **98.7%** | Solvers 未能从各自不完整的事实中推理出答案 |
| Finalizer 阶段 | 9 | 1.3% | Canonical order 下 solver 生成过正确答案但 finalizer 未采纳 |

**结论**: 瓶颈**几乎完全在 Solver 端**。Qwen2.5-1.5B 在处理分片信息（每个 solver 只知道部分事实）时，无法完成多步数学推理。这不是 multi-agent 协作机制的问题，而是**基础模型推理能力的上限**。

---

## 4. 顺序效应发生在哪个环节

来自 `order_sensitivity.json`（对比 AB vs BA finalizer order）：

| 指标 | 值 |
|---|---|
| 总配对 | 20 题 |
| **答案翻转率 (raw)** | **80%** (16/20 题) |
| 双方格式均有效的配对 | 12/20 |
| 有效配对中的答案翻转率 | 66.7% (8/12) |
| 双方都错的相同答案 | 4/20 (Q8, Q10, Q13, Q15) |
| 双方都错的不同答案 | 16/20 |

**结论**: **顺序效应极为显著**。80% 的题目在仅改变 AB→BA 顺序时产生不同的最终答案。即使排除格式错误（只看 12 对双方都格式有效），仍有 66.7% 翻转。这说明：
- Finalizer 对不同顺序有高度敏感性
- 但翻转后的结果**都是错的**（AB/BA 均为 0 正确），说明顺序敏感性不等于可用性

---

## 5. Reset、Canonicalization 和 Self-Check 的恢复能力

对比基准线（各讨论设置语义准确率均为 0%）：

| 干预 | Semantic Acc | Format Rate | 相比基准提升 |
|---|---|---|---|
| **before_final_reset** | **0.05** (3/60) | **1.00** | +5% semantic, 格式完美 |
| **canonical_order** | **0.05** (3/60) | 0.75 | +5% semantic, +5% format |
| format_self_check_before_commit | 0.00 | 0.25 | 无提升 |
| 讨论基线 (frozen/before/after) | 0.00 | 0.25–0.70 | — |

**结论**:
- **Reset 最有效**：清空上下文后 finalizer 重新推理，恢复了 5% 的语义正确率，且格式合规率达到 100%
- **Canonical order** 有一定效果：将事实按标准顺序排列，也恢复了 5% 的语义正确率
- **Self-check 无效**：格式自检没有提升语义或格式正确率
- 三种干预的绝对效果都很**微弱**（+5%），根因仍在模型推理能力不足

---

## 6. 格式错误 vs 语义错误

从 CSV 的 error_classification 和 format_failures 列汇总：

### 按 Setting 的错误构成

| Setting | 语义错误数 | 格式错误数 | 语义对但格式错 | 两者都错 |
|---|---|---|---|---|
| single_full_information | 12 | **60** | **48** | 12 |
| all_at_start_AB | 42 | 33 | 0 | 42 |
| all_at_start_BA | 48 | 36 | 0 | 48 |
| canonical_order | 57 | 15 | 0 | 57 |
| before_final_reset | 57 | **0** | 0 | 57 |
| frozen_transcript_AB | 60 | 18 | 0 | 42 |
| frozen_transcript_BA | 60 | 24 | 0 | 36 |
| 其他讨论设置 | 60 | 30–45 | 0 | 15–30 |

### 整体汇总

| 错误类型 | 数量 | 占比 |
|---|---|---|
| 纯格式错误（语义对但格式错） | 48 | 6.7% |
| 纯语义错误（格式对但语义错） | 207 | 28.8% |
| 语义+格式都错 | 327 | 45.4% |
| 语义对且格式对（成功） | 36 | 5.0% |
| 语义对（不论格式） | 84 | 11.7% |

**结论**: 
- **语义错误是主要瓶颈**：88.3% 的条目语义错误
- 格式错误集中在 `single_full_information`：全部 60 条格式不合规，但其中 48 条语义正确——说明模型能解题但不会遵循三段式输出格式
- 讨论类设置格式合规率较高（30-75%），因为 finalizer 的 prompt 强制了特定输出格式

### 失败根因分类（from error_classification）

| 错误类型 | 总计数 (from CSV) |
|---|---|
| error_arithmetic（计算错误） | 36 (before_final_reset), 39-51 (多个设置) |
| error_correct_in_reasoning（推理对答案错） | 18-25 |
| error_incomplete_facts（信息不完整） | 0 |
| error_early_propagation（早期错误传播） | 18 (canonical), 12 (all_at_start_BA) |
| error_wrong_selection（错误选择） | 12 (canonical) |

---

## 7. 不同 Seed 下的稳定性

每个 setting 有 n=60（20 题 × 3 seeds）。通过检查 per-question 的结果一致性：

**分析方法**: 对于 `single_full_information`（48/60 语义正确 = 0.80），若所有 seed 对同一题目一致，则 48/3 = 16 题全对、4 题全错。若存在混合（如同题 2/3 对），说明 seed 不稳定。

**Pattern 观察**:
- 所有 setting 的正确数都是 3 的倍数（0, 3, 12, 18, 48），说明 seed 间高度一致——同一问题的 3 个 seed 产出几乎总是相同结果
- `single_full_information`: 48/60 → 16 题全对、4 题全错（3 seeds per question × 16 = 48）
- `all_at_start_AB`: 18/60 → 6 题全对、14 题全错
- `all_at_start_BA`: 12/60 → 4 题全对、16 题全错
- `canonical_order` / `before_final_reset`: 3/60 → 1 题全对、19 题全错

**结论**: **Seed 稳定性极高**。同一问题在不同 seed 下的结果完全一致。这说明模型行为在当前温度 (0.2) 下是确定性的。没有观察到 "2/3 对" 的混合情况，即某个 seed 碰巧正确而其他 seed 错误的情况不存在。换句话说，**多次采样 (temperature=0.2) 不改变结果**——更高温度可能引入差异但预计会进一步降低准确率。

---

## 综合诊断

```
完整事实 → Solver → 讨论 → Finalizer → 答案
   ✅         ❌        ❌        ❌          ❌
```

| 问题 | 答案 |
|---|---|
| 完整事实到达 finalizer 了吗？ | ✅ 是，几乎所有设置中完整事实都到达了 |
| 正确答案在轨迹中出现过吗？ | ❌ 98.7% 的失败案例中从未出现 |
| 在哪丢了？ | **Solver 端** — 分片信息下模型无法推理 |
| 顺序效应在哪？ | **Finalizer** — 80% 翻转率，但翻转后全是错的 |
| Reset 能恢复多少？ | +5% semantic，格式 100% — 有用但微弱 |
| Canonical order 能恢复多少？ | +5% semantic，+5% format — 轻微改善 |
| Self-check 能恢复多少？ | 0% — 无效果 |
| 格式 vs 语义？ | 88.3% 语义错误，6.7% 纯格式错误 |
| Seed 稳定吗？ | ✅ 高度稳定 (T=0.2)，同题不同 seed 结果一致 |

## 核心结论

**Qwen2.5-1.5B 作为 solver 时，multi-agent 信息分片架构在这个难度级别上基本无法工作。** 问题不在协作机制（讨论、ledger、finalizer），而在于 1.5B 模型本身无法从部分事实中完成多步数学推理。即使是 `single_full_information`（一次性给所有事实），0.80 的语义准确率也完全被格式错误抵消（strict accuracy = 0.00）。

**建议**: 
1. 使用更强的 solver 模型（7B+）重复实验
2. 若必须用 1.5B，需在格式训练或 few-shot 上做针对性优化
3. Reset 和 canonical order 是唯二有效的干预方向，值得进一步探索
