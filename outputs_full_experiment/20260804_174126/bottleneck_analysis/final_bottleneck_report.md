# GSM8K Multi-Agent 瓶颈证据完整分析报告

**实验**: `20260804_174126` | **模型**: Qwen2.5-1.5B | **生成时间**: 2026-08-06
**数据集**: 20 题 GSM8K 变体 | **讨论轮数**: 2 | **Temperature**: 0.2 | **Seeds**: 3

---

## 目录

1. [逐题阶段化分析表（canonical_order）](#一逐题阶段化分析表)
2. [瓶颈漏斗](#二瓶颈漏斗)
3. [关键分离指标：Solver 问题 vs Finalizer 问题](#三关键分离指标)
4. [典型案例深度剖析](#四典型案例深度剖析)
5. [各 Setting 对比漏斗](#五各-setting-对比漏斗)
6. [核心结论](#六核心结论)

---

## 一、逐题阶段化分析表

以下针对 **`canonical_order`** 设置（信息按标准顺序逐步注入，最具诊断价值）逐题分析。
每题 3 seed，取 majority vote (>=2/3)。

### 图例
- ✅ = 是/通过
- ❌ = 否/未通过
- ⚠️ = 部分/不确定
- — = 不适用

### 各题分析

| # | Gold | 题目概要 | 1. A事实披露 | 2. B事实披露 | 3. 首次完整轮次 | 4. A自述 | 5. B自述 | 6. 互相吸收 | 7. 事实失真 | 8. 推理中出现正确答案 | 9. 正确候选 | 10. Finalizer见候选 | 11. 首个错误阶段 | 最终 |
|---|------|----------|-------------|-------------|----------------|---------|---------|------------|-----------|---------------------|------------|-------------------|-----------------|------|
| 1 | 42 | Julie读书页数 | ✅ | ✅ | R1 after_simultaneous | ✅ | ✅ | ⚠️ A吸B✅ B吸A❌ | B方缺少A的关键数值120 | ✅ | ✅ | ❌ | finalizer_rejected | ❌ |
| 2 | 5 | Betty买钱包 | ✅ | ✅ | R1 after_simultaneous | ✅ | ❌ | ❌ 双方互不吸收 | B遗漏$15; A/B均未交叉引用对方数值 | ✅ | ✅ | ❌ | finalizer_rejected | ❌ |
| 3 | 16 | Ken盒子重量 | ✅ | ✅ | R1 after_simultaneous | ✅ | ✅ | ❌ 双方互不吸收 | 16出现在R1推理中但未被提取为候选 | ✅ | ❌ | ❌ | not_captured | ❌ |
| 4 | 41 | Alexis买鞋 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏30,46,38; B遗漏16 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 5 | 38 | Bella买邮票 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏11,9; B遗漏13 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 6 | 5 | Ann买上衣 | ✅ | ✅ | R1 after_simultaneous | ✅ | ❌ | ❌ | B遗漏2,10,4; 5出现在R1推理中 | ✅ | ❌ | ✅ | not_captured | ❌ |
| 7 | 1200 | Noah卖画 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏60,30,8; B遗漏4 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 8 | 1920 | Carolyn练琴 | ✅ | ✅ | R1 after_simultaneous | ❌ | ✅ | ❌ | A遗漏20,3 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 9 | 45 | 作业三部分 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏25; B遗漏2 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 10 | 840 | James工资 | ✅ | ✅ | R1 after_simultaneous | ✅ | ❌ | ❌ | B遗漏20% | ✅ | ❌ | ❌ | not_captured | ❌ |
| 11 | 520 | Tara笔记本 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏1000,20; B遗漏65 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 12 | 3 | Roger买pack | ✅ | ✅ | R1 after_simultaneous | ❌ | ✅ | ❌ | A遗漏13 | ✅ | ✅ | ✅ | none | ✅ |
| 13 | 28 | 减肥四人组 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏103,27; B遗漏7 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 14 | 768 | 菜园产量 | ✅ | ✅ | R1 after_simultaneous | ✅ | ✅ | ✅ 双方正确互吸 | 无失真 | ❌ | ❌ | ✅ | solver_never | ❌ |
| 15 | 100 | Jennifer买罐头 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏40; B遗漏50 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 16 | 700 | Irene收入 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏500,40; B遗漏20,50 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 17 | 35 | Winwin彩票 | ✅ | ✅ | R2 after_simultaneous | ✅ | ✅ | ❌ | 事实复述OK但计算顺序混乱 | ✅ | ✅ | ✅ | finalizer_rejected | ❌ |
| 18 | 200 | John存钱罐 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏25; B遗漏400 | ❌ | ❌ | ❌ | solver_never | ❌ |
| 19 | 135 | Wickham盘子 | ✅ | ✅ | R1 after_simultaneous | ❌ | ✅ | ❌ | A遗漏30 | ❌ | ❌ | ✅ | solver_never | ❌ |
| 20 | 43 | 安全帽计数 | ✅ | ✅ | R1 after_simultaneous | ❌ | ❌ | ❌ | A遗漏26,15,24; B遗漏4,6,12 | ❌ | ❌ | ❌ | solver_never | ❌ |

### 逐题错误阶段详细分布（60 traces = 20题 x 3 seeds）

| 错误阶段 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| **solver_never_produced_correct** | 39 | 65.0% | Solver 从头到尾未产生过正确答案 |
| **correct_in_reasoning_not_captured** | 9 | 15.0% | 推理中出现过正确答案但未被提取为正式候选 |
| **correct_candidate_not_passed_to_finalizer** | 9 | 15.0% | 正确候选存在但未被传给 Finalizer |
| **none_correct** | 3 | 5.0% | 最终正确（仅 Q12 的 3 个 seed） |

---

## 二、瓶颈漏斗

针对 `canonical_order`（20 题，majority vote）：

```
20 题
  │
  ├─→ 20 题事实完整公开 (100%)
  │     （canonical_order 中所有事实按序注入，无遗漏）
  │
  ├─→ 1 题事实没有失真 (5%)
  │     （仅 Q14：A/B 均正确复述自身事实且正确吸收对方事实）
  │     事实失真表现在：Agent 输出中遗漏关键数值、未在推理中使用对方数据
  │
  ├─→ 7 题完成全局状态重建 (35%)
  │     （正确答案在任意 Agent 的 reasoning 中出现过）
  │     Q1, Q2, Q3, Q6, Q10, Q12, Q17
  │
  ├─→ 4 题形成完整推理计划 (20%)
  │     （正确答案进入正式候选字段 candidate_appearances）
  │     Q1, Q2, Q12, Q17
  │     ⚠️ Q3, Q6, Q10 推理中有正确答案但未被提取为候选
  │
  ├─→ 4 题出现正确候选 (20%)
  │     （同上，majority vote 确认）
  │
  └─→ 1 题最终正确 (5%)
        （仅 Q12：Roger 买 pack，13+3+2=18, 18/6=3）
```

### 损失点定位

| 漏斗阶段 | 存活数 | 损失数 | 损失率 | 损失原因 |
|----------|--------|--------|--------|----------|
| 事实完整公开 | 20 | 0 | 0% | — |
| → 事实无失真 | 1 | 19 | 95% | **Agent 输出遗漏关键数值** |
| → 全局状态重建 | 7 | — | — | (此处跳跃是因为重建可以在有失真情况下发生) |
| → 正确候选生成 | 4 | 3 (of 7) | 43% | **候选提取失败**：正确答案在推理中但未成候选 |
| → 最终正确 | 1 | 3 (of 4) | 75% | **Finalizer 拒绝正确候选** |

---

## 三、关键分离指标

### 3.1 指标定义

**candidate_emergence_given_complete_disclosure**
> 事实已完整公开时，正确候选出现的比例。
> **衡量 Solver 端能力**：给定所有事实，Agent 能否将其转化为正确的形式化候选答案？

**final_retention_given_correct_candidate**
> 正确候选已存在时，最终答案保留正确候选的比例。
> **衡量 Finalizer 端能力**：正确答案已经有人提出来了，Finalizer 能不能识别并采纳？

### 3.2 canonical_order 指标

| 指标 | 计算 | 值 |
|------|------|-----|
| 事实完整公开 | — | 20/20 = 100% |
| candidate_emergence_given_complete_disclosure | 4/20 | **20.0%** |
| final_retention_given_correct_candidate | 1/4 | **25.0%** |

> **解读**：
> - 即使事实全部公开，只有 20% 的题目能产生正确候选 → **Solver 瓶颈严重**
> - 即使在罕见的"正确候选已存在"的情况下，只有 25% 被保留 → **Finalizer 瓶颈同样严重**
> - 两者相乘：0.20 × 0.25 = 0.05 = 5%，恰好等于最终准确率

### 3.3 不同 Setting 对比

| Setting | Facts Complete | Candidate Emergence | Final Retention | Final Accuracy |
|---------|---------------|--------------------|-----------------|----------------|
| **single_full_information** (基线) | 100% | N/A (无候选机制) | N/A | **80%** |
| all_at_start_AB | 100% | 4/20 = 20% | 6/4 = 150%* | 6/20 = 30% |
| all_at_start_BA | 100% | 3/20 = 15% | 4/3 = 133%* | 4/20 = 20% |
| **canonical_order** | 100% | 4/20 = 20% | 1/4 = 25% | 1/20 = 5% |
| before_final_reset | 100% | 0/20 = 0% | N/A | 1/20 = 5% |
| before_final_transcript | 100% | 0/20 = 0% | N/A | 0/20 = 0% |

> \* `all_at_start` 的 retention > 100% 是因为 **Finalizer 在 Solver 未产生正确候选的情况下独立解出了正确答案**。这发生在 Finalizer 直接从讨论 transcript 中提取事实并自行计算时（例如 Q10 的 $840/week）。此时候选追踪为空但最终答案正确。

### 3.4 分离 Solver 问题与 Finalizer 问题

```
                   Solver 端                    Finalizer 端
                (事实→候选)                    (候选→答案)
                     │                              │
    single_full:  80% (无候选机制，直接求解)         │
                     │                              │
    all_at_start:  20% ────────────────────────→ 150%* (独立求解补位)
                     │                              │
    canonical:     20% ────────────────────────→  25%
                     │                              │
    reset:          0% ────────────────────────→  N/A
```

| 设置 | Solver 问题严重度 | Finalizer 问题严重度 | 主要瓶颈 |
|------|------------------|---------------------|---------|
| canonical_order | **80% 损失** | **75% 损失** | 双重瓶颈 |
| all_at_start | **80% 损失** | -35% (Finalizer 补位) | Solver 为主 |
| before_final_reset | **100% 损失** | — | Solver 完败 |

---

## 四、典型案例深度剖析

### 4.1 Q1 (Gold: 42) — Finalizer 拒绝正确候选

**事实分配**:
- A: Julie read 12 pages yesterday. Today she read twice as many.
- B: Book has 120 pages. Tomorrow reads half of remaining.

**事件时间线**:

| 阶段 | Agent | 输出 | 正确? |
|------|-------|------|-------|
| R1 Send | Solver A | 开始计算 12×2=24... | ⚠️ 进行中 |
| R1 Send | Solver B | 需要更多信息 | ❌ |
| R2 Send | Solver A | **42** | ✅ |
| R2 Send | Solver B | **42** | ✅ |
| Solver Final A | — | **42** | ✅ |
| Solver Final B | — | **42** | ✅ |
| **Finalizer** | — | **36** | ❌ |

**Finalizer 推理**: "Calculating the remaining pages after yesterday and today, then dividing by two for tomorrow's reading."

**诊断**: 
- 4 次正确候选 (2 次 R2 + 2 次 Final)，全部为 42
- Finalizer 选择 "recomputed" 模式，自行计算得到 36
- 36 = 120 - 12 - 24 = 84, 84/2 = 42... 但 Finalizer 算成了 36
- 可能计算路径: 12+24=36 (pages read), 错误地将 36 作为答案
- **根因**: Finalizer 的独立计算有算术错误，覆盖了 Solver 的正确候选

### 4.2 Q2 (Gold: 5) — Finalizer 产生幻觉答案

**事实分配**:
- A: Wallet costs $100. Betty has half.
- B: Parents give $15. Grandparents give twice that.

**事件时间线**:

| 阶段 | Agent | 输出 | 正确? |
|------|-------|------|-------|
| R2 Send | Solver A | **5** | ✅ |
| R2 Send | Solver B | **5** | ✅ |
| Solver Finals | A, B | **5** | ✅ |
| **Finalizer** | — | **30** | ❌ |

**诊断**:
- 正确候选: 4/4 = 5
- Finalizer 输出 30, Reason: "Calculating the total amount given by parents and grandparents, subtracting what she already has, and finding the difference."
- 30 = 15 + 15×2? 不, 30 应该是 15×2 = grandparents' contribution
- Finalizer 取了 grandparents 的钱 (30) 作为最终答案，完全忽略了 Betty 已有的 $50 和总价 $100
- **根因**: Finalizer 的计算链断裂，只完成了部分计算

### 4.3 Q17 (Gold: 35) — Finalizer 幻觉"四舍五入"

**事实分配**:
- A: Winwin wins $50. Pays 20% tax.
- B: Pays $5 processing fee.

**事件时间线**:

| 阶段 | Agent | 输出 | 正确? |
|------|-------|------|-------|
| R1 Send | Solver A | 45-10=35 推理中 | ⚠️ (顺序错但结果对) |
| R2 Send | Both | Undetermined (信息不足) | ❌ (倒退!) |
| Solver Final A | — | **35** | ✅ |
| Solver Final B | — | **35** (推理混乱) | ✅ |
| **Finalizer** | — | **38** | ❌ |

**Finalizer 原文**: "Applying the tax rate to the winnings correctly results in $40 after tax, subtracting the $5 fee leaves $35, which rounds down to $38 due to rounding rules."

**诊断**:
- Finalizer **自己算出了 35** (40-5=35)！
- 但随后凭空加入 "rounds down to $38 due to rounding rules"
- 这是纯粹的**幻觉**——没有任何 rounding 规则能让 35 变成 38
- **根因**: Finalizer 产生了置信度不足导致的随机偏离

### 4.4 Q10 (Gold: 840) — Finalizer 补位成功

**事实分配**:
- A: Main job $20/hr, 30 hrs/week.
- B: Second job 20% less, half the hours.

**事件时间线**:

| 阶段 | Agent | 输出 | 正确? |
|------|-------|------|-------|
| R1 Send | Solver A | 20×30=600, 20×0.8=16, 30/2=15... (截断) | ⚠️ |
| R1 Send | Solver B | "信息不足" | ❌ |
| R2 Send | Both | "缺少第二份工作的时薪和工时" | ❌ (倒退!) |
| Solver Finals | A, B | 空答案 | ❌ |
| **Finalizer** | — | **$840/week** | ✅ |

**诊断**:
- Solver A 在 R1 开始了正确计算但被截断 (max_new_tokens=384)
- Solver B 始终声称信息不足（即使在 all_at_start 已拥有全部事实）
- 两个 Solver 均未产生任何候选答案
- **Finalizer 独立完成了计算**: 20×30=600, 16×15=240, 600+240=840
- 但 Finalizer 的 Reason 文本与答案矛盾（说"信息不足"但给出了 840）
- **根因**: 截断 + Solver B 能力不足 → 但 Finalizer 有独立求解能力

### 4.5 Q12 (Gold: 3) — 唯一成功案例

**事实分配**:
- A: 13 players. Each needs one pouch.
- B: 3 coaches, 2 helpers. 6 pouches per pack.

**为什么这道题成功了？**
1. **计算步骤简单**: 13+3+2=18, 18/6=3（仅两步整数运算）
2. **Solver B 在 R1 就完成了完整计算**: 清楚列出 13+3+2=18, 18/6=3
3. **信息依赖简单**: 不涉及百分比、分数、多步嵌套
4. **Finalizer 正确采纳**: 选择了 recomputed 模式但得到了相同答案

**对比 Q1-Q20**: 其他题目涉及乘法、百分比、比率、多步嵌套计算（如 20% lower, half of remaining, 20% tax 等），模型无法在分片信息下完成这些较复杂的推理。

### 4.6 Q14 (Gold: 768) — 唯一无事实失真案例

**事实分配**:
- A: 237 potatoes. 60 fewer cucumbers than potatoes.
- B: Twice as many peppers as cucumbers.

**诊断**:
- 两个 Solver 都正确复述了自身事实
- 两个 Solver 都正确吸收了对方的事实
- 但正确答案 768 从未在推理中出现
- Solver 们的计算: 237 + (237-60) + 2×(237-60) = 237+177+354 = 768
- 实际上模型未能完成这个三步计算链——信息都在，但算不出来
- **根因**: 即使事实无失真，**多步算术能力不足**仍是瓶颈

---

## 五、各 Setting 对比漏斗

### 5.1 all_at_start_AB (事实一开始全部给出，A先发言)

```
20 → 20(事实完整) → 4(正确候选) → 6(最终正确)
                     ↑ Solver损失80%   ↑ Finalizer补位+2
```

### 5.2 all_at_start_BA (事实一开始全部给出，B先发言)

```
20 → 20(事实完整) → 3(正确候选) → 4(最终正确)
                     ↑ Solver损失85%   ↑ Finalizer补位+1
```

### 5.3 canonical_order (标准顺序逐步注入)

```
20 → 20 → 1(无失真) → 7(推理中有) → 4(正确候选) → 1(最终正确)
                    ↓ 95%失真       ↓ 43%提取失败    ↓ 75%Finalizer拒绝
```

### 5.4 before_final_reset (讨论后清空上下文，Finalizer 重置)

```
20 → 0(正确候选) → 1(最终正确)
     ↑ Solver 100%失败  ↑ Finalizer 独立算对1题 (Q7: 1200)
```

### 5.5 before_final_transcript (讨论 transcript + 事实给 Finalizer)

```
20 → 0(正确候选) → 0(最终正确)
     ↑ 完全失败
```

---

## 六、核心结论

### 6.1 瓶颈定位

```
                     Solver 端瓶颈              Finalizer 端瓶颈
                    (事实 → 候选)              (候选 → 答案)
                         │                          │
    ┌────────────────────┼──────────────────────────┤
    │                    ▼                          ▼
    │   canonical_order: 80% 的题目无法生成正确候选  75% 的正确候选被拒绝
    │   all_at_start:    80% 的题目无法生成正确候选  Finalizer 可独立补位 2-3 题
    │   before_final_reset: 100% 无法生成正确候选    Finalizer 可独立补位 1 题
    │
    │   根因: Qwen2.5-1.5B 在分片信息下无法完成多步数学推理
    │   表现: 65% 的 trace 中正确答案从未出现
    │         95% 的题目存在事实复述/吸收失真
    └────────────────────────────────────────────────
```

### 6.2 关键数字

| 指标 | 值 | 解读 |
|------|-----|------|
| **基线天花板** (single_full) | 80% semantic | 模型在完整信息下能解 16/20 |
| **信息完整→正确候选** | **20%** (canonical) | Multi-agent 分片使 Solver 能力降至 1/4 |
| **正确候选→最终正确** | **25%** (canonical) | Finalizer 丢失了 3/4 的正确候选 |
| **端到端准确率** | **5%** (canonical) | 0.20 × 0.25 = 0.05 |
| **事实失真率** | **95%** | 仅 Q14 无失真 |
| **Solver 未产生率** | **65%** | 2/3 的 trace 中正确答案从未出现 |
| **Finalizer 幻觉率** | **75%** (有候选时) | Q1: 42→36, Q2: 5→30, Q17: 35→38 |

### 6.3 失败模式分类

| 失败模式 | 代表题目 | 机制 |
|----------|---------|------|
| **Solver 能力不足** | Q4,5,7,8,9,11,13,15,16,18,20 (11题) | 即使事实齐全，Solver 无法完成算术推理 |
| **候选提取失败** | Q3,6,10 (3题) | 正确答案在 reasoning 中出现但未被 capture 为候选 |
| **Finalizer 计算错误** | Q1, Q2 (2题) | Solver 给出了正确候选，Finalizer 独立计算时算错 |
| **Finalizer 幻觉** | Q17 (1题) | Finalizer 自己算出 35 又随机改成 38 |
| **唯一成功** | Q12 (1题) | 计算最简单 (18/6=3)，两步整数运算 |
| **无失真但算不出** | Q14 (1题) | 事实完美复述+吸收，但三步运算链断裂 |
| **信息时序问题** | Q17 (1题) | R1 有正确答案，R2 倒退声称"信息不足" |

### 6.4 建议

1. **升级 Solver 模型**: Qwen2.5-1.5B 在分片信息下的算术推理能力严重不足。建议使用 7B+ 模型重复实验。
2. **Finalizer 不应 recompute**: 当 Solver 已经给出明确候选时，Finalizer 应优先采纳而非重新计算。可加入 "trust-but-verify" 机制而非 "always-recompute"。
3. **候选提取需要改进**: Q3, Q6, Q10 中正确答案出现在 reasoning 中但未被标记为候选。需要更好的答案提取机制。
4. **抗幻觉机制**: Q17 的 Finalizer 在算出正确答案后随机偏离，需要 consistency check 机制。
5. **截断问题**: Q10 的 Solver A 在计算中途被截断 (max_new_tokens=384)，增加 token 限制可能改善。

### 6.5 两个分离指标的最终答案

```
candidate_emergence_given_complete_disclosure = 4/20 = 20.0%
  → 事实完整公开后，只有 20% 的题目能生成正确候选
  → 这是 Solver 端瓶颈：给定所有事实，Agent 缺乏将其转化为正确答案的能力

final_retention_given_correct_candidate = 1/4 = 25.0%
  → 正确候选存在时，只有 25% 被 Finalizer 保留
  → 这是 Finalizer 端瓶颈：正确答案已有人提出，Finalizer 却无法识别/采纳
```

---

*报告生成于 2026-08-06 | 数据来源: `outputs_full_experiment/20260804_174126/`*
