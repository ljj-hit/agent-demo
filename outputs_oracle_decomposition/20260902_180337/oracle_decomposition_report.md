# Oracle Decomposition 实验简报

## 实验设置

- 数据集：`data/20.json`
- 样本数：20
- planner：`E_beam_verify_repair`
- proposal backend：`llm`
- 模型路径：`qwen2.5-1.5B`
- device：`cuda`
- temperature：0.0
- seed：42
- 输出目录：`outputs_oracle_decomposition/20260902_180337`

本实验比较 O0 到 O6 七个 oracle level。O0-O5 保持同一个 planner 生成 dependency program，只逐步增加 planner 可见的 oracle context；O6 直接使用完整 oracle program，作为 upper bound。

## 汇总结果

| Level | 可见 Oracle 信息 | Final Accuracy | Edge F1 | Executable Rate | Candidate Emergence |
|---|---|---:|---:|---:|---:|
| O0 | none | 0.0% | 3.54% | 100.0% | 0.0% |
| O1 | goal | 0.0% | 3.54% | 100.0% | 0.0% |
| O2 | goal, relevant_facts | 0.0% | 3.54% | 100.0% | 0.0% |
| O3 | goal, relevant_facts, fact_binding | 0.0% | 3.54% | 100.0% | 0.0% |
| O4 | goal, relevant_facts, fact_binding, local_relations | 0.0% | 2.00% | 100.0% | 0.0% |
| O5 | goal, relevant_facts, fact_binding, local_relations, topology | 0.0% | 2.00% | 100.0% | 0.0% |
| O6 | full oracle program | 100.0% | 97.40% | 100.0% | 100.0% |

## 简要结论

1. O0-O5 的 executable rate 均为 100%，说明生成的 IR 基本满足可执行性约束，执行器和保存流程正常。

2. O0-O5 的 final accuracy 和 candidate emergence rate 均为 0%，说明在不直接提供完整 executable program 的情况下，当前 LLM planner 没有产生正确答案候选。

3. 从 O0 到 O5，逐步加入 goal、relevant fact 标记、fact binding、local relations 和 topology 后，final accuracy 没有提升。这表明当前瓶颈不在执行器，而更可能在 planner 对 `OracleContext` 的利用能力、LLM 输出格式稳定性，或 dependency expansion 的生成质量。

4. O6 达到 100% final accuracy 和 100% candidate emergence，说明 oracle plan 构造、IR 执行和 evaluation upper bound 是有效的。O0-O5 与 O6 的巨大差距可以作为后续分析 dependency program recovery 失败位置的主要证据。

## 备注

本报告只基于当前 `metrics.csv` 做简要总结，不包含额外实验结果分析。
