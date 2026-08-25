# 实验结果与结论

## 结果

本次实验使用 `data/20.json`，共 20 题，完整运行六组 causal matrix 和五种 generated planner，共得到 440 条 case 结果。运行模式为 `hybrid`，使用本地 `qwen2.5-1.5B`，设备为 `cuda`。

模型调用统计：

```text
llm_call_count: 368
total_tokens: 118196
llm_runtime_seconds: 1070.126
```

Oracle 上界结果：

| Setting | Final Accuracy | Dependency Edge F1 | Executable Rate |
|---|---:|---:|---:|
| Gold Facts + Fresh + Oracle Plan | 1.00 | 0.974 | 1.00 |
| LLM Facts + Fresh + Oracle Plan | 0.95 | 0.956 | 0.95 |

Fresh evidence 下的 generated plan 最好结果：

| Facts | Best Planner | Final Accuracy | Dependency Edge F1 | Executable Rate |
|---|---|---:|---:|---:|
| Gold Facts | C/D/E | 0.20 | 0.098 | 1.00 |
| LLM Facts | C/D/E | 0.20 | 0.098 | 1.00 |

三类 gap：

```text
Fact Extraction Gap:
final_accuracy_gap = 0.05
fact_correctness_gap = 0.0134

Dependency Recovery Gap:
B_backward final_accuracy_gap = 0.85
C/D/E final_accuracy_gap = 0.80

History Contamination Gap:
所有 generated planner 的 executable_rate_gap 约为 1.00
```

History 条件下，所有 generated setting 的 executable rate 都降为 0，主要原因是 history derived facts 与 original facts 发生冲突，并被 fact conflict checker 拦截。

## 结论

当前框架已经能够完成：

```text
fact provenance 保留
backward dependency search
executable IR 执行
hard verification
oracle plan 对照
六组 causal matrix
token/time 记录
```

但从结果看，在没有 gold plan 的情况下，仅依赖分散 evidence 自动恢复 dependency structure 仍然不稳定。

核心结论是：

```text
主要瓶颈是 Dependency Recovery，而不是 Fact Extraction。
```

原因是：

```text
Gold Facts + Oracle Plan 可以达到 1.00 accuracy；
LLM Facts + Oracle Plan 仍有 0.95 accuracy；
但 Generated Plan 最好只有 0.20 accuracy。
```

因此，fact extraction 带来的损失较小，而 generated planner 生成的 dependency graph 与 oracle graph 差距较大。

History contamination 的影响主要体现在 executability 上：当前 conflict checker 能够检测污染事实，但也会导致 history setting 下 executable rate 归零。

总体回答研究问题：

```text
当前 search + verification 方法可以恢复部分可执行 dependency program，
但尚不能稳定恢复接近 Oracle Plan 的 dependency graph。
下一步应重点改进 dependency expansion、verification-guided repair 和 relation fact support 的使用。
```

