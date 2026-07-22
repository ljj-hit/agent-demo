# Hidden-GSM8K 五种设置 trace 离线重算报告

## 口径

本报告从 `outputs_hidden_gsm8k` 下五份 `traces_all.json` 离线生成，共 120 条 trace；没有调用模型或在线裁判。数值答案正确性、格式合规性、来源一致性是三个互不覆盖的维度。明确写出的数值答案即使格式违规也会保留并参与正确性比较。`computed` 按 `recomputed` 的语义读取，但另记格式问题。

## 汇总

| 设置 | 变体 | N | 答案正确 | 正确率 | 格式问题 | 来源不一致 | 原始 correct | 因解耦恢复的正确项 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_full | - | 20 | 15 | 75.0% | 20 | 0 | 14 | 1 |
| single_partial | - | 40 | 0 | 0.0% | 40 | 0 | 0 | 0 |
| multi_partial | - | 20 | 1 | 5.0% | 2 | 3 | 0 | 1 |
| multi_partial_verifier | - | 20 | 1 | 5.0% | 4 | 15 | 0 | 1 |
| oracle_broadcast | - | 20 | 10 | 50.0% | 5 | 7 | 3 | 7 |

### single_partial 变体明细

| 变体 | N | 答案正确 | 正确率 | 格式问题 |
|---|---:|---:|---:|---:|
| A | 20 | 0 | 0.0% | 20 |
| B | 20 | 0 | 0.0% | 20 |

## 总体结论

离线数值正确 27/120（22.5%）；有格式问题 71/120；有来源不一致 25/120。原始 `correct` 为真 17/120，格式与正确性解耦后恢复 10 个正确答案。

格式问题分布：`answer_not_on_first_line` 57 次；`more_than_three_reasoning_sentences` 55 次；`computed_used_instead_of_recomputed` 9 次；`finalizer_physical_line_count_not_three` 2 次；`finalizer_not_three_labeled_lines` 1 次。

来源一致性问题分布：`recomputed_without_numeric_answer` 18 次；`selected_source_solver_a_unavailable` 5 次；`selected_source_missing` 1 次；`selected_source_solver_b_unavailable` 1 次。

## 被原逻辑因格式清空、但离线恢复为正确的条目

- `single_full` q8：恢复答案 `1920`；格式问题：answer_not_on_first_line, more_than_three_reasoning_sentences；来源问题：无。
- `multi_partial` q1：恢复答案 `42`；格式问题：computed_used_instead_of_recomputed；来源问题：无。
- `multi_partial_verifier` q1：恢复答案 `42`；格式问题：computed_used_instead_of_recomputed；来源问题：无。
- `oracle_broadcast` q1：恢复答案 `42`；格式问题：computed_used_instead_of_recomputed；来源问题：无。
- `oracle_broadcast` q8：恢复答案 `1920`；格式问题：computed_used_instead_of_recomputed；来源问题：无。
- `oracle_broadcast` q11：恢复答案 `520`；格式问题：computed_used_instead_of_recomputed；来源问题：无。
- `oracle_broadcast` q12：恢复答案 `3`；格式问题：无；来源问题：selected_source_solver_a_unavailable。
- `oracle_broadcast` q14：恢复答案 `768`；格式问题：无；来源问题：selected_source_solver_a_unavailable。
- `oracle_broadcast` q17：恢复答案 `35`；格式问题：finalizer_physical_line_count_not_three；来源问题：无。
- `oracle_broadcast` q18：恢复答案 `200`；格式问题：computed_used_instead_of_recomputed；来源问题：无。

## 解读限制

`single_partial` 的 A/B 是同一设置下两个独立变体，因此该目录有 40 条而非 20 条；报告既分变体展示，也保留全部逐题记录。离线判断只认可明确标签或 trace 中已经保存的数值预测，不会从长推理过程里挑一个碰巧等于 gold 的数字。来源一致性使用 trace 保存的候选答案；若所选候选因上游格式错误而不可用，会记为来源问题，而不会反过来清空最终数值答案。
