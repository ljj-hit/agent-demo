# 50题三种设置表现分析报告

数据来源仅限以下三次输出：

- `outputs/20260619_225524`：Single Agent
- `outputs/20260620_003621`：Multi-Agent
- `outputs/20260619_232557`：Multi-Agent + Verifier

测试脚本参考 `run_qwen_local.py`：每题先由 solver 作答，再由 finalizer 给最终答案；`multi_verifier` 额外加入 verifier，verifier 会输出 `verified_answer`、`chosen_solver` 等字段，finalizer 再根据 verifier 结果作答。

## 总体正确率

| 设置 | 题数 | 正确数 | 正确率 | 平均 token |
|---|---:|---:|---:|---:|
| Single Agent | 50 | 34 | 68% | 2748.94 |
| Multi-Agent | 50 | 37 | 74% | 4587.68 |
| Multi-Agent + Verifier | 50 | 47 | 94% | 7719.82 |

结论：两个 solver 的多智能体设置比单智能体提升 6 个百分点；加入 verifier 后提升明显，比普通 Multi-Agent 高 20 个百分点，比 Single Agent 高 26 个百分点。代价是 token 消耗也最高。

## Multi-Agent 具体表现

该设置中，50 题里有 20 题在过程中出现过错误：Solver A 错 16 题，Solver B 错 14 题，两者都错 10 题；最终错 13 题，另有 7 题虽然某个 solver 错了但 finalizer 选对或纠正了。

| 题号 | A | B | Final | 初答到终答变化 | 带偏/纠正判断 |
|---:|---|---|---|---|---|
| 2 | 错 757 | 错 165 | 错 757 | A错→终错；B错→终错 | 双 solver 都错，finalizer 采纳 A，A 带偏 |
| 3 | 错 18 | 错 20 | 错 20 | A错→终错；B错→终错 | 双错，finalizer 采纳 B，B 带偏 |
| 4 | 错 285 | 错 456 | 错 456 | A错→终错；B错→终错 | 双错，finalizer 采纳 B |
| 6 | 错 29 | 对 27 | 对 27 | A错→终对；B对→终对 | B 纠正了 A 的错误 |
| 8 | 错 300 | 错 90 | 错 90 | A错→终错；B错→终错 | 双错，finalizer 采纳 B |
| 10 | 错 624 | 对 434 | 对 434 | A错→终对；B对→终对 | B 纠正了 A 的错误 |
| 12 | 对 402 | 错 287 | 错 1/4 | A对→终错；B错→终错 | finalizer 被 B/中间分数抽取带偏，错过 A |
| 13 | 错 7 | 错 7.25 | 错 7.25 | A错→终错；B错→终错 | 双错，finalizer 采纳 B |
| 14 | 错 9 | 错 15 | 错 9 | A错→终错；B错→终错 | 双错，finalizer 采纳 A |
| 19 | 错 333 | 错 162 | 错 333 | A错→终错；B错→终错 | 双错，finalizer 采纳 A |
| 26 | 错 35 | 错 180 | 错 180 | A错→终错；B错→终错 | 双错，finalizer 采纳 B |
| 29 | 错 214 | 错 4 | 错 -214 | A错→终错；B错→终错 | 双错，finalizer 又引入符号/费用理解错误 |
| 30 | 错 11 | 对 7 | 对 7 | A错→终对；B对→终对 | B 纠正了 A 的错误 |
| 33 | 错 24 | 对 8 | 对 8 | A错→终对；B对→终对 | B 纠正了 A 的错误 |
| 35 | 对 31 | 错 941 | 对 31 | A对→终对；B错→终对 | finalizer 选 A，避开 B |
| 37 | 错 24 | 错 24 | 错 24 | A错→终错；B错→终错 | 双错且答案一致，finalizer 被共识带偏 |
| 45 | 对 12 | 错 16 | 错 16 | A对→终错；B错→终错 | finalizer 错选 B |
| 46 | 错 24 | 对 96 | 错 24 | A错→终错；B对→终错 | finalizer 错选 A |
| 47 | 对 680 | 错 128 | 对 680 | A对→终对；B错→终对 | finalizer 选 A，避开 B |
| 49 | 错 790 | 对 2620 | 对 2620 | A错→终对；B对→终对 | B 纠正了 A 的错误 |

简要归纳：普通 Multi-Agent 的主要风险是没有显式校验者。两个 solver 都错时 finalizer 基本无法自救；一个 solver 对、一个错时，finalizer 有时能选对，但第 12、45、46 题说明它也会错过正确 solver。

## Multi-Agent + Verifier 具体表现

该设置中，50 题里有 20 题在过程中出现过错误：Solver A 错 16 题，Solver B 错 12 题，两者都错 8 题；最终只错 3 题。Verifier 在这些问题中给出正确 `verified_answer` 19 次，但第 45 题 verifier 本身失败；第 2、29 题 verifier 已给出正确答案，finalizer 没有跟随。

| 题号 | A | B | Verifier 选择/答案 | Final 是否跟 verifier | 初答到终答变化 | 带偏/发现判断 |
|---:|---|---|---|---|---|---|
| 2 | 错 252 | 错 353 | uncertain / 对 253 | 否 | A错→终错；B错→终错；V对→终错 | Verifier 发现并修正，但 finalizer 未跟随，被 A 带偏 |
| 3 | 错 20 | 错 20 | uncertain / 对 15 | 是 | A错→终对；B错→终对；V对→终对 | Verifier 发现双错并纠正 |
| 4 | 错 798 | 错 627 | uncertain / 对 741 | 是 | A错→终对；B错→终对；V对→终对 | Verifier 发现双错并纠正 |
| 8 | 错 300 | 错 300 | uncertain / 对 780 | 是 | A错→终对；B错→终对；V对→终对 | Verifier 发现双错并纠正 |
| 10 | 错 704 | 对 434 | 格式异常 / 对 434 | 是 | A错→终对；B对→终对；V对→终对 | Verifier 答案正确，但 `chosen_solver` 无效，主要靠 verified answer 稳住 |
| 12 | 对 402 | 错 0.25 | uncertain / 对 402 | 是 | A对→终对；B错→终对；V对→终对 | Verifier 发现 B 错，保留 A 的正确答案 |
| 13 | 错 7 | 对 6 | 格式异常 / 对 6 | 是 | A错→终对；B对→终对；V对→终对 | Verifier 识别 B 对，但 `chosen_solver` 格式异常 |
| 14 | 错 3 | 对 7 | uncertain / 对 7 | 是 | A错→终对；B对→终对；V对→终对 | Verifier 答案正确，但 critique 有矛盾，发现质量一般 |
| 19 | 错 36 | 错 162 | uncertain / 对 324 | 是 | A错→终对；B错→终对；V对→终对 | Verifier 发现双错并纠正 |
| 22 | 错 183 | 对 366 | uncertain / 对 366 | 是 | A错→终对；B对→终对；V对→终对 | Verifier 答案正确，但误判 B 也错；结果层面纠正成功 |
| 23 | 对 10 | 错 4 | uncertain / 对 10 | 是 | A对→终对；B错→终对；V对→终对 | Verifier 发现 B 错并保留 A |
| 26 | 错 60 | 错 140 | uncertain / 对 125 | 是 | A错→终对；B错→终对；V对→终对 | Verifier 发现双错并纠正 |
| 29 | 错 12 | 对 36 | uncertain / 对 36 | 否 | A错→终错；B对→终错；V对→终错 | Verifier 给出正确答案但诊断不清，finalizer 未跟随并输出 11 |
| 33 | 错 24 | 对 8 | 格式异常 / 对 8 | 是 | A错→终对；B对→终对；V对→终对 | Verifier 答案正确，但错误批评 B；结果未受影响 |
| 35 | 错 21 | 对 31 | corrected / 对 31 | 是 | A错→终对；B对→终对；V对→终对 | Verifier 明确发现 A 错，采纳/修正为 B 的正确答案 |
| 37 | 错 24 | 错 25 | uncertain / 对 22 | 是 | A错→终对；B错→终对；V对→终对 | Verifier 发现双错并纠正 |
| 38 | 对 20 | 错 50 | 格式异常 / 对 20 | 是 | A对→终对；B错→终对；V对→终对 | Verifier 发现 B 错，保留 A |
| 40 | 错 96 | 对 64 | uncertain / 对 64 | 是 | A错→终对；B对→终对；V对→终对 | Verifier 答案正确，但没有准确指出 A 错 |
| 45 | 对 12 | 错 14 | uncertain / 错 14 | 否 | A对→终错；B错→终错；V错→终错 | Verifier 未发现 B 错且输出无效，finalizer 被 B 带偏 |
| 46 | 错 24 | 错 24 | uncertain / 对 96 | 是 | A错→终对；B错→终对；V对→终对 | Verifier 发现双错并纠正 |

简要归纳：Verifier 的最大贡献是能在两个 solver 都错时直接给出纠正答案，这解释了第 3、4、8、19、26、37、46 题的提升。主要问题有两类：一是 `chosen_solver` 经常输出 `A|B|corrected|uncertain` 这种模板残留或过多使用 `uncertain`，选择字段不可靠；二是 verifier 的结果答案常常正确，但 critique 会误判 solver 过程。最终错误的 3 题中，第 2、29 题是 finalizer 没有跟 verifier，第 45 题是 verifier 未发现错误。

## 总结

Single Agent 基线正确率最低。Multi-Agent 能靠两个 solver 的互补性修正一部分错误，但缺少强校验时，finalizer 容易被错误 solver 或错误共识带偏。Multi-Agent + Verifier 表现最好，核心收益来自 verifier 对错误 solver 的显式纠偏；不过 verifier 的选择字段和解释可靠性不足，建议后续强化 JSON schema 约束，并让 finalizer 明确优先服从 verifier 的 `verified_answer`。
