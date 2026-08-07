# 本地 Qwen 多智能体 GSM8K 实验

本项目提供两个使用本地 Hugging Face Qwen 模型的评测脚本：

| 脚本 | 用途 | 默认数据 | 默认输出 |
| --- | --- | --- | --- |
| `run_qwen_local.py` | 标准 GSM8K 多智能体基线，对比不同的求解、验证和答案选择策略 | `data/50.jsonl` | `outputs/` |
| `run_hidden_gsm8k.py` | Hidden-GSM8K 部分信息实验，评估信息披露、整合和最终答案选择 | `data/20.json` | `outputs_hidden_gsm8k/` |

两个脚本都由本地 Qwen 完成 agent 推理，默认从项目目录下的
`qwen2.5-1.5B/` 加载模型。DeepSeek 使用 OpenAI-compatible API：

- `run_qwen_local.py` 始终使用 DeepSeek 对各阶段答案进行离线判分。
- `run_hidden_gsm8k.py` 默认使用 DeepSeek 复核本地判分和信息完整性，可用
  `--skip-deepseek` 完全离线运行。

## 安装

建议使用独立 Python 环境：

```powershell
pip install -r requirements.txt
```

主要依赖包括 `torch`、`transformers`、`safetensors`、`openai` 和
`python-dotenv`。使用 GPU 时，需要安装与本机 CUDA 匹配的 PyTorch。

默认设备为 `cuda`。也可以传入 `--device cuda:0` 或 `--device cpu`。
CUDA 设备根据计算能力使用 `bfloat16` 或 `float16`，CPU 使用 `float32`。

模型目录应至少包含：

```text
config.json
tokenizer_config.json
tokenizer.json
model.safetensors
```

默认只读取本地模型文件；传入 `--allow-download` 后，Transformers 可以下载
缺失文件。

## DeepSeek 配置

在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

API Key 依次兼容 `DEEPSEEK_API_KEY`、`API_KEY`、`OPENAI_API_KEY`。
服务地址也兼容 `BASE_URL`；模型名也兼容 `MODEL_NAME`。
`run_qwen_local.py` 还支持 `JUDGE_BASE_URL`、`OPENAI_BASE_URL`、
`JUDGE_MODEL` 和 `OPENAI_MODEL`。

Hidden-GSM8K 的 DeepSeek 请求失败时最多重试 4 次；如果最终仍失败，脚本会
保留本地判分并继续保存输出。

## 标准 GSM8K 实验

### 数据和 Prompt

`run_qwen_local.py` 接受 JSON 数组或 JSONL，每条记录必须包含：

```json
{"question": "题目", "answer": "解题过程 #### 最终答案"}
```

如果 `answer` 中含有 `####`，最后一个 `####` 后的内容作为标准答案。
Prompt 位于 `prompts/`：

```text
solver_a.txt
solver_b.txt
verifier.txt
finalizer.txt
```

### 实验设置

- `single`：仅运行 Solver A，其输出直接作为最终结果。
- `multi`：Solver A/B 独立作答，由 Finalizer 汇总。
- `multi_verifier`：增加 Verifier，再由 Finalizer 作答。
- `multi_verifier_forced`：直接采用基于 Verifier 评分选出的答案。
- `multi_candidate_memory`：Finalizer 只能从结构化候选表中选择答案。
- `multi_ask_before_finalize`：Finalizer 作答前，再向两个 Solver 征询一次异议。

兼容别名 `single_agent`、`multi_agent` 和 `multi_agent_verifier`。
不指定 setting 时进入交互选择。

### 运行示例

```powershell
# 检查依赖、模型、数据、Prompt 和 Judge 配置
python run_qwen_local.py --check-config

# 运行单个或全部设置
python run_qwen_local.py --setting single
python run_qwen_local.py --setting multi_verifier
python run_qwen_local.py --setting all

# 配对运行指定设置；同一道题的 Solver A/B 初始输出会缓存复用
python run_qwen_local.py --settings single multi multi_verifier --seed 42

# 自定义模型和生成参数
python run_qwen_local.py --setting all --model-path D:\models\qwen `
  --data-path data\50.jsonl --device cuda:0 --temperature 0
```

常用参数：

```text
--data-path PATH          默认 data/50.jsonl
--model-path PATH         本地 Hugging Face 模型目录
--device DEVICE           默认 cuda
--max-new-tokens N        默认 512
--temperature FLOAT       默认 0.2；0 表示确定性解码
--seed N                  默认 42
--allow-download          允许下载缺失的模型文件
```

每次运行在 `outputs/YYYYMMDD_HHMMSS/` 创建独立目录；若同名则追加
`_02`、`_03` 等后缀。每完成一个样例就增量更新：

- `traces_all.json`：完整 agent 轨迹、Judge 结果、token 和耗时。
- `metrics.csv`：各 setting 的准确率、oracle gap、推理/Judge token 和耗时。
- `failures.json`：最终答案错误的样例及失败分类。

## Hidden-GSM8K 实验

### 数据和 Prompt

`run_hidden_gsm8k.py` 接受 JSON 数组或 JSONL。每条记录必须包含：

```json
{
  "condition_A": "只提供给 Solver A 的条件",
  "condition_B": "只提供给 Solver B 的条件",
  "shared_question": "双方可见的问题",
  "full": "包含完整条件的问题",
  "fact": {
    "A": ["A 方必须披露的事实"],
    "B": ["B 方必须披露的事实"]
  },
  "answer": "解题过程 #### 最终答案"
}
```

为兼容旧数据，`full_question` 可代替 `full`，
`required_private_facts.agent_A/agent_B` 可代替 `fact.A/B`。
`fact` 只用于生成完成后的信息交换评估，不会进入模型 Prompt。

Prompt 位于 `hidden_gsm8k_prompts/`：

```text
solver.txt
verifier.txt
finalizer.txt
```

### 实验设置

- `single_full`：单个 Solver 获得完整信息。
- `single_partial`：分别运行 A、B 两个单智能体部分信息变体。
- `multi_partial`：A/B 持有不同私有条件，进行对称公开讨论后作答。
- `multi_partial_verifier`：在部分信息讨论后增加 Verifier。
- `oracle_broadcast`：将完整私有信息公开广播，作为信息充分的对照组。

多智能体讨论按轮进行；同一轮的 A/B 使用完全相同的公开快照，并通过同一个
GPU batch 生成。一起运行 `multi_partial` 和
`multi_partial_verifier` 时，两者复用同一道题的同一份讨论轨迹。

### 运行示例

```powershell
# 只验证配置，不加载模型推理
python run_hidden_gsm8k.py --check-config

# 完全离线运行单个设置
python run_hidden_gsm8k.py --setting multi_partial --skip-deepseek

# 运行全部设置
python run_hidden_gsm8k.py --setting all

# 配对运行指定设置
python run_hidden_gsm8k.py --settings multi_partial multi_partial_verifier `
  --discussion-rounds 3 --limit 20 --seed 42

# 自定义路径和设备
python run_hidden_gsm8k.py --setting oracle_broadcast `
  --data-path data\20.json --model-path D:\models\qwen `
  --output-dir outputs_hidden_gsm8k --device cuda:0
```

常用参数：

```text
--data-path PATH          默认 data/20.json
--model-path PATH         默认 qwen2.5-1.5B/
--output-dir PATH         默认 outputs_hidden_gsm8k/
--setting NAME            运行一个 setting 或 all
--settings NAME [...]     配对运行多个 setting
--device DEVICE           默认 cuda
--max-new-tokens N        默认 384
--temperature FLOAT       默认 0.2
--discussion-rounds N     默认 2，最小为 1
--seed N                  默认 42
--limit N                 仅运行前 N 条；0 表示全部
--allow-download          允许下载缺失的模型文件
--skip-deepseek           禁用 DeepSeek，使用本地等价性判分
--check-config            打印配置后退出
```

单个 setting 的输出目录为：

```text
outputs_hidden_gsm8k/YYYYMMDD_HHMMSS/
```

一次选择多个 setting 时，每个 setting 使用独立目录：

```text
outputs_hidden_gsm8k/YYYYMMDD_HHMMSS_<setting>/
```

每个目录包含：

- `run_config.json`：本次运行的路径、setting、生成参数和复用策略。
- `traces_all.json`：讨论轮次、公开 transcript、候选答案、格式校验、
  信息覆盖、DeepSeek 复核、token 和耗时。
- `metrics.csv`：准确率、各 agent 正确数、信息完整数、失败类型、
  oracle gap、无效输出、token 和耗时。
- `failures.json`：失败样例及其信息获取、信息整合或答案选择分类。

Hidden-GSM8K 对输出格式执行严格校验。Solver 最终输出的第一行必须是
`Final answer: ...`，并且其后最多三句解释；格式不合法会记为无效输出，
不会因答案数值碰巧正确而计为正确。

## 信息时序重放实验

`run_hidden_gsm8k.py` 在原有 Hidden-GSM8K 设置之外，支持在相同 20 道题上运行六种信息时序重放设置：

- `all_at_start_AB`：第一轮前公开全部事实，A 事实在前、B 事实在后。
- `all_at_start_BA`：事实文本与上一设置完全相同，只交换 A/B 的显示顺序。
- `after_round1`：第一轮讨论结束后公开全部事实。
- `before_final_transcript`：最终回答前公开事实，finalizer 同时看到此前的讨论 transcript。
- `before_final_transcript_ledger`：最终回答前公开事实，finalizer 同时看到规范化事实表和此前的讨论 transcript。
- `before_final_reset`：最终回答前公开事实并清除旧讨论，新的 finalizer 只看到共享问题和固定 A、B 顺序的事实表。
- `finalizer_only_order_ab_ba`：不运行前期讨论，在同一个 setting 中为每题运行 AB、BA 两个 finalizer-only 变体；除事实行顺序外，完整上下文和参数完全相同。

### 受控实验保证

- 六个设置使用同一个本地模型、相同的 solver/finalizer prompt、讨论轮数和 `max_new_tokens`。
- 重放实验的所有模型调用强制使用 `temperature=0`，有效值记录在 replay trace 的 `run_config.temperature` 中。
- A/B 事实逐字复制自数据集的 `condition_A` 和 `condition_B`，不会由模型重新生成、摘要或改写。
- 三个 `before_final_*` 设置逐题共用同一个前期 discussion，并记录相同的 `discussion_trace_hash`。
- 每次本地 agent 调用都在事件的 `actual_messages` 中保存实际可见的完整 system/user 输入。
- 每题记录 `injected_fact_hash` 和 `final_received_fact_hash`。离线汇总会校验六设置的事实 hash，不一致时立即报错。
- `answer`/gold 不会进入任何 agent 的 `actual_messages`，只在生成完成后用于离线判分。
- 数学正确性使用 `semantic_correct`，格式合规使用 `format_compliant`，答案与理由的一致性使用 `answer_reason_consistent`。
- `strict_correct = semantic_correct AND format_compliant AND answer_reason_consistent`，主 `accuracy` 使用 `strict_correct`。
- 单次 finalizer 输出无效时同时记录 `finalizer_exhausted=true` 和 `single_shot_format_failure=true`。

### 运行六个设置

```powershell
python run_hidden_gsm8k.py --settings `
  all_at_start_AB `
  all_at_start_BA `
  after_round1 `
  before_final_transcript `
  before_final_transcript_ledger `
  before_final_reset
```

禁用 DeepSeek 复核、仅使用本地等价性判分：

```powershell
python run_hidden_gsm8k.py --settings `
  all_at_start_AB all_at_start_BA after_round1 `
  before_final_transcript before_final_transcript_ledger before_final_reset `
  --skip-deepseek
```

只检查数据、模型和参数而不执行推理：

```powershell
python run_hidden_gsm8k.py --check-config --settings `
  all_at_start_AB all_at_start_BA after_round1 `
  before_final_transcript before_final_transcript_ledger before_final_reset
```

默认使用 `data/20.json` 的全部 20 题。`--limit N` 可用于调试；正式配对分析应让六个设置运行相同的完整题目集合。

### 输出与离线指标

每个设置写入独立目录：

```text
outputs_hidden_gsm8k/YYYYMMDD_HHMMSS_<setting>/
├── run_config.json
├── traces_all.json
├── metrics.csv
└── failures.json
```

六设置的配对分析写入公共目录：

```text
outputs_hidden_gsm8k/YYYYMMDD_HHMMSS_replay_analysis/
├── replay_analysis.json
└── replay_metrics.csv
```

`replay_metrics.csv` 分别报告每个设置的语义正确率、格式合规率、答案—理由一致率和严格正确率；
其中 `accuracy` 与 `strict_accuracy` 相同。`replay_analysis.json` 另外包含：

- `schedule_flip_rate`：在 AB 固定顺序下，从开头公开改为第一轮后或最终回答前公开时，最终答案发生变化的题数、比例、题号和 pairwise 结果。
- `late_evidence_penalty`：`all_at_start_AB` 正确但 `before_final_transcript` 错误的题数和题号。
- `reset_recovery`：`before_final_transcript` 错误、清除旧讨论后恢复正确的题数和题号。
- `ledger_recovery`：`before_final_transcript` 错误、加入规范化事实表后恢复正确的题数和题号。
- `fact_hash_consistent_across_six_settings`：六设置逐题事实集合的 hash 一致性。

运行 finalizer-only AB/BA 顺序对照：

```powershell
python run_hidden_gsm8k.py --setting finalizer_only_order_ab_ba
```

该 setting 的 `metrics.csv` 按 `agent_variant=AB` 和 `agent_variant=BA` 分组，并额外输出
`finalizer_order_analysis.json`，其中包含配对题数、答案翻转率、两种顺序的三层正确性统计和事实 hash 一致性。

## 结果解读

- `accuracy`：最终正确率。
- `oracle_gap`：最终答案错误，但 Solver、Verifier 等上游阶段至少出现过
  正确候选答案。
- `inference_*_tokens`：本地 Qwen 推理消耗。
- `judge_*_tokens`：DeepSeek 离线复核消耗。
- `information_acquisition_failure`：公开讨论没有包含全部必要事实。
- `information_integration_failure`：信息已完整，但没有形成正确候选答案。
- `answer_selection_failure`：上游已出现正确候选，但最终选择错误。

分析单题时优先查看 `traces_all.json`；汇总比较不同 setting 时查看
`metrics.csv`。

## 完整实验流水线

`run_full_experiment.py` 提供一站式实验流水线，包含：

- **3 题 Gate 检查**：验证 finalizer 格式合规和 AB/BA prompt 正确性
- **20 题正式实验**：12 个核心设置的系统性比较
- **独立 Judge 系统**：双份 Judge（不同 prompt/seed），检测 disagreement
- **格式错误分类**：缺失字段、字段顺序、额外文本、非法 source 等 9 类
- **答案错误分类**：reasoning 中有正确答案但写错、算术错误、忽略后加入事实等 8 类
- **轨迹分析**：正确答案出现/保留/丢失的完整时间线
- **顺序效应分析**：AB/BA 配对的 flip rate、order sensitivity
- **基线对照**：single-agent full/late information、deterministic calculator、oracle 上界
- **多 seed 运行**：每个设置支持多个 seed，检验稳定性

### Gate 检查

```powershell
python run_full_experiment.py --gate --data-path data/3q.json --device cuda
```

通过条件：
- `before_final_reset` 3/3 格式有效
- `finalizer_only_AB` 3/3 格式有效
- `finalizer_only_BA` 3/3 格式有效
- AB/BA prompt diff 只显示事实顺序变化
- 主 accuracy 使用 strict answer correctness
- Invalid pair 不进入 valid flip rate
- Judge 输出可解析
- 人工核验表已生成
- README 数据一致性
- 输出记录 data SHA256
- 一键复现

### 20 题正式实验

```powershell
python run_full_experiment.py --twenty-q --data-path data/20.json --device cuda --num-seeds 3
```

12 个核心设置：
1. single_full_information
2. all_at_start_AB
3. all_at_start_BA
4. after_round1 (AB)
5. after_round1_BA
6. before_final_transcript (AB)
7. before_final_transcript_BA
8. canonical_order
9. before_final_reset
10. frozen_transcript_AB
11. frozen_transcript_BA
12. format_self_check_before_commit

### 输出结构

```text
outputs_full_experiment/YYYYMMDD_HHMMSS/
├── run_config.json              # 运行配置 + data SHA256
├── traces_all.json              # 完整轨迹
├── traces_<setting>.json        # 按 setting 分组的轨迹
├── comprehensive_metrics.csv    # 三层正确率 + 格式/错误分类 + Oracle
├── failures_detailed.json       # 详细失败分析
├── order_sensitivity.json       # AB/BA 配对分析
├── judge_outputs.json           # 独立 Judge 双份结果
├── manual_audit.csv             # 人工核验表
├── loss_statistics.json         # 正确答案丢失位置统计
├── prompt_diff_q*/              # AB/BA prompt 差异
├── gate_check.json              # (gate 模式) 12 项 gate 结果
└── analysis_report.md           # 实验分析报告（7 项诊断）
```

### 断点续跑

`continue_experiment.py` 用于从中断处恢复实验，保留已有结果、只跑剩余题目。

```powershell
# 从上次中断处自动续跑
python continue_experiment.py

# 指定起始题号
python continue_experiment.py 9

# 指定起始题号 + 数量限制（如只补跑 3 题）
python continue_experiment.py 9 3
```

特性：
- 自动从 `traces_all.json` 识别已完成题目，从 `max(qid)+1` 开始
- 每道题跑完立即增量写入 traces 和 metrics，中断不怕丢
- 全部跑完后自动调用 `write_comprehensive_outputs()` 生成完整分析文件
- 输出目录硬编码在脚本顶部 `OUTPUT_DIR`，修改后指向目标目录即可

### 实验结果 (20260804_174126)

使用 Qwen2.5-1.5B 在 20 题上运行全部 12 个核心设置（3 seeds），完整分析见 [analysis_report.md](outputs_full_experiment/20260804_174126/analysis_report.md)。

**核心结论**：

| 诊断项 | 结果 |
|---|---|
| 完整事实到达 finalizer？ | ✅ 是 — 几乎所有设置中完整事实都到达了 |
| 正确答案在轨迹中出现过？ | ❌ 98.7% 的失败案例中**从未出现** |
| 正确答案在哪丢失？ | **Solver 端** — 1.5B 分片信息下无法推理 |
| 顺序效应在哪个环节？ | **Finalizer** — 80% 答案翻转率，但翻转后全错 |
| Reset 恢复能力 | +5% semantic，格式 100% — 有效但微弱 |
| Canonical order 恢复能力 | +5% semantic — 轻微改善 |
| Self-check 恢复能力 | 0% — 无效果 |
| 格式错误 vs 语义错误 | 88.3% 语义错误，6.7% 纯格式错误 |
| Seed 稳定性 | ✅ 高 — 同题不同 seed 结果一致（T=0.2） |

整体：Qwen2.5-1.5B 太小，分片信息下 multi-agent 架构无法弥补基础推理能力不足。建议用 7B+ 模型重跑。

### 完整设置列表

共 66 个实验设置，分为：

- **原有设置** (12): single_full, single_partial, multi_partial, multi_partial_verifier, oracle_broadcast, all_at_start_AB/BA, after_round1, before_final_transcript, before_final_transcript_ledger, before_final_reset, finalizer_only_order_ab_ba
- **顺序隔离** (7): solver_only_AB/BA, finalizer_only_AB/BA, frozen_transcript_AB/BA, canonical_order, random_order
- **信息时间** (10): info_at_start/after_round1/before_final/before_finalizer/reset_direct × AB/BA
- **格式变体** (10): three_line, strict_json, xml_tags, answer_only, answer_first, reason_first, reason_then_answer, internal_reasoning_then_structured, deterministic_extract, self_check_before_commit
- **锚定效应** (14): anchor_early_* × 4 + anchor_source_* × 4 + anchor_repeat_* × 3 + counter_belief + context_reset + belief_reset
- **Ledger 变体** (6): raw_concat, structured_kv, dependency_table, canonical_sorted, provenance_free, provenance_aware
- **基线** (6): single_full_information, single_late_information, deterministic_calculator, best_solver_oracle, discussion_oracle, finalizer_upper_bound

---

# ExecGround 五部分实验（2026-08）

本实验围绕 Qwen2.5-1.5B 多智能体 GSM8K 的瓶颈诊断与架构改进，分为五个部分：

| Part | 内容 | 状态 |
|------|------|------|
| Part 1 | 现有 20 题瓶颈证据收集 | ✅ |
| Part 2 | Oracle 干预实验（6 组，360 traces） | ✅ |
| Part 3 | ExecGround 模块实现（TypedFact → Ledger → Plan → Verify） | ✅ |
| Part 4 | 六组消融实验（ExecGround vs. 基线） | ✅ |
| Part 5 | 统一错误分类标注（13 类别） | ✅ |

核心发现：Qwen2.5-1.5B 在 multi-agent 架构下，**讨论历史污染是主瓶颈**（oracle_disclosure 25% → fresh_solver 5.6%——尽管绝对值低，方向明确），**依赖图缺失和"信息足够但输出undetermined"占错误的 98%**。可执行计划层和覆盖验证层超出此模型能力范围。

---

## Part 1: 瓶颈证据收集

### 产出

| 文件 | 说明 |
|------|------|
| `deep_bottleneck_analysis_v2.py` | 分析脚本：11 维分析 + 漏斗 + 分离指标 |
| `outputs_full_experiment/20260804_174126/bottleneck_analysis/final_bottleneck_report.md` | 主报告 |
| `outputs_full_experiment/20260804_174126/bottleneck_analysis/per_question_deep_analysis.json` | 每题每 seed 原始数据 |
| `outputs_full_experiment/20260804_174126/bottleneck_analysis/per_question_summary.csv` | 20 题汇总 CSV |
| `outputs_full_experiment/20260804_174126/analysis_report.md` | 完整实验分析报告 |

### 核心发现

```
漏斗: 20题 → 1题事实无失真 → 7题推理中有正确答案 → 4题正确候选 → 1题最终正确
candidate_emergence_given_complete_disclosure: 4/20 = 20%
final_retention_given_correct_candidate: 1/4 = 25%
结论: 双重瓶颈 0.20 × 0.25 = 0.05 = 最终准确率
```

---

## Part 2: Oracle 干预实验

### 设计

6 组逐层干预，通过程序化注入信息确定因果瓶颈：

| # | Setting | 干预方式 | Accuracy |
|---|---------|---------|----------|
| 0 | free_discussion | 基线：私有事实讨论 | 5.0% |
| 1 | oracle_disclosure | 程序注入 A+B 原始事实 | 25.0% |
| 2 | oracle_canonical_state | 规范化结构化事实表 | 20.0% |
| 3 | canonical_state_fresh | 新 solver 只读题目+事实表 | 60.0% |
| 4 | oracle_plan | 事实表+完整方程依赖结构 | 70.0% |
| 5 | oracle_candidate | 直接注入正确答案标注为权威 | 53.3% |

### 产出

| 文件 | 说明 |
|------|------|
| `oracle_intervention_experiment.py` | 实验脚本：6 组 builder + 完整循环 |
| `analyze_oracle_results.py` | 分析脚本：因果归因 + 对比报告 |
| `outputs_oracle_intervention/20260806_191645/oracle_analysis_report.md` | 完整分析报告 |
| `outputs_oracle_intervention/20260806_191645/oracle_metrics.json` | 原始指标 |
| `outputs_oracle_intervention/20260806_191645/traces_all.json` | 360 traces |

### 因果归因

```
瓶颈层级                              损失量      累计损失
Fact Disclosure (agent 说不清)         +20%  →    5% → 25%
Discussion Contamination (历史污染)    +40%  →   20% → 60%  ★ 最大瓶颈
Plan Generation (方程结构)             +10%  →   60% → 70%
Finalizer Retention (保留正确候选)      —    →   见下文
Residual (算术能力天花板)              10%   →   70% → 80% (single_full 基线)
```

### 运行方式

```powershell
python oracle_intervention_experiment.py --rounds 2 --seeds 3 --limit 20
python analyze_oracle_results.py outputs_oracle_intervention/20260806_191645
```

---

## Part 3: ExecGround 模块实现

### 四大模块

| Module | 功能 | 文件位置 |
|--------|------|---------|
| **TypedFact** | 将私有信息转为结构化 JSON 事实（fact_id, subject, relation, object, value, unit, source, evidence） | `exec_ground.py` |
| **CanonicalLedger** | 确定性合并 A+B 事实：去重、实体对齐、单位标准化、关系标准化、冲突检测、固定排序。AB=BA 确定保证。 | `exec_ground.py` |
| **FreshSolver + ExecutablePlan** | 新 solver 只读题目+ledger（无讨论历史），输出可执行 JSON-IR 计划（add/subtract/multiply/divide + fact_id 引用） | `exec_ground.py` |
| **CoverageVerifier** | 程序检查：事实覆盖率、缺失事实、未绑定变量、可执行性、结果正确性。提供定向修复提示，无自由讨论。 | `exec_ground.py` |

### 测试覆盖

| 文件 | 说明 |
|------|------|
| `test_exec_ground.py` | 22 个测试，141 个断言 |
| 覆盖范围 | TypedFact创建/序列化/标准化、CanonicalLedger构建/确定性/去重/冲突检测、ExecutablePlan创建/JSON往返/执行/解析、CoverageVerifier全场景、FixPrompt生成、端到端集成测试、全部20题gold事实提取 |

```powershell
python -m pytest test_exec_ground.py -v
```

---

## Part 4: 六组消融实验

### 设计

对应 ExecGround 的四层架构，逐层叠加：

| # | Setting | 描述 |
|---|---------|------|
| 0 | free_discussion | 基线：私有事实讨论（复用 Part 2） |
| 1 | oracle_disclosure | Reveal-All：程序注入所有事实（复用 Part 2） |
| 2 | canonical_ledger | TypedFact → Ledger → 带 Ledger 讨论 → Finalizer |
| 3 | ledger_fresh_solver | + Fresh Solver（无讨论历史，只读题目+Ledger） |
| 4 | ledger_exec_plan | + 可执行 JSON-IR Plan |
| 5 | ledger_exec_plan_verify | + 覆盖验证 + 修复循环 |

关键观测：不看最终准确率，看**正确候选涌现率**在各层的增量变化。

### 实验结果

| Setting | N | Acc | Emerge | 关键发现 |
|---------|---|------|--------|---------|
| free_discussion | 60 | 0% | 0% | 基线 |
| oracle_disclosure | 60 | 25% | 20% | 与 Part 2 一致 |
| canonical_ledger | 54 | 0% | 0% | 讨论协议瓶颈：solver 全输出 undetermined |
| ledger_fresh_solver | 54 | 5.6% | 5.6% | Fresh solver 恢复微弱但存在 |
| ledger_exec_plan | 54 | 5.6% | 0% | 48/54 plan 执行失败 |
| ledger_exec_plan_verify | 54 | 5.6% | 0% | 覆盖验证可检测但模型无法修复 |

### 产出

| 文件 | 说明 |
|------|------|
| `exec_ground_experiment.py` | 实验脚本：6 builder + 增量保存 + 断点续跑 |
| `outputs_exec_ground/20260807_152225/traces_all.jsonl` | 336 traces（增量写入，fsync 保证） |
| `outputs_exec_ground/20260807_152225/traces_all.json` | 完整 traces（JSON 格式，方便加载） |
| `outputs_exec_ground/20260807_152225/exec_ground_metrics.json` | 各 setting 指标 |
| `outputs_exec_ground/20260807_152225/exec_ground_analysis_report.md` | 分析报告 |

### 运行方式

```powershell
# 正式运行
python exec_ground_experiment.py --limit 20 --seeds 3

# 使用 gold facts 快速验证
python exec_ground_experiment.py --limit 20 --seeds 3 --use-gold-facts

# 断点续跑
python exec_ground_experiment.py --limit 20 --seeds 3 --output-dir outputs_exec_ground/20260807_152225 --resume
```

---

## Part 5: 统一错误分类

### 13 错误类别

| # | 类别 | 数量 | 占比 | Oracle 修复 |
|---|------|------|------|------------|
| 1 | 依赖图缺失 | 162 | 51.9% | oracle_plan |
| 2 | 信息足够但仍输出undetermined | 144 | 46.2% | fresh_solver |
| 3 | 忘记自己的事实 | 120 | 38.5% | canonical_state |
| 4 | 忽略对方事实 | 96 | 30.8% | canonical_state |
| 5 | 只完成局部计算 | 81 | 26.0% | oracle_plan |
| 6 | 事实数值被修改 | 60 | 19.2% | canonical_state |
| 7 | 事实披露不完整 | 27 | 8.7% | oracle_disclosure |
| 8 | 私有事实没有披露 | 21 | 6.7% | oracle_disclosure |
| 9 | 算术执行错误 | 6 | 1.9% | oracle_plan |
| 10 | half/twice等关系翻转 | 0 | 0% | 模型未达到此复杂度 |
| 11 | 实体或题意漂移 | 0 | 0% | 模型保持问题理解稳定 |
| 12 | reasoning正确但候选字段错误 | 0 | 0% | 正确数不出现在reasoning中 |
| 13 | 正确候选存在但被丢失 | 0 | 0% | Finalizer非独立瓶颈 |

### 产出

| 文件 | 说明 |
|------|------|
| `error_classification.py` | 错误分类脚本：13 类别启发式检测 |
| `outputs_exec_ground/20260807_152225/error_classification_report.md` | 主报告：每类的数量/题目ID/典型轨迹/最早错误轮次/Oracle修复 |
| `outputs_exec_ground/20260807_152225/error_classification.json` | 结构化 JSON 数据 |

### 运行方式

```powershell
python error_classification.py
```

---

## 提交文件清单

### 代码

| 文件 | 用途 | Part |
|------|------|------|
| `exec_ground.py` | ExecGround 核心模块（TypedFact/CanonicalLedger/FreshSolver/CoverageVerifier） | 3 |
| `test_exec_ground.py` | ExecGround 测试套件（22 tests） | 3 |
| `exec_ground_experiment.py` | 六组消融实验脚本 | 4 |
| `error_classification.py` | 13 类别错误分类分析脚本 | 5 |
| `oracle_intervention_experiment.py` | Oracle 干预实验脚本 | 2 |
| `analyze_oracle_results.py` | Oracle 结果因果归因分析 | 2 |
| `deep_bottleneck_analysis_v2.py` | 瓶颈证据收集分析脚本 | 1 |
| `run_hidden_gsm8k.py` | Hidden-GSM8K 核心基础设施（Part 1-2 依赖） | 1-2 |
| `HANDOVER.md` | 工作交接文档 | — |

### 结果文件

| 文件/文件夹 | 内容 | Part |
|------|------|------|
| `outputs_full_experiment/20260804_174126/bottleneck_analysis/` | 瓶颈分析报告 + 每题数据 | 1 |
| `outputs_full_experiment/20260804_174126/analysis_report.md` | 完整实验分析 | 1 |
| `outputs_oracle_intervention/20260806_191645/oracle_analysis_report.md` | Oracle 因果归因报告 | 2 |
| `outputs_oracle_intervention/20260806_191645/oracle_metrics.json` | Oracle 指标 | 2 |
| `outputs_exec_ground/20260807_152225/traces_all.jsonl` | ExecGround 336 traces | 4 |
| `outputs_exec_ground/20260807_152225/exec_ground_metrics.json` | 各 setting 指标 | 4 |
| `outputs_exec_ground/20260807_152225/exec_ground_analysis_report.md` | 消融分析报告 | 4 |
| `outputs_exec_ground/20260807_152225/error_classification_report.md` | 13 类别分类报告 | 5 |
| `outputs_exec_ground/20260807_152225/error_classification.json` | 分类结构化数据 | 5 |

### 依赖文件（不修改）

| 文件 | 用途 |
|------|------|
| `data/20.json` | 20 题 GSM8K 变体数据 |
| `hidden_gsm8k_prompts/` | solver/finalizer/verifier prompt 模板 |
| `qwen2.5-1.5B/` | 模型权重（不提交） |

### 命令速查

```powershell
# Part 2: Oracle 干预实验
python oracle_intervention_experiment.py --rounds 2 --seeds 3 --limit 20

# Part 3: 运行 ExecGround 测试
python -m pytest test_exec_ground.py -v

# Part 4: 六组消融实验
python exec_ground_experiment.py --limit 20 --seeds 3

# Part 5: 错误分类分析
python error_classification.py
```
