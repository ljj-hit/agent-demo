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
