# 本地 Qwen 多智能体 GSM8K 实验

`run_qwen_local.py` 使用本地 Hugging Face Qwen 模型完成 Solver、Verifier 和 Finalizer 推理，并通过 OpenAI-compatible API 调用 DeepSeek 模型判断各阶段答案是否正确。

## 运行流程

脚本提供六种实验模式：

- `single`：Solver A 作答，Finalizer 生成最终答案。
- `multi`：Solver A、Solver B 独立作答，Finalizer 汇总。
- `multi_verifier`：在两个 Solver 后增加 Verifier，再由 Finalizer 作答。
- `multi_verifier_forced`：直接采用基于 Verifier 评分得到的答案，不再调用 Finalizer 生成答案。
- `multi_candidate_memory`：将 Solver 和 Verifier 的答案整理为候选表，要求 Finalizer 只能从候选答案中选择；若两次选择均无效，则按脚本规则回退到候选答案。
- `multi_ask_before_finalize`：生成候选表后，再询问两个 Solver 是否有最终异议，并把异议交给 Finalizer 参考。

不传 `--setting` 时，脚本会在终端中交互式提示选择一种模式。使用 `--setting all` 会依次运行全部模式。同一道题的 Solver A、Solver B 本地推理结果会在不同模式间缓存复用。

## 依赖

运行需要以下 Python 包：

```text
torch
transformers
python-dotenv
openai
```

默认使用 CUDA；如需使用 CPU，请传入 `--device cpu`。CUDA 模式下，计算能力 8.0 及以上使用 `bfloat16`，其他可用 CUDA 设备使用 `float16`；CPU 使用 `float32`。

## 本地模型

默认模型目录为：

```text
D:\agentdemo\multi_agent_gsm8k\qwen2.5-1.5B
```

可通过 `--model-path` 指定其他目录。脚本运行推理前会检查模型目录中是否存在：

```text
config.json
tokenizer_config.json
tokenizer.json
model.safetensors
```

默认仅加载本地文件。传入 `--allow-download` 后，Transformers 可以下载缺失的模型文件。

## Judge API 配置

脚本会读取项目根目录的 `.env`，并覆盖当前进程中的同名环境变量。至少需要配置一个 API Key：

```env
DEEPSEEK_API_KEY=your_api_key
JUDGE_BASE_URL=https://api.deepseek.com
JUDGE_MODEL=deepseek-v4-flash
JUDGE_MAX_ATTEMPTS=4
```

API Key 依次读取 `DEEPSEEK_API_KEY`、`API_KEY`、`OPENAI_API_KEY`。地址依次读取 `JUDGE_BASE_URL`、`DEEPSEEK_BASE_URL`、`BASE_URL`、`OPENAI_BASE_URL`；模型名依次读取 `JUDGE_MODEL`、`DEEPSEEK_MODEL`、`MODEL_NAME`、`OPENAI_MODEL`。

Judge 请求失败时会进行指数退避重试，默认最多 4 次，等待时间最大为 10 秒。

## 数据和 Prompt

默认数据文件为 `data/50.jsonl`。`--data-path` 支持 JSON 数组和 JSONL；每条记录必须包含：

```json
{"question": "题目", "answer": "解答过程 #### 最终答案"}
```

如果 `answer` 中包含 `####`，脚本取最后一个 `####` 后的内容作为标准答案。

Prompt 从 `prompts/` 读取。按所选模式使用以下文件：

- `solver_a.txt`
- `solver_b.txt`（多智能体模式）
- `verifier.txt`（含 Verifier 的模式）
- `finalizer.txt`

## 运行命令

检查配置：

```powershell
python run_qwen_local.py --check-config
```

运行单个模式或全部模式：

```powershell
python run_qwen_local.py --setting single
python run_qwen_local.py --setting multi
python run_qwen_local.py --setting multi_verifier
python run_qwen_local.py --setting multi_verifier_forced
python run_qwen_local.py --setting multi_candidate_memory
python run_qwen_local.py --setting multi_ask_before_finalize
python run_qwen_local.py --setting all
```

也支持 `single_agent`、`multi_agent`、`multi_agent_verifier` 三个别名。

常用参数：

```text
--data-path PATH          数据集路径，默认 data/50.jsonl
--model-path PATH         本地 Hugging Face 模型目录
--device DEVICE           默认 cuda，也可使用 cuda:0 或 cpu
--max-new-tokens N        默认 512
--temperature FLOAT       默认 0.2；设为 0 时关闭采样
--allow-download          允许 Transformers 下载缺失文件
```

例如：

```powershell
python run_qwen_local.py --setting all --model-path D:\models\qwen2.5-1.5B --device cuda:0 --temperature 0
```

## 输出

每次推理运行会创建独立目录：

```text
outputs/YYYYMMDD_HHMMSS/
```

若同一秒内目录重名，会追加 `_02`、`_03` 等后缀。每完成一道题，脚本都会增量更新：

- `traces_all.json`：每道题的各阶段原始输出、抽取答案、Judge 结果、Verifier 决策、候选表、最终答案、正确性变化、oracle gap 和 token 用量。
- `metrics.csv`：各模式的题数、正确数、准确率、oracle gap 统计和 token 统计。
- `failures.json`：最终答案错误的样例及失败分类。

`metrics.csv` 包含以下字段：

```text
setting, setting_name, num_examples, correct, accuracy,
oracle_gap_count, oracle_gap_rate, oracle_gap_question_ids,
prompt_tokens, completion_tokens, total_tokens, avg_total_tokens
```

这里的 `oracle_gap` 表示最终答案错误，但 Solver A、Solver B 或 Verifier 中至少有一个阶段已得到正确答案。Token 统计同时包含本地模型调用和 Judge API 调用。
