# agent-demo
a multiple agents framework using local model
# Local Qwen GSM8K Runner

`run_qwen_local.py` 用本地 Qwen 模型运行 GSM8K-style 多智能体实验。它复用 `run.py` 里的数据读取、prompt、答案抽取、setting 定义和输出逻辑，但 agent 推理由本地 Qwen 完成；每个 agent 输出的对错和过程分数仍由 DeepSeek/OpenAI-compatible judge API 评估。

## 实验设置

支持三种设置：

- `single`：只运行 Solver A，再由 Finalizer 输出最终答案
- `multi`：Solver A 和 Solver B 独立作答，再由 Finalizer 汇总
- `multi_verifier`：Solver A/B 独立作答，Verifier 检查并给出建议答案，再交给 Finalizer

也可以使用别名：

- `single_agent`
- `multi_agent`
- `multi_agent_verifier`
- `all`

## 依赖

```bash
pip install -r requirements.txt
```

主要依赖：

- `torch`
- `transformers`
- `safetensors`
- `openai`
- `python-dotenv`

如果使用 CUDA，需安装匹配本机 CUDA 版本的 PyTorch。

## 模型与数据

默认本地模型路径：

```text
D:\agentdemo\multi_agent_gsm8k\qwen2.5-1.5B
```

模型目录至少需要包含：

- `config.json`
- `tokenizer_config.json`
- `tokenizer.json`
- `model.safetensors`

默认数据集：

```text
data/50q.jsonl
```

可以通过 `--model-path` 和 `--data-path` 覆盖默认路径。

## API 配置

虽然 agent 推理使用本地 Qwen，但 judge 仍需要 API。复制 `.env.example` 为 `.env`，并配置：

```env
API_KEY=your_api_key_here
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
```

代码也兼容以下环境变量名：

- API key：`DEEPSEEK_API_KEY`、`API_KEY`、`OPENAI_API_KEY`
- base URL：`JUDGE_BASE_URL`、`DEEPSEEK_BASE_URL`、`BASE_URL`、`OPENAI_BASE_URL`
- judge model：`JUDGE_MODEL`、`DEEPSEEK_MODEL`、`MODEL_NAME`、`OPENAI_MODEL`

## 检查配置

正式运行前建议先检查依赖、模型、数据和 judge 配置：

```bash
python run_qwen_local.py --check-config
```

指定 CPU 或自定义路径：

```bash
python run_qwen_local.py --check-config --device cpu --model-path D:\path\to\qwen --data-path data\50q.jsonl
```

## 运行命令

运行单个设置：

```bash
python run_qwen_local.py --setting single
python run_qwen_local.py --setting multi
python run_qwen_local.py --setting multi_verifier
```

一次运行全部设置：

```bash
python run_qwen_local.py --setting all
```

常用参数：

```bash
python run_qwen_local.py ^
  --setting multi_verifier ^
  --device cuda ^
  --max-new-tokens 512 ^
  --temperature 0.2 ^
  --data-path data\50q.jsonl ^
  --model-path D:\agentdemo\multi_agent_gsm8k\qwen2.5-1.5B
```

参数说明：

- `--setting`：选择实验设置；为空时会进入交互选择
- `--device`：`cuda`、`cuda:0` 或 `cpu`
- `--max-new-tokens`：每次本地生成的最大新 token 数，默认 `512`
- `--temperature`：本地 Qwen 采样温度，默认 `0.2`；设为 `0` 时使用确定性解码
- `--allow-download`：允许 Transformers 下载缺失模型文件；默认只读取本地文件

## 输出文件

每次运行会在 `outputs/` 下创建时间戳目录，例如：

```text
outputs/20260620_003621/
```

目录内包含：

- `traces_all.json`：每题完整轨迹，包括 solver、verifier、finalizer 输出和 judge 结果
- `metrics.csv`：各 setting 的正确率、token 总量和平均 token
- `failures.json`：最终答错的样例

`traces_all.json` 中常用字段：

- `solver_a_answer`、`solver_b_answer`
- `solver_a_correct`、`solver_b_correct`
- `verifier_output`
- `verifier_chosen_solver`
- `verifier_gold_grade`
- `final_prediction`
- `finalizer_gold_grade`
- `finalizer_followed_verifier`
- `correctness_changes`
- `correct`

## 结果解读

`metrics.csv` 是总体对比入口：

- `accuracy`：最终正确率
- `correct`：正确题数
- `total_tokens`：本地模型调用和 judge 调用的 token 总量
- `avg_total_tokens`：每题平均 token

如果要分析某题为什么错，优先看 `traces_all.json`：

1. 看 `solver_a_correct` 和 `solver_b_correct`，判断初答谁错。
2. 看 `verifier_output.verified_answer` 和 `verifier_gold_grade.correct`，判断 verifier 是否发现错误。
3. 看 `finalizer_followed_verifier`，判断 finalizer 是否跟随 verifier。
4. 看 `correctness_changes`，判断从初答到终答是否被纠正或带偏。

## 注意事项

- 本地 Qwen 只负责生成答案；最终对错以 judge API 的评估结果为准。
- `multi_verifier` 的 token 成本最高，因为额外调用 verifier，并且 judge 也会评估 verifier 输出。
- `run_qwen_local.py` 会缓存同一题的 Solver A/B 输出；运行 `all` 时，不同 setting 可复用 solver 结果，减少重复本地推理。
- PowerShell 中文显示乱码通常是终端编码问题，输出文件本身按 UTF-8 写入。
