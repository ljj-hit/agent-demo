import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT_FALLBACK = Path(r"D:\agentdemo\multi_agent_gsm8k")
DEFAULT_DATA_PATH = ROOT / "data" / "50q.jsonl"
DEFAULT_MODEL_PATH = Path(r"D:\agentdemo\multi_agent_gsm8k\qwen2.5-1.5B")
DEFAULT_JUDGE_BASE_URL = "https://api.deepseek.com"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
API_KEY_NAMES = ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY")


def make_timestamped_output_dir(base_dir: Path, run_started_at: datetime) -> Path:
    base_dir.mkdir(exist_ok=True)
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    suffix = 2
    while output_dir.exists():
        output_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir()
    return output_dir


def load_benchmark_module():
    import run as local_run

    if hasattr(local_run, "resolve_data_path"):
        return local_run

    fallback_run = BENCHMARK_ROOT_FALLBACK / "run.py"
    if not fallback_run.exists():
        return local_run

    spec = importlib.util.spec_from_file_location("agentdemo_benchmark_run", fallback_run)
    if spec is None or spec.loader is None:
        return local_run
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark_module()
SETTINGS = benchmark.SETTINGS
SETTING_NAMES = benchmark.SETTING_NAMES
SETTING_ALIASES = benchmark.SETTING_ALIASES


def dependency_status() -> dict[str, bool]:
    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
    }


def load_ml_dependencies():
    missing = [name for name, present in dependency_status().items() if not present]
    if missing:
        raise SystemExit(
            "Missing Python package(s): "
            + ", ".join(missing)
            + ". Install them in this environment before running local Qwen inference."
        )

    import torch
    import transformers.utils.import_utils as transformers_import_utils

    transformers_import_utils._sklearn_available = False
    transformers_import_utils._scipy_available = False

    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def validate_model_path(model_path: Path) -> None:
    required_files = ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")
    missing = [name for name in required_files if not (model_path / name).exists()]
    if missing:
        raise SystemExit(
            f"Model directory is missing file(s): {', '.join(missing)}\n"
            f"Checked: {model_path}"
        )


def choose_cuda_dtype(torch):
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


class LocalQwenModel:
    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = load_ml_dependencies()
        validate_model_path(model_path)

        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.torch = torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Install CUDA-enabled PyTorch or pass --device cpu.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch_dtype = choose_cuda_dtype(torch) if device.startswith("cuda") else torch.float32
        print(f"dtype: {torch_dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0, input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = int(generated_ids.shape[-1])
        usage = {
            "prompt_tokens": int(input_len),
            "completion_tokens": completion_tokens,
            "total_tokens": int(input_len + completion_tokens),
        }
        return text, usage


def load_api_dependencies():
    if importlib.util.find_spec("dotenv") is None:
        raise SystemExit("Missing Python package: python-dotenv. Install dependencies first.")
    if importlib.util.find_spec("openai") is None:
        raise SystemExit("Missing Python package: openai. Install dependencies first.")

    from dotenv import load_dotenv
    from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

    return load_dotenv, OpenAI, (APIConnectionError, APITimeoutError, RateLimitError)


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def masked(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def clamp_score(value: object, default: Decimal = Decimal("0")) -> Decimal:
    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError):
        score = default
    return max(Decimal("0"), min(Decimal("10"), score))


def format_score(score: Decimal) -> float:
    return float(score.quantize(Decimal("0.01")))


def distance_grade_against_gold(answer: object | None, gold_answer: str) -> dict:
    extracted_answer = benchmark.extract_answer(answer)
    correct = benchmark.is_correct(extracted_answer, gold_answer)
    pred_num = benchmark.to_decimal(extracted_answer)
    gold_num = benchmark.to_decimal(gold_answer)

    absolute_error = None
    relative_error = None
    if correct:
        score = Decimal("10")
    elif pred_num is None or gold_num is None:
        score = Decimal("0")
    else:
        absolute_error = abs(pred_num - gold_num)
        denominator = abs(gold_num)
        if denominator == 0:
            score = max(Decimal("0"), Decimal("10") - absolute_error)
        else:
            relative_error = absolute_error / denominator
            score = Decimal("10") * max(Decimal("0"), Decimal("1") - relative_error)

    return {
        "answer": extracted_answer,
        "gold_answer": gold_answer,
        "correct": correct,
        "score": format_score(score),
        "score_scale": "0-10",
        "absolute_error": str(absolute_error) if absolute_error is not None else None,
        "relative_error": str(relative_error) if relative_error is not None else None,
    }


def parse_judge_response(text: str, fallback_answer: str, gold_answer: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}

    raw_correct = data.get("correct", False)
    if isinstance(raw_correct, str):
        correct = raw_correct.strip().lower() in {"true", "yes", "1", "correct"}
    else:
        correct = bool(raw_correct)
    judged_answer = str(data.get("normalized_prediction") or fallback_answer).strip()
    fallback_grade = distance_grade_against_gold(judged_answer, gold_answer)
    result_score = Decimal("10") if correct else clamp_score(
        data.get("result_score", data.get("score")),
        Decimal(str(fallback_grade["score"])),
    )
    process_score = clamp_score(data.get("process_score"), Decimal("0"))
    score = clamp_score(
        data.get("score"),
        (result_score * Decimal("0.7")) + (process_score * Decimal("0.3")),
    )
    return {
        "answer": judged_answer,
        "gold_answer": gold_answer,
        "correct": correct,
        "score": format_score(score),
        "score_scale": "0-10",
        "result_score": format_score(result_score),
        "process_score": format_score(process_score),
        "absolute_error": data.get("absolute_error", fallback_grade["absolute_error"]),
        "relative_error": data.get("relative_error", fallback_grade["relative_error"]),
        "judge_reason": str(data.get("reason", data.get("feedback", ""))).strip(),
    }


def deepseek_grade_agent_output(
    judge_client,
    retryable_errors: tuple[type[Exception], ...],
    judge_model: str,
    agent_name: str,
    question: str,
    agent_output: str,
    gold_answer: str,
    context: str = "",
) -> tuple[dict, dict]:
    fallback_answer = benchmark.extract_answer(agent_output)
    system_prompt = (
        "You are a strict but fair evaluator for GSM8K-style multi-agent math reasoning. "
        "Evaluate both the agent's reasoning process and its final result against the gold answer. "
        "Mark correct when the mathematical value is equivalent even if formatting differs, "
        "for example fractions vs decimals, commas, currency signs, units, or explanatory text. "
        "The result_score must be based on distance from the gold answer: 10 means equivalent, "
        "and the result_score must decrease as the numeric gap grows. "
        "The process_score must judge whether the setup, arithmetic, reasoning steps, and use of "
        "available evidence are sound. Penalize unsupported guesses, copied wrong answers, missing "
        "steps, arithmetic mistakes, and contradictions. The overall score should combine process "
        "quality and result quality on a 0-10 scale. "
        "Return only valid JSON."
    )
    user_prompt = (
        "Agent being evaluated:\n"
        f"{agent_name}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Gold answer:\n"
        f"{gold_answer}\n\n"
        "Additional context available to this agent, if any:\n"
        f"{context or '<none>'}\n\n"
        "Agent output to evaluate:\n"
        f"{agent_output}\n\n"
        "Extracted prediction:\n"
        f"{fallback_answer}\n\n"
        "Return this JSON schema exactly:\n"
        "{\n"
        '  "correct": true,\n'
        '  "result_score": 10,\n'
        '  "process_score": 10,\n'
        '  "score": 10,\n'
        '  "absolute_error": "0",\n'
        '  "relative_error": "0",\n'
        '  "normalized_prediction": "...",\n'
        '  "normalized_gold": "...",\n'
        '  "reason": "brief explanation of process and result scoring"\n'
        "}"
    )
    max_attempts = int(env_value("JUDGE_MAX_ATTEMPTS", "MODEL_MAX_ATTEMPTS", default="4"))
    for attempt in range(1, max_attempts + 1):
        try:
            resp = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            return parse_judge_response(text, fallback_answer, gold_answer), benchmark.usage_dict(resp)
        except retryable_errors:
            if attempt == max_attempts:
                raise
            delay_seconds = min(2 ** (attempt - 1), 10)
            print(f"DeepSeek judge call failed; retrying in {delay_seconds}s ({attempt}/{max_attempts})")
            time.sleep(delay_seconds)

    raise RuntimeError("DeepSeek judge call failed unexpectedly.")


def cached_local_call(
    cache: dict | None,
    cache_key: tuple | None,
    local_model: LocalQwenModel,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict]:
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]
    result = local_model.generate(system_prompt, user_prompt)
    if cache is not None and cache_key is not None:
        cache[cache_key] = result
    return result


def run_one_local(
    local_model: LocalQwenModel,
    judge_client,
    judge_retryable_errors: tuple[type[Exception], ...],
    judge_model: str,
    prompts: dict,
    example: dict,
    setting: str,
    example_index: int | None = None,
    model_cache: dict | None = None,
) -> dict:
    question = example["question"]
    gold_answer = example["gold_answer"]
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    example_key = (example_index, question) if example_index is not None else question

    solver_a_output, usage = cached_local_call(
        model_cache,
        ("solver_a", example_key),
        local_model,
        prompts["solver_a"],
        question,
    )
    benchmark.add_usage(token_usage, usage)
    solver_a_grade, usage = deepseek_grade_agent_output(
        judge_client,
        judge_retryable_errors,
        judge_model,
        "Solver A",
        question,
        solver_a_output,
        gold_answer,
    )
    benchmark.add_usage(token_usage, usage)

    solver_b_output = ""
    solver_b_grade = None
    verifier_text = ""
    verifier_output = {}
    verifier_gold_grade = None

    if setting in {"multi", "multi_verifier"}:
        solver_b_output, usage = cached_local_call(
            model_cache,
            ("solver_b", example_key),
            local_model,
            prompts["solver_b"],
            question,
        )
        benchmark.add_usage(token_usage, usage)
        solver_b_grade, usage = deepseek_grade_agent_output(
            judge_client,
            judge_retryable_errors,
            judge_model,
            "Solver B",
            question,
            solver_b_output,
            gold_answer,
        )
        benchmark.add_usage(token_usage, usage)

    if setting == "single":
        final_input = f"Question:\n{question}\n\nSolver A output:\n{solver_a_output}"
    elif setting == "multi":
        final_input = (
            f"Question:\n{question}\n\n"
            f"Solver A output:\n{solver_a_output}\n\n"
            f"Solver B output:\n{solver_b_output}"
        )
    else:
        verifier_input = (
            f"Question:\n{question}\n\n"
            f"Gold answer:\n{gold_answer}\n\n"
            f"Solver A output:\n{solver_a_output}\n\n"
            f"Solver B output:\n{solver_b_output}"
        )
        verifier_text, usage = local_model.generate(prompts["verifier"], verifier_input)
        benchmark.add_usage(token_usage, usage)
        verifier_output = benchmark.parse_verifier(verifier_text)
        verifier_gold_grade, usage = deepseek_grade_agent_output(
            judge_client,
            judge_retryable_errors,
            judge_model,
            "Verifier",
            question,
            verifier_text,
            gold_answer,
            context=(
                f"Solver A output:\n{solver_a_output}\n\n"
                f"Solver B output:\n{solver_b_output}"
            ),
        )
        benchmark.add_usage(token_usage, usage)
        final_input = (
            f"Question:\n{question}\n\n"
            f"Solver A output:\n{solver_a_output}\n\n"
            f"Solver B output:\n{solver_b_output}\n\n"
            f"Verifier JSON:\n{json.dumps(verifier_output, ensure_ascii=False)}"
        )

    final_output, usage = local_model.generate(prompts["finalizer"], final_input)
    benchmark.add_usage(token_usage, usage)
    final_prediction = benchmark.extract_answer(final_output)
    finalizer_gold_grade, usage = deepseek_grade_agent_output(
        judge_client,
        judge_retryable_errors,
        judge_model,
        "Finalizer",
        question,
        final_output,
        gold_answer,
        context=final_input,
    )
    benchmark.add_usage(token_usage, usage)
    correct = finalizer_gold_grade["correct"]
    verifier_chosen_solver = verifier_output.get("chosen_solver") if verifier_output else None
    finalizer_followed_verifier = (
        benchmark.answers_match(final_prediction, verifier_output.get("verified_answer"))
        if verifier_output
        else None
    )
    correctness_changes = {
        "solver_a": {
            "initial_correct": solver_a_grade["correct"],
            "final_correct": correct,
            "changed": solver_a_grade["correct"] != correct,
        }
    }
    if solver_b_grade is not None:
        correctness_changes["solver_b"] = {
            "initial_correct": solver_b_grade["correct"],
            "final_correct": correct,
            "changed": solver_b_grade["correct"] != correct,
        }
    if verifier_gold_grade is not None:
        correctness_changes["verifier"] = {
            "initial_correct": verifier_gold_grade["correct"],
            "final_correct": correct,
            "changed": verifier_gold_grade["correct"] != correct,
        }

    return {
        "question": question,
        "example_index": example_index,
        "gold_answer": gold_answer,
        "setting": setting,
        "setting_name": SETTING_NAMES[setting],
        "solver_a_output": solver_a_output,
        "solver_b_output": solver_b_output,
        "solver_a_answer": solver_a_grade["answer"],
        "solver_b_answer": solver_b_grade["answer"] if solver_b_grade else "",
        "solver_a_correct": solver_a_grade["correct"],
        "solver_b_correct": solver_b_grade["correct"] if solver_b_grade else None,
        "solver_a_score": solver_a_grade["score"],
        "solver_b_score": solver_b_grade["score"] if solver_b_grade else None,
        "solver_a_process_score": solver_a_grade["process_score"],
        "solver_b_process_score": solver_b_grade["process_score"] if solver_b_grade else None,
        "solver_a_result_score": solver_a_grade["result_score"],
        "solver_b_result_score": solver_b_grade["result_score"] if solver_b_grade else None,
        "solver_grades": {
            "A": solver_a_grade,
            "B": solver_b_grade,
        },
        "verifier_text": verifier_text,
        "verifier_output": verifier_output,
        "verifier_chosen_solver": verifier_chosen_solver,
        "verifier_gold_grade": verifier_gold_grade,
        "verifier_score": verifier_gold_grade["score"] if verifier_gold_grade else None,
        "verifier_process_score": verifier_gold_grade["process_score"] if verifier_gold_grade else None,
        "verifier_result_score": verifier_gold_grade["result_score"] if verifier_gold_grade else None,
        "final_prediction": final_prediction,
        "finalizer_output": final_output,
        "finalizer_gold_grade": finalizer_gold_grade,
        "finalizer_score": finalizer_gold_grade["score"],
        "finalizer_process_score": finalizer_gold_grade["process_score"],
        "finalizer_result_score": finalizer_gold_grade["result_score"],
        "finalizer_followed_verifier": finalizer_followed_verifier,
        "correctness_changes": correctness_changes,
        "correct": correct,
        "score": finalizer_gold_grade["score"],
        "score_scale": finalizer_gold_grade["score_scale"],
        "token_usage": token_usage,
    }


def print_gpu_info(device: str) -> None:
    torch, _, _ = load_ml_dependencies()
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"gpu: {torch.cuda.get_device_name(index)}")
        print(f"total vram: {props.total_memory / 1024**3:.2f} GB")


def load_judge_client():
    load_dotenv, OpenAI, retryable_errors = load_api_dependencies()
    env_path = ROOT / ".env"
    load_dotenv(env_path, override=True)
    api_key = env_value(*API_KEY_NAMES)
    base_url = env_value("JUDGE_BASE_URL", "DEEPSEEK_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", default=DEFAULT_JUDGE_BASE_URL)
    model = env_value("JUDGE_MODEL", "DEEPSEEK_MODEL", "MODEL_NAME", "OPENAI_MODEL", default=DEFAULT_JUDGE_MODEL)
    if not api_key:
        raise SystemExit(
            "Missing DeepSeek judge API key. Add API_KEY or DEEPSEEK_API_KEY to .env/environment."
        )
    return OpenAI(api_key=api_key, base_url=base_url), retryable_errors, model, base_url, api_key


def main() -> None:
    run_started_at = datetime.now()
    parser = argparse.ArgumentParser(
        description="Run the GSM8K multi-agent demo with local Qwen for all agent inference."
    )
    parser.add_argument(
        "--setting",
        choices=list(SETTING_ALIASES),
        default="",
        help="single/single_agent, multi/multi_agent, multi_verifier/multi_agent_verifier, or all.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate local model, prompts, and data without running inference.",
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Dataset path. Supports the same JSON/JSONL formats as run.py.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the local Qwen Hugging Face model directory.",
    )
    parser.add_argument("--device", default="cuda", help="Use cuda, cuda:0, or cpu.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to download missing model files. Default is local files only.",
    )
    args = parser.parse_args()

    if args.setting:
        selected_setting = SETTING_ALIASES[args.setting]
    elif args.check_config:
        selected_setting = "all"
    else:
        selected_setting = benchmark.choose_setting_interactively()

    settings = SETTINGS if selected_setting == "all" else [selected_setting]
    data_path = benchmark.resolve_data_path(args.data_path)
    model_path = Path(args.model_path).resolve()

    prompts = {
        name: benchmark.read_text(benchmark.PROMPT_DIR / f"{name}.txt")
        for name in benchmark.prompt_names_for_settings(settings)
    }
    try:
        examples = benchmark.load_examples(data_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Failed to load dataset: {exc}") from exc

    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    judge_client, judge_retryable_errors, judge_model, judge_base_url, judge_api_key = load_judge_client()

    if args.check_config:
        print("Configuration OK")
        deps = dependency_status()
        print(f"torch: {'found' if deps['torch'] else 'missing'}")
        print(f"transformers: {'found' if deps['transformers'] else 'missing'}")
        if all(deps.values()):
            print_gpu_info(args.device)
        print(f"model_path: {model_path}")
        print(f"device: {args.device}")
        print(f"data_path: {data_path}")
        print(f"examples: {len(examples)}")
        print(f"settings: {', '.join(SETTING_NAMES[s] for s in settings)}")
        print(f"judge_base_url: {judge_base_url}")
        print(f"judge_model: {judge_model}")
        print(f"judge_api_key: {masked(judge_api_key)}")
        print(f"outputs: {benchmark.OUTPUT_DIR}")
        return

    print_gpu_info(args.device)
    print(f"Loading local model from {model_path}")
    local_model = LocalQwenModel(
        model_path=model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        local_files_only=not args.allow_download,
    )

    model_cache = {}
    traces = []
    output_dir = make_timestamped_output_dir(benchmark.OUTPUT_DIR, run_started_at)
    print(f"Writing incremental outputs to {output_dir}")
    benchmark.write_outputs(traces, output_dir=output_dir, print_table=False)

    started_at = time.time()
    for setting in settings:
        for idx, example in enumerate(examples, start=1):
            print(f"[{SETTING_NAMES[setting]}] {idx}/{len(examples)}")
            traces.append(
                run_one_local(
                    local_model,
                    judge_client,
                    judge_retryable_errors,
                    judge_model,
                    prompts,
                    example,
                    setting,
                    example_index=idx,
                    model_cache=model_cache,
                )
            )
            benchmark.write_outputs(traces, output_dir=output_dir, print_table=False)

    benchmark.write_outputs(traces, output_dir=output_dir)
    print(f"Done in {time.time() - started_at:.1f}s. Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
