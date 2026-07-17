import argparse
import csv
import importlib.util
import io
import json
import math
import os
import re
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "50.jsonl"
PROMPT_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_MODEL_PATH = Path(r"D:\agentdemo\multi_agent_gsm8k\qwen2.5-1.5B")
DEFAULT_JUDGE_BASE_URL = "https://api.deepseek.com"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
API_KEY_NAMES = ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY")
SETTINGS = [
    "single", "multi", "multi_verifier", "multi_verifier_forced",
    "multi_candidate_memory", "multi_ask_before_finalize",
]
SETTING_NAMES = {
    "single": "Single Agent",
    "multi": "Multi-Agent",
    "multi_verifier": "Multi-Agent + Verifier",
    "multi_verifier_forced": "Multi-Agent + Forced Verifier",
    "multi_candidate_memory": "Multi-Agent + Candidate Memory",
    "multi_ask_before_finalize": "Multi-Agent + Ask Before Finalize",
}
SETTING_ALIASES = {
    "single": "single", "single_agent": "single",
    "multi": "multi", "multi_agent": "multi",
    "multi_verifier": "multi_verifier",
    "multi_agent_verifier": "multi_verifier", "all": "all",
    "multi_verifier_forced": "multi_verifier_forced",
    "multi_candidate_memory": "multi_candidate_memory",
    "multi_ask_before_finalize": "multi_ask_before_finalize",
}

MULTI_SETTINGS = set(SETTINGS) - {"single"}
VERIFIER_SETTINGS = set(SETTINGS) - {"single", "multi"}


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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def resolve_data_path(data_path: str) -> Path:
    path = Path(data_path)
    return path if path.is_absolute() else ROOT / path


def strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        return "\n".join(lines).strip()
    return stripped


def iter_dataset_items(data_path: Path) -> list[tuple[int, dict]]:
    text = strip_markdown_json_fence(data_path.read_text(encoding="utf-8"))
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{data_path} is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(f"{data_path} must contain a JSON array or JSONL records.")
        return list(enumerate(data, start=1))
    items = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append((line_no, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{data_path}:{line_no} is not valid JSON: {exc}") from exc
    return items


def load_examples(data_path: Path) -> list[dict]:
    examples = []
    for item_no, item in iter_dataset_items(data_path):
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise ValueError(f"{data_path}:{item_no} must contain 'question' and 'answer'.")
        answer = str(item["answer"]).strip()
        examples.append({"question": str(item["question"]).strip(), "answer": answer,
                         "gold_answer": extract_gold(answer)})
    if not examples:
        raise ValueError(f"{data_path} does not contain any examples.")
    return examples


def extract_gold(answer: str) -> str:
    return answer.split("####")[-1].strip() if "####" in answer else extract_answer(answer)


def extract_answer(text: object | None) -> str:
    if not text:
        return ""
    text = str(text)
    stripped = text.strip()
    match = re.search(r"Final Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if re.fullmatch(r"\d{1,2}:\d{2}", stripped):
        return stripped
    multi = r"-?\d+(?:\.\d+)?(?:\s*(?:,|and|x|by)\s*-?\d+(?:\.\d+)?)+"
    if re.fullmatch(multi, stripped, flags=re.IGNORECASE):
        return stripped
    fractions = re.findall(r"-?\d+\s*/\s*-?\d+", text)
    if fractions:
        return fractions[-1].replace(" ", "")
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else stripped


def to_decimal(value: object | None) -> Decimal | None:
    if value is None:
        return None
    cleaned = str(value).replace(",", "").strip()
    fraction = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", cleaned)
    if fraction:
        numerator, denominator = map(Decimal, fraction.groups())
        return None if denominator == 0 else numerator / denominator
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def normalize_answer(value: object | None) -> str:
    answer = extract_answer(value)
    decimal_value = to_decimal(answer)
    if decimal_value is not None:
        return str(decimal_value.normalize())
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", " "))
    if numbers:
        return ",".join(str(to_decimal(n) or n) for n in numbers)
    return re.sub(r"\s+", " ", answer).strip().lower()


def is_correct(prediction: object, gold: object) -> bool:
    prediction = extract_answer(prediction)
    pred_num, gold_num = to_decimal(prediction), to_decimal(gold)
    return (normalize_answer(prediction) == normalize_answer(gold)
            if pred_num is None or gold_num is None else pred_num == gold_num)


def answers_match(left: object | None, right: object | None) -> bool:
    return bool(left and right) and normalize_answer(left) == normalize_answer(right)


def usage_dict(resp) -> dict:
    usage = getattr(resp, "usage", None)
    return {key: int(getattr(usage, key, 0) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def add_usage(total: dict, part: dict) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] += part.get(key, 0)


def prompt_names_for_settings(settings: list[str]) -> set[str]:
    names = {"solver_a", "finalizer"}
    if any(s in MULTI_SETTINGS for s in settings):
        names.add("solver_b")
    if any(s in VERIFIER_SETTINGS for s in settings):
        names.add("verifier")
    return names


def bounded_number(value: object, minimum: float, maximum: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(max(minimum, min(maximum, number)), 4)


def parse_verifier(text: str) -> dict:
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(data, dict):
        data = {}

    raw_scores = data.get("solver_scores")
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    score_a = bounded_number(raw_scores.get("A", raw_scores.get("a")), 0, 10)
    score_b = bounded_number(raw_scores.get("B", raw_scores.get("b")), 0, 10)
    scores = {"A": 0 if score_a is None else score_a, "B": 0 if score_b is None else score_b}
    chosen = str(data.get("chosen_solver", "uncertain")).strip().lower()
    chosen = {"a": "A", "b": "B", "corrected": "corrected", "uncertain": "uncertain"}.get(
        chosen, "uncertain"
    )
    verified_answer = str(data.get("verified_answer", "")).strip()
    if not data:
        verified_answer = extract_answer(text)
    critique = str(data.get("critique", "")).strip()
    if not critique and not data:
        critique = "Verifier did not return valid JSON; extracted an answer from its text."
    return {
        "verified_answer": verified_answer,
        "chosen_solver": chosen,
        "solver_scores": scores,
        "confidence": bounded_number(data.get("confidence"), 0, 1) or 0,
        "critique": critique,
    }


def resolve_verifier_decision(
    verifier_output: dict,
    solver_a_output: str,
    solver_b_output: str,
) -> dict:
    """Turn the verifier's rubric scores into one explicit, usable decision."""
    scores = verifier_output.get("solver_scores", {})
    chosen = verifier_output.get("chosen_solver", "uncertain")
    score_a, score_b = scores.get("A"), scores.get("B")
    if score_a is not None and score_b is not None:
        preferred = "A" if score_a > score_b else "B" if score_b > score_a else "tie"
    else:
        preferred = "unknown"
    effective_choice = chosen
    adjustment = ""

    # `corrected` is intentionally not replaced: the prompt allows the verifier to
    # reject both solvers and derive its own answer. For A/B, unequal scores are the
    # objective tie-breaker required by the verifier rubric.
    if chosen in {"A", "B", "uncertain"} and preferred in {"A", "B"}:
        effective_choice = preferred
        if chosen != preferred:
            adjustment = (
                f"chosen_solver={chosen!r} conflicted with scores "
                f"A={scores.get('A')}, B={scores.get('B')}; used {preferred}."
            )

    solver_answers = {
        "A": extract_answer(solver_a_output),
        "B": extract_answer(solver_b_output),
    }
    verified_answer = str(verifier_output.get("verified_answer", "")).strip()
    if effective_choice in solver_answers:
        selected_answer = solver_answers[effective_choice]
        # Prefer the verifier's normalized form when it represents the same value.
        if verified_answer and answers_match(verified_answer, selected_answer):
            selected_answer = verified_answer
    else:
        selected_answer = verified_answer

    return {
        "effective_choice": effective_choice,
        "selected_answer": selected_answer,
        "adjustment_reason": adjustment,
    }


def choose_setting_interactively() -> str:
    options = [(str(i), setting) for i, setting in enumerate(SETTINGS, start=1)]
    print("\nSelect experiment mode:")
    for number, setting in options:
        print(f"{number}. {SETTING_NAMES[setting]} ({setting})")
    aliases = {number: setting for number, setting in options}
    aliases.update({key: value for key, value in SETTING_ALIASES.items() if value != "all"})
    while True:
        choice = input(f"Enter 1-{len(options)}: ").strip().lower()
        if choice in aliases:
            return aliases[choice]
        print(f"Invalid choice. Please enter 1-{len(options)}.")


def classify_failure(trace: dict) -> tuple[str, list[str], str]:
    a_correct = bool(trace.get("solver_a_correct"))
    b_correct = bool(trace.get("solver_b_correct"))
    verifier_correct = bool(trace.get("verifier_correct"))
    solver_correct = a_correct or b_correct
    types = []
    if not solver_correct and not verifier_correct:
        types.append("没人会")
    if solver_correct:
        types.append("finalizer 丢答案")
    if solver_correct and trace.get("verifier_correct") is False:
        types.append("verifier 判断错")
    if verifier_correct:
        types.append("finalizer 不听 verifier")

    # More specific downstream failures take precedence; failure_types retains overlaps.
    priority = ["finalizer 不听 verifier", "verifier 判断错", "finalizer 丢答案", "没人会"]
    primary = next((item for item in priority if item in types), "没人会")
    if primary == "finalizer 不听 verifier":
        reason = "Verifier 已给出正确答案，但 finalizer 最终没有采用该答案。"
    elif primary == "verifier 判断错":
        sources = "、".join(name for name, ok in (("Solver A", a_correct), ("Solver B", b_correct)) if ok)
        reason = f"{sources} 已答对，但 verifier 给出了错误判断，finalizer 也未保留正确答案。"
    elif primary == "finalizer 丢答案":
        sources = "、".join(name for name, ok in (("Solver A", a_correct), ("Solver B", b_correct)) if ok)
        reason = f"{sources} 已答对，但 finalizer 没有采纳其答案。"
    else:
        reason = "Solver A、Solver B 和 verifier 均未产生正确答案。"
    return primary, types, reason


def build_failures(traces: list[dict]) -> list[dict]:
    failures = []
    for trace in traces:
        if trace["correct"]:
            continue
        primary, types, reason = classify_failure(trace)
        failures.append({
            "question_id": trace.get("example_index"),
            "question": trace["question"],
            "setting": trace["setting"],
            "gold": trace["gold_answer"],
            "solver_a_answer": trace.get("solver_a_answer", ""),
            "solver_a_correct": trace.get("solver_a_correct"),
            "solver_b_answer": trace.get("solver_b_answer", ""),
            "solver_b_correct": trace.get("solver_b_correct"),
            "verifier_answer": trace.get("verifier_answer", ""),
            "verifier_correct": trace.get("verifier_correct"),
            "final_answer": trace["final_prediction"],
            "final_correct": trace["correct"],
            "failure_type": primary,
            "failure_types": types,
            "short_reason": reason,
        })
    return failures


def write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    try:
        os.replace(temporary, path)
    except PermissionError:
        path.write_text(content, encoding="utf-8")
        temporary.unlink(missing_ok=True)


def write_outputs(traces: list[dict], output_dir: Path, print_table: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_dir / "traces_all.json", json.dumps(traces, ensure_ascii=False, indent=2))
    grouped = {s: {"n": 0, "correct": 0, "oracle_gap": 0, "oracle_gap_ids": [], "prompt_tokens": 0,
                   "completion_tokens": 0, "total_tokens": 0} for s in SETTINGS}
    for trace in traces:
        stats = grouped[trace["setting"]]
        stats["n"] += 1
        stats["correct"] += int(trace["correct"])
        if trace.get("oracle_gap"):
            stats["oracle_gap"] += 1
            stats["oracle_gap_ids"].append(trace.get("example_index"))
        add_usage(stats, trace["token_usage"])
    buffer = io.StringIO()
    fields = ["setting", "setting_name", "num_examples", "correct", "accuracy",
              "oracle_gap_count", "oracle_gap_rate", "oracle_gap_question_ids",
              "prompt_tokens", "completion_tokens", "total_tokens", "avg_total_tokens"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for setting, stats in grouped.items():
        n = stats["n"]
        writer.writerow({"setting": setting, "setting_name": SETTING_NAMES[setting],
                         "num_examples": n, "correct": stats["correct"],
                         "accuracy": round(stats["correct"] / n, 4) if n else 0,
                         "oracle_gap_count": stats["oracle_gap"],
                         "oracle_gap_rate": round(stats["oracle_gap"] / n, 4) if n else 0,
                         "oracle_gap_question_ids": json.dumps(stats["oracle_gap_ids"]),
                         "prompt_tokens": stats["prompt_tokens"],
                         "completion_tokens": stats["completion_tokens"],
                         "total_tokens": stats["total_tokens"],
                         "avg_total_tokens": round(stats["total_tokens"] / n, 2) if n else 0})
    write_text_atomic(output_dir / "metrics.csv", buffer.getvalue())
    write_text_atomic(output_dir / "failures.json", json.dumps(build_failures(traces), ensure_ascii=False, indent=2))
    if print_table:
        print("\nAccuracy and token summary")
        for setting, stats in grouped.items():
            accuracy = stats["correct"] / stats["n"] if stats["n"] else 0
            gap_rate = stats["oracle_gap"] / stats["n"] if stats["n"] else 0
            print(f"{SETTING_NAMES[setting]:<32} N={stats['n']:<3} accuracy={accuracy:.4f} "
                  f"oracle_gap={stats['oracle_gap']} ({gap_rate:.4f}) ids={stats['oracle_gap_ids']} "
                  f"total_tokens={stats['total_tokens']}")
    return output_dir


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


def parse_judge_response(text: str, prediction: str, gold_answer: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("DeepSeek judge did not return JSON.")
        data = json.loads(match.group(0))

    if "correct" not in data:
        raise ValueError("DeepSeek judge JSON is missing 'correct'.")
    raw_correct = data["correct"]
    if isinstance(raw_correct, str):
        normalized = raw_correct.strip().lower()
        if normalized not in {"true", "false", "yes", "no", "1", "0", "correct", "incorrect"}:
            raise ValueError("DeepSeek judge returned an invalid 'correct' value.")
        correct = normalized in {"true", "yes", "1", "correct"}
    else:
        correct = bool(raw_correct)
    return {
        "answer": prediction,
        "gold_answer": gold_answer,
        "correct": correct,
        "judge_conclusion": str(data.get("conclusion", "")).strip(),
    }


def deepseek_grade_agent_output(
    judge_client,
    retryable_errors: tuple[type[Exception], ...],
    judge_model: str,
    agent_name: str,
    question: str,
    agent_output: str,
    gold_answer: str,
) -> tuple[dict, dict]:
    fallback_answer = (
        str(parse_verifier(agent_output).get("verified_answer", "")).strip()
        if agent_name == "Verifier"
        else extract_answer(agent_output)
    )
    system_prompt = (
        "Judge only whether a predicted final answer is equivalent to the gold answer. "
        "Ignore reasoning quality. Treat mathematically equivalent formats as equal. "
        "Return only compact valid JSON."
    )
    user_prompt = (
        f"Agent: {agent_name}\nQuestion: {question}\n"
        f"Gold: {gold_answer}\nPrediction: {fallback_answer}\n"
        'Return: {"correct": true, "conclusion": "correct"}'
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
            return parse_judge_response(text, fallback_answer, gold_answer), usage_dict(resp)
        except retryable_errors + (ValueError,):
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


def make_candidate_table(solver_a_output: str, solver_b_output: str, verifier_output: dict) -> dict:
    scores = verifier_output.get("solver_scores", {}) if verifier_output else {}
    candidates = [
        {
            "source": "solver_a",
            "answer": extract_answer(solver_a_output),
            "reason": solver_a_output,
            "verifier_score": scores.get("A"),
        },
        {
            "source": "solver_b",
            "answer": extract_answer(solver_b_output),
            "reason": solver_b_output,
            "verifier_score": scores.get("B"),
        },
    ]
    if verifier_output:
        candidates.append({
            "source": "verifier",
            "answer": str(verifier_output.get("verified_answer", "")),
            "reason": str(verifier_output.get("critique", "")),
            "chosen_solver": verifier_output.get("chosen_solver"),
            "confidence": verifier_output.get("confidence"),
        })
    decision = resolve_verifier_decision(verifier_output, solver_a_output, solver_b_output)
    return {"candidates": candidates, "verifier_decision": decision}


def candidate_answers(candidate_table: dict) -> list[str]:
    return [
        str(candidate.get("answer", "")).strip()
        for candidate in candidate_table.get("candidates", [])
        if str(candidate.get("answer", "")).strip()
    ]


def answer_is_candidate(answer: str, candidate_table: dict) -> bool:
    return any(answers_match(answer, candidate) for candidate in candidate_answers(candidate_table))


def candidate_fallback(candidate_table: dict) -> tuple[str, str]:
    candidates = candidate_table.get("candidates", [])
    decision = candidate_table.get("verifier_decision", {})
    selected_answer = str(decision.get("selected_answer", "")).strip()
    if selected_answer:
        return selected_answer, f"verifier_decision:{decision.get('effective_choice', 'unknown')}"
    for preferred_source in ("verifier", "solver_a", "solver_b"):
        for candidate in candidates:
            answer = str(candidate.get("answer", "")).strip()
            if candidate.get("source") == preferred_source and answer:
                return answer, preferred_source
    return "", "none"


def ask_solver_for_final_objection(
    local_model: LocalQwenModel,
    solver_prompt: str,
    question: str,
    original_output: str,
    candidate_table: dict,
) -> tuple[str, dict]:
    prompt = (
        f"Question:\n{question}\n\nYour original answer:\n{original_output}\n\n"
        f"Current candidate table:\n{json.dumps(candidate_table, ensure_ascii=False)}\n\n"
        "候选表是否遗漏了重要依据？如果你不同意其中基于评分得到的当前决定，"
        "请给出最后一次简短的数学反驳；否则明确回答没有异议。"
    )
    return local_model.generate(solver_prompt, prompt)


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
    add_usage(token_usage, usage)
    solver_a_grade, usage = deepseek_grade_agent_output(
        judge_client,
        judge_retryable_errors,
        judge_model,
        "Solver A",
        question,
        solver_a_output,
        gold_answer,
    )
    add_usage(token_usage, usage)

    solver_b_output = ""
    solver_b_grade = None
    verifier_text = ""
    verifier_output = {}
    verifier_decision = {}
    verifier_gold_grade = None
    candidate_table = {}
    solver_objections = {}
    candidate_enforcement = {"required": False, "valid": None, "retried": False, "fallback_used": False}

    if setting in MULTI_SETTINGS:
        solver_b_output, usage = cached_local_call(
            model_cache,
            ("solver_b", example_key),
            local_model,
            prompts["solver_b"],
            question,
        )
        add_usage(token_usage, usage)
        solver_b_grade, usage = deepseek_grade_agent_output(
            judge_client,
            judge_retryable_errors,
            judge_model,
            "Solver B",
            question,
            solver_b_output,
            gold_answer,
        )
        add_usage(token_usage, usage)

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
            f"Solver A output:\n{solver_a_output}\n\n"
            f"Solver B output:\n{solver_b_output}"
        )
        verifier_text, usage = local_model.generate(prompts["verifier"], verifier_input)
        add_usage(token_usage, usage)
        verifier_output = parse_verifier(verifier_text)
        verifier_decision = resolve_verifier_decision(
            verifier_output, solver_a_output, solver_b_output
        )
        effective_verifier_output = dict(verifier_output)
        effective_verifier_output["chosen_solver"] = verifier_decision["effective_choice"]
        effective_verifier_output["verified_answer"] = verifier_decision["selected_answer"]
        decision_answer = verifier_decision.get("selected_answer", "")
        verifier_gold_grade, usage = deepseek_grade_agent_output(
            judge_client,
            judge_retryable_errors,
            judge_model,
            "Verifier Decision",
            question,
            f"Final Answer: {decision_answer}",
            gold_answer,
        )
        add_usage(token_usage, usage)
        final_input = (
            f"Question:\n{question}\n\n"
            f"Solver A output:\n{solver_a_output}\n\n"
            f"Solver B output:\n{solver_b_output}\n\n"
            f"Verifier JSON:\n{json.dumps(effective_verifier_output, ensure_ascii=False)}"
        )

    if setting == "multi_candidate_memory":
        candidate_table = make_candidate_table(solver_a_output, solver_b_output, verifier_output)
        candidate_enforcement["required"] = True
        final_input = (
            f"Question:\n{question}\n\nStructured candidate table:\n"
            f"{json.dumps(candidate_table, ensure_ascii=False, indent=2)}\n\n"
            "You must select an answer from this table. Explain briefly why, then end with "
            "`Final Answer: <answer>`."
        )
    elif setting == "multi_ask_before_finalize":
        candidate_table = make_candidate_table(solver_a_output, solver_b_output, verifier_output)
        for source, prompt_name, original_output in (
            ("solver_a", "solver_a", solver_a_output),
            ("solver_b", "solver_b", solver_b_output),
        ):
            objection, usage = ask_solver_for_final_objection(
                local_model, prompts[prompt_name], question, original_output, candidate_table
            )
            add_usage(token_usage, usage)
            solver_objections[source] = objection
        final_input += (
            "\n\nFinal objections from solvers:\n"
            f"{json.dumps(solver_objections, ensure_ascii=False, indent=2)}\n\n"
            "Consider these last objections before producing the final answer."
        )

    if setting == "multi_verifier_forced" and verifier_decision.get("selected_answer"):
        final_output = (
            "Verifier-forced setting: using the score-based verifier decision directly.\n"
            f"Final Answer: {verifier_decision['selected_answer']}"
        )
    else:
        final_output, usage = local_model.generate(prompts["finalizer"], final_input)
        add_usage(token_usage, usage)
    if setting == "multi_candidate_memory":
        first_prediction = extract_answer(final_output)
        candidate_enforcement["first_prediction"] = first_prediction
        if not answer_is_candidate(first_prediction, candidate_table):
            candidate_enforcement["retried"] = True
            allowed_answers = candidate_answers(candidate_table)
            retry_input = (
                f"{final_input}\n\nYour previous answer `{first_prediction}` was not one of the "
                f"allowed candidate answers: {json.dumps(allowed_answers, ensure_ascii=False)}. "
                "Choose exactly one allowed answer and explain why. End with `Final Answer: <answer>`."
            )
            final_output, usage = local_model.generate(prompts["finalizer"], retry_input)
            add_usage(token_usage, usage)
        candidate_enforcement["valid"] = answer_is_candidate(extract_answer(final_output), candidate_table)
        if not candidate_enforcement["valid"]:
            fallback_answer, fallback_source = candidate_fallback(candidate_table)
            candidate_enforcement.update({
                "fallback_used": True,
                "fallback_source": fallback_source,
                "invalid_finalizer_output": final_output,
            })
            final_output = (
                "Candidate constraint enforced after two invalid finalizer selections.\n"
                f"Selected source: {fallback_source}\nFinal Answer: {fallback_answer}"
            )
            candidate_enforcement["valid"] = bool(fallback_answer)
    final_prediction = extract_answer(final_output)
    finalizer_gold_grade, usage = deepseek_grade_agent_output(
        judge_client,
        judge_retryable_errors,
        judge_model,
        "Finalizer",
        question,
        final_output,
        gold_answer,
    )
    add_usage(token_usage, usage)
    correct = finalizer_gold_grade["correct"]
    verifier_answer = str(verifier_decision.get("selected_answer", ""))
    verifier_correct = verifier_gold_grade["correct"] if verifier_gold_grade else None
    oracle_gap = bool(
        not correct
        and (solver_a_grade["correct"] or bool(solver_b_grade and solver_b_grade["correct"])
             or bool(verifier_gold_grade and verifier_gold_grade["correct"]))
    )
    finalizer_followed_verifier = (
        answers_match(final_prediction, verifier_decision.get("selected_answer"))
        if verifier_decision.get("selected_answer")
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
        "solver_grades": {
            "A": solver_a_grade,
            "B": solver_b_grade,
        },
        "verifier_text": verifier_text,
        "verifier_output": verifier_output,
        "verifier_decision": verifier_decision,
        "verifier_gold_grade": verifier_gold_grade,
        "verifier_answer": verifier_answer,
        "verifier_correct": verifier_correct,
        "final_prediction": final_prediction,
        "finalizer_output": final_output,
        "finalizer_gold_grade": finalizer_gold_grade,
        "finalizer_followed_verifier": finalizer_followed_verifier,
        "candidate_table": candidate_table,
        "candidate_enforcement": candidate_enforcement,
        "solver_objections": solver_objections,
        "correctness_changes": correctness_changes,
        "correct": correct,
        "oracle_gap": oracle_gap,
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
        help=("single, multi, multi_verifier, multi_verifier_forced, "
              "multi_candidate_memory, multi_ask_before_finalize, or all."),
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
        selected_setting = choose_setting_interactively()

    settings = SETTINGS if selected_setting == "all" else [selected_setting]
    data_path = resolve_data_path(args.data_path)
    model_path = Path(args.model_path).resolve()

    prompts = {
        name: read_text(PROMPT_DIR / f"{name}.txt")
        for name in prompt_names_for_settings(settings)
    }
    try:
        examples = load_examples(data_path)
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
        print(f"outputs: {OUTPUT_DIR}")
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
    output_dir = make_timestamped_output_dir(OUTPUT_DIR, run_started_at)
    print(f"Writing incremental outputs to {output_dir}")
    write_outputs(traces, output_dir=output_dir, print_table=False)

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
            write_outputs(traces, output_dir=output_dir, print_table=False)

    write_outputs(traces, output_dir=output_dir)
    print(f"Done in {time.time() - started_at:.1f}s. Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
