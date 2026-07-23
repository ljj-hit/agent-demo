"""Hidden-GSM8K: controlled partial-information multi-agent evaluation on local Qwen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

ROOT = Path(__file__).resolve().parent

# =============================================================================
# USER CONFIGURATION
# Change paths and defaults here. No question text is stored in this script.
# Examples: DATA_PATH = ROOT / "data" / "my_questions.json"
#           DATA_PATH = Path(r"D:\datasets\my_questions.jsonl")
# Relative paths are resolved from the directory containing this script.
# Command-line arguments can override these values.
# =============================================================================
DATA_PATH = ROOT / "data" / "3q.json"
MODEL_PATH = ROOT / "qwen2.5-1.5B"
OUTPUT_BASE_DIR = ROOT / "outputs_hidden_gsm8k"
PROMPT_DIR = ROOT / "hidden_gsm8k_prompts"
PROMPT_PATHS = {
    "solver": PROMPT_DIR / "solver.txt",
    "verifier": PROMPT_DIR / "verifier.txt",
    "finalizer": PROMPT_DIR / "finalizer.txt",
}

DEFAULT_DEVICE = "cuda"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_DISCUSSION_ROUNDS = 2
DEFAULT_SEED = 42
DEFAULT_LIMIT = 0                    # 0 means all records in DATA_PATH.
DEFAULT_ALLOW_DOWNLOAD = False       # Keep local model loading offline by default.
DEFAULT_SKIP_DEEPSEEK = False        # False means DeepSeek judging is enabled.
DEFAULT_JUDGE_MAX_ATTEMPTS = 4
DEFAULT_FINALIZER_MAX_ATTEMPTS = 1  # Finalizer is never retried: a malformed selection is invalid.

# Leave empty to show the interactive setting menu. Example:
# DEFAULT_SELECTED_SETTINGS = ("multi_partial", "multi_partial_verifier")
DEFAULT_SELECTED_SETTINGS: tuple[str, ...] = ()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_API_KEY_ENV_NAMES = ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY")

SETTINGS = ("single_full", "single_partial", "multi_partial", "multi_partial_verifier", "oracle_broadcast")
REPLAY_SETTINGS = ("all_at_start_AB", "all_at_start_BA", "after_round1",
                   "before_final_transcript", "before_final_transcript_ledger", "before_final_reset")
SETTINGS = SETTINGS + REPLAY_SETTINGS
SETTING_NAMES = {
    "single_full": "Single Agent - Full Information",
    "single_partial": "Single Agent - Partial Information (A and B)",
    "multi_partial": "Multi-Agent - Partial Information",
    "multi_partial_verifier": "Multi-Agent - Partial Information + Verifier",
    "oracle_broadcast": "Oracle Broadcast",
    "all_at_start_AB": "Information Replay - All at Start (A then B)",
    "all_at_start_BA": "Information Replay - All at Start (B then A)",
    "after_round1": "Information Replay - Reveal after Round 1",
    "before_final_transcript": "Information Replay - Before Final with Transcript",
    "before_final_transcript_ledger": "Information Replay - Before Final with Transcript + Ledger",
    "before_final_reset": "Information Replay - Before Final after Context Reset",
}
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

# Backward-compatible aliases for imports from earlier revisions.
DEFAULT_DATA = DATA_PATH
DEFAULT_MODEL = MODEL_PATH
DEFAULT_OUTPUT = OUTPUT_BASE_DIR


def read_json_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    values = json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    required = {"condition_A", "condition_B", "shared_question", "answer"}
    for i, value in enumerate(values, 1):
        if not isinstance(value, dict) or not required <= value.keys():
            raise ValueError(f"record {i} must contain {sorted(required)}")
        # Prefer the new concise names, while accepting datasets produced by
        # the previous schema.  `fact` is evaluation metadata and must never
        # be used to construct a model prompt.
        full = value.get("full", value.get("full_question"))
        fact = value.get("fact", value.get("required_private_facts"))
        if not isinstance(full, str) or not full.strip():
            raise ValueError(f"record {i} must contain a non-empty 'full' field")
        if not isinstance(fact, dict):
            raise ValueError(f"record {i} must contain a 'fact' object")
        normalized_fact = {}
        for side in ("A", "B"):
            side_facts = fact.get(side, fact.get(f"agent_{side}"))
            if isinstance(side_facts, str):
                side_facts = [side_facts]
            if not isinstance(side_facts, list) or not all(isinstance(x, str) and x.strip() for x in side_facts):
                raise ValueError(f"record {i} fact must provide a non-empty string list for side {side}")
            normalized_fact[side] = side_facts
        value["full"] = full
        value["fact"] = normalized_fact
    return values


def extract_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    match = re.search(r"(?:final_answer|Final Answer)\s*[\":=]+\s*([^\n\"}]+)", text, re.I)
    if match:
        return match.group(1).strip()
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+)?", text)
    return nums[-1].replace(",", "") if nums else text


def decimal(value: Any) -> Decimal | None:
    text = extract_answer(value).replace(",", "").strip()
    frac = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", text)
    try:
        if frac:
            a, b = map(Decimal, frac.groups())
            return None if b == 0 else a / b
        return Decimal(text) if re.fullmatch(r"-?\d+(?:\.\d+)?", text) else None
    except InvalidOperation:
        return None


def equivalent(left: Any, right: Any) -> bool:
    a, b = decimal(left), decimal(right)
    if a is not None and b is not None:
        return a == b
    norm = lambda x: re.sub(r"\s+", " ", extract_answer(x).lower()).strip()
    return bool(norm(left)) and norm(left) == norm(right)


def _legacy_explicitly_undetermined(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return (text in UNDETERMINED_ANSWERS or
            any(phrase in text for phrase in ("cannot determine", "cannot be determined", "can't determine",
                                               "not enough information", "insufficient information", "无法确定", "不能确定")))


def _legacy_concludingly_undetermined(value: Any) -> bool:
    """Conservatively reject unlabeled prose that says the answer is unknown."""
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text:
        return True
    phrases = ("cannot determine", "cannot be determined", "can't determine", "not enough information",
               "insufficient information", "unable to determine", "answer is undetermined", "无法确定", "不能确定")
    # With no explicit answer label, any unresolved-insufficiency statement is
    # treated as authoritative. This guarantees that a coincidental gold
    # number elsewhere in the prose cannot be scored as an answer.
    return any(phrase in text for phrase in phrases)


def _legacy_extract_labeled_answer(text: Any, label: str) -> str:
    """Extract only the declared answer, never an incidental number in reasoning."""
    raw = re.sub(r"[*`]", "", str(text or ""))
    matches = re.findall(rf"(?im){re.escape(label)}\s*[:：=]\s*(.+?)\s*$", raw)
    if not matches:
        return ""
    declared = re.sub(r"[*`]+", "", matches[-1]).strip()
    return "" if explicitly_undetermined(declared) else extract_answer(declared)


def extract_current_answer(text: Any) -> str:
    return extract_labeled_answer(text, "Current answer")


def explicitly_undetermined(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    phrases = ("cannot determine", "cannot be determined", "can't determine", "not enough information",
               "insufficient information", "\u65e0\u6cd5\u786e\u5b9a", "\u4e0d\u80fd\u786e\u5b9a")
    return (text in UNDETERMINED_ANSWERS or any(phrase in text for phrase in phrases) or
            bool(re.search(r"\b(?:undetermined|unknown|insufficient)\b", text)))


def concludingly_undetermined(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text:
        return True
    phrases = ("cannot determine", "cannot be determined", "can't determine", "not enough information",
               "insufficient information", "unable to determine", "answer is undetermined",
               "impossible to determine", "impossible to calculate", "impossible to conclude",
               "cannot conclude", "cannot calculate", "cannot answer",
               "\u65e0\u6cd5\u786e\u5b9a", "\u4e0d\u80fd\u786e\u5b9a")
    return (any(phrase in text for phrase in phrases) or
            bool(re.search(r"\b(?:undetermined|unknown|insufficient)\b", text)))


def extract_labeled_answer(text: Any, label: str) -> str:
    """Extract a declared answer using encoding-safe punctuation patterns."""
    raw = re.sub(r"[*`]", "", str(text or ""))
    matches = re.findall(rf"(?im){re.escape(label)}\s*[:\uFF1A=]\s*(.+?)\s*$", raw)
    if not matches:
        return ""
    declared = matches[-1].strip()
    return "" if concludingly_undetermined(declared) else extract_answer(declared)


def extract_free_text_answer(text: Any, label: str) -> tuple[str, str]:
    """Return a safe answer plus an auditable extraction method."""
    raw = str(text or "").strip()
    labeled = extract_labeled_answer(raw, label)
    label_present = bool(re.search(rf"(?i){re.escape(label)}\s*[:：=]", re.sub(r"[*`]", "", raw)))
    if label_present:
        return labeled, "explicit_label" if labeled else "explicit_undetermined"
    # Some small models emit the discussion-format declaration even when a
    # final answer was requested. An explicit undetermined current answer is
    # authoritative and must never fall through to a numbered-list digit.
    if label.lower() == "final answer":
        normalized = re.sub(r"[*`]", "", raw)
        current_present = bool(re.search(r"(?i)Current answer\s*[:\uFF1A=]", normalized))
        if current_present and not extract_labeled_answer(raw, "Current answer"):
            return "", "explicit_current_undetermined"
    if concludingly_undetermined(raw):
        return "", "concluding_undetermined"
    fallback = extract_answer(raw)
    return (fallback, "safe_natural_language_fallback") if decimal(fallback) is not None else ("", "no_supported_answer")


def parse_solver_final(text: Any) -> tuple[str, str]:
    """Validate the solver contract instead of recovering from malformed prose."""
    # Preserve leading whitespace/newlines so an answer on the second physical
    # line cannot be silently promoted to the required first line. A terminal
    # newline emitted by the tokenizer is harmless.
    raw = str(text or "").rstrip("\r\n")
    lines = raw.splitlines()
    if not lines or not re.fullmatch(r"Final answer\s*[:\uFF1A]\s*.+", lines[0], re.I):
        return "", "first line must be `Final answer: ...`"
    answer = extract_labeled_answer(lines[0], "Final answer")
    if not answer and not explicitly_undetermined(lines[0]):
        return "", "Final answer is empty or unsupported"
    sentence_count = 0
    for line in (line.strip() for line in lines[1:] if line.strip()):
        # Protect decimal points before splitting. Each non-empty physical line
        # counts as at least one sentence, so unpunctuated bullet-style reasons
        # cannot bypass the three-sentence limit.
        protected = re.sub(r"(?<=\d)\.(?=\d)", "\uE000", line)
        sentence_count += len([part for part in re.split(r"[.!?\u3002\uFF01\uFF1F]+", protected)
                               if part.strip()])
    if sentence_count > 3:
        return "", "solver reasoning exceeds three sentences"
    return answer, ""


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "correct"}:
            return True
        if normalized in {"false", "no", "0", "incorrect"}:
            return False
    return default if value is None else bool(value)


def parse_object(text: str, defaults: dict) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        value = json.loads(match.group(0)) if match else {}
    result = dict(defaults)
    if isinstance(value, dict):
        result.update(value)
    return result


def raw_json_object(text: str) -> dict | None:
    """Parse the model's actual JSON object without filling default fields."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def blank_usage() -> dict:
    return {k: 0 for k in USAGE_KEYS}


def add_usage(target: dict, usage: dict) -> None:
    for key in USAGE_KEYS:
        target[key] = target.get(key, 0) + int(usage.get(key, 0) or 0)


def derived_seed(base_seed: int, *scope: Any) -> int:
    """Derive a stable seed that does not depend on Python's randomized hash."""
    material = "|".join([str(base_seed), *(str(value) for value in scope)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2 ** 31)


def reseed_model(model: Any, seed: int) -> None:
    random.seed(seed)
    model.torch.manual_seed(seed)
    if model.torch.cuda.is_available():
        model.torch.cuda.manual_seed_all(seed)


def dependency_status() -> dict[str, bool]:
    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "safetensors": importlib.util.find_spec("safetensors") is not None,
        "openai": importlib.util.find_spec("openai") is not None,
        "dotenv": importlib.util.find_spec("dotenv") is not None,
    }


def load_ml_dependencies():
    status = dependency_status()
    missing = [name for name in ("torch", "transformers", "safetensors") if not status[name]]
    if missing:
        raise SystemExit("Missing Python package(s): " + ", ".join(missing) +
                         ". Install them in the active environment before local Qwen inference.")
    import torch
    import transformers.utils.import_utils as transformers_import_utils

    # These optional integrations are not used by Hidden-GSM8K. Disable them
    # before importing AutoModel to avoid unrelated broken scipy/sklearn wheels.
    transformers_import_utils._sklearn_available = False
    transformers_import_utils._scipy_available = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return torch, AutoModelForCausalLM, AutoTokenizer


def load_api_dependencies():
    status = dependency_status()
    missing = [name for name in ("openai", "dotenv") if not status[name]]
    if missing:
        display = ["python-dotenv" if name == "dotenv" else name for name in missing]
        raise SystemExit("Missing DeepSeek package(s): " + ", ".join(display) + ".")
    from dotenv import load_dotenv
    from openai import OpenAI
    return load_dotenv, OpenAI


def validate_model_path(model_path: Path) -> None:
    required = ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        raise SystemExit(f"Model directory is missing: {', '.join(missing)}\nChecked: {model_path}")


def choose_model_dtype(torch, device: str):
    if not device.startswith("cuda"):
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


class LocalQwen:
    def __init__(self, path: Path, device: str, max_new_tokens: int, temperature: float, allow_download: bool):
        torch, AutoModelForCausalLM, AutoTokenizer = load_ml_dependencies()
        validate_model_path(path)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable; install CUDA PyTorch or use --device cpu.")
        self.torch, self.device = torch, device
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=not allow_download, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        dtype = choose_model_dtype(torch, device)
        print(f"model dtype: {dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, local_files_only=not allow_download, trust_remote_code=True).to(device).eval()

    def call(self, system: str, user: str, temperature: float | None = None) -> tuple[str, dict, float]:
        return self.call_batch([(system, user)], temperature=temperature)[0]

    def call_batch(self, requests: list[tuple[str, str]], temperature: float | None = None) -> list[tuple[str, dict, float]]:
        """Generate a logically simultaneous batch for symmetric solver turns."""
        started = time.perf_counter()
        rendered = [self.tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
                    for system, user in requests]
        inputs = self.tokenizer(rendered, return_tensors="pt", padding=True)
        prompt_tokens = [int(value) for value in inputs["attention_mask"].sum(dim=1).tolist()]
        padded_width = int(inputs["input_ids"].shape[-1])
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        kwargs = {"max_new_tokens": self.max_new_tokens, "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id}
        generation_temperature = self.temperature if temperature is None else temperature
        if generation_temperature > 0:
            kwargs.update(do_sample=True, temperature=generation_temperature, top_p=.9)
        else:
            kwargs["do_sample"] = False
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        batch_elapsed = time.perf_counter() - started
        results = []
        for index in range(len(requests)):
            generated = outputs[index, padded_width:]
            token_ids = generated.tolist()
            if self.tokenizer.eos_token_id in token_ids:
                completion_tokens = token_ids.index(self.tokenizer.eos_token_id) + 1
                generated = generated[:completion_tokens]
            else:
                completion_tokens = len(token_ids)
            usage = {"prompt_tokens": prompt_tokens[index], "completion_tokens": completion_tokens,
                     "total_tokens": prompt_tokens[index] + completion_tokens}
            results.append((self.tokenizer.decode(generated, skip_special_tokens=True).strip(), usage,
                            batch_elapsed / len(requests)))
        return results


VERIFIER_DEFAULT = {"information_sufficient": False, "revealed_facts": [], "candidate_checks": [], "verified_answer": "", "selected_source": "none", "missing_information": []}
FINALIZER_DEFAULT = {"final_answer": "", "selected_source": "none", "reason": ""}

UNDETERMINED_ANSWERS = {"", "unknown", "undetermined", "cannot determine", "cannot be determined", "insufficient information", "none", "n/a"}


def model_event(model: LocalQwen, agent: str, system: str, user: str, phase: str, parser_defaults: dict | None,
                temperature: float | None = None) -> dict:
    try:
        raw, usage, elapsed = model.call(system, user, temperature=temperature)
    except TypeError:
        # Compatibility with simple test doubles and older imported model wrappers.
        raw, usage, elapsed = model.call(system, user)
    if parser_defaults is None:
        return {"agent": agent, "phase": phase, "actual_input": user,
                "actual_messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "output": raw,
                "raw_output": raw, "token_usage": usage, "runtime_seconds": elapsed}
    try:
        parsed, parse_error = parse_object(raw, parser_defaults), ""
    except (ValueError, json.JSONDecodeError) as exc:
        parsed, parse_error = dict(parser_defaults), str(exc)
    return {"agent": agent, "phase": phase, "actual_input": user,
            "actual_messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "raw_output": raw,
            "parsed_output": parsed, "parse_error": parse_error, "token_usage": usage, "runtime_seconds": elapsed}


def paired_model_events(model: LocalQwen, system: str, specs: dict[str, tuple[str, str, dict | None]],
                        temperature: float | None = None) -> dict[str, dict]:
    """Run A/B in one model batch when supported; fall back only for test doubles."""
    sides = ("A", "B")
    if hasattr(model, "call_batch"):
        try:
            results = model.call_batch([(system, specs[side][1]) for side in sides], temperature=temperature)
        except TypeError:
            results = model.call_batch([(system, specs[side][1]) for side in sides])
        events = {}
        for side, (raw, usage, elapsed) in zip(sides, results):
            agent, user, defaults = specs[side]
            if defaults is None:
                events[side] = {"agent": agent, "phase": "", "actual_input": user,
                                "actual_messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "output": raw,
                                "raw_output": raw, "token_usage": usage, "runtime_seconds": elapsed,
                                "generated_in_parallel_batch": True}
                continue
            else:
                try:
                    parsed, parse_error = parse_object(raw, defaults), ""
                except (ValueError, json.JSONDecodeError) as exc:
                    parsed, parse_error = dict(defaults), str(exc)
            events[side] = {"agent": agent, "phase": "", "actual_input": user,
                            "actual_messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "raw_output": raw,
                            "parsed_output": parsed, "parse_error": parse_error, "token_usage": usage,
                            "runtime_seconds": elapsed, "generated_in_parallel_batch": True}
        return events
    return {side: model_event(model, specs[side][0], system, specs[side][1], "", specs[side][2],
                              temperature=temperature) for side in sides}


def public_transcript(rounds: list[dict]) -> str:
    lines = []
    for event in rounds:
        public_turn = event.get("raw_output", "").strip()
        if public_turn:
            label = f'{event["agent"]} round={event.get("round", "-")} stage={event.get("stage", event["phase"])}'
            lines.append(f'{label}:\n{public_turn}')
    return "\n".join(lines) or "(nothing disclosed yet)"


def run_discussion(model: LocalQwen, solver_prompt: str, item: dict, oracle: bool, rounds_count: int = DEFAULT_DISCUSSION_ROUNDS) -> dict:
    events: list[dict] = []
    round_records: list[dict] = []
    oracle_text = ""
    if oracle:
        oracle_text = f'ORACLE PUBLIC FACT A (verbatim): {item["condition_A"]}\nORACLE PUBLIC FACT B (verbatim): {item["condition_B"]}'

    for round_no in range(1, rounds_count + 1):
        # A and B speak once in one GPU batch from the exact same pre-round public
        # snapshot; neither input contains the peer's same-round output.
        pre_round_transcript = "\n".join(x for x in (oracle_text, public_transcript(events)) if x)
        outbound_specs = {}
        for side in ("A", "B"):
            purpose = ("Share your information, reason as far as possible, and state exactly what is still missing."
                       if round_no == 1 else "Correct mistakes, fill gaps, and advance the solution using earlier messages.")
            user = (f'Role: solver_{side.lower()}\nDiscussion round: {round_no} of {rounds_count}\nPurpose: {purpose}\n'
                    f'Shared question: {item["shared_question"]}\nYour private fact: {item[f"condition_{side}"]}\n'
                    f'Public transcript through the previous completed round:\n{pre_round_transcript}\n'
                    "Think about the complete problem, not only your fragment. Explain what your facts imply, disclose exact useful facts, "
                    "state what information is missing, and respond to earlier claims when present. Begin with exactly one separate line "
                    "`Current answer: <answer>` or `Current answer: undetermined`, then give your reasoning. Write directly to the other solver in natural text; "
                    "do not output JSON. You cannot see the peer's same-round message.")
            outbound_specs[side] = (f"solver_{side.lower()}", user, None)
        outbound = paired_model_events(model, solver_prompt, outbound_specs)
        for side in ("A", "B"):
            event = outbound[side]
            event["phase"] = f"discussion_round_{round_no}_send"
            event["round"] = round_no
            event["stage"] = "send"
            event["current_answer"], event["current_answer_extraction"] = extract_free_text_answer(event["raw_output"], "Current answer")
            event["current_answer_explicit"] = event["current_answer_extraction"].startswith("explicit_")
        events.extend([outbound["A"], outbound["B"]])
        round_records.append({"round": round_no, "purpose": purpose,
                              "pre_round_public_transcript": pre_round_transcript,
                              "simultaneous_turn": {side.lower(): outbound[side] for side in ("A", "B")}})

    final_specs = {}
    transcript = "\n".join(x for x in (oracle_text, public_transcript(events)) if x)
    for side in ("A", "B"):
        user = (f'Role: solver_{side.lower()}\nShared question: {item["shared_question"]}\n'
                f'Your private fact: {item[f"condition_{side}"]}\n'
                f'Public transcript after {rounds_count} symmetric rounds:\n{transcript}\n'
                "The public transcript may contain the facts missing from your private input. Before answering undetermined, extract the "
                "other solver's disclosed numbers and relationships, combine them with your private fact, and check whether the complete "
                "calculation is now possible. Solve using all available information. Put `Final answer: ...` on the FIRST line, "
                "then give at most three sentences of reasoning. Use natural text; do not output JSON.")
        final_specs[side] = (f"solver_{side.lower()}", user, None)
    final_batch = paired_model_events(model, solver_prompt, final_specs)
    finals = {}
    for side in ("A", "B"):
        final_batch[side]["phase"] = "solver_final"
        answer, format_error = parse_solver_final(final_batch[side].get("raw_output", ""))
        final_batch[side]["answer"] = answer
        final_batch[side]["answer_extraction"] = "strict_solver_final" if not format_error else "invalid_format"
        final_batch[side]["validation_error"] = format_error
        final_batch[side]["invalid_output"] = bool(format_error)
        finals[side.lower()] = final_batch[side]
    result = {"protocol": "symmetric_one_turn_per_round", "round_records": round_records,
              "symmetry_guarantees": {"same_round_send_uses_identical_public_snapshot": True,
                                       "paired_solver_generation_uses_one_gpu_batch": True,
                                       "cross_agent_channel_is_raw_public_text_only": True},
              "discussion_events": events, "public_transcript": transcript,
              "solver_finals": finals}
    if oracle:
        result["oracle_public_information"] = oracle_text
    return result


def replay_facts(item: dict, order: str = "AB") -> str:
    """Render frozen dataset facts verbatim; this function never asks a model to rewrite them."""
    return "\n".join(f"FACT {side} (verbatim): {item[f'condition_{side}']}" for side in order)


def replay_fact_hash(item: dict) -> str:
    """Order-independent identity of the exact A/B fact collection."""
    canonical = json.dumps({"A": item["condition_A"], "B": item["condition_B"]},
                           ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def replay_ledger(item: dict) -> str:
    """A deterministic table: labels are normalized, fact values remain byte-for-byte unchanged."""
    return f"| side | fact (verbatim) |\n|---|---|\n| A | {item['condition_A']} |\n| B | {item['condition_B']} |"


def run_replay_discussion(model: LocalQwen, solver_prompt: str, item: dict, reveal_after_round: int | None,
                          order: str = "AB", rounds_count: int = DEFAULT_DISCUSSION_ROUNDS) -> dict:
    """Discussion protocol for timing replay. No solver has an undisclosed private fact."""
    events, round_records = [], []
    facts = replay_facts(item, order)
    for round_no in range(1, rounds_count + 1):
        visible_facts = facts if reveal_after_round is not None and round_no > reveal_after_round else ""
        pre_round_transcript = "\n".join(x for x in (visible_facts, public_transcript(events)) if x)
        specs = {}
        for side in ("A", "B"):
            purpose = ("Share your reasoning and state exactly what information is still missing."
                       if round_no == 1 else "Correct mistakes, fill gaps, and advance the solution using earlier messages.")
            user = (f'Role: solver_{side.lower()}\nDiscussion round: {round_no} of {rounds_count}\nPurpose: {purpose}\n'
                    f'Shared question: {item["shared_question"]}\n'
                    f'Public transcript through the previous completed round:\n{pre_round_transcript or "(nothing disclosed yet)"}\n'
                    "Use only information actually visible above. Begin with exactly one separate line "
                    "`Current answer: <answer>` or `Current answer: undetermined`, then give your reasoning. "
                    "Write directly to the other solver in natural text; do not output JSON. You cannot see the peer's same-round message.")
            specs[side] = (f"solver_{side.lower()}", user, None)
        outbound = paired_model_events(model, solver_prompt, specs, temperature=0.0)
        for side in ("A", "B"):
            event = outbound[side]
            event.update(phase=f"discussion_round_{round_no}_send", round=round_no, stage="send")
            event["current_answer"], event["current_answer_extraction"] = extract_free_text_answer(
                event["raw_output"], "Current answer")
            event["current_answer_explicit"] = event["current_answer_extraction"].startswith("explicit_")
        events.extend((outbound["A"], outbound["B"]))
        round_records.append({"round": round_no, "purpose": purpose,
                              "facts_visible": bool(visible_facts),
                              "pre_round_public_transcript": pre_round_transcript or "(nothing disclosed yet)",
                              "simultaneous_turn": {side.lower(): outbound[side] for side in ("A", "B")}})

    visible_at_solver_final = reveal_after_round is not None and rounds_count >= reveal_after_round
    transcript = "\n".join(x for x in (facts if visible_at_solver_final else "", public_transcript(events)) if x)
    specs = {}
    for side in ("A", "B"):
        user = (f'Role: solver_{side.lower()}\nShared question: {item["shared_question"]}\n'
                f'Public transcript after {rounds_count} symmetric rounds:\n{transcript or "(nothing disclosed yet)"}\n'
                "Use only the visible information. Put `Final answer: ...` on the FIRST line, then give at most three "
                "sentences of reasoning. Use natural text; do not output JSON.")
        specs[side] = (f"solver_{side.lower()}", user, None)
    final_batch = paired_model_events(model, solver_prompt, specs, temperature=0.0)
    finals = {}
    for side in ("A", "B"):
        event = final_batch[side]
        answer, error = parse_solver_final(event.get("raw_output", ""))
        event.update(phase="solver_final", answer=answer,
                     answer_extraction="strict_solver_final" if not error else "invalid_format",
                     validation_error=error, invalid_output=bool(error))
        finals[side.lower()] = event
    information_timeline = [
        {"checkpoint": "after_simultaneous_turn", "round": row["round"],
         "information_complete": row["facts_visible"],
         "side_revealed": {"A": row["facts_visible"], "B": row["facts_visible"]}}
        for row in round_records]
    return {"protocol": "information_timing_replay", "round_records": round_records,
            "discussion_events": events, "public_transcript": transcript or "(nothing disclosed yet)",
            "solver_finals": finals, "facts_visible_at_solver_final": visible_at_solver_final,
            "information_timeline": information_timeline,
            "reveal_after_round": reveal_after_round, "fact_order": order}


def coverage_score(fact: str, public: str) -> float:
    tokens = set(re.findall(r"[a-z0-9.$%/]+", fact.lower()))
    seen = set(re.findall(r"[a-z0-9.$%/]+", public.lower()))
    return len(tokens & seen) / max(1, len(tokens))


def atomic_facts(condition: str) -> list[str]:
    """Split a condition into auditable units without changing the dataset."""
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", condition.strip()) if part.strip()]
    return parts or [condition.strip()]


def fact_is_public(fact: str, public: str) -> tuple[bool, float]:
    score = coverage_score(fact, public)
    required_numbers = set(re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?|\d+%", fact.lower()))
    public_numbers = set(re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?|\d+%", public.lower()))
    # Numbers must survive disclosure; lexical overlap permits concise paraphrases.
    return score >= .60 and required_numbers <= public_numbers, score


def objective_information(item: dict, discussion: dict) -> dict:
    if discussion.get("oracle_public_information"):
        return {"required_fact_units": item["fact"], "side_revealed": {"A": True, "B": True},
                "unit_coverage_scores": {"A": [1.0] * len(item["fact"]["A"]), "B": [1.0] * len(item["fact"]["B"])},
                "information_complete": True, "assessment_method": "oracle verbatim disclosure"}
    public = discussion["public_transcript"]
    units, revealed, scores = {}, {}, {}
    for side in ("A", "B"):
        # Gold fact units are only used here, after generation, to measure
        # information exchange. They are deliberately absent from all prompts.
        units[side] = item["fact"][side]
        checks = [fact_is_public(fact, public) for fact in units[side]]
        revealed[side] = all(ok for ok, _ in checks)
        scores[side] = [round(score, 4) for _, score in checks]
    return {"required_fact_units": units, "side_revealed": revealed, "unit_coverage_scores": scores,
            "information_complete": all(revealed.values()), "assessment_method": "atomic lexical+numeric coverage",
            "needs_semantic_review": not all(revealed.values())}


def add_information_timeline(item: dict, discussion: dict) -> None:
    oracle = discussion.get("oracle_public_information", "")
    events = discussion.get("discussion_events", [])
    timeline, accumulated = [], []

    def checkpoint(label: str, round_no: int | None) -> dict:
        public = "\n".join(x for x in (oracle, public_transcript(accumulated)) if x)
        snapshot_input = {"public_transcript": public}
        if oracle:
            snapshot_input["oracle_public_information"] = oracle
        snapshot = objective_information(item, snapshot_input)
        row = {"checkpoint": label, "round": round_no, "public_event_count": len(accumulated),
               "information_complete": snapshot["information_complete"], "side_revealed": snapshot["side_revealed"]}
        timeline.append(row)
        return row

    state = checkpoint("discussion_start", None)
    for round_no in sorted({event["round"] for event in events}):
        sends = [event for event in events if event["round"] == round_no]
        for event in sends:
            event["information_complete_at_generation"] = state["information_complete"]
        accumulated.extend(sends)
        state = checkpoint("after_simultaneous_turn", round_no)
    discussion["information_timeline"] = timeline
    first = next((x for x in timeline if x["information_complete"]), None)
    discussion["first_complete_checkpoint"] = first
    discussion["first_complete_after_public_event"] = first["public_event_count"] if first else None


def single_call(model: LocalQwen, prompt: str, item: dict, side: str | None) -> dict:
    if side is None:
        # Only the full-information setting receives the complete problem.
        user = (f'Role: solver_a\nFull question: {item["full"]}\n'
                'Solve the complete problem carefully. Put `Final answer: ...` on the FIRST line, then give at most three sentences of reasoning. '
                'Use natural text; do not output JSON.')
    else:
        user = (f'Role: solver_{side.lower()}\nShared question: {item["shared_question"]}\n'
                f'Your private fact: {item[f"condition_{side}"]}\nAnalyze what can be concluded and clearly identify missing information. '
                'If the answer is determined, put `Final answer: ...` on the FIRST line, then give at most three sentences of reasoning; '
                'otherwise put `Final answer: undetermined` on the first line and explain why in at most three sentences. '
                'Use natural text; do not output JSON.')
    event = model_event(model, f'solver_{(side or "a").lower()}', prompt, user, "single_final", None)
    answer, format_error = parse_solver_final(event.get("raw_output", ""))
    event.update(answer=answer,
                 answer_extraction="strict_solver_final" if not format_error else "invalid_format",
                 validation_error=format_error, invalid_output=bool(format_error))
    return event


def event_answer(event: dict | None, key: str = "final_answer") -> str:
    event = event or {}
    parsed = event.get("parsed_output", {})
    if key in parsed:
        value = str(parsed.get(key, "")).strip()
        return "" if explicitly_undetermined(value) else extract_answer(value)
    # Free-text solver outputs must explicitly declare a final answer. This
    # prevents "cannot determine" prose that merely mentions the gold number
    # from being scored as correct.
    if key == "final_answer" and "answer" in event:
        return str(event.get("answer", ""))
    return extract_free_text_answer(event.get("raw_output", ""), "Final answer")[0]


def candidate_appearances(trace: dict) -> list[dict]:
    """Answers visible before final selection, with information state at appearance time."""
    if trace.get("single_event"):
        answer = event_answer(trace["single_event"])
        return [{"source": trace["single_event"]["agent"], "phase": "single_final", "answer": answer,
                 "information_complete_at_appearance": trace["information"]["information_complete"]}]
    discussion = trace.get("discussion") or {}
    appearances = []
    timeline = {row.get("round"): row for row in discussion.get("information_timeline", [])
                if row.get("checkpoint") == "after_simultaneous_turn"}
    for event in discussion.get("discussion_events", []):
        answer = event.get("current_answer", "")
        if answer:
            state = timeline.get(event.get("round"), {})
            appearances.append({"source": event["agent"], "phase": event["phase"], "round": event.get("round"),
                                "answer": answer, "information_complete_at_appearance": bool(state.get("information_complete"))})
    complete_after_discussion = bool(discussion.get(
        "facts_visible_at_solver_final",
        trace.get("information", {}).get("information_complete")))
    for side in ("a", "b"):
        event = discussion.get("solver_finals", {}).get(side)
        if event:
            appearances.append({"source": event["agent"], "phase": "solver_final", "answer": event_answer(event),
                                "information_complete_at_appearance": complete_after_discussion})
    verifier = trace.get("verifier_event")
    if verifier and not verifier.get("invalid_output"):
        appearances.append({"source": "verifier", "phase": "verification", "answer": event_answer(verifier, "verified_answer"),
                            "information_complete_at_appearance": complete_after_discussion})
    return appearances


def parse_fixed_finalizer(text: str) -> tuple[dict, str]:
    """Parse exactly three labeled lines; never recover by asking the model again."""
    # A terminal newline emitted by the tokenizer is harmless. Any leading or
    # internal blank line is still a fourth-format line and must be rejected.
    raw = str(text or "").rstrip("\r\n")
    lines = raw.splitlines()
    labels = ("Selected source", "Final answer", "Reason")
    if len(lines) != 3:
        return dict(FINALIZER_DEFAULT), "expected exactly three lines"
    values = {}
    for line, label in zip(lines, labels):
        match = re.fullmatch(rf"{re.escape(label)}\s*[:\uFF1A]\s*(.*)", line, re.I)
        if not match:
            return dict(FINALIZER_DEFAULT), f"expected line `{label}: ...`"
        values[label] = match.group(1).strip()
    source = values["Selected source"].lower()
    if source not in {"solver_a", "solver_b", "verifier", "recomputed", "none"}:
        return dict(FINALIZER_DEFAULT), "selected_source is not an allowed value"
    if not values["Reason"]:
        return dict(FINALIZER_DEFAULT), "Reason must not be empty"
    answer = "" if explicitly_undetermined(values["Final answer"]) else extract_answer(values["Final answer"])
    return {"selected_source": source, "final_answer": answer, "reason": values["Reason"]}, ""


def source_consistency_error(parsed: dict, candidates: dict, *, allow_none: bool) -> str:
    source, answer = parsed.get("selected_source", "none"), parsed.get("final_answer", "")
    allowed = {"solver_a", "solver_b", "verifier", "recomputed"} | ({"none"} if allow_none else set())
    if source not in allowed:
        return "selected_source is not an allowed value"
    if source in {"solver_a", "solver_b", "verifier"} and source not in candidates:
        return f"selected source {source} is unavailable or invalid"
    if source in candidates and not equivalent(answer, candidates[source]):
        return f"answer does not match selected source {source}"
    if source == "none" and answer:
        return "selected_source none must have an empty answer"
    if source == "recomputed" and decimal(answer) is None:
        return "selected_source recomputed requires a supported numeric answer"
    return ""


def has_explicit_identical_candidate_rejection(reason: Any) -> bool:
    """Require an auditable rejection declaration without guessing semantics."""
    text = re.sub(r"\s+", " ", str(reason or "").strip())
    chinese_prefix = "\u62d2\u7edd\u76f8\u540c\u5019\u9009\uff0c\u56e0\u4e3a"
    if text.startswith(chinese_prefix) and text[len(chinese_prefix):].strip(" :\uFF1A\uFF0C,"):
        return True
    prefixes = ("Reject identical candidates because", "拒绝相同候选，因为")
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()) and text[len(prefix):].strip(" :：，,"):
            return True
    return False


def verifier_explains_identical_candidate_rejection(parsed: dict) -> bool:
    """Require a concrete contradiction when the verifier rejects agreed answers."""
    checks = parsed.get("candidate_checks", [])
    return any(check.get("source") in {"solver_a", "solver_b"} and
               str(check.get("status", "")).strip().lower() == "unsupported" and
               bool(str(check.get("reason", "")).strip())
               for check in checks if isinstance(check, dict))


def verifier_consistency_error(event: dict, candidates: dict) -> str:
    raw = raw_json_object(event.get("raw_output", ""))
    if raw is None:
        return "verifier response is not a valid JSON object"
    missing = [key for key in ("verified_answer", "selected_source") if key not in raw]
    if missing:
        return "verifier missing required field(s): " + ", ".join(missing)
    parsed = event.get("parsed_output", {})
    parsed["verified_answer"] = ("" if explicitly_undetermined(raw.get("verified_answer"))
                                 else extract_answer(raw.get("verified_answer")))
    parsed["selected_source"] = str(raw.get("selected_source", "none")).strip().lower()
    # Reuse the common source-consistency validator, whose canonical answer
    # field is named final_answer.
    parsed["final_answer"] = parsed["verified_answer"]
    error = source_consistency_error(parsed, candidates, allow_none=True)
    if error:
        return error
    a_answer, b_answer = candidates.get("solver_a", ""), candidates.get("solver_b", "")
    if (parsed["selected_source"] == "recomputed" and a_answer and b_answer and
            equivalent(a_answer, b_answer) and not equivalent(parsed["verified_answer"], a_answer) and
            not verifier_explains_identical_candidate_rejection(parsed)):
        return ("different recomputation of identical solver candidates requires an unsupported "
                "candidate_check with a non-empty contradiction reason")
    if parsed["selected_source"] in candidates:
        parsed["verified_answer"] = candidates[parsed["selected_source"]]
    return ""


def call_finalizer_once(model: LocalQwen, system: str, user: str, candidates: dict) -> dict:
    # Selection should be deterministic. Solver creativity must not leak into
    # the final source choice or cause answer drift.
    event = model_event(model, "finalizer", system, user, "finalization", None, temperature=0.0)
    parsed, error = parse_fixed_finalizer(event.get("raw_output", ""))
    if not error:
        error = source_consistency_error(parsed, candidates, allow_none=True)
    a_answer, b_answer = candidates.get("solver_a", ""), candidates.get("solver_b", "")
    if (not error and parsed["selected_source"] == "recomputed" and a_answer and b_answer and
            equivalent(a_answer, b_answer) and not equivalent(parsed["final_answer"], a_answer)):
        if not has_explicit_identical_candidate_rejection(parsed.get("reason")):
            error = ("different recomputation requires Reason to start with "
                     "`Reject identical candidates because` or `拒绝相同候选，因为`, followed by an explanation")
    # Preserve the selected candidate's exact representation after proving
    # mathematical equivalence (for example, normalize 42.0 back to 42).
    if not error and parsed["selected_source"] in candidates:
        parsed["final_answer"] = candidates[parsed["selected_source"]]
    event.update(parsed_output=parsed, parse_error="", validation_error=error,
                 attempts=[{"attempt": 1, "raw_output": event.get("raw_output", ""),
                            "parsed_output": parsed, "validation_error": error,
                            "token_usage": event.get("token_usage", blank_usage()),
                            "runtime_seconds": event.get("runtime_seconds", 0.0)}],
                 retry_count=0, recovered_after_retry=False, retry_exhausted=False,
                 invalid_output=bool(error))
    return event


def finish_multi(model: LocalQwen, prompts: dict, item: dict, discussion: dict, with_verifier: bool) -> tuple[dict | None, dict]:
    raw_candidates = {"solver_a": event_answer(discussion["solver_finals"]["a"]),
                      "solver_b": event_answer(discussion["solver_finals"]["b"])}
    candidates = {source: answer for source, answer in raw_candidates.items() if decimal(answer) is not None}
    verifier = None
    if with_verifier:
        user = f'Shared question: {item["shared_question"]}\nPublic transcript:\n{discussion["public_transcript"]}\nCandidates: {json.dumps(candidates, ensure_ascii=False)}'
        verifier = model_event(model, "verifier", prompts["verifier"], user, "verification", VERIFIER_DEFAULT)
        verifier_error = verifier_consistency_error(verifier, candidates)
        verifier["validation_error"] = verifier_error
        verifier["invalid_output"] = bool(verifier_error)
    if verifier and verifier.get("invalid_output"):
        report = {"usable": False, "validation_error": verifier.get("validation_error", "invalid verifier output")}
    else:
        report = verifier["parsed_output"] if verifier else "(no verifier in this setting)"
    # Empty/undetermined outputs are deliberately not offered as selectable
    # sources. This prevents a small model from blindly choosing solver_a just
    # because it appears first in the candidate object.
    # GSM8K targets are numeric. Prose such as "the answer cannot be
    # calculated" must not become a selectable candidate merely because it is
    # non-empty or happens to mention an incidental number.
    finalizer_candidates = dict(candidates)
    if verifier and not verifier.get("invalid_output"):
        verified_answer = event_answer(verifier, "verified_answer")
        if decimal(verified_answer) is not None:
            finalizer_candidates["verifier"] = verified_answer
    available_sources = list(finalizer_candidates) + ["recomputed", "none"]
    user = (f'Shared question: {item["shared_question"]}\nPublic transcript:\n{discussion["public_transcript"]}\n'
            f'Valid non-empty candidates: {json.dumps(finalizer_candidates, ensure_ascii=False)}\n'
            f'Available selected_source values for this question: {json.dumps(available_sources, ensure_ascii=False)}\n'
            f'Verifier report: {json.dumps(report, ensure_ascii=False)}\n'
            "For solver_a, solver_b, or verifier, copy that source's candidate answer exactly; never calculate a replacement value.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, finalizer_candidates)
    return verifier, finalizer


def collect_events(trace: dict) -> list[dict]:
    events = []
    if trace.get("single_event"):
        events.append(trace["single_event"])
    discussion = trace.get("discussion") or {}
    events += discussion.get("discussion_events", [])
    events += list(discussion.get("solver_finals", {}).values())
    if trace.get("verifier_event"):
        events.append(trace["verifier_event"])
    if trace.get("finalizer_event"):
        events.append(trace["finalizer_event"])
    return events


def classify(trace: dict, gold: str) -> tuple[str | None, bool]:
    complete = bool(trace.get("information", {}).get("information_complete"))
    appearances = trace.get("candidate_appearances", [])
    supported_correct_appeared = any(as_bool(x.get("correct"), equivalent(x.get("answer"), gold)) and x.get("information_complete_at_appearance", False)
                                     for x in appearances if x.get("answer"))
    lucky = any(as_bool(x.get("correct"), equivalent(x.get("answer"), gold)) and not x.get("information_complete_at_appearance", False)
                for x in appearances if x.get("answer"))
    if trace["correct"]:
        return None, lucky
    if trace.get("invalid_output"):
        return "invalid_output", lucky
    if not complete:
        return "information_acquisition_failure", lucky
    if not supported_correct_appeared:
        return "information_integration_failure", lucky
    return "answer_selection_failure", lucky


def build_trace(model: LocalQwen, prompts: dict, item: dict, qid: int, setting: str, discussion_cache: dict,
                variant: str = "", discussion_rounds: int = DEFAULT_DISCUSSION_ROUNDS) -> dict:
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    trace = {"question_id": qid, "setting": setting, "shared_question": item["shared_question"], "gold_answer": gold}
    if setting == "single_partial":
        trace["agent_variant"] = variant
    if setting.startswith("single"):
        side = variant if setting == "single_partial" else None
        event = single_call(model, prompts["solver"], item, side)
        prediction = event["answer"]
        trace.update(single_event=event, final_prediction=prediction, candidate_answers={event["agent"]: prediction}, information={"information_complete": side is None, "side_revealed": {"A": side is None or side == "A", "B": side is None or side == "B"}})
        trace["invalid_output"] = bool(event.get("invalid_output"))
    else:
        cache_key = "oracle" if setting == "oracle_broadcast" else "partial"
        if cache_key not in discussion_cache:
            discussion_cache[cache_key] = run_discussion(model, prompts["solver"], item, cache_key == "oracle", discussion_rounds)
            add_information_timeline(item, discussion_cache[cache_key])
        discussion = discussion_cache[cache_key]
        with_verifier = setting in {"multi_partial_verifier", "oracle_broadcast"}
        verifier, finalizer = finish_multi(model, prompts, item, discussion, with_verifier)
        candidates = {"solver_a": event_answer(discussion["solver_finals"]["a"]), "solver_b": event_answer(discussion["solver_finals"]["b"])}
        if verifier and not verifier.get("invalid_output"):
            candidates["verifier"] = event_answer(verifier, "verified_answer")
        trace.update(discussion=discussion, discussion_cache_key=cache_key, finalizer_event=finalizer,
                     final_prediction=event_answer(finalizer), candidate_answers=candidates, information=objective_information(item, discussion))
        trace["invalid_output"] = bool(finalizer.get("invalid_output"))
        trace["finalizer_retry_count"] = int(finalizer.get("retry_count", 0))
        trace["finalizer_recovered"] = bool(finalizer.get("recovered_after_retry"))
        trace["finalizer_exhausted"] = bool(finalizer.get("retry_exhausted"))
        if verifier is not None:
            trace["verifier_event"] = verifier
    trace["correct_before_judge"] = equivalent(trace["final_prediction"], gold)
    # Keep semantic correctness for auditing, but malformed output is never a
    # correct experiment result, regardless of whether DeepSeek is enabled.
    trace["correct"] = not trace.get("invalid_output", False) and trace["correct_before_judge"]
    trace["candidate_appearances"] = candidate_appearances(trace)
    for appearance in trace["candidate_appearances"]:
        appearance["correct_before_judge"] = equivalent(appearance["answer"], gold)
        appearance["correct"] = appearance["correct_before_judge"]
    trace["per_agent_correctness"] = {source: equivalent(answer, gold) for source, answer in trace["candidate_answers"].items()}
    if trace.get("finalizer_event"):
        trace["per_agent_correctness"]["finalizer"] = trace["correct"]
    usage, agent_usage, timing = blank_usage(), defaultdict(blank_usage), defaultdict(float)
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"]); add_usage(agent_usage[event["agent"]], event["token_usage"])
        timing[event["phase"]] += event["runtime_seconds"]
    trace.update(inference_token_usage=usage, per_agent_token_usage=dict(agent_usage), phase_runtime_seconds=dict(timing), total_runtime_seconds=time.perf_counter() - started)
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def build_replay_trace(model: LocalQwen, prompts: dict, item: dict, qid: int, setting: str,
                       discussion: dict) -> dict:
    """Finalize one replay condition from an already-created discussion object."""
    started = time.perf_counter()
    facts_ab, facts_ba = replay_facts(item, "AB"), replay_facts(item, "BA")
    ledger = replay_ledger(item)
    old_transcript = public_transcript(discussion.get("discussion_events", []))
    if setting == "before_final_reset":
        evidence_view = f'Canonical fact table:\n{ledger}'
        context_policy = "reset; no prior discussion or candidates"
    elif setting == "before_final_transcript_ledger":
        evidence_view = (f'Newly disclosed facts (verbatim):\n{facts_ab}\nCanonical fact table:\n{ledger}\n'
                         f'Prior discussion transcript:\n{old_transcript}')
        context_policy = "prior discussion plus verbatim facts and canonical ledger"
    elif setting == "before_final_transcript":
        evidence_view = f'Newly disclosed facts (verbatim):\n{facts_ab}\nPrior discussion transcript:\n{old_transcript}'
        context_policy = "prior discussion plus verbatim facts"
    else:
        evidence_view = discussion["public_transcript"]
        context_policy = "facts already present in discussion transcript"

    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence_view}\n'
            'Valid non-empty candidates: {}\nAvailable selected_source values for this question: ["recomputed", "none"]\n'
            'Verifier report: "(no verifier in this setting)"\n'
            "Recompute from the visible evidence. Return exactly the required three-line finalizer format.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, {})
    # Semantic scoring is deliberately independent of the strict three-line
    # format validator. A labeled numeric answer can therefore be correct
    # while the same event is separately counted as format-noncompliant.
    prediction, semantic_extraction = extract_free_text_answer(
        finalizer.get("raw_output", ""), "Final answer")
    gold = extract_answer(item["answer"])
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))
    fact_hash = replay_fact_hash(item)
    discussion_hash = hashlib.sha256(json.dumps(
        [{"actual_messages": event.get("actual_messages"), "raw_output": event.get("raw_output")}
         for event in discussion.get("discussion_events", []) +
         list(discussion.get("solver_finals", {}).values())],
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    trace = {
        "question_id": qid, "setting": setting, "shared_question": item["shared_question"],
        "gold_answer": gold, "discussion": discussion, "discussion_object_id": id(discussion),
        "discussion_trace_hash": discussion_hash,
        "finalizer_event": finalizer, "final_prediction": prediction,
        "semantic_answer_extraction": semantic_extraction,
        "candidate_answers": {}, "information": {"information_complete": True,
        "side_revealed": {"A": True, "B": True}, "assessment_method": "verbatim scheduled injection"},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "injected_fact_hash": fact_hash, "final_received_fact_hash": fact_hash,
        "fact_hash_algorithm": "sha256(canonical-json-sort-keys)",
        "fact_text_order_at_initial_reveal": "BA" if setting == "all_at_start_BA" else "AB",
        "final_context_policy": context_policy, "semantic_correct": semantic_correct,
        "format_compliant": format_compliant, "correct_before_judge": semantic_correct,
        "correct": semantic_correct, "invalid_output": not format_compliant,
        "finalizer_retry_count": 0, "finalizer_recovered": False, "finalizer_exhausted": False,
    }
    trace["candidate_appearances"] = candidate_appearances(trace)
    for appearance in trace["candidate_appearances"]:
        appearance["correct_before_judge"] = equivalent(appearance["answer"], gold)
        appearance["correct"] = appearance["correct_before_judge"]
    trace["per_agent_correctness"] = {"finalizer": semantic_correct}
    usage, agent_usage, timing = blank_usage(), defaultdict(blank_usage), defaultdict(float)
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
        add_usage(agent_usage[event["agent"]], event["token_usage"])
        timing[event["phase"]] += event["runtime_seconds"]
    trace.update(inference_token_usage=usage, per_agent_token_usage=dict(agent_usage),
                 phase_runtime_seconds=dict(timing), total_runtime_seconds=time.perf_counter() - started)
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def deepseek_review(traces: list[dict]) -> tuple[dict, dict, float]:
    entries = []
    for trace_index, trace in enumerate(traces):
        if not trace["correct_before_judge"] and not trace.get("invalid_output"):
            entries.append({"id": f"{trace_index}:final", "setting": trace["setting"], "target": "final", "question": trace["shared_question"],
                            "gold": trace["gold_answer"], "prediction": trace["final_prediction"], "local_correct": False})
        for appearance_index, appearance in enumerate(trace.get("candidate_appearances", [])):
            if appearance.get("answer") and not appearance["correct_before_judge"]:
                entries.append({"id": f"{trace_index}:candidate:{appearance_index}", "setting": trace["setting"], "question": trace["shared_question"],
                                "target": f'{appearance["source"]}/{appearance["phase"]}', "gold": trace["gold_answer"],
                                "prediction": appearance["answer"], "local_correct": False})
        information = trace.get("information", {})
        discussion = trace.get("discussion") or {}
        if information.get("needs_semantic_review") and discussion:
            entries.append({"id": f"{trace_index}:information", "setting": trace["setting"], "target": "information_completeness",
                            "required_facts": information.get("required_fact_units", {}),
                            "public_transcript": discussion.get("public_transcript", ""),
                            "instruction": "Decide whether every required fact is explicitly disclosed or unambiguously paraphrased in the public transcript."})
    if not entries:
        return {}, blank_usage(), 0.0
    load_dotenv, OpenAI = load_api_dependencies()
    load_dotenv(ROOT / ".env", override=True)
    key = next((os.getenv(x) for x in DEEPSEEK_API_KEY_ENV_NAMES if os.getenv(x)), None)
    if not key:
        raise SystemExit("Missing DEEPSEEK_API_KEY or API_KEY in .env.")
    client = OpenAI(api_key=key, base_url=os.getenv("DEEPSEEK_BASE_URL", os.getenv("BASE_URL", DEEPSEEK_BASE_URL)))
    model = os.getenv("DEEPSEEK_MODEL", os.getenv("MODEL_NAME", DEEPSEEK_MODEL))
    user = ("Return a valid JSON object with one result for every id. For answer targets, judge only mathematical equivalence to gold; "
            "format_issue=true only for an actually equivalent representation. For information_completeness targets, set correct=true only "
            "when every required fact is explicitly present or unambiguously paraphrased in the public transcript; do not require matching wording. "
            "Explain missing facts briefly in reason.\n" + json.dumps(entries, ensure_ascii=False))
    started = time.perf_counter()
    last_error = None
    for attempt in range(1, DEFAULT_JUDGE_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Return only valid JSON with schema {"results":[{"id":"0:final","correct":true,"format_issue":false,"reason":""}]}. Preserve every input id exactly.'},
                    {"role": "user", "content": user},
                ],
            )
            parsed = parse_object(response.choices[0].message.content, {"results": []})
            if not isinstance(parsed.get("results"), list):
                raise ValueError("DeepSeek JSON field 'results' is not a list.")
            usage = {k: int(getattr(response.usage, k, 0) or 0) for k in USAGE_KEYS}
            reviews = {str(x["id"]): x for x in parsed["results"] if isinstance(x, dict) and "id" in x}
            return reviews, usage, time.perf_counter() - started
        except Exception as exc:
            last_error = exc
            if attempt < DEFAULT_JUDGE_MAX_ATTEMPTS:
                delay = min(2 ** (attempt - 1), 8)
                print(f"DeepSeek judge failed ({attempt}/{DEFAULT_JUDGE_MAX_ATTEMPTS}); retrying in {delay}s: {exc}")
                time.sleep(delay)
    error = f"DeepSeek judge failed after {DEFAULT_JUDGE_MAX_ATTEMPTS} attempts: {last_error}"
    print(f"WARNING: {error}. Keeping local correctness judgments and saving outputs.")
    return {"__judge_error__": error}, blank_usage(), time.perf_counter() - started


def write_outputs(traces: list[dict], output_dir: Path, run_config: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_config is not None:
        (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "traces_all.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    failure_fields = ("question_id", "setting", "agent_variant", "shared_question", "gold_answer", "final_prediction",
                      "failure_type", "invalid_output", "finalizer_retry_count", "finalizer_recovered", "finalizer_exhausted",
                      "lucky_guess", "oracle_gap", "information", "candidate_answers",
                      "candidate_appearances", "per_agent_correctness", "deepseek_judge")
    failures = [{key: trace[key] for key in failure_fields if key in trace}
                for trace in traces if not trace.get("correct")]
    (output_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    grouped = defaultdict(list)
    for t in traces:
        grouped[(t["setting"], t.get("agent_variant") or "")].append(t)
    fields = ["setting", "agent_variant", "n", "correct", "accuracy", "solver_a_correct", "solver_b_correct", "verifier_correct", "finalizer_correct", "information_complete",
              "fail_information_acquisition", "fail_information_integration", "fail_answer_selection", "invalid_output",
              "finalizer_retry_count", "finalizer_recovered", "finalizer_exhausted",
              "oracle_gap", "oracle_gap_ids", "lucky_guess", "format_issue_corrected",
              "prompt_tokens", "completion_tokens", "total_tokens", "judge_total_tokens",
              "inference_runtime_seconds", "judge_runtime_seconds", "end_to_end_runtime_seconds"]
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=fields); writer.writeheader()
    for (setting, variant), rows in grouped.items():
        correct = sum(bool(x["correct"]) for x in rows)
        writer.writerow({"setting": setting, "agent_variant": variant, "n": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 4),
            "solver_a_correct": sum(bool(x.get("per_agent_correctness", {}).get("solver_a")) for x in rows),
            "solver_b_correct": sum(bool(x.get("per_agent_correctness", {}).get("solver_b")) for x in rows),
            "verifier_correct": sum(bool(x.get("per_agent_correctness", {}).get("verifier")) for x in rows),
            "finalizer_correct": sum(bool(x.get("per_agent_correctness", {}).get("finalizer")) for x in rows),
            "information_complete": sum(bool(x.get("information", {}).get("information_complete")) for x in rows),
            "fail_information_acquisition": sum(x["failure_type"] == "information_acquisition_failure" for x in rows),
            "fail_information_integration": sum(x["failure_type"] == "information_integration_failure" for x in rows),
            "fail_answer_selection": sum(x["failure_type"] == "answer_selection_failure" for x in rows),
            "invalid_output": sum(bool(x.get("invalid_output")) for x in rows),
            "finalizer_retry_count": sum(int(x.get("finalizer_retry_count", 0)) for x in rows),
            "finalizer_recovered": sum(bool(x.get("finalizer_recovered")) for x in rows),
            "finalizer_exhausted": sum(bool(x.get("finalizer_exhausted")) for x in rows),
            "oracle_gap": sum(bool(x.get("oracle_gap")) for x in rows),
            "oracle_gap_ids": ",".join(str(x["question_id"]) for x in rows if x.get("oracle_gap")),
            "lucky_guess": sum(x["lucky_guess"] for x in rows),
            "format_issue_corrected": sum(bool(x.get("deepseek_judge", {}).get("final", {}).get("format_issue")) for x in rows),
            "prompt_tokens": sum(x["inference_token_usage"]["prompt_tokens"] for x in rows), "completion_tokens": sum(x["inference_token_usage"]["completion_tokens"] for x in rows),
            "total_tokens": sum(x["inference_token_usage"]["total_tokens"] for x in rows), "judge_total_tokens": sum(x.get("judge_token_usage", {}).get("total_tokens", 0) for x in rows),
            "inference_runtime_seconds": round(sum(x["total_runtime_seconds"] for x in rows), 3),
            "judge_runtime_seconds": round(sum(x.get("judge_runtime_seconds", 0) for x in rows), 3),
            "end_to_end_runtime_seconds": round(sum(x["total_runtime_seconds"] + x.get("judge_runtime_seconds", 0) for x in rows), 3)})
    (output_dir / "metrics.csv").write_text(buf.getvalue(), encoding="utf-8-sig")


def write_replay_analysis(traces: list[dict], output_dir: Path) -> None:
    """Write paired, offline timing metrics. Correctness never depends on format compliance."""
    rows = [t for t in traces if t["setting"] in REPLAY_SETTINGS]
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    by_setting = {setting: {int(t["question_id"]): t for t in rows if t["setting"] == setting}
                  for setting in REPLAY_SETTINGS}
    common_ids = sorted(set.intersection(*(set(values) for values in by_setting.values()))) if all(by_setting.values()) else []
    accuracy = {}
    for setting in REPLAY_SETTINGS:
        values = list(by_setting[setting].values())
        accuracy[setting] = {
            "n": len(values),
            "correct": sum(bool(t.get("semantic_correct", t.get("correct_before_judge"))) for t in values),
            "accuracy": round(sum(bool(t.get("semantic_correct", t.get("correct_before_judge"))) for t in values) / len(values), 4) if values else 0,
            "format_compliant": sum(bool(t.get("format_compliant", not t.get("invalid_output"))) for t in values),
            "format_compliance_rate": round(sum(bool(t.get("format_compliant", not t.get("invalid_output"))) for t in values) / len(values), 4) if values else 0,
        }
    def ok(setting: str, qid: int) -> bool:
        trace = by_setting[setting][qid]
        return bool(trace.get("semantic_correct", trace.get("correct_before_judge")))
    def answer_key(setting: str, qid: int) -> str:
        prediction = by_setting[setting][qid]["final_prediction"]
        numeric = decimal(prediction)
        return f"decimal:{numeric.normalize()}" if numeric is not None else (
            "text:" + re.sub(r"\s+", " ", extract_answer(prediction).lower()).strip())
    timing_settings = ("all_at_start_AB", "after_round1", "before_final_transcript")
    flips = []
    pairwise_flips = {}
    baseline = "all_at_start_AB"
    for comparison in timing_settings[1:]:
        ids = [qid for qid in common_ids if
               answer_key(baseline, qid) != answer_key(comparison, qid)]
        pairwise_flips[f"{baseline}_vs_{comparison}"] = {
            "count": len(ids), "denominator": len(common_ids),
            "rate": round(len(ids) / len(common_ids), 4) if common_ids else 0, "question_ids": ids}
    for qid in common_ids:
        predictions = {setting: answer_key(setting, qid)
                       for setting in timing_settings}
        if len(set(predictions.values())) > 1:
            flips.append({"question_id": qid, "predictions": predictions})
    late_penalty_ids = [qid for qid in common_ids
                        if ok("all_at_start_AB", qid) and not ok("before_final_transcript", qid)]
    reset_ids = [qid for qid in common_ids if not ok("before_final_transcript", qid)
                 and ok("before_final_reset", qid)]
    ledger_ids = [qid for qid in common_ids if not ok("before_final_transcript", qid)
                  and ok("before_final_transcript_ledger", qid)]
    hashes_consistent = all(len({by_setting[s][qid]["injected_fact_hash"] for s in REPLAY_SETTINGS}) == 1
                            for qid in common_ids)
    if common_ids and not hashes_consistent:
        raise RuntimeError("Replay invariant violated: a question has different injected fact hashes.")
    result = {
        "per_setting": accuracy,
        "paired_question_count": len(common_ids),
        "schedule_flip_rate": {"count": len(flips), "denominator": len(common_ids),
                               "rate": round(len(flips) / len(common_ids), 4) if common_ids else 0,
                               "definition": "prediction changes across AB-ordered all-at-start, after-round1, or before-final-transcript",
                               "question_ids": [x["question_id"] for x in flips], "details": flips,
                               "pairwise": pairwise_flips},
        "late_evidence_penalty": {"count": len(late_penalty_ids), "question_ids": late_penalty_ids},
        "reset_recovery": {"count": len(reset_ids), "question_ids": reset_ids},
        "ledger_recovery": {"count": len(ledger_ids), "question_ids": ledger_ids},
        "fact_hash_consistent_across_six_settings": hashes_consistent,
    }
    (output_dir / "replay_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=("setting", "n", "correct", "accuracy",
                                              "format_compliant", "format_compliance_rate"))
    writer.writeheader()
    for setting in REPLAY_SETTINGS:
        writer.writerow({"setting": setting, **accuracy[setting]})
    (output_dir / "replay_metrics.csv").write_text(buf.getvalue(), encoding="utf-8-sig")


def choose_settings_interactively() -> list[str]:
    options = [(str(index), setting) for index, setting in enumerate(SETTINGS, 1)]
    print("\nSelect one or more Hidden-GSM8K settings:")
    for number, setting in options:
        print(f"{number}. {SETTING_NAMES[setting]} ({setting})")
    aliases = {number: setting for number, setting in options}
    aliases.update({setting: setting for setting in SETTINGS})
    while True:
        choice = input("Enter choices (for example 1,3,4), setting names, or all: ").strip().lower()
        if choice == "all":
            return list(SETTINGS)
        selected = []
        for part in re.split(r"[\s,]+", choice):
            if not part or part not in aliases:
                selected = []
                break
            setting = aliases[part]
            if setting not in selected:
                selected.append(setting)
        if selected:
            return selected
        print(f"Invalid choice. Use numbers 1-{len(options)}, setting names, or all.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hidden-GSM8K local-Qwen multi-agent experiment")
    parser.add_argument("--data-path", default=str(DATA_PATH)); parser.add_argument("--model-path", default=str(MODEL_PATH)); parser.add_argument("--output-dir", default=str(OUTPUT_BASE_DIR))
    setting_group = parser.add_mutually_exclusive_group()
    setting_group.add_argument("--setting", choices=(*SETTINGS, "all"), help="Run one setting, or all settings.")
    setting_group.add_argument("--settings", nargs="+", choices=SETTINGS, help="Run multiple selected settings.")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--discussion-rounds", type=int, default=DEFAULT_DISCUSSION_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--allow-download", action="store_true", default=DEFAULT_ALLOW_DOWNLOAD)
    parser.add_argument("--skip-deepseek", action="store_true", default=DEFAULT_SKIP_DEEPSEEK)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if args.settings:
        selected_settings = list(dict.fromkeys(args.settings))
    elif args.setting:
        selected_settings = list(SETTINGS) if args.setting == "all" else [args.setting]
    elif DEFAULT_SELECTED_SETTINGS:
        invalid_settings = [setting for setting in DEFAULT_SELECTED_SETTINGS if setting not in SETTINGS]
        if invalid_settings:
            parser.error(f"Invalid DEFAULT_SELECTED_SETTINGS: {invalid_settings}")
        selected_settings = list(dict.fromkeys(DEFAULT_SELECTED_SETTINGS))
    elif args.check_config:
        selected_settings = list(SETTINGS)
    else:
        selected_settings = choose_settings_interactively()
    if args.discussion_rounds < 1:
        parser.error("--discussion-rounds must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    data_path, model_path = Path(args.data_path).resolve(), Path(args.model_path).resolve()
    prompts = {name: path.read_text(encoding="utf-8").strip() for name, path in PROMPT_PATHS.items()}
    items = read_json_records(data_path); items = items[:args.limit or None]
    if args.check_config:
        print(json.dumps({"data_path": str(data_path), "records": len(items), "model_path": str(model_path),
                          "model_exists": model_path.exists(), "output_base_dir": str(Path(args.output_dir).resolve()),
                          "prompt_paths": {name: str(path) for name, path in PROMPT_PATHS.items()},
                          "settings": selected_settings, "device": args.device, "temperature": args.temperature,
                          "max_new_tokens": args.max_new_tokens, "discussion_rounds": args.discussion_rounds,
                          "finalizer_max_attempts": DEFAULT_FINALIZER_MAX_ATTEMPTS,
                          "seed": args.seed, "limit": args.limit, "deepseek_enabled": not args.skip_deepseek,
                          "deepseek_base_url": DEEPSEEK_BASE_URL, "deepseek_model": DEEPSEEK_MODEL}, ensure_ascii=False, indent=2)); return
    model = LocalQwen(model_path, args.device, args.max_new_tokens, args.temperature, args.allow_download)
    reseed_model(model, args.seed)
    output_base = Path(args.output_dir).resolve()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dirs = {setting: output_base / (run_stamp if len(selected_settings) == 1 else f"{run_stamp}_{setting}")
                   for setting in selected_settings}
    run_config = {"script": "run_hidden_gsm8k.py", "data_path": str(data_path), "model_path": str(model_path),
                  "settings": selected_settings, "device": args.device, "temperature": args.temperature,
                  "max_new_tokens": args.max_new_tokens, "discussion_rounds": args.discussion_rounds, "seed": args.seed,
                  "finalizer_max_attempts": DEFAULT_FINALIZER_MAX_ATTEMPTS,
                  "deepseek_enabled": not args.skip_deepseek,
                  "shared_discussion_settings": sorted({"multi_partial", "multi_partial_verifier"} & set(selected_settings)),
                  "discussion_reuse_scope": "same partial-information condition only; oracle and single-agent settings are distinct",
                  "discussion_reused_across_selected_settings": len({"multi_partial", "multi_partial_verifier"} & set(selected_settings)) > 1,
                  "seed_scope": "stable SHA-256 derivation by question and generation scope",
                  "replay_settings": list(REPLAY_SETTINGS),
                  "replay_temperature": 0.0,
                  "replay_fact_source": "condition_A/condition_B copied verbatim from the dataset",
                  "replay_gold_visibility": "offline scoring only; never included in actual_messages",
                  "started_at": datetime.now().isoformat(timespec="seconds")}
    traces = []
    for qid, item in enumerate(items, 1):
        cache = {}; question_traces = []
        # Generate one partial-information A/B discussion per question before
        # evaluating any setting, then reuse that exact object for every
        # selected setting that differs only in verifier/finalizer policy.
        if {"multi_partial", "multi_partial_verifier"} & set(selected_settings):
            reseed_model(model, derived_seed(args.seed, qid, "shared_partial_discussion"))
            cache["partial"] = run_discussion(model, prompts["solver"], item, False, args.discussion_rounds)
            add_information_timeline(item, cache["partial"])
        selected_replay = set(REPLAY_SETTINGS) & set(selected_settings)
        if selected_replay:
            if "all_at_start_AB" in selected_replay:
                cache["replay_all_AB"] = run_replay_discussion(
                    model, prompts["solver"], item, 0, "AB", args.discussion_rounds)
            if "all_at_start_BA" in selected_replay:
                cache["replay_all_BA"] = run_replay_discussion(
                    model, prompts["solver"], item, 0, "BA", args.discussion_rounds)
            if "after_round1" in selected_replay:
                cache["replay_after_round1"] = run_replay_discussion(
                    model, prompts["solver"], item, 1, "AB", args.discussion_rounds)
            if selected_replay & {"before_final_transcript", "before_final_transcript_ledger", "before_final_reset"}:
                cache["replay_before_final_shared"] = run_replay_discussion(
                    model, prompts["solver"], item, None, "AB", args.discussion_rounds)
        for setting in selected_settings:
            variants = ("A", "B") if setting == "single_partial" else ("",)
            for variant in variants:
                print(f"[{qid}/{len(items)}] {setting}{'_' + variant if variant else ''}")
                reseed_model(model, derived_seed(args.seed, qid, setting, variant or "default"))
                if setting in REPLAY_SETTINGS:
                    replay_cache_key = {
                        "all_at_start_AB": "replay_all_AB",
                        "all_at_start_BA": "replay_all_BA",
                        "after_round1": "replay_after_round1",
                        "before_final_transcript": "replay_before_final_shared",
                        "before_final_transcript_ledger": "replay_before_final_shared",
                        "before_final_reset": "replay_before_final_shared",
                    }[setting]
                    trace = build_replay_trace(model, prompts, item, qid, setting, cache[replay_cache_key])
                    trace["discussion_cache_key"] = replay_cache_key
                else:
                    trace = build_trace(model, prompts, item, qid, setting, cache, variant, args.discussion_rounds)
                trace["run_config"] = {key: run_config[key] for key in ("model_path", "device", "temperature", "max_new_tokens", "discussion_rounds", "seed")}
                if setting in REPLAY_SETTINGS:
                    trace["run_config"]["temperature"] = 0.0
                question_traces.append(trace)
        if not args.skip_deepseek:
            reviews, judge_usage, judge_time = deepseek_review(question_traces)
            judge_error = reviews.pop("__judge_error__", None)
            for i, trace in enumerate(question_traces):
                fallback_reason = ("invalid finalizer output" if trace.get("invalid_output") else
                                   "skipped: locally correct" if trace["correct_before_judge"] else "missing judge row")
                fallback = {"correct": trace["correct_before_judge"], "format_issue": False, "reason": fallback_reason,
                            "deepseek_reviewed": False}
                final_review = reviews.get(f"{i}:final", fallback)
                if f"{i}:final" in reviews:
                    final_review["deepseek_reviewed"] = True
                candidate_reviews = []
                for appearance_index, appearance in enumerate(trace.get("candidate_appearances", [])):
                    review_id = f"{i}:candidate:{appearance_index}"
                    reason = "skipped: locally correct" if appearance["correct_before_judge"] else "missing judge row"
                    review = reviews.get(review_id, {"correct": appearance["correct_before_judge"],
                                                     "format_issue": False, "reason": reason,
                                                     "deepseek_reviewed": False})
                    if review_id in reviews:
                        review["deepseek_reviewed"] = True
                    appearance["deepseek_judge"] = review
                    appearance["correct"] = as_bool(review.get("correct"), appearance["correct_before_judge"])
                    candidate_reviews.append(review)
                trace["deepseek_judge"] = {"final": final_review, "candidate_appearances": candidate_reviews}
                if judge_error:
                    trace["deepseek_judge_error"] = judge_error
                judged_correct = as_bool(final_review.get("correct"), trace["correct_before_judge"])
                if trace["setting"] in REPLAY_SETTINGS:
                    trace["semantic_correct"] = judged_correct
                    trace["format_compliant"] = not bool(trace.get("invalid_output"))
                    trace["correct"] = judged_correct
                else:
                    trace["correct"] = False if trace.get("invalid_output") else judged_correct
                info_review = reviews.get(f"{i}:information")
                if info_review is not None:
                    trace["information"]["deepseek_semantic_review"] = info_review
                    if as_bool(info_review.get("correct")):
                        trace["information"]["information_complete"] = True
                        trace["information"]["side_revealed"] = {"A": True, "B": True}
                        trace["information"]["assessment_method"] = "DeepSeek semantic fact-disclosure review"
                        for appearance in trace.get("candidate_appearances", []):
                            if appearance.get("phase") in {"solver_final", "verification"}:
                                appearance["information_complete_at_appearance"] = True
                for source in trace["per_agent_correctness"]:
                    matching = [x for x in trace["candidate_appearances"] if x["source"] == source and x["phase"] in {"single_final", "solver_final", "verification"}]
                    if matching:
                        trace["per_agent_correctness"][source] = matching[-1]["correct"]
                if "finalizer" in trace["per_agent_correctness"]:
                    trace["per_agent_correctness"]["finalizer"] = trace["correct"]
                trace["judge_token_usage"] = {
                    key: judge_usage[key] // len(question_traces) + (1 if i < judge_usage[key] % len(question_traces) else 0)
                    for key in USAGE_KEYS
                }
                trace["judge_runtime_seconds"] = judge_time / len(question_traces)
                trace["judge_batch_shared"] = True
                trace["judge_batch_question_id"] = qid
                trace["failure_type"], trace["lucky_guess"] = classify(trace, trace["gold_answer"])
        for trace in question_traces:
            # Historical/original oracle-gap definition: a correct answer was
            # available from a solver turn/final or verifier, but the finalizer lost it.
            trace["oracle_gap"] = bool(not trace["correct"] and any(
                x.get("source") in {"solver_a", "solver_b", "verifier"} and bool(x.get("correct"))
                for x in trace.get("candidate_appearances", [])))
            trace["failure_type"], trace["lucky_guess"] = classify(trace, trace["gold_answer"])
        traces.extend(question_traces)
        for setting, directory in output_dirs.items():
            setting_config = dict(run_config, setting=setting, output_dir=str(directory))
            write_outputs([x for x in traces if x["setting"] == setting], directory, setting_config)
        write_replay_analysis(traces, output_base / f"{run_stamp}_replay_analysis")
    for setting, directory in output_dirs.items():
        setting_config = dict(run_config, setting=setting, output_dir=str(directory))
        write_outputs([x for x in traces if x["setting"] == setting], directory, setting_config)
        print(f"Wrote {sum(x['setting'] == setting for x in traces)} {setting} traces to {directory}")
    write_replay_analysis(traces, output_base / f"{run_stamp}_replay_analysis")


if __name__ == "__main__":
    main()
