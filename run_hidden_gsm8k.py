"""Hidden-GSM8K: controlled partial-information multi-agent evaluation on local Qwen."""
from __future__ import annotations

import argparse
import csv
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
DATA_PATH = ROOT / "data" / "20.json"
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

# Leave empty to show the interactive setting menu. Example:
# DEFAULT_SELECTED_SETTINGS = ("multi_partial", "multi_partial_verifier")
DEFAULT_SELECTED_SETTINGS: tuple[str, ...] = ()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_API_KEY_ENV_NAMES = ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY")

SETTINGS = ("single_full", "single_partial", "multi_partial", "multi_partial_verifier", "oracle_broadcast")
SETTING_NAMES = {
    "single_full": "Single Agent - Full Information",
    "single_partial": "Single Agent - Partial Information (A and B)",
    "multi_partial": "Multi-Agent - Partial Information",
    "multi_partial_verifier": "Multi-Agent - Partial Information + Verifier",
    "oracle_broadcast": "Oracle Broadcast",
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


def blank_usage() -> dict:
    return {k: 0 for k in USAGE_KEYS}


def add_usage(target: dict, usage: dict) -> None:
    for key in USAGE_KEYS:
        target[key] = target.get(key, 0) + int(usage.get(key, 0) or 0)


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

    def call(self, system: str, user: str) -> tuple[str, dict, float]:
        return self.call_batch([(system, user)])[0]

    def call_batch(self, requests: list[tuple[str, str]]) -> list[tuple[str, dict, float]]:
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
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=.9)
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
FINALIZER_DEFAULT = {"final_answer": "", "selected_source": "recomputed", "used_public_facts": [], "followed_verifier": False, "reason": ""}


def model_event(model: LocalQwen, agent: str, system: str, user: str, phase: str, parser_defaults: dict | None) -> dict:
    raw, usage, elapsed = model.call(system, user)
    if parser_defaults is None:
        return {"agent": agent, "phase": phase, "actual_input": user, "output": raw,
                "raw_output": raw, "token_usage": usage, "runtime_seconds": elapsed}
    try:
        parsed, parse_error = parse_object(raw, parser_defaults), ""
    except (ValueError, json.JSONDecodeError) as exc:
        parsed, parse_error = dict(parser_defaults), str(exc)
    return {"agent": agent, "phase": phase, "actual_input": user, "raw_output": raw,
            "parsed_output": parsed, "parse_error": parse_error, "token_usage": usage, "runtime_seconds": elapsed}


def paired_model_events(model: LocalQwen, system: str, specs: dict[str, tuple[str, str, dict | None]]) -> dict[str, dict]:
    """Run A/B in one model batch when supported; fall back only for test doubles."""
    sides = ("A", "B")
    if hasattr(model, "call_batch"):
        results = model.call_batch([(system, specs[side][1]) for side in sides])
        events = {}
        for side, (raw, usage, elapsed) in zip(sides, results):
            agent, user, defaults = specs[side]
            if defaults is None:
                events[side] = {"agent": agent, "phase": "", "actual_input": user, "output": raw,
                                "raw_output": raw, "token_usage": usage, "runtime_seconds": elapsed,
                                "generated_in_parallel_batch": True}
                continue
            else:
                try:
                    parsed, parse_error = parse_object(raw, defaults), ""
                except (ValueError, json.JSONDecodeError) as exc:
                    parsed, parse_error = dict(defaults), str(exc)
            events[side] = {"agent": agent, "phase": "", "actual_input": user, "raw_output": raw,
                            "parsed_output": parsed, "parse_error": parse_error, "token_usage": usage,
                            "runtime_seconds": elapsed, "generated_in_parallel_batch": True}
        return events
    return {side: model_event(model, specs[side][0], system, specs[side][1], "", specs[side][2]) for side in sides}


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
        # A and B send in one GPU batch from the exact same pre-round public
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
                    "state what information is missing, and respond to earlier claims when present. Write directly to the other solver in natural text; "
                    "do not output JSON. You cannot see the peer's same-round message.")
            outbound_specs[side] = (f"solver_{side.lower()}", user, None)
        outbound = paired_model_events(model, solver_prompt, outbound_specs)
        for side in ("A", "B"):
            event = outbound[side]
            event["phase"] = f"discussion_round_{round_no}_send"
            event["round"] = round_no
            event["stage"] = "send"
        events.extend([outbound["A"], outbound["B"]])

        # Message delivery is the only cross-agent channel. Each solver keeps
        # its own sent message locally and receives exactly the peer's public
        # output; no private memory or hidden state is synchronized.
        review_specs = {}
        for side in ("A", "B"):
            peer = "B" if side == "A" else "A"
            user = (f'Role: solver_{side.lower()}\nReview round: {round_no} of {rounds_count}\n'
                    f'Shared question: {item["shared_question"]}\nYour private fact: {item[f"condition_{side}"]}\n'
                    f'Public transcript through the previous completed round:\n{pre_round_transcript}\n'
                    f'Your sent message this round:\n{outbound[side]["raw_output"]}\n'
                    f'Peer message received from solver_{peer.lower()}:\n{outbound[peer]["raw_output"]}\n'
                    f'Review solver_{peer.lower()}\'s message against your own facts. Check calculations, resolve contradictions, identify remaining gaps, '
                    "and give the best updated reasoning or answer. Address the peer directly in natural text; do not output JSON.")
            review_specs[side] = (f"solver_{side.lower()}", user, None)
        reviews = paired_model_events(model, solver_prompt, review_specs)
        for side in ("A", "B"):
            event = reviews[side]
            event["phase"] = f"discussion_round_{round_no}_review"
            event["round"] = round_no
            event["stage"] = "review"
        events.extend([reviews["A"], reviews["B"]])
        round_records.append({"round": round_no, "purpose": purpose,
                              "pre_round_public_transcript": pre_round_transcript,
                              "simultaneous_send": {side.lower(): outbound[side] for side in ("A", "B")},
                              "message_delivery": {
                                  "solver_a_received_from_solver_b": outbound["B"]["raw_output"],
                                  "solver_b_received_from_solver_a": outbound["A"]["raw_output"],
                              },
                              "simultaneous_review": {side.lower(): reviews[side] for side in ("A", "B")}})

    final_specs = {}
    transcript = "\n".join(x for x in (oracle_text, public_transcript(events)) if x)
    for side in ("A", "B"):
        user = (f'Role: solver_{side.lower()}\nShared question: {item["shared_question"]}\n'
                f'Your private fact: {item[f"condition_{side}"]}\n'
                f'Public transcript after {rounds_count} symmetric rounds:\n{transcript}\n'
                "Solve the complete problem using all available information. Show a concise calculation and end with `Final answer: ...`. "
                "Use natural text; do not output JSON.")
        final_specs[side] = (f"solver_{side.lower()}", user, None)
    final_batch = paired_model_events(model, solver_prompt, final_specs)
    finals = {}
    for side in ("A", "B"):
        final_batch[side]["phase"] = "solver_final"
        finals[side.lower()] = final_batch[side]
    result = {"protocol": "symmetric_free_text_send_then_review", "round_records": round_records,
              "symmetry_guarantees": {"same_round_send_uses_identical_public_snapshot": True,
                                       "same_round_reviews_do_not_see_each_other": True,
                                       "paired_solver_generation_uses_one_gpu_batch": True,
                                       "cross_agent_channel_is_raw_public_text_only": True},
              "discussion_events": events, "public_transcript": transcript,
              "solver_finals": finals}
    if oracle:
        result["oracle_public_information"] = oracle_text
    return result


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
            "information_complete": all(revealed.values()), "assessment_method": "atomic lexical+numeric coverage"}


def add_information_timeline(item: dict, discussion: dict) -> None:
    oracle = discussion.get("oracle_public_information", "")
    events = discussion.get("discussion_events", [])
    timeline, accumulated = [], []

    def checkpoint(label: str, round_no: int | None) -> dict:
        public = "\n".join(x for x in (oracle, public_transcript(accumulated)) if x)
        snapshot = objective_information(item, {"public_transcript": public})
        row = {"checkpoint": label, "round": round_no, "public_event_count": len(accumulated),
               "information_complete": snapshot["information_complete"], "side_revealed": snapshot["side_revealed"]}
        timeline.append(row)
        return row

    state = checkpoint("discussion_start", None)
    for round_no in sorted({event["round"] for event in events}):
        sends = [event for event in events if event["round"] == round_no and event.get("stage") == "send"]
        reviews = [event for event in events if event["round"] == round_no and event.get("stage") == "review"]
        for event in sends:
            event["information_complete_at_generation"] = state["information_complete"]
        accumulated.extend(sends)
        state = checkpoint("after_simultaneous_send", round_no)
        for event in reviews:
            event["information_complete_at_generation"] = state["information_complete"]
        accumulated.extend(reviews)
        state = checkpoint("after_simultaneous_review", round_no)
    discussion["information_timeline"] = timeline
    first = next((x for x in timeline if x["information_complete"]), None)
    discussion["first_complete_checkpoint"] = first
    discussion["first_complete_after_public_event"] = first["public_event_count"] if first else None


def single_call(model: LocalQwen, prompt: str, item: dict, side: str | None) -> dict:
    if side is None:
        # Only the full-information setting receives the complete problem.
        user = (f'Role: solver_a\nFull question: {item["full"]}\n'
                'Solve the complete problem carefully. Show a concise calculation and end with `Final answer: ...`. Use natural text; do not output JSON.')
    else:
        user = (f'Role: solver_{side.lower()}\nShared question: {item["shared_question"]}\n'
                f'Your private fact: {item[f"condition_{side}"]}\nAnalyze what can be concluded and clearly identify missing information. '
                'If the answer is determined, show a concise calculation and end with `Final answer: ...`; otherwise say that it cannot yet be determined. '
                'Use natural text; do not output JSON.')
    return model_event(model, f'solver_{(side or "a").lower()}', prompt, user, "single_final", None)


def event_answer(event: dict | None, key: str = "final_answer") -> str:
    event = event or {}
    parsed = event.get("parsed_output", {})
    if key in parsed:
        return str(parsed.get(key, "")).strip()
    return extract_answer(event.get("raw_output", ""))


def candidate_appearances(trace: dict) -> list[dict]:
    """Answers visible before final selection, with information state at appearance time."""
    if trace.get("single_event"):
        answer = event_answer(trace["single_event"])
        return [{"source": trace["single_event"]["agent"], "phase": "single_final", "answer": answer,
                 "information_complete_at_appearance": trace["information"]["information_complete"]}]
    discussion = trace.get("discussion") or {}
    appearances = []
    # Free-form discussion messages are preserved verbatim but are not treated
    # as formal candidates. Only each solver's explicit final response enters
    # answer-selection metrics.
    complete_after_discussion = bool(trace.get("information", {}).get("information_complete"))
    for side in ("a", "b"):
        event = discussion.get("solver_finals", {}).get(side)
        if event:
            appearances.append({"source": event["agent"], "phase": "solver_final", "answer": event_answer(event),
                                "information_complete_at_appearance": complete_after_discussion})
    verifier = trace.get("verifier_event")
    if verifier:
        appearances.append({"source": "verifier", "phase": "verification", "answer": event_answer(verifier, "verified_answer"),
                            "information_complete_at_appearance": complete_after_discussion})
    return appearances


def finish_multi(model: LocalQwen, prompts: dict, item: dict, discussion: dict, with_verifier: bool) -> tuple[dict | None, dict]:
    candidates = {"solver_a": event_answer(discussion["solver_finals"]["a"]), "solver_b": event_answer(discussion["solver_finals"]["b"])}
    verifier = None
    if with_verifier:
        user = f'Shared question: {item["shared_question"]}\nPublic transcript:\n{discussion["public_transcript"]}\nCandidates: {json.dumps(candidates, ensure_ascii=False)}'
        verifier = model_event(model, "verifier", prompts["verifier"], user, "verification", VERIFIER_DEFAULT)
    report = verifier["parsed_output"] if verifier else "(no verifier in this setting)"
    user = (f'Shared question: {item["shared_question"]}\nPublic transcript:\n{discussion["public_transcript"]}\n'
            f'Candidates: {json.dumps(candidates, ensure_ascii=False)}\nVerifier report: {json.dumps(report, ensure_ascii=False)}')
    finalizer = model_event(model, "finalizer", prompts["finalizer"], user, "finalization", FINALIZER_DEFAULT)
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
        prediction = event_answer(event)
        trace.update(single_event=event, final_prediction=prediction, candidate_answers={event["agent"]: prediction}, information={"information_complete": side is None, "side_revealed": {"A": side is None or side == "A", "B": side is None or side == "B"}})
    else:
        cache_key = "oracle" if setting == "oracle_broadcast" else "partial"
        if cache_key not in discussion_cache:
            discussion_cache[cache_key] = run_discussion(model, prompts["solver"], item, cache_key == "oracle", discussion_rounds)
            add_information_timeline(item, discussion_cache[cache_key])
        discussion = discussion_cache[cache_key]
        with_verifier = setting in {"multi_partial_verifier", "oracle_broadcast"}
        verifier, finalizer = finish_multi(model, prompts, item, discussion, with_verifier)
        candidates = {"solver_a": event_answer(discussion["solver_finals"]["a"]), "solver_b": event_answer(discussion["solver_finals"]["b"])}
        if verifier:
            candidates["verifier"] = event_answer(verifier, "verified_answer")
        trace.update(discussion=discussion, discussion_cache_key=cache_key, finalizer_event=finalizer,
                     final_prediction=event_answer(finalizer), candidate_answers=candidates, information=objective_information(item, discussion))
        if verifier is not None:
            trace["verifier_event"] = verifier
    trace["correct_before_judge"] = equivalent(trace["final_prediction"], gold)
    trace["correct"] = trace["correct_before_judge"]
    trace["candidate_appearances"] = candidate_appearances(trace)
    for appearance in trace["candidate_appearances"]:
        appearance["correct_before_judge"] = equivalent(appearance["answer"], gold)
        appearance["correct"] = appearance["correct_before_judge"]
    trace["per_agent_correctness"] = {source: equivalent(answer, gold) for source, answer in trace["candidate_answers"].items()}
    if trace.get("finalizer_event"):
        trace["per_agent_correctness"]["finalizer"] = trace["correct_before_judge"]
    usage, agent_usage, timing = blank_usage(), defaultdict(blank_usage), defaultdict(float)
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"]); add_usage(agent_usage[event["agent"]], event["token_usage"])
        timing[event["phase"]] += event["runtime_seconds"]
    trace.update(inference_token_usage=usage, per_agent_token_usage=dict(agent_usage), phase_runtime_seconds=dict(timing), total_runtime_seconds=time.perf_counter() - started)
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def deepseek_review(traces: list[dict]) -> tuple[dict, dict, float]:
    entries = []
    for trace_index, trace in enumerate(traces):
        if not trace["correct_before_judge"]:
            entries.append({"id": f"{trace_index}:final", "setting": trace["setting"], "target": "final", "question": trace["shared_question"],
                            "gold": trace["gold_answer"], "prediction": trace["final_prediction"], "local_correct": False})
        for appearance_index, appearance in enumerate(trace.get("candidate_appearances", [])):
            if appearance.get("answer") and not appearance["correct_before_judge"]:
                entries.append({"id": f"{trace_index}:candidate:{appearance_index}", "setting": trace["setting"], "question": trace["shared_question"],
                                "target": f'{appearance["source"]}/{appearance["phase"]}', "gold": trace["gold_answer"],
                                "prediction": appearance["answer"], "local_correct": False})
    if not entries:
        return {}, blank_usage(), 0.0
    load_dotenv, OpenAI = load_api_dependencies()
    load_dotenv(ROOT / ".env", override=True)
    key = next((os.getenv(x) for x in DEEPSEEK_API_KEY_ENV_NAMES if os.getenv(x)), None)
    if not key:
        raise SystemExit("Missing DEEPSEEK_API_KEY or API_KEY in .env.")
    client = OpenAI(api_key=key, base_url=os.getenv("DEEPSEEK_BASE_URL", os.getenv("BASE_URL", DEEPSEEK_BASE_URL)))
    model = os.getenv("DEEPSEEK_MODEL", os.getenv("MODEL_NAME", DEEPSEEK_MODEL))
    user = ("Return a valid JSON object. Review only the locally incorrect GSM8K predictions below. Judge only mathematical equivalence "
            "to gold, not reasoning quality. format_issue=true only when the prediction is actually equivalent "
            "because of formatting, units, wording, fractions, or another equivalent representation. Return one JSON result for every id.\n" + json.dumps(entries, ensure_ascii=False))
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
                      "failure_type", "lucky_guess", "oracle_gap", "information", "candidate_answers",
                      "candidate_appearances", "per_agent_correctness", "deepseek_judge")
    failures = [{key: trace[key] for key in failure_fields if key in trace}
                for trace in traces if not trace.get("correct")]
    (output_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    grouped = defaultdict(list)
    for t in traces:
        grouped[(t["setting"], t.get("agent_variant") or "")].append(t)
    fields = ["setting", "agent_variant", "n", "correct", "accuracy", "solver_a_correct", "solver_b_correct", "verifier_correct", "finalizer_correct", "information_complete",
              "fail_information_acquisition", "fail_information_integration", "fail_answer_selection", "lucky_guess", "format_issue_corrected",
              "oracle_accuracy", "oracle_gap_accuracy", "oracle_pairwise_rescues", "prompt_tokens", "completion_tokens", "total_tokens", "judge_total_tokens",
              "inference_runtime_seconds", "judge_runtime_seconds", "end_to_end_runtime_seconds"]
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=fields); writer.writeheader()
    oracle = {t["question_id"]: t["correct"] for t in traces if t["setting"] == "oracle_broadcast"}
    for (setting, variant), rows in grouped.items():
        correct = sum(bool(x["correct"]) for x in rows)
        oracle_rows = [oracle[x["question_id"]] for x in rows if x["question_id"] in oracle]
        oracle_accuracy = sum(oracle_rows) / len(oracle_rows) if oracle_rows else None
        accuracy = correct / len(rows)
        writer.writerow({"setting": setting, "agent_variant": variant, "n": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 4),
            "solver_a_correct": sum(bool(x.get("per_agent_correctness", {}).get("solver_a")) for x in rows),
            "solver_b_correct": sum(bool(x.get("per_agent_correctness", {}).get("solver_b")) for x in rows),
            "verifier_correct": sum(bool(x.get("per_agent_correctness", {}).get("verifier")) for x in rows),
            "finalizer_correct": sum(bool(x.get("per_agent_correctness", {}).get("finalizer")) for x in rows),
            "information_complete": sum(bool(x.get("information", {}).get("information_complete")) for x in rows),
            "fail_information_acquisition": sum(x["failure_type"] == "information_acquisition_failure" for x in rows),
            "fail_information_integration": sum(x["failure_type"] == "information_integration_failure" for x in rows),
            "fail_answer_selection": sum(x["failure_type"] == "answer_selection_failure" for x in rows), "lucky_guess": sum(x["lucky_guess"] for x in rows),
            "format_issue_corrected": sum(bool(x.get("deepseek_judge", {}).get("final", {}).get("format_issue")) for x in rows),
            "oracle_accuracy": "" if oracle_accuracy is None else round(oracle_accuracy, 4),
            "oracle_gap_accuracy": "" if oracle_accuracy is None else round(oracle_accuracy - accuracy, 4),
            "oracle_pairwise_rescues": "" if not oracle_rows else sum(bool(oracle.get(x["question_id"])) and not x["correct"] for x in rows),
            "prompt_tokens": sum(x["inference_token_usage"]["prompt_tokens"] for x in rows), "completion_tokens": sum(x["inference_token_usage"]["completion_tokens"] for x in rows),
            "total_tokens": sum(x["inference_token_usage"]["total_tokens"] for x in rows), "judge_total_tokens": sum(x.get("judge_token_usage", {}).get("total_tokens", 0) for x in rows),
            "inference_runtime_seconds": round(sum(x["total_runtime_seconds"] for x in rows), 3),
            "judge_runtime_seconds": round(sum(x.get("judge_runtime_seconds", 0) for x in rows), 3),
            "end_to_end_runtime_seconds": round(sum(x["total_runtime_seconds"] + x.get("judge_runtime_seconds", 0) for x in rows), 3)})
    (output_dir / "metrics.csv").write_text(buf.getvalue(), encoding="utf-8-sig")


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
                          "seed": args.seed, "limit": args.limit, "deepseek_enabled": not args.skip_deepseek,
                          "deepseek_base_url": DEEPSEEK_BASE_URL, "deepseek_model": DEEPSEEK_MODEL}, ensure_ascii=False, indent=2)); return
    model = LocalQwen(model_path, args.device, args.max_new_tokens, args.temperature, args.allow_download)
    random.seed(args.seed); model.torch.manual_seed(args.seed)
    if model.torch.cuda.is_available(): model.torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config = {"script": "run_hidden_gsm8k.py", "data_path": str(data_path), "model_path": str(model_path),
                  "settings": selected_settings, "device": args.device, "temperature": args.temperature,
                  "max_new_tokens": args.max_new_tokens, "discussion_rounds": args.discussion_rounds, "seed": args.seed,
                  "deepseek_enabled": not args.skip_deepseek, "started_at": datetime.now().isoformat(timespec="seconds")}
    traces = []
    for qid, item in enumerate(items, 1):
        cache = {}; question_traces = []
        for setting in selected_settings:
            variants = ("A", "B") if setting == "single_partial" else ("",)
            for variant in variants:
                print(f"[{qid}/{len(items)}] {setting}{'_' + variant if variant else ''}")
                trace = build_trace(model, prompts, item, qid, setting, cache, variant, args.discussion_rounds)
                trace["run_config"] = {key: run_config[key] for key in ("model_path", "device", "temperature", "max_new_tokens", "discussion_rounds", "seed")}
                question_traces.append(trace)
        if not args.skip_deepseek:
            reviews, judge_usage, judge_time = deepseek_review(question_traces)
            judge_error = reviews.pop("__judge_error__", None)
            for i, trace in enumerate(question_traces):
                fallback_reason = "skipped: locally correct" if trace["correct_before_judge"] else "missing judge row"
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
                trace["correct"] = as_bool(final_review.get("correct"), trace["correct_before_judge"])
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
        oracle_row = next((x for x in question_traces if x["setting"] == "oracle_broadcast"), None)
        if oracle_row is not None:
            for trace in question_traces:
                trace["oracle_gap"] = bool(oracle_row["correct"] and not trace["correct"])
        traces.extend(question_traces); write_outputs(traces, output_dir, run_config)
    write_outputs(traces, output_dir, run_config); print(f"Wrote {len(traces)} traces to {output_dir}")


if __name__ == "__main__":
    main()
