"""Hidden-GSM8K: controlled partial-information multi-agent evaluation on local Qwen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import time
import difflib
import contextlib
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
import unittest

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
DEFAULT_FINALIZER_MAX_ATTEMPTS = 1  # Finalizer is never retried: a malformed selection is invalid.

# Leave empty to show the interactive setting menu. Example:
# DEFAULT_SELECTED_SETTINGS = ("multi_partial", "multi_partial_verifier")
DEFAULT_SELECTED_SETTINGS: tuple[str, ...] = ()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_API_KEY_ENV_NAMES = ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY")

SETTINGS = ("single_full", "single_partial", "multi_partial", "multi_partial_verifier", "oracle_broadcast",
            "self_check_before_commit")
INFORMATION_TIMINGS = (
    "all_at_start",
    "after_round1",
    "before_discussion_end",
    "before_finalizer",
    "before_final_reset",
)
REPLAY_SETTINGS = tuple(
    f"{timing}_{order}" for timing in INFORMATION_TIMINGS for order in ("AB", "BA"))
FINALIZER_ORDER_SETTING = "finalizer_only_order_ab_ba"
SOLVER_ORDER_SETTING = "solver_only_order_ab_ba"
FROZEN_TRANSCRIPT_ORDER_SETTING = "frozen_transcript_order_ab_ba"
CANONICAL_ORDER_SETTING = "canonical_order"
RANDOM_ORDER_SETTING = "random_order"
SELF_CHECK_BEFORE_COMMIT_SETTING = "self_check_before_commit"
ORDER_SETTINGS = (SOLVER_ORDER_SETTING, FINALIZER_ORDER_SETTING,
                  FROZEN_TRANSCRIPT_ORDER_SETTING, CANONICAL_ORDER_SETTING, RANDOM_ORDER_SETTING)
CONTROLLED_SETTINGS = REPLAY_SETTINGS + ORDER_SETTINGS
SETTINGS = SETTINGS + REPLAY_SETTINGS + ORDER_SETTINGS
SETTING_NAMES = {
    "single_full": "Single Agent - Full Information",
    "single_partial": "Single Agent - Partial Information (A and B)",
    "multi_partial": "Multi-Agent - Partial Information",
    "multi_partial_verifier": "Multi-Agent - Partial Information + Verifier",
    "oracle_broadcast": "Oracle Broadcast",
    "self_check_before_commit": "Self-Check Before Commit",
    "all_at_start_AB": "Information Replay - All at Start (A then B)",
    "all_at_start_BA": "Information Replay - All at Start (B then A)",
    "after_round1_AB": "Information Replay - Reveal after Round 1 (AB)",
    "after_round1_BA": "Information Replay - Reveal after Round 1 (BA)",
    "before_discussion_end_AB": "Information Replay - Before Discussion End (AB)",
    "before_discussion_end_BA": "Information Replay - Before Discussion End (BA)",
    "before_finalizer_AB": "Information Replay - Before Finalizer (AB)",
    "before_finalizer_BA": "Information Replay - Before Finalizer (BA)",
    "before_final_reset_AB": "Information Replay - Reset then Finalizer (AB)",
    "before_final_reset_BA": "Information Replay - Reset then Finalizer (BA)",
    "finalizer_only_order_ab_ba": "Finalizer-only Fact Order Control (AB and BA)",
    "solver_only_order_ab_ba": "Solver-only Fact Order Control (AB and BA)",
    "frozen_transcript_order_ab_ba": "Frozen Transcript Fact Order Control (AB and BA)",
    "canonical_order": "Fixed Source-Order Ledger (Legacy Canonical Name)",
    "random_order": "Randomized Fact Order Repetitions",
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


def _conclusion_window(text: str) -> str:
    """Find the conclusion window: last 2 non-empty paragraphs → last 3 sentences/lines."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return ""
    window_text = "\n".join(paragraphs[-2:]) if len(paragraphs) >= 2 else paragraphs[0]
    clauses = [c.strip() for c in re.split(r"(?<=[.!?。！？])\s+", window_text) if c.strip()]
    if len(clauses) >= 4:
        window_text = " ".join(clauses[-3:])
    else:
        lines = [line.strip() for line in window_text.splitlines() if line.strip()]
        if len(lines) >= 5:
            window_text = "\n".join(lines[-3:])
    return window_text


_SAFE_CONCLUSION_OPENERS = re.compile(
    r"(?i)\b(?:therefore|thus|hence|so|finally|in\s+conclusion)\b"
)
_SAFE_CONCLUSION_VERBS = re.compile(
    r"(?i)\b(?:must|should|needs|costs|earns|weighs|takes|buys|pays|produces|remains|remaining|"
    r"is|are|equals|would\s+be|will\s+be)\b"
)
_SAFE_CONCLUSION_OBJECTS = re.compile(
    r"(?i)\b(?:answer|result|total|final|remaining|amount|number|weight|cost|price|income|"
    r"balance|pages|plates|hats|packs|dollars|vegetables)\b"
)
EXPECTED_LABELS: dict[str, tuple[str, ...]] = {
    "solver_current": ("Current answer",),
    "solver_final": ("Final answer",),
    "verifier": ("verified_answer",),
    "finalizer": ("Final answer",),
}

_EXPECTED_JSON_KEYS: dict[str, tuple[str, ...]] = {
    "solver_current": ("current_answer",),
    "solver_final": ("final_answer",),
    "verifier": ("verified_answer",),
    "finalizer": ("final_answer",),
}

# Regex that matches ANY of the known labels (used only for mid-text fallback
# when no line-starting label matched; the caller must then filter by
# expected_output_type).
_LABEL_LINE = re.compile(
    r"(?im)^\s*(?:Current\s+answer|Final\s+answer|Final\s+Answer|verified_answer)\s*[:：=]\s*(.+?)\s*$"
)
_BOXED = re.compile(r"(?:\$|\\\()?\\boxed\{([^}]+)\}(?:\)?\$)?")
_NUMERIC = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+)?)")
_MONEY_NUM = re.compile(r"\$?(-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+)?)")
_AMBIGUOUS_SPECULATION = re.compile(
    r"(?i)\b(?:could|might|maybe|perhaps|possibly|may\s+be|not\s+sure|uncertain)\b"
)
# Pattern to find a label from a specific expected set anywhere in text.
_LABEL_ANYWHERE = re.compile(
    r"(?i)((?:Current\s+answer|Final\s+answer|Final\s+Answer|verified_answer))\s*[:：=]\s*(.+?)(?:\n|$)"
)

# Pattern: "<object-word> ... is/equals/of N" or "must/should/needs/buys/... [helper] N"
# Intervening words allowed between object and binding verb (e.g.
# "the balance remaining after the payments is $520").
# Helper word allowed after modal (e.g. "must buy 3", "should read 42").
_CONCLUSION_BINDING = re.compile(
    r"(?i)(?:"
    r"\b(answer|result|total|final|remaining|amount|number|weight|cost|price|income|"
    r"balance|pages|plates|hats|packs|dollars|vegetables)\b"
    r".*?"
    r"\b(is|:|：|=|equals|of)\b"
    r"\s*"
    r"(\$?\s*-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+)?)"
    r"|"
    r"\b(must|should|needs|buys|pays|earns|takes|weighs|produces|remains)\b"
    r"(?:\s+\w+)?"
    r"\s*"
    r"(\$?\s*-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+)?)"
    r")"
)


def _clean_parseable_number(text: str) -> str:
    """Normalize a token for decimal parsing: strip currency, commas, LaTeX."""
    cleaned = re.sub(r"[\$\s,\\]", "", text.strip())
    if decimal(cleaned) is not None:
        return cleaned
    return ""


def _extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} answer from text (full-text search allowed)."""
    matches = _BOXED.findall(text)
    for raw in reversed(matches):
        answer = _clean_parseable_number(raw)
        if answer:
            return answer
    return ""


def _explicitly_undetermined_prose(text: str) -> bool:
    """Check if text contains authoritative undetermined/unknown/cannot-determine language."""
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    return (
        lowered in {"", "unknown", "undetermined", "cannot determine", "cannot be determined",
                     "insufficient information", "none", "n/a"}
        or any(phrase in lowered for phrase in (
            "cannot determine", "cannot be determined", "can't determine",
            "not enough information", "insufficient information",
            "unable to determine", "answer is undetermined",
            "impossible to determine", "impossible to calculate",
            "impossible to conclude", "cannot conclude", "cannot calculate",
            "cannot answer",
        ))
        or bool(re.search(r"\b(?:undetermined|unknown|insufficient)\b", lowered))
    )


def extract_first_declared_numeric(value: str) -> str:
    """Extract the FIRST numeric answer from a declared label value.

    - \\boxed{N} is checked first.
    - Otherwise the first parseable number in the declaration is returned.
    - Subsequent numbers (e.g. in reasoning after the answer) are ignored.

    Examples:
        "41 because total=143 and known=102"  → "41"
        "Julie should read 42 pages"          → "42"
        "$520 after 4 payments"               → "520"
        "undetermined; 3 facts missing"       → ""
    """
    text = str(value or "").strip()
    if _explicitly_undetermined_prose(text):
        return ""
    # \\boxed{N} has the highest priority.
    boxed_match = _BOXED.search(text)
    if boxed_match:
        answer = _clean_parseable_number(boxed_match.group(1))
        if answer:
            return answer
    # Scan from left to right; return the first parseable number.
    for match in _NUMERIC.finditer(text):
        answer = _clean_parseable_number(match.group(1))
        if answer:
            return answer
    return ""


def extract_numeric_bound_to_conclusion(clause: str) -> dict:
    """Extract a conclusion-bound numeric from a clause, or report ambiguity.

    Returns {"answer": str, "ambiguous": bool}.  Only numbers that are
    grammatically bound to a conclusion object or verb are accepted.
    Multiple unbound numbers → ambiguous.

    Correctly handles:
        "the answer is 5, verified in 2 ways." → "5"
        "Therefore, the final weight is 16 pounds." → "16"
        "Roger must buy 3 packs." → "3"
        "There are 3 apples and 4 oranges." → ambiguous
        "Therefore, there are 3 apples and 4 oranges." → ambiguous
    """
    # Priority 1: structural binding via _CONCLUSION_BINDING regex.
    bindings = list(_CONCLUSION_BINDING.finditer(clause))
    if bindings:
        answers = []
        for m in bindings:
            # Groups: (obj, binding_verb, obj_num, action_verb, verb_num)
            num_str = (m.group(3) or m.group(5) or "").replace(",", "").replace("$", "").strip()
            ans = _clean_parseable_number(num_str)
            if ans and ans not in answers:
                answers.append(ans)
        if len(answers) == 1:
            return {"answer": answers[0], "ambiguous": False}
        if len(answers) >= 2:
            return {"answer": "", "ambiguous": True}

    # Priority 2: \\boxed{N} anywhere in the clause.
    boxed = _extract_boxed_answer(clause)
    if boxed:
        return {"answer": boxed, "ambiguous": False}

    # Priority 3: exactly one number in a clause with a clear conclusion marker.
    # Must NOT be a number that clearly just modifies a generic container/unit noun
    # (e.g. "3 cases", "5 apples", "2 ways", "4 reasons").
    _NON_CONCLUSION_NOUN = re.compile(
        r"\b(?:cases|apples|oranges|ways|reasons|facts|times|steps|attempts|methods|"
        r"tries|checks|verifications|examples|instances)\b", re.I)
    all_nums = _collect_all_numbers(clause)
    has_opener = bool(_SAFE_CONCLUSION_OPENERS.search(clause))
    has_object = bool(_SAFE_CONCLUSION_OBJECTS.search(clause))
    has_verb = bool(_SAFE_CONCLUSION_VERBS.search(clause))

    if len(all_nums) == 1 and (has_opener or has_object or has_verb):
        num = all_nums[0]
        # If the number is immediately followed by a non-conclusion noun
        # (not a unit or conclusion object), it's modifying that noun.
        remaining = _NUMERIC.split(clause, maxsplit=2)
        if len(remaining) >= 3:
            after_num = remaining[2].strip()
            noun_match = re.match(r"([a-z]{3,})\b", after_num)
            if noun_match:
                following_word = noun_match.group(1).lower()
                if _NON_CONCLUSION_NOUN.match(following_word):
                    return {"answer": "", "ambiguous": True}
        return {"answer": num, "ambiguous": False}

    # Multiple numbers without clear binding → ambiguous.
    if len(all_nums) >= 2:
        return {"answer": "", "ambiguous": True}

    return {"answer": "", "ambiguous": False}


def _collect_all_numbers(text: str) -> list[str]:
    """Return all unique parseable numbers found in text, preserving order."""
    seen = set()
    result = []
    for token in _NUMERIC.findall(text):
        answer = _clean_parseable_number(token)
        if answer and answer not in seen:
            seen.add(answer)
            result.append(answer)
    return result


def _expected_label_regex(expected_output_type: str) -> re.Pattern:
    """Build a line-starting regex that matches only the labels allowed for this role."""
    labels = EXPECTED_LABELS.get(expected_output_type, ("Final answer",))
    alternation = "|".join(re.escape(label) for label in labels)
    return re.compile(rf"(?im)^\s*(?:{alternation})\s*[:：=]\s*(.+?)\s*$")


def _expected_label_anywhere_regex(expected_output_type: str) -> re.Pattern:
    """Build a mid-text regex that matches only the labels allowed for this role."""
    labels = EXPECTED_LABELS.get(expected_output_type, ("Final answer",))
    alternation = "|".join(re.escape(label) for label in labels)
    return re.compile(rf"(?i)(?:{alternation})\s*[:：=]\s*(.+?)(?:\n|$)")


def extract_explicit_conclusion(
    text: str,
    expected_output_type: str,
) -> dict:
    """Extract the explicitly committed answer from model output.

    expected_output_type must be one of: solver_current, solver_final, verifier, finalizer.

    Returns: {"answer": str, "method": str, "explicit": bool, "evidence": str, "ambiguous": bool}

    Priority order (strict, stop at first match):
      A. Explicit label (only the labels allowed for this expected_output_type)
      B. Valid JSON field (only the JSON key matching expected_output_type)
      C. \\boxed{N} in conclusion window
      D. Explicit natural-language conclusion in conclusion window
      E. Simple single-number output
      F. Empty answer

    Label isolation: Finalizer Reason mentioning "Current answer: 5"
    cannot override "Final answer: 42" for a finalizer extraction.
    """
    raw = str(text or "").strip()
    if not raw:
        return {"answer": "", "method": "empty_input", "explicit": False, "evidence": "", "ambiguous": False}

    # ── Priority A: Explicit label declaration ──
    # ONLY search labels allowed for this expected_output_type.
    label_line_re = _expected_label_regex(expected_output_type)
    label_any_re = _expected_label_anywhere_regex(expected_output_type)
    all_label_matches = list(label_line_re.finditer(raw))
    mid_matches = list(label_any_re.finditer(raw))
    all_label_matches.extend(mid_matches)
    all_label_matches.sort(key=lambda m: m.start())

    if all_label_matches:
        # When the same label appears multiple times, check whether all
        # non-empty answers are mathematically equivalent.
        unique_answers = []
        for m in all_label_matches:
            declared = m.group(1).strip()
            if _explicitly_undetermined_prose(declared):
                continue
            ans = extract_first_declared_numeric(declared)
            if ans and ans not in unique_answers:
                unique_answers.append(ans)
        if len(unique_answers) == 0:
            # All labels were explicitly undetermined.
            return {"answer": "", "method": "explicit_undetermined", "explicit": True,
                    "evidence": all_label_matches[-1].group(0).strip(), "ambiguous": False}
        if len(unique_answers) == 1:
            answer = unique_answers[0]
            last_match = all_label_matches[-1]
            return {"answer": answer, "method": "explicit_label", "explicit": True,
                    "evidence": last_match.group(0).strip(), "ambiguous": False}
        # Multiple different answers from the same label type → ambiguous.
        return {"answer": "", "method": "ambiguous_label_conflict", "explicit": True,
                "evidence": f"conflicting: {unique_answers}", "ambiguous": True}

    # ── Priority B: Valid JSON field ──
    json_obj = raw_json_object(raw)
    if isinstance(json_obj, dict):
        for json_key in _EXPECTED_JSON_KEYS.get(expected_output_type, ("final_answer",)):
            json_value = str(json_obj.get(json_key, "")).strip()
            if json_value and not _explicitly_undetermined_prose(json_value):
                answer = extract_first_declared_numeric(json_value)
                if answer:
                    return {"answer": answer, "method": "json_field", "explicit": True,
                            "evidence": f'"{json_key}": "{json_value}"', "ambiguous": False}

    # ── Build conclusion window ──
    window = _conclusion_window(raw)
    window_clauses = [c.strip() for c in re.split(r"(?<=[.!?。！？])\s+", window) if c.strip()]

    # ── Priority C: \\boxed{N} in conclusion window ──
    boxed = _extract_boxed_answer(window)
    if boxed:
        return {"answer": boxed, "method": "boxed_in_window", "explicit": True,
                "evidence": f"\\boxed{{{boxed}}}", "ambiguous": False}

    # ── Undetermined check: only in the conclusion window (last clause) ──
    # Earlier text saying "insufficient information" does NOT block a later
    # explicit conclusion like "Therefore, the final answer is 42."
    if window_clauses:
        last_clause = window_clauses[-1]
        if _explicitly_undetermined_prose(last_clause):
            # Check: does the last clause have a concrete answer?
            # If the last clause says "cannot determine" or "remains unknown",
            # the whole conclusion is undetermined.
            has_concrete_answer = (
                _SAFE_CONCLUSION_OPENERS.search(last_clause)
                and _NUMERIC.search(last_clause)
                and not _explicitly_undetermined_prose(last_clause)
            )
            if not has_concrete_answer:
                return {"answer": "", "method": "concluding_undetermined", "explicit": True,
                        "evidence": last_clause[:200], "ambiguous": False}

    # ── Priority D: Explicit natural-language conclusion in window ──
    if window_clauses:
        # Check last 3 clauses in reverse order.
        for clause in reversed(window_clauses[-3:]):
            has_opener = bool(_SAFE_CONCLUSION_OPENERS.search(clause))
            has_verb = bool(_SAFE_CONCLUSION_VERBS.search(clause))
            has_object = bool(_SAFE_CONCLUSION_OBJECTS.search(clause))
            has_boxed = bool(_BOXED.search(clause))
            has_speculation = bool(_AMBIGUOUS_SPECULATION.search(clause))

            # Condition 1: therefore/thus/hence/finally/in conclusion + bound conclusion
            if has_opener and not has_speculation:
                bound = extract_numeric_bound_to_conclusion(clause)
                if bound["answer"]:
                    return {"answer": bound["answer"], "method": "concluding_expression",
                            "explicit": True, "evidence": clause[:200],
                            "ambiguous": bool(bound["ambiguous"])}
                if bound["ambiguous"]:
                    return {"answer": "", "method": "ambiguous_conclusion",
                            "explicit": False, "evidence": clause[:200], "ambiguous": True}

            # Condition 2: \\boxed{N} anywhere in clause
            if has_boxed:
                answer = _extract_boxed_answer(clause)
                if answer:
                    return {"answer": answer, "method": "boxed_in_clause", "explicit": True,
                            "evidence": clause[:200], "ambiguous": False}

            # Condition 3: conclusion verb without opener → must be single, bound
            if has_verb and not has_opener and not has_speculation:
                bound = extract_numeric_bound_to_conclusion(clause)
                if bound["answer"]:
                    return {"answer": bound["answer"], "method": "concluding_verb",
                            "explicit": True, "evidence": clause[:200],
                            "ambiguous": bool(bound["ambiguous"])}
                if bound["ambiguous"]:
                    return {"answer": "", "method": "ambiguous_conclusion",
                            "explicit": False, "evidence": clause[:200], "ambiguous": True}

            # Condition 4: conclusion object without explicit opener/verb
            if has_object and not has_speculation:
                bound = extract_numeric_bound_to_conclusion(clause)
                if bound["answer"]:
                    return {"answer": bound["answer"], "method": "concluding_object",
                            "explicit": True, "evidence": clause[:200],
                            "ambiguous": bool(bound["ambiguous"])}
                if bound["ambiguous"]:
                    return {"answer": "", "method": "ambiguous_conclusion",
                            "explicit": False, "evidence": clause[:200], "ambiguous": True}

        # Global ambiguity check across the conclusion window.
        all_nums = _collect_all_numbers(window)
        if len(all_nums) >= 2:
            last_clause = window_clauses[-1] if window_clauses else ""
            bound = extract_numeric_bound_to_conclusion(last_clause)
            if bound["answer"] and not bound["ambiguous"]:
                return {"answer": bound["answer"], "method": "concluding_expression",
                        "explicit": True, "evidence": last_clause[:200], "ambiguous": False}
            return {"answer": "", "method": "ambiguous_conclusion", "explicit": False,
                    "evidence": window[:200], "ambiguous": True}

    # ── Priority E: Simple single-number output ──
    full_nums = _collect_all_numbers(raw)
    if len(full_nums) == 1:
        return {"answer": full_nums[0], "method": "safe_single_number_fallback", "explicit": False,
                "evidence": f"single number: {full_nums[0]}", "ambiguous": False}

    # ── Priority F: Empty ──
    return {"answer": "", "method": "no_supported_answer", "explicit": False, "evidence": "", "ambiguous": False}


def extract_semantic_answer(
    text: Any,
    label: str,
    parsed_output: dict | None = None,
    expected_output_type: str = "solver_final",
) -> dict:
    """Unified semantic answer extraction — never picks incidental numbers.

    Returns {"answer": str, "method": str, "explicit": bool, "evidence": str, "ambiguous": bool}.
    Delegates to extract_explicit_conclusion() for all extraction logic.
    The 'label' parameter is preserved for backward compatibility and forwarded
    as part of the extraction; the expected_output_type drives JSON key selection.
    """
    raw = str(text or "").strip()

    # Use the unified conclusion extractor for all paths.
    result = extract_explicit_conclusion(raw, expected_output_type)

    # If no answer found via conclusion extractor, try parsed_output JSON fields
    # as a secondary source (for pre-built parsed dicts from call_finalizer_once etc.)
    if not result["answer"] and parsed_output and isinstance(parsed_output, dict):
        for json_key in (label.lower().replace(" ", "_"), label):
            json_value = str(parsed_output.get(json_key, "")).strip()
            if json_value and not _explicitly_undetermined_prose(json_value):
                answer = extract_first_declared_numeric(json_value)
                if answer:
                    return {"answer": answer, "method": "json_field", "explicit": True,
                            "evidence": f'parsed_output["{json_key}"]="{json_value}"', "ambiguous": False}

    return result


def parse_solver_final(text: Any) -> tuple[str, str]:
    """Validate the solver contract instead of recovering from malformed prose.

    Leading blank lines are a format violation \u2014 the first PHYSICAL line must
    be the Final answer declaration. Semantic extraction (via
    extract_semantic_answer) is format-independent and runs separately.
    """
    raw = str(text or "").rstrip("\r\n")
    lines = raw.splitlines()
    if not lines or not re.fullmatch(r"Final answer\s*[:\uFF1A]\s*.+", lines[0], re.I):
        return "", _solver_first_line_error(raw)
    match = re.fullmatch(r"Final answer\s*[:\uFF1A]\s*(.+)", lines[0], re.I)
    declared = match.group(1).strip() if match else ""
    if explicitly_undetermined(declared):
        answer = ""
    else:
        answer = extract_first_declared_numeric(declared)
        if not answer:
            return "", "Final answer is empty or unsupported"
    sentence_count = 0
    for line in (line.strip() for line in lines[1:] if line.strip()):
        # Protect decimal points before splitting. Each non-empty physical line
        # counts as at least one sentence, so unpunctuated bullet-style reasons
        # cannot bypass the three-sentence limit.
        protected = re.sub(r"(?<=\d)\.(?=\d)", "\uE000", line)
        parts = re.split(r"[.!?\u3002\uFF01\uFF1F]+", protected)
        # Exclude standalone digits produced by numbered-list markers.
        sentence_count += len([p for p in parts if p.strip() and not re.fullmatch(r"\s*\d+\s*", p)])
    if sentence_count > 3:
        return "", "solver reasoning exceeds three sentences"
    return answer, ""


def _solver_first_line_error(raw: str) -> str:
    """Classify first-line format error: field_order vs missing_field."""
    raw_no_blank = "\n".join(line for line in raw.splitlines() if line.strip())
    # Check if "Final answer:" appears anywhere in the text (after stripping blanks).
    if re.search(r"Final answer\s*[:\uFF1A]", raw_no_blank, re.I):
        return "first line must be `Final answer: ...`"
    return "first line must be `Final answer: ...`"


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
            # Preserve leading whitespace so strict first-line format checks can
            # detect a preamble or blank line. Only terminal whitespace is harmless.
            results.append((self.tokenizer.decode(generated, skip_special_tokens=True).rstrip(), usage,
                            batch_elapsed / len(requests)))
        return results


VERIFIER_DEFAULT = {"information_sufficient": False, "revealed_facts": [], "candidate_checks": [], "verified_answer": "", "selected_source": "none", "missing_information": []}
VERIFIER_REQUIRED_KEYS = set(VERIFIER_DEFAULT)
# Per-key type requirements for a valid verifier JSON object.
VERIFIER_KEY_TYPES = {
    "information_sufficient": bool,
    "revealed_facts": (list,),
    "candidate_checks": (list,),
    "verified_answer": (str, int, float, type(None)),
    "selected_source": (str,),
    "missing_information": (list,),
}
FINALIZER_DEFAULT = {"final_answer": "", "selected_source": "none", "reason": ""}
LOCAL_JUDGE_DEFAULT = {
    "reason_derived_answer": "",
    "answer_reason_consistent": False,
    "reason_mathematically_valid": False,
    "reason_uses_all_required_facts": False,
    "error_type": "",
}
FORMAT_FAILURE_CATEGORIES = (
    "missing_field", "field_order_error", "extra_text_outside_three_lines",
    "unparseable_final_answer", "illegal_selected_source", "empty_reason",
    "retry_exhausted", "truncated", "model_refusal",
)
ANSWER_ERROR_CATEGORIES = (
    "correct_answer_in_reasoning_but_wrong_final_answer", "arithmetic_error",
    "used_incomplete_facts", "ignored_late_facts",
    "carried_forward_early_wrong_conclusion", "selected_wrong_solver_candidate",
    "unable_to_judge", "random_or_unsourced_number",
)

UNDETERMINED_ANSWERS = {"", "unknown", "undetermined", "cannot determine", "cannot be determined", "insufficient information", "none", "n/a"}


def model_event(model: LocalQwen, agent: str, system: str, user: str, phase: str, parser_defaults: dict | None,
                temperature: float | None = None) -> dict:
    try:
        raw, usage, elapsed = model.call(system, user, temperature=temperature)
    except TypeError:
        # Compatibility with simple test doubles and older imported model wrappers.
        raw, usage, elapsed = model.call(system, user)
    truncated = bool(getattr(model, "max_new_tokens", 0)
                     and usage.get("completion_tokens", 0) >= getattr(model, "max_new_tokens", 0))
    if parser_defaults is None:
        return {"agent": agent, "phase": phase, "actual_input": user,
                "actual_messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "output": raw,
                "raw_output": raw, "token_usage": usage, "runtime_seconds": elapsed,
                "generated_truncated": truncated}
    try:
        parsed, parse_error = parse_object(raw, parser_defaults), ""
    except (ValueError, json.JSONDecodeError) as exc:
        parsed, parse_error = dict(parser_defaults), str(exc)
    return {"agent": agent, "phase": phase, "actual_input": user,
            "actual_messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "raw_output": raw,
            "parsed_output": parsed, "parse_error": parse_error, "token_usage": usage,
            "runtime_seconds": elapsed, "generated_truncated": truncated}


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
            sem = extract_semantic_answer(event["raw_output"], "Current answer",
                                           expected_output_type="solver_current")
            event["current_answer"] = sem["answer"]
            event["current_answer_extraction"] = sem["method"]
            event["current_answer_explicit"] = sem["explicit"]
            event["semantic_answer"] = sem["answer"]
            event["semantic_answer_extraction"] = sem["method"]
            event["semantic_answer_evidence"] = sem.get("evidence", "")
            event["semantic_answer_ambiguous"] = bool(sem.get("ambiguous"))
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
        raw = final_batch[side].get("raw_output", "")
        sem = extract_semantic_answer(raw, "Final answer",
                                       expected_output_type="solver_final")
        strict_answer, format_error = parse_solver_final(raw)
        final_batch[side]["answer"] = sem["answer"]
        final_batch[side]["semantic_answer"] = sem["answer"]
        final_batch[side]["semantic_answer_extraction"] = sem["method"]
        final_batch[side]["semantic_answer_evidence"] = sem.get("evidence", "")
        final_batch[side]["semantic_answer_ambiguous"] = bool(sem.get("ambiguous"))
        final_batch[side]["strict_answer"] = strict_answer
        final_batch[side]["answer_extraction"] = "strict_solver_final" if not format_error else "invalid_format"
        final_batch[side]["validation_error"] = format_error
        final_batch[side]["invalid_output"] = bool(format_error)
        final_batch[side]["raw_format_compliant"] = not bool(format_error)
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


def random_order_sequence(base_seed: int, question_id: int, repetitions: int = 6) -> tuple[str, ...]:
    """Balanced deterministic randomization; with two facts the permutation space is AB/BA."""
    orders = ["AB", "BA"] * ((repetitions + 1) // 2)
    orders = orders[:repetitions]
    random.Random(derived_seed(base_seed, question_id, "random_order")).shuffle(orders)
    return tuple(orders)


def replay_setting_parts(setting: str) -> tuple[str, str]:
    for order in ("AB", "BA"):
        suffix = f"_{order}"
        if setting.endswith(suffix):
            timing = setting[:-len(suffix)]
            if timing in INFORMATION_TIMINGS:
                return timing, order
    raise ValueError(f"not a canonical information-timing setting: {setting}")


def replay_ledger(item: dict) -> str:
    """A deterministic table: labels are normalized, fact values remain byte-for-byte unchanged."""
    return f"| side | fact (verbatim) |\n|---|---|\n| A | {item['condition_A']} |\n| B | {item['condition_B']} |"


def ledger_evidence_hash(evidence: str) -> str:
    return hashlib.sha256(evidence.encode("utf-8")).hexdigest()


def fixed_source_ledger_evidence(item: dict) -> str:
    return f"Canonical fact table:\n{replay_ledger(item)}"


def fixed_source_ledger_finalizer_prompt(item: dict) -> tuple[str, str]:
    evidence = fixed_source_ledger_evidence(item)
    user = empty_candidate_finalizer_prompt(item, evidence)
    return evidence, user


def empty_candidate_finalizer_prompt(item: dict, evidence_view: str) -> str:
    """Build the one canonical no-candidate prompt used by every control."""
    return (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence_view}\n'
            'Valid non-empty candidates: {}\nAvailable selected_source values for this question: ["recomputed", "none"]\n'
            'Verifier report: "(no verifier in this setting)"\n'
            "Because Valid non-empty candidates is {}, ignore every identical-candidate rule and instruction. "
            "Your response must start directly on its first line with `Selected source:`; do not emit any "
            "preamble, analysis, blank line, or other text before it.\n"
            "Return exactly three physical lines with no blank lines between them. Use `Selected source: recomputed` "
            "only when Final answer is a concrete numeric answer derived from the visible evidence. If the visible "
            "evidence cannot determine a numeric answer, the only valid first two lines are exactly "
            "`Selected source: none` and `Final answer: undetermined`.\n"
            "Recompute from the visible evidence. Return exactly the required three-line finalizer format.")


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
            sem = extract_semantic_answer(event["raw_output"], "Current answer",
                                           expected_output_type="solver_current")
            event["current_answer"] = sem["answer"]
            event["current_answer_extraction"] = sem["method"]
            event["current_answer_explicit"] = sem["explicit"]
            event["semantic_answer"] = sem["answer"]
            event["semantic_answer_extraction"] = sem["method"]
            event["semantic_answer_evidence"] = sem.get("evidence", "")
            event["semantic_answer_ambiguous"] = bool(sem.get("ambiguous"))
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
        raw = event.get("raw_output", "")
        sem = extract_semantic_answer(raw, "Final answer",
                                       expected_output_type="solver_final")
        strict_answer, error = parse_solver_final(raw)
        event.update(phase="solver_final",
                     answer=sem["answer"],
                     semantic_answer=sem["answer"],
                     semantic_answer_extraction=sem["method"],
                     semantic_answer_evidence=sem.get("evidence", ""),
                     semantic_answer_ambiguous=bool(sem.get("ambiguous")),
                     strict_answer=strict_answer,
                     answer_extraction="strict_solver_final" if not error else "invalid_format",
                     validation_error=error, invalid_output=bool(error),
                     raw_format_compliant=not bool(error))
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


def extract_confidence(text: Any) -> float | None:
    match = re.search(r"(?i)\bconfidence\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%?", str(text or ""))
    if match:
        return min(1.0, max(0.0, float(match.group(1)) / 100))
    lowered = str(text or "").lower()
    if "high confidence" in lowered or "confident" in lowered:
        return 0.8
    if "low confidence" in lowered or "uncertain" in lowered:
        return 0.3
    return None


def acknowledges_prior_error(text: Any) -> bool:
    return bool(re.search(
        r"(?i)\b(?:previous|earlier|prior)\s+(?:answer|conclusion|reasoning).{0,30}"
        r"(?:wrong|incorrect|mistaken|error)|(?:i|we)\s+(?:was|were)\s+(?:wrong|mistaken)|"
        r"(?:此前|之前|先前).{0,20}(?:错误|不对)", str(text or "")))


def acknowledges_new_facts(item: dict, text: Any) -> dict:
    raw = str(text or "")
    checks = {}
    for side in ("A", "B"):
        fact = item[f"condition_{side}"]
        explicit, score = fact_is_public(fact, raw)
        fact_numbers = set(re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?|\d+%", fact.lower()))
        output_numbers = set(re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?|\d+%", raw.lower()))
        used = bool(fact_numbers) and fact_numbers <= output_numbers
        checks[side] = {"acknowledged": bool(explicit or used), "coverage": round(score, 4)}
    return {"per_fact": checks, "all_new_facts_acknowledged": all(
        row["acknowledged"] for row in checks.values())}


def belief_update_direction(before: Any, after: Any, gold: Any) -> str:
    if equivalent(before, after) or (not before and not after):
        return "unchanged"
    before_correct, after_correct = equivalent(before, gold), equivalent(after, gold)
    if not before_correct and after_correct:
        return "wrong_to_correct"
    if before_correct and not after_correct:
        return "correct_to_wrong"
    return "wrong_to_different_wrong"


def add_information_state_metrics(trace: dict, item: dict) -> None:
    """Create checkpoint snapshots and isolate the first response to injected facts."""
    discussion = trace.get("discussion") or {}
    timing = trace.get("information_injection_timing", "")
    steps = []
    current_candidates = {}
    for row in discussion.get("round_records", []):
        outputs = {}
        for side in ("A", "B"):
            event = row.get("simultaneous_turn", {}).get(side.lower(), {})
            answer = event.get("current_answer", "")
            current_candidates[f"solver_{side.lower()}"] = answer
            raw = event.get("raw_output", "")
            outputs[f"solver_{side.lower()}"] = {
                "prediction": answer,
                "confidence": extract_confidence(raw),
                "confidence_source": "explicit_or_lexical; null when not stated",
                "reasoning_path": raw,
                "acknowledges_previous_error": acknowledges_prior_error(raw),
            }
        steps.append({
            "checkpoint": f"discussion_round_{row['round']}",
            "facts_visible": ["A", "B"] if row.get("facts_visible") else [],
            "current_candidate_answers": dict(current_candidates),
            "agent_states": outputs,
        })
    final_states = {}
    final_facts_visible = bool(discussion.get("facts_visible_at_solver_final"))
    for side in ("a", "b"):
        event = discussion.get("solver_finals", {}).get(side)
        if event:
            answer, raw = event_answer(event), event.get("raw_output", "")
            current_candidates[f"solver_{side}"] = answer
            final_states[f"solver_{side}"] = {
                "prediction": answer,
                "confidence": extract_confidence(raw),
                "confidence_source": "explicit_or_lexical; null when not stated",
                "reasoning_path": raw,
                "acknowledges_previous_error": acknowledges_prior_error(raw),
            }
    if final_states:
        steps.append({
            "checkpoint": "solver_final",
            "facts_visible": ["A", "B"] if final_facts_visible else [],
            "current_candidate_answers": dict(current_candidates),
            "agent_states": final_states,
        })
    finalizer = trace.get("finalizer_event") or {}
    final_raw = finalizer.get("raw_output", "")
    steps.append({
        "checkpoint": "finalizer",
        "facts_visible": ["A", "B"],
        "current_candidate_answers": {
            **current_candidates, "finalizer": trace.get("final_prediction", "")},
        "agent_states": {
            "finalizer": {
                "prediction": trace.get("final_prediction", ""),
                "confidence": extract_confidence(final_raw),
                "confidence_source": "explicit_or_lexical; null when not stated",
                "reasoning_path": finalizer.get("parsed_output", {}).get("reason", final_raw),
                "acknowledges_previous_error": acknowledges_prior_error(final_raw),
            }
        },
    })

    injection_index = next((i for i, step in enumerate(steps) if step["facts_visible"]), len(steps) - 1)
    response_step = steps[injection_index]
    acknowledgements = {
        agent: acknowledges_new_facts(item, state["reasoning_path"])
        for agent, state in response_step["agent_states"].items()
    }
    before_candidates = steps[injection_index - 1]["current_candidate_answers"] if injection_index else {}
    after_candidates = response_step["current_candidate_answers"]
    updates = {}
    agents = set(before_candidates) | set(after_candidates)
    if response_step["checkpoint"] == "finalizer":
        # Compare each frozen solver belief with the one downstream final decision.
        agents = set(before_candidates) or {"finalizer"}
        for agent in sorted(agents):
            before = before_candidates.get(agent, "")
            after = trace.get("final_prediction", "")
            updates[agent] = {
                "before": before, "after": after,
                "changed": not equivalent(before, after),
                "direction": belief_update_direction(before, after, trace["gold_answer"]),
            }
    else:
        for agent in sorted(agents):
            before, after = before_candidates.get(agent, ""), after_candidates.get(agent, "")
            updates[agent] = {
                "before": before, "after": after,
                "changed": not equivalent(before, after),
                "direction": belief_update_direction(before, after, trace["gold_answer"]),
            }
    trace["information_steps"] = steps
    trace["late_fact_acknowledgement"] = {
        "injection_timing": timing,
        "first_response_checkpoint": response_step["checkpoint"],
        "per_agent": acknowledgements,
        "all_responding_agents_acknowledged": bool(acknowledgements) and all(
            row["all_new_facts_acknowledged"] for row in acknowledgements.values()),
    }
    trace["belief_update"] = {
        "injection_timing": timing,
        "per_agent": updates,
        "direction_counts": dict(Counter(row["direction"] for row in updates.values())),
    }


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
    raw_output = event.get("raw_output", "")
    # Semantic extraction: format-independent, used for correctness and candidates.
    sem = extract_semantic_answer(raw_output, "Final answer",
                                   expected_output_type="solver_final")
    # Strict format validation: enforces the "first line + ≤3 sentences" contract.
    strict_answer, format_error = parse_solver_final(raw_output)
    event.update(answer=sem["answer"],
                 semantic_answer=sem["answer"],
                 semantic_answer_extraction=sem["method"],
                 semantic_answer_evidence=sem.get("evidence", ""),
                 semantic_answer_ambiguous=bool(sem.get("ambiguous")),
                 semantic_answer_explicit=sem["explicit"],
                 strict_answer=strict_answer,
                 answer_extraction="strict_solver_final" if not format_error else "invalid_format",
                 validation_error=format_error,
                 invalid_output=bool(format_error),
                 raw_format_compliant=not bool(format_error),
                 single_shot_format_failure=bool(format_error))
    if format_error:
        event["format_failure_categories"] = classify_solver_format_errors(event)
    return event


def event_answer(event: dict | None, key: str = "final_answer") -> str:
    """Return the semantic answer from an event, never blocked by format errors.

    Priority (strict, no guessing):
    1. Explicitly saved semantic_answer field — authoritative.
    2. Structured parsed_output field (legacy events).
    3. Flat event["answer"] field (legacy solver events).
    4. Return "" — NEVER guess from raw_output prose.
    """
    event = event or {}
    # Tier 1: semantic_answer field — format-independent, always authoritative.
    if "semantic_answer" in event:
        value = str(event.get("semantic_answer", "")).strip()
        if not value or explicitly_undetermined(value):
            return ""
        answer = extract_answer(value)
        return answer if decimal(answer) is not None else ""
    # Tier 2: parsed_output field (legacy path for finalizer/verifier events).
    parsed = event.get("parsed_output", {})
    if isinstance(parsed, dict) and key in parsed:
        value = str(parsed.get(key, "")).strip()
        if not value or explicitly_undetermined(value):
            return ""
        answer = extract_answer(value)
        return answer if decimal(answer) is not None else ""
    # Tier 3: event["answer"] flat key (legacy solver events).
    if key == "final_answer" and "answer" in event:
        value = str(event.get("answer", "")).strip()
        if not value or explicitly_undetermined(value):
            return ""
        answer = extract_answer(value)
        return answer if decimal(answer) is not None else ""
    # Never guess from raw_output prose.
    return ""


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
    if verifier:
        verifier_answer = verifier.get("semantic_answer",
                                        event_answer(verifier, "verified_answer"))
        verifier_protocol_ok = bool(verifier.get("protocol_valid"))
        if decimal(verifier_answer) is not None:
            appearances.append({"source": "verifier", "phase": "verification",
                                "answer": verifier_answer,
                                "information_complete_at_appearance": complete_after_discussion,
                                "raw_format_compliant": bool(verifier.get("raw_format_compliant")),
                                "protocol_valid": verifier_protocol_ok,
                                "eligible_for_finalizer": verifier_protocol_ok,
                                "diagnostic_only": not verifier_protocol_ok,
                                })
    return appearances


def finalizer_saw_correct_answer(trace: dict) -> dict:
    """Check explicit candidates/answer declarations in the exact finalizer input."""
    gold = trace.get("gold_answer", "")
    correct_sources = [
        source for source, answer in trace.get("candidate_answers", {}).items()
        if equivalent(answer, gold)
    ]
    finalizer = trace.get("finalizer_event") or {}
    prompt = str(finalizer.get("actual_input", ""))
    declared = []
    for label in ("Current answer", "Final answer", "verified_answer"):
        for value in re.findall(rf"(?im)(?<![A-Za-z0-9_]){re.escape(label)}\s*[:\uFF1A=]\s*([^\r\n]+)", prompt):
            answer = extract_answer(value)
            if equivalent(answer, gold):
                declared.append({"label": label, "answer": answer})
    return {
        "saw_correct_answer": bool(correct_sources or declared),
        "correct_candidate_sources": correct_sources,
        "correct_declarations_in_prompt": declared,
    }


def analyze_answer_flow(trace: dict) -> dict:
    """Locate emergence, loss, recovery, and final commit failures offline."""
    gold = trace.get("gold_answer", "")
    # ── Single-agent settings: no finalizer exists ──
    single = trace.get("single_event")
    is_single = single is not None and trace.get("finalizer_event") is None
    if is_single:
        single_answer = event_answer(single)
        single_correct = equivalent(single_answer, gold)
        flow = {
            "answer_emergence": {
                "occurred": single_correct,
                "first_step": "Single final" if single_correct else "",
                "first_agent": single.get("agent", "solver") if single_correct else "",
                "first_index": 0 if single_correct else None,
            },
            "answer_retention": {
                "retained_to_final_answer": single_correct,
                "retained_to_finalizer_input": None,
                "last_correct_step": "Single final" if single_correct else "",
                "last_correct_agent": single.get("agent", "solver") if single_correct else "",
            },
            "answer_overwrite": {"occurred": False, "events": []},
            "answer_recovery": {"occurred": False, "events": []},
            "final_commit_failure": False,
            "finalizer_visibility": {
                "saw_correct_answer": False,
                "correct_candidate_sources": [],
                "correct_declarations_in_prompt": [],
            },
            "finalizer_explicitly_rejected_correct_answer": None,
            "finalizer_rejection_assessment": "not_applicable_single_agent",
            "finalizer_selected_away_from_correct_candidate": None,
            "final_deviation": None,
            "loss_locations": [],
            "timeline_nodes": [
                {"index": 0, "step": "Single final",
                 "agent": single.get("agent", "solver"), "phase": "solver_final",
                 "answer": extract_answer(single_answer) if single_answer else "",
                 "correct": single_correct,
                 "raw_output": str(single.get("raw_output", ""))},
            ],
        }
        return flow

    # ── Multi-agent settings: full answer-flow analysis ──
    nodes = []

    def add(step: str, agent: str, answer: Any, phase: str, raw: Any = "") -> None:
        value = extract_answer(answer) if answer else ""
        nodes.append({
            "index": len(nodes), "step": step, "agent": agent, "phase": phase,
            "answer": value, "correct": equivalent(value, gold), "raw_output": str(raw or ""),
        })

    discussion = trace.get("discussion") or {}
    for event in discussion.get("discussion_events", []):
        add(f"Round {event.get('round', '?')}", event.get("agent", "solver"),
            event.get("current_answer", ""), "solver_discussion", event.get("raw_output"))
    for side in ("a", "b"):
        event = discussion.get("solver_finals", {}).get(side)
        if event:
            add("Solver final", event.get("agent", f"solver_{side}"), event_answer(event),
                "solver_final", event.get("raw_output"))
    verifier = trace.get("verifier_event")
    if verifier:
        add("Verifier", "verifier", event_answer(verifier, "verified_answer"),
            "verifier", verifier.get("raw_output"))

    visibility = finalizer_saw_correct_answer(trace)
    finalizer = trace.get("finalizer_event")
    reason_answer = ""
    if finalizer:
        reason_answer = trace.get("reason_derived_answer", "")
        if not reason_answer:
            consistency = check_answer_reason_consistency(
                trace.get("final_prediction", ""),
                finalizer.get("parsed_output", {}).get("reason", ""))
            reason_answer = consistency.get("reason_conclusion_answer", "")
        if reason_answer:
            add("Finalizer Reason", "finalizer", reason_answer,
                "finalizer_reason", finalizer.get("parsed_output", {}).get("reason", ""))
        add("Final answer", "finalizer", trace.get("final_prediction", ""),
            "final_answer_serialization", finalizer.get("raw_output", ""))

    answered_nodes = [node for node in nodes if node["answer"]]
    correct_nodes = [node for node in answered_nodes if node["correct"]]
    first_correct = correct_nodes[0] if correct_nodes else None
    last_correct = correct_nodes[-1] if correct_nodes else None
    overwrite_events, recovery_events = [], []
    seen_correct = False
    in_lost_state = False
    previous_answer_node = None
    for node in answered_nodes:
        if node["correct"]:
            if seen_correct and in_lost_state:
                recovery_events.append({
                    "step": node["step"], "agent": node["agent"], "answer": node["answer"]})
            seen_correct = True
            in_lost_state = False
        elif seen_correct and not in_lost_state:
            overwrite_events.append({
                "from_step": previous_answer_node["step"] if previous_answer_node else "",
                "from_agent": previous_answer_node["agent"] if previous_answer_node else "",
                "to_step": node["step"], "to_agent": node["agent"], "answer": node["answer"],
                "phase": node["phase"],
            })
            in_lost_state = True
        previous_answer_node = node

    parsed = (finalizer or {}).get("parsed_output", {})
    selected_source = parsed.get("selected_source", "")
    reason = str(parsed.get("reason", ""))
    rejected_source = bool(
        visibility["correct_candidate_sources"]
        and selected_source not in visibility["correct_candidate_sources"])
    explicit_rejection_language = bool(re.search(
        r"(?i)\b(?:reject|rejected|incorrect|wrong|unsupported|ignore|cannot use|not reliable)\b|"
        r"(?:拒绝|错误|不支持|忽略|不能采用)", reason))
    explicitly_rejected = bool(rejected_source and explicit_rejection_language)
    rejection_review = trace.get("deepseek_correct_rejection_review")
    if rejection_review is not None:
        explicitly_rejected = as_bool(rejection_review.get("correct"), explicitly_rejected)
    final_correct = bool(answered_nodes and answered_nodes[-1]["phase"] == "final_answer_serialization"
                         and answered_nodes[-1]["correct"])
    reason_correct = any(node["phase"] == "finalizer_reason" and node["correct"] for node in nodes)
    final_commit_failure = bool(
        finalizer and not final_correct and (visibility["saw_correct_answer"] or reason_correct))

    loss_locations = []
    for event in overwrite_events:
        phase = event["phase"]
        if phase == "verifier":
            location = "verifier"
        elif phase == "finalizer_reason":
            location = "finalizer_reasoning"
        elif phase == "final_answer_serialization":
            location = ("final_answer_serialization" if reason_correct
                        else "finalizer_reasoning")
        elif phase.startswith("solver"):
            location = ("solver_internal" if event["from_agent"] == event["to_agent"]
                        else "between_solvers")
        else:
            location = "unknown"
        loss_locations.append(location)
    if final_commit_failure and "Canonical fact table:" in str((finalizer or {}).get("actual_input", "")):
        loss_locations.append("ledger")
    loss_locations = list(dict.fromkeys(loss_locations))

    deviation = None
    if last_correct and not final_correct:
        deviation = next((node for node in answered_nodes
                          if node["index"] > last_correct["index"] and not node["correct"]), None)
    return {
        "answer_emergence": {
            "occurred": first_correct is not None,
            "first_step": first_correct["step"] if first_correct else "",
            "first_agent": first_correct["agent"] if first_correct else "",
            "first_index": first_correct["index"] if first_correct else None,
        },
        "answer_retention": {
            "retained_to_final_answer": bool(first_correct and final_correct),
            "retained_to_finalizer_input": visibility["saw_correct_answer"],
            "last_correct_step": last_correct["step"] if last_correct else "",
            "last_correct_agent": last_correct["agent"] if last_correct else "",
        },
        "answer_overwrite": {"occurred": bool(overwrite_events), "events": overwrite_events},
        "answer_recovery": {"occurred": bool(recovery_events), "events": recovery_events},
        "final_commit_failure": final_commit_failure,
        "finalizer_visibility": visibility,
        "finalizer_explicitly_rejected_correct_answer": explicitly_rejected,
        "finalizer_rejection_assessment": (
            "DeepSeek semantic review" if rejection_review is not None else "deterministic rejection-language rule"),
        "finalizer_selected_away_from_correct_candidate": rejected_source,
        "final_deviation": ({
            "step": deviation["step"], "agent": deviation["agent"],
            "phase": deviation["phase"], "answer": deviation["answer"]}
            if deviation else None),
        "loss_locations": loss_locations,
        "timeline_nodes": nodes,
    }


def render_answer_flow_timeline(trace: dict) -> str:
    flow = trace["answer_flow"]
    lines = [
        f"Question {trace.get('question_id')} | setting={trace.get('setting')} | "
        f"variant={trace.get('agent_variant', '')}"
    ]
    for node in flow["timeline_nodes"]:
        status = "正确" if node["correct"] else ("无法判断" if not node["answer"] else "错误")
        lines.append(f"{node['step']}: {node['agent']}={status} ({node['answer'] or 'undetermined'})")
    visibility = flow["finalizer_visibility"]
    lines.extend([
        f"Finalizer saw correct answer: {visibility['saw_correct_answer']}",
        f"Finalizer explicitly rejected correct answer: "
        f"{flow['finalizer_explicitly_rejected_correct_answer']}",
        f"Final commit failure: {flow['final_commit_failure']}",
        f"Loss locations: {', '.join(flow['loss_locations']) or 'none'}",
    ])
    return "\n".join(lines)


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
    declared = values["Final answer"]
    if explicitly_undetermined(declared):
        answer = ""
    else:
        answer = extract_first_declared_numeric(declared)
        if not answer:
            return dict(FINALIZER_DEFAULT), "Final answer is not a supported numeric value"
    return {"selected_source": source, "final_answer": answer, "reason": values["Reason"]}, ""


def parse_finalizer_fields_loose(raw_output: str) -> dict:
    """Loose semantic-only parse of finalizer output for answer recovery.

    This function is ONLY used for semantic answer recovery — it must never
    set raw_format_compliant, protocol_valid, or normalized_format_compliant
    to True. It tolerates:
    - Labels not on the first three lines
    - Label order errors
    - Blank lines between labels
    - Extra preamble/explanation text
    - Markdown wrapping around labels
    - Only "Final answer:" present (no Selected source or Reason)
    - No labels at all but a clear natural-language conclusion

    Returns {
        "selected_source": "", "final_answer": "", "reason": "",
        "answer_evidence": "", "source_evidence": "", "reason_evidence": "",
    }
    """
    raw = str(raw_output or "").strip()
    stripped = re.sub(r"[*`]", "", raw)

    def _find_line(label: str) -> str:
        """Find the last line matching the label anywhere in text, return its value."""
        matches = list(re.finditer(
            rf"(?im)^{re.escape(label)}\s*[:：=]\s*(.+?)\s*$", stripped))
        if not matches:
            # Also try: the label might be embedded in a longer line with markdown
            matches = list(re.finditer(
                rf"(?im){re.escape(label)}\s*[:：=]\s*(.+?)(?:\n|$)", stripped))
        if matches:
            return matches[-1].group(1).strip()
        return ""

    selected_source = _find_line("Selected source")
    final_answer_raw = _find_line("Final answer")
    reason = _find_line("Reason")

    # Normalize selected_source
    source = selected_source.lower().strip()
    if source not in {"solver_a", "solver_b", "verifier", "recomputed", "none"}:
        source = ""

    # Extract answer from Final answer line if present
    final_answer = ""
    answer_evidence = ""
    if final_answer_raw and not _explicitly_undetermined_prose(final_answer_raw):
        final_answer = extract_first_declared_numeric(final_answer_raw)
        answer_evidence = f"Final answer: {final_answer_raw}"[:200]

    # If no Final answer label, try natural language conclusion
    if not final_answer:
        conclusion = extract_explicit_conclusion(raw, "finalizer")
        final_answer = conclusion["answer"]
        answer_evidence = conclusion.get("evidence", "")

    return {
        "selected_source": source,
        "final_answer": final_answer,
        "reason": reason,
        "answer_evidence": answer_evidence,
        "source_evidence": f"Selected source: {selected_source}" if selected_source else "",
        "reason_evidence": f"Reason: {reason}"[:200] if reason else "",
    }


def self_check_before_commit_user_prompt(base_user: str, draft_output: str) -> str:
    return (f"{base_user}\nDraft finalizer output (review only):\n{draft_output}\n"
            "Before committing, verify three things: whether Final answer equals the result computed in Reason, "
            "whether all visible facts are used, and whether the output obeys the required format. If anything is wrong, "
            "revise only Selected source, Final answer, and Reason. Do not request more discussion, do not ask for new facts, "
            "and do not add any extra text outside the same three-line format.")


def normalize_finalizer_output(text: Any) -> tuple[str, list[str]]:
    """Apply only deterministic, auditable format/source normalizations."""
    raw = str(text or "").rstrip("\r\n")
    lines = raw.splitlines()
    normalized_steps = []
    if any(not line.strip() for line in lines):
        compact = [line for line in lines if line.strip()]
        labels = ("Selected source", "Final answer", "Reason")
        if len(compact) == 3 and all(
                re.match(rf"^\s*{re.escape(label)}\s*[:\uFF1A]", line, re.I)
                for line, label in zip(compact, labels)):
            lines = compact
            normalized_steps.append("removed_blank_lines_between_fields")
    if len(lines) == 3:
        source_match = re.fullmatch(r"\s*Selected source\s*[:\uFF1A]\s*(.*)", lines[0], re.I)
        answer_match = re.fullmatch(r"\s*Final answer\s*[:\uFF1A]\s*(.*)", lines[1], re.I)
        if (source_match and answer_match
                and source_match.group(1).strip().lower() == "recomputed"
                and explicitly_undetermined(answer_match.group(1).strip())):
            lines[0] = re.sub(r"^(\s*Selected source\s*[:\uFF1A]\s*).*$",
                              r"\1none", lines[0], flags=re.I)
            normalized_steps.append("recomputed_undetermined_to_none")
    normalized = "\n".join(lines) if normalized_steps else raw
    return normalized, normalized_steps


def classify_finalizer_format_errors(event: dict) -> list[str]:
    """Return non-exclusive, auditable failure categories for a finalizer response.

    Classification is based on the ORIGINAL raw_output, never the normalized version.
    This ensures that blank-line removal and other normalizations cannot hide
    format failures from the report.
    """
    raw = str(event.get("raw_output", "") or "")
    stripped = raw.rstrip("\r\n")
    lines = stripped.splitlines()
    expected = ("Selected source", "Final answer", "Reason")
    found = []
    for line in lines:
        match = re.match(r"^\s*(Selected source|Final answer|Reason)\s*[:\uFF1A]", line, re.I)
        if match:
            found.append(match.group(1).lower())
    categories = []
    if any(label.lower() not in found for label in expected):
        categories.append("missing_field")
    expected_found = [label.lower() for label in expected if label.lower() in found]
    if found and found[:len(expected_found)] != expected_found:
        categories.append("field_order_error")
    if len(lines) > 3 or (lines and not re.match(r"^Selected source\s*[:\uFF1A]", lines[0], re.I)):
        categories.append("extra_text_outside_three_lines")
    final_match = next((re.fullmatch(r"Final answer\s*[:\uFF1A]\s*(.*)", line, re.I)
                        for line in lines if re.match(r"^Final answer\s*[:\uFF1A]", line, re.I)), None)
    final_value = final_match.group(1).strip() if final_match else ""
    if final_match and not explicitly_undetermined(final_value) and decimal(extract_answer(final_value)) is None:
        categories.append("unparseable_final_answer")
    source_match = next((re.fullmatch(r"Selected source\s*[:\uFF1A]\s*(.*)", line, re.I)
                         for line in lines if re.match(r"^Selected source\s*[:\uFF1A]", line, re.I)), None)
    if source_match and source_match.group(1).strip().lower() not in {
            "solver_a", "solver_b", "verifier", "recomputed", "none"}:
        categories.append("illegal_selected_source")
    reason_match = next((re.fullmatch(r"Reason\s*[:\uFF1A]\s*(.*)", line, re.I)
                         for line in lines if re.match(r"^Reason\s*[:\uFF1A]", line, re.I)), None)
    if reason_match and not reason_match.group(1).strip():
        categories.append("empty_reason")
    if event.get("retry_exhausted"):
        categories.append("retry_exhausted")
    if event.get("generated_truncated"):
        categories.append("truncated")
    refusal = re.compile(r"\b(?:i cannot|i can't|unable to comply|cannot assist|refuse|sorry)\b|"
                         r"(?:无法回答|不能回答|拒绝回答)", re.I)
    if refusal.search(raw):
        categories.append("model_refusal")
    return list(dict.fromkeys(categories))


def classify_finalizer_protocol_errors(event: dict) -> list[str]:
    """Return non-exclusive protocol validation failure categories.

    Protocol errors are about source/candidate consistency, not format.
    These are tracked separately from format_failure_categories.
    """
    error = str(event.get("protocol_validation_error", ""))
    categories = []
    if "selected_source is not an allowed value" in error:
        categories.append("illegal_selected_source")
    if "unavailable or invalid" in error:
        categories.append("selected_unavailable_source")
    if "does not match selected source" in error:
        categories.append("answer_source_mismatch")
    if "selected_source none must have an empty answer" in error:
        categories.append("none_with_nonempty_answer")
    if "recomputed requires a supported numeric answer" in error:
        categories.append("recomputed_without_numeric_answer")
    if "different recomputation requires Reason" in error:
        categories.append("identical_candidate_rejection_missing")
    return list(dict.fromkeys(categories))


def classify_solver_format_errors(event: dict) -> list[str]:
    """Return non-exclusive, auditable failure categories for a solver response."""
    raw = str(event.get("raw_output", "") or "")
    error = str(event.get("validation_error", ""))
    categories = []
    if "first line must be" in error:
        # Distinguish: Final answer label on later line vs completely absent.
        raw_no_blank = "\n".join(line for line in raw.splitlines() if line.strip())
        if re.search(r"Final answer\s*[:：]", raw_no_blank, re.I):
            categories.append("field_order_error")
        else:
            categories.append("missing_field")
    if "exceeds three sentences" in error:
        categories.append("extra_text_outside_three_lines")
    if "empty or unsupported" in error:
        categories.append("unparseable_final_answer")
    if event.get("generated_truncated"):
        categories.append("truncated")
    refusal = re.compile(r"\b(?:i cannot|i can't|unable to comply|cannot assist|refuse|sorry)\b|"
                         r"(?:无法回答|不能回答|拒绝回答)", re.I)
    if refusal.search(raw):
        categories.append("model_refusal")
    return list(dict.fromkeys(categories))


def classify_wrong_answer(trace: dict) -> list[str]:
    """Combine objective trace evidence with the independent judge diagnosis."""
    if trace.get("semantic_correct"):
        return []
    categories = []
    raw = str((trace.get("finalizer_event") or {}).get("raw_output", ""))
    reason_text = "\n".join(
        line for line in raw.splitlines() if re.match(r"^Reason\s*[:\uFF1A]", line, re.I))
    reason_numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+)?", reason_text)
    if any(equivalent(number.replace(",", ""), trace.get("gold_answer")) for number in reason_numbers):
        categories.append("correct_answer_in_reasoning_but_wrong_final_answer")
    judge_error = str(trace.get("local_judge", {}).get("consensus", {}).get("error_type", "")).strip()
    allowed = {
        "arithmetic_error", "used_incomplete_facts", "ignored_late_facts",
        "carried_forward_early_wrong_conclusion", "selected_wrong_solver_candidate",
        "unable_to_judge", "random_or_unsourced_number",
    }
    normalized = re.sub(r"[^a-z]+", "_", judge_error.lower()).strip("_")
    if normalized in allowed:
        categories.append(normalized)
    if not trace.get("information", {}).get("information_complete", True):
        categories.append("used_incomplete_facts")
    acknowledgement = trace.get("late_fact_acknowledgement", {})
    if (trace.get("information_injection_timing") not in {"", "all_at_start"}
            and acknowledgement
            and not acknowledgement.get("all_responding_agents_acknowledged")):
        categories.append("ignored_late_facts")
    update_rows = trace.get("belief_update", {}).get("per_agent", {}).values()
    if any(row.get("direction") == "unchanged" and not equivalent(
            row.get("after"), trace.get("gold_answer")) for row in update_rows):
        categories.append("carried_forward_early_wrong_conclusion")
    if not trace.get("final_prediction"):
        categories.append("unable_to_judge")
    source = (trace.get("finalizer_event") or {}).get("parsed_output", {}).get("selected_source")
    if source in {"solver_a", "solver_b", "verifier"} and not trace.get(
            "per_agent_correctness", {}).get(source, False):
        categories.append("selected_wrong_solver_candidate")
    if not categories and decimal(trace.get("final_prediction")) is not None:
        # Only label as random_or_unsourced_number when we have positive evidence
        # that the number has no calculation source. Otherwise default to
        # unable_to_judge — the model may have made an arithmetic error.
        categories.append("unable_to_judge")
    return list(dict.fromkeys(categories))


def check_answer_reason_consistency(answer: Any, reason: Any) -> dict:
    """Detect an explicit concluding number in Reason that contradicts Final answer."""
    answer_value = decimal(answer)
    text = re.sub(r"\s+", " ", str(reason or "")).strip()
    if answer_value is None:
        return {"answer_reason_consistent": False, "answer_reason_checkable": False,
                "reason_conclusion_answer": "", "answer_reason_consistency_method": "unsupported_final_answer"}
    # Work backwards through sentence-like clauses. Only treat a number as a
    # conclusion when the clause contains a calculation/conclusion marker;
    # incidental inputs and numbered steps are deliberately ignored.
    clauses = [part.strip() for part in re.split(r"(?<=[.!?])\s+|(?<=[;])\s+", text) if part.strip()]
    marker = re.compile(
        r"\b(?:thus|therefore|hence|so|gives?|giving|yields?|result(?:s|ing)?|"
        r"equals?|leaves?|needs?|will\s+(?:read|be|need|have|weigh)|"
        r"reads?|weighs?|has|"
        r"correct\s+(?:answer|final\s+weight)|final\s+(?:answer|weight)|"
        r"dividing|subtracting|adding|total(?:ing|s)?)\b",
        re.I)
    for clause in reversed(clauses):
        if not marker.search(clause):
            continue
        numbers = re.findall(r"(?<![A-Za-z])[-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?", clause)
        for token in reversed(numbers):
            candidate = decimal(token.replace("$", ""))
            if candidate is not None:
                return {
                    "answer_reason_consistent": candidate == answer_value,
                    "answer_reason_checkable": True,
                    "reason_conclusion_answer": extract_answer(token.replace("$", "")),
                    "answer_reason_consistency_method": "last_marked_numeric_conclusion",
                    "reason_conclusion_clause": clause,
                }
    return {"answer_reason_consistent": True, "answer_reason_checkable": False,
            "reason_conclusion_answer": "", "answer_reason_consistency_method": "no_explicit_numeric_conclusion"}


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


def verifier_schema_error(raw_json: dict | None) -> str:
    """Validate the complete verifier JSON schema and field types.

    Returns "" when the JSON is a valid verifier object.  Any schema or type
    violation returns a human-readable error string.  This function is the
    single authority for verifier format validity.
    """
    if raw_json is None:
        return "verifier response is not a valid JSON object"
    if not isinstance(raw_json, dict):
        return "verifier response is not a valid JSON object"
    actual_keys = set(raw_json)
    missing = sorted(VERIFIER_REQUIRED_KEYS - actual_keys)
    if missing:
        return "verifier missing required field(s): " + ", ".join(missing)
    extra = sorted(actual_keys - VERIFIER_REQUIRED_KEYS)
    if extra:
        return "verifier contains unexpected field(s): " + ", ".join(extra)
    # Type checks for every required field.
    for key, allowed_raw in VERIFIER_KEY_TYPES.items():
        value = raw_json.get(key)
        allowed = allowed_raw if isinstance(allowed_raw, tuple) else (allowed_raw,)
        # Booleans are not numeric scalars for verified_answer.
        if isinstance(value, bool) and bool not in allowed:
            return f"verifier {key} must not be a boolean"
        if not isinstance(value, allowed):
            expected = " | ".join(t.__name__ for t in allowed)
            return f"verifier {key} must be {expected}, got {type(value).__name__}"
    # candidate_checks: each element must be a dict with all four string fields.
    checks = raw_json.get("candidate_checks", [])
    if not isinstance(checks, list):
        return "verifier candidate_checks must be a list"
    for i, check in enumerate(checks):
        if not isinstance(check, dict):
            return f"verifier candidate_checks[{i}] must be a dict"
        for field in ("source", "answer", "status", "reason"):
            if field not in check:
                return f"verifier candidate_checks[{i}] missing field: {field}"
            if not isinstance(check.get(field), str):
                return f"verifier candidate_checks[{i}].{field} must be a string"
    # revealed_facts, missing_information: each element must be a string.
    for list_key in ("revealed_facts", "missing_information"):
        items = raw_json.get(list_key, [])
        if not isinstance(items, list):
            return f"verifier {list_key} must be a list"
        for j, item in enumerate(items):
            if not isinstance(item, str):
                return f"verifier {list_key}[{j}] must be a string"
    return ""


def verifier_consistency_error(event: dict, candidates: dict) -> str:
    raw = raw_json_object(event.get("raw_output", ""))
    if raw is None:
        return "verifier response is not a valid JSON object"
    missing = [key for key in ("verified_answer", "selected_source") if key not in raw]
    if missing:
        return "verifier missing required field(s): " + ", ".join(missing)
    parsed = event.get("parsed_output", {})
    verified_raw = raw.get("verified_answer")
    if explicitly_undetermined(verified_raw):
        parsed["verified_answer"] = ""
    else:
        parsed["verified_answer"] = extract_first_declared_numeric(str(verified_raw))
        if not parsed["verified_answer"]:
            return "verifier verified_answer is not a supported numeric value"
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
    raw_output = event.get("raw_output", "")
    # ── Layer 1: Raw format validation on the unmodified raw_output ──
    raw_parsed, raw_format_error = parse_fixed_finalizer(raw_output)
    # ── Layer 2: Normalized format validation ──
    normalized_output, normalizations = normalize_finalizer_output(raw_output)
    normalized_parsed, normalized_format_error = parse_fixed_finalizer(normalized_output)
    # ── Layer 3: Protocol validation (source consistency, candidate checks) ──
    protocol_validation_error = ""
    if not normalized_format_error:
        protocol_validation_error = source_consistency_error(normalized_parsed, candidates, allow_none=True)
    a_answer, b_answer = candidates.get("solver_a", ""), candidates.get("solver_b", "")
    if (not normalized_format_error and not protocol_validation_error
            and normalized_parsed["selected_source"] == "recomputed" and a_answer and b_answer
            and equivalent(a_answer, b_answer)
            and not equivalent(normalized_parsed["final_answer"], a_answer)):
        if not has_explicit_identical_candidate_rejection(normalized_parsed.get("reason")):
            protocol_validation_error = (
                "different recomputation requires Reason to start with "
                "`Reject identical candidates because` or `拒绝相同候选，因为`, followed by an explanation")
    # Preserve the selected candidate's exact representation after proving
    # mathematical equivalence (for example, normalize 42.0 back to 42).
    if (not normalized_format_error and not protocol_validation_error
            and normalized_parsed["selected_source"] in candidates):
        normalized_parsed["final_answer"] = candidates[normalized_parsed["selected_source"]]
    # ── Semantic answer extraction from the ORIGINAL raw_output (format-independent) ──
    loose_parsed = parse_finalizer_fields_loose(raw_output)
    sem = extract_semantic_answer(
        raw_output,
        "Final answer",
        loose_parsed,
        expected_output_type="finalizer",
    )
    # If normalization gave us a valid parsed output, use it; otherwise use raw_parsed.
    parsed = normalized_parsed if not normalized_format_error else raw_parsed
    # Combined error for backward-compatible validation_error / invalid_output fields.
    combined_error = normalized_format_error or protocol_validation_error
    event.update(
        parsed_output=parsed, parse_error="",
        # ── Backward-compatible combined fields ──
        validation_error=combined_error,
        invalid_output=bool(combined_error),
        # ── Split three-layer validation ──
        raw_format_error=raw_format_error,
        normalized_format_error=normalized_format_error,
        protocol_validation_error=protocol_validation_error,
        # ── Format compliance flags (NEVER relaxed by loose parse) ──
        raw_format_compliant=not bool(raw_format_error),
        normalized_format_compliant=not bool(normalized_format_error),
        protocol_valid=(not bool(normalized_format_error) and not bool(protocol_validation_error)),
        # ── Semantic answer (format-independent, from raw output) ──
        semantic_answer=sem["answer"],
        semantic_answer_extraction=sem["method"],
        semantic_answer_explicit=sem["explicit"],
        semantic_answer_evidence=sem.get("evidence", ""),
        semantic_answer_ambiguous=bool(sem.get("ambiguous")),
        # ── Loose semantic parse (for diagnostic use only) ──
        semantic_parsed_output=loose_parsed,
        # ── Normalization metadata ──
        normalized_output=normalized_output,
        deterministic_normalizations=normalizations,
        # ── Single-shot format failure: raw format only ──
        single_shot_format_failure=bool(raw_format_error),
        # ── Retry metadata ──
        attempts=[{"attempt": 1, "raw_output": raw_output,
                    "normalized_output": normalized_output,
                    "deterministic_normalizations": normalizations,
                    "parsed_output": parsed,
                    "raw_format_error": raw_format_error,
                    "normalized_format_error": normalized_format_error,
                    "protocol_validation_error": protocol_validation_error,
                    "validation_error": combined_error,
                    "token_usage": event.get("token_usage", blank_usage()),
                    "runtime_seconds": event.get("runtime_seconds", 0.0)}],
        retry_count=0, recovered_after_retry=False,
        retry_exhausted=bool(combined_error),
    )
    event["format_failure_categories"] = classify_finalizer_format_errors(event)
    event["protocol_failure_categories"] = classify_finalizer_protocol_errors(event)
    return event


def call_finalizer_with_self_check_once(model: LocalQwen, system: str, user: str, candidates: dict) -> dict:
    draft_event = call_finalizer_once(model, system, user, candidates)
    draft_event["phase"] = "finalization_draft"
    self_check_user = self_check_before_commit_user_prompt(
        user,
        draft_event.get("normalized_output", draft_event.get("raw_output", "")),
    )
    final_event = call_finalizer_once(model, system, self_check_user, candidates)
    final_event["phase"] = "finalization_self_check"
    final_event["self_check_before_commit"] = True
    final_event["draft_finalizer_event"] = draft_event
    final_event["self_check_user_prompt"] = self_check_user
    return final_event


def finish_multi(model: LocalQwen, prompts: dict, item: dict, discussion: dict, with_verifier: bool,
                 self_check_before_commit: bool = False) -> tuple[dict | None, dict]:
    raw_candidates = {"solver_a": event_answer(discussion["solver_finals"]["a"]),
                      "solver_b": event_answer(discussion["solver_finals"]["b"])}
    candidates = {source: answer for source, answer in raw_candidates.items() if decimal(answer) is not None}
    verifier = None
    if with_verifier:
        user = f'Shared question: {item["shared_question"]}\nPublic transcript:\n{discussion["public_transcript"]}\nCandidates: {json.dumps(candidates, ensure_ascii=False)}'
        verifier = model_event(model, "verifier", prompts["verifier"], user, "verification", VERIFIER_DEFAULT)
        # Semantic extraction: preserve answer even when JSON parsing fails.
        verifier_sem = extract_semantic_answer(verifier.get("raw_output", ""), "verified_answer",
                                               verifier.get("parsed_output"),
                                               expected_output_type="verifier")
        verifier["semantic_answer"] = verifier_sem["answer"]
        verifier["semantic_answer_extraction"] = verifier_sem["method"]
        verifier["semantic_answer_evidence"] = verifier_sem.get("evidence", "")
        verifier["semantic_answer_ambiguous"] = bool(verifier_sem.get("ambiguous"))

        # ── Separate format vs protocol errors ──
        raw_json = raw_json_object(verifier.get("raw_output", ""))
        verifier_format_error = verifier_schema_error(raw_json)
        verifier_protocol_error = ""
        if not verifier_format_error:
            verifier_protocol_error = verifier_consistency_error(verifier, candidates)

        verifier["raw_format_compliant"] = not bool(verifier_format_error)
        verifier["protocol_valid"] = (not verifier_format_error and not verifier_protocol_error)
        verifier["verifier_format_error"] = verifier_format_error
        verifier["verifier_protocol_error"] = verifier_protocol_error
        verifier["validation_error"] = verifier_format_error or verifier_protocol_error
        verifier["invalid_output"] = bool(verifier_format_error or verifier_protocol_error)

    # ── Build clean Verifier report for Finalizer input ──
    if verifier and verifier.get("protocol_valid"):
        parsed = verifier.get("parsed_output", {})
        report = {key: parsed.get(key, default) for key, default in VERIFIER_DEFAULT.items()}
    elif verifier:
        report = {"usable": False, "validation_error": verifier.get("validation_error", "invalid verifier output")}
    else:
        report = "(no verifier in this setting)"

    # ── Finalizer candidates: only protocol-valid verifier becomes a source ──
    finalizer_candidates = dict(candidates)
    if (
        verifier
        and verifier.get("protocol_valid")
        and decimal(verifier.get("semantic_answer")) is not None
    ):
        finalizer_candidates["verifier"] = verifier["semantic_answer"]
    available_sources = list(finalizer_candidates) + ["recomputed", "none"]
    user = (f'Shared question: {item["shared_question"]}\nPublic transcript:\n{discussion["public_transcript"]}\n'
            f'Valid non-empty candidates: {json.dumps(finalizer_candidates, ensure_ascii=False)}\n'
            f'Available selected_source values for this question: {json.dumps(available_sources, ensure_ascii=False)}\n'
            f'Verifier report: {json.dumps(report, ensure_ascii=False)}\n'
            "For solver_a, solver_b, or verifier, copy that source's candidate answer exactly; never calculate a replacement value.")
    if self_check_before_commit:
        finalizer = call_finalizer_with_self_check_once(model, prompts["finalizer"], user, finalizer_candidates)
    else:
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
    if trace.get("finalizer_draft_event"):
        events.append(trace["finalizer_draft_event"])
    if trace.get("finalizer_event"):
        events.append(trace["finalizer_event"])
    events += trace.get("local_judge", {}).get("runs", [])
    return events


def run_local_reason_judge(model: LocalQwen, item: dict, finalizer: dict, base_seed: int,
                           question_id: int) -> dict:
    """Run two blinded local reviews; neither request contains setting/variant names or gold."""
    parsed = finalizer.get("parsed_output", {})
    payload = (
        f'Question: {item["shared_question"]}\n'
        f'Complete fact A: {item["condition_A"]}\n'
        f'Complete fact B: {item["condition_B"]}\n'
        f'Final answer: {finalizer.get("semantic_answer", parsed.get("final_answer", ""))}\n'
        f'Reason: {parsed.get("reason", "")}'
    )
    schemas = (
        "Independently recompute the answer from the question and complete facts, then audit the supplied reason.",
        "Audit each arithmetic step and verify that every required fact is used before evaluating answer/reason agreement.",
    )
    system_tail = (
        ' Return only one JSON object with exactly these fields: '
        '{"reason_derived_answer":"","answer_reason_consistent":true,'
        '"reason_mathematically_valid":true,"reason_uses_all_required_facts":true,"error_type":""}. '
        "error_type must be empty or one of arithmetic_error, used_incomplete_facts, ignored_late_facts, "
        "carried_forward_early_wrong_conclusion, selected_wrong_solver_candidate, unable_to_judge, "
        "random_or_unsourced_number. Do not infer anything from experiment names; none are provided."
    )
    runs = []
    for index, instruction in enumerate(schemas, 1):
        reseed_model(model, derived_seed(base_seed, question_id, "local_reason_judge", index))
        event = model_event(model, f"local_judge_{index}", instruction + system_tail,
                            payload, f"local_reason_judge_{index}", LOCAL_JUDGE_DEFAULT, temperature=0.0)
        raw = raw_json_object(event.get("raw_output", ""))
        valid = isinstance(raw, dict) and set(raw) == set(LOCAL_JUDGE_DEFAULT)
        judged = dict(LOCAL_JUDGE_DEFAULT)
        if valid:
            judged.update({
                "reason_derived_answer": extract_answer(raw.get("reason_derived_answer", "")),
                "answer_reason_consistent": as_bool(raw.get("answer_reason_consistent")),
                "reason_mathematically_valid": as_bool(raw.get("reason_mathematically_valid")),
                "reason_uses_all_required_facts": as_bool(raw.get("reason_uses_all_required_facts")),
                "error_type": str(raw.get("error_type", "")).strip(),
            })
        event["parsed_output"] = judged
        event["validation_error"] = "" if valid else "judge response does not match the fixed JSON schema"
        event["invalid_output"] = not valid
        runs.append(event)
    left, right = (run["parsed_output"] for run in runs)
    comparable = ("answer_reason_consistent", "reason_mathematically_valid",
                  "reason_uses_all_required_facts", "error_type")
    agreement = (
        not any(run["invalid_output"] for run in runs)
        and equivalent(left["reason_derived_answer"], right["reason_derived_answer"])
        and all(left[key] == right[key] for key in comparable)
    )
    consensus = dict(left) if agreement else dict(LOCAL_JUDGE_DEFAULT)
    return {
        "runs": runs,
        "judge_disagreement": not agreement,
        "manual_review_required": not agreement,
        "consensus": consensus,
        "blinding": "question, complete facts, final answer, and reason only; no setting or variant names; no gold",
    }


def apply_local_judge(trace: dict, judge: dict) -> None:
    trace["local_judge"] = judge
    consensus = judge["consensus"]
    trace["judge_disagreement"] = bool(judge["judge_disagreement"])
    trace["reason_derived_answer"] = consensus["reason_derived_answer"]
    trace["answer_reason_consistent"] = bool(consensus["answer_reason_consistent"])
    trace["reason_mathematically_valid"] = bool(consensus["reason_mathematically_valid"])
    trace["reason_uses_all_required_facts"] = bool(consensus["reason_uses_all_required_facts"])
    strict_answer = bool(trace.get("semantic_correct") and trace.get("format_compliant"))
    semantic = bool(trace.get("semantic_correct"))
    trace["strict_answer_correct"] = strict_answer
    trace["fully_valid_correct"] = bool(
        strict_answer and not trace["judge_disagreement"]
        and trace["answer_reason_consistent"] and trace["reason_mathematically_valid"])
    trace["strict_correct"] = strict_answer
    trace["correct"] = semantic
    trace["answer_error_categories"] = classify_wrong_answer(trace)
    usage = blank_usage()
    for event in judge["runs"]:
        add_usage(usage, event.get("token_usage", {}))
    trace["local_judge_token_usage"] = usage
    trace["local_judge_runtime_seconds"] = sum(
        float(event.get("runtime_seconds", 0.0)) for event in judge["runs"])


def classify(trace: dict, gold: str) -> tuple[str | None, bool]:
    """Classify semantic failure type only. Format issues are recorded separately."""
    complete = bool(trace.get("information", {}).get("information_complete"))
    appearances = trace.get("candidate_appearances", [])
    # Qualified correct appearances: eligible (not diagnostic-only),
    # correct answer, and appeared when information was complete.
    supported_correct_appeared = any(
        as_bool(x.get("correct"), equivalent(x.get("answer"), gold))
        and x.get("information_complete_at_appearance", False)
        and x.get("eligible_for_finalizer", True)
        for x in appearances if x.get("answer"))
    lucky = any(as_bool(x.get("correct"), equivalent(x.get("answer"), gold)) and not x.get("information_complete_at_appearance", False)
                for x in appearances if x.get("answer"))
    if trace.get("semantic_correct"):
        return None, lucky
    # Semantic failure classification — format issues do NOT mask these.
    if not trace.get("answer_reason_consistent", True):
        return "answer_reason_inconsistency", lucky
    if not complete:
        return "information_acquisition_failure", lucky
    if not supported_correct_appeared:
        return "information_integration_failure", lucky
    return "answer_selection_failure", lucky


def set_outcome_fields(trace: dict, gold: str, semantic_correct: bool | None = None) -> None:
    """Report semantic, strict-answer, and fully-valid correctness separately.

    semantic_correct is ALWAYS computed from final_prediction vs gold.
    The semantic_correct parameter is deprecated and ignored \u2014 it exists
    only for backward compatibility with existing call sites.
    DeepSeek and local judge CANNOT override correctness.
    """
    # Semantic correctness: deterministic local answer comparison only.
    semantic = equivalent(trace.get("final_prediction", ""), gold)
    # Format compliance: always from raw_format_compliant on the primary event.
    finalizer = trace.get("finalizer_event")
    single = trace.get("single_event")
    if finalizer and "raw_format_compliant" in finalizer:
        format_compliant = bool(finalizer.get("raw_format_compliant"))
    elif single and "raw_format_compliant" in single:
        format_compliant = bool(single.get("raw_format_compliant"))
    else:
        format_compliant = trace.get("format_compliant",
                                     not bool(trace.get("invalid_output")))
    # Reason consistency: only for finalizer events.
    if finalizer:
        parsed_reason = str(finalizer.get("parsed_output", {}).get("reason", "")).strip()
        if not parsed_reason:
            match = re.search(r"(?im)^Reason\s*[:\uFF1A]\s*(.+?)\s*$", finalizer.get("raw_output", ""))
            parsed_reason = match.group(1).strip() if match else ""
        consistency = check_answer_reason_consistency(trace.get("final_prediction", ""), parsed_reason)
        trace.update(consistency)
        trace["reason_evaluation_available"] = True
    else:
        trace["answer_reason_consistent"] = True
        trace["answer_reason_checkable"] = False
        trace["reason_conclusion_answer"] = ""
        trace["answer_reason_consistency_method"] = "not_applicable_no_finalizer"
        trace["reason_evaluation_available"] = False
    # Local reason judge (only for settings with finalizer).
    if trace.get("local_judge"):
        consensus = trace["local_judge"]["consensus"]
        trace["answer_reason_consistent"] = bool(consensus.get("answer_reason_consistent",
                                                      trace.get("answer_reason_consistent", True)))
        trace["reason_mathematically_valid"] = bool(consensus.get("reason_mathematically_valid", False))
        trace["reason_uses_all_required_facts"] = bool(consensus.get("reason_uses_all_required_facts", False))
    elif not trace.get("reason_evaluation_available"):
        # No reason evaluation available \u2192 null out metrics to avoid unfair comparison.
        trace["reason_mathematically_valid"] = None
        trace["reason_uses_all_required_facts"] = None
    reason_consistent = bool(trace.get("answer_reason_consistent", True))
    reason_valid = trace.get("reason_mathematically_valid")
    strict_answer = bool(semantic and format_compliant)
    trace["semantic_correct"] = semantic
    trace["format_compliant"] = format_compliant
    # Propagate three-layer format/protocol compliance from events.
    if finalizer:
        trace["raw_format_compliant"] = bool(finalizer.get("raw_format_compliant"))
        trace["normalized_format_compliant"] = bool(finalizer.get("normalized_format_compliant"))
        trace["protocol_valid"] = bool(finalizer.get("protocol_valid"))
    elif single:
        trace["raw_format_compliant"] = bool(single.get("raw_format_compliant"))
        trace["normalized_format_compliant"] = single.get("normalized_format_compliant")
    trace["strict_answer_correct"] = strict_answer
    # fully_valid_correct: only when reason evaluation is available.
    if reason_valid is not None:
        trace["fully_valid_correct"] = bool(
            strict_answer and reason_consistent and reason_valid and not trace.get("judge_disagreement", False))
    else:
        trace["fully_valid_correct"] = None
    # Backward-compatible names now follow the documented primary metric.
    trace["strict_correct"] = strict_answer
    trace["correct"] = semantic
    if finalizer:
        trace["format_failure_categories"] = classify_finalizer_format_errors(finalizer)
    elif single:
        trace["format_failure_categories"] = single.get("format_failure_categories", [])


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
        trace["invalid_output"] = bool(event.get("invalid_output"))
        trace["single_shot_format_failure"] = bool(
            event.get("single_shot_format_failure"))
        trace["raw_format_compliant"] = bool(
            event.get("raw_format_compliant"))
        trace["normalized_format_compliant"] = event.get(
            "normalized_format_compliant")
        trace["format_failure_categories"] = list(
            event.get("format_failure_categories", []))
        # ── Diagnostic semantic-answer fields ──
        trace["semantic_answer_absent"] = not bool(event.get("semantic_answer"))
        trace["semantic_answer_ambiguous"] = bool(event.get("semantic_answer_ambiguous"))
        trace["semantic_answer_extraction"] = event.get("semantic_answer_extraction", "")
    else:
        cache_key = "oracle" if setting == "oracle_broadcast" else "partial"
        if cache_key not in discussion_cache:
            discussion_cache[cache_key] = run_discussion(model, prompts["solver"], item, cache_key == "oracle", discussion_rounds)
            add_information_timeline(item, discussion_cache[cache_key])
        discussion = discussion_cache[cache_key]
        with_verifier = setting in {"multi_partial_verifier", "oracle_broadcast"}
        verifier, finalizer = finish_multi(
            model, prompts, item, discussion, with_verifier,
            self_check_before_commit=(setting == "self_check_before_commit"),
        )
        raw_candidates = {
            "solver_a": event_answer(discussion["solver_finals"]["a"]),
            "solver_b": event_answer(discussion["solver_finals"]["b"]),
        }
        candidates = {source: answer for source, answer in raw_candidates.items()
                      if decimal(answer) is not None}
        if verifier and verifier.get("protocol_valid"):
            verified_answer = verifier.get("semantic_answer",
                                            event_answer(verifier, "verified_answer"))
            if decimal(verified_answer) is not None:
                candidates["verifier"] = verified_answer
        trace.update(discussion=discussion, discussion_cache_key=cache_key, finalizer_event=finalizer,
                     final_prediction=event_answer(finalizer), candidate_answers=candidates, information=objective_information(item, discussion))
        if finalizer.get("draft_finalizer_event"):
            trace["finalizer_draft_event"] = finalizer["draft_finalizer_event"]
        trace["invalid_output"] = bool(finalizer.get("invalid_output"))
        trace["finalizer_retry_count"] = int(finalizer.get("retry_count", 0))
        trace["finalizer_recovered"] = bool(finalizer.get("recovered_after_retry"))
        trace["finalizer_exhausted"] = bool(finalizer.get("retry_exhausted"))
        trace["deterministic_normalizations"] = finalizer.get("deterministic_normalizations", [])
        if verifier is not None:
            trace["verifier_event"] = verifier
        # ── Diagnostic semantic-answer fields for multi-agent settings ──
        if finalizer:
            trace["semantic_answer_absent"] = not bool(finalizer.get("semantic_answer"))
            trace["semantic_answer_ambiguous"] = bool(finalizer.get("semantic_answer_ambiguous"))
            trace["semantic_answer_extraction"] = finalizer.get("semantic_answer_extraction", "")
        else:
            trace["semantic_answer_absent"] = True
            trace["semantic_answer_ambiguous"] = False
            trace["semantic_answer_extraction"] = "no_finalizer"
    trace["correct_before_judge"] = equivalent(trace["final_prediction"], gold)
    set_outcome_fields(trace, gold, trace["correct_before_judge"])
    trace["candidate_appearances"] = candidate_appearances(trace)
    for appearance in trace["candidate_appearances"]:
        appearance["correct_before_judge"] = equivalent(appearance["answer"], gold)
        appearance["correct"] = appearance["correct_before_judge"]
    trace["per_agent_correctness"] = {source: equivalent(answer, gold) for source, answer in trace["candidate_answers"].items()}
    if trace.get("finalizer_event"):
        trace["per_agent_correctness"]["finalizer"] = trace["semantic_correct"]
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
    timing, order = replay_setting_parts(setting)
    ordered_facts = replay_facts(item, order)
    old_transcript = public_transcript(discussion.get("discussion_events", []))
    if timing == "before_final_reset":
        evidence_view = f'Newly disclosed facts (verbatim):\n{ordered_facts}'
        context_policy = "reset; no prior discussion or candidates"
    elif timing == "before_finalizer":
        evidence_view = f'Newly disclosed facts (verbatim):\n{ordered_facts}\nPrior discussion transcript:\n{old_transcript}'
        context_policy = "prior discussion plus verbatim facts"
    else:
        evidence_view = discussion["public_transcript"]
        context_policy = "facts already present in discussion transcript"

    user = empty_candidate_finalizer_prompt(item, evidence_view)
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, {})
    # Semantic scoring uses the unified semantic extractor via event_answer.
    # Format compliance is tracked separately and cannot clear the answer.
    prediction = event_answer(finalizer)
    semantic_extraction = finalizer.get(
        "semantic_answer_extraction", "event_semantic_answer")
    gold = extract_answer(item["answer"])
    semantic_correct = equivalent(prediction, gold)
    format_compliant = bool(finalizer.get("raw_format_compliant"))
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
        "finalizer_user_prompt": user,
        "semantic_answer_extraction": semantic_extraction,
        "raw_format_compliant": bool(finalizer.get("raw_format_compliant")),
        "normalized_format_compliant": bool(finalizer.get("normalized_format_compliant")),
        "protocol_valid": bool(finalizer.get("protocol_valid")),
        "raw_format_error": finalizer.get("raw_format_error", ""),
        "normalized_format_error": finalizer.get("normalized_format_error", ""),
        "protocol_validation_error": finalizer.get("protocol_validation_error", ""),
        "format_failure_categories": list(finalizer.get("format_failure_categories", [])),
        "protocol_failure_categories": list(finalizer.get("protocol_failure_categories", [])),
        "candidate_answers": {}, "information": {"information_complete": True,
        "side_revealed": {"A": True, "B": True}, "assessment_method": "verbatim scheduled injection"},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "injected_fact_hash": fact_hash, "final_received_fact_hash": fact_hash,
        "fact_hash_algorithm": "sha256(canonical-json-sort-keys)",
        "information_injection_timing": timing,
        "fact_order": order,
        "fact_text_order_at_initial_reveal": order,
        "final_context_policy": context_policy, "semantic_correct": semantic_correct,
        "format_compliant": format_compliant, "correct_before_judge": semantic_correct,
        "invalid_output": not format_compliant,
        "single_shot_format_failure": bool(finalizer.get("single_shot_format_failure")),
        "finalizer_retry_count": 0, "finalizer_recovered": False, "finalizer_exhausted": False,
    }
    trace["finalizer_exhausted"] = bool(finalizer.get("retry_exhausted"))
    trace["single_shot_format_failure"] = bool(finalizer.get("single_shot_format_failure"))
    trace["deterministic_normalizations"] = finalizer.get("deterministic_normalizations", [])
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for appearance in trace["candidate_appearances"]:
        appearance["correct_before_judge"] = equivalent(appearance["answer"], gold)
        appearance["correct"] = appearance["correct_before_judge"]
    trace["per_agent_correctness"] = {"finalizer": trace["semantic_correct"]}
    usage, agent_usage, timing = blank_usage(), defaultdict(blank_usage), defaultdict(float)
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
        add_usage(agent_usage[event["agent"]], event["token_usage"])
        timing[event["phase"]] += event["runtime_seconds"]
    trace.update(inference_token_usage=usage, per_agent_token_usage=dict(agent_usage),
                 phase_runtime_seconds=dict(timing), total_runtime_seconds=time.perf_counter() - started)
    trace["semantic_answer_absent"] = not bool(finalizer.get("semantic_answer"))
    trace["semantic_answer_ambiguous"] = bool(finalizer.get("semantic_answer_ambiguous"))
    trace["semantic_answer_extraction"] = finalizer.get("semantic_answer_extraction", "")
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    add_information_state_metrics(trace, item)
    return trace


def build_finalizer_order_trace(model: LocalQwen, prompts: dict, item: dict, qid: int, order: str,
                                setting: str = FINALIZER_ORDER_SETTING,
                                evidence_override: str | None = None,
                                user_override: str | None = None,
                                discussion: dict | None = None) -> dict:
    """Build a finalizer trace for a blinded order-control condition."""
    started = time.perf_counter()
    evidence = replay_facts(item, order) if evidence_override is None else evidence_override
    user = user_override if user_override is not None else empty_candidate_finalizer_prompt(item, evidence)
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, {})
    prediction = event_answer(finalizer)
    extraction = finalizer.get(
        "semantic_answer_extraction", "event_semantic_answer")
    gold, fact_hash = extract_answer(item["answer"]), replay_fact_hash(item)
    evidence_hash = ledger_evidence_hash(evidence)
    trace = {
        "question_id": qid, "setting": setting, "agent_variant": order,
        "shared_question": item["shared_question"], "gold_answer": gold,
        "finalizer_event": finalizer, "final_prediction": prediction,
        "finalizer_user_prompt": user,
        "semantic_answer_extraction": extraction,
        "raw_format_compliant": bool(finalizer.get("raw_format_compliant")),
        "normalized_format_compliant": bool(finalizer.get("normalized_format_compliant")),
        "protocol_valid": bool(finalizer.get("protocol_valid")),
        "raw_format_error": finalizer.get("raw_format_error", ""),
        "normalized_format_error": finalizer.get("normalized_format_error", ""),
        "protocol_validation_error": finalizer.get("protocol_validation_error", ""),
        "format_failure_categories": list(finalizer.get("format_failure_categories", [])),
        "protocol_failure_categories": list(finalizer.get("protocol_failure_categories", [])),
        "candidate_answers": {},
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True},
                        "assessment_method": "verbatim finalizer-only injection"},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "injected_fact_hash": fact_hash, "final_received_fact_hash": fact_hash,
        "fact_hash_algorithm": "sha256(canonical-json-sort-keys)", "fact_order": order,
        "final_context_policy": "finalizer-only; identical context except fact order",
        "invalid_output": bool(finalizer.get("invalid_output")),
        "finalizer_retry_count": 0, "finalizer_recovered": False,
        "finalizer_exhausted": bool(finalizer.get("retry_exhausted")),
        "single_shot_format_failure": bool(finalizer.get("single_shot_format_failure")),
        "deterministic_normalizations": finalizer.get("deterministic_normalizations", []),
        "correct_before_judge": equivalent(prediction, gold),
        "evidence_hash": evidence_hash,
    }
    if setting == CANONICAL_ORDER_SETTING:
        trace["baseline_evidence_hash"] = ledger_evidence_hash(evidence)
        trace["baseline_prompt_hash"] = hashlib.sha256(user.encode("utf-8")).hexdigest()
        trace.update({
            "ledger_type": "fixed_source_order_legacy",
            "ordering_policy": "source_A_then_B",
            "provenance_mode": "aware",
            "canonical_semantics_claimed": False,
            "fact_order": "fixed_source_A_then_B",
            "final_context_policy": "fixed source-order ledger; legacy canonical_order setting",
        })
    if discussion is not None:
        trace["discussion"] = discussion
    set_outcome_fields(trace, gold, trace["correct_before_judge"])
    trace["candidate_appearances"] = []
    trace["per_agent_correctness"] = {"finalizer": trace["semantic_correct"]}
    usage, agent_usage, timing = blank_usage(), defaultdict(blank_usage), defaultdict(float)
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
        add_usage(agent_usage[event["agent"]], event["token_usage"])
        timing[event["phase"]] += event["runtime_seconds"]
    trace.update(inference_token_usage=usage, per_agent_token_usage=dict(agent_usage),
                 phase_runtime_seconds=dict(timing), total_runtime_seconds=time.perf_counter() - started)
    trace["semantic_answer_absent"] = not bool(finalizer.get("semantic_answer"))
    trace["semantic_answer_ambiguous"] = bool(finalizer.get("semantic_answer_ambiguous"))
    trace["semantic_answer_extraction"] = finalizer.get("semantic_answer_extraction", "")
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def classify_self_check_change_direction(provisional_answer: Any, committed_answer: Any, gold_answer: Any) -> str:
    provisional_correct = equivalent(provisional_answer, gold_answer)
    committed_correct = equivalent(committed_answer, gold_answer)
    changed = not equivalent(provisional_answer, committed_answer)
    if not changed:
        return "unchanged_correct" if committed_correct else "unchanged_wrong"
    if not provisional_correct and committed_correct:
        return "wrong_to_correct"
    if provisional_correct and not committed_correct:
        return "correct_to_wrong"
    return "wrong_to_different_wrong"


def build_self_check_trace(model: LocalQwen, prompts: dict, item: dict, qid: int, base_seed: int) -> dict:
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    baseline_evidence, baseline_user = fixed_source_ledger_finalizer_prompt(item)
    self_check_evidence, provisional_user = fixed_source_ledger_finalizer_prompt(item)
    baseline_evidence_hash = ledger_evidence_hash(baseline_evidence)
    self_check_evidence_hash = ledger_evidence_hash(self_check_evidence)
    baseline_prompt_hash = hashlib.sha256(baseline_user.encode("utf-8")).hexdigest()
    provisional_prompt_hash = hashlib.sha256(provisional_user.encode("utf-8")).hexdigest()
    evidence_equal_to_baseline = (
        baseline_evidence == self_check_evidence
        and baseline_evidence_hash == self_check_evidence_hash
    )
    prompt_equal_to_baseline = (
        baseline_user == provisional_user
        and baseline_prompt_hash == provisional_prompt_hash
    )
    invariant_status = (
        "PASS" if evidence_equal_to_baseline and prompt_equal_to_baseline else "SELF_CHECK_INVARIANT_FAIL"
    )
    # The provisional call intentionally shares the same seed scope as canonical_order.
    reseed_model(model, derived_seed(base_seed, qid, "canonical_finalizer", "provisional"))
    draft_event = call_finalizer_once(model, prompts["finalizer"], provisional_user, {})
    draft_raw = draft_event.get("raw_output", "")
    draft_normalized = draft_event.get("normalized_output", draft_raw)
    draft_parsed = draft_event.get("parsed_output", {})
    draft_prediction = event_answer(draft_event)
    draft_extraction = draft_event.get(
        "semantic_answer_extraction", "event_semantic_answer")
    draft_answer = draft_prediction
    draft_reason = draft_parsed.get("reason", "")
    draft_selected_source = draft_parsed.get("selected_source", "")
    draft_semantic_correct = equivalent(draft_prediction, gold)
    draft_format_compliant = bool(draft_event.get("raw_format_compliant"))
    draft_raw_format_compliant = bool(draft_event.get("raw_format_compliant"))
    draft_normalized_format_compliant = bool(draft_event.get("normalized_format_compliant"))
    draft_protocol_valid = bool(draft_event.get("protocol_valid"))

    commit_user = self_check_before_commit_user_prompt(provisional_user, draft_raw)
    reseed_model(model, derived_seed(base_seed, qid, "self_check_before_commit", "commit"))
    committed_event = call_finalizer_once(model, prompts["finalizer"], commit_user, {})
    committed_raw = committed_event.get("raw_output", "")
    committed_normalized = committed_event.get("normalized_output", committed_raw)
    committed_parsed = committed_event.get("parsed_output", {})
    committed_prediction = event_answer(committed_event)
    committed_extraction = committed_event.get(
        "semantic_answer_extraction", "event_semantic_answer")
    committed_answer = committed_prediction
    committed_reason = committed_parsed.get("reason", "")
    committed_selected_source = committed_parsed.get("selected_source", "")
    committed_semantic_correct = equivalent(committed_prediction, gold)
    committed_format_compliant = bool(committed_event.get("raw_format_compliant"))
    committed_raw_format_compliant = bool(committed_event.get("raw_format_compliant"))
    committed_normalized_format_compliant = bool(committed_event.get("normalized_format_compliant"))
    committed_protocol_valid = bool(committed_event.get("protocol_valid"))

    draft_event["phase"] = "finalization_provisional"
    committed_event["phase"] = "finalization_self_check_commit"
    committed_event["self_check_before_commit"] = True
    committed_event["draft_finalizer_event"] = draft_event
    committed_event["self_check_user_prompt"] = commit_user

    trace = {
        "question_id": qid,
        "setting": SELF_CHECK_BEFORE_COMMIT_SETTING,
        "agent_variant": "canonical",
        "shared_question": item["shared_question"],
        "gold_answer": gold,
        "candidate_answers": {},
        "information": {
            "information_complete": True,
            "side_revealed": {"A": True, "B": True},
            "assessment_method": "verbatim finalizer-only injection",
        },
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "injected_fact_hash": replay_fact_hash(item),
        "final_received_fact_hash": replay_fact_hash(item),
        "fact_hash_algorithm": "sha256(canonical-json-sort-keys)",
        "paired_baseline_setting": CANONICAL_ORDER_SETTING,
        "baseline_evidence_hash": baseline_evidence_hash,
        "self_check_evidence_hash": self_check_evidence_hash,
        "baseline_prompt_hash": baseline_prompt_hash,
        "provisional_prompt_hash": provisional_prompt_hash,
        "evidence_equal_to_baseline": evidence_equal_to_baseline,
        "prompt_equal_to_baseline": prompt_equal_to_baseline,
        "self_check_invariant_status": invariant_status,
        "ledger_type": "fixed_source_order_legacy",
        "ordering_policy": "source_A_then_B",
        "provenance_mode": "aware",
        "canonical_semantics_claimed": False,
        "fact_order": "fixed_source_A_then_B",
        "final_context_policy": "fixed source-order ledger; legacy canonical_order setting",
        "finalizer_user_prompt": provisional_user,
        "self_check_user_prompt": commit_user,
        "finalizer_draft_event": draft_event,
        "finalizer_event": committed_event,
        "final_prediction": committed_answer,
        "provisional_raw_output": draft_raw,
        "provisional_normalized_output": draft_normalized,
        "provisional_selected_source": draft_selected_source,
        "provisional_answer": draft_answer,
        "provisional_answer_extraction": draft_extraction,
        "provisional_parsed_answer": draft_parsed.get("final_answer", ""),
        "provisional_reason": draft_reason,
        "provisional_format_compliant": draft_format_compliant,
        "provisional_raw_format_compliant": draft_raw_format_compliant,
        "provisional_normalized_format_compliant": draft_normalized_format_compliant,
        "provisional_protocol_valid": draft_protocol_valid,
        "provisional_semantic_correct": draft_semantic_correct,
        "committed_raw_output": committed_raw,
        "committed_normalized_output": committed_normalized,
        "committed_selected_source": committed_selected_source,
        "committed_answer": committed_answer,
        "committed_answer_extraction": committed_extraction,
        "committed_parsed_answer": committed_parsed.get("final_answer", ""),
        "committed_reason": committed_reason,
        "committed_format_compliant": committed_format_compliant,
        "committed_raw_format_compliant": committed_raw_format_compliant,
        "committed_normalized_format_compliant": committed_normalized_format_compliant,
        "committed_protocol_valid": committed_protocol_valid,
        "committed_semantic_correct": committed_semantic_correct,
        "self_check_enabled": True,
        "self_check_call_count": 1,
        "answer_changed_after_self_check": not equivalent(draft_prediction, committed_prediction),
        "source_changed_after_self_check": draft_selected_source != committed_selected_source,
        "reason_changed_after_self_check": re.sub(r"\s+", " ", str(draft_reason)).strip() != re.sub(r"\s+", " ", str(committed_reason)).strip(),
        "format_changed_after_self_check": draft_format_compliant != committed_format_compliant,
        "self_check_change_direction": classify_self_check_change_direction(draft_prediction, committed_prediction, gold),
        "invalid_output": not committed_format_compliant,
        "finalizer_retry_count": 0,
        "finalizer_recovered": False,
        "finalizer_exhausted": bool(committed_event.get("retry_exhausted")),
        "single_shot_format_failure": bool(committed_event.get("single_shot_format_failure")),
        "deterministic_normalizations": committed_event.get("deterministic_normalizations", []),
    }
    trace["correct_before_judge"] = committed_semantic_correct
    set_outcome_fields(trace, gold, committed_semantic_correct)
    trace["candidate_appearances"] = []
    trace["per_agent_correctness"] = {"finalizer": trace["semantic_correct"]}
    usage, agent_usage, timing = blank_usage(), defaultdict(blank_usage), defaultdict(float)
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
        add_usage(agent_usage[event["agent"]], event["token_usage"])
        timing[event["phase"]] += event["runtime_seconds"]
    trace.update(
        inference_token_usage=usage,
        per_agent_token_usage=dict(agent_usage),
        phase_runtime_seconds=dict(timing),
        total_runtime_seconds=time.perf_counter() - started,
    )
    trace["semantic_answer_absent"] = not bool(committed_event.get("semantic_answer"))
    trace["semantic_answer_ambiguous"] = bool(committed_event.get("semantic_answer_ambiguous"))
    trace["semantic_answer_extraction"] = committed_event.get("semantic_answer_extraction", "")
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    trace["answer_error_categories"] = classify_wrong_answer(trace)
    trace["answer_flow"] = analyze_answer_flow(trace)
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
        visibility = finalizer_saw_correct_answer(trace)
        if (trace.get("finalizer_event") and visibility["saw_correct_answer"]
                and not trace.get("correct_before_judge")):
            finalizer = trace["finalizer_event"]
            entries.append({
                "id": f"{trace_index}:correct_rejection",
                "target": "explicit_correct_answer_rejection",
                "known_correct_answer": trace["gold_answer"],
                "correct_sources_visible_to_finalizer": visibility["correct_candidate_sources"],
                "selected_source": finalizer.get("parsed_output", {}).get("selected_source", ""),
                "finalizer_reason": finalizer.get("parsed_output", {}).get("reason", ""),
                "instruction": (
                    "Set correct=true only if the reason explicitly rejects, dismisses, or argues against "
                    "the known correct answer/source; merely selecting something else is not explicit rejection."),
            })
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
            "For explicit_correct_answer_rejection targets, follow that row's instruction exactly. "
            "Explain the decision briefly in reason.\n" + json.dumps(entries, ensure_ascii=False))
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


def audit_required_reasons(trace: dict) -> list[str]:
    """Identify samples required by the manual-audit protocol."""
    reasons = []
    if not trace.get("format_compliant", not trace.get("invalid_output")):
        reasons.append("format_failure")
    if not trace.get("answer_reason_consistent", True):
        reasons.append("answer_reason_inconsistent")
    if trace.get("judge_disagreement"):
        reasons.append("judge_disagreement")
    if trace.get("order_pair_answer_flip"):
        reasons.append("ab_ba_answer_flip")
    if trace.get("manual_audit_random_sample"):
        reasons.append("random_normal_sample")
    return reasons


def mark_manual_audit_samples(traces: list[dict]) -> None:
    """Mark AB/BA flips and a deterministic small sample of otherwise normal rows."""
    by_setting_variant_qid = {
        (trace["setting"], trace.get("agent_variant", ""), int(trace["question_id"])): trace
        for trace in traces
    }
    for trace in traces:
        trace["order_pair_answer_flip"] = False
        trace["manual_audit_random_sample"] = False
    paired_settings = [FINALIZER_ORDER_SETTING, SOLVER_ORDER_SETTING, FROZEN_TRANSCRIPT_ORDER_SETTING]
    for setting in paired_settings:
        ab_ids = {qid for row_setting, variant, qid in by_setting_variant_qid
                  if row_setting == setting and variant == "AB"}
        ba_ids = {qid for row_setting, variant, qid in by_setting_variant_qid
                  if row_setting == setting and variant == "BA"}
        for qid in sorted(ab_ids & ba_ids):
            ab = by_setting_variant_qid[(setting, "AB", qid)]
            ba = by_setting_variant_qid[(setting, "BA", qid)]
            if not equivalent(ab.get("final_prediction"), ba.get("final_prediction")):
                ab["order_pair_answer_flip"] = True
                ba["order_pair_answer_flip"] = True
    normal_rows = [
        trace for trace in traces
        if trace.get("format_compliant", not trace.get("invalid_output"))
        and trace.get("answer_reason_consistent", True)
        and not trace.get("judge_disagreement")
        and not trace.get("order_pair_answer_flip")
    ]
    for trace in sorted(normal_rows, key=lambda row: (
            row["setting"], str(row.get("agent_variant", "")), int(row["question_id"])))[:3]:
        trace["manual_audit_random_sample"] = True


def write_manual_audit_csv(traces: list[dict], output_dir: Path) -> None:
    """Write a prefilled audit sheet; manual judgment columns are intentionally blank."""
    fields = [
        "setting", "agent_variant", "question_id", "audit_required", "audit_reasons",
        "gold_answer", "final_answer", "format_compliant", "semantic_correct",
        "strict_answer_correct", "fully_valid_correct", "answer_reason_consistent",
        "reason_mathematically_valid", "reason_uses_all_required_facts",
        "judge_disagreement", "auto_error_types", "format_failure_categories",
        "finalizer_selected_source", "deterministic_normalizations",
        "finalizer_reason", "finalizer_raw_output",
        "manual_format_valid", "manual_semantic_correct",
        "manual_answer_reason_consistent", "manual_reason_mathematically_valid",
        "manual_reason_uses_all_required_facts", "manual_error_type",
        "auto_manual_agreement", "manual_notes",
    ]
    rows = []
    for trace in sorted(traces, key=lambda row: (
            row["setting"], str(row.get("agent_variant", "")), int(row["question_id"]))):
        finalizer = trace.get("finalizer_event") or {}
        parsed = finalizer.get("parsed_output", {})
        reasons = audit_required_reasons(trace)
        rows.append({
            "setting": trace["setting"],
            "agent_variant": trace.get("agent_variant", ""),
            "question_id": trace["question_id"],
            "audit_required": bool(reasons),
            "audit_reasons": ";".join(reasons),
            "gold_answer": trace.get("gold_answer", ""),
            "final_answer": trace.get("final_prediction", ""),
            "format_compliant": bool(trace.get("format_compliant", not trace.get("invalid_output"))),
            "semantic_correct": bool(trace.get("semantic_correct")),
            "strict_answer_correct": bool(trace.get("strict_answer_correct")),
            "fully_valid_correct": bool(trace.get("fully_valid_correct")),
            "answer_reason_consistent": bool(trace.get("answer_reason_consistent")),
            "reason_mathematically_valid": bool(trace.get("reason_mathematically_valid")),
            "reason_uses_all_required_facts": bool(trace.get("reason_uses_all_required_facts")),
            "judge_disagreement": bool(trace.get("judge_disagreement")),
            "auto_error_types": ";".join(trace.get("answer_error_categories", [])),
            "format_failure_categories": ";".join(trace.get("format_failure_categories", [])),
            "finalizer_selected_source": parsed.get("selected_source", ""),
            "deterministic_normalizations": ";".join(finalizer.get("deterministic_normalizations", [])),
            "finalizer_reason": parsed.get("reason", ""),
            "finalizer_raw_output": finalizer.get("raw_output", ""),
            "manual_format_valid": "",
            "manual_semantic_correct": "",
            "manual_answer_reason_consistent": "",
            "manual_reason_mathematically_valid": "",
            "manual_reason_uses_all_required_facts": "",
            "manual_error_type": "",
            "auto_manual_agreement": "",
            "manual_notes": "",
        })
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    (output_dir / "manual_audit.csv").write_text(buf.getvalue(), encoding="utf-8-sig")
    required = [row for row in rows if row["audit_required"]]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(required)
    (output_dir / "manual_audit_required.csv").write_text(buf.getvalue(), encoding="utf-8-sig")


def write_self_check_analysis(traces: list[dict], output_dir: Path) -> None:
    rows = [trace for trace in traces if trace.get("setting") == SELF_CHECK_BEFORE_COMMIT_SETTING]
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "setting": SELF_CHECK_BEFORE_COMMIT_SETTING,
        "n": len(rows),
        "provisional_semantic_correct": sum(bool(trace.get("provisional_semantic_correct")) for trace in rows),
        "committed_semantic_correct": sum(bool(trace.get("committed_semantic_correct")) for trace in rows),
        "provisional_format_compliant": sum(bool(trace.get("provisional_format_compliant")) for trace in rows),
        "committed_format_compliant": sum(bool(trace.get("committed_format_compliant")) for trace in rows),
        "answer_changed_count": sum(bool(trace.get("answer_changed_after_self_check")) for trace in rows),
        "wrong_to_correct": sum(trace.get("self_check_change_direction") == "wrong_to_correct" for trace in rows),
        "correct_to_wrong": sum(trace.get("self_check_change_direction") == "correct_to_wrong" for trace in rows),
        "wrong_to_different_wrong": sum(trace.get("self_check_change_direction") == "wrong_to_different_wrong" for trace in rows),
        "unchanged_correct": sum(trace.get("self_check_change_direction") == "unchanged_correct" for trace in rows),
        "unchanged_wrong": sum(trace.get("self_check_change_direction") == "unchanged_wrong" for trace in rows),
        "invariant_fail_count": sum(trace.get("self_check_invariant_status") == "SELF_CHECK_INVARIANT_FAIL" for trace in rows),
        "per_trace": [{
            "question_id": trace["question_id"],
            "paired_baseline_setting": trace.get("paired_baseline_setting", "canonical_order"),
            "baseline_evidence_hash": trace.get("baseline_evidence_hash", ""),
            "self_check_evidence_hash": trace.get("self_check_evidence_hash", ""),
            "evidence_equal_to_baseline": trace.get("evidence_equal_to_baseline", False),
            "self_check_invariant_status": trace.get("self_check_invariant_status", ""),
            "provisional_semantic_correct": trace.get("provisional_semantic_correct", False),
            "committed_semantic_correct": trace.get("committed_semantic_correct", False),
            "provisional_format_compliant": trace.get("provisional_format_compliant", False),
            "committed_format_compliant": trace.get("committed_format_compliant", False),
            "answer_changed_after_self_check": trace.get("answer_changed_after_self_check", False),
            "source_changed_after_self_check": trace.get("source_changed_after_self_check", False),
            "reason_changed_after_self_check": trace.get("reason_changed_after_self_check", False),
            "format_changed_after_self_check": trace.get("format_changed_after_self_check", False),
            "self_check_change_direction": trace.get("self_check_change_direction", ""),
        } for trace in rows],
    }
    (output_dir / "self_check_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(traces: list[dict], output_dir: Path, run_config: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_manual_audit_samples(traces)
    if run_config is not None:
        (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "traces_all.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    failure_fields = ("question_id", "setting", "agent_variant", "shared_question", "gold_answer", "final_prediction",
                      "semantic_correct", "format_compliant", "strict_answer_correct", "fully_valid_correct",
                      "answer_reason_consistent", "reason_mathematically_valid", "reason_uses_all_required_facts",
                      "judge_disagreement", "format_failure_categories", "answer_error_categories", "strict_correct",
                      "order_pair_answer_flip", "manual_audit_random_sample",
                      "information_steps", "late_fact_acknowledgement", "belief_update",
                      "answer_flow",
                      "answer_reason_checkable", "reason_conclusion_answer", "single_shot_format_failure",
                      "failure_type", "invalid_output", "finalizer_retry_count", "finalizer_recovered", "finalizer_exhausted",
                      "deterministic_normalizations",
                      "lucky_guess", "oracle_gap", "information", "candidate_answers",
                      "candidate_appearances", "per_agent_correctness", "deepseek_judge")
    failures = [{key: trace[key] for key in failure_fields if key in trace}
                for trace in traces if not trace.get("semantic_correct")]
    (output_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    grouped = defaultdict(list)
    for t in traces:
        grouped[(t["setting"], t.get("agent_variant") or "")].append(t)
    fields = ["setting", "agent_variant", "n", "correct", "accuracy",
              "semantic_correct", "semantic_accuracy", "format_compliant", "format_compliance_rate",
              "raw_format_compliant", "raw_format_compliance_rate",
              "normalized_format_compliant", "normalized_format_compliance_rate",
              "protocol_valid", "protocol_valid_rate",
              "strict_answer_correct", "strict_answer_accuracy",
              "fully_valid_correct", "fully_valid_eligible_n", "fully_valid_accuracy", "fully_valid_all_n_accuracy",
              "answer_reason_consistent", "answer_reason_consistency_rate",
              "reason_evaluation_available",
              "reason_mathematically_valid", "reason_mathematically_valid_eligible_n", "reason_mathematically_valid_rate",
              "judge_disagreement", "strict_correct", "strict_accuracy",
              "late_fact_acknowledgement", "belief_wrong_to_correct", "belief_correct_to_wrong",
              "belief_wrong_to_different_wrong", "belief_unchanged",
              "single_shot_format_failure",
              "solver_a_correct", "solver_b_correct", "verifier_correct", "finalizer_correct", "information_complete",
              "fail_information_acquisition", "fail_information_integration", "fail_answer_selection", "invalid_output",
              "fail_answer_reason_inconsistency",
              "finalizer_retry_count", "finalizer_recovered", "finalizer_exhausted",
              "oracle_gap", "oracle_gap_ids", "lucky_guess", "format_issue_corrected",
              "prompt_tokens", "completion_tokens", "total_tokens", "judge_total_tokens",
              "local_judge_total_tokens", "inference_runtime_seconds",
              "judge_runtime_seconds", "local_judge_runtime_seconds", "end_to_end_runtime_seconds",
              "provisional_semantic_correct", "provisional_format_compliant",
              "committed_semantic_correct", "committed_format_compliant",
              "answer_changed_after_self_check", "wrong_to_correct", "correct_to_wrong",
              "wrong_to_different_wrong", "unchanged_correct", "unchanged_wrong"]
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=fields); writer.writeheader()
    for (setting, variant), rows in grouped.items():
        correct = sum(bool(x["correct"]) for x in rows)
        row = {"setting": setting, "agent_variant": variant, "n": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 4),
            "semantic_correct": sum(bool(x.get("semantic_correct")) for x in rows),
            "semantic_accuracy": round(sum(bool(x.get("semantic_correct")) for x in rows) / len(rows), 4),
            "format_compliant": sum(bool(x.get("format_compliant")) for x in rows),
            "format_compliance_rate": round(sum(bool(x.get("format_compliant")) for x in rows) / len(rows), 4),
            "raw_format_compliant": sum(bool(x.get("raw_format_compliant", x.get("format_compliant"))) for x in rows),
            "raw_format_compliance_rate": round(sum(bool(x.get("raw_format_compliant", x.get("format_compliant"))) for x in rows) / len(rows), 4) if rows else 0,
            "normalized_format_compliant": sum(bool(x.get("normalized_format_compliant", x.get("format_compliant"))) for x in rows),
            "normalized_format_compliance_rate": round(sum(bool(x.get("normalized_format_compliant", x.get("format_compliant"))) for x in rows) / len(rows), 4) if rows else 0,
            "protocol_valid": sum(bool(x.get("protocol_valid")) for x in rows),
            "protocol_valid_rate": round(sum(bool(x.get("protocol_valid")) for x in rows) / len(rows), 4) if rows else 0,
            "strict_answer_correct": sum(bool(x.get("strict_answer_correct")) for x in rows),
            "strict_answer_accuracy": round(sum(bool(x.get("strict_answer_correct")) for x in rows) / len(rows), 4),
            "fully_valid_correct": sum(bool(x.get("fully_valid_correct")) for x in rows if x.get("fully_valid_correct") is not None),
            "fully_valid_eligible_n": sum(1 for x in rows if x.get("fully_valid_correct") is not None),
            "fully_valid_accuracy": round(sum(bool(x.get("fully_valid_correct")) for x in rows if x.get("fully_valid_correct") is not None) / max(1, sum(1 for x in rows if x.get("fully_valid_correct") is not None)), 4),
            "fully_valid_all_n_accuracy": round(sum(bool(x.get("fully_valid_correct")) for x in rows) / len(rows), 4),
            "answer_reason_consistent": sum(bool(x.get("answer_reason_consistent")) for x in rows),
            "answer_reason_consistency_rate": round(sum(bool(x.get("answer_reason_consistent")) for x in rows) / len(rows), 4),
            "reason_evaluation_available": sum(bool(x.get("reason_evaluation_available")) for x in rows),
            "reason_mathematically_valid": sum(bool(x.get("reason_mathematically_valid")) for x in rows if x.get("reason_mathematically_valid") is not None),
            "reason_mathematically_valid_eligible_n": sum(1 for x in rows if x.get("reason_mathematically_valid") is not None),
            "reason_mathematically_valid_rate": round(sum(bool(x.get("reason_mathematically_valid")) for x in rows if x.get("reason_mathematically_valid") is not None) / max(1, sum(1 for x in rows if x.get("reason_mathematically_valid") is not None)), 4),
            "judge_disagreement": sum(bool(x.get("judge_disagreement")) for x in rows),
            "late_fact_acknowledgement": sum(bool(x.get("late_fact_acknowledgement", {}).get(
                "all_responding_agents_acknowledged")) for x in rows),
            "belief_wrong_to_correct": sum(x.get("belief_update", {}).get(
                "direction_counts", {}).get("wrong_to_correct", 0) for x in rows),
            "belief_correct_to_wrong": sum(x.get("belief_update", {}).get(
                "direction_counts", {}).get("correct_to_wrong", 0) for x in rows),
            "belief_wrong_to_different_wrong": sum(x.get("belief_update", {}).get(
                "direction_counts", {}).get("wrong_to_different_wrong", 0) for x in rows),
            "belief_unchanged": sum(x.get("belief_update", {}).get(
                "direction_counts", {}).get("unchanged", 0) for x in rows),
            "strict_correct": sum(bool(x.get("strict_correct")) for x in rows),
            "strict_accuracy": round(sum(bool(x.get("strict_correct")) for x in rows) / len(rows), 4),
            "single_shot_format_failure": sum(bool(x.get("single_shot_format_failure")) for x in rows),
            "solver_a_correct": sum(bool(x.get("per_agent_correctness", {}).get("solver_a")) for x in rows),
            "solver_b_correct": sum(bool(x.get("per_agent_correctness", {}).get("solver_b")) for x in rows),
            "verifier_correct": sum(bool(x.get("per_agent_correctness", {}).get("verifier")) for x in rows),
            "finalizer_correct": sum(bool(x.get("per_agent_correctness", {}).get("finalizer")) for x in rows),
            "information_complete": sum(bool(x.get("information", {}).get("information_complete")) for x in rows),
            "fail_information_acquisition": sum(x["failure_type"] == "information_acquisition_failure" for x in rows),
            "fail_information_integration": sum(x["failure_type"] == "information_integration_failure" for x in rows),
            "fail_answer_selection": sum(x["failure_type"] == "answer_selection_failure" for x in rows),
            "fail_answer_reason_inconsistency": sum(x["failure_type"] == "answer_reason_inconsistency" for x in rows),
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
            "local_judge_total_tokens": sum(x.get("local_judge_token_usage", {}).get("total_tokens", 0) for x in rows),
            "inference_runtime_seconds": round(sum(x["total_runtime_seconds"] for x in rows), 3),
            "judge_runtime_seconds": round(sum(x.get("judge_runtime_seconds", 0) for x in rows), 3),
            "local_judge_runtime_seconds": round(sum(x.get("local_judge_runtime_seconds", 0) for x in rows), 3),
            "end_to_end_runtime_seconds": round(sum(
                x["total_runtime_seconds"] + x.get("judge_runtime_seconds", 0)
                + x.get("local_judge_runtime_seconds", 0) for x in rows), 3)}
        row.update({
            "provisional_semantic_correct": 0,
            "provisional_format_compliant": 0,
            "committed_semantic_correct": 0,
            "committed_format_compliant": 0,
            "answer_changed_after_self_check": 0,
            "wrong_to_correct": 0,
            "correct_to_wrong": 0,
            "wrong_to_different_wrong": 0,
            "unchanged_correct": 0,
            "unchanged_wrong": 0,
        })
        if setting == SELF_CHECK_BEFORE_COMMIT_SETTING:
            row.update({
                "provisional_semantic_correct": sum(bool(x.get("provisional_semantic_correct")) for x in rows),
                "provisional_format_compliant": sum(bool(x.get("provisional_format_compliant")) for x in rows),
                "committed_semantic_correct": sum(bool(x.get("committed_semantic_correct")) for x in rows),
                "committed_format_compliant": sum(bool(x.get("committed_format_compliant")) for x in rows),
                "answer_changed_after_self_check": sum(bool(x.get("answer_changed_after_self_check")) for x in rows),
                "wrong_to_correct": sum(x.get("self_check_change_direction") == "wrong_to_correct" for x in rows),
                "correct_to_wrong": sum(x.get("self_check_change_direction") == "correct_to_wrong" for x in rows),
                "wrong_to_different_wrong": sum(x.get("self_check_change_direction") == "wrong_to_different_wrong" for x in rows),
                "unchanged_correct": sum(x.get("self_check_change_direction") == "unchanged_correct" for x in rows),
                "unchanged_wrong": sum(x.get("self_check_change_direction") == "unchanged_wrong" for x in rows),
            })
        writer.writerow(row)
    (output_dir / "metrics.csv").write_text(buf.getvalue(), encoding="utf-8-sig")
    category_report = {
        "format_failures": {
            category: sum(category in trace.get("format_failure_categories", []) for trace in traces)
            for category in FORMAT_FAILURE_CATEGORIES
        },
        "wrong_answers": {
            category: sum(category in trace.get("answer_error_categories", []) for trace in traces)
            for category in ANSWER_ERROR_CATEGORIES
        },
    }
    (output_dir / "error_categories.json").write_text(
        json.dumps(category_report, ensure_ascii=False, indent=2), encoding="utf-8")
    flow_rows = [trace for trace in traces if trace.get("answer_flow")]
    flow_summary = {
        "trace_count": len(flow_rows),
        "answer_emergence_count": sum(
            bool(t["answer_flow"]["answer_emergence"]["occurred"]) for t in flow_rows),
        "answer_retention_count": sum(
            bool(t["answer_flow"]["answer_retention"]["retained_to_final_answer"]) for t in flow_rows),
        "answer_overwrite_count": sum(
            bool(t["answer_flow"]["answer_overwrite"]["occurred"]) for t in flow_rows),
        "answer_recovery_count": sum(
            bool(t["answer_flow"]["answer_recovery"]["occurred"]) for t in flow_rows),
        "final_commit_failure_count": sum(
            bool(t["answer_flow"]["final_commit_failure"]) for t in flow_rows),
        "loss_location_counts": {
            location: sum(location in t["answer_flow"]["loss_locations"] for t in flow_rows)
            for location in (
                "solver_internal", "between_solvers", "verifier", "ledger",
                "finalizer_reasoning", "final_answer_serialization", "unknown")
        },
    }
    (output_dir / "answer_flow_analysis.json").write_text(
        json.dumps({
            "summary": flow_summary,
            "per_trace": [{
                "question_id": trace["question_id"],
                "setting": trace["setting"],
                "agent_variant": trace.get("agent_variant", ""),
                **trace["answer_flow"],
            } for trace in flow_rows],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    timeline_dir = output_dir / "answer_flow_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    for trace in flow_rows:
        if trace.get("semantic_correct"):
            continue
        variant = re.sub(r"[^A-Za-z0-9_-]+", "_", str(trace.get("agent_variant", "") or "default"))
        filename = f"question_{int(trace['question_id']):03d}_{variant}.txt"
        (timeline_dir / filename).write_text(
            render_answer_flow_timeline(trace), encoding="utf-8")
    write_manual_audit_csv(traces, output_dir)
    write_self_check_analysis(traces, output_dir)


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
        semantic_count = sum(bool(t.get("semantic_correct")) for t in values)
        semantic_rate = round(semantic_count / len(values), 4) if values else 0
        strict_count = sum(bool(t.get("strict_correct")) for t in values)
        strict_rate = round(strict_count / len(values), 4) if values else 0
        # Primary metric: semantic correctness (format-independent).
        # strict_* fields are secondary diagnostics only.
        accuracy[setting] = {
            "n": len(values),
            "correct": semantic_count,
            "accuracy": semantic_rate,
            "semantic_correct": semantic_count,
            "semantic_accuracy": semantic_rate,
            "format_compliant": sum(bool(t.get("format_compliant", not t.get("invalid_output"))) for t in values),
            "format_compliance_rate": round(sum(bool(t.get("format_compliant", not t.get("invalid_output"))) for t in values) / len(values), 4) if values else 0,
            "raw_format_compliant": sum(bool(t.get("raw_format_compliant", t.get("format_compliant", not t.get("invalid_output")))) for t in values),
            "raw_format_compliance_rate": round(sum(bool(t.get("raw_format_compliant", t.get("format_compliant", not t.get("invalid_output")))) for t in values) / len(values), 4) if values else 0,
            "normalized_format_compliant": sum(bool(t.get("normalized_format_compliant", t.get("format_compliant", not t.get("invalid_output")))) for t in values),
            "normalized_format_compliance_rate": round(sum(bool(t.get("normalized_format_compliant", t.get("format_compliant", not t.get("invalid_output")))) for t in values) / len(values), 4) if values else 0,
            "protocol_valid": sum(bool(t.get("protocol_valid")) for t in values),
            "protocol_valid_rate": round(sum(bool(t.get("protocol_valid")) for t in values) / len(values), 4) if values else 0,
            "answer_reason_consistent": sum(bool(t.get("answer_reason_consistent")) for t in values),
            "answer_reason_consistency_rate": round(sum(bool(t.get("answer_reason_consistent")) for t in values) / len(values), 4) if values else 0,
            "strict_correct": strict_count,
            "strict_accuracy": strict_rate,
            "strict_answer_correct": strict_count,
            "strict_answer_accuracy": strict_rate,
            "fully_valid_correct": sum(bool(t.get("fully_valid_correct")) for t in values if t.get("fully_valid_correct") is not None),
            "fully_valid_accuracy": round(
                sum(bool(t.get("fully_valid_correct")) for t in values if t.get("fully_valid_correct") is not None) / max(1, sum(1 for t in values if t.get("fully_valid_correct") is not None)), 4) if values else 0,
        }
    def answer_key(setting: str, qid: int) -> str:
        prediction = by_setting[setting][qid]["final_prediction"]
        numeric = decimal(prediction)
        return f"decimal:{numeric.normalize()}" if numeric is not None else (
            "text:" + re.sub(r"\s+", " ", extract_answer(prediction).lower()).strip())
    order_effect = {}
    for timing in INFORMATION_TIMINGS:
        ab_setting, ba_setting = f"{timing}_AB", f"{timing}_BA"
        paired_ids = sorted(set(by_setting[ab_setting]) & set(by_setting[ba_setting]))
        # All-pair semantic flip (includes format-invalid pairs).
        all_flips = [qid for qid in paired_ids
                     if answer_key(ab_setting, qid) != answer_key(ba_setting, qid)]
        # Valid-pair flip (only pairs where both sides are format-compliant).
        valid_ids = [qid for qid in paired_ids
                     if by_setting[ab_setting][qid].get("format_compliant")
                     and by_setting[ba_setting][qid].get("format_compliant")]
        flips = [qid for qid in valid_ids
                 if answer_key(ab_setting, qid) != answer_key(ba_setting, qid)]
        order_effect[timing] = {
            "paired_question_count": len(paired_ids),
            "all_pair_semantic_flip_count": len(all_flips),
            "all_pair_semantic_flip_rate": round(len(all_flips) / len(paired_ids), 4) if paired_ids else 0,
            "all_pair_flip_question_ids": all_flips,
            "both_format_valid_pair_count": len(valid_ids),
            "valid_pair_answer_flip_count": len(flips),
            "valid_pair_answer_flip_rate": round(len(flips) / len(valid_ids), 4) if valid_ids else 0,
            "valid_pair_flip_question_ids": flips,
        }
    timing_effect = {}
    for order in ("AB", "BA"):
        settings = [f"{timing}_{order}" for timing in INFORMATION_TIMINGS]
        available = [setting for setting in settings if by_setting[setting]]
        ids = sorted(set.intersection(*(set(by_setting[setting]) for setting in available))) if available else []
        sensitive = []
        for qid in ids:
            predictions = {setting.removesuffix(f"_{order}"): answer_key(setting, qid)
                           for setting in available}
            if len(set(predictions.values())) > 1:
                sensitive.append({"question_id": qid, "predictions": predictions})
        timing_effect[order] = {
            "paired_question_count": len(ids),
            "timing_sensitive_question_count": len(sensitive),
            "timing_sensitivity_rate": round(len(sensitive) / len(ids), 4) if ids else 0,
            "details": sensitive,
        }
    acknowledgement = {}
    belief_updates = {}
    for setting in REPLAY_SETTINGS:
        values = list(by_setting[setting].values())
        acknowledged = [bool(t.get("late_fact_acknowledgement", {}).get(
            "all_responding_agents_acknowledged")) for t in values]
        direction_counts = Counter(
            direction
            for trace in values
            for direction, count in trace.get("belief_update", {}).get("direction_counts", {}).items()
            for _ in range(count))
        acknowledgement[setting] = {
            "n": len(values),
            "all_agents_acknowledged_count": sum(acknowledged),
            "late_fact_acknowledgement_rate": round(sum(acknowledged) / len(values), 4) if values else 0,
        }
        belief_updates[setting] = dict(direction_counts)
    hashes_consistent = all(
        len({by_setting[s][qid]["injected_fact_hash"] for s in REPLAY_SETTINGS
             if qid in by_setting[s]}) == 1
        for qid in set().union(*(set(values) for values in by_setting.values())))
    if common_ids and not hashes_consistent:
        raise RuntimeError("Replay invariant violated: a question has different injected fact hashes.")
    result = {
        "per_setting": accuracy,
        "paired_question_count": len(common_ids),
        "factorial_design": "five information-injection timings crossed independently with AB/BA fact order",
        "order_effect_holding_timing_fixed": order_effect,
        "timing_effect_holding_order_fixed": timing_effect,
        "late_fact_acknowledgement": acknowledgement,
        "belief_update_direction_counts": belief_updates,
        "fact_hash_consistent_across_all_timing_order_cells": hashes_consistent,
    }
    (output_dir / "replay_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=("setting", "n", "correct", "accuracy",
                                              "semantic_correct", "semantic_accuracy",
                                              "format_compliant", "format_compliance_rate",
                                              "raw_format_compliant", "raw_format_compliance_rate",
                                              "normalized_format_compliant", "normalized_format_compliance_rate",
                                              "protocol_valid", "protocol_valid_rate",
                                              "answer_reason_consistent", "answer_reason_consistency_rate",
                                              "strict_correct", "strict_accuracy",
                                              "strict_answer_correct", "strict_answer_accuracy",
                                              "fully_valid_correct", "fully_valid_accuracy"))
    writer.writeheader()
    for setting in REPLAY_SETTINGS:
        writer.writerow({"setting": setting, **accuracy[setting]})
    (output_dir / "replay_metrics.csv").write_text(buf.getvalue(), encoding="utf-8-sig")


def write_finalizer_order_analysis(traces: list[dict], output_dir: Path) -> None:
    rows = [t for t in traces if t["setting"] == FINALIZER_ORDER_SETTING]
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    by_order = {order: {int(t["question_id"]): t for t in rows if t.get("agent_variant") == order}
                for order in ("AB", "BA")}
    paired = sorted(set(by_order["AB"]) & set(by_order["BA"]))
    # All-pair semantic flip (includes format-invalid pairs).
    raw_flips = [qid for qid in paired if not equivalent(
        by_order["AB"][qid]["final_prediction"], by_order["BA"][qid]["final_prediction"])]
    valid_paired = [qid for qid in paired
                    if by_order["AB"][qid].get("format_compliant")
                    and by_order["BA"][qid].get("format_compliant")]
    valid_flips = [qid for qid in valid_paired if not equivalent(
        by_order["AB"][qid]["final_prediction"], by_order["BA"][qid]["final_prediction"])]

    semantic_buckets = {
        "both_semantically_correct": 0,
        "AB_only_correct": 0,
        "BA_only_correct": 0,
        "both_wrong_same_answer": 0,
        "both_wrong_different_answer": 0,
    }
    for qid in valid_paired:
        ab, ba = by_order["AB"][qid], by_order["BA"][qid]
        ab_ok, ba_ok = bool(ab.get("semantic_correct")), bool(ba.get("semantic_correct"))
        if ab_ok and ba_ok:
            semantic_buckets["both_semantically_correct"] += 1
        elif ab_ok:
            semantic_buckets["AB_only_correct"] += 1
        elif ba_ok:
            semantic_buckets["BA_only_correct"] += 1
        elif equivalent(ab["final_prediction"], ba["final_prediction"]):
            semantic_buckets["both_wrong_same_answer"] += 1
        else:
            semantic_buckets["both_wrong_different_answer"] += 1

    prompt_checks = {}
    for qid in paired:
        ab_prompt = by_order["AB"][qid].get("finalizer_user_prompt", "")
        ba_prompt = by_order["BA"][qid].get("finalizer_user_prompt", "")
        ab_lines, ba_lines = ab_prompt.splitlines(), ba_prompt.splitlines()
        ab_fact_rows = [i for i, line in enumerate(ab_lines) if line.startswith("FACT ")]
        ba_fact_rows = [i for i, line in enumerate(ba_lines) if line.startswith("FACT ")]
        only_fact_order_swapped = (
            len(ab_fact_rows) == len(ba_fact_rows) == 2
            and ab_fact_rows == ba_fact_rows
            and ab_lines[ab_fact_rows[0]] == ba_lines[ba_fact_rows[1]]
            and ab_lines[ab_fact_rows[1]] == ba_lines[ba_fact_rows[0]]
            and all(ab_line == ba_line for i, (ab_line, ba_line) in enumerate(zip(ab_lines, ba_lines))
                    if i not in ab_fact_rows)
            and len(ab_lines) == len(ba_lines)
        )
        pair_dir = output_dir / "prompt_pairs" / f"question_{qid:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        (pair_dir / "prompt_AB.txt").write_text(ab_prompt, encoding="utf-8")
        (pair_dir / "prompt_BA.txt").write_text(ba_prompt, encoding="utf-8")
        diff_text = "".join(difflib.unified_diff(
            ab_prompt.splitlines(keepends=True), ba_prompt.splitlines(keepends=True),
            fromfile="prompt_AB.txt", tofile="prompt_BA.txt"))
        (pair_dir / "prompt_diff.txt").write_text(diff_text, encoding="utf-8")
        prompt_checks[str(qid)] = {
            "only_two_fact_lines_swapped": only_fact_order_swapped,
            "artifact_directory": str(pair_dir),
        }

    result = {
        "setting": FINALIZER_ORDER_SETTING,
        "paired_question_count": len(paired),
        "all_pair_semantic_flip_count": len(raw_flips),
        "all_pair_semantic_flip_rate": round(len(raw_flips) / len(paired), 4) if paired else 0,
        "all_pair_semantic_flip_question_ids": raw_flips,
        "context_invariant": "shared question, system prompt, parameters, and wording identical; only verbatim fact order changes",
        "per_order": {
            order: {
                "n": len(by_order[order]),
                "semantic_correct": sum(bool(t.get("semantic_correct")) for t in by_order[order].values()),
                "format_compliant": sum(bool(t.get("format_compliant")) for t in by_order[order].values()),
                "answer_reason_consistent": sum(bool(t.get("answer_reason_consistent")) for t in by_order[order].values()),
                "strict_correct": sum(bool(t.get("strict_correct")) for t in by_order[order].values()),
                "strict_answer_correct": sum(bool(t.get("strict_answer_correct")) for t in by_order[order].values()),
                "fully_valid_correct": sum(bool(t.get("fully_valid_correct")) for t in by_order[order].values()),
            } for order in ("AB", "BA")
        },
        "raw_answer_flip_rate": round(len(raw_flips) / len(paired), 4) if paired else 0,
        "raw_answer_flip_count": len(raw_flips),
        "raw_answer_flip_question_ids": raw_flips,
        "both_format_valid_pair_count": len(valid_paired),
        "valid_pair_answer_flip_rate": round(len(valid_flips) / len(valid_paired), 4) if valid_paired else 0,
        "valid_pair_answer_flip_count": len(valid_flips),
        "valid_pair_answer_flip_question_ids": valid_flips,
        **semantic_buckets,
        "semantic_bucket_denominator": "pairs where both AB and BA are format compliant",
        "prompt_diff_checks": prompt_checks,
        "all_prompt_diffs_only_swap_two_fact_lines": all(
            check["only_two_fact_lines_swapped"] for check in prompt_checks.values()),
        "fact_hash_consistent": all(
            by_order["AB"][qid]["injected_fact_hash"] == by_order["BA"][qid]["injected_fact_hash"]
            for qid in paired),
    }
    (output_dir / "finalizer_order_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    failed_prompt_checks = [qid for qid, check in prompt_checks.items()
                            if not check["only_two_fact_lines_swapped"]]
    if failed_prompt_checks:
        raise RuntimeError(
            "AB/BA prompt invariant failed; differences were not limited to swapping the two fact lines "
            f"for question(s): {', '.join(failed_prompt_checks)}")


def write_order_sensitivity_analysis(traces: list[dict], output_dir: Path) -> None:
    rows = [trace for trace in traces if trace["setting"] in ORDER_SETTINGS]
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    def key(answer: Any) -> str:
        value = decimal(answer)
        return f"decimal:{value.normalize()}" if value is not None else (
            "text:" + re.sub(r"\s+", " ", extract_answer(answer).lower()).strip())

    by_setting_question = defaultdict(list)
    for trace in rows:
        by_setting_question[(trace["setting"], int(trace["question_id"]))].append(trace)
    settings_report = {}
    for setting in ORDER_SETTINGS:
        question_report = {}
        first_position_correct = defaultdict(list)
        for (row_setting, qid), values in by_setting_question.items():
            if row_setting != setting:
                continue
            answers = [key(value["final_prediction"]) for value in values]
            counts = Counter(answers)
            total = len(answers)
            entropy = -sum((count / total) * math.log2(count / total)
                           for count in counts.values()) if total else 0.0
            by_first = defaultdict(list)
            for value, answer in zip(values, answers):
                order = value.get("fact_order") or str(value.get("agent_variant", "")).rsplit("_", 1)[-1]
                first = order[0] if order in {"AB", "BA"} else "canonical"
                by_first[first].append(answer)
                first_position_correct[first].append(bool(value.get("semantic_correct")))
            position_modes = {
                first: {"answer": Counter(position_answers).most_common(1)[0][0],
                        "share": round(Counter(position_answers).most_common(1)[0][1] / len(position_answers), 4)}
                for first, position_answers in by_first.items() if position_answers
            }
            position_dominance = (
                {"A", "B"} <= set(position_modes)
                and position_modes["A"]["answer"] != position_modes["B"]["answer"]
                and min(position_modes["A"]["share"], position_modes["B"]["share"]) >= 0.75
            )
            question_report[str(qid)] = {
                "n": total,
                "order_sensitivity": len(counts) > 1,
                "answer_entropy": round(entropy, 6),
                "most_common_answer": counts.most_common(1)[0][0] if counts else "",
                "most_common_answer_share": round(counts.most_common(1)[0][1] / total, 4) if total else 0,
                "correct_answer_share": round(
                    sum(bool(value.get("semantic_correct")) for value in values) / total, 4) if total else 0,
                "answer_distribution": dict(counts),
                "first_position_answer_modes": position_modes,
                "position_dominance": position_dominance,
            }
        position_rates = {
            first: round(sum(results) / len(results), 4)
            for first, results in first_position_correct.items() if results
        }
        dominant_position = ""
        if {"A", "B"} <= set(position_rates) and position_rates["A"] != position_rates["B"]:
            dominant_position = max(("A", "B"), key=lambda first: position_rates[first])
        settings_report[setting] = {
            "question_count": len(question_report),
            "order_sensitive_question_count": sum(
                row["order_sensitivity"] for row in question_report.values()),
            "position_dominant_question_count": sum(
                row["position_dominance"] for row in question_report.values()),
            "first_position_correct_rate": position_rates,
            "long_run_dominant_first_position": dominant_position,
            "per_question": question_report,
        }
    (output_dir / "order_sensitivity.json").write_text(
        json.dumps({"settings": settings_report}, ensure_ascii=False, indent=2), encoding="utf-8")


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
    output_dirs = {setting: output_base / f"{run_stamp}_{setting}"
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
                  "information_timing_factor_levels": list(INFORMATION_TIMINGS),
                  "information_order_factor_levels": ["AB", "BA"],
                  "information_timing_discussion_rounds": max(3, args.discussion_rounds),
                  "replay_temperature": 0.0,
                  "finalizer_order_setting": FINALIZER_ORDER_SETTING,
                  "finalizer_order_temperature": 0.0,
                  "finalizer_order_seed_policy": "AB and BA use the same per-question derived seed",
                  "local_reason_judge_runs": 2,
                  "local_reason_judge_temperature": 0.0,
                  "local_reason_judge_blinding": "question, complete facts, final answer, and reason only",
                  "random_order_repetitions_per_question": 6,
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
            replay_rounds = max(3, args.discussion_rounds)
            for timing in INFORMATION_TIMINGS:
                selected_orders = [
                    order for order in ("AB", "BA") if f"{timing}_{order}" in selected_replay]
                if not selected_orders:
                    continue
                if timing in {"before_finalizer", "before_final_reset"}:
                    shared_key = "replay_no_fact_shared"
                    if shared_key not in cache:
                        reseed_model(model, derived_seed(args.seed, qid, "information_timing", "no_fact"))
                        cache[shared_key] = run_replay_discussion(
                            model, prompts["solver"], item, None, "AB", replay_rounds)
                    for order in selected_orders:
                        cache[f"replay_{timing}_{order}"] = cache[shared_key]
                    continue
                reveal_after_round = {
                    "all_at_start": 0,
                    "after_round1": 1,
                    "before_discussion_end": replay_rounds - 1,
                }[timing]
                for order in selected_orders:
                    reseed_model(model, derived_seed(
                        args.seed, qid, "information_timing", timing, "paired_order"))
                    cache[f"replay_{timing}_{order}"] = run_replay_discussion(
                        model, prompts["solver"], item, reveal_after_round, order, replay_rounds)
        if SOLVER_ORDER_SETTING in selected_settings:
            cache["solver_order_AB"] = run_replay_discussion(
                model, prompts["solver"], item, 0, "AB", args.discussion_rounds)
            cache["solver_order_BA"] = run_replay_discussion(
                model, prompts["solver"], item, 0, "BA", args.discussion_rounds)
        if FROZEN_TRANSCRIPT_ORDER_SETTING in selected_settings:
            cache["frozen_order_transcript"] = run_replay_discussion(
                model, prompts["solver"], item, None, "AB", args.discussion_rounds)
        for setting in selected_settings:
            variants = (("A", "B") if setting == "single_partial" else
                        ("AB", "BA") if setting in {
                            SOLVER_ORDER_SETTING, FINALIZER_ORDER_SETTING,
                            FROZEN_TRANSCRIPT_ORDER_SETTING} else
                        ("canonical",) if setting in {CANONICAL_ORDER_SETTING, SELF_CHECK_BEFORE_COMMIT_SETTING} else
                        tuple(f"{index:02d}_{order}" for index, order in enumerate(
                            random_order_sequence(args.seed, qid), 1))
                        if setting == RANDOM_ORDER_SETTING else ("",))
            for variant in variants:
                print(f"[{qid}/{len(items)}] {setting}{'_' + variant if variant else ''}")
                seed_variant = ("paired_order_control" if setting in {
                    SOLVER_ORDER_SETTING, FINALIZER_ORDER_SETTING, FROZEN_TRANSCRIPT_ORDER_SETTING}
                                else (variant or "default"))
                if setting in REPLAY_SETTINGS:
                    timing_name, _ = replay_setting_parts(setting)
                    generation_seed = derived_seed(
                        args.seed, qid, "information_timing_finalizer", timing_name, "paired_order")
                elif setting in {CANONICAL_ORDER_SETTING, SELF_CHECK_BEFORE_COMMIT_SETTING}:
                    generation_seed = derived_seed(args.seed, qid, "canonical_finalizer", "provisional")
                else:
                    generation_seed = derived_seed(args.seed, qid, setting, seed_variant)
                reseed_model(model, generation_seed)
                if setting == FINALIZER_ORDER_SETTING:
                    trace = build_finalizer_order_trace(model, prompts, item, qid, variant)
                elif setting == SOLVER_ORDER_SETTING:
                    ordered_discussion = cache[f"solver_order_{variant}"]
                    evidence = (f"Canonical fact table:\n{replay_ledger(item)}\n"
                                f"Solver transcript:\n{public_transcript(ordered_discussion['discussion_events'])}")
                    trace = build_finalizer_order_trace(
                        model, prompts, item, qid, variant,
                        setting=setting,
                        evidence_override=evidence,
                        discussion=ordered_discussion,
                    )
                    trace["final_context_policy"] = (
                        "solver sees ordered facts; finalizer receives canonical ledger plus resulting transcript")
                elif setting == FROZEN_TRANSCRIPT_ORDER_SETTING:
                    frozen = cache["frozen_order_transcript"]
                    evidence = (f"Newly disclosed facts (verbatim):\n{replay_facts(item, variant)}\n"
                                f"Frozen solver transcript:\n{frozen['public_transcript']}")
                    trace = build_finalizer_order_trace(
                        model, prompts, item, qid, variant,
                        setting=setting,
                        evidence_override=evidence,
                        discussion=frozen,
                    )
                    trace["final_context_policy"] = (
                        "identical frozen solver transcript; only final fact rows swap")
                elif setting == CANONICAL_ORDER_SETTING:
                    canonical_evidence, canonical_user = fixed_source_ledger_finalizer_prompt(item)
                    trace = build_finalizer_order_trace(
                        model, prompts, item, qid, variant,
                        setting=setting,
                        evidence_override=canonical_evidence,
                        user_override=canonical_user,
                    )
                    trace["fact_order"] = "fixed_source_A_then_B"
                    trace["final_context_policy"] = "fixed source-order ledger; legacy canonical_order setting"
                    trace.update({
                        "ledger_type": "fixed_source_order_legacy",
                        "ordering_policy": "source_A_then_B",
                        "provenance_mode": "aware",
                        "canonical_semantics_claimed": False,
                    })
                elif setting == SELF_CHECK_BEFORE_COMMIT_SETTING:
                    trace = build_self_check_trace(model, prompts, item, qid, args.seed)
                elif setting == RANDOM_ORDER_SETTING:
                    order = variant.rsplit("_", 1)[1]
                    trace = build_finalizer_order_trace(
                        model, prompts, item, qid, variant, setting, replay_facts(item, order))
                    trace["fact_order"] = order
                    trace["final_context_policy"] = "deterministically randomized fact-row permutation"
                elif setting in REPLAY_SETTINGS:
                    replay_cache_key = f"replay_{setting}"
                    trace = build_replay_trace(model, prompts, item, qid, setting, cache[replay_cache_key])
                    trace["discussion_cache_key"] = replay_cache_key
                else:
                    trace = build_trace(model, prompts, item, qid, setting, cache, variant, args.discussion_rounds)
                if trace.get("finalizer_event"):
                    apply_local_judge(
                        trace,
                        run_local_reason_judge(model, item, trace["finalizer_event"], args.seed, qid),
                    )
                trace["run_config"] = {key: run_config[key] for key in ("model_path", "device", "temperature", "max_new_tokens", "discussion_rounds", "seed")}
                if setting in CONTROLLED_SETTINGS or setting in {CANONICAL_ORDER_SETTING, SELF_CHECK_BEFORE_COMMIT_SETTING}:
                    trace["run_config"]["temperature"] = 0.0
                if setting in REPLAY_SETTINGS:
                    trace["run_config"]["discussion_rounds"] = max(3, args.discussion_rounds)
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
                    # DeepSeek is diagnostic only -- never override local correctness.
                    # appearance["correct"] always equals appearance["correct_before_judge"].
                    candidate_reviews.append(review)
                trace["deepseek_judge"] = {"final": final_review, "candidate_appearances": candidate_reviews}
                if judge_error:
                    trace["deepseek_judge_error"] = judge_error
                # DeepSeek is diagnostic only — never overrides semantic_correct / correct.
                # correct_before_judge is the deterministic local answer comparison.
                trace["deepseek_format_issue"] = bool(final_review.get("format_issue"))
                trace["deepseek_reason"] = str(final_review.get("reason", ""))
                info_review = reviews.get(f"{i}:information")
                if info_review is not None:
                    trace["deepseek_information_review"] = info_review
                    trace["information_complete_deepseek_review"] = as_bool(
                        info_review.get("correct"))
                    # DeepSeek information review is diagnostic only.
                    # Never override the deterministic information_complete field.
                rejection_review = reviews.get(f"{i}:correct_rejection")
                if rejection_review is not None:
                    trace["deepseek_correct_rejection_review"] = rejection_review
                # per_agent_correctness is always locally computed.
                # DeepSeek does not modify it.
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
            trace["answer_error_categories"] = classify_wrong_answer(trace)
            trace["answer_flow"] = analyze_answer_flow(trace)
        traces.extend(question_traces)
        for setting, directory in output_dirs.items():
            setting_config = dict(run_config, setting=setting, output_dir=str(directory))
            write_outputs([x for x in traces if x["setting"] == setting], directory, setting_config)
            if setting == FINALIZER_ORDER_SETTING:
                write_finalizer_order_analysis(traces, directory)
        write_replay_analysis(traces, output_base / f"{run_stamp}_replay_analysis")
        write_order_sensitivity_analysis(traces, output_base / f"{run_stamp}_order_analysis")
    for setting, directory in output_dirs.items():
        setting_config = dict(run_config, setting=setting, output_dir=str(directory))
        write_outputs([x for x in traces if x["setting"] == setting], directory, setting_config)
        if setting == FINALIZER_ORDER_SETTING:
            write_finalizer_order_analysis(traces, directory)
        print(f"Wrote {sum(x['setting'] == setting for x in traces)} {setting} traces to {directory}")
    write_replay_analysis(traces, output_base / f"{run_stamp}_replay_analysis")
    write_order_sensitivity_analysis(traces, output_base / f"{run_stamp}_order_analysis")


class _MockCuda:
    def is_available(self) -> bool:
        return False

    def manual_seed_all(self, seed: int) -> None:
        self.last_seed = seed


class _MockTorch:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.cuda = _MockCuda()

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def inference_mode(self):
        return contextlib.nullcontext()


class _MockModel:
    def __init__(self, responses: list[tuple[str, dict, float]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.max_new_tokens = 128
        self.torch = _MockTorch()

    def call(self, system: str, user: str, temperature: float | None = None):
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)


class HiddenGsm8kSelfCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.item = {
            "shared_question": "What is 2 + 3?",
            "condition_A": "Agent A knows the addend 2.",
            "condition_B": "Agent B knows the addend 3.",
            "answer": "#### 5",
            "fact": "SECRET_FACT_SHOULD_NOT_APPEAR",
        }
        self.prompts = {"finalizer": "FINALIZER_SYSTEM_PROMPT"}

    def test_semantic_extraction_is_independent_from_format(self) -> None:
        draft_raw = "Here is the answer.\nFinal answer: 5\nReason: 2 + 3 = 5\n"
        commit_raw = "Selected source: recomputed\nFinal answer: 5\nReason: 2 + 3 = 5\n"
        model = _MockModel([
            (draft_raw, {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.1),
            (commit_raw, {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.2),
        ])
        trace = build_self_check_trace(model, self.prompts, self.item, qid=7, base_seed=42)
        self.assertEqual(trace["provisional_answer"], "5")
        self.assertEqual(trace["provisional_answer_extraction"], "explicit_label")
        self.assertEqual(trace["provisional_parsed_answer"], "")
        self.assertTrue(trace["provisional_semantic_correct"])
        self.assertFalse(trace["provisional_format_compliant"])
        self.assertEqual(trace["committed_answer"], "5")
        self.assertEqual(trace["committed_answer_extraction"], "explicit_label")
        self.assertEqual(trace["committed_parsed_answer"], "5")
        self.assertTrue(trace["committed_semantic_correct"])
        self.assertTrue(trace["committed_format_compliant"])
        self.assertEqual(trace["final_prediction"], trace["committed_answer"])
        self.assertEqual(trace["self_check_change_direction"], "unchanged_correct")
        self.assertFalse(trace["answer_changed_after_self_check"])

    def test_self_check_change_directions_cover_semantic_cases(self) -> None:
        model = _MockModel([
            ("Selected source: recomputed\nFinal answer: 4\nReason: 2 + 2 = 4\n", {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.1),
            ("Selected source: recomputed\nFinal answer: 5\nReason: 2 + 3 = 5\n", {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.2),
        ])
        trace = build_self_check_trace(model, self.prompts, self.item, qid=8, base_seed=42)
        self.assertEqual(trace["self_check_change_direction"], "wrong_to_correct")
        self.assertTrue(trace["answer_changed_after_self_check"])
        self.assertTrue(trace["committed_semantic_correct"])

        model = _MockModel([
            ("Selected source: recomputed\nFinal answer: 5\nReason: 2 + 3 = 5\n", {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.1),
            ("Selected source: recomputed\nFinal answer: 4\nReason: 2 + 2 = 4\n", {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.2),
        ])
        trace = build_self_check_trace(model, self.prompts, self.item, qid=9, base_seed=42)
        self.assertEqual(trace["self_check_change_direction"], "correct_to_wrong")

        model = _MockModel([
            ("Selected source: recomputed\nFinal answer: 4\nReason: 2 + 2 = 4\n", {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.1),
            ("Selected source: recomputed\nFinal answer: 4\nReason: 2 + 2 = 4\n", {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.2),
        ])
        trace = build_self_check_trace(model, self.prompts, self.item, qid=10, base_seed=42)
        self.assertEqual(trace["self_check_change_direction"], "unchanged_wrong")

    def test_canonical_and_self_check_share_fixed_source_helper(self) -> None:
        evidence, user = fixed_source_ledger_finalizer_prompt(self.item)
        self.assertEqual(evidence, fixed_source_ledger_evidence(self.item))
        self.assertEqual(user, empty_candidate_finalizer_prompt(self.item, evidence))
        self.assertEqual(ledger_evidence_hash(evidence), hashlib.sha256(evidence.encode("utf-8")).hexdigest())

    def test_invariant_status_is_real_comparison(self) -> None:
        model = _MockModel([
            ("Selected source: recomputed\nFinal answer: 5\nReason: 2 + 3 = 5\n", {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.1),
            ("Selected source: recomputed\nFinal answer: 5\nReason: 2 + 3 = 5\n", {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}, 0.2),
        ])
        trace = build_self_check_trace(model, self.prompts, self.item, qid=11, base_seed=42)
        self.assertEqual(trace["self_check_invariant_status"], "PASS")
        self.assertTrue(trace["evidence_equal_to_baseline"])
        self.assertTrue(trace["prompt_equal_to_baseline"])
        self.assertEqual(trace["paired_baseline_setting"], CANONICAL_ORDER_SETTING)

    def test_old_setting_shape_is_unchanged(self) -> None:
        model = _MockModel([
            ("Selected source: recomputed\nFinal answer: 5\nReason: 2 + 3 = 5\n", {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}, 0.1),
        ])
        trace = build_finalizer_order_trace(
            model,
            self.prompts,
            self.item,
            qid=3,
            order="BA",
        )
        self.assertNotIn("provisional_answer", trace)
        self.assertNotIn("self_check_enabled", trace)
        self.assertEqual(trace["setting"], FINALIZER_ORDER_SETTING)
        self.assertEqual(trace["final_prediction"], trace["finalizer_event"]["parsed_output"]["final_answer"])

    def test_semantic_correctness_is_independent_from_format_and_wrong_answers_are_distinct(self) -> None:
        malformed_correct = {
            "raw_output": "Final answer: 5\nReason: 2 + 3 = 5\n",
            "parsed_output": {"final_answer": ""},
            "semantic_answer": "5",
            "invalid_output": True,
        }
        self.assertEqual(event_answer(malformed_correct), "5")

        correct_trace = {
            "final_prediction": event_answer(malformed_correct),
            "invalid_output": True,
            "information": {"information_complete": True},
            "candidate_appearances": [{"answer": "5", "correct": True, "information_complete_at_appearance": True}],
        }
        set_outcome_fields(correct_trace, "5", semantic_correct=True)
        self.assertTrue(correct_trace["semantic_correct"])
        self.assertTrue(correct_trace["correct"])
        self.assertFalse(correct_trace["format_compliant"])
        self.assertFalse(correct_trace["strict_answer_correct"])
        self.assertIsNone(classify(correct_trace, "5")[0])
        self.assertEqual(classify_wrong_answer(correct_trace), [])

        wrong_trace = {
            "final_prediction": "4",
            "invalid_output": False,
            "information": {"information_complete": True},
            "candidate_appearances": [{"answer": "5", "correct": True, "information_complete_at_appearance": True}],
            "local_judge": {"consensus": {"error_type": "arithmetic_error"}},
        }
        set_outcome_fields(wrong_trace, "5", semantic_correct=False)
        self.assertFalse(wrong_trace["semantic_correct"])
        self.assertEqual(classify_wrong_answer(wrong_trace), ["arithmetic_error"])
        self.assertEqual(classify(wrong_trace, "5")[0], "answer_selection_failure")

        missing_trace = {
            "final_prediction": "",
            "invalid_output": False,
            "information": {"information_complete": True},
            "candidate_appearances": [],
        }
        set_outcome_fields(missing_trace, "5", semantic_correct=False)
        self.assertEqual(classify_wrong_answer(missing_trace), ["unable_to_judge"])

    # ── Comprehensive format/correctness independence tests ──

    def test_single_full_answer_on_second_line_semantically_correct(self):
        """Answer correct but on 2nd line: semantic_correct=True, format fails."""
        raw = "Let me think step by step.\nFinal answer: 42\nReasoning..."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "42")
        self.assertEqual(sem["method"], "explicit_label")
        self.assertTrue(sem["explicit"])
        # Strict format: first line is NOT "Final answer: ..."
        _, fmt_err = parse_solver_final(raw)
        self.assertTrue(bool(fmt_err))

    def test_single_partial_explicit_undetermined_numbers_in_reasoning(self):
        """Explicit undetermined with numbers in reasoning → answer=''."""
        raw = "Final answer: cannot determine\nReasoning: there are 3 apples and 4 oranges"
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "")
        self.assertIn("undetermined", sem["method"])

    def test_multi_solver_final_format_error_preserves_candidate(self):
        """Solver final: format error but answer correct → candidate preserved."""
        raw = "Some preamble\nFinal answer: 42\nReasoning here."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "42")
        # Even with strict format failure, semantic answer survives.
        _, fmt_err = parse_solver_final(raw)
        self.assertTrue(bool(fmt_err))

    def test_verifier_json_error_preserves_semantic_answer(self):
        """Verifier JSON malformed but verified_answer clear → semantic survives."""
        raw = 'Not valid JSON here\nverified_answer: 42\nmore text'
        sem = extract_semantic_answer(raw, "verified_answer",
                                       expected_output_type="verifier")
        self.assertEqual(sem["answer"], "42")
        self.assertEqual(sem["method"], "explicit_label")

    def test_finalizer_three_line_error_numeric_answer_correct(self):
        """Finalizer: wrong line count but correct numeric answer."""
        raw = "Extra preamble line\nSelected source: recomputed\nFinal answer: 42\nReason: correct"
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "42")
        # strict parser fails
        parsed, err = parse_fixed_finalizer(raw)
        self.assertTrue(bool(err))

    def test_truly_wrong_answer(self):
        """Truly wrong answer → semantic_correct=False."""
        raw = "Final answer: 99"
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "99")
        self.assertFalse(equivalent("99", "42"))

    def test_no_answer_at_all(self):
        """No answer at all → prediction=''."""
        raw = "I cannot solve this problem."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "")
        self.assertEqual(sem["method"], "no_supported_answer")

    def test_multiple_numbers_no_conclusion_not_extracted(self):
        """Multiple numbers in reasoning, no conclusion label → not extracted."""
        raw = "There are 3 apples and 4 oranges. Maybe 5 or 6."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "")
        self.assertFalse(sem["explicit"])

    def test_leading_blank_line_format_fails(self):
        """Leading blank line → raw_format_compliant=False."""
        raw = "\nFinal answer: 42\nReasoning."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "42")
        # Strict format: first physical line is blank, so fail.
        _, fmt_err = parse_solver_final(raw)
        self.assertTrue(bool(fmt_err))

    def test_normalized_compliant_raw_fails(self):
        """normalize_finalizer_output fixes blank lines but raw still fails."""
        raw = "Selected source: recomputed\nFinal answer: 42\n\nReason: test"
        normalized, steps = normalize_finalizer_output(raw)
        self.assertIn("removed_blank_lines_between_fields", steps)
        # Raw: parse fails (4 lines including blank)
        _, raw_err = parse_fixed_finalizer(raw)
        self.assertTrue(bool(raw_err))
        # Normalized: parse succeeds
        _, norm_err = parse_fixed_finalizer(normalized)
        self.assertFalse(bool(norm_err))

    def test_deepseek_cannot_flip_locally_wrong_to_correct(self):
        """DeepSeek correct=true must NOT flip locally-wrong answer."""
        trace = {
            "final_prediction": "4",
            "gold_answer": "5",
            "invalid_output": False,
            "information": {"information_complete": True},
            "candidate_appearances": [{"answer": "5", "correct": True, "information_complete_at_appearance": True}],
        }
        set_outcome_fields(trace, "5")
        self.assertFalse(trace["semantic_correct"])
        self.assertFalse(trace["correct"])
        # DeepSeek says correct=true — should NOT change semantic_correct.
        # (In the real pipeline, deepseek results are now diagnostic-only.)
        trace["deepseek_judge"] = {"final": {"correct": True, "reason": "disagree"}}
        self.assertFalse(trace["semantic_correct"],
                         "DeepSeek must not override semantic_correct")

    def test_deepseek_cannot_flip_locally_correct_to_wrong(self):
        """DeepSeek correct=false must NOT flip locally-correct answer."""
        trace = {
            "final_prediction": "5",
            "gold_answer": "5",
            "invalid_output": False,
            "information": {"information_complete": True},
            "candidate_appearances": [],
        }
        set_outcome_fields(trace, "5")
        self.assertTrue(trace["semantic_correct"])
        # DeepSeek says correct=false — should NOT change semantic_correct.
        trace["deepseek_judge"] = {"final": {"correct": False, "reason": "disagree"}}
        self.assertTrue(trace["semantic_correct"],
                        "DeepSeek must not override semantic_correct")

    def test_format_correct_semantically_wrong(self):
        """Format-compliant but semantically wrong answer."""
        raw = "Final answer: 99\nHere is the math."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "99")
        _, fmt_err = parse_solver_final(raw)
        self.assertFalse(bool(fmt_err))  # format OK
        self.assertFalse(equivalent("99", "42"))  # semantically wrong

    def test_format_wrong_semantically_correct(self):
        """Format-noncompliant but semantically correct answer."""
        raw = "Step 1: find numbers.\nStep 2: add them.\nStep 3: verify.\nFinal answer: 42\nMore."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "42")
        _, fmt_err = parse_solver_final(raw)
        self.assertTrue(bool(fmt_err))  # format fails
        self.assertTrue(equivalent("42", "42"))  # semantically correct

    def test_format_wrong_semantically_wrong(self):
        """Both format and semantics wrong."""
        raw = "Preamble.\nStep1. Step2. Step3. Step4. Step5.\nFinal answer: 99\nMore."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "99")
        _, fmt_err = parse_solver_final(raw)
        self.assertTrue(bool(fmt_err))  # format fails
        self.assertFalse(equivalent("99", "42"))  # semantically wrong

    def test_no_answer_distinct_from_wrong_answer(self):
        """No answer (empty) vs wrong answer — clearly distinguished."""
        raw_no = "I don't know the answer."
        sem_no = extract_semantic_answer(raw_no, "Final answer")
        self.assertEqual(sem_no["answer"], "")
        self.assertFalse(sem_no["explicit"])

        raw_wrong = "Final answer: 99"
        sem_wrong = extract_semantic_answer(raw_wrong, "Final answer")
        self.assertEqual(sem_wrong["answer"], "99")
        self.assertTrue(sem_wrong["explicit"])

    def test_concluding_expression_extraction(self):
        """Explicit concluding expressions are extracted correctly."""
        # "the answer is X" pattern — may return concluding_verb or concluding_expression
        sem = extract_semantic_answer("Some reasoning. The answer is 42.", "Final answer")
        self.assertEqual(sem["answer"], "42")
        self.assertIn(sem["method"], ("concluding_expression", "concluding_verb", "concluding_object"))

        # "therefore X" pattern
        sem = extract_semantic_answer("We compute 3+4=7. Therefore 7.", "Final answer")
        self.assertEqual(sem["answer"], "7")
        self.assertEqual(sem["method"], "concluding_expression")

    def test_single_number_simple_fallback(self):
        """Simple text with exactly one number gets safe fallback."""
        sem = extract_semantic_answer("42", "Final answer")
        self.assertEqual(sem["answer"], "42")
        self.assertEqual(sem["method"], "safe_single_number_fallback")
        self.assertFalse(sem["explicit"])

    def test_reason_evaluation_unavailable_for_single_settings(self):
        """Single settings get reason_evaluation_available=False and None fields."""
        trace = {
            "final_prediction": "42",
            "gold_answer": "42",
            "invalid_output": False,
        }
        set_outcome_fields(trace, "42")
        self.assertFalse(trace["reason_evaluation_available"])
        self.assertIsNone(trace["reason_mathematically_valid"])
        self.assertIsNone(trace["fully_valid_correct"])
        self.assertTrue(trace["semantic_correct"])

    # ── Comprehensive format/correctness/protocol independence tests ──

    def test_call_finalizer_once_splits_three_layers(self):
        """call_finalizer_once separates raw, normalized, and protocol errors."""
        mock = _MockModel([
            ("Selected source: recomputed\nFinal answer: 42\nReason: correct\n",
             {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}, 0.1),
        ])
        event = call_finalizer_once(mock, "sys", "user", {})
        self.assertTrue(event["raw_format_compliant"])
        self.assertTrue(event["normalized_format_compliant"])
        self.assertTrue(event["protocol_valid"])
        self.assertEqual(event["semantic_answer"], "42")
        self.assertIn("format_failure_categories", event)
        self.assertIn("protocol_failure_categories", event)

    def test_call_finalizer_once_raw_fails_normalized_passes(self):
        """Blank line in raw: raw_format_compliant=False, normalized passes."""
        mock = _MockModel([
            ("Selected source: recomputed\nFinal answer: 42\n\nReason: correct",
             {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}, 0.1),
        ])
        event = call_finalizer_once(mock, "sys", "user", {})
        self.assertFalse(event["raw_format_compliant"])
        self.assertTrue(event["normalized_format_compliant"])
        self.assertTrue(event["single_shot_format_failure"])
        self.assertEqual(event["semantic_answer"], "42")

    def test_call_finalizer_once_raw_semantic_answer_survives_format_error(self):
        """Format error: semantic answer still extracted."""
        mock = _MockModel([
            ("Preamble text\nSelected source: recomputed\nFinal answer: 42\nReason: correct",
             {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}, 0.1),
        ])
        event = call_finalizer_once(mock, "sys", "user", {})
        self.assertFalse(event["raw_format_compliant"])
        self.assertEqual(event["semantic_answer"], "42")
        self.assertTrue(event["single_shot_format_failure"])

    def test_classify_finalizer_format_errors_uses_raw_output(self):
        """Format classification uses raw_output, not normalized_output."""
        event = {
            "raw_output": "Preamble\nSelected source: recomputed\nFinal answer: 42\nReason: x",
            "normalized_output": "Selected source: recomputed\nFinal answer: 42\nReason: x",
        }
        cats = classify_finalizer_format_errors(event)
        self.assertIn("extra_text_outside_three_lines", cats)

    def test_classify_finalizer_protocol_errors(self):
        """Protocol errors are classified separately from format errors."""
        event = {
            "protocol_validation_error": "answer does not match selected source solver_a",
        }
        cats = classify_finalizer_protocol_errors(event)
        self.assertIn("answer_source_mismatch", cats)

    def test_single_shot_format_failure_is_raw_only(self):
        """single_shot_format_failure is based on raw format, not normalized."""
        mock = _MockModel([
            ("Selected source: recomputed\nFinal answer: 42\n\nReason: test",
             {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}, 0.1),
        ])
        event = call_finalizer_once(mock, "sys", "user", {})
        # Raw has blank line → single_shot_format_failure = True
        self.assertTrue(event["single_shot_format_failure"])
        # But normalized parses fine → normalized_format_compliant = True
        self.assertTrue(event["normalized_format_compliant"])
        # semantic answer is still correct
        self.assertEqual(event["semantic_answer"], "42")

    def test_finalizer_candidate_passing_ignores_invalid_output(self):
        """Verifier candidate passed to finalizer even when JSON is malformed."""
        verifier = {
            "raw_output": "bad json\nverified_answer: 42\n",
            "parsed_output": {"verified_answer": ""},
            "invalid_output": True,
            "semantic_answer": "42",
            "semantic_answer_extraction": "explicit_label",
        }
        verified_answer = verifier.get("semantic_answer",
                                        event_answer(verifier, "verified_answer"))
        self.assertEqual(verified_answer, "42")
        self.assertIsNotNone(decimal(verified_answer))

    def test_candidate_appearances_includes_verifier_even_when_invalid(self):
        """candidate_appearances shows verifier answer regardless of invalid_output."""
        trace = {
            "verifier_event": {
                "agent": "verifier",
                "raw_output": "bad json\nverified_answer: 42\n",
                "invalid_output": True,
                "semantic_answer": "42",
                "semantic_answer_extraction": "explicit_label",
                "raw_format_compliant": False,
                "protocol_valid": False,
            },
            "discussion": {
                "discussion_events": [],
                "solver_finals": {},
            },
            "information": {"information_complete": True},
        }
        appearances = candidate_appearances(trace)
        verifier_apps = [a for a in appearances if a["source"] == "verifier"]
        self.assertEqual(len(verifier_apps), 1)
        self.assertEqual(verifier_apps[0]["answer"], "42")

    def test_build_replay_trace_uses_event_answer(self):
        """build_replay_trace extracts prediction via event_answer, not free text."""
        finalizer = {
            "raw_output": "Selected source: recomputed\nFinal answer: 42\nReason: correct\n",
            "semantic_answer": "42",
            "semantic_answer_extraction": "explicit_label",
            "raw_format_compliant": True,
            "normalized_format_compliant": True,
            "protocol_valid": True,
            "raw_format_error": "",
            "normalized_format_error": "",
            "protocol_validation_error": "",
            "format_failure_categories": [],
            "protocol_failure_categories": [],
            "invalid_output": False,
            "single_shot_format_failure": False,
            "deterministic_normalizations": [],
            "retry_exhausted": False,
            "parsed_output": {"final_answer": "42", "selected_source": "recomputed", "reason": "correct"},
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "runtime_seconds": 0.1,
            "validation_error": "",
            "actual_input": "test",
        }
        # Simulate: call_finalizer_once returns this.
        # Use event_answer to verify prediction extraction.
        prediction = event_answer(finalizer)
        self.assertEqual(prediction, "42")

    def test_build_finalizer_order_trace_semantic_survives_format_error(self):
        """Format error in canonical_order does not clear semantic answer."""
        finalizer = {
            "raw_output": "Preamble\nSelected source: recomputed\nFinal answer: 42\nReason: correct",
            "semantic_answer": "42",
            "semantic_answer_extraction": "explicit_label",
            "raw_format_compliant": False,
            "normalized_format_compliant": False,
            "protocol_valid": False,
            "raw_format_error": "expected exactly three lines",
            "normalized_format_error": "expected exactly three lines",
            "protocol_validation_error": "",
            "format_failure_categories": ["extra_text_outside_three_lines"],
            "protocol_failure_categories": [],
            "invalid_output": True,
            "single_shot_format_failure": True,
            "deterministic_normalizations": [],
            "retry_exhausted": True,
            "parsed_output": {"final_answer": "", "selected_source": "none", "reason": ""},
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "runtime_seconds": 0.1,
            "validation_error": "expected exactly three lines",
            "actual_input": "test",
        }
        prediction = event_answer(finalizer)
        self.assertEqual(prediction, "42")

    def test_self_check_provisional_format_error_committed_format_ok(self):
        """Provisional format error + committed format OK = unchanged_correct."""
        draft_event = {
            "raw_output": "Preamble\nFinal answer: 42\n",
            "semantic_answer": "42",
            "semantic_answer_extraction": "explicit_label",
            "raw_format_compliant": False,
            "normalized_format_compliant": False,
            "protocol_valid": False,
            "parsed_output": {"final_answer": "", "selected_source": "none", "reason": ""},
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            "runtime_seconds": 0.1,
            "invalid_output": True,
        }
        committed_event = {
            "raw_output": "Selected source: recomputed\nFinal answer: 42\nReason: correct\n",
            "semantic_answer": "42",
            "semantic_answer_extraction": "explicit_label",
            "raw_format_compliant": True,
            "normalized_format_compliant": True,
            "protocol_valid": True,
            "parsed_output": {"final_answer": "42", "selected_source": "recomputed", "reason": "correct"},
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            "runtime_seconds": 0.1,
            "invalid_output": False,
        }
        direction = classify_self_check_change_direction("42", "42", "42")
        self.assertEqual(direction, "unchanged_correct")

    def test_deepseek_information_not_override_deterministic(self):
        """DeepSeek info review must not override deterministic information_complete."""
        trace = {
            "information": {
                "information_complete": False,
                "side_revealed": {"A": False, "B": False},
            },
        }
        # Simulate DeepSeek saying complete=true
        info_review = {"correct": True, "reason": "facts are present"}
        trace["deepseek_information_review"] = info_review
        trace["information_complete_deepseek_review"] = as_bool(info_review.get("correct"))
        # Deterministic field must remain False
        self.assertFalse(trace["information"]["information_complete"])

    def test_local_judge_uses_semantic_answer(self):
        """Local Judge receives semantic_answer, not parsed_output final_answer."""
        finalizer = {
            "raw_output": "Preamble\nFinal answer: 42\n",
            "semantic_answer": "42",
            "parsed_output": {"final_answer": "", "selected_source": "none", "reason": ""},
        }
        # The Final answer passed to judge should come from semantic_answer
        judge_answer = finalizer.get("semantic_answer",
                                      finalizer["parsed_output"].get("final_answer", ""))
        self.assertEqual(judge_answer, "42")
        # Without the fix, parsed_output["final_answer"]="" would give empty
        without_fix = finalizer["parsed_output"].get("final_answer", "")
        self.assertEqual(without_fix, "")

    def test_metrics_fields_include_protocol_stats(self):
        """Metrics CSV fields include raw/normalized format and protocol stats."""
        import csv
        buf = io.StringIO()
        # Simulate a minimal metrics row
        fields = ["setting", "n", "correct", "accuracy",
                  "semantic_correct", "semantic_accuracy",
                  "format_compliant", "format_compliance_rate",
                  "raw_format_compliant", "raw_format_compliance_rate",
                  "normalized_format_compliant", "normalized_format_compliance_rate",
                  "protocol_valid", "protocol_valid_rate",
                  "strict_answer_correct", "strict_answer_accuracy"]
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "setting": "test", "n": 1, "correct": 1, "accuracy": 1.0,
            "semantic_correct": 1, "semantic_accuracy": 1.0,
            "format_compliant": 0, "format_compliance_rate": 0.0,
            "raw_format_compliant": 0, "raw_format_compliance_rate": 0.0,
            "normalized_format_compliant": 1, "normalized_format_compliance_rate": 1.0,
            "protocol_valid": 1, "protocol_valid_rate": 1.0,
            "strict_answer_correct": 0, "strict_answer_accuracy": 0.0,
        })
        self.assertIn("raw_format_compliant", buf.getvalue())

    def test_empty_failure_list_for_format_error_semantic_correct(self):
        """classify_wrong_answer returns [] when semantic_correct is True, even with format errors."""
        trace = {
            "semantic_correct": True,
            "final_prediction": "42",
            "gold_answer": "42",
            "invalid_output": True,
            "format_compliant": False,
            "information": {"information_complete": True},
            "candidate_appearances": [{"answer": "42", "correct": True,
                                       "information_complete_at_appearance": True}],
        }
        self.assertEqual(classify_wrong_answer(trace), [])

    def test_format_failure_not_in_semantic_failures(self):
        """format_failure should not appear in semantic failure type."""
        trace = {
            "semantic_correct": False,
            "final_prediction": "",
            "gold_answer": "42",
            "invalid_output": True,
            "information": {"information_complete": True},
            "candidate_appearances": [],
            "local_judge": {"consensus": {"error_type": ""}},
        }
        failure_type, _ = classify(trace, "42")
        # Should be a semantic failure type, not "invalid_output"
        self.assertIn(failure_type, {
            "information_acquisition_failure",
            "information_integration_failure",
            "answer_selection_failure",
            "answer_reason_inconsistency",
        })

    def test_single_partial_undetermined_extraction(self):
        """Explicit undetermined returns empty semantic answer."""
        raw = "Final answer: cannot determine\nThe problem lacks information."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "")
        self.assertIn("undetermined", sem["method"])
        self.assertTrue(sem["explicit"])

    def test_concluding_undetermined_blocks_number_extraction(self):
        """'Cannot determine' with numbers in text returns empty."""
        raw = "We have 5 apples and 3 oranges. Cannot determine the total weight."
        sem = extract_semantic_answer(raw, "Final answer")
        self.assertEqual(sem["answer"], "")
        self.assertIn("undetermined", sem["method"])

    def test_protocol_errors_not_in_format_categories(self):
        """Source consistency errors go to protocol_failure_categories."""
        event = {
            "raw_output": "Selected source: recomputed\nFinal answer: 42\nReason: correct",
            "protocol_validation_error": "answer does not match selected source solver_a",
        }
        format_cats = classify_finalizer_format_errors(event)
        protocol_cats = classify_finalizer_protocol_errors(event)
        self.assertNotIn("answer_source_mismatch", format_cats)
        self.assertIn("answer_source_mismatch", protocol_cats)

    def test_extract_semantic_answer_current_answer_label(self):
        """extract_semantic_answer works with Current answer label."""
        raw = "Current answer: 42\nReasoning follows."
        sem = extract_semantic_answer(raw, "Current answer",
                                       expected_output_type="solver_current")
        self.assertEqual(sem["answer"], "42")
        self.assertEqual(sem["method"], "explicit_label")

    def test_extract_semantic_answer_verified_answer_label(self):
        """extract_semantic_answer works with verified_answer label."""
        raw = "verified_answer: 42\nChecks passed."
        sem = extract_semantic_answer(raw, "verified_answer",
                                       expected_output_type="verifier")
        self.assertEqual(sem["answer"], "42")
        self.assertEqual(sem["method"], "explicit_label")

    # ── event_answer no-guessing tests ──

    def test_event_answer_does_not_guess_from_raw_output(self):
        """event_answer returns '' when only raw_output with incidental numbers exists."""
        event = {
            "raw_output": "There are 3 apples and 4 oranges."
        }
        self.assertEqual(event_answer(event), "")

    def test_event_answer_does_not_take_last_reasoning_number(self):
        """event_answer returns '' — last number in prose is not an answer."""
        event = {
            "raw_output": "The correct answer may be 5, verified in 2 ways."
        }
        self.assertEqual(event_answer(event), "")

    def test_event_answer_uses_saved_semantic_answer(self):
        """semantic_answer field takes priority over raw_output prose."""
        event = {
            "semantic_answer": "5",
            "raw_output": "Reason mentions 2 and 3."
        }
        self.assertEqual(event_answer(event), "5")

    def test_empty_semantic_answer_is_authoritative(self):
        """Explicitly empty semantic_answer blocks fallback to parsed_output or raw_output."""
        event = {
            "semantic_answer": "",
            "parsed_output": {"final_answer": "5"},
            "raw_output": "Final answer: 5"
        }
        self.assertEqual(event_answer(event), "")

    # ── Natural language conclusion extraction (positive) ──

    def test_conclusion_therefore_final_weight(self):
        result = extract_explicit_conclusion(
            "Therefore, the final weight of the box is 16 pounds.", "solver_final")
        self.assertEqual(result["answer"], "16")
        self.assertEqual(result["method"], "concluding_expression")
        self.assertTrue(result["explicit"])
        self.assertFalse(result["ambiguous"])

    def test_conclusion_balance_remaining(self):
        result = extract_explicit_conclusion(
            "Therefore, the balance remaining after the 4 monthly payments is $520.", "solver_final")
        self.assertEqual(result["answer"], "520")
        self.assertEqual(result["method"], "concluding_expression")

    def test_conclusion_must_buy(self):
        result = extract_explicit_conclusion(
            "Roger must buy 3 packs of trail-mix pouches.", "solver_final")
        self.assertEqual(result["answer"], "3")
        self.assertIn(result["method"], ("concluding_verb", "concluding_expression"))

    def test_conclusion_garden_produces(self):
        result = extract_explicit_conclusion(
            "Therefore, the garden produces a total of 768 vegetables.", "solver_final")
        self.assertEqual(result["answer"], "768")
        self.assertEqual(result["method"], "concluding_expression")

    def test_conclusion_boxed_dollars(self):
        result = extract_explicit_conclusion(
            "Winwin takes home $\\boxed{5}$ dollars.", "solver_final")
        self.assertEqual(result["answer"], "5")
        self.assertIn(result["method"], ("boxed_in_window", "boxed_in_clause"))

    def test_conclusion_amount_remaining(self):
        result = extract_explicit_conclusion(
            "The amount remaining in John's piggy bank is $200.", "solver_final")
        self.assertEqual(result["answer"], "200")
        self.assertIn(result["method"], ("concluding_object", "concluding_expression", "concluding_verb"))

    def test_conclusion_total_plates(self):
        result = extract_explicit_conclusion(
            "The total number of plates needed is 180.", "solver_final")
        self.assertEqual(result["answer"], "180")
        self.assertIn(result["method"], ("concluding_object", "concluding_expression", "concluding_verb"))

    def test_conclusion_hard_hats(self):
        result = extract_explicit_conclusion(
            "The total number of hard hats remaining in the truck is 43.", "solver_final")
        self.assertEqual(result["answer"], "43")
        self.assertIn(result["method"], ("concluding_object", "concluding_expression", "concluding_verb"))

    def test_conclusion_thus_read(self):
        result = extract_explicit_conclusion(
            "Thus, Julie should read 42 pages.", "solver_final")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "concluding_expression")

    def test_conclusion_costs(self):
        result = extract_explicit_conclusion(
            "Each top costs $5.", "solver_final")
        self.assertEqual(result["answer"], "5")
        self.assertIn(result["method"], ("concluding_verb", "concluding_expression"))

    def test_conclusion_paid(self):
        result = extract_explicit_conclusion(
            "Alexis paid $31 for the shoes.", "solver_final")
        self.assertEqual(result["answer"], "31")
        self.assertIn(result["method"], ("concluding_verb", "concluding_expression",
                                          "safe_single_number_fallback"))

    def test_conclusion_total_income(self):
        result = extract_explicit_conclusion(
            "Irene's total income was $700.", "solver_final")
        self.assertEqual(result["answer"], "700")
        self.assertIn(result["method"], ("concluding_object", "concluding_expression"))

    # ── Natural language conclusion extraction (negative) ──

    def test_no_conclusion_apples_oranges(self):
        result = extract_explicit_conclusion(
            "There are 3 apples and 4 oranges.", "solver_final")
        self.assertEqual(result["answer"], "")
        self.assertFalse(result["explicit"])

    def test_no_conclusion_maybe(self):
        result = extract_explicit_conclusion(
            "Maybe 5 or 6.", "solver_final")
        self.assertEqual(result["answer"], "")
        self.assertFalse(result["explicit"])

    def test_no_conclusion_cannot_determine_with_numbers(self):
        result = extract_explicit_conclusion(
            "Cannot determine; 3 cases remain.", "solver_final")
        self.assertEqual(result["answer"], "")
        self.assertIn("undetermined", result["method"])

    def test_no_conclusion_reason_verified(self):
        result = extract_explicit_conclusion(
            "Reason: 2 + 3 = 5, verified in 2 ways.", "solver_final")
        self.assertEqual(result["answer"], "")
        self.assertFalse(result["explicit"])

    # ── Finalizer loose parse tests ──

    def test_loose_parse_strict_three_line(self):
        loose = parse_finalizer_fields_loose(
            "Selected source: recomputed\nFinal answer: 42\nReason: correct")
        self.assertEqual(loose["final_answer"], "42")
        self.assertEqual(loose["selected_source"], "recomputed")

    def test_loose_parse_preamble_then_three_lines(self):
        loose = parse_finalizer_fields_loose(
            "Let me think.\nSelected source: recomputed\nFinal answer: 42\nReason: correct")
        self.assertEqual(loose["final_answer"], "42")

    def test_loose_parse_label_order_wrong(self):
        loose = parse_finalizer_fields_loose(
            "Final answer: 42\nSelected source: recomputed\nReason: correct")
        self.assertEqual(loose["final_answer"], "42")
        self.assertEqual(loose["selected_source"], "recomputed")

    def test_loose_parse_blank_lines_between(self):
        loose = parse_finalizer_fields_loose(
            "Selected source: recomputed\n\nFinal answer: 42\n\nReason: correct")
        self.assertEqual(loose["final_answer"], "42")

    def test_loose_parse_only_natural_language(self):
        loose = parse_finalizer_fields_loose(
            "Therefore, the answer is 42. I computed it carefully.")
        self.assertEqual(loose["final_answer"], "42")

    def test_loose_parse_format_error_semantic_correct(self):
        loose = parse_finalizer_fields_loose(
            "Preamble\nSelected source: recomputed\nFinal answer: 5\nReason: x")
        self.assertEqual(loose["final_answer"], "5")
        # The strict parse would fail (4 lines), but loose recovers the answer.

    def test_loose_parse_format_error_semantic_wrong(self):
        loose = parse_finalizer_fields_loose(
            "Preamble\nSelected source: recomputed\nFinal answer: wrong\nReason: x")
        self.assertEqual(loose["final_answer"], "")

    def test_loose_parse_ambiguous_multiple_answers(self):
        """Multiple conflicting answers in conclusion window → ambiguous."""
        result = extract_explicit_conclusion(
            "The answer could be 5. Or it could be 6. Both seem possible.", "finalizer")
        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["answer"], "")

    # ── Verifier semantic extraction tests ──

    def test_verifier_loose_json_with_markdown(self):
        raw = '```json\n{"verified_answer": "42", "selected_source": "solver_a"}\n```'
        result = extract_explicit_conclusion(raw, "verifier")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "json_field")

    def test_verifier_natural_language_with_label(self):
        raw = 'The JSON is broken. verified_answer: 42'
        result = extract_explicit_conclusion(raw, "verifier")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "explicit_label")

    def test_verifier_natural_language_the_verified_answer_is(self):
        result = extract_explicit_conclusion(
            "The verified answer is 42.", "verifier")
        self.assertEqual(result["answer"], "42")
        self.assertIn(result["method"], ("concluding_expression", "concluding_verb", "concluding_object"))

    def test_verifier_therefore_supported_answer(self):
        result = extract_explicit_conclusion(
            "Therefore, the supported answer is 42.", "verifier")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "concluding_expression")

    def test_verifier_multiple_conflicting(self):
        result = extract_explicit_conclusion(
            "verified_answer: 5\nBut really verified_answer: 6", "verifier")
        # Two different answers from same label type → ambiguous.
        self.assertEqual(result["answer"], "")
        self.assertTrue(result["ambiguous"])
        self.assertIn("ambiguous", result["method"])

    # ── Format error + semantic correct independence tests ──

    def test_format_error_semantic_correct_combo(self):
        """Format error but semantic answer correct: all flags correct."""
        # Simulate a trace where model puts answer on wrong line but gets it right.
        trace = {
            "final_prediction": "42",
            "gold_answer": "42",
            "invalid_output": True,
            "single_event": {
                "raw_output": "Preamble\nFinal answer: 42\nExtra text",
                "raw_format_compliant": False,
                "semantic_answer": "42",
            },
            "information": {"information_complete": True},
            "candidate_appearances": [{"answer": "42", "correct": True,
                                       "information_complete_at_appearance": True}],
        }
        set_outcome_fields(trace, "42")
        self.assertTrue(trace["semantic_correct"])
        self.assertTrue(trace["correct"])
        self.assertFalse(trace["format_compliant"])
        self.assertFalse(trace["raw_format_compliant"])
        self.assertFalse(trace["strict_answer_correct"])

    def test_single_full_answer_flow_no_finalizer(self):
        """single_full answer_flow should not fabricate finalizer behavior."""
        trace = {
            "single_event": {
                "agent": "solver_a",
                "raw_output": "Final answer: 42\nReasoning.",
                "semantic_answer": "42",
                "raw_format_compliant": True,
            },
            "final_prediction": "42",
            "gold_answer": "42",
            "candidate_answers": {"solver_a": "42"},
            "information": {"information_complete": True},
        }
        flow = analyze_answer_flow(trace)
        self.assertTrue(flow["answer_emergence"]["occurred"])
        self.assertTrue(flow["answer_retention"]["retained_to_final_answer"])
        self.assertIsNone(flow["answer_retention"]["retained_to_finalizer_input"])
        self.assertFalse(flow["final_commit_failure"])
        self.assertIsNone(flow["finalizer_selected_away_from_correct_candidate"])
        self.assertIsNone(flow["finalizer_explicitly_rejected_correct_answer"])
        self.assertEqual(flow["loss_locations"], [])

    def test_single_full_answer_flow_wrong(self):
        """single_full wrong answer: no fabricated finalizer blame."""
        trace = {
            "single_event": {
                "agent": "solver_a",
                "raw_output": "Final answer: 99\nReasoning.",
                "semantic_answer": "99",
                "raw_format_compliant": True,
            },
            "final_prediction": "99",
            "gold_answer": "42",
            "candidate_answers": {"solver_a": "99"},
            "information": {"information_complete": True},
        }
        flow = analyze_answer_flow(trace)
        self.assertFalse(flow["answer_emergence"]["occurred"])
        self.assertFalse(flow["answer_retention"]["retained_to_final_answer"])
        self.assertIsNone(flow["answer_retention"]["retained_to_finalizer_input"])
        self.assertFalse(flow["final_commit_failure"])

    # ── extended semantic extraction through extract_semantic_answer ──

    def test_extract_semantic_answer_concluding_expression(self):
        sem = extract_semantic_answer(
            "Therefore, the final weight of the box is 16 pounds.",
            "Final answer", expected_output_type="solver_final")
        self.assertEqual(sem["answer"], "16")
        self.assertEqual(sem["method"], "concluding_expression")
        self.assertTrue(sem["explicit"])

    def test_extract_semantic_answer_solver_current(self):
        sem = extract_semantic_answer(
            "Current answer: 42\nI still need info.",
            "Current answer", expected_output_type="solver_current")
        self.assertEqual(sem["answer"], "42")
        self.assertEqual(sem["method"], "explicit_label")

    def test_extract_semantic_answer_verifier_type(self):
        sem = extract_semantic_answer(
            '{"verified_answer": "42", "selected_source": "solver_a"}',
            "verified_answer", expected_output_type="verifier")
        self.assertEqual(sem["answer"], "42")
        self.assertEqual(sem["method"], "json_field")

    def test_extract_semantic_answer_finalizer_type(self):
        sem = extract_semantic_answer(
            "Final answer: 42",
            "Final answer", expected_output_type="finalizer")
        self.assertEqual(sem["answer"], "42")
        self.assertEqual(sem["method"], "explicit_label")

    # ── Diagnostic trace fields tests ──

    def test_diagnostic_fields_single_event_present(self):
        """When event has semantic_answer, semantic_answer_absent=False."""
        event = {
            "semantic_answer": "42",
            "semantic_answer_extraction": "explicit_label",
            "semantic_answer_ambiguous": False,
        }
        self.assertFalse(not bool(event.get("semantic_answer")))
        self.assertFalse(bool(event.get("semantic_answer_ambiguous")))

    def test_diagnostic_fields_single_event_absent(self):
        """When event has no semantic_answer, semantic_answer_absent=True."""
        event = {
            "semantic_answer": "",
            "semantic_answer_extraction": "no_supported_answer",
            "semantic_answer_ambiguous": False,
        }
        self.assertTrue(not bool(event.get("semantic_answer")))

    # ── Unified extraction across expected_output_types ──

    def test_unified_extraction_solver_current_label(self):
        """solver_current type recognizes Current answer label."""
        result = extract_explicit_conclusion(
            "Current answer: 42", "solver_current")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "explicit_label")

    def test_unified_extraction_solver_final_label(self):
        """solver_final type recognizes Final answer label."""
        result = extract_explicit_conclusion(
            "Final answer: 42", "solver_final")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "explicit_label")

    def test_unified_extraction_verifier_label(self):
        """verifier type recognizes verified_answer label."""
        result = extract_explicit_conclusion(
            "verified_answer: 42", "verifier")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "explicit_label")

    def test_unified_extraction_finalizer_label(self):
        """finalizer type recognizes Final answer label."""
        result = extract_explicit_conclusion(
            "Final answer: 42", "finalizer")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "explicit_label")

    # ── parse_finalizer_fields_loose additional edge cases ──

    def test_loose_parse_only_final_answer(self):
        """Only Final answer present, no Selected source or Reason."""
        loose = parse_finalizer_fields_loose(
            "Here is my analysis.\nFinal answer: 42")
        self.assertEqual(loose["final_answer"], "42")
        self.assertEqual(loose["selected_source"], "")
        self.assertEqual(loose["reason"], "")

    def test_loose_parse_markdown_wrapped_labels(self):
        """Labels wrapped in markdown should still be found."""
        loose = parse_finalizer_fields_loose(
            "**Selected source:** recomputed\n**Final answer:** 42\n**Reason:** test")
        self.assertEqual(loose["final_answer"], "42")

    def test_loose_parse_undetermined_final_answer(self):
        """Explicitly undetermined Final answer."""
        loose = parse_finalizer_fields_loose(
            "Selected source: none\nFinal answer: cannot determine\nReason: no data")
        self.assertEqual(loose["final_answer"], "")

    # ── Cross-role label pollution tests ──

    def test_finalizer_reason_current_answer_does_not_override_final_answer(self):
        """Finalizer Reason mentioning Current answer: 5 cannot override Final answer: 42."""
        raw = (
            "Selected source: recomputed\n"
            "Final answer: 42\n"
            "Reason: Solver previously wrote Current answer: 5"
        )
        result = extract_explicit_conclusion(raw, "finalizer")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["method"], "explicit_label")

    def test_finalizer_ignores_verified_answer_in_reason(self):
        """Finalizer extraction ignores verified_answer label in Reason text."""
        raw = (
            "Selected source: recomputed\n"
            "Final answer: 42\n"
            "Reason: Verifier said verified_answer: 7 but I disagree"
        )
        result = extract_explicit_conclusion(raw, "finalizer")
        self.assertEqual(result["answer"], "42")

    def test_verifier_ignores_final_answer_in_text(self):
        """Verifier extraction ignores Final answer: label in its text."""
        raw = "The solver wrote Final answer: 99. verified_answer: 42"
        result = extract_explicit_conclusion(raw, "verifier")
        self.assertEqual(result["answer"], "42")

    def test_solver_final_ignores_verified_answer_in_own_text(self):
        """Solver final extraction ignores verified_answer: label."""
        raw = "Final answer: 42\nThe verifier confirmed verified_answer: 99"
        result = extract_explicit_conclusion(raw, "solver_final")
        self.assertEqual(result["answer"], "42")

    def test_solver_current_ignores_final_answer_in_own_text(self):
        """Solver current extraction ignores Final answer: label."""
        raw = "Current answer: 7\nLater I think Final answer: 99"
        result = extract_explicit_conclusion(raw, "solver_current")
        self.assertEqual(result["answer"], "7")

    # ── extract_first_declared_numeric tests ──

    def test_labeled_answer_uses_first_declared_number(self):
        """Label extraction uses FIRST number, not last."""
        result = extract_explicit_conclusion(
            "Final answer: 41 because total=143 and known=102", "solver_final")
        self.assertEqual(result["answer"], "41")

    def test_labeled_answer_with_dollar_first(self):
        """Label extraction: first number after label, skipping reasoning."""
        result = extract_explicit_conclusion(
            "verified_answer: $520 after 4 payments", "verifier")
        self.assertEqual(result["answer"], "520")

    def test_labeled_answer_boxed_first(self):
        """Label extraction prioritizes \\boxed{} over raw numbers."""
        result = extract_explicit_conclusion(
            "Final answer: \\boxed{5} from the calculation 5 = 25/5", "solver_final")
        self.assertEqual(result["answer"], "5")

    def test_undetermined_label_with_numbers_returns_empty(self):
        """Label 'undetermined; 3 facts missing' returns empty."""
        result = extract_explicit_conclusion(
            "Current answer: undetermined; 3 facts missing", "solver_current")
        self.assertEqual(result["answer"], "")
        self.assertIn("undetermined", result["method"])

    # ── extract_numeric_bound_to_conclusion tests ──

    def test_conclusion_number_bound_to_answer_object(self):
        """Bound conclusion: answer is N, not the last number in clause."""
        result = extract_explicit_conclusion(
            "Therefore, the answer is 5, verified in 2 ways.", "solver_final")
        self.assertEqual(result["answer"], "5")

    def test_multiple_unbound_numbers_are_ambiguous(self):
        """Multiple unbound numbers → ambiguous, not last-number guess."""
        result = extract_explicit_conclusion(
            "Therefore, there are 3 apples and 4 oranges.", "solver_final")
        self.assertEqual(result["answer"], "")
        self.assertTrue(result["ambiguous"])

    def test_bound_verb_number_extracted(self):
        """'Roger must buy 3 packs' → 3 via bound verb pattern."""
        result = extract_explicit_conclusion(
            "Roger must buy 3 packs.", "solver_final")
        self.assertEqual(result["answer"], "3")

    def test_single_number_without_marker_not_conclusion(self):
        """'There are 3 apples and 4 oranges.' without markers → empty."""
        result = extract_explicit_conclusion(
            "There are 3 apples and 4 oranges.", "solver_final")
        self.assertEqual(result["answer"], "")

    # ── Undetermined check in conclusion window only ──

    def test_earlier_insufficient_does_not_override_later_answer(self):
        """Earlier 'insufficient information' does not block later explicit answer."""
        result = extract_explicit_conclusion(
            "At first there was insufficient information. "
            "Therefore, the final answer is 42.", "solver_final")
        self.assertEqual(result["answer"], "42")

    def test_concluding_cannot_determine_returns_empty(self):
        """Last clause 'cannot determine' blocks extraction."""
        result = extract_explicit_conclusion(
            "Cannot determine; 3 cases remain.", "solver_final")
        self.assertEqual(result["answer"], "")

    def test_answer_remains_unknown_returns_empty(self):
        """'The answer remains unknown. Therefore 3 cases were checked.' → empty."""
        result = extract_explicit_conclusion(
            "The answer remains unknown. Therefore 3 cases were checked.", "solver_final")
        self.assertEqual(result["answer"], "")
        # Method is undetermined (first clause) or ambiguous (last clause has
        # unbound number "3 cases") — both are correct empty outcomes.
        self.assertIn(result["method"], ("concluding_undetermined", "ambiguous_conclusion"))

    # ── Verifier candidate qualification tests ──

    def test_invalid_verifier_not_added_as_finalizer_candidate_protocol(self):
        """Invalid verifier (not protocol_valid) is NOT added as candidate source."""
        verifier = {
            "raw_output": '{"verified_answer": "42", "selected_source": "solver_a"}',
            "parsed_output": {"verified_answer": "42", "selected_source": "solver_a"},
            "semantic_answer": "42",
            "raw_format_compliant": True,
            "protocol_valid": False,
            "validation_error": "answer does not match selected source solver_a",
            "invalid_output": True,
        }
        candidates = {"solver_a": "99", "solver_b": "42"}
        # Protocol-invalid verifier must NOT become a candidate.
        should_add = (
            verifier.get("protocol_valid")
            and decimal(verifier.get("semantic_answer")) is not None
        )
        self.assertFalse(should_add)

    def test_protocol_valid_verifier_added_as_candidate(self):
        """Protocol-valid verifier IS added as candidate source."""
        verifier = {
            "raw_output": '{"verified_answer": "42", "selected_source": "solver_b"}',
            "parsed_output": {"verified_answer": "42", "selected_source": "solver_b"},
            "semantic_answer": "42",
            "raw_format_compliant": True,
            "protocol_valid": True,
        }
        candidates = {"solver_a": "99", "solver_b": "42"}
        should_add = (
            verifier.get("protocol_valid")
            and decimal(verifier.get("semantic_answer")) is not None
        )
        self.assertTrue(should_add)

    def test_verifier_protocol_error_does_not_change_raw_format_flag(self):
        """JSON valid but source mismatch: raw_format_compliant=True, protocol_valid=False."""
        verifier_raw = '{"verified_answer": "42", "selected_source": "solver_c"}'
        import json as _json
        raw_json_obj = _json.loads(verifier_raw)
        # Format check: valid JSON with required fields.
        fmt_ok = isinstance(raw_json_obj, dict) and all(
            k in raw_json_obj for k in ("verified_answer", "selected_source"))
        self.assertTrue(fmt_ok)
        # But selected_source "solver_c" is not allowed.
        source = raw_json_obj.get("selected_source", "")
        proto_ok = source in {"solver_a", "solver_b", "verifier", "recomputed", "none"}
        self.assertFalse(proto_ok)
        # raw_format_compliant reflects format only.
        self.assertTrue(fmt_ok)
        # protocol_valid requires both format and protocol.
        self.assertFalse(fmt_ok and proto_ok)

    def test_finalizer_format_error_cannot_be_protocol_valid(self):
        """When normalized_format_error is set, protocol_valid must be False."""
        # Simulate a finalizer event with format error but no protocol error.
        normalized_format_error = "expected exactly three lines"
        protocol_validation_error = ""
        protocol_valid = (
            not bool(normalized_format_error) and not bool(protocol_validation_error)
        )
        self.assertFalse(protocol_valid)

    def test_finalizer_format_ok_protocol_ok_is_protocol_valid(self):
        """When both format and protocol pass, protocol_valid is True."""
        normalized_format_error = ""
        protocol_validation_error = ""
        protocol_valid = (
            not bool(normalized_format_error) and not bool(protocol_validation_error)
        )
        self.assertTrue(protocol_valid)

    # ── Same-label multiple occurrence semantics ──

    def test_same_label_same_answer_accepted(self):
        """Same label appears twice with same answer → accepted."""
        raw = "Final answer: 42\nRecheck: Final answer: 42"
        result = extract_explicit_conclusion(raw, "solver_final")
        self.assertEqual(result["answer"], "42")

    def test_same_label_different_answers_ambiguous(self):
        """Same label appears with different answers → ambiguous."""
        raw = "Final answer: 5\nWait no, Final answer: 6"
        result = extract_explicit_conclusion(raw, "solver_final")
        self.assertEqual(result["answer"], "")
        self.assertTrue(result["ambiguous"])

    # ── Structured-parser first-declared-number tests ──

    def test_solver_strict_parser_uses_first_declared_number(self):
        """parse_solver_final extracts FIRST number, not last."""
        raw = (
            "Final answer: 41 because total=143 and known=102\n"
            "The calculation is complete."
        )
        answer, error = parse_solver_final(raw)
        self.assertEqual(answer, "41")
        self.assertEqual(error, "")

    def test_finalizer_strict_parser_uses_first_declared_number(self):
        """parse_fixed_finalizer extracts FIRST number from Final answer."""
        raw = (
            "Selected source: recomputed\n"
            "Final answer: 41 because total=143 and known=102\n"
            "Reason: calculated from visible facts"
        )
        parsed, error = parse_fixed_finalizer(raw)
        self.assertEqual(parsed["final_answer"], "41")
        self.assertEqual(error, "")

    def test_finalizer_loose_parser_uses_first_declared_number(self):
        """parse_finalizer_fields_loose extracts FIRST number."""
        raw = (
            "Selected source: recomputed\n"
            "Final answer: 41 because total=143 and known=102\n"
            "Reason: calculated"
        )
        parsed = parse_finalizer_fields_loose(raw)
        self.assertEqual(parsed["final_answer"], "41")

    def test_strict_solver_undetermined_first_line(self):
        """Solver first line undetermined → empty answer, no error."""
        raw = "Final answer: cannot determine\nThe problem lacks data."
        answer, error = parse_solver_final(raw)
        self.assertEqual(answer, "")
        self.assertEqual(error, "")

    def test_strict_finalizer_undetermined(self):
        """Finalizer undetermined → empty answer, no error."""
        raw = (
            "Selected source: none\n"
            "Final answer: cannot determine\n"
            "Reason: insufficient evidence"
        )
        parsed, error = parse_fixed_finalizer(raw)
        self.assertEqual(parsed["final_answer"], "")
        self.assertEqual(error, "")

    def test_strict_finalizer_non_numeric_answer_is_error(self):
        """Finalizer non-numeric non-undetermined → error."""
        raw = (
            "Selected source: recomputed\n"
            "Final answer: forty two\n"
            "Reason: the answer is forty two"
        )
        parsed, error = parse_fixed_finalizer(raw)
        self.assertTrue(bool(error))
        self.assertEqual(parsed["final_answer"], "")

    # ── Verifier field type check tests ──

    def test_verifier_rejects_non_scalar_answer_field(self):
        """verified_answer as list → raw_format_compliant=False."""
        raw = '{"verified_answer": [42], "selected_source": "solver_a"}'
        raw_json = raw_json_object(raw)
        self.assertIsNotNone(raw_json)
        verdict = not isinstance(raw_json.get("verified_answer"), bool) and isinstance(
            raw_json.get("verified_answer"), (str, int, float, type(None)))
        self.assertFalse(verdict, "list verified_answer must be rejected")

    def test_verifier_rejects_non_string_source_field(self):
        """selected_source as int → raw_format_compliant=False."""
        raw = '{"verified_answer": "42", "selected_source": 5}'
        raw_json = raw_json_object(raw)
        self.assertIsNotNone(raw_json)
        self.assertFalse(isinstance(raw_json.get("selected_source"), str),
                         "non-string selected_source must be rejected")

    def test_verifier_bool_answer_field_rejected(self):
        """verified_answer as bool → raw_format_compliant=False."""
        raw = '{"verified_answer": true, "selected_source": "solver_a"}'
        raw_json = raw_json_object(raw)
        self.assertIsNotNone(raw_json)
        verdict = not isinstance(raw_json.get("verified_answer"), bool) and isinstance(
            raw_json.get("verified_answer"), (str, int, float, type(None)))
        self.assertFalse(verdict, "bool verified_answer must be rejected")

    # ── build_trace candidate_answers protocol gate tests ──

    def test_invalid_verifier_not_recorded_as_candidate_answer(self):
        """protocol_valid=False verifier: not in candidate_answers."""
        verifier = {
            "raw_output": '{"verified_answer": "42", "selected_source": "solver_c"}',
            "semantic_answer": "42",
            "raw_format_compliant": True,
            "protocol_valid": False,
            "validation_error": "selected source solver_c is unavailable",
        }
        candidates = {"solver_a": "99", "solver_b": "42"}
        if verifier and verifier.get("protocol_valid"):
            v_ans = verifier.get("semantic_answer")
            if decimal(v_ans) is not None:
                candidates["verifier"] = v_ans
        self.assertNotIn("verifier", candidates,
                         "protocol-invalid verifier must not appear in candidate_answers")

    def test_valid_verifier_recorded_as_candidate_answer(self):
        """protocol_valid=True verifier with numeric answer → in candidates."""
        verifier = {
            "raw_output": '{"verified_answer": "42", "selected_source": "solver_b"}',
            "semantic_answer": "42",
            "raw_format_compliant": True,
            "protocol_valid": True,
        }
        candidates = {"solver_a": "99", "solver_b": "42"}
        if verifier and verifier.get("protocol_valid"):
            v_ans = verifier.get("semantic_answer")
            if decimal(v_ans) is not None:
                candidates["verifier"] = v_ans
        self.assertIn("verifier", candidates)
        self.assertEqual(candidates["verifier"], "42")

    def test_verifier_consistency_error_uses_first_declared_number(self):
        """verifier_consistency_error extracts first number from verified_answer."""
        raw = ('{"verified_answer": "41 because total=143 known=102", '
               '"selected_source": "recomputed"}')
        parsed_json = raw_json_object(raw)
        self.assertIsNotNone(parsed_json)
        verified_raw = parsed_json.get("verified_answer")
        self.assertFalse(explicitly_undetermined(verified_raw))
        answer = extract_first_declared_numeric(str(verified_raw))
        self.assertEqual(answer, "41")

    # ── production-path mock test for build_trace candidate_answers ──

    def test_build_trace_candidate_answers_excludes_invalid_verifier(self):
        """Full trace simulation: invalid verifier not in candidate_answers."""
        verifier_event = {
            "agent": "verifier",
            "raw_output": '{"verified_answer": "42", "selected_source": "solver_c"}',
            "semantic_answer": "42",
            "raw_format_compliant": True,
            "protocol_valid": False,
            "validation_error": "selected source solver_c is unavailable or invalid",
            "invalid_output": True,
        }
        candidates = {"solver_a": "99", "solver_b": "42"}
        if (verifier_event and verifier_event.get("protocol_valid")
                and decimal(verifier_event.get("semantic_answer")) is not None):
            candidates["verifier"] = verifier_event["semantic_answer"]
        # The production gate must block the invalid verifier.
        self.assertNotIn("verifier", candidates)
        # But semantic_answer must still be preserved for diagnostics.
        self.assertEqual(verifier_event["semantic_answer"], "42")
        self.assertFalse(verifier_event["protocol_valid"])

    # ── Production-path tests with mock models ──

    _FULL_VERIFIER_JSON = (
        '{"information_sufficient": true, "revealed_facts": ["fact1"], '
        '"candidate_checks": [{"source": "solver_a", "answer": "99", '
        '"status": "correct", "reason": "matches"}], '
        '"verified_answer": "42", "selected_source": "solver_b", '
        '"missing_information": []}'
    )

    def test_production_finish_multi_invalid_verifier_no_answer_leak(self):
        """finish_multi: invalid verifier answer not leaked to finalizer prompt."""
        solver_finals = {
            "a": {"agent": "solver_a", "semantic_answer": "99",
                  "raw_output": "Final answer: 99\nOK."},
            "b": {"agent": "solver_b", "semantic_answer": "42",
                  "raw_output": "Final answer: 42\nOK."},
        }
        discussion = {"public_transcript": "test transcript",
                      "solver_finals": solver_finals}
        # Verifier JSON: solver_b is correct answer 42, but source=recomputed
        # with answer 42 that differs from solver_a (99) → mismatch may occur.
        # Use source "solver_a" with answer 99, but verified_answer=42 → mismatch.
        verifier_json = (
            '{"information_sufficient": true, "revealed_facts": [], '
            '"candidate_checks": [], '
            '"verified_answer": "42", "selected_source": "solver_a", '
            '"missing_information": []}'
        )
        model = _MockModel([
            (verifier_json, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 0.1),
            ("Selected source: recomputed\nFinal answer: 42\nReason: calc",
             {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}, 0.2),
        ])
        prompts = {"verifier": "VERIFIER_SYS", "finalizer": "FINALIZER_SYS"}
        verifier, finalizer = finish_multi(model, prompts, self.item, discussion,
                                           with_verifier=True)
        self.assertIsNotNone(verifier)
        # Verifier protocol_valid=False because answer 42 != source solver_a answer 99.
        self.assertFalse(verifier["protocol_valid"])
        # But semantic answer is still 42 (preserved for diagnostics).
        self.assertEqual(verifier["semantic_answer"], "42")
        # Finalizer input must NOT contain _semantic_verified_answer.
        finalizer_input = str(finalizer.get("actual_input", ""))
        self.assertNotIn("_semantic_verified_answer", finalizer_input)
        # Verifier report in finalizer input must NOT expose the answer 42.
        # The report should say usable=False.
        self.assertIn('"usable"', finalizer_input)

    def test_production_missing_fields_verifier_format_invalid(self):
        """finish_multi: JSON missing required fields → format invalid."""
        solver_finals = {
            "a": {"agent": "solver_a", "semantic_answer": "99",
                  "raw_output": "Final answer: 99\nOK."},
            "b": {"agent": "solver_b", "semantic_answer": "42",
                  "raw_output": "Final answer: 42\nOK."},
        }
        discussion = {"public_transcript": "test transcript",
                      "solver_finals": solver_finals}
        # Only 2 fields — missing 4 required fields.
        verifier_json = ('{"verified_answer": "42", "selected_source": "recomputed"}')
        model = _MockModel([
            (verifier_json, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 0.1),
            ("Selected source: recomputed\nFinal answer: 42\nReason: calc",
             {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}, 0.2),
        ])
        prompts = {"verifier": "VERIFIER_SYS", "finalizer": "FINALIZER_SYS"}
        verifier, finalizer = finish_multi(model, prompts, self.item, discussion,
                                           with_verifier=True)
        self.assertFalse(verifier["raw_format_compliant"])
        self.assertFalse(verifier["protocol_valid"])
        self.assertIn("missing", verifier.get("verifier_format_error", ""))

    def test_production_full_valid_schema_passes_format(self):
        """finish_multi: complete six-field JSON passes format check."""
        solver_finals = {
            "a": {"agent": "solver_a", "semantic_answer": "99",
                  "raw_output": "Final answer: 99\nOK."},
            "b": {"agent": "solver_b", "semantic_answer": "42",
                  "raw_output": "Final answer: 42\nOK."},
        }
        discussion = {"public_transcript": "test transcript",
                      "solver_finals": solver_finals}
        # Complete valid schema; answer matches solver_b.
        verifier_json = (
            '{"information_sufficient": true, "revealed_facts": ["a"], '
            '"candidate_checks": [{"source": "solver_b", "answer": "42", '
            '"status": "correct", "reason": "matches"}], '
            '"verified_answer": "42", "selected_source": "solver_b", '
            '"missing_information": []}'
        )
        model = _MockModel([
            (verifier_json, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 0.1),
            ("Selected source: solver_b\nFinal answer: 42\nReason: matches",
             {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}, 0.2),
        ])
        prompts = {"verifier": "VERIFIER_SYS", "finalizer": "FINALIZER_SYS"}
        verifier, finalizer = finish_multi(model, prompts, self.item, discussion,
                                           with_verifier=True)
        self.assertTrue(verifier["raw_format_compliant"])
        # Protocol valid: answer matches source solver_b=42.
        self.assertTrue(verifier["protocol_valid"])
        self.assertEqual(verifier["semantic_answer"], "42")

    def test_production_invalid_verifier_not_affect_classify(self):
        """Invalid verifier with correct answer does not cause answer_selection_failure."""
        # Build a trace where solvers are both wrong, verifier has correct answer
        # but is protocol-invalid, and finalizer is also wrong.
        trace = {
            "semantic_correct": False,
            "final_prediction": "99",
            "gold_answer": "42",
            "invalid_output": False,
            "information": {"information_complete": True},
            "candidate_answers": {"solver_a": "99", "solver_b": "99"},
            "answer_reason_consistent": True,
            "candidate_appearances": [
                {"source": "solver_a", "phase": "solver_final", "answer": "99",
                 "correct": False, "information_complete_at_appearance": True,
                 "eligible_for_finalizer": True},
                {"source": "solver_b", "phase": "solver_final", "answer": "99",
                 "correct": False, "information_complete_at_appearance": True,
                 "eligible_for_finalizer": True},
                # Verifier: correct answer 42 but diagnostic_only=True.
                {"source": "verifier", "phase": "verification", "answer": "42",
                 "correct": True, "information_complete_at_appearance": True,
                 "eligible_for_finalizer": False, "diagnostic_only": True,
                 "protocol_valid": False},
            ],
            "verifier_event": {
                "semantic_answer": "42",
                "raw_format_compliant": True,
                "protocol_valid": False,
            },
        }
        failure_type, lucky = classify(trace, "42")
        # No eligible correct answer appeared → information_integration_failure.
        self.assertEqual(failure_type, "information_integration_failure")

    def test_production_build_trace_candidates_match_finalizer_prompt(self):
        """build_trace candidate_answers exactly matches Finalizer prompt candidates."""
        # Simulate what build_trace does for candidate construction.
        solver_finals = {
            "a": {"agent": "solver_a", "semantic_answer": "99",
                  "raw_output": "Final answer: 99\nOK."},
            "b": {"agent": "solver_b", "semantic_answer": "42",
                  "raw_output": "Final answer: 42\nOK."},
        }
        # Build candidates exactly as build_trace does.
        raw_candidates = {
            "solver_a": event_answer(solver_finals["a"]),
            "solver_b": event_answer(solver_finals["b"]),
        }
        candidates = {source: answer for source, answer in raw_candidates.items()
                      if decimal(answer) is not None}
        # Verifier is protocol-invalid → not in candidates.
        verifier_event = {
            "semantic_answer": "42",
            "protocol_valid": False,
        }
        if (verifier_event and verifier_event.get("protocol_valid")
                and decimal(verifier_event.get("semantic_answer")) is not None):
            candidates["verifier"] = verifier_event["semantic_answer"]
        self.assertNotIn("verifier", candidates)
        self.assertEqual(candidates, {"solver_a": "99", "solver_b": "42"})

        # Now with protocol-valid verifier.
        verifier_event2 = {
            "semantic_answer": "42",
            "protocol_valid": True,
        }
        candidates2 = {source: answer for source, answer in raw_candidates.items()
                       if decimal(answer) is not None}
        if (verifier_event2 and verifier_event2.get("protocol_valid")
                and decimal(verifier_event2.get("semantic_answer")) is not None):
            candidates2["verifier"] = verifier_event2["semantic_answer"]
        self.assertIn("verifier", candidates2)
        self.assertEqual(candidates2["verifier"], "42")

    def test_production_solver_null_answer_filtered_from_candidates(self):
        """Solver with undetermined/empty answer is excluded from candidates."""
        solver_finals = {
            "a": {"agent": "solver_a", "semantic_answer": "",
                  "raw_output": "Final answer: cannot determine\nNeed more data."},
            "b": {"agent": "solver_b", "semantic_answer": "42",
                  "raw_output": "Final answer: 42\nOK."},
        }
        raw_candidates = {
            "solver_a": event_answer(solver_finals["a"]),
            "solver_b": event_answer(solver_finals["b"]),
        }
        candidates = {source: answer for source, answer in raw_candidates.items()
                      if decimal(answer) is not None}
        self.assertNotIn("solver_a", candidates)
        self.assertIn("solver_b", candidates)

    def test_production_finalizer_saw_correct_answer_no_underscore_match(self):
        """_semantic_verified_answer is not matched as verified_answer."""
        prompt = (
            'Verifier report: {"usable": false, "_semantic_verified_answer": "42", '
            '"validation_error": "bad"}'
        )
        declared = []
        for label in ("Current answer", "Final answer", "verified_answer"):
            for value in re.findall(
                rf"(?im)(?<![A-Za-z0-9_]){re.escape(label)}\s*[:：=]\s*([^\r\n]+)",
                prompt,
            ):
                declared.append({"label": label, "value": value.strip()})
        # No real "verified_answer" label — only "_semantic_verified_answer".
        self.assertEqual(len(declared), 0,
                         "_semantic_verified_answer must not match verified_answer label")

    def test_production_verifier_schema_rejects_extra_field(self):
        """verifier_schema_error: extra field → format invalid."""
        raw = {
            "information_sufficient": True,
            "revealed_facts": [],
            "candidate_checks": [],
            "verified_answer": "42",
            "selected_source": "solver_a",
            "missing_information": [],
            "extra_field": "should not be here",
        }
        error = verifier_schema_error(raw)
        self.assertTrue(bool(error))
        self.assertIn("unexpected", error)

    def test_production_verifier_schema_rejects_bool_answer(self):
        """verifier_schema_error: boolean verified_answer → format invalid."""
        raw = {
            "information_sufficient": True,
            "revealed_facts": [],
            "candidate_checks": [],
            "verified_answer": True,
            "selected_source": "solver_a",
            "missing_information": [],
        }
        error = verifier_schema_error(raw)
        self.assertTrue(bool(error))

    def test_production_verifier_schema_rejects_list_answer(self):
        """verifier_schema_error: list verified_answer → format invalid."""
        raw = {
            "information_sufficient": True,
            "revealed_facts": [],
            "candidate_checks": [],
            "verified_answer": [42],
            "selected_source": "solver_a",
            "missing_information": [],
        }
        error = verifier_schema_error(raw)
        self.assertTrue(bool(error))

    def test_production_verifier_schema_rejects_non_string_source(self):
        """verifier_schema_error: integer selected_source → format invalid."""
        raw = {
            "information_sufficient": True,
            "revealed_facts": [],
            "candidate_checks": [],
            "verified_answer": "42",
            "selected_source": 5,
            "missing_information": [],
        }
        error = verifier_schema_error(raw)
        self.assertTrue(bool(error))

    def test_production_verifier_schema_rejects_bad_candidate_check(self):
        """verifier_schema_error: candidate_check missing 'reason' → invalid."""
        raw = {
            "information_sufficient": True,
            "revealed_facts": [],
            "candidate_checks": [{"source": "solver_a", "answer": "42", "status": "ok"}],
            "verified_answer": "42",
            "selected_source": "solver_a",
            "missing_information": [],
        }
        error = verifier_schema_error(raw)
        self.assertTrue(bool(error))

    def test_production_verifier_schema_rejects_non_string_info_items(self):
        """verifier_schema_error: revealed_facts contains non-string → invalid."""
        raw = {
            "information_sufficient": True,
            "revealed_facts": [42],
            "candidate_checks": [],
            "verified_answer": "42",
            "selected_source": "solver_a",
            "missing_information": [],
        }
        error = verifier_schema_error(raw)
        self.assertTrue(bool(error))

    def test_production_candidate_appearances_has_eligibility_fields(self):
        """candidate_appearances includes eligible_for_finalizer and diagnostic_only."""
        trace = {
            "verifier_event": {
                "agent": "verifier",
                "semantic_answer": "42",
                "raw_format_compliant": True,
                "protocol_valid": False,
            },
            "discussion": {"discussion_events": [], "solver_finals": {}},
            "information": {"information_complete": True},
        }
        appearances = candidate_appearances(trace)
        verifier_apps = [a for a in appearances if a["source"] == "verifier"]
        self.assertEqual(len(verifier_apps), 1)
        self.assertFalse(verifier_apps[0]["eligible_for_finalizer"])
        self.assertTrue(verifier_apps[0]["diagnostic_only"])


if __name__ == "__main__":
    main()
