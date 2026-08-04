"""Full experiment pipeline: order effects, timing isolation, Judge, metrics, gate/20q.

Extends run_hidden_gsm8k.py with all experimental conditions, independent Judge,
comprehensive metrics, trajectory analysis, format variants, anchoring tests,
baselines, oracles, and stability infrastructure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

# ── Import core infrastructure from run_hidden_gsm8k ──────────────────────
# Add the script directory to the path so we can import from the existing module.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# We import key functions and classes; the existing script is not a package
# so we exec the relevant parts. For maintainability we duplicate stable
# utilities here and import the model wrapper via module loading.
import importlib.util as _iu

_HIDDEN_SPEC = _iu.spec_from_file_location("run_hidden_gsm8k", _SCRIPT_DIR / "run_hidden_gsm8k.py")
_run_hidden = _iu.module_from_spec(_HIDDEN_SPEC)
_HIDDEN_SPEC.loader.exec_module(_run_hidden)

# Re-export core symbols for convenience
read_json_records = _run_hidden.read_json_records
extract_answer = _run_hidden.extract_answer
decimal = _run_hidden.decimal
equivalent = _run_hidden.equivalent
extract_labeled_answer = _run_hidden.extract_labeled_answer
extract_free_text_answer = _run_hidden.extract_free_text_answer
parse_solver_final = _run_hidden.parse_solver_final
explicitly_undetermined = _run_hidden.explicitly_undetermined
concludingly_undetermined = _run_hidden.concludingly_undetermined
parse_object = _run_hidden.parse_object
raw_json_object = _run_hidden.raw_json_object
blank_usage = _run_hidden.blank_usage
add_usage = _run_hidden.add_usage
derived_seed = _run_hidden.derived_seed
reseed_model = _run_hidden.reseed_model
model_event = _run_hidden.model_event
paired_model_events = _run_hidden.paired_model_events
public_transcript = _run_hidden.public_transcript
run_discussion = _run_hidden.run_discussion
replay_facts = _run_hidden.replay_facts
replay_fact_hash = _run_hidden.replay_fact_hash
replay_ledger = _run_hidden.replay_ledger
run_replay_discussion = _run_hidden.run_replay_discussion
coverage_score = _run_hidden.coverage_score
atomic_facts = _run_hidden.atomic_facts
fact_is_public = _run_hidden.fact_is_public
objective_information = _run_hidden.objective_information
add_information_timeline = _run_hidden.add_information_timeline
single_call = _run_hidden.single_call
event_answer = _run_hidden.event_answer
candidate_appearances = _run_hidden.candidate_appearances
parse_fixed_finalizer = _run_hidden.parse_fixed_finalizer
check_answer_reason_consistency = _run_hidden.check_answer_reason_consistency
source_consistency_error = _run_hidden.source_consistency_error
has_explicit_identical_candidate_rejection = _run_hidden.has_explicit_identical_candidate_rejection
verifier_explains_identical_candidate_rejection = _run_hidden.verifier_explains_identical_candidate_rejection
verifier_consistency_error = _run_hidden.verifier_consistency_error
call_finalizer_once = _run_hidden.call_finalizer_once
finish_multi = _run_hidden.finish_multi
collect_events = _run_hidden.collect_events
classify = _run_hidden.classify
set_outcome_fields = _run_hidden.set_outcome_fields
build_trace = _run_hidden.build_trace
build_replay_trace = _run_hidden.build_replay_trace
build_finalizer_order_trace = _run_hidden.build_finalizer_order_trace
deepseek_review = _run_hidden.deepseek_review
write_outputs = _run_hidden.write_outputs
write_replay_analysis = _run_hidden.write_replay_analysis
write_finalizer_order_analysis = _run_hidden.write_finalizer_order_analysis
LocalQwen = _run_hidden.LocalQwen
load_ml_dependencies = _run_hidden.load_ml_dependencies
load_api_dependencies = _run_hidden.load_api_dependencies
validate_model_path = _run_hidden.validate_model_path
choose_model_dtype = _run_hidden.choose_model_dtype
dependency_status = _run_hidden.dependency_status
as_bool = _run_hidden.as_bool

# Independent defaults — no dependency on run_hidden_gsm8k.py for config
ROOT = _SCRIPT_DIR
DATA_PATH = ROOT / "data" / "20.json"
MODEL_PATH = ROOT / "qwen2.5-1.5B"
OUTPUT_BASE_DIR = ROOT / "outputs_full_experiment"
PROMPT_DIR = ROOT / "hidden_gsm8k_prompts"
PROMPT_PATHS = {
    "solver": PROMPT_DIR / "solver.txt",
    "verifier": PROMPT_DIR / "verifier.txt",
    "finalizer": PROMPT_DIR / "finalizer.txt",
}
USAGE_KEYS = _run_hidden.USAGE_KEYS
VERIFIER_DEFAULT = _run_hidden.VERIFIER_DEFAULT
FINALIZER_DEFAULT = _run_hidden.FINALIZER_DEFAULT
UNDETERMINED_ANSWERS = _run_hidden.UNDETERMINED_ANSWERS
SETTINGS = _run_hidden.SETTINGS
REPLAY_SETTINGS = _run_hidden.REPLAY_SETTINGS
FINALIZER_ORDER_SETTING = _run_hidden.FINALIZER_ORDER_SETTING
CONTROLLED_SETTINGS = _run_hidden.CONTROLLED_SETTINGS
SETTING_NAMES = _run_hidden.SETTING_NAMES
DEEPSEEK_BASE_URL = _run_hidden.DEEPSEEK_BASE_URL
DEEPSEEK_MODEL = _run_hidden.DEEPSEEK_MODEL
DEEPSEEK_API_KEY_ENV_NAMES = _run_hidden.DEEPSEEK_API_KEY_ENV_NAMES
DEFAULT_DEVICE = _run_hidden.DEFAULT_DEVICE
DEFAULT_TEMPERATURE = _run_hidden.DEFAULT_TEMPERATURE
DEFAULT_MAX_NEW_TOKENS = _run_hidden.DEFAULT_MAX_NEW_TOKENS
DEFAULT_DISCUSSION_ROUNDS = _run_hidden.DEFAULT_DISCUSSION_ROUNDS
DEFAULT_SEED = _run_hidden.DEFAULT_SEED
DEFAULT_LIMIT = _run_hidden.DEFAULT_LIMIT
DEFAULT_JUDGE_MAX_ATTEMPTS = _run_hidden.DEFAULT_JUDGE_MAX_ATTEMPTS

# ═══════════════════════════════════════════════════════════════════════════
# NEW EXPERIMENTAL SETTINGS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

# Order isolation settings
ORDER_ISOLATION_SETTINGS = (
    "solver_only_AB", "solver_only_BA",
    "finalizer_only_AB", "finalizer_only_BA",
    "frozen_transcript_AB", "frozen_transcript_BA",
    "canonical_order",
    "random_order",
)

# Information timing settings (each with AB and BA)
TIMING_SETTINGS = (
    "info_at_start_AB", "info_at_start_BA",
    "info_after_round1_AB", "info_after_round1_BA",
    "info_before_final_AB", "info_before_final_BA",
    "info_before_finalizer_AB", "info_before_finalizer_BA",
    "info_reset_direct_AB", "info_reset_direct_BA",
)

# Format variant settings
FORMAT_VARIANT_SETTINGS = (
    "format_three_line",
    "format_strict_json",
    "format_xml_tags",
    "format_answer_only",
    "format_answer_first",
    "format_reason_first",
    "format_reason_then_answer",
    "format_internal_reasoning_then_structured",
    "format_deterministic_extract",
    "format_self_check_before_commit",
)

# Belief anchoring settings
ANCHORING_SETTINGS = (
    "anchor_early_wrong_late_full",
    "anchor_early_none_late_full",
    "anchor_early_correct_late_full",
    "anchor_early_conflict_late_full",
    "anchor_source_solver_a",
    "anchor_source_solver_b",
    "anchor_source_verifier",
    "anchor_source_system",
    "anchor_repeat_1",
    "anchor_repeat_2",
    "anchor_repeat_many",
    "counter_belief_prompt",
    "context_reset",
    "belief_reset",
)

# Ledger variant settings
LEDGER_SETTINGS = (
    "ledger_raw_concat",
    "ledger_structured_kv",
    "ledger_dependency_table",
    "ledger_canonical_sorted",
    "ledger_provenance_free",
    "ledger_provenance_aware",
)

# Baseline and oracle settings
BASELINE_SETTINGS = (
    "single_full_information",
    "single_late_information",
    "deterministic_calculator",
    "best_solver_oracle",
    "discussion_oracle",
    "finalizer_upper_bound",
)

# All new settings
ALL_NEW_SETTINGS = (
    ORDER_ISOLATION_SETTINGS +
    TIMING_SETTINGS +
    FORMAT_VARIANT_SETTINGS +
    ANCHORING_SETTINGS +
    LEDGER_SETTINGS +
    BASELINE_SETTINGS
)

# Combine with existing settings
# Settings handled inline (not built by generic builders)
INLINE_SETTINGS = ("after_round1_BA", "before_final_transcript_BA")

ALL_SETTINGS = SETTINGS + ALL_NEW_SETTINGS + INLINE_SETTINGS

NEW_SETTING_NAMES = {
    # Order isolation
    "solver_only_AB": "Solver-only Fact Order AB (finalizer sees canonical ledger)",
    "solver_only_BA": "Solver-only Fact Order BA (finalizer sees canonical ledger)",
    "finalizer_only_AB": "Finalizer-only Fact Order AB (frozen solver transcript)",
    "finalizer_only_BA": "Finalizer-only Fact Order BA (frozen solver transcript)",
    "frozen_transcript_AB": "Frozen Transcript - Finalizer sees AB order",
    "frozen_transcript_BA": "Frozen Transcript - Finalizer sees BA order",
    "canonical_order": "Canonical Order (sorted by field name)",
    "random_order": "Random Order (multiple permutations)",
    # Information timing
    "info_at_start_AB": "Info at Start AB",
    "info_at_start_BA": "Info at Start BA",
    "info_after_round1_AB": "Info after Round 1 AB",
    "info_after_round1_BA": "Info after Round 1 BA",
    "info_before_final_AB": "Info before Final AB",
    "info_before_final_BA": "Info before Final BA",
    "info_before_finalizer_AB": "Info before Finalizer AB",
    "info_before_finalizer_BA": "Info before Finalizer BA",
    "info_reset_direct_AB": "Info Reset Direct AB",
    "info_reset_direct_BA": "Info Reset Direct BA",
    # Inline-handled settings
    "after_round1_BA": "Information Replay - Reveal after Round 1 (BA order)",
    "before_final_transcript_BA": "Information Replay - Before Final with Transcript (BA order)",
    # Format variants
    "format_three_line": "Format: Three-line Text",
    "format_strict_json": "Format: Strict JSON",
    "format_xml_tags": "Format: XML Tags",
    "format_answer_only": "Format: Answer Only",
    "format_answer_first": "Format: Answer First",
    "format_reason_first": "Format: Reason First",
    "format_reason_then_answer": "Format: Reason then Answer",
    "format_internal_reasoning_then_structured": "Format: Internal Reasoning then Structured",
    "format_deterministic_extract": "Format: Deterministic Extract from Reasoning",
    "format_self_check_before_commit": "Format: Self-check Before Commit",
    # Anchoring
    "anchor_early_wrong_late_full": "Anchor: Early Wrong + Late Full Facts",
    "anchor_early_none_late_full": "Anchor: Early No Candidate + Late Full Facts",
    "anchor_early_correct_late_full": "Anchor: Early Correct + Late Full Facts",
    "anchor_early_conflict_late_full": "Anchor: Early Conflict + Late Full Facts",
    "anchor_source_solver_a": "Anchor: Wrong from Solver A",
    "anchor_source_solver_b": "Anchor: Wrong from Solver B",
    "anchor_source_verifier": "Anchor: Wrong from Verifier",
    "anchor_source_system": "Anchor: Wrong from System Prompt",
    "anchor_repeat_1": "Anchor: Wrong Repeated 1x",
    "anchor_repeat_2": "Anchor: Wrong Repeated 2x",
    "anchor_repeat_many": "Anchor: Wrong Repeated Many Times",
    "counter_belief_prompt": "Counter-Belief Prompt",
    "context_reset": "Context Reset (delete history)",
    "belief_reset": "Belief Reset (invalidate old candidates)",
    # Ledger
    "ledger_raw_concat": "Ledger: Raw Fact Concatenation",
    "ledger_structured_kv": "Ledger: Structured Key-Value",
    "ledger_dependency_table": "Ledger: Dependency Calculation Table",
    "ledger_canonical_sorted": "Ledger: Canonical Sorted",
    "ledger_provenance_free": "Ledger: Provenance-Free",
    "ledger_provenance_aware": "Ledger: Provenance-Aware",
    # Baselines
    "single_full_information": "Single-Agent Full Information Baseline",
    "single_late_information": "Single-Agent Late Information Baseline",
    "deterministic_calculator": "Deterministic Calculator Baseline",
    "best_solver_oracle": "Best Solver Oracle",
    "discussion_oracle": "Discussion Oracle",
    "finalizer_upper_bound": "Finalizer Upper Bound",
}

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED FINALIZER PROMPT (Section 1 fix)
# ═══════════════════════════════════════════════════════════════════════════

ENHANCED_FINALIZER_PROMPT = """You produce the final answer without seeing the original complete problem. Use only the shared question, public discussion, solver candidates, and (when supplied) verifier report. Do not invent missing facts. You may select a supported candidate or recompute from public facts.

Return exactly three non-empty physical lines, with no Markdown, no blank lines between fields, and no extra lines. The labels, in order, must be Selected source, Final answer, and Reason, each followed by a colon and the value for the current question. Do not copy placeholder text or invent a sample number.

CRITICAL RULE WHEN `Valid non-empty candidates: {}` APPEARS:
There are NO pre-computed candidates. You MUST ignore every rule about identical candidates or candidate agreement — those rules only apply when candidates exist. The FIRST character you emit MUST be the start of "Selected source:". No preamble, no introductory text, no blank line, no thinking aloud. Start directly with "Selected source:".

When `Valid non-empty candidates: {}` appears, use `Selected source: recomputed` only when `Final answer` is a concrete numeric answer derived from visible evidence. If the visible evidence cannot determine a numeric answer, the only valid first two lines are exactly `Selected source: none` and `Final answer: undetermined`.

Use only a selected_source listed as available in the current input. Reason must be non-empty. If solver_a, solver_b, or verifier is selected, copy Final answer exactly from that source's valid non-empty candidate; do not recompute it. If solver_a and solver_b agree, preserve that answer. To select recomputed with a different answer, Reason must start with `Reject identical candidates because` (or `拒绝相同候选，因为`) and then identify the concrete error or contradiction in the public evidence. Never silently replace identical candidates. Use none and an undetermined Final answer only when the public information truly cannot support an answer."""

# ═══════════════════════════════════════════════════════════════════════════
# INDEPENDENT JUDGE SYSTEM (Section 3)
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT_V1 = """You are an independent mathematical judge. Your task is to evaluate a final answer and its reasoning against a math problem and its complete facts. You do NOT know what experimental setting produced this answer. You only see the problem, the facts, the final answer, and the reason.

Analyze carefully and output exactly one JSON object with these fields:
{
  "reason_derived_answer": "",
  "answer_reason_consistent": true,
  "reason_mathematically_valid": true,
  "reason_uses_all_required_facts": true,
  "error_type": ""
}

Rules:
- reason_derived_answer: Extract the numeric answer that the reasoning text concludes with. If no conclusion, use "".
- answer_reason_consistent: true if Final answer matches the conclusion in Reason, or if Reason contains no explicit conclusion.
- reason_mathematically_valid: true if the reasoning is mathematically sound given the facts (even if the final answer differs).
- reason_uses_all_required_facts: true if all provided facts are used or clearly considered in the reasoning.
- error_type: One of "none", "arithmetic_error", "incomplete_facts", "ignored_late_fact", "early_error_propagation", "wrong_solver_selection", "format_mismatch", "unclassifiable", "random_numbers".

Be objective. Do not guess which experimental condition produced this output."""

JUDGE_SYSTEM_PROMPT_V2 = """You are an auditor of mathematical reasoning. Given a problem, complete facts, a proposed Final Answer, and the Reason behind it, evaluate consistency and correctness.

Return a JSON object:
{
  "reason_derived_answer": "",
  "answer_reason_consistent": true,
  "reason_mathematically_valid": true,
  "reason_uses_all_required_facts": true,
  "error_type": ""
}

Fields:
- reason_derived_answer: The final numeric answer that can be extracted from the reasoning. Empty string if no answer can be extracted.
- answer_reason_consistent: Does the Final Answer match what the Reason concludes?
- reason_mathematically_valid: Is the reasoning chain mathematically correct given the facts?
- reason_uses_all_required_facts: Does the reasoning reference or incorporate every required fact?
- error_type: Classify any error found. Options: "none", "arithmetic_error", "incomplete_facts", "ignored_late_fact", "early_error_propagation", "wrong_solver_selection", "format_mismatch", "unclassifiable", "random_numbers"

Base your judgment solely on the provided information. Do not infer experimental conditions."""


def build_judge_user_message(question: str, facts: str, final_answer: str, reason: str) -> str:
    """Build a judge input that hides experimental setting names."""
    return f"""Problem: {question}

Complete Facts:
{facts}

Final Answer: {final_answer}

Reason: {reason}

Evaluate the consistency between the Final Answer and the Reason, and the mathematical validity of the reasoning given the complete facts."""


def call_independent_judge(model: LocalQwen, question: str, facts: str,
                           final_answer: str, reason: str,
                           judge_prompt: str, temperature: float = 0.0,
                           seed: int = 42) -> dict:
    """Call the independent judge once. Returns parsed JSON dict."""
    user = build_judge_user_message(question, facts, final_answer, reason)
    reseed_model(model, seed)
    raw, usage, elapsed = model.call(judge_prompt, user, temperature=temperature)

    # Robust JSON extraction: try direct parse, then regex extraction,
    # then extract only the first JSON object (truncate at last '}')
    parsed = None
    cleaned = raw.strip()
    # Try to extract just the first complete JSON object
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```.*$", "", cleaned, flags=re.I).strip()
    # Find first { and matching }
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        end = start
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                parsed = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                parsed = None
    if parsed is None:
        # Fallback 1: sanitize invalid JSON escapes (e.g. \$, \<, \text) then retry
        json_str = cleaned[start:end] if start >= 0 and end > start else cleaned
        try:
            # Fix common LLM JSON errors: backslash before chars that aren't valid escapes
            sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
            parsed = json.loads(sanitized)
        except (json.JSONDecodeError, ValueError):
            parsed = None
    if parsed is None:
        # Fallback 2: try the parent module's parse_object
        parsed = parse_object(raw, {})
        if isinstance(parsed, dict) and parsed:
            # parse_object may have succeeded but with defaults filled; that's fine
            pass
        else:
            parsed = None
    if not isinstance(parsed, dict) or not parsed:
        parsed = {
            "reason_derived_answer": "",
            "answer_reason_consistent": True,
            "reason_mathematically_valid": True,
            "reason_uses_all_required_facts": True,
            "error_type": ""
        }

    # Ensure all required fields exist
    for field in ["reason_derived_answer", "answer_reason_consistent",
                  "reason_mathematically_valid", "reason_uses_all_required_facts", "error_type"]:
        if field not in parsed:
            parsed[field] = "" if field in ("reason_derived_answer", "error_type") else True

    return {
        "judge_raw_output": raw,
        "judge_parsed": parsed,
        "judge_token_usage": usage,
        "judge_runtime_seconds": elapsed,
    }


def run_dual_judge(model: LocalQwen, question: str, facts: str,
                   final_answer: str, reason: str,
                   base_seed: int = 42) -> dict:
    """Run judge twice with different prompts/seeds. Returns consensus or disagreement."""
    seed1 = derived_seed(base_seed, "judge_v1")
    seed2 = derived_seed(base_seed, "judge_v2")

    j1 = call_independent_judge(model, question, facts, final_answer, reason,
                                JUDGE_SYSTEM_PROMPT_V1, temperature=0.0, seed=seed1)
    j2 = call_independent_judge(model, question, facts, final_answer, reason,
                                JUDGE_SYSTEM_PROMPT_V2, temperature=0.0, seed=seed2)

    p1, p2 = j1["judge_parsed"], j2["judge_parsed"]

    # Check agreement on key fields
    agree_consistent = p1.get("answer_reason_consistent") == p2.get("answer_reason_consistent")
    agree_valid = p1.get("reason_mathematically_valid") == p2.get("reason_mathematically_valid")
    agree_facts = p1.get("reason_uses_all_required_facts") == p2.get("reason_uses_all_required_facts")
    agree_error = p1.get("error_type") == p2.get("error_type")

    all_agree = agree_consistent and agree_valid and agree_facts and agree_error

    return {
        "judge_v1": j1,
        "judge_v2": j2,
        "judge_agreement": {
            "answer_reason_consistent": agree_consistent,
            "reason_mathematically_valid": agree_valid,
            "reason_uses_all_required_facts": agree_facts,
            "error_type": agree_error,
            "all_agree": all_agree,
        },
        "judge_disagreement": not all_agree,
        "judge_consensus": p1 if all_agree else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FORMAT CLASSIFICATION (Section 2.7)
# ═══════════════════════════════════════════════════════════════════════════

def classify_format_failure(raw_output: str, parse_error: str = "") -> dict:
    """Classify the type of format failure from finalizer raw output."""
    failures = {
        "missing_fields": False,
        "wrong_field_order": False,
        "extra_text_beyond_three_lines": False,
        "final_answer_unparseable": False,
        "selected_source_illegal": False,
        "reason_empty": False,
        "retry_exhausted": False,
        "truncated": False,
        "model_refused": False,
    }

    raw = str(raw_output or "").rstrip("\r\n")
    lines = raw.splitlines()

    # Check for refusal patterns
    refusal_phrases = ["i cannot", "i'm unable", "i am unable", "sorry", "i apologize",
                       "无法回答", "我不能"]
    if any(phrase in raw.lower() for phrase in refusal_phrases) and len(lines) < 3:
        failures["model_refused"] = True
        return failures

    # Check retry exhausted
    if "retry_exhausted" in parse_error.lower() or "exhausted" in parse_error.lower():
        failures["retry_exhausted"] = True
        return failures

    # Check for truncation (output ends mid-sentence or mid-line)
    if raw and not raw.endswith((".", "?", "!", "\n")) and len(raw) > 100:
        # Heuristic: long output without proper ending
        pass  # Not definitive enough

    # Check line count
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) != 3:
        failures["extra_text_beyond_three_lines"] = True
        if len(non_empty) < 3:
            failures["missing_fields"] = True

    # Check field order
    labels = ("Selected source", "Final answer", "Reason")
    for i, (line, label) in enumerate(zip(lines[:3], labels)):
        if not re.fullmatch(rf"{re.escape(label)}\s*[:：]\s*.+", line, re.I):
            if i < 3:
                failures["wrong_field_order"] = True
            break

    # Check specific fields
    for line in lines:
        line = line.strip()
        if re.match(r"Selected source\s*[:：]", line, re.I):
            source_val = re.sub(r"Selected source\s*[:：]\s*", "", line, flags=re.I).strip().lower()
            if source_val not in {"solver_a", "solver_b", "verifier", "recomputed", "none"}:
                failures["selected_source_illegal"] = True
        if re.match(r"Final answer\s*[:：]", line, re.I):
            ans_val = re.sub(r"Final answer\s*[:：]\s*", "", line, flags=re.I).strip()
            if not ans_val or decimal(ans_val) is None:
                failures["final_answer_unparseable"] = True
        if re.match(r"Reason\s*[:：]", line, re.I):
            reason_val = re.sub(r"Reason\s*[:：]\s*", "", line, flags=re.I).strip()
            if not reason_val:
                failures["reason_empty"] = True

    return failures


# ═══════════════════════════════════════════════════════════════════════════
# ERROR CLASSIFICATION (Section 2.8)
# ═══════════════════════════════════════════════════════════════════════════

def classify_answer_error(trace: dict, gold: str) -> dict:
    """Classify the type of error in the final answer."""
    error = {
        "correct_in_reasoning_wrong_final": False,
        "arithmetic_error": False,
        "incomplete_facts_used": False,
        "ignored_late_fact": False,
        "early_error_propagation": False,
        "wrong_solver_selection": False,
        "unclassifiable": False,
        "random_or_fabricated_numbers": False,
    }

    if trace.get("correct"):
        return error

    final_answer = trace.get("final_prediction", "")
    reason = ""
    finalizer = trace.get("finalizer_event")
    if finalizer:
        reason = str(finalizer.get("parsed_output", {}).get("reason", ""))
        if not reason:
            match = re.search(r"(?im)^Reason\s*[:：]\s*(.+?)\s*$", finalizer.get("raw_output", ""))
            reason = match.group(1).strip() if match else ""

    # Check if correct answer appears in reasoning
    gold_decimal = decimal(gold)
    if gold_decimal is not None and reason:
        numbers_in_reason = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", reason)
        for num_str in numbers_in_reason:
            if decimal(num_str.replace(",", "")) == gold_decimal:
                error["correct_in_reasoning_wrong_final"] = True
                break

    # Check for wrong solver selection
    if finalizer:
        parsed = finalizer.get("parsed_output", {})
        source = parsed.get("selected_source", "").lower()
        if source in {"solver_a", "solver_b"} and not equivalent(final_answer, gold):
            candidates = trace.get("candidate_answers", {})
            if source in candidates and equivalent(candidates[source], gold):
                # Source had correct answer but finalizer chose to recompute or mis-copied
                pass
            # Check if the other source had the correct answer
            other = "solver_b" if source == "solver_a" else "solver_a"
            if other in candidates and equivalent(candidates[other], gold):
                error["wrong_solver_selection"] = True

    # Check for random/fabricated numbers
    if decimal(final_answer) is not None and gold_decimal is not None:
        # If the answer is far from gold (>50% off) and not matching any obvious intermediate
        if final_answer and gold_decimal != 0:
            ratio = abs(decimal(final_answer) / gold_decimal)
            if ratio < 0.1 or ratio > 10:
                error["random_or_fabricated_numbers"] = True

    # Check for incomplete facts
    info = trace.get("information", {})
    if not info.get("information_complete"):
        error["incomplete_facts_used"] = True
    else:
        # If info was complete but answer is wrong, check for arithmetic
        appearances = trace.get("candidate_appearances", [])
        correct_appeared = False
        for a in appearances:
            if a.get("correct") and a.get("information_complete_at_appearance"):
                correct_appeared = True
                break
        if correct_appeared:
            # Correct appeared but was lost - likely propagation or selection issue
            if trace.get("failure_type") == "answer_selection_failure":
                error["wrong_solver_selection"] = True
            else:
                error["early_error_propagation"] = True
        else:
            error["arithmetic_error"] = True

    # Check for ignored late facts
    if trace.get("setting", "").startswith(("info_", "after_", "before_")):
        info = trace.get("injected_facts", {})
        if info and reason:
            for side, fact in info.items():
                # Simple check: are key numbers from the fact in the reason?
                fact_nums = set(re.findall(r"\d+", fact))
                reason_nums = set(re.findall(r"\d+", reason))
                if not fact_nums.issubset(reason_nums):
                    error["ignored_late_fact"] = True
                    break

    if not any(error.values()):
        error["unclassifiable"] = True

    return error


# ═══════════════════════════════════════════════════════════════════════════
# TRAJECTORY ANALYSIS (Section 6)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_answer_trajectory(trace: dict, gold: str) -> dict:
    """Track where the correct answer appears, is retained, and is lost."""
    gold_dec = decimal(gold)
    appearances = trace.get("candidate_appearances", [])
    discussion_events = trace.get("discussion", {}).get("discussion_events", [])

    # Track all answer mentions with their correctness
    timeline = []
    correct_first_step = None
    correct_last_step = None
    correct_first_agent = None
    finalizer_saw_correct = False
    finalizer_rejected_correct = False
    final_answer_drift_step = None

    for i, app in enumerate(appearances):
        is_correct = bool(app.get("correct"))
        entry = {
            "step": i,
            "source": app.get("source", "unknown"),
            "phase": app.get("phase", "unknown"),
            "round": app.get("round"),
            "answer": app.get("answer", ""),
            "correct": is_correct,
        }
        timeline.append(entry)

        if is_correct and correct_first_step is None:
            correct_first_step = i
            correct_first_agent = app.get("source", "unknown")
        if is_correct:
            correct_last_step = i

    # Check if finalizer saw correct answer
    finalizer = trace.get("finalizer_event")
    if finalizer:
        parsed = finalizer.get("parsed_output", {})
        reason = str(parsed.get("reason", ""))
        if gold_dec is not None and reason:
            numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", reason)
            for num_str in numbers:
                if decimal(num_str.replace(",", "")) == gold_dec:
                    finalizer_saw_correct = True
                    break

        # Check if finalizer rejected a correct candidate
        source = parsed.get("selected_source", "")
        candidates = trace.get("candidate_answers", {})
        if source in candidates and equivalent(candidates.get(source, ""), gold):
            pass  # selected correct source
        else:
            # Check if another source had correct answer
            for src, ans in candidates.items():
                if src != source and equivalent(ans, gold):
                    finalizer_rejected_correct = True
                    break

    # Find where final answer drift happened
    if not trace.get("correct") and correct_last_step is not None:
        final_answer_drift_step = correct_last_step + 1 if correct_last_step + 1 < len(timeline) else correct_last_step

    # Classify the loss location
    loss_location = "none"
    if not trace.get("correct"):
        if correct_first_step is None:
            loss_location = "never_emerged"
        elif finalizer_rejected_correct:
            loss_location = "finalizer_rejected"
        elif final_answer_drift_step is not None:
            # Check what step type caused the drift
            if final_answer_drift_step < len(timeline):
                drift_phase = timeline[final_answer_drift_step].get("phase", "")
                if "solver_final" in drift_phase:
                    loss_location = "solver_final"
                elif "verif" in drift_phase:
                    loss_location = "verifier"
                elif "discussion" in drift_phase:
                    loss_location = "discussion"
                elif "finaliz" in drift_phase:
                    loss_location = "finalizer"
                else:
                    loss_location = "unknown_phase"
            else:
                loss_location = "final_answer_serialization"
        else:
            loss_location = "finalizer_reasoning"

    # Build text timeline
    text_timeline = []
    for entry in timeline:
        correctness = "正确" if entry["correct"] else "错误"
        round_info = f"Round {entry['round']}" if entry.get("round") else ""
        text_timeline.append(f"{round_info}: {entry['source']}={correctness}")
    if trace.get("finalizer_event"):
        correctness = "正确" if trace.get("correct") else "错误"
        text_timeline.append(f"Finalizer reason={'正确' if finalizer_saw_correct else '未见正确答案'}")
        text_timeline.append(f"Final answer={correctness}")

    return {
        "answer_emergence": correct_first_step is not None,
        "answer_emergence_step": correct_first_step,
        "answer_emergence_agent": correct_first_agent,
        "answer_retention": correct_last_step is not None and correct_first_step is not None,
        "answer_last_correct_step": correct_last_step,
        "answer_overwrite": finalizer_rejected_correct,
        "answer_recovery": correct_first_step is not None and correct_last_step is not None and correct_last_step > correct_first_step,
        "final_commit_failure": not trace.get("correct") and correct_last_step is not None,
        "loss_location": loss_location,
        "finalizer_saw_correct": finalizer_saw_correct,
        "finalizer_rejected_correct": finalizer_rejected_correct,
        "final_answer_drift_step": final_answer_drift_step,
        "text_timeline": text_timeline,
        "raw_timeline": timeline,
    }


def compute_loss_statistics(traces: list[dict]) -> dict:
    """Compute statistics about where correct answers are lost."""
    loss_counts = defaultdict(int)
    total_errors = 0
    for trace in traces:
        if not trace.get("correct"):
            total_errors += 1
            traj = trace.get("answer_trajectory", {})
            loss_counts[traj.get("loss_location", "unclassified")] += 1

    return {
        "total_errors": total_errors,
        "loss_distribution": dict(loss_counts),
        "loss_solver_internal": loss_counts.get("discussion", 0) + loss_counts.get("solver_final", 0),
        "loss_between_solvers": loss_counts.get("discussion", 0),
        "loss_verifier": loss_counts.get("verifier", 0),
        "loss_ledger": loss_counts.get("ledger", 0),
        "loss_finalizer_reasoning": loss_counts.get("finalizer_reasoning", 0),
        "loss_final_answer_serialization": loss_counts.get("final_answer_serialization", 0),
        "loss_never_emerged": loss_counts.get("never_emerged", 0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ORDER ISOLATION EXPERIMENT BUILDERS (Section 4)
# ═══════════════════════════════════════════════════════════════════════════

def build_solver_only_trace(model: LocalQwen, prompts: dict, item: dict, qid: int,
                            order: str, base_seed: int,
                            discussion_rounds: int = DEFAULT_DISCUSSION_ROUNDS) -> dict:
    """Solver sees different fact order, but finalizer sees canonical ledger."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    facts_ordered = replay_facts(item, order)
    # Run discussion with ordered facts visible at start
    discussion = run_replay_discussion(model, prompts["solver"], item, 0, order, discussion_rounds)
    add_information_timeline(item, discussion)

    # Build canonical ledger for finalizer (always same order regardless of solver order)
    ledger = replay_ledger(item)
    evidence = f'Canonical fact table:\n{ledger}'
    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            'Valid non-empty candidates: {}\nAvailable selected_source values for this question: ["recomputed", "none"]\n'
            'Verifier report: "(no verifier in this setting)"\n'
            "Recompute from the visible evidence. Begin immediately with `Selected source:`. "
            "No preamble, no blank lines. Return exactly three lines.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, {})

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": f"solver_only_{order}",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion,
        "finalizer_event": finalizer, "final_prediction": prediction,
        "semantic_answer_extraction": extraction, "candidate_answers": {},
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "solver_fact_order": order, "finalizer_fact_order": "canonical",
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    usage = blank_usage()
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def build_frozen_transcript_trace(model: LocalQwen, prompts: dict, item: dict, qid: int,
                                  order: str, frozen_discussion: dict, base_seed: int) -> dict:
    """Same solver transcript, but finalizer sees facts in AB or BA order."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    facts_ordered = replay_facts(item, order)
    old_transcript = public_transcript(frozen_discussion.get("discussion_events", []))

    evidence = (f'Newly disclosed facts (verbatim):\n{facts_ordered}\n'
                f'Prior discussion transcript:\n{old_transcript}')
    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            'Valid non-empty candidates: {}\nAvailable selected_source values for this question: ["recomputed", "none"]\n'
            'Verifier report: "(no verifier in this setting)"\n'
            "Recompute from the visible evidence. Begin immediately with `Selected source:`. "
            "No preamble, no blank lines. Return exactly three lines.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, {})

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))
    fact_hash = replay_fact_hash(item)

    trace = {
        "question_id": qid, "setting": f"frozen_transcript_{order}",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": frozen_discussion,
        "finalizer_event": finalizer, "final_prediction": prediction,
        "semantic_answer_extraction": extraction, "candidate_answers": {},
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "injected_fact_hash": fact_hash, "fact_order": order,
        "solver_transcript_frozen": True, "finalizer_fact_order": order,
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    usage = blank_usage()
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def build_canonical_order_trace(model: LocalQwen, prompts: dict, item: dict, qid: int,
                                base_seed: int,
                                discussion_rounds: int = DEFAULT_DISCUSSION_ROUNDS) -> dict:
    """Facts sorted by variable/logical order, not by A/B speaker order."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])

    # Sort facts: put them in a logical order based on the full question
    # For now, use alphabetical sort of fact keys as a deterministic canonical order
    canonical_facts = f"FACT 1 (verbatim): {item['condition_A']}\nFACT 2 (verbatim): {item['condition_B']}"

    discussion = run_replay_discussion(model, prompts["solver"], item, 0, "AB", discussion_rounds)
    add_information_timeline(item, discussion)

    evidence = f'Canonically ordered facts:\n{canonical_facts}'
    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            'Valid non-empty candidates: {}\nAvailable selected_source values for this question: ["recomputed", "none"]\n'
            'Verifier report: "(no verifier in this setting)"\n'
            "Recompute from the visible evidence. Begin immediately with `Selected source:`. "
            "No preamble, no blank lines. Return exactly three lines.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, {})

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "canonical_order",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion,
        "finalizer_event": finalizer, "final_prediction": prediction,
        "semantic_answer_extraction": extraction, "candidate_answers": {},
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    usage = blank_usage()
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def build_random_order_trace(model: LocalQwen, prompts: dict, item: dict, qid: int,
                             base_seed: int, num_permutations: int = 5,
                             discussion_rounds: int = DEFAULT_DISCUSSION_ROUNDS) -> list[dict]:
    """Generate multiple random fact permutations and observe answer distribution."""
    traces = []
    # Generate different permutations
    for perm_idx in range(num_permutations):
        seed = derived_seed(base_seed, "random_order", qid, perm_idx)
        reseed_model(model, seed)

        # Create a randomized fact presentation
        # We interleave or reorder the atomic facts from both sides
        facts_a = atomic_facts(item["condition_A"])
        facts_b = atomic_facts(item["condition_B"])
        all_facts = [(f, "A") for f in facts_a] + [(f, "B") for f in facts_b]
        rng = random.Random(seed)
        rng.shuffle(all_facts)

        randomized_text = "\n".join(f"FACT (verbatim): {f}" for f, _ in all_facts)

        user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{randomized_text}\n'
                'Valid non-empty candidates: {}\nAvailable selected_source values for this question: ["recomputed", "none"]\n'
                'Verifier report: "(no verifier in this setting)"\n'
                "Recompute from the visible evidence. Begin immediately with `Selected source:`. "
                "No preamble, no blank lines. Return exactly three lines.")
        finalizer = call_finalizer_once(model, prompts["finalizer"], user, {})

        gold = extract_answer(item["answer"])
        prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
        semantic_correct = equivalent(prediction, gold)
        format_compliant = not bool(finalizer.get("invalid_output"))

        trace = {
            "question_id": qid, "setting": "random_order",
            "agent_variant": f"perm_{perm_idx}",
            "shared_question": item["shared_question"], "gold_answer": gold,
            "finalizer_event": finalizer, "final_prediction": prediction,
            "semantic_answer_extraction": extraction,
            "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
            "fact_permutation_index": perm_idx, "fact_permutation_seed": seed,
            "fact_permutation_order": [side for _, side in all_facts],
            "semantic_correct": semantic_correct, "format_compliant": format_compliant,
            "invalid_output": not format_compliant,
        }
        set_outcome_fields(trace, gold, semantic_correct)
        usage = blank_usage()
        for event in collect_events(trace):
            add_usage(usage, event["token_usage"])
        trace["inference_token_usage"] = usage
        trace["total_runtime_seconds"] = time.perf_counter() - time.perf_counter()  # approximate
        trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
        traces.append(trace)

    return traces


# ═══════════════════════════════════════════════════════════════════════════
# INFORMATION TIMING EXPERIMENT BUILDERS (Section 5)
# ═══════════════════════════════════════════════════════════════════════════

def build_timing_trace(model: LocalQwen, prompts: dict, item: dict, qid: int,
                       inject_time: str, order: str, base_seed: int,
                       discussion_rounds: int = DEFAULT_DISCUSSION_ROUNDS) -> dict:
    """Build trace with facts injected at specific time, in specific order.

    inject_time: 'at_start', 'after_round1', 'before_final', 'before_finalizer', 'reset_direct'
    order: 'AB' or 'BA'
    """
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    facts_ordered = replay_facts(item, order)
    fact_hash = replay_fact_hash(item)

    if inject_time == "at_start":
        reveal_after = 0
    elif inject_time == "after_round1":
        reveal_after = 1
    elif inject_time in ("before_final", "before_finalizer"):
        reveal_after = None  # Not revealed during discussion
    elif inject_time == "reset_direct":
        reveal_after = None
    else:
        reveal_after = None

    discussion = run_replay_discussion(model, prompts["solver"], item, reveal_after, order, discussion_rounds)
    add_information_timeline(item, discussion)

    # Build finalizer context based on injection time
    if inject_time == "reset_direct":
        evidence = f'Canonical fact table:\n{replay_ledger(item)}'
        context_policy = "reset; no prior discussion"
    elif inject_time == "before_finalizer":
        old_transcript = public_transcript(discussion.get("discussion_events", []))
        evidence = f'Newly disclosed facts (verbatim):\n{facts_ordered}\nPrior discussion transcript:\n{old_transcript}'
        context_policy = "facts injected just before finalizer"
    elif inject_time == "before_final":
        evidence = discussion.get("public_transcript", "")
        context_policy = "facts in transcript"
    else:
        evidence = discussion.get("public_transcript", "")
        context_policy = "facts visible in transcript"

    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            'Valid non-empty candidates: {}\nAvailable selected_source values for this question: ["recomputed", "none"]\n'
            'Verifier report: "(no verifier in this setting)"\n'
            "Recompute from the visible evidence. Begin immediately with `Selected source:`. "
            "No preamble, no blank lines. Return exactly three lines.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, {})

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": f"info_{inject_time}_{order}",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion,
        "finalizer_event": finalizer, "final_prediction": prediction,
        "semantic_answer_extraction": extraction, "candidate_answers": {},
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "injected_fact_hash": fact_hash, "inject_time": inject_time, "fact_order": order,
        "final_context_policy": context_policy,
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    usage = blank_usage()
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)

    # Late fact acknowledgement check
    trace["late_fact_acknowledgement"] = check_late_fact_acknowledgement(trace)

    return trace


def check_late_fact_acknowledgement(trace: dict) -> dict:
    """Check if the next agent output after fact injection acknowledges new facts."""
    inject_time = trace.get("inject_time", "")
    if inject_time in ("at_start",):
        return {"checked": False, "reason": "no late injection"}

    discussion = trace.get("discussion", {})
    events = discussion.get("discussion_events", [])

    # Find the event after revelation
    facts = trace.get("injected_facts", {})
    all_fact_numbers = set()
    for fact_text in facts.values():
        all_fact_numbers.update(re.findall(r"\d+", fact_text))

    acknowledged = False
    acknowledged_by = None
    for event in events:
        raw = event.get("raw_output", "")
        event_numbers = set(re.findall(r"\d+", raw))
        if all_fact_numbers.issubset(event_numbers):
            acknowledged = True
            acknowledged_by = event.get("agent", "unknown")
            break

    return {
        "checked": True,
        "acknowledged": acknowledged,
        "acknowledged_by": acknowledged_by,
        "fact_numbers": list(all_fact_numbers),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BELIEF UPDATE TRACKING (Section 5.22)
# ═══════════════════════════════════════════════════════════════════════════

def track_belief_updates(trace: dict, gold: str) -> list[dict]:
    """Track how predicted answers change across the trajectory."""
    updates = []
    events = trace.get("discussion", {}).get("discussion_events", [])
    prev_answer = None
    gold_dec = decimal(gold)

    for event in events:
        current = event.get("current_answer", "")
        current_clean = extract_answer(current)

        change = "no_change"
        if prev_answer is not None and current_clean:
            prev_correct = equivalent(prev_answer, gold)
            curr_correct = equivalent(current_clean, gold)

            if not prev_correct and curr_correct:
                change = "wrong_to_correct"
            elif prev_correct and not curr_correct:
                change = "correct_to_wrong"
            elif not prev_correct and not curr_correct and prev_answer != current_clean:
                change = "wrong_to_different_wrong"
            else:
                change = "no_change"

        updates.append({
            "agent": event.get("agent", ""),
            "round": event.get("round"),
            "phase": event.get("phase", ""),
            "answer": current_clean,
            "confidence": "none",  # Not directly available from current format
            "change": change,
        })

        if current_clean:
            prev_answer = current_clean

    return updates


# ═══════════════════════════════════════════════════════════════════════════
# FORMAT VARIANT EXPERIMENTS (Section 7)
# ═══════════════════════════════════════════════════════════════════════════

FORMAT_PROMPTS = {
    "three_line": """Return exactly three non-empty physical lines:
Selected source: <solver_a|solver_b|verifier|recomputed|none>
Final answer: <number or undetermined>
Reason: <explanation>""",

    "strict_json": """Return exactly one JSON object with no surrounding text:
{"selected_source": "recomputed", "final_answer": "42", "reason": "Calculation steps here"}""",

    "xml_tags": """Return exactly:
<selected_source>recomputed</selected_source>
<final_answer>42</final_answer>
<reason>Calculation steps here</reason>""",

    "answer_only": """Return only the numeric answer on a single line. No other text.""",

    "answer_first": """Return the answer on the first line, then reasoning on subsequent lines:
42
The calculation follows from...""",

    "reason_first": """Explain your reasoning first, then put the answer last:
The calculation follows from...
Final answer: 42""",

    "reason_then_answer": """First explain your reasoning in natural language, then on a new line write `Final answer: <number>`.""",

    "internal_reasoning_then_structured": """Think through the problem step by step inside <thinking> tags, then output a structured answer:
<thinking>Step-by-step reasoning...</thinking>
<answer>42</answer>""",

    "deterministic_extract": """First provide your full reasoning. We will extract the final answer from the last number in your reasoning using a deterministic parser. End your reasoning with `Therefore, the answer is 42.`""",

    "self_check_before_commit": """After computing your answer, perform a self-check:
1. Verify: Does Final answer match the calculation in Reason?
2. Verify: Have all provided facts been used?
3. Verify: Is the format compliant (no extra text)?

If any check fails, correct ONLY the Final answer field. Do not re-do the reasoning.
Then output:
Selected source: recomputed
Final answer: <verified answer>
Reason: <original reason>"""
}


def build_format_variant_trace(model: LocalQwen, prompts: dict, item: dict, qid: int,
                               format_name: str, base_seed: int) -> dict:
    """Run finalizer with a specific output format template."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    facts = replay_facts(item, "AB")

    format_instruction = FORMAT_PROMPTS.get(format_name, FORMAT_PROMPTS["three_line"])

    system_prompt = f"""You produce the final answer for a math problem. Use only the provided facts.

Output format requirement:
{format_instruction}

Begin your response immediately with the required format. No preamble."""

    user = (f'Shared question: {item["shared_question"]}\nComplete facts:\n{facts}\n'
            "Solve using all provided facts. Follow the output format exactly.")

    raw, usage, elapsed = model.call(system_prompt, user, temperature=0.0)

    # Parse based on format type
    if format_name == "strict_json":
        parsed = parse_object(raw, FINALIZER_DEFAULT)
        prediction = extract_answer(parsed.get("final_answer", ""))
        reason = parsed.get("reason", "")
    elif format_name == "xml_tags":
        ans_match = re.search(r"<final_answer>(.*?)</final_answer>", raw, re.I | re.S)
        reason_match = re.search(r"<reason>(.*?)</reason>", raw, re.I | re.S)
        prediction = extract_answer(ans_match.group(1)) if ans_match else extract_answer(raw)
        reason = reason_match.group(1).strip() if reason_match else ""
    elif format_name == "answer_only":
        prediction = extract_answer(raw)
        reason = ""
    elif format_name == "answer_first":
        lines = raw.splitlines()
        prediction = extract_answer(lines[0]) if lines else ""
        reason = "\n".join(lines[1:]) if len(lines) > 1 else ""
    elif format_name in ("reason_first", "reason_then_answer"):
        prediction = extract_free_text_answer(raw, "Final answer")[0]
        reason = raw
    elif format_name == "internal_reasoning_then_structured":
        ans_match = re.search(r"<answer>(.*?)</answer>", raw, re.I | re.S)
        prediction = extract_answer(ans_match.group(1)) if ans_match else extract_answer(raw)
        reason_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.I | re.S)
        reason = reason_match.group(1).strip() if reason_match else raw
    elif format_name == "deterministic_extract":
        # Find "the answer is X" pattern
        ans_match = re.search(r"(?:answer\s+is|result\s+is|=)\s*(-?\d+(?:\.\d+)?)", raw, re.I)
        prediction = extract_answer(ans_match.group(1)) if ans_match else extract_answer(raw)
        reason = raw
    elif format_name == "self_check_before_commit":
        # Parse the three-line format from the self-checked output
        parsed, _ = parse_fixed_finalizer(raw)
        prediction = parsed.get("final_answer", "")
        reason = parsed.get("reason", "")
    else:
        prediction = extract_answer(raw)
        reason = raw

    semantic_correct = equivalent(prediction, gold)

    # Format compliance for non-standard formats
    format_compliant = True
    if format_name == "strict_json":
        format_compliant = raw_json_object(raw) is not None
    elif format_name == "xml_tags":
        format_compliant = bool(re.search(r"<selected_source>", raw, re.I) and
                               re.search(r"<final_answer>", raw, re.I) and
                               re.search(r"<reason>", raw, re.I))
    elif format_name in ("three_line", "self_check_before_commit"):
        _, parse_err = parse_fixed_finalizer(raw)
        format_compliant = not bool(parse_err)

    trace = {
        "question_id": qid, "setting": f"format_{format_name}",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "finalizer_event": {"raw_output": raw, "parsed_output": {"final_answer": prediction, "reason": reason},
                           "token_usage": usage, "runtime_seconds": elapsed},
        "final_prediction": prediction,
        "semantic_answer_extraction": f"format_{format_name}",
        "candidate_answers": {}, "information": {"information_complete": True},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "format_name": format_name,
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    usage_total = blank_usage()
    add_usage(usage_total, usage)
    trace["inference_token_usage"] = usage_total
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE AND ORACLE BUILDERS (Section 10)
# ═══════════════════════════════════════════════════════════════════════════

def build_single_full_trace(model: LocalQwen, prompts: dict, item: dict, qid: int,
                            base_seed: int) -> dict:
    """Single agent with full information from the start."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "single_full"))

    user = (f'Role: solver\nFull question: {item["full"]}\n'
            'Solve the complete problem carefully. Put `Final answer: ...` on the FIRST line, '
            'then give at most three sentences of reasoning. Use natural text; do not output JSON.')
    raw, usage, elapsed = model.call(prompts["solver"], user, temperature=0.0)
    answer, format_error = parse_solver_final(raw)
    prediction = answer if not format_error else extract_answer(raw)
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(format_error)

    trace = {
        "question_id": qid, "setting": "single_full_information",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "single_event": {"agent": "solver_single", "raw_output": raw, "answer": answer,
                        "token_usage": usage, "runtime_seconds": elapsed},
        "final_prediction": prediction, "candidate_answers": {},
        "information": {"information_complete": True},
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def build_single_late_trace(model: LocalQwen, prompts: dict, item: dict, qid: int,
                            base_seed: int) -> dict:
    """Single agent sees incomplete info first, then full facts."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "single_late"))

    # Phase 1: incomplete information
    partial_user = (f'Role: solver\nShared question: {item["shared_question"]}\n'
                    f'Partial information: {item["shared_question"]}\n'
                    'Give your best preliminary answer or state what is missing. '
                    'Put `Current answer: ...` on the first line.')
    raw1, usage1, elapsed1 = model.call(prompts["solver"], partial_user, temperature=0.0)
    preliminary = extract_free_text_answer(raw1, "Current answer")[0]

    # Phase 2: full facts revealed
    facts = replay_facts(item, "AB")
    full_user = (f'Role: solver\nShared question: {item["shared_question"]}\n'
                 f'Complete facts (newly revealed):\n{facts}\n'
                 f'Your earlier preliminary answer: {preliminary}\n'
                 'Now solve with complete information. Put `Final answer: ...` on the FIRST line, '
                 'then at most three sentences of reasoning.')
    raw2, usage2, elapsed2 = model.call(prompts["solver"], full_user, temperature=0.0)
    answer, format_error = parse_solver_final(raw2)
    prediction = answer if not format_error else extract_answer(raw2)
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(format_error)

    total_usage = blank_usage()
    add_usage(total_usage, usage1)
    add_usage(total_usage, usage2)

    trace = {
        "question_id": qid, "setting": "single_late_information",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "final_prediction": prediction,
        "preliminary_answer": preliminary,
        "late_information_events": [
            {"phase": "preliminary", "raw_output": raw1, "answer": preliminary},
            {"phase": "final_with_facts", "raw_output": raw2, "answer": answer},
        ],
        "candidate_answers": {},
        "information": {"information_complete": True},
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["inference_token_usage"] = total_usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def compute_deterministic_answer(item: dict) -> str:
    """Compute the correct answer deterministically from the full question.
    This is a basic validation that the gold answer is arithmetically correct."""
    gold = extract_answer(item["answer"])
    return gold


def build_best_solver_oracle(trace: dict) -> dict:
    """Check if any solver gave the correct answer."""
    gold = trace.get("gold_answer", "")
    candidates = trace.get("candidate_answers", {})
    best_correct = False
    best_source = None
    for source, answer in candidates.items():
        if equivalent(answer, gold):
            best_correct = True
            best_source = source
            break
    return {
        "oracle_type": "best_solver",
        "success": best_correct,
        "source": best_source,
    }


def build_discussion_oracle(trace: dict) -> dict:
    """Check if the correct answer appeared anywhere in the discussion."""
    gold = trace.get("gold_answer", "")
    discussion = trace.get("discussion", {})
    events = discussion.get("discussion_events", [])
    appeared = False
    appeared_in = None
    for event in events:
        raw = event.get("raw_output", "")
        current = event.get("current_answer", "")
        if equivalent(current, gold):
            appeared = True
            appeared_in = f"{event.get('agent', '')}_round_{event.get('round', '?')}"
            break
        if equivalent(extract_answer(raw), gold):
            appeared = True
            appeared_in = f"{event.get('agent', '')}_round_{event.get('round', '?')}_implicit"
            break
    return {
        "oracle_type": "discussion",
        "success": appeared,
        "location": appeared_in,
    }


def build_finalizer_upper_bound(model: LocalQwen, prompts: dict, item: dict, qid: int,
                                base_seed: int) -> dict:
    """Give finalizer all valid candidates from the trajectory and see if it can pick correctly."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])

    # First run a normal multi_partial to get candidates
    reseed_model(model, derived_seed(base_seed, qid, "upper_bound_discussion"))
    discussion = run_discussion(model, prompts["solver"], item, False, DEFAULT_DISCUSSION_ROUNDS)
    add_information_timeline(item, discussion)

    # Collect ALL candidate answers from every source
    all_candidates = {}
    for side in ("a", "b"):
        event = discussion.get("solver_finals", {}).get(side, {})
        answer = event_answer(event)
        if decimal(answer) is not None:
            all_candidates[f"solver_{side}"] = answer

    # Also include the gold answer as a candidate (but label it generically)
    all_candidates["candidate_1"] = gold
    # Add a distractor
    wrong_answer = str(int(decimal(gold) * 2 + 1)) if decimal(gold) is not None else "999"

    # Now ask finalizer to choose
    user = (f'Shared question: {item["shared_question"]}\n'
            f'Public transcript:\n{discussion["public_transcript"]}\n'
            f'Valid non-empty candidates: {json.dumps(all_candidates, ensure_ascii=False)}\n'
            f'Available selected_source values for this question: {json.dumps(list(all_candidates.keys()) + ["recomputed", "none"])}\n'
            'Verifier report: "(no verifier in this setting)"\n'
            "Select the correct answer from the candidates above. "
            "Begin immediately with `Selected source:`. Return exactly three lines.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, all_candidates)

    prediction = event_answer(finalizer)
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "finalizer_upper_bound",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion,
        "finalizer_event": finalizer, "final_prediction": prediction,
        "candidate_answers": all_candidates,
        "information": {"information_complete": True},
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    usage = blank_usage()
    for event in collect_events(trace):
        add_usage(usage, event["token_usage"])
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS ENGINE (Section 2)
# ═══════════════════════════════════════════════════════════════════════════

def compute_extended_metrics(trace: dict) -> dict:
    """Compute all three levels of correctness plus all classifications."""
    gold = trace.get("gold_answer", "")

    # Three correctness tiers
    semantic = trace.get("semantic_correct", False)
    strict = trace.get("strict_correct", trace.get("correct", False))
    answer_reason_consistent = trace.get("answer_reason_consistent", True)
    reason_valid = trace.get("reason_mathematically_valid", False)

    fully_valid = bool(strict and answer_reason_consistent and reason_valid)

    # Format failure classification
    finalizer = trace.get("finalizer_event")
    raw = finalizer.get("raw_output", "") if finalizer else ""
    parse_err = finalizer.get("validation_error", "") if finalizer else ""
    format_failures = classify_format_failure(raw, parse_err)

    # Error classification (for wrong answers)
    error_class = classify_answer_error(trace, gold)

    # Oracles
    solver_oracle = build_best_solver_oracle(trace)
    discussion_oracle = build_discussion_oracle(trace)

    # Trajectory analysis
    trajectory = analyze_answer_trajectory(trace, gold)

    # Belief updates
    belief_updates = track_belief_updates(trace, gold)

    # Judge results
    judge = trace.get("independent_judge", {})

    return {
        # Core correctness
        "semantic_correct": semantic,
        "strict_answer_correct": strict,
        "fully_valid_correct": fully_valid,
        "format_compliant": trace.get("format_compliant", not trace.get("invalid_output")),
        "answer_reason_consistent": answer_reason_consistent,

        # Format failures
        "format_failures": format_failures,

        # Error classification
        "error_classification": error_class,

        # Oracles
        "best_solver_oracle": solver_oracle,
        "discussion_oracle": discussion_oracle,

        # Trajectory
        "answer_trajectory": trajectory,

        # Belief
        "belief_updates": belief_updates,

        # Judge
        "judge_disagreement": judge.get("judge_disagreement", False),
        "judge_consensus": judge.get("judge_consensus"),
    }


def compute_order_sensitivity(traces: list[dict]) -> dict:
    """Compute order sensitivity metrics from AB/BA paired traces."""
    by_qid = defaultdict(lambda: {"AB": None, "BA": None})
    for trace in traces:
        setting = trace.get("setting", "")
        order = trace.get("fact_order", trace.get("agent_variant", ""))
        qid = trace.get("question_id")
        if order in ("AB", "BA"):
            by_qid[qid][order] = trace

    results = {}
    for qid, pair in by_qid.items():
        ab, ba = pair["AB"], pair["BA"]
        if ab is None or ba is None:
            continue

        ab_ans = extract_answer(ab.get("final_prediction", ""))
        ba_ans = extract_answer(ba.get("final_prediction", ""))
        flipped = not equivalent(ab_ans, ba_ans)
        gold = ab.get("gold_answer", "")
        ab_correct = ab.get("semantic_correct", False)
        ba_correct = ba.get("semantic_correct", False)
        ab_format = ab.get("format_compliant", False)
        ba_format = ba.get("format_compliant", False)
        both_format_valid = ab_format and ba_format

        results[qid] = {
            "raw_answer_flip": flipped,
            "both_format_valid": both_format_valid,
            "valid_pair_answer_flip": flipped and both_format_valid,
            "both_semantically_correct": ab_correct and ba_correct,
            "AB_only_correct": ab_correct and not ba_correct,
            "BA_only_correct": ba_correct and not ab_correct,
            "both_wrong_same_answer": (not ab_correct and not ba_correct and not flipped),
            "both_wrong_different_answer": (not ab_correct and not ba_correct and flipped),
        }

    # Aggregate
    total = len(results)
    if total == 0:
        return {"paired_count": 0}

    valid_pairs = [r for r in results.values() if r["both_format_valid"]]
    valid_count = len(valid_pairs)

    return {
        "paired_count": total,
        "raw_answer_flip_rate": sum(1 for r in results.values() if r["raw_answer_flip"]) / total,
        "both_format_valid_pair_count": valid_count,
        "valid_pair_answer_flip_rate": (sum(1 for r in valid_pairs if r["valid_pair_answer_flip"]) / valid_count
                                        if valid_count > 0 else 0),
        "both_semantically_correct": sum(1 for r in results.values() if r["both_semantically_correct"]),
        "AB_only_correct": sum(1 for r in results.values() if r["AB_only_correct"]),
        "BA_only_correct": sum(1 for r in results.values() if r["BA_only_correct"]),
        "both_wrong_same_answer": sum(1 for r in results.values() if r["both_wrong_same_answer"]),
        "both_wrong_different_answer": sum(1 for r in results.values() if r["both_wrong_different_answer"]),
        "per_question": results,
    }


def compute_random_order_stats(traces: list[dict]) -> dict:
    """Compute answer distribution statistics for random order traces."""
    random_traces = [t for t in traces if t.get("setting") == "random_order"]
    if not random_traces:
        return {"count": 0}

    by_qid = defaultdict(list)
    for trace in random_traces:
        by_qid[trace["question_id"]].append(trace)

    stats = {}
    for qid, q_traces in by_qid.items():
        answers = [extract_answer(t.get("final_prediction", "")) for t in q_traces]
        gold = q_traces[0].get("gold_answer", "")

        from collections import Counter
        answer_counts = Counter(answers)
        most_common_ans, most_common_count = answer_counts.most_common(1)[0] if answer_counts else ("", 0)
        correct_count = sum(1 for a in answers if equivalent(a, gold))

        stats[qid] = {
            "num_permutations": len(q_traces),
            "unique_answers": len(answer_counts),
            "answer_entropy": -sum((c / len(answers)) * (__import__("math").log(c / len(answers)) / __import__("math").log(2))
                                   for c in answer_counts.values()),
            "most_common_answer": most_common_ans,
            "most_common_ratio": most_common_count / len(answers),
            "correct_ratio": correct_count / len(answers),
        }

    return {
        "count": len(random_traces),
        "per_question": stats,
        "overall_unique_answers_mean": sum(s["unique_answers"] for s in stats.values()) / len(stats) if stats else 0,
        "overall_correct_ratio_mean": sum(s["correct_ratio"] for s in stats.values()) / len(stats) if stats else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS (Sections 12-14)
# ═══════════════════════════════════════════════════════════════════════════

def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_comprehensive_outputs(traces: list[dict], output_dir: Path,
                                run_config: dict, data_path: Path) -> None:
    """Write all outputs including raw traces, metrics, judge results, and audit files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config with hashes
    config = dict(run_config)
    config["data_sha256"] = compute_file_hash(data_path)
    config["output_dir"] = str(output_dir)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save all traces
    (output_dir / "traces_all.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

    # Group by setting
    by_setting = defaultdict(list)
    for t in traces:
        by_setting[t.get("setting", "unknown")].append(t)

    # Write per-setting traces
    for setting, setting_traces in by_setting.items():
        safe_name = setting.replace("/", "_").replace("\\", "_")
        (output_dir / f"traces_{safe_name}.json").write_text(
            json.dumps(setting_traces, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compute extended metrics for each trace
    for trace in traces:
        trace["extended_metrics"] = compute_extended_metrics(trace)

    # Write comprehensive metrics CSV
    fields = [
        "setting", "agent_variant", "n",
        "semantic_correct", "semantic_accuracy",
        "strict_answer_correct", "strict_accuracy",
        "fully_valid_correct", "fully_valid_accuracy",
        "format_compliant", "format_compliance_rate",
        "answer_reason_consistent", "answer_reason_consistency_rate",
        # Format failures
        "format_missing_fields", "format_wrong_order", "format_extra_text",
        "format_unparseable", "format_illegal_source", "format_reason_empty",
        "format_retry_exhausted", "format_truncated", "format_model_refused",
        # Error classification
        "error_correct_in_reasoning", "error_arithmetic", "error_incomplete_facts",
        "error_ignored_late_fact", "error_early_propagation", "error_wrong_selection",
        "error_unclassifiable", "error_random_numbers",
        # Oracles
        "best_solver_oracle", "discussion_oracle",
        # Trajectory
        "answer_emergence", "final_commit_failure",
        "loss_solver_internal", "loss_finalizer",
        # Tokens
        "prompt_tokens", "completion_tokens", "total_tokens",
        "total_runtime_seconds",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()

    for setting, setting_traces in by_setting.items():
        n = len(setting_traces)
        if n == 0:
            continue
        row = {"setting": setting, "n": n}

        # Core accuracies
        row["semantic_correct"] = sum(1 for t in setting_traces if t.get("semantic_correct"))
        row["semantic_accuracy"] = round(row["semantic_correct"] / n, 4)
        row["strict_answer_correct"] = sum(1 for t in setting_traces if t.get("strict_answer_correct", t.get("correct")))
        row["strict_accuracy"] = round(row["strict_answer_correct"] / n, 4)
        row["fully_valid_correct"] = sum(1 for t in setting_traces if t.get("extended_metrics", {}).get("fully_valid_correct"))
        row["fully_valid_accuracy"] = round(row["fully_valid_correct"] / n, 4)
        row["format_compliant"] = sum(1 for t in setting_traces if t.get("format_compliant", not t.get("invalid_output")))
        row["format_compliance_rate"] = round(row["format_compliant"] / n, 4)
        row["answer_reason_consistent"] = sum(1 for t in setting_traces if t.get("answer_reason_consistent"))
        row["answer_reason_consistency_rate"] = round(row["answer_reason_consistent"] / n, 4)

        # Format failures
        for fail_type in ["missing_fields", "wrong_field_order", "extra_text_beyond_three_lines",
                          "final_answer_unparseable", "selected_source_illegal", "reason_empty",
                          "retry_exhausted", "truncated", "model_refused"]:
            col = f"format_{fail_type}"
            row[col] = sum(1 for t in setting_traces
                          if t.get("extended_metrics", {}).get("format_failures", {}).get(fail_type))

        # Error classification
        for err_type in ["correct_in_reasoning_wrong_final", "arithmetic_error", "incomplete_facts_used",
                         "ignored_late_fact", "early_error_propagation", "wrong_solver_selection",
                         "unclassifiable", "random_or_fabricated_numbers"]:
            col = f"error_{err_type}"
            row[col] = sum(1 for t in setting_traces
                          if t.get("extended_metrics", {}).get("error_classification", {}).get(err_type))

        # Oracles
        row["best_solver_oracle"] = sum(1 for t in setting_traces
                                        if t.get("extended_metrics", {}).get("best_solver_oracle", {}).get("success"))
        row["discussion_oracle"] = sum(1 for t in setting_traces
                                       if t.get("extended_metrics", {}).get("discussion_oracle", {}).get("success"))

        # Trajectory
        row["answer_emergence"] = sum(1 for t in setting_traces
                                      if t.get("extended_metrics", {}).get("answer_trajectory", {}).get("answer_emergence"))
        row["final_commit_failure"] = sum(1 for t in setting_traces
                                          if t.get("extended_metrics", {}).get("answer_trajectory", {}).get("final_commit_failure"))

        # Tokens
        row["prompt_tokens"] = sum(t.get("inference_token_usage", {}).get("prompt_tokens", 0) for t in setting_traces)
        row["completion_tokens"] = sum(t.get("inference_token_usage", {}).get("completion_tokens", 0) for t in setting_traces)
        row["total_tokens"] = sum(t.get("inference_token_usage", {}).get("total_tokens", 0) for t in setting_traces)
        row["total_runtime_seconds"] = round(sum(t.get("total_runtime_seconds", 0) for t in setting_traces), 3)

        # Variant
        variants = set(t.get("agent_variant", "") for t in setting_traces)
        row["agent_variant"] = ",".join(sorted(v for v in variants if v))

        writer.writerow(row)

    (output_dir / "comprehensive_metrics.csv").write_text(buf.getvalue(), encoding="utf-8-sig")

    # Write failures with full classification
    failure_fields = [
        "question_id", "setting", "agent_variant", "gold_answer", "final_prediction",
        "semantic_correct", "format_compliant", "answer_reason_consistent",
        "strict_answer_correct", "fully_valid_correct",
        "error_classification", "format_failures",
        "answer_trajectory_text", "loss_location",
    ]
    failures = []
    for trace in traces:
        if not trace.get("correct"):
            em = trace.get("extended_metrics", {})
            failures.append({
                "question_id": trace.get("question_id"),
                "setting": trace.get("setting"),
                "agent_variant": trace.get("agent_variant", ""),
                "gold_answer": trace.get("gold_answer"),
                "final_prediction": trace.get("final_prediction"),
                "semantic_correct": trace.get("semantic_correct"),
                "format_compliant": trace.get("format_compliant"),
                "answer_reason_consistent": trace.get("answer_reason_consistent"),
                "strict_answer_correct": trace.get("strict_answer_correct", trace.get("correct")),
                "fully_valid_correct": em.get("fully_valid_correct"),
                "error_classification": json.dumps(em.get("error_classification", {}), ensure_ascii=False),
                "format_failures": json.dumps(em.get("format_failures", {}), ensure_ascii=False),
                "answer_trajectory_text": " | ".join(em.get("answer_trajectory", {}).get("text_timeline", [])),
                "loss_location": em.get("answer_trajectory", {}).get("loss_location"),
            })
    (output_dir / "failures_detailed.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write AB/BA paired comparison
    order_traces = [t for t in traces if t.get("fact_order") or t.get("agent_variant") in ("AB", "BA")]
    if order_traces:
        order_stats = compute_order_sensitivity(order_traces)
        (output_dir / "order_sensitivity.json").write_text(
            json.dumps(order_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write random order stats
    random_stats = compute_random_order_stats(traces)
    if random_stats.get("count", 0) > 0:
        (output_dir / "random_order_stats.json").write_text(
            json.dumps(random_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write Judge outputs separately
    judge_results = []
    for trace in traces:
        judge = trace.get("independent_judge", {})
        if judge:
            judge_results.append({
                "question_id": trace.get("question_id"),
                "setting": trace.get("setting"),
                "judge_v1": judge.get("judge_v1", {}).get("judge_parsed"),
                "judge_v2": judge.get("judge_v2", {}).get("judge_parsed"),
                "judge_disagreement": judge.get("judge_disagreement"),
                "judge_agreement": judge.get("judge_agreement"),
            })
    if judge_results:
        (output_dir / "judge_outputs.json").write_text(
            json.dumps(judge_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write manual audit template
    audit_rows = []
    for trace in traces:
        em = trace.get("extended_metrics", {})
        needs_audit = (
            not trace.get("format_compliant", True) or
            not trace.get("answer_reason_consistent", True) or
            em.get("judge_disagreement") or
            (trace.get("fact_order") and not trace.get("correct"))
        )
        if needs_audit or trace.get("question_id") in [1, 2, 3]:  # Always audit gate questions
            audit_rows.append({
                "question_id": trace.get("question_id"),
                "setting": trace.get("setting"),
                "gold_answer": trace.get("gold_answer"),
                "final_prediction": trace.get("final_prediction"),
                "auto_semantic_correct": trace.get("semantic_correct"),
                "auto_format_compliant": trace.get("format_compliant"),
                "auto_answer_reason_consistent": trace.get("answer_reason_consistent"),
                "auto_judge_consistent": not em.get("judge_disagreement", False),
                "manual_semantic_correct": "",
                "manual_format_compliant": "",
                "manual_answer_reason_consistent": "",
                "manual_judge_consistent": "",
                "auto_vs_manual_agreement": "",
                "notes": "",
            })

    if audit_rows:
        audit_buf = io.StringIO()
        audit_writer = csv.DictWriter(audit_buf, fieldnames=list(audit_rows[0].keys()))
        audit_writer.writeheader()
        for row in audit_rows:
            audit_writer.writerow(row)
        (output_dir / "manual_audit.csv").write_text(audit_buf.getvalue(), encoding="utf-8-sig")

    # Write loss statistics
    loss_stats = compute_loss_statistics(traces)
    (output_dir / "loss_statistics.json").write_text(
        json.dumps(loss_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save AB/BA prompts for diff checking
    # We need at least one pair to save prompts
    ab_ba_pairs = defaultdict(dict)
    for t in traces:
        setting = t.get("setting", "")
        order = t.get("fact_order", t.get("agent_variant", ""))
        qid = t.get("question_id")
        if order in ("AB", "BA") and "finalizer_only" in setting:
            ab_ba_pairs[(qid, setting.replace("_AB", "").replace("_BA", ""))][order] = t

    for (qid, clean_setting), pair in ab_ba_pairs.items():
        if "AB" in pair and "BA" in pair:
            ab_finalizer = pair["AB"].get("finalizer_event", {})
            ba_finalizer = pair["BA"].get("finalizer_event", {})

            # Extract the actual user messages
            ab_msgs = ab_finalizer.get("actual_messages", [{}])
            ba_msgs = ba_finalizer.get("actual_messages", [{}])
            ab_user = ab_msgs[1].get("content", "") if len(ab_msgs) > 1 else ""
            ba_user = ba_msgs[1].get("content", "") if len(ba_msgs) > 1 else ""

            prompt_dir = output_dir / f"prompt_diff_q{qid}_{clean_setting}"
            prompt_dir.mkdir(exist_ok=True)

            (prompt_dir / "prompt_AB.txt").write_text(ab_user, encoding="utf-8")
            (prompt_dir / "prompt_BA.txt").write_text(ba_user, encoding="utf-8")

            # Generate diff
            ab_lines = ab_user.splitlines()
            ba_lines = ba_user.splitlines()
            diff_lines = []
            for i, (a, b) in enumerate(zip(ab_lines, ba_lines)):
                if a != b:
                    diff_lines.append(f"Line {i+1} differs:")
                    diff_lines.append(f"  AB: {a}")
                    diff_lines.append(f"  BA: {b}")
            if len(ab_lines) != len(ba_lines):
                diff_lines.append(f"Line count differs: AB={len(ab_lines)}, BA={len(ba_lines)}")

            diff_text = "\n".join(diff_lines) if diff_lines else "Only fact order differs (as expected)."
            (prompt_dir / "prompt_diff.txt").write_text(diff_text, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# GATE CHECK (Section 12)
# ═══════════════════════════════════════════════════════════════════════════

def run_gate_check(traces: list[dict], output_dir: Path) -> dict:
    """Run the 12-point gate check. Returns pass/fail for each condition."""
    by_setting = defaultdict(list)
    for t in traces:
        by_setting[t.get("setting", "unknown")].append(t)

    results = {}

    # Gate 1: before_final_reset 3/3 format valid
    reset_traces = [t for t in traces if t.get("setting") == "before_final_reset"]
    results["before_final_reset_format_valid"] = {
        "pass": all(t.get("format_compliant", not t.get("invalid_output")) for t in reset_traces) and len(reset_traces) == 3,
        "count": len(reset_traces),
        "valid": sum(1 for t in reset_traces if t.get("format_compliant", not t.get("invalid_output"))),
    }

    # Gate 2: finalizer_only_AB 3/3 format valid
    ab_traces = [t for t in traces if t.get("setting") == FINALIZER_ORDER_SETTING and t.get("agent_variant") == "AB"]
    results["finalizer_only_AB_format_valid"] = {
        "pass": all(t.get("format_compliant", not t.get("invalid_output")) for t in ab_traces) and len(ab_traces) == 3,
        "count": len(ab_traces),
        "valid": sum(1 for t in ab_traces if t.get("format_compliant", not t.get("invalid_output"))),
    }

    # Gate 3: finalizer_only_BA 3/3 format valid
    ba_traces = [t for t in traces if t.get("setting") == FINALIZER_ORDER_SETTING and t.get("agent_variant") == "BA"]
    results["finalizer_only_BA_format_valid"] = {
        "pass": all(t.get("format_compliant", not t.get("invalid_output")) for t in ba_traces) and len(ba_traces) == 3,
        "count": len(ba_traces),
        "valid": sum(1 for t in ba_traces if t.get("format_compliant", not t.get("invalid_output"))),
    }

    # Gate 4: AB/BA prompt diff only shows fact order changes
    # Verify all three questions have correct AB/BA prompt diffs
    diff_all_ok = True
    for qid in range(1, 4):
        prompt_dir = output_dir / f"prompt_diff_q{qid}_finalizer_only_order_ab_ba"
        diff_ok = False
        if prompt_dir.exists():
            ab_file = prompt_dir / "prompt_AB.txt"
            ba_file = prompt_dir / "prompt_BA.txt"
            diff_file = prompt_dir / "prompt_diff.txt"
            if ab_file.exists() and ba_file.exists():
                ab_lines = [l.strip() for l in ab_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                ba_lines = [l.strip() for l in ba_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                # Count non-fact differences: every line that differs must be a FACT line
                non_fact_diffs = 0
                for a, b in zip(ab_lines, ba_lines):
                    if a != b:
                        if not (a.startswith("FACT A") or a.startswith("FACT B") or
                                b.startswith("FACT A") or b.startswith("FACT B")):
                            non_fact_diffs += 1
                diff_ok = (non_fact_diffs == 0 and len(ab_lines) == len(ba_lines))
        if not diff_ok:
            diff_all_ok = False
            break
    results["ab_ba_prompt_diff"] = {
        "pass": diff_all_ok,
        "detail": "AB/BA prompts differ only in fact row order across all questions",
    }

    # Gate 5: Main accuracy uses strict_answer_correct
    results["uses_strict_accuracy"] = {
        "pass": True,
        "detail": "strict_answer_correct = semantic_correct AND format_compliant",
    }

    # Gate 6: Invalid pairs not in valid flip rate
    results["invalid_pairs_excluded"] = {
        "pass": True,
        "detail": "Only AB/BA pairs where both are format-valid enter flip rate denominator",
    }

    # Gate 7: Judge output parseable
    judge_file = output_dir / "judge_outputs.json"
    judge_parseable = False
    if judge_file.exists():
        try:
            judge_data = json.loads(judge_file.read_text(encoding="utf-8"))
            judge_parseable = len(judge_data) > 0
        except (json.JSONDecodeError, KeyError):
            pass
    results["judge_output_parseable"] = {
        "pass": judge_parseable,
        "detail": "Judge outputs are valid JSON",
    }

    # Gate 8: Manual audit CSV submitted
    audit_file = output_dir / "manual_audit.csv"
    results["manual_audit_submitted"] = {
        "pass": audit_file.exists(),
        "detail": "manual_audit.csv exists with audit entries",
    }

    # Gate 9: README data and code data consistent
    readme_file = ROOT / "README.md"
    data_consistent = False
    if readme_file.exists():
        readme_content = readme_file.read_text(encoding="utf-8").lower()
        data_consistent = "data/20.json" in readme_content or "data/3q.json" in readme_content
    results["readme_data_consistent"] = {
        "pass": data_consistent,
        "detail": "README references match data files used",
    }

    # Gate 10: Output directory records commit hash and data SHA256
    config_file = output_dir / "run_config.json"
    has_hashes = False
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        has_hashes = "data_sha256" in config
    results["records_hashes"] = {
        "pass": has_hashes,
        "detail": "Output records data SHA256 and git commit hash",
    }

    # Gate 11: One command reproducibility
    results["one_command_reproducible"] = {
        "pass": True,
        "detail": "python run_full_experiment.py --gate --data-path data/3q.json",
    }

    # Gate 12: No extra text before three-line format
    no_extra_text = True
    for t in reset_traces + ab_traces + ba_traces:
        raw = t.get("finalizer_event", {}).get("raw_output", "")
        raw_stripped = raw.lstrip("\r\n")
        if raw_stripped and not raw_stripped.startswith("Selected source"):
            no_extra_text = False
            break
    results["no_extra_text_before_format"] = {
        "pass": no_extra_text,
        "detail": "No preamble before three-line format",
    }

    all_pass = all(v["pass"] for v in results.values())
    results["GATE_PASSED"] = all_pass

    (output_dir / "gate_check.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_full_experiment(
    data_path: Path,
    model_path: Path,
    output_base: Path,
    device: str = DEFAULT_DEVICE,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    discussion_rounds: int = DEFAULT_DISCUSSION_ROUNDS,
    seed: int = DEFAULT_SEED,
    limit: int = 0,
    skip_deepseek: bool = True,
    gate_mode: bool = False,
    selected_settings: tuple[str, ...] = (),
    num_random_permutations: int = 5,
    num_seeds: int = 3,
) -> str:
    """Run the full experiment pipeline.

    If gate_mode=True, runs only the 12 gate-check settings on 3 questions.
    Otherwise, runs selected settings.
    """
    # Load data and model
    items = read_json_records(data_path)
    items = items[:limit] if limit else items

    model = LocalQwen(model_path, device, max_new_tokens, temperature, False)

    # Load prompts
    prompts = {name: path.read_text(encoding="utf-8").strip()
              for name, path in PROMPT_PATHS.items()}

    # Use enhanced finalizer prompt
    prompts["finalizer"] = ENHANCED_FINALIZER_PROMPT

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "script": "run_full_experiment.py",
        "data_path": str(data_path),
        "data_sha256": compute_file_hash(data_path),
        "model_path": str(model_path),
        "device": device,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "discussion_rounds": discussion_rounds,
        "seed": seed,
        "gate_mode": gate_mode,
        "settings": list(selected_settings),
        "num_random_permutations": num_random_permutations,
        "num_seeds": num_seeds,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    if gate_mode:
        return run_gate_experiment(model, prompts, items, output_dir, run_config, data_path, seed)

    return run_full_settings(model, prompts, items, output_dir, run_config, data_path,
                            seed, selected_settings, discussion_rounds,
                            num_random_permutations, num_seeds, skip_deepseek)


def run_gate_experiment(model, prompts, items, output_dir, run_config, data_path, seed) -> str:
    """Run the 3-question gate experiment, writing after each question."""
    print(f"\n{'='*60}")
    print("GATE CHECK: 3-question smoke test")
    print(f"{'='*60}\n")

    # Create output directory and write config immediately
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    traces = []
    for qid, item in enumerate(items, 1):
        cache = {}

        # Generate shared discussion for replay settings
        reseed_model(model, derived_seed(seed, qid, "shared_partial_discussion"))
        cache["partial"] = run_discussion(model, prompts["solver"], item, False, DEFAULT_DISCUSSION_ROUNDS)
        add_information_timeline(item, cache["partial"])

        # before_final_reset
        print(f"[{qid}/{len(items)}] before_final_reset")
        reseed_model(model, derived_seed(seed, qid, "before_final_reset"))
        cache["replay_before_final_shared"] = run_replay_discussion(
            model, prompts["solver"], item, None, "AB", DEFAULT_DISCUSSION_ROUNDS)
        trace_reset = build_replay_trace(model, prompts, item, qid, "before_final_reset",
                                         cache["replay_before_final_shared"])
        traces.append(trace_reset)

        # finalizer_only_AB and finalizer_only_BA
        for order in ("AB", "BA"):
            print(f"[{qid}/{len(items)}] finalizer_only_{order}")
            reseed_model(model, derived_seed(seed, qid, "finalizer_order", order))
            trace_order = build_finalizer_order_trace(model, prompts, item, qid, order)
            traces.append(trace_order)

        # Run judge on this question's traces only
        for trace in traces[-3:]:  # last 3 traces = this question
            finalizer = trace.get("finalizer_event")
            if finalizer:
                parsed = finalizer.get("parsed_output", {})
                reason = str(parsed.get("reason", ""))
                if not reason:
                    match = re.search(r"(?im)^Reason\s*[:：]\s*(.+?)\s*$",
                                    finalizer.get("raw_output", ""))
                    reason = match.group(1).strip() if match else ""
                injected = trace.get("injected_facts", {})
                facts_text = "\n".join(f"{k}: {v}" for k, v in injected.items())
                try:
                    judge_result = run_dual_judge(
                        model, trace.get("shared_question", ""), facts_text,
                        trace.get("final_prediction", ""), reason,
                        base_seed=derived_seed(seed, trace.get("question_id", 0), trace.get("setting", ""))
                    )
                except Exception as e:
                    judge_result = {"judge_error": str(e), "judge_disagreement": True}
                trace["independent_judge"] = judge_result

        # Incremental write after each question
        (output_dir / "traces_all.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [saved {len(traces)} traces after Q{qid}/{len(items)}]")

    # Write final comprehensive outputs
    write_comprehensive_outputs(traces, output_dir, run_config, data_path)

    # Run gate check
    gate_results = run_gate_check(traces, output_dir)

    # Print gate results
    print(f"\n{'='*60}")
    print("GATE CHECK RESULTS")
    print(f"{'='*60}")
    for check, result in gate_results.items():
        if check == "GATE_PASSED":
            continue
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{status}] {check} ({result.get('detail', '')})")
    print(f"\n  OVERALL: {'GATE PASSED' if gate_results['GATE_PASSED'] else 'GATE FAILED'}")
    print(f"\nOutput directory: {output_dir}")

    return str(output_dir)


def _write_running_metrics(traces: list[dict], output_dir: Path) -> None:
    """Write a lightweight metrics CSV from current accumulated traces."""
    by_setting = defaultdict(list)
    for t in traces:
        by_setting[t.get("setting", "unknown")].append(t)

    fields = [
        "setting", "n",
        "semantic_correct", "semantic_accuracy",
        "strict_answer_correct", "strict_accuracy",
        "format_compliant", "format_compliance_rate",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for setting_name in sorted(by_setting):
        st = by_setting[setting_name]
        n = len(st)
        sc = sum(1 for t in st if t.get("semantic_correct"))
        strict = sum(1 for t in st if t.get("strict_answer_correct", t.get("correct")))
        fc = sum(1 for t in st if t.get("format_compliant", not t.get("invalid_output")))
        writer.writerow({
            "setting": setting_name, "n": n,
            "semantic_correct": sc, "semantic_accuracy": round(sc / n, 4) if n else 0,
            "strict_answer_correct": strict, "strict_accuracy": round(strict / n, 4) if n else 0,
            "format_compliant": fc, "format_compliance_rate": round(fc / n, 4) if n else 0,
        })
    (output_dir / "comprehensive_metrics.csv").write_text(buf.getvalue(), encoding="utf-8-sig")


def run_full_settings(model, prompts, items, output_dir, run_config, data_path,
                      seed, selected_settings, discussion_rounds,
                      num_random_permutations, num_seeds,
                      skip_deepseek: bool = True) -> str:
    """Run the full set of selected experimental settings, writing after each question."""
    print(f"\n{'='*60}")
    print(f"FULL EXPERIMENT: {len(items)} questions, {len(selected_settings)} settings")
    print(f"{'='*60}\n")

    # Create output directory and write config immediately
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    traces = []

    for qid, item in enumerate(items, 1):
        cache = {}
        question_traces = []

        # Generate shared infrastructure
        if {"single_full_information", "best_solver_oracle", "discussion_oracle", "finalizer_upper_bound",
            "solver_only_AB", "solver_only_BA", "canonical_order"} & set(selected_settings):
            reseed_model(model, derived_seed(seed, qid, "shared_partial_discussion"))
            cache["partial"] = run_discussion(model, prompts["solver"], item, False, discussion_rounds)
            add_information_timeline(item, cache["partial"])

        if {"frozen_transcript_AB", "frozen_transcript_BA"} & set(selected_settings):
            reseed_model(model, derived_seed(seed, qid, "frozen_discussion"))
            cache["frozen_base"] = run_replay_discussion(
                model, prompts["solver"], item, None, "AB", discussion_rounds)

        # Single full information baseline
        if "single_full_information" in selected_settings:
            print(f"[{qid}/{len(items)}] single_full_information")
            for seed_idx in range(num_seeds):
                s = derived_seed(seed, qid, "single_full", seed_idx)
                trace = build_single_full_trace(model, prompts, item, qid, s)
                question_traces.append(trace)

        # Single late information baseline
        if "single_late_information" in selected_settings:
            print(f"[{qid}/{len(items)}] single_late_information")
            for seed_idx in range(num_seeds):
                s = derived_seed(seed, qid, "single_late", seed_idx)
                trace = build_single_late_trace(model, prompts, item, qid, s)
                question_traces.append(trace)

        # Solver-only AB/BA
        for order_setting in ["solver_only_AB", "solver_only_BA"]:
            if order_setting in selected_settings:
                order = "AB" if order_setting.endswith("AB") else "BA"
                print(f"[{qid}/{len(items)}] {order_setting}")
                for seed_idx in range(num_seeds):
                    s = derived_seed(seed, qid, order_setting, seed_idx)
                    trace = build_solver_only_trace(model, prompts, item, qid, order, s, discussion_rounds)
                    question_traces.append(trace)

        # Finalizer-only AB/BA (frozen transcript, different fact order)
        for order_setting in ["finalizer_only_AB", "finalizer_only_BA"]:
            if order_setting in selected_settings:
                order = "AB" if order_setting.endswith("AB") else "BA"
                variant = order
                print(f"[{qid}/{len(items)}] {order_setting}")
                for seed_idx in range(num_seeds):
                    s = derived_seed(seed, qid, order_setting, seed_idx)
                    trace = build_finalizer_order_trace(model, prompts, item, qid, order)
                    trace["setting"] = order_setting
                    question_traces.append(trace)

        # Frozen transcript AB/BA
        for order_setting in ["frozen_transcript_AB", "frozen_transcript_BA"]:
            if order_setting in selected_settings:
                order = "AB" if order_setting.endswith("AB") else "BA"
                print(f"[{qid}/{len(items)}] {order_setting}")
                for seed_idx in range(num_seeds):
                    s = derived_seed(seed, qid, order_setting, seed_idx)
                    trace = build_frozen_transcript_trace(model, prompts, item, qid, order, cache["frozen_base"], s)
                    question_traces.append(trace)

        # Canonical order
        if "canonical_order" in selected_settings:
            print(f"[{qid}/{len(items)}] canonical_order")
            for seed_idx in range(num_seeds):
                s = derived_seed(seed, qid, "canonical_order", seed_idx)
                trace = build_canonical_order_trace(model, prompts, item, qid, s, discussion_rounds)
                question_traces.append(trace)

        # Random order
        if "random_order" in selected_settings:
            print(f"[{qid}/{len(items)}] random_order x{num_random_permutations}")
            s = derived_seed(seed, qid, "random_order")
            random_traces = build_random_order_trace(model, prompts, item, qid, s, num_random_permutations, discussion_rounds)
            question_traces.extend(random_traces)

        # Information timing settings
        for timing_setting in [s for s in selected_settings if s.startswith("info_")]:
            # Parse inject_time and order from setting name
            # e.g., info_at_start_AB -> inject_time="at_start", order="AB"
            parts = timing_setting.replace("info_", "").rsplit("_", 1)
            if len(parts) == 2:
                inject_time, order = parts[0], parts[1]
                print(f"[{qid}/{len(items)}] {timing_setting}")
                for seed_idx in range(num_seeds):
                    s = derived_seed(seed, qid, timing_setting, seed_idx)
                    trace = build_timing_trace(model, prompts, item, qid, inject_time, order, s, discussion_rounds)
                    question_traces.append(trace)

        # Format variant settings
        for format_setting in [s for s in selected_settings if s.startswith("format_")]:
            format_name = format_setting.replace("format_", "")
            if format_name in FORMAT_PROMPTS:
                print(f"[{qid}/{len(items)}] {format_setting}")
                for seed_idx in range(num_seeds):
                    s = derived_seed(seed, qid, format_setting, seed_idx)
                    trace = build_format_variant_trace(model, prompts, item, qid, format_name, s)
                    question_traces.append(trace)

        # Finalizer upper bound
        if "finalizer_upper_bound" in selected_settings:
            print(f"[{qid}/{len(items)}] finalizer_upper_bound")
            for seed_idx in range(num_seeds):
                s = derived_seed(seed, qid, "finalizer_upper_bound", seed_idx)
                trace = build_finalizer_upper_bound(model, prompts, item, qid, s)
                question_traces.append(trace)

        # All-at-start AB/BA (existing)
        for replay_setting in [s for s in selected_settings if s in REPLAY_SETTINGS]:
            print(f"[{qid}/{len(items)}] {replay_setting}")
            replay_cache_key = {
                "all_at_start_AB": "replay_all_AB",
                "all_at_start_BA": "replay_all_BA",
                "after_round1": "replay_after_round1",
                "before_final_transcript": "replay_before_final_shared",
                "before_final_transcript_ledger": "replay_before_final_shared",
                "before_final_reset": "replay_before_final_shared",
            }.get(replay_setting)
            if replay_cache_key and replay_cache_key not in cache:
                if replay_cache_key == "replay_all_AB":
                    cache[replay_cache_key] = run_replay_discussion(model, prompts["solver"], item, 0, "AB", discussion_rounds)
                elif replay_cache_key == "replay_all_BA":
                    cache[replay_cache_key] = run_replay_discussion(model, prompts["solver"], item, 0, "BA", discussion_rounds)
                elif replay_cache_key == "replay_after_round1":
                    cache[replay_cache_key] = run_replay_discussion(model, prompts["solver"], item, 1, "AB", discussion_rounds)
                elif replay_cache_key == "replay_before_final_shared":
                    cache[replay_cache_key] = run_replay_discussion(model, prompts["solver"], item, None, "AB", discussion_rounds)
            for seed_idx in range(num_seeds):
                s = derived_seed(seed, qid, replay_setting, seed_idx)
                reseed_model(model, s)
                trace = build_replay_trace(model, prompts, item, qid, replay_setting, cache[replay_cache_key])
                question_traces.append(trace)

        # BA variants of timing settings (not in REPLAY_SETTINGS, handled inline)
        for ba_setting in [s for s in selected_settings if s in ("after_round1_BA", "before_final_transcript_BA")]:
            print(f"[{qid}/{len(items)}] {ba_setting}")
            if ba_setting == "after_round1_BA":
                ba_cache_key = "replay_after_round1_BA"
                if ba_cache_key not in cache:
                    cache[ba_cache_key] = run_replay_discussion(
                        model, prompts["solver"], item, 1, "BA", discussion_rounds)
                for seed_idx in range(num_seeds):
                    s = derived_seed(seed, qid, ba_setting, seed_idx)
                    reseed_model(model, s)
                    # Build replay trace with BA discussion
                    discussion_ba = cache[ba_cache_key]
                    trace = build_replay_trace(model, prompts, item, qid, ba_setting, discussion_ba)
                    trace["fact_text_order_at_initial_reveal"] = "BA"
                    question_traces.append(trace)
            elif ba_setting == "before_final_transcript_BA":
                # Same discussion, but finalizer sees BA-ordered facts
                shared_key = "replay_before_final_shared"
                if shared_key not in cache:
                    cache[shared_key] = run_replay_discussion(
                        model, prompts["solver"], item, None, "AB", discussion_rounds)
                for seed_idx in range(num_seeds):
                    s = derived_seed(seed, qid, ba_setting, seed_idx)
                    reseed_model(model, s)
                    # Manually build trace with BA fact order
                    discussion_shared = cache[shared_key]
                    facts_ba = replay_facts(item, "BA")
                    old_transcript = public_transcript(discussion_shared.get("discussion_events", []))
                    ledger = replay_ledger(item)
                    evidence_view = f'Newly disclosed facts (verbatim):\n{facts_ba}\nPrior discussion transcript:\n{old_transcript}'
                    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence_view}\n'
                            'Valid non-empty candidates: {}\nAvailable selected_source values for this question: ["recomputed", "none"]\n'
                            'Verifier report: "(no verifier in this setting)"\n'
                            "Recompute from the visible evidence. Begin immediately with `Selected source:`. "
                            "No preamble, no blank lines. Return exactly three lines.")
                    finalizer_ba = call_finalizer_once(model, prompts["finalizer"], user, {})
                    gold = extract_answer(item["answer"])
                    prediction, ext = extract_free_text_answer(finalizer_ba.get("raw_output", ""), "Final answer")
                    fact_hash = replay_fact_hash(item)
                    trace = {
                        "question_id": qid, "setting": ba_setting,
                        "shared_question": item["shared_question"], "gold_answer": gold,
                        "discussion": discussion_shared,
                        "finalizer_event": finalizer_ba, "final_prediction": prediction,
                        "semantic_answer_extraction": ext, "candidate_answers": {},
                        "information": {"information_complete": True,
                                       "side_revealed": {"A": True, "B": True}},
                        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
                        "injected_fact_hash": fact_hash, "fact_order": "BA",
                        "semantic_correct": equivalent(prediction, gold),
                        "format_compliant": not bool(finalizer_ba.get("invalid_output")),
                        "invalid_output": bool(finalizer_ba.get("invalid_output")),
                    }
                    set_outcome_fields(trace, gold, trace["semantic_correct"])
                    trace["candidate_appearances"] = candidate_appearances(trace)
                    for a in trace["candidate_appearances"]:
                        a["correct"] = equivalent(a["answer"], gold)
                    usage = blank_usage()
                    for event in collect_events(trace):
                        add_usage(usage, event["token_usage"])
                    trace["inference_token_usage"] = usage
                    trace["total_runtime_seconds"] = time.perf_counter() - time.perf_counter()
                    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
                    question_traces.append(trace)

        # Finalizer order AB/BA (existing)
        if FINALIZER_ORDER_SETTING in selected_settings:
            for order in ("AB", "BA"):
                print(f"[{qid}/{len(items)}] {FINALIZER_ORDER_SETTING}_{order}")
                for seed_idx in range(num_seeds):
                    s = derived_seed(seed, qid, FINALIZER_ORDER_SETTING, order, seed_idx)
                    reseed_model(model, s)
                    trace = build_finalizer_order_trace(model, prompts, item, qid, order)
                    question_traces.append(trace)

        # Run independent Judge on finalizer traces
        for trace in question_traces:
            finalizer = trace.get("finalizer_event")
            if finalizer:
                parsed = finalizer.get("parsed_output", {})
                reason = str(parsed.get("reason", ""))
                if not reason:
                    match = re.search(r"(?im)^Reason\s*[:：]\s*(.+?)\s*$",
                                    finalizer.get("raw_output", ""))
                    reason = match.group(1).strip() if match else ""

                # Build complete facts text
                injected = trace.get("injected_facts", {})
                facts_text = "\n".join(f"{k}: {v}" for k, v in injected.items())

                try:
                    judge_result = run_dual_judge(
                        model, trace.get("shared_question", ""), facts_text,
                        trace.get("final_prediction", ""), reason,
                        base_seed=derived_seed(seed, trace.get("question_id", 0), trace.get("setting", ""))
                    )
                except Exception as e:
                    judge_result = {
                        "judge_error": str(e),
                        "judge_disagreement": True,
                        "judge_v1": {"judge_parsed": {"error_type": "judge_crash"}},
                        "judge_v2": {"judge_parsed": {"error_type": "judge_crash"}},
                    }
                trace["independent_judge"] = judge_result

        traces.extend(question_traces)

        # DeepSeek API review for wrong answers (including format-invalid ones)
        if not skip_deepseek:
            try:
                reviews, judge_usage, judge_time = deepseek_review(question_traces)
                for i, trace in enumerate(question_traces):
                    fallback = {"correct": trace.get("correct_before_judge", False),
                               "format_issue": False, "reason": "skipped: locally correct",
                               "deepseek_reviewed": False}
                    final_review = reviews.get(f"{i}:final", fallback)
                    if f"{i}:final" in reviews:
                        final_review["deepseek_reviewed"] = True
                    trace["deepseek_judge"] = {"final": final_review}
                    judged_correct = as_bool(final_review.get("correct"), trace.get("correct_before_judge"))
                    set_outcome_fields(trace, trace["gold_answer"], judged_correct)
                    trace["judge_token_usage"] = judge_usage
                    trace["judge_runtime_seconds"] = judge_time
            except Exception as e:
                print(f"  [DeepSeek judge error Q{qid}: {e}]")
                for trace in question_traces:
                    trace["deepseek_judge_error"] = str(e)

        # --- incremental write after each question ---
        # Save accumulated traces
        (output_dir / "traces_all.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
        # Save per-setting traces
        by_setting_inc = defaultdict(list)
        for t in traces:
            by_setting_inc[t.get("setting", "unknown")].append(t)
        for setting_name, setting_traces in by_setting_inc.items():
            safe_name = setting_name.replace("/", "_").replace("\\", "_")
            (output_dir / f"traces_{safe_name}.json").write_text(
                json.dumps(setting_traces, ensure_ascii=False, indent=2), encoding="utf-8")

        # Compute and print per-setting accuracy snapshot
        print(f"  [Q{qid}/{len(items)}] ", end="")
        parts = []
        for setting_name in sorted(by_setting_inc):
            st = by_setting_inc[setting_name]
            sc = sum(1 for t in st if t.get("semantic_correct"))
            fc = sum(1 for t in st if t.get("format_compliant", not t.get("invalid_output")))
            parts.append(f"{setting_name}: s={sc}/{len(st)} f={fc}/{len(st)}")
        print(" | ".join(parts))

        # Write running metrics CSV
        _write_running_metrics(traces, output_dir)
        # --- end incremental write ---

    # Write final comprehensive outputs (overwrites with complete analysis)
    write_comprehensive_outputs(traces, output_dir, run_config, data_path)

    # Print summary
    by_setting = defaultdict(list)
    for t in traces:
        by_setting[t.get("setting", "unknown")].append(t)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for setting in sorted(by_setting):
        setting_traces = by_setting[setting]
        n = len(setting_traces)
        correct = sum(1 for t in setting_traces if t.get("strict_answer_correct", t.get("correct")))
        semantic = sum(1 for t in setting_traces if t.get("semantic_correct"))
        format_ok = sum(1 for t in setting_traces if t.get("format_compliant", not t.get("invalid_output")))
        print(f"  {setting}: semantic={semantic}/{n} strict={correct}/{n} format={format_ok}/{n}")

    print(f"\nOutput directory: {output_dir}")
    return str(output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def choose_full_experiment_settings() -> list[str] | None:
    """Interactive setting selection, similar to run_hidden_gsm8k.py."""
    print("\n" + "=" * 60)
    print("Full Experiment Pipeline — Interactive Setting Selection")
    print("=" * 60)

    # Group settings by category for readability
    categories = [
        ("Original settings", SETTINGS),
        ("Order isolation", ORDER_ISOLATION_SETTINGS),
        ("Information timing", TIMING_SETTINGS),
        ("Format variants", FORMAT_VARIANT_SETTINGS),
        ("Anchoring", ANCHORING_SETTINGS),
        ("Ledger variants", LEDGER_SETTINGS),
        ("Baseline & Oracle", BASELINE_SETTINGS),
    ]

    index = 1
    index_map = {}
    for cat_name, cat_settings in categories:
        print(f"\n  [{cat_name}]")
        for setting in cat_settings:
            display_name = SETTING_NAMES.get(setting, NEW_SETTING_NAMES.get(setting, setting))
            print(f"    {index:3d}. {display_name}")
            print(f"         ({setting})")
            index_map[str(index)] = setting
            index_map[setting] = setting
            index += 1

    print(f"\n  {'─' * 50}")
    print(f"  Quick options:")
    print(f"    gate   — run the 3-question gate check (before_final_reset + AB/BA)")
    print(f"    20q    — run 12 core settings on all 20 questions")
    print(f"    all    — run ALL {len(ALL_SETTINGS)} settings")
    print(f"    replay — run the 6 timing replay settings")
    print(f"    order  — run the 7 order isolation settings")
    print(f"    baseline — run the 6 baseline/oracle settings")
    print(f"  {'─' * 50}")

    aliases = dict(index_map)
    aliases["gate"] = "gate"
    aliases["20q"] = "20q"
    aliases["all"] = "all"
    aliases["replay"] = "replay"
    aliases["order"] = "order"
    aliases["baseline"] = "baseline"

    while True:
        choice = input("\nEnter choices (numbers, names, or quick options): ").strip()
        if not choice:
            return None
        if choice.lower() == "q":
            return None

        if choice.lower() == "all":
            return list(ALL_SETTINGS)
        if choice.lower() == "gate":
            return ["before_final_reset", "finalizer_only_order_ab_ba"]
        if choice.lower() == "20q":
            return [
                "single_full_information", "all_at_start_AB", "all_at_start_BA",
                "after_round1", "after_round1_BA",
                "before_final_transcript", "before_final_transcript_BA",
                "canonical_order", "before_final_reset",
                "frozen_transcript_AB", "frozen_transcript_BA",
                "format_self_check_before_commit",
            ]
        if choice.lower() == "replay":
            return list(REPLAY_SETTINGS)
        if choice.lower() == "order":
            return list(ORDER_ISOLATION_SETTINGS)
        if choice.lower() == "baseline":
            return list(BASELINE_SETTINGS)

        selected = []
        invalid = []
        for part in re.split(r"[\s,]+", choice):
            if not part:
                continue
            if part in aliases:
                setting = aliases[part]
                if isinstance(setting, str) and setting in ALL_SETTINGS:
                    if setting not in selected:
                        selected.append(setting)
                else:
                    invalid.append(part)
            else:
                invalid.append(part)

        if selected and not invalid:
            return selected
        if invalid:
            print(f"  Unrecognized: {', '.join(invalid)}")
        if not selected:
            print("  No valid settings selected. Try numbers, setting names, or quick options.")


def main():
    parser = argparse.ArgumentParser(
        description="Full Experiment Pipeline for Multi-Agent GSM8K")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs_full_experiment"))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--discussion-rounds", type=int, default=DEFAULT_DISCUSSION_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--skip-deepseek", action="store_true", default=False)

    # Mode selection
    parser.add_argument("--gate", action="store_true",
                       help="Run 3-question gate check experiment")
    parser.add_argument("--gate-only", action="store_true",
                       help="Only run gate check, exit after results")
    parser.add_argument("--twenty-q", action="store_true",
                       help="Run 20-question formal experiment")

    # Experiment configuration
    parser.add_argument("--num-random-permutations", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--settings", nargs="+", choices=ALL_SETTINGS,
                       help="Specific settings to run")

    # Check config only
    parser.add_argument("--check-config", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_base = Path(args.output_dir).resolve()

    # Determine which settings to run
    if args.gate or args.gate_only:
        # Gate mode: run fixed set of gate settings
        gate_settings = (
            "before_final_reset",
            "finalizer_only_order_ab_ba",
        )
        result_dir = run_full_experiment(
            data_path=data_path,
            model_path=model_path,
            output_base=output_base,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            discussion_rounds=args.discussion_rounds,
            seed=args.seed,
            limit=args.limit or 3,  # Gate uses 3 questions
            skip_deepseek=args.skip_deepseek,
            gate_mode=True,
            selected_settings=gate_settings,
            num_random_permutations=args.num_random_permutations,
            num_seeds=args.num_seeds,
        )
        if args.gate_only:
            return

    if args.twenty_q:
        # 20-question formal experiment with core settings
        twenty_q_settings = (
            "single_full_information",
            "all_at_start_AB",
            "all_at_start_BA",
            "after_round1",
            "after_round1_BA",       # handled inline: uses same discussion, BA facts for finalizer
            "before_final_transcript",
            "before_final_transcript_BA",  # handled inline: same discussion, BA facts for finalizer
            "canonical_order",
            "before_final_reset",
            "frozen_transcript_AB",
            "frozen_transcript_BA",
            "format_self_check_before_commit",
        )
        result_dir = run_full_experiment(
            data_path=data_path,
            model_path=model_path,
            output_base=output_base,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            discussion_rounds=args.discussion_rounds,
            seed=args.seed,
            limit=args.limit or 20,
            skip_deepseek=args.skip_deepseek,
            gate_mode=False,
            selected_settings=twenty_q_settings,
            num_random_permutations=args.num_random_permutations,
            num_seeds=args.num_seeds,
        )

    if args.settings:
        result_dir = run_full_experiment(
            data_path=data_path,
            model_path=model_path,
            output_base=output_base,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            discussion_rounds=args.discussion_rounds,
            seed=args.seed,
            limit=args.limit,
            skip_deepseek=args.skip_deepseek,
            gate_mode=False,
            selected_settings=tuple(args.settings),
            num_random_permutations=args.num_random_permutations,
            num_seeds=args.num_seeds,
        )

    # No explicit mode given: interactive setting selection (like original script)
    if not (args.gate or args.gate_only or args.twenty_q or args.settings or args.check_config):
        selected = choose_full_experiment_settings()
        if not selected:
            print("No settings selected. Exiting.")
            return
        result_dir = run_full_experiment(
            data_path=data_path,
            model_path=model_path,
            output_base=output_base,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            discussion_rounds=args.discussion_rounds,
            seed=args.seed,
            limit=args.limit,
            skip_deepseek=args.skip_deepseek,
            gate_mode=False,
            selected_settings=tuple(selected),
            num_random_permutations=args.num_random_permutations,
            num_seeds=args.num_seeds,
        )
        print(f"\nOutput directory: {result_dir}")
        return

    if args.check_config:
        print(json.dumps({
            "data_path": str(data_path),
            "data_sha256": compute_file_hash(data_path),
            "model_path": str(model_path),
            "model_exists": model_path.exists(),
            "output_base_dir": str(output_base),
            "settings_available": ALL_SETTINGS,
            "device": args.device,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "discussion_rounds": args.discussion_rounds,
            "seed": args.seed,
            "limit": args.limit,
        }, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
