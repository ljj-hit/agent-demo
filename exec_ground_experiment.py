#!/usr/bin/env python3
"""
ExecGround Six-Group Ablation Experiment (生死实验)

Six settings on the existing 20 problems to measure how correct-candidate emergence
rate changes layer by layer:

  Setting 0: free_discussion           — original baseline (private facts, free discussion)
  Setting 1: reveal_all                — all raw facts disclosed before discussion
  Setting 2: canonical_ledger          — TypedFact → CanonicalLedger → discussion with ledger
  Setting 3: ledger_fresh_solver       — TypedFact → CanonicalLedger → fresh solver (NO discussion)
  Setting 4: ledger_exec_plan          — TypedFact → Ledger → fresh solver → executable plan
  Setting 5: ledger_exec_plan_verify   — TypedFact → Ledger → plan → coverage verify → fix loop

Key observation: NOT final accuracy, but correct-candidate-emergence rate per layer.

Usage:
  python exec_ground_experiment.py --limit 20 --seeds 3
  python exec_ground_experiment.py --setting 3 --limit 5 --seeds 1  # single setting
  python exec_ground_experiment.py --use-gold-facts  # use rule-based facts (no LLM extraction)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# ── Import infrastructure ──
import importlib.util as _iu

_HIDDEN_SPEC = _iu.spec_from_file_location("run_hidden_gsm8k", _SCRIPT_DIR / "run_hidden_gsm8k.py")
_run_hidden = _iu.module_from_spec(_HIDDEN_SPEC)
_HIDDEN_SPEC.loader.exec_module(_run_hidden)

_ORACLE_SPEC = _iu.spec_from_file_location("oracle_intervention_experiment",
                                            _SCRIPT_DIR / "oracle_intervention_experiment.py")
_oracle_exp = _iu.module_from_spec(_ORACLE_SPEC)
_ORACLE_SPEC.loader.exec_module(_oracle_exp)

# ── Import ExecGround ──
import exec_ground as eg

# Re-export core symbols
read_json_records      = _run_hidden.read_json_records
extract_answer         = _run_hidden.extract_answer
decimal                = _run_hidden.decimal
equivalent             = _run_hidden.equivalent
extract_labeled_answer = _run_hidden.extract_labeled_answer
extract_free_text_answer = _run_hidden.extract_free_text_answer
parse_solver_final     = _run_hidden.parse_solver_final
explicitly_undetermined = _run_hidden.explicitly_undetermined
parse_object           = _run_hidden.parse_object
blank_usage            = _run_hidden.blank_usage
add_usage              = _run_hidden.add_usage
derived_seed           = _run_hidden.derived_seed
reseed_model           = _run_hidden.reseed_model
model_event            = _run_hidden.model_event
paired_model_events    = _run_hidden.paired_model_events
public_transcript      = _run_hidden.public_transcript
run_discussion         = _run_hidden.run_discussion
replay_facts           = _run_hidden.replay_facts
replay_ledger          = _run_hidden.replay_ledger
run_replay_discussion  = _run_hidden.run_replay_discussion
atomic_facts           = _run_hidden.atomic_facts
add_information_timeline = _run_hidden.add_information_timeline
single_call            = _run_hidden.single_call
event_answer           = _run_hidden.event_answer
candidate_appearances  = _run_hidden.candidate_appearances
parse_fixed_finalizer  = _run_hidden.parse_fixed_finalizer
check_answer_reason_consistency = _run_hidden.check_answer_reason_consistency
source_consistency_error = _run_hidden.source_consistency_error
call_finalizer_once    = _run_hidden.call_finalizer_once
collect_events         = _run_hidden.collect_events
classify               = _run_hidden.classify
set_outcome_fields     = _run_hidden.set_outcome_fields
LocalQwen              = _run_hidden.LocalQwen
USAGE_KEYS             = _run_hidden.USAGE_KEYS
VERIFIER_DEFAULT       = _run_hidden.VERIFIER_DEFAULT
FINALIZER_DEFAULT      = _run_hidden.FINALIZER_DEFAULT

# ── Config ──
ROOT = _SCRIPT_DIR
DATA_PATH = ROOT / "data" / "20.json"
MODEL_PATH = ROOT / "qwen2.5-1.5B"
OUTPUT_BASE_DIR = ROOT / "outputs_exec_ground"
PROMPT_DIR = ROOT / "hidden_gsm8k_prompts"

DEFAULT_DEVICE = "cuda"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_DISCUSSION_ROUNDS = 2
DEFAULT_SEED = 42
DEFAULT_MAX_VERIFY_ROUNDS = 2

EXEC_GROUND_SETTINGS = [
    "free_discussion",
    "reveal_all",
    "canonical_ledger",
    "ledger_fresh_solver",
    "ledger_exec_plan",
    "ledger_exec_plan_verify",
]

SETTING_LABELS = {
    "free_discussion":          "0. Free Discussion (baseline: private facts, free discussion)",
    "reveal_all":               "1. Reveal-All (all raw facts disclosed before discussion)",
    "canonical_ledger":         "2. Canonical Ledger (TypedFact → Ledger → discussion with ledger)",
    "ledger_fresh_solver":      "3. Ledger + Fresh Solver (no discussion history)",
    "ledger_exec_plan":         "4. Ledger + Executable Plan (structured JSON-IR)",
    "ledger_exec_plan_verify":  "5. Ledger + Exec Plan + Coverage Verify (verification loop)",
}

# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED FINALIZER — accepts non-standard source names like fresh_solver,
# executed_plan, verified_plan (the standard call_finalizer_once rejects them).
# ═══════════════════════════════════════════════════════════════════════════════

EXTENDED_FINALIZER_SYSTEM = """You produce the final answer. Use only the shared question and the evidence provided.

Return exactly three non-empty physical lines, with no Markdown, no blank lines:
Selected source: <source_name>
Final answer: <number or undetermined>
Reason: <explanation>

Use only a selected_source listed as available. Reason must be non-empty.
If a source is selected, copy its answer exactly; do not recompute a different answer."""


def call_extended_finalizer(model, system, user, candidates, allowed_sources):
    """Custom finalizer that accepts extended source names.

    Unlike call_finalizer_once which hardcodes allowed sources as
    {solver_a, solver_b, verifier, recomputed, none}, this accepts
    any source name listed in allowed_sources.
    """
    event = model_event(model, "finalizer", system, user, "finalization", None, temperature=0.0)
    raw = event.get("raw_output", "")
    parsed, error = _parse_extended_finalizer(raw, allowed_sources)

    event.update(
        parsed_output=parsed,
        parse_error="",
        validation_error=error,
        invalid_output=bool(error),
    )
    return event


def _parse_extended_finalizer(text, allowed_sources):
    """Parse finalizer output allowing any source in allowed_sources."""
    raw = str(text or "").rstrip("\r\n")
    lines = raw.splitlines()
    labels = ("Selected source", "Final answer", "Reason")

    if len(lines) != 3:
        return dict(FINALIZER_DEFAULT), f"expected exactly three lines, got {len(lines)}"

    values = {}
    for line, label in zip(lines, labels):
        match = re.fullmatch(rf"{re.escape(label)}\s*[:：]\s*(.*)", line, re.I)
        if not match:
            return dict(FINALIZER_DEFAULT), f"expected line `{label}: ...`"
        values[label] = match.group(1).strip()

    source = values["Selected source"].lower()
    if source not in allowed_sources:
        return dict(FINALIZER_DEFAULT), f"selected_source '{source}' is not allowed (valid: {sorted(allowed_sources)})"

    if not values["Reason"]:
        return dict(FINALIZER_DEFAULT), "Reason must not be empty"

    answer = "" if explicitly_undetermined(values["Final answer"]) else extract_answer(values["Final answer"])
    return {"selected_source": source, "final_answer": answer, "reason": values["Reason"]}, ""


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

FRESH_SOLVER_FINAL_SYSTEM = """You are a careful mathematical problem solver. You receive a question and a clean, verified fact table.

These facts are guaranteed to be correct and complete. Use ONLY these facts.
Do NOT guess. Do NOT refer to any prior discussion — you have no access to it.
Put `Final answer: ...` on the FIRST line, then at most three sentences of reasoning."""

DISCUSSION_WITH_LEDGER_USER = """Role: solver_{side}
Discussion round: {round_no} of {total_rounds}
Purpose: {purpose}
Shared question: {question}

CANONICAL FACT LEDGER (authoritative, all facts from both agents):
{ledger_text}

Public transcript through the previous completed round:
{transcript}

Use the Canonical Fact Ledger as your authoritative source. Reason about the complete problem.
Begin with exactly one separate line `Current answer: <answer>` or `Current answer: undetermined`,
then give your reasoning. Write directly to the other solver in natural text; do not output JSON.
You cannot see the peer's same-round message."""


def load_prompts():
    """Load prompt files from disk."""
    prompts = {}
    for name in ["solver", "verifier", "finalizer"]:
        path = PROMPT_DIR / f"{name}.txt"
        if path.exists():
            prompts[name] = path.read_text(encoding="utf-8")
    return prompts


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_candidate_emergence(trace: dict, gold: str) -> dict:
    """Check if correct answer appeared as a candidate at any point."""
    appearances = trace.get("candidate_appearances", [])
    correct_appearances = []
    for a in appearances:
        ans = a.get("answer", "")
        is_correct = equivalent(ans, gold)
        if is_correct:
            correct_appearances.append(a)

    emerged = len(correct_appearances) > 0
    emerged_with_complete_info = any(
        a.get("information_complete_at_appearance", False)
        for a in correct_appearances
    )

    return {
        "correct_candidate_emerged": emerged,
        "correct_candidate_count": len(correct_appearances),
        "emerged_with_complete_information": emerged_with_complete_info,
        "total_candidates": len(appearances),
        "correct_appearances": [{
            "source": a.get("source", ""),
            "phase": a.get("phase", ""),
            "answer": a.get("answer", ""),
        } for a in correct_appearances],
    }


def compute_fact_distortion_rate(agent_outputs: list[str], required_facts: list[str]) -> dict:
    """Check what fraction of required fact numbers are missing from agent outputs."""
    all_output = " ".join(agent_outputs)
    total_fact_nums = set()
    found_fact_nums = set()
    for fact in required_facts:
        nums = set(re.findall(r"\b(\d+(?:\.\d+)?)\b", fact))
        total_fact_nums.update(nums)
        for n in nums:
            if n in all_output:
                found_fact_nums.add(n)
    if not total_fact_nums:
        return {"distortion_rate": 0.0, "missing_count": 0, "total_count": 0}
    missing = total_fact_nums - found_fact_nums
    return {
        "distortion_rate": len(missing) / len(total_fact_nums),
        "missing_count": len(missing),
        "total_count": len(total_fact_nums),
    }


def compute_typed_fact_quality(facts: list[eg.TypedFact], condition_text: str) -> dict:
    """Measure typed fact extraction quality against source text."""
    if not facts:
        return {"coverage": 0.0, "avg_evidence_length": 0, "facts_with_values": 0, "total": 0}

    source_nums = set(re.findall(r"\b(\d+(?:\.\d+)?)\b", condition_text))
    fact_nums = set()
    for f in facts:
        if f.value is not None:
            fact_nums.add(str(int(f.value)) if f.value == int(f.value) else str(f.value))

    coverage = len(fact_nums & source_nums) / max(1, len(source_nums))
    return {
        "coverage": coverage,
        "num_coverage": f"{len(fact_nums & source_nums)}/{len(source_nums)}",
        "avg_evidence_length": sum(len(f.evidence) for f in facts) / max(1, len(facts)),
        "facts_with_values": sum(1 for f in facts if f.value is not None),
        "total": len(facts),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_free_discussion_trace(model, prompts, item, qid, base_seed, discussion_rounds=2):
    """Setting 0: Baseline — use existing oracle experiment builder."""
    return _oracle_exp.build_free_discussion_trace(
        model, prompts, item, qid, base_seed, discussion_rounds)


def build_reveal_all_trace(model, prompts, item, qid, base_seed, discussion_rounds=2):
    """Setting 1: Reveal-All — use existing oracle_disclosure builder."""
    return _oracle_exp.build_oracle_disclosure_trace(
        model, prompts, item, qid, base_seed, discussion_rounds)


def build_canonical_ledger_trace(model, prompts, item, qid, base_seed,
                                  discussion_rounds=2, use_gold_facts=False):
    """Setting 2: TypedFact → CanonicalLedger → discussion with ledger → finalizer.

    Phase 1: Each agent extracts TypedFacts (NO answering)
    Phase 2: Build CanonicalLedger
    Phase 3: Agents discuss with ledger as reference
    Phase 4: Finalizer selects answer
    """
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "canonical_ledger"))

    # Phase 1: Extract typed facts
    if use_gold_facts:
        facts_a = eg.extract_gold_typed_facts(item["condition_A"], "A")
        facts_b = eg.extract_gold_typed_facts(item["condition_B"], "B")
        fact_method = "gold_rule_based"
    else:
        reseed_model(model, derived_seed(base_seed, qid, "typed_fact_A"))
        facts_a = eg.extract_typed_facts(model, item["condition_A"], "A", temperature=0.0)
        reseed_model(model, derived_seed(base_seed, qid, "typed_fact_B"))
        facts_b = eg.extract_typed_facts(model, item["condition_B"], "B", temperature=0.0)
        fact_method = "llm_extraction"

    # Phase 2: Build canonical ledger
    ledger = eg.CanonicalLedger.build(facts_a, facts_b, item)
    fact_quality_a = compute_typed_fact_quality(facts_a, item["condition_A"])
    fact_quality_b = compute_typed_fact_quality(facts_b, item["condition_B"])

    # Phase 3: Discussion with ledger as reference
    # Custom discussion loop where agents see the ledger
    discussion = _run_discussion_with_ledger(
        model, prompts["solver"], item, ledger, discussion_rounds)
    add_information_timeline(item, discussion)

    # Phase 4: Finalizer
    transcript = public_transcript(discussion.get("discussion_events", []))
    evidence = f"{ledger.to_text()}\n\nDiscussion transcript:\n{transcript}"

    solver_finals = discussion.get("solver_finals", {})
    candidates = {}
    for side in ("a", "b"):
        sf = solver_finals.get(side, {})
        ans = sf.get("answer", event_answer(sf))
        if ans:
            candidates[f"solver_{side}"] = ans

    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            f'Valid non-empty candidates: {json.dumps(candidates)}\n'
            f'Available selected_source values for this question: ["solver_a", "solver_b", "recomputed", "none"]\n'
            f'Verifier report: "(no verifier in this setting)"\n'
            "Use the canonical ledger as the authoritative source. "
            "Select a supported candidate or recompute. Begin immediately with `Selected source:`.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, candidates)

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "canonical_ledger",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion, "finalizer_event": finalizer,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "exec_ground": {
            "fact_method": fact_method,
            "typed_facts_A": [f.to_dict() for f in facts_a],
            "typed_facts_B": [f.to_dict() for f in facts_b],
            "ledger_fact_count": len(ledger.facts),
            "ledger_conflict_count": len(ledger.conflicts),
            "ledger_text": ledger.to_text(),
            "fact_quality_A": fact_quality_a,
            "fact_quality_B": fact_quality_b,
        },
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


def build_ledger_fresh_solver_trace(model, prompts, item, qid, base_seed,
                                     discussion_rounds=2, use_gold_facts=False):
    """Setting 3: TypedFact → Ledger → Fresh Solver (NO discussion history) → Finalizer.

    Fresh solver reads ONLY question + canonical ledger. NO discussion transcript.
    """
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "ledger_fresh"))

    # Phase 1: Extract typed facts
    if use_gold_facts:
        facts_a = eg.extract_gold_typed_facts(item["condition_A"], "A")
        facts_b = eg.extract_gold_typed_facts(item["condition_B"], "B")
        fact_method = "gold_rule_based"
    else:
        reseed_model(model, derived_seed(base_seed, qid, "typed_fact_A"))
        facts_a = eg.extract_typed_facts(model, item["condition_A"], "A", temperature=0.0)
        reseed_model(model, derived_seed(base_seed, qid, "typed_fact_B"))
        facts_b = eg.extract_typed_facts(model, item["condition_B"], "B", temperature=0.0)
        fact_method = "llm_extraction"

    # Phase 2: Build canonical ledger
    ledger = eg.CanonicalLedger.build(facts_a, facts_b, item)
    fact_quality_a = compute_typed_fact_quality(facts_a, item["condition_A"])
    fact_quality_b = compute_typed_fact_quality(facts_b, item["condition_B"])

    # Phase 3: Fresh solver (no discussion!)
    reseed_model(model, derived_seed(base_seed, qid, "fresh_solver_call"))
    fresh_user = (
        f'Shared question: {item["shared_question"]}\n\n'
        f'{ledger.to_text()}\n\n'
        'You are seeing this problem for the first time. The facts above are guaranteed correct '
        'and complete. Compute the answer. Put `Final answer: ...` on the FIRST line, '
        'then at most three sentences of reasoning.'
    )
    fresh_raw, fresh_usage, fresh_elapsed = model.call(
        FRESH_SOLVER_FINAL_SYSTEM, fresh_user, temperature=0.0)
    fresh_answer, fresh_fmt_err = parse_solver_final(fresh_raw)
    fresh_prediction = fresh_answer if not fresh_fmt_err else extract_answer(fresh_raw)

    # Phase 4: Finalizer
    candidates = {"fresh_solver": fresh_prediction}
    evidence = (
        f"{ledger.to_text()}\n\n"
        f"--- Fresh Solver (NO discussion history, fact-ledger only) ---\n"
        f"Fresh solver answer: {fresh_prediction}\n"
        f"Fresh solver reasoning: {fresh_raw[:500]}"
    )

    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            f'Valid non-empty candidates: {json.dumps(candidates)}\n'
            f'Available selected_source values for this question: ["fresh_solver", "recomputed", "none"]\n'
            f'Verifier report: "(no verifier in this setting)"\n'
            "The fresh_solver saw ONLY the canonical fact ledger (no discussion). "
            "Select the best-supported candidate or recompute. Begin immediately with `Selected source:`.")
    finalizer = call_extended_finalizer(model, prompts["finalizer"], user, candidates,
                                          {"fresh_solver", "recomputed", "none"})

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "ledger_fresh_solver",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "finalizer_event": finalizer,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "fresh_solver_event": {
            "agent": "fresh_solver", "raw_output": fresh_raw,
            "answer": fresh_answer, "prediction": fresh_prediction,
            "token_usage": fresh_usage, "runtime_seconds": fresh_elapsed,
        },
        "fresh_solver_correct": equivalent(fresh_prediction, gold),
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "exec_ground": {
            "fact_method": fact_method,
            "typed_facts_A": [f.to_dict() for f in facts_a],
            "typed_facts_B": [f.to_dict() for f in facts_b],
            "ledger_fact_count": len(ledger.facts),
            "ledger_conflict_count": len(ledger.conflicts),
            "ledger_text": ledger.to_text(),
            "fact_quality_A": fact_quality_a,
            "fact_quality_B": fact_quality_b,
        },
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    trace["candidate_appearances"].append({
        "source": "fresh_solver", "phase": "fresh_solve",
        "answer": fresh_prediction,
        "information_complete_at_appearance": True,
        "correct": equivalent(fresh_prediction, gold),
    })
    usage = blank_usage()
    add_usage(usage, fresh_usage)
    for event in collect_events(trace):
        add_usage(usage, event.get("token_usage", blank_usage()))
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def build_ledger_exec_plan_trace(model, prompts, item, qid, base_seed,
                                  discussion_rounds=2, use_gold_facts=False):
    """Setting 4: TypedFact → Ledger → Fresh Solver → Executable Plan → Execute → Finalizer.

    Fresh solver outputs executable JSON-IR. Plan is executed to get the answer.
    """
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "ledger_exec_plan"))

    # Phase 1: Extract typed facts
    if use_gold_facts:
        facts_a = eg.extract_gold_typed_facts(item["condition_A"], "A")
        facts_b = eg.extract_gold_typed_facts(item["condition_B"], "B")
        fact_method = "gold_rule_based"
    else:
        reseed_model(model, derived_seed(base_seed, qid, "typed_fact_A"))
        facts_a = eg.extract_typed_facts(model, item["condition_A"], "A", temperature=0.0)
        reseed_model(model, derived_seed(base_seed, qid, "typed_fact_B"))
        facts_b = eg.extract_typed_facts(model, item["condition_B"], "B", temperature=0.0)
        fact_method = "llm_extraction"

    # Phase 2: Build canonical ledger
    ledger = eg.CanonicalLedger.build(facts_a, facts_b, item)
    fact_quality_a = compute_typed_fact_quality(facts_a, item["condition_A"])
    fact_quality_b = compute_typed_fact_quality(facts_b, item["condition_B"])

    # Phase 3: Fresh solver → executable plan
    reseed_model(model, derived_seed(base_seed, qid, "exec_plan_solver"))
    plan, plan_raw, plan_usage = eg.fresh_solve(
        model, item["shared_question"], ledger, temperature=0.0)

    # Phase 4: Execute plan
    executed_result = eg.execute_plan(plan, ledger)
    execution_correct = (
        executed_result is not None
        and decimal(gold) is not None
        and math.isclose(executed_result, float(decimal(gold)), rel_tol=1e-9)
    )

    # Phase 5: Finalizer - presents both executed result and raw plan
    executed_str = f"{executed_result}" if executed_result is not None else "EXECUTION FAILED"
    candidates = {
        "executed_plan": executed_str,
    }

    evidence = (
        f"{ledger.to_text()}\n\n"
        f"--- Executable Plan (JSON-IR) ---\n"
        f"{json.dumps(plan.to_json(), indent=2)}\n\n"
        f"--- Plan Execution Result ---\n"
        f"Executed answer: {executed_str}\n"
    )

    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            f'Valid non-empty candidates: {json.dumps(candidates)}\n'
            f'Available selected_source values for this question: ["executed_plan", "recomputed", "none"]\n'
            f'Verifier report: "(no verifier in this setting)"\n'
            "The executed_plan answer was computed deterministically from the canonical ledger. "
            "Select it or recompute. Begin immediately with `Selected source:`.")
    finalizer = call_extended_finalizer(model, prompts["finalizer"], user, candidates,
                                          {"executed_plan", "recomputed", "none"})

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "ledger_exec_plan",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "finalizer_event": finalizer,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "executed_plan_result": executed_result,
        "executed_plan_correct": execution_correct,
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "exec_ground": {
            "fact_method": fact_method,
            "typed_facts_A": [f.to_dict() for f in facts_a],
            "typed_facts_B": [f.to_dict() for f in facts_b],
            "ledger_fact_count": len(ledger.facts),
            "ledger_conflict_count": len(ledger.conflicts),
            "ledger_text": ledger.to_text(),
            "fact_quality_A": fact_quality_a,
            "fact_quality_B": fact_quality_b,
            "plan_json": plan.to_json(),
            "plan_raw": plan_raw,
            "plan_step_count": len(plan.steps),
            "plan_executable": executed_result is not None,
            "plan_execution_correct": execution_correct,
        },
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    trace["candidate_appearances"].append({
        "source": "executed_plan", "phase": "plan_execution",
        "answer": executed_str,
        "information_complete_at_appearance": True,
        "correct": execution_correct,
    })
    usage = blank_usage()
    add_usage(usage, plan_usage)
    for event in collect_events(trace):
        add_usage(usage, event.get("token_usage", blank_usage()))
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


def build_ledger_exec_plan_verify_trace(model, prompts, item, qid, base_seed,
                                         discussion_rounds=2, use_gold_facts=False,
                                         max_verify_rounds=2):
    """Setting 5: TypedFact → Ledger → Plan → Coverage Verify → Fix Loop → Finalizer.

    Full ExecGround pipeline with verification loop.
    """
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "ledger_exec_verify"))

    # Phase 1: Extract typed facts
    if use_gold_facts:
        facts_a = eg.extract_gold_typed_facts(item["condition_A"], "A")
        facts_b = eg.extract_gold_typed_facts(item["condition_B"], "B")
        fact_method = "gold_rule_based"
    else:
        reseed_model(model, derived_seed(base_seed, qid, "typed_fact_A"))
        facts_a = eg.extract_typed_facts(model, item["condition_A"], "A", temperature=0.0)
        reseed_model(model, derived_seed(base_seed, qid, "typed_fact_B"))
        facts_b = eg.extract_typed_facts(model, item["condition_B"], "B", temperature=0.0)
        fact_method = "llm_extraction"

    # Phase 2: Build canonical ledger
    ledger = eg.CanonicalLedger.build(facts_a, facts_b, item)
    fact_quality_a = compute_typed_fact_quality(facts_a, item["condition_A"])
    fact_quality_b = compute_typed_fact_quality(facts_b, item["condition_B"])

    # Phase 3: Fresh solver → executable plan
    reseed_model(model, derived_seed(base_seed, qid, "exec_plan_solver"))
    plan, plan_raw, plan_usage = eg.fresh_solve(
        model, item["shared_question"], ledger, temperature=0.0)

    # Phase 4: Coverage verification + fix loop
    verify_rounds = []
    current_plan = plan
    total_fix_usage = blank_usage()

    for round_no in range(max_verify_rounds + 1):
        report = eg.verify_coverage(current_plan, ledger, gold)

        round_info = {
            "round": round_no,
            "is_clean": report.is_clean,
            "facts_total": report.facts_total,
            "facts_used": report.facts_used,
            "missing_facts": report.missing_facts,
            "unbound_variables": report.unbound_variables,
            "executable": report.executable,
            "computed_result": report.computed_result,
            "expected_result": report.expected_result,
            "result_matches": report.result_matches,
            "fix_hints": report.fix_hints,
        }
        verify_rounds.append(round_info)

        if report.is_clean or round_no >= max_verify_rounds:
            break

        if not report.has_fixable_issues:
            break

        # Fix iteration
        fix_prompt = eg.generate_fix_prompt(report, current_plan, ledger)
        fix_user = (
            f"QUESTION:\n{item['shared_question']}\n\n"
            f"{ledger.to_text()}\n\n"
            f"{fix_prompt}"
        )
        reseed_model(model, derived_seed(base_seed, qid, f"fix_round_{round_no}"))
        fix_raw, fix_usage, _ = model.call(eg.FRESH_SOLVER_SYSTEM, fix_user, temperature=0.0)
        current_plan = eg._parse_plan_json(fix_raw)
        round_info["fix_raw"] = fix_raw
        add_usage(total_fix_usage, fix_usage)

    # Phase 5: Execute final plan
    final_result = eg.execute_plan(current_plan, ledger)
    final_correct = (
        final_result is not None
        and decimal(gold) is not None
        and math.isclose(final_result, float(decimal(gold)), rel_tol=1e-9)
    )

    # Phase 6: Finalizer
    final_str = f"{final_result}" if final_result is not None else "EXECUTION FAILED"
    candidates = {"verified_plan": final_str}

    evidence = (
        f"{ledger.to_text()}\n\n"
        f"--- Final Executable Plan (after {len(verify_rounds)} verify rounds) ---\n"
        f"{json.dumps(current_plan.to_json(), indent=2)}\n\n"
        f"--- Verification Summary ---\n"
        f"Final round clean: {verify_rounds[-1]['is_clean'] if verify_rounds else 'N/A'}\n"
        f"Facts used: {verify_rounds[-1].get('facts_used', '?')}/{verify_rounds[-1].get('facts_total', '?')}\n"
        f"Executable: {verify_rounds[-1].get('executable', '?')}\n"
        f"Result: {final_str}\n"
    )

    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            f'Valid non-empty candidates: {json.dumps(candidates)}\n'
            f'Available selected_source values for this question: ["verified_plan", "recomputed", "none"]\n'
            f'Verifier report: "(coverage verification already performed)"\n'
            "The verified_plan was checked for completeness and correctness. "
            "Select it or recompute. Begin immediately with `Selected source:`.")
    finalizer = call_extended_finalizer(model, prompts["finalizer"], user, candidates,
                                          {"verified_plan", "recomputed", "none"})

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "ledger_exec_plan_verify",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "finalizer_event": finalizer,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "verified_plan_result": final_result,
        "verified_plan_correct": final_correct,
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "exec_ground": {
            "fact_method": fact_method,
            "typed_facts_A": [f.to_dict() for f in facts_a],
            "typed_facts_B": [f.to_dict() for f in facts_b],
            "ledger_fact_count": len(ledger.facts),
            "ledger_conflict_count": len(ledger.conflicts),
            "ledger_text": ledger.to_text(),
            "fact_quality_A": fact_quality_a,
            "fact_quality_B": fact_quality_b,
            "initial_plan_json": plan.to_json(),
            "initial_plan_raw": plan_raw,
            "final_plan_json": current_plan.to_json(),
            "verify_rounds": verify_rounds,
            "total_verify_rounds": len(verify_rounds),
            "final_plan_clean": verify_rounds[-1]["is_clean"] if verify_rounds else False,
        },
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    trace["candidate_appearances"].append({
        "source": "verified_plan", "phase": "verified_execution",
        "answer": final_str,
        "information_complete_at_appearance": True,
        "correct": final_correct,
    })
    usage = blank_usage()
    add_usage(usage, plan_usage)
    add_usage(usage, total_fix_usage)
    for event in collect_events(trace):
        add_usage(usage, event.get("token_usage", blank_usage()))
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


# ── Builder dispatch ──

BUILDERS = {
    "free_discussion":          build_free_discussion_trace,
    "reveal_all":               build_reveal_all_trace,
    "canonical_ledger":         build_canonical_ledger_trace,
    "ledger_fresh_solver":      build_ledger_fresh_solver_trace,
    "ledger_exec_plan":         build_ledger_exec_plan_trace,
    "ledger_exec_plan_verify":  build_ledger_exec_plan_verify_trace,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DISCUSSION WITH LEDGER (for Setting 2)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_discussion_with_ledger(
    model: LocalQwen,
    solver_prompt: str,
    item: dict,
    ledger: eg.CanonicalLedger,
    rounds_count: int = 2,
) -> dict:
    """Run a two-agent discussion where both agents see the canonical ledger."""
    events: list[dict] = []
    round_records: list[dict] = []
    ledger_text = ledger.to_text()

    for round_no in range(1, rounds_count + 1):
        pre_round_transcript = "\n".join(x for x in (public_transcript(events),) if x)
        outbound_specs = {}
        for side in ("A", "B"):
            purpose = (
                "Share your reasoning and state exactly what information is still missing."
                if round_no == 1
                else "Correct mistakes, fill gaps, and advance the solution using earlier messages."
            )
            user = DISCUSSION_WITH_LEDGER_USER.format(
                side=side.lower(),
                round_no=round_no,
                total_rounds=rounds_count,
                purpose=purpose,
                question=item["shared_question"],
                ledger_text=ledger_text,
                transcript=pre_round_transcript or "(nothing disclosed yet)",
            )
            outbound_specs[side] = (f"solver_{side.lower()}", user, None)
        outbound = paired_model_events(model, solver_prompt, outbound_specs, temperature=0.0)
        for side in ("A", "B"):
            event = outbound[side]
            event["phase"] = f"discussion_round_{round_no}_send"
            event["round"] = round_no
            event["stage"] = "send"
            event["current_answer"], event["current_answer_extraction"] = extract_free_text_answer(
                event["raw_output"], "Current answer")
            event["current_answer_explicit"] = event["current_answer_extraction"].startswith("explicit_")
        events.extend([outbound["A"], outbound["B"]])
        round_records.append({
            "round": round_no, "purpose": purpose,
            "pre_round_public_transcript": pre_round_transcript or "(nothing disclosed yet)",
            "simultaneous_turn": {side.lower(): outbound[side] for side in ("A", "B")},
        })

    transcript = "\n".join(x for x in (public_transcript(events),) if x)
    final_specs = {}
    for side in ("A", "B"):
        user = (
            f'Role: solver_{side.lower()}\nShared question: {item["shared_question"]}\n'
            f'{ledger_text}\n\n'
            f'Public transcript after {rounds_count} symmetric rounds:\n{transcript or "(nothing disclosed yet)"}\n'
            "Use the canonical ledger as the authoritative source. "
            "Put `Final answer: ...` on the FIRST line, then at most three sentences of reasoning. "
            "Use natural text; do not output JSON."
        )
        final_specs[side] = (f"solver_{side.lower()}", user, None)
    final_batch = paired_model_events(model, solver_prompt, final_specs, temperature=0.0)
    finals = {}
    for side in ("A", "B"):
        event = final_batch[side]
        answer, error = parse_solver_final(event.get("raw_output", ""))
        event.update(phase="solver_final", answer=answer,
                     answer_extraction="strict_solver_final" if not error else "invalid_format",
                     validation_error=error, invalid_output=bool(error))
        finals[side.lower()] = event

    return {
        "protocol": "symmetric_with_canonical_ledger",
        "round_records": round_records,
        "discussion_events": events,
        "public_transcript": transcript or "(nothing disclosed yet)",
        "solver_finals": finals,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_setting_metrics(traces: list[dict], setting: str) -> dict:
    """Compute comprehensive metrics for a setting."""
    total = len(traces)
    if total == 0:
        return {"setting": setting, "total_traces": 0}

    correct = sum(1 for t in traces if t.get("semantic_correct"))
    format_ok = sum(1 for t in traces if t.get("format_compliant"))
    reason_consistent = sum(1 for t in traces if t.get("answer_reason_consistent"))

    # Candidate emergence
    emergence_data = []
    for t in traces:
        gold = t.get("gold_answer", "")
        emergence_data.append(compute_candidate_emergence(t, gold))

    emerged = sum(1 for e in emergence_data if e["correct_candidate_emerged"])
    emerged_complete = sum(1 for e in emergence_data if e["emerged_with_complete_information"])

    # Fact distortion (for settings with discussion)
    distortion_rates = []
    for t in traces:
        disc = t.get("discussion", {})
        all_outputs = []
        for event in disc.get("discussion_events", []):
            all_outputs.append(event.get("raw_output", ""))
        facts_a = t.get("injected_facts", {}).get("A", "")
        facts_b = t.get("injected_facts", {}).get("B", "")
        if isinstance(facts_a, str) and isinstance(facts_b, str):
            required = [facts_a, facts_b]
            distortion_rates.append(compute_fact_distortion_rate(all_outputs, required))

    avg_distortion = (
        sum(d["distortion_rate"] for d in distortion_rates) / len(distortion_rates)
        if distortion_rates else 0.0
    )

    # ExecGround-specific metrics
    eg_metrics = {}
    eg_traces = [t for t in traces if "exec_ground" in t]
    if eg_traces:
        ledger_fact_counts = [t["exec_ground"]["ledger_fact_count"] for t in eg_traces]
        eg_metrics["avg_ledger_fact_count"] = sum(ledger_fact_counts) / len(ledger_fact_counts)

        fact_qualities_a = [t["exec_ground"].get("fact_quality_A", {}).get("coverage", 0)
                           for t in eg_traces]
        fact_qualities_b = [t["exec_ground"].get("fact_quality_B", {}).get("coverage", 0)
                           for t in eg_traces]
        eg_metrics["avg_fact_quality_A"] = sum(fact_qualities_a) / max(1, len(fact_qualities_a))
        eg_metrics["avg_fact_quality_B"] = sum(fact_qualities_b) / max(1, len(fact_qualities_b))

    # Plan-specific metrics
    plan_traces = [t for t in traces
                   if t.get("setting") in ("ledger_exec_plan", "ledger_exec_plan_verify")]
    if plan_traces:
        plan_exec_correct = sum(
            1 for t in plan_traces
            if t.get("executed_plan_correct") or t.get("verified_plan_correct")
        )
        eg_metrics["plan_execution_correct"] = plan_exec_correct
        eg_metrics["plan_execution_correct_rate"] = plan_exec_correct / len(plan_traces)

        plan_steps = [
            t.get("exec_ground", {}).get("plan_step_count", 0)
            for t in plan_traces
        ]
        eg_metrics["avg_plan_steps"] = sum(plan_steps) / max(1, len(plan_steps))

    # Verify-specific metrics
    verify_traces = [t for t in traces if t.get("setting") == "ledger_exec_plan_verify"]
    if verify_traces:
        clean_plans = sum(
            1 for t in verify_traces
            if t.get("exec_ground", {}).get("final_plan_clean", False)
        )
        eg_metrics["final_plan_clean_rate"] = clean_plans / len(verify_traces)
        avg_rounds = sum(
            len(t.get("exec_ground", {}).get("verify_rounds", []))
            for t in verify_traces
        ) / len(verify_traces)
        eg_metrics["avg_verify_rounds"] = avg_rounds

    return {
        "setting": setting,
        "label": SETTING_LABELS.get(setting, setting),
        "total_traces": total,
        "semantic_accuracy": correct / total,
        "correct_count": correct,
        "format_compliance": format_ok / total,
        "answer_reason_consistency": reason_consistent / total,
        "correct_candidate_emergence": emerged / total,
        "correct_candidate_emerged_with_complete_info": emerged_complete / total,
        "avg_fact_distortion_rate": avg_distortion,
        "exec_ground_metrics": eg_metrics,
    }


def run_experiment(
    model: LocalQwen,
    items: list[dict],
    settings: list[str],
    seeds: int = 3,
    discussion_rounds: int = 2,
    use_gold_facts: bool = False,
    max_verify_rounds: int = 2,
    base_seed: int = DEFAULT_SEED,
    output_dir: Path | None = None,
    resume: bool = False,
) -> tuple[list[dict], dict]:
    """Run the six-group ablation experiment.

    If output_dir is provided, each trace is saved immediately after completion
    (incremental write). Combined with resume=True, already-completed traces
    are skipped on restart.

    Returns (all_traces, metrics_by_setting).
    """
    prompts = load_prompts()
    all_traces: list[dict] = []

    # ── Incremental save setup ──
    traces_jsonl_path: Path | None = None
    completed_keys: set[tuple[str, int, int]] = set()  # (setting, seed, qid)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        traces_jsonl_path = output_dir / "traces_all.jsonl"

        if resume and traces_jsonl_path.exists():
            # Load already-completed traces
            with open(traces_jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        all_traces.append(t)
                        completed_keys.add((
                            t.get("setting", ""),
                            t.get("seed", 0),
                            t.get("question_id", 0),
                        ))
                    except json.JSONDecodeError:
                        pass
            print(f"  Resumed {len(all_traces)} previously completed traces from {traces_jsonl_path}")

    # ── Run ──
    total_combinations = len(settings) * len(items) * seeds
    current = len(all_traces)

    for setting in settings:
        builder = BUILDERS.get(setting)
        if builder is None:
            print(f"  WARNING: Unknown setting '{setting}', skipping")
            continue

        label = SETTING_LABELS.get(setting, setting)
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        for seed_idx in range(seeds):
            for item_idx, item in enumerate(items):
                current += 1
                qid = item_idx + 1

                # Skip already-completed traces when resuming
                if (setting, seed_idx + 1, qid) in completed_keys:
                    print(f"  [{current}/{total_combinations}] Q{qid:02d} seed={seed_idx+1} ... (skipped, already done)")
                    continue

                print(f"  [{current}/{total_combinations}] Q{qid:02d} seed={seed_idx+1} ...", end=" ", flush=True)

                try:
                    # leder_exec_plan_verify has an extra max_verify_rounds param
                    if setting == "ledger_exec_plan_verify":
                        trace = builder(model, prompts, item, qid, base_seed,
                                       discussion_rounds, use_gold_facts, max_verify_rounds)
                    elif setting in ("canonical_ledger", "ledger_fresh_solver",
                                     "ledger_exec_plan"):
                        trace = builder(model, prompts, item, qid, base_seed,
                                       discussion_rounds, use_gold_facts)
                    else:
                        trace = builder(model, prompts, item, qid, base_seed, discussion_rounds)

                    trace["seed"] = seed_idx + 1
                    all_traces.append(trace)

                    # ── Immediate write to disk ──
                    if traces_jsonl_path is not None:
                        with open(traces_jsonl_path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                            fh.flush()
                            os.fsync(fh.fileno())

                    status = "OK" if trace.get("semantic_correct") else "X"
                    pred = trace.get("final_prediction", "?")
                    gold = trace.get("gold_answer", "?")
                    print(f"{status} pred={pred} gold={gold}")

                except Exception as exc:
                    print(f"ERROR: {exc}")
                    import traceback
                    traceback.print_exc()

    # Compute metrics
    metrics_by_setting = {}
    for setting in settings:
        setting_traces = [t for t in all_traces if t.get("setting") == setting]
        metrics_by_setting[setting] = compute_setting_metrics(setting_traces, setting)

    return all_traces, metrics_by_setting


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ExecGround Six-Group Ablation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=0,
                       help="Max questions to use (0 = all)")
    parser.add_argument("--seeds", type=int, default=3,
                       help="Number of seeds per question per setting")
    parser.add_argument("--rounds", type=int, default=2,
                       help="Discussion rounds")
    parser.add_argument("--setting", type=str, default="",
                       help="Run a single setting only")
    parser.add_argument("--settings", type=str, nargs="*", default=None,
                       help="Run specific settings (space-separated)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--use-gold-facts", action="store_true",
                       help="Use rule-based gold facts instead of LLM extraction")
    parser.add_argument("--max-verify-rounds", type=int, default=2,
                       help="Max coverage verification rounds")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default="",
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing output-dir, skipping already-completed traces")
    args = parser.parse_args()

    # Determine settings
    if args.setting:
        settings = [args.setting]
    elif args.settings:
        settings = list(args.settings)
    else:
        settings = list(EXEC_GROUND_SETTINGS)

    # Determine output dir early (needed for incremental writes)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_BASE_DIR / timestamp

    print(f"ExecGround Six-Group Ablation Experiment")
    print(f"  Output dir: {output_dir}")
    print(f"  Settings: {settings}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Use gold facts: {args.use_gold_facts}")
    print(f"  Max verify rounds: {args.max_verify_rounds}")
    print(f"  Resume: {args.resume}")

    # Load data
    items = read_json_records(DATA_PATH)
    if args.limit > 0:
        items = items[:args.limit]
    print(f"  Questions: {len(items)}")

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = LocalQwen(
        MODEL_PATH,
        args.device,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_TEMPERATURE,
        False,
    )
    print("  Model loaded.")

    # Run experiment (traces saved incrementally to output_dir/traces_all.jsonl)
    all_traces, metrics = run_experiment(
        model=model,
        items=items,
        settings=settings,
        seeds=args.seeds,
        discussion_rounds=args.rounds,
        use_gold_facts=args.use_gold_facts,
        max_verify_rounds=args.max_verify_rounds,
        base_seed=args.base_seed,
        output_dir=output_dir,
        resume=args.resume,
    )

    # Save final combined traces (single JSON for easy loading)
    traces_path = output_dir / "traces_all.json"
    traces_path.write_text(json.dumps(all_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTraces saved to: {traces_path}")

    # Also save JSONL if not already (for resume compatibility)
    jsonl_path = output_dir / "traces_all.jsonl"
    if not jsonl_path.exists():
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for t in all_traces:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Save metrics
    metrics_path = output_dir / "exec_ground_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metrics saved to: {metrics_path}")

    # Print summary
    print(f"\n{'='*90}")
    print("  EXECGROUND SIX-GROUP ABLATION RESULTS")
    print(f"{'='*90}")
    print(f"{'Setting':<30} {'Acc':>6} {'Emerge':>7} {'Format':>7} {'#Traces':>8}")
    print(f"{'-'*30} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")

    for setting in settings:
        m = metrics.get(setting, {})
        if not m:
            continue
        label = SETTING_LABELS.get(setting, setting)[:28]
        acc = m.get("semantic_accuracy", 0)
        emerge = m.get("correct_candidate_emergence", 0)
        fmt = m.get("format_compliance", 0)
        n = m.get("total_traces", 0)
        print(f"{label:<30} {acc:6.1%} {emerge:7.1%} {fmt:7.1%} {n:>8}")

    print(f"{'='*90}")

    # Layer-by-layer emergence analysis
    print(f"\n--- Layer-by-Layer Correct Candidate Emergence ---")
    for setting in settings:
        m = metrics.get(setting, {})
        if not m:
            continue
        acc = m.get("semantic_accuracy", 0)
        emerge = m.get("correct_candidate_emergence", 0)
        emerge_ci = m.get("correct_candidate_emerged_with_complete_info", 0)
        egm = m.get("exec_ground_metrics", {})
        label = SETTING_LABELS.get(setting, setting)
        print(f"  {label}:")
        print(f"    Final accuracy:     {acc:.1%}")
        print(f"    Candidate emerged:  {emerge:.1%}")
        print(f"    Emerged w/ complete: {emerge_ci:.1%}")
        if egm:
            for k, v in egm.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")

    # Generate analysis report
    report_path = output_dir / "exec_ground_analysis_report.md"
    _generate_report(metrics, settings, report_path)
    print(f"\nAnalysis report saved to: {report_path}")


def _generate_report(metrics: dict, settings: list[str], path: Path):
    """Generate a markdown analysis report."""
    lines = [
        "# ExecGround Six-Group Ablation Experiment — Analysis Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Settings**: {len(settings)}",
        "",
        "## Layer-by-Layer Correct Candidate Emergence",
        "",
        "The key question: at which layer does the correct answer first emerge as a candidate?",
        "",
        "| # | Setting | Accuracy | Candidate Emergence | Emerged w/ Complete Info | Format OK |",
        "|---|---------|----------|--------------------|-------------------------|----------|",
    ]

    for i, setting in enumerate(settings):
        m = metrics.get(setting, {})
        if not m:
            continue
        acc = m.get("semantic_accuracy", 0)
        emerge = m.get("correct_candidate_emergence", 0)
        emerge_ci = m.get("correct_candidate_emerged_with_complete_info", 0)
        fmt = m.get("format_compliance", 0)
        lines.append(
            f"| {i} | {setting} | {acc:.1%} | {emerge:.1%} | {emerge_ci:.1%} | {fmt:.1%} |"
        )

    lines += [
        "",
        "## Causal Attribution",
        "",
        "Each layer adds one capability. The Δ in emergence rate reveals the bottleneck:",
        "",
    ]

    prev_emerge = 0.0
    for i, setting in enumerate(settings):
        m = metrics.get(setting, {})
        if not m:
            continue
        emerge = m.get("correct_candidate_emergence", 0)
        delta = emerge - prev_emerge
        label = SETTING_LABELS.get(setting, setting)

        if i == 0:
            lines.append(f"- **{setting}** (baseline): emergence = {emerge:.1%}")
        else:
            lines.append(f"- **{setting}**: emergence = {emerge:.1%} (Δ = {delta:+.1%})")
            if delta > 0.05:
                lines.append(f"  - ★ MAJOR BOTTLENECK: {label} contributes {delta:.0%} improvement")
            elif delta > 0.01:
                lines.append(f"  - Minor improvement: +{delta:.0%}")
            else:
                lines.append(f"  - Negligible effect: {delta:+.1%}")

        prev_emerge = emerge

    lines += [
        "",
        "## Interpretation Guide",
        "",
        "1. If **ledger + fresh solver** already recovers most accuracy:",
        "   → Main problem is discussion history contamination and state confusion.",
        "   → Keep the architecture simple.",
        "",
        "2. If **executable plan** is needed for significant recovery:",
        "   → Core problem is cross-fact dependency construction failure.",
        "   → Focus on dependency graph and programmatic reasoning.",
        "",
        "3. If **even oracle state + oracle plan** cannot recover:",
        "   → Re-examine problem splitting, data labeling, and model base capability.",
        "   → Do NOT rush to write paper conclusions.",
        "",
        "## ExecGround-Specific Metrics",
        "",
    ]

    for setting in settings:
        m = metrics.get(setting, {})
        egm = m.get("exec_ground_metrics", {})
        if not egm:
            continue
        lines.append(f"### {setting}")
        for k, v in egm.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.3f}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
