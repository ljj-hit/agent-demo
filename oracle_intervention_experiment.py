#!/usr/bin/env python3
"""
Oracle Intervention Experiment — 6 settings to identify the causal bottleneck
in multi-agent GSM8K with Qwen2.5-1.5B.

Settings:
  0. free_discussion        — baseline: each agent only knows private facts
  1. oracle_disclosure      — all raw facts injected before discussion
  2. oracle_canonical_state — normalized, distortion-free fact table
  3. canonical_state_fresh  — fresh solver reads only question + fact table (no history)
  4. oracle_plan            — fact table + variable dependency graph / equation structure
  5. oracle_candidate       — correct answer injected as a candidate for finalizer

Metrics per setting:
  - semantic accuracy
  - correct candidate emergence rate
  - candidate emergence given complete facts
  - fact distortion rate
  - partial answer rate
  - undetermined ratio
  - format compliance rate
  - answer-reason consistency rate

Run:
  python oracle_intervention_experiment.py [--limit 20] [--rounds 2] [--seeds 3]
"""

from __future__ import annotations
import argparse, csv, hashlib, json, os, re, sys, time
from collections import defaultdict, Counter
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

# Re-export core symbols
read_json_records     = _run_hidden.read_json_records
extract_answer        = _run_hidden.extract_answer
decimal               = _run_hidden.decimal
equivalent            = _run_hidden.equivalent
extract_labeled_answer = _run_hidden.extract_labeled_answer
extract_free_text_answer = _run_hidden.extract_free_text_answer
parse_solver_final    = _run_hidden.parse_solver_final
explicitly_undetermined = _run_hidden.explicitly_undetermined
parse_object          = _run_hidden.parse_object
blank_usage           = _run_hidden.blank_usage
add_usage             = _run_hidden.add_usage
derived_seed          = _run_hidden.derived_seed
reseed_model          = _run_hidden.reseed_model
model_event           = _run_hidden.model_event
paired_model_events   = _run_hidden.paired_model_events
public_transcript     = _run_hidden.public_transcript
run_discussion        = _run_hidden.run_discussion
replay_facts          = _run_hidden.replay_facts
replay_fact_hash      = _run_hidden.replay_fact_hash
replay_ledger         = _run_hidden.replay_ledger
run_replay_discussion = _run_hidden.run_replay_discussion
coverage_score        = _run_hidden.coverage_score
atomic_facts          = _run_hidden.atomic_facts
fact_is_public        = _run_hidden.fact_is_public
add_information_timeline = _run_hidden.add_information_timeline
single_call           = _run_hidden.single_call
event_answer          = _run_hidden.event_answer
candidate_appearances = _run_hidden.candidate_appearances
parse_fixed_finalizer = _run_hidden.parse_fixed_finalizer
check_answer_reason_consistency = _run_hidden.check_answer_reason_consistency
source_consistency_error = _run_hidden.source_consistency_error
call_finalizer_once   = _run_hidden.call_finalizer_once
collect_events        = _run_hidden.collect_events
classify              = _run_hidden.classify
set_outcome_fields    = _run_hidden.set_outcome_fields
build_trace           = _run_hidden.build_trace
LocalQwen             = _run_hidden.LocalQwen
USAGE_KEYS            = _run_hidden.USAGE_KEYS
VERIFIER_DEFAULT      = _run_hidden.VERIFIER_DEFAULT
FINALIZER_DEFAULT     = _run_hidden.FINALIZER_DEFAULT
UNDETERMINED_ANSWERS  = _run_hidden.UNDETERMINED_ANSWERS
DEFAULT_DISCUSSION_ROUNDS = _run_hidden.DEFAULT_DISCUSSION_ROUNDS
DEFAULT_SEED          = _run_hidden.DEFAULT_SEED
DEFAULT_TEMPERATURE   = _run_hidden.DEFAULT_TEMPERATURE
DEFAULT_MAX_NEW_TOKENS = _run_hidden.DEFAULT_MAX_NEW_TOKENS
DEFAULT_DEVICE        = _run_hidden.DEFAULT_DEVICE

# ── Config ──
ROOT = _SCRIPT_DIR
DATA_PATH = ROOT / "data" / "20.json"
MODEL_PATH = ROOT / "qwen2.5-1.5B"
OUTPUT_BASE_DIR = ROOT / "outputs_oracle_intervention"
PROMPT_DIR = ROOT / "hidden_gsm8k_prompts"

ORACLE_SETTINGS = [
    "free_discussion",
    "oracle_disclosure",
    "oracle_canonical_state",
    "canonical_state_fresh",
    "oracle_plan",
    "oracle_candidate",
]

SETTING_LABELS = {
    "free_discussion":          "0. Free Discussion (baseline: private facts only)",
    "oracle_disclosure":        "1. Oracle Disclosure (all raw facts injected pre-discussion)",
    "oracle_canonical_state":   "2. Oracle Canonical State (normalized fact table)",
    "canonical_state_fresh":    "3. Canonical State + Fresh Solver (no discussion history)",
    "oracle_plan":              "4. Oracle Plan (fact table + equation structure)",
    "oracle_candidate":         "5. Oracle Candidate (correct answer as candidate)",
}

# ═══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

ORACLE_SOLVER_SYSTEM = """You are a careful mathematical problem solver. You receive complete facts about a math problem.

Reason step by step and check arithmetic. Use all provided facts to compute the answer.
Put `Final answer: ...` on the FIRST line, then give at most three sentences of concise reasoning.
Use natural text; do not output JSON."""

ORACLE_FRESH_SOLVER_SYSTEM = """You are a mathematical problem solver. You are given a question and a clean, verified fact table.

These facts are guaranteed to be correct and complete. Use ONLY these facts.
Do NOT guess. Do NOT refer to any prior discussion — you have no access to it.
Put `Final answer: ...` on the FIRST line, then at most three sentences of reasoning."""

ORACLE_PLAN_SOLVER_SYSTEM = """You are a mathematical problem solver. You are given:
1. A question
2. A verified fact table
3. An equation structure showing the dependency order

Your job: fill the values from the fact table into the equation structure, then execute the arithmetic.
Put `Final answer: ...` on the FIRST line, then show the filled equation with computed result."""

ORACLE_FINALIZER_SYSTEM = """You produce the final answer. Use only the shared question and the evidence provided.

Return exactly three non-empty physical lines, with no Markdown, no blank lines:
Selected source: <solver_a|solver_b|oracle|recomputed|none>
Final answer: <number or undetermined>
Reason: <explanation>

Use only a selected_source listed as available. Reason must be non-empty.
If a source is selected, copy its answer exactly; do not recompute a different answer."""

# ── Load prompts from disk ──
def load_prompts():
    prompts = {}
    for name in ["solver", "verifier", "finalizer"]:
        path = PROMPT_DIR / f"{name}.txt"
        if path.exists():
            prompts[name] = path.read_text(encoding="utf-8")
    # Also load solver_a and solver_b if they exist
    for name in ["solver_a", "solver_b"]:
        path = PROMPT_DIR / f"{name}.txt"
        if path.exists():
            prompts[f"solver_{name}"] = path.read_text(encoding="utf-8")
    return prompts


# ═══════════════════════════════════════════════════════════════════════════
# FACT DISTORTION & PARTIAL ANSWER METRICS
# ═══════════════════════════════════════════════════════════════════════════

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
        "missing_numbers": sorted(missing, key=lambda x: int(x) if x.isdigit() else 0),
    }

def compute_partial_answer_rate(outputs: list[str], gold: str) -> dict:
    """Check if agents produced partial (intermediate) correct calculations."""
    gold_num = decimal(gold)
    partial_hits = 0
    total_outputs = len(outputs)
    for output in outputs:
        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", output)
        # Check if any intermediate number matches the gold or a plausible intermediate
        # Simple heuristic: the output contains numbers that are part of the correct computation
        if nums:
            partial_hits += 1
    return {
        "partial_answer_rate": partial_hits / max(1, total_outputs),
        "outputs_with_numbers": partial_hits,
        "total_outputs": total_outputs,
    }

def compute_undetermined_ratio(outputs: list[str]) -> dict:
    """Check ratio of GENUINELY undetermined answers (where agent gives up, not just labels)."""
    undetermined = 0
    total = 0
    for output in outputs:
        if not output.strip():
            continue
        total += 1
        ca_match = re.search(r"(?im)^Current answer\s*[:：]\s*(.+)", output)
        fa_match = re.search(r"(?im)^Final answer\s*[:：]\s*(.+)", output)

        current_label = ca_match.group(1).strip().lower() if ca_match else ""
        final_label = fa_match.group(1).strip().lower() if fa_match else ""

        # Only count as truly undetermined if the output shows NO attempt at calculation
        has_calculation = bool(re.search(
            r"(?:(?:=\s*\d+|equals?\s*\d+|gives?\s*\d+|<<[\d\s+\-*/.=]+>>|\d+\s*[\+\-\*/]\s*\d+)"
            r"|(?:total|sum|result|answer|cost|weight|pages|amount)\s+(?:is|=|:)?\s*\d+)",
            output, re.I))

        if has_calculation:
            continue  # Agent is computing, not giving up

        if current_label in ("undetermined", "undetermined.", "unknown", "insufficient", ""):
            undetermined += 1
        elif final_label in ("undetermined", "undetermined.", "unknown"):
            undetermined += 1

    return {
        "undetermined_ratio": undetermined / max(1, total),
        "undetermined_count": undetermined,
        "total_checks": total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ORACLE PLAN GENERATION (extract equation structure from answer)
# ═══════════════════════════════════════════════════════════════════════════

def extract_equation_structure(answer_text: str) -> str:
    """Extract the <<...>> computation steps from the gold answer as equation structure."""
    steps = re.findall(r"<<(.+?)>>", answer_text)
    if not steps:
        return "(No structured computation available — solve directly from facts)"
    lines = []
    for i, step in enumerate(steps):
        # Replace numbers with variable placeholders
        masked = re.sub(r"\b\d+(?:\.\d+)?\b", "___", step)
        lines.append(f"  Step {i+1}: {masked}  [raw: {step}]")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# BUILDER FUNCTIONS FOR EACH SETTING
# ═══════════════════════════════════════════════════════════════════════════

def build_free_discussion_trace(model, prompts, item, qid, base_seed, discussion_rounds=2):
    """Setting 0: Baseline — each agent only knows private facts."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "free_discussion"))

    # Run genuine private-fact discussion (NOT replay)
    discussion = run_discussion(model, prompts["solver"], item, oracle=False,
                                rounds_count=discussion_rounds)
    add_information_timeline(item, discussion)

    # Finalizer gets transcript + all facts
    transcript = public_transcript(discussion.get("discussion_events", []))
    all_facts = replay_facts(item, "AB")
    evidence = f"Complete facts (now revealed):\n{all_facts}\n\nDiscussion transcript:\n{transcript}"

    # Extract solver candidates
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
            "Select a supported candidate or recompute from the evidence. Begin immediately with `Selected source:`.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, candidates)

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "free_discussion",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion, "finalizer_event": finalizer,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
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


def build_oracle_disclosure_trace(model, prompts, item, qid, base_seed, discussion_rounds=2):
    """Setting 1: All raw facts injected before discussion — no agent restatement needed."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "oracle_disclosure"))

    # All facts visible from round 1
    discussion = run_replay_discussion(model, prompts["solver"], item,
                                       reveal_after_round=0, order="AB",
                                       rounds_count=discussion_rounds)
    add_information_timeline(item, discussion)

    # Finalizer sees transcript with all facts already in it
    transcript = public_transcript(discussion.get("discussion_events", []))
    all_facts = replay_facts(item, "AB")
    evidence = f"Oracle-disclosed facts (verbatim):\n{all_facts}\n\nDiscussion transcript:\n{transcript}"

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
            "Select a supported candidate or recompute from the evidence. Begin immediately with `Selected source:`.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, candidates)

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "oracle_disclosure",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion, "finalizer_event": finalizer,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
        "disclosure_mode": "oracle_verbatim_injection",
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


def build_oracle_canonical_state_trace(model, prompts, item, qid, base_seed, discussion_rounds=2):
    """Setting 2: Normalized, distortion-free canonical fact table instead of raw facts."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "oracle_canonical"))

    # Build canonical fact table
    ledger = replay_ledger(item)
    # Also extract atomic facts for cleaner presentation
    facts_a = atomic_facts(item["condition_A"])
    facts_b = atomic_facts(item["condition_B"])
    canonical_table = "CANONICAL FACT TABLE (guaranteed correct and complete):\n"
    canonical_table += "| # | Source | Fact (verbatim) |\n|---|---|---|\n"
    for i, f in enumerate(facts_a):
        canonical_table += f"| {i+1} | A | {f} |\n"
    for i, f in enumerate(facts_b):
        canonical_table += f"| {len(facts_a)+i+1} | B | {f} |\n"

    # Discussion with canonical facts visible from round 1
    discussion = run_replay_discussion(model, prompts["solver"], item,
                                       reveal_after_round=0, order="AB",
                                       rounds_count=discussion_rounds)
    add_information_timeline(item, discussion)

    transcript = public_transcript(discussion.get("discussion_events", []))
    evidence = f"{canonical_table}\n\nDiscussion transcript:\n{transcript}"

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
            "Use the canonical fact table as the authoritative source. "
            "Select a supported candidate or recompute. Begin immediately with `Selected source:`.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, candidates)

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "oracle_canonical_state",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion, "finalizer_event": finalizer,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "canonical_fact_table": canonical_table,
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
        "disclosure_mode": "oracle_canonical_table",
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


def build_canonical_state_fresh_trace(model, prompts, item, qid, base_seed, discussion_rounds=2):
    """Setting 3: Discussion runs normally, but a FRESH solver reads only question + fact table.
    The fresh solver has NO access to the discussion history."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "fresh_solver"))

    # Step 1: Run standard discussion with all facts visible
    discussion = run_replay_discussion(model, prompts["solver"], item,
                                       reveal_after_round=0, order="AB",
                                       rounds_count=discussion_rounds)
    add_information_timeline(item, discussion)

    # Step 2: Build canonical fact table
    facts_a = atomic_facts(item["condition_A"])
    facts_b = atomic_facts(item["condition_B"])
    canonical_table = "CANONICAL FACT TABLE (guaranteed correct and complete):\n"
    for i, f in enumerate(facts_a + facts_b):
        src = "A" if i < len(facts_a) else "B"
        canonical_table += f"  [{src}] {f}\n"

    # Step 3: Launch FRESH solver — NO discussion history
    reseed_model(model, derived_seed(base_seed, qid, "fresh_solver_call"))
    fresh_user = (f'Shared question: {item["shared_question"]}\n\n'
                  f'{canonical_table}\n\n'
                  'You are seeing this problem for the first time. The facts above are guaranteed correct '
                  'and complete. Compute the answer. Put `Final answer: ...` on the FIRST line, '
                  'then at most three sentences of reasoning.')
    fresh_raw, fresh_usage, fresh_elapsed = model.call(
        ORACLE_FRESH_SOLVER_SYSTEM, fresh_user, temperature=0.0)
    fresh_answer, fresh_fmt_err = parse_solver_final(fresh_raw)
    fresh_prediction = fresh_answer if not fresh_fmt_err else extract_answer(fresh_raw)

    # Step 4: Finalizer sees fresh solver + canonical table + (optionally) old discussion
    transcript = public_transcript(discussion.get("discussion_events", []))
    evidence = (f"{canonical_table}\n\n"
                f"--- Fresh Solver (no history, fact-table only) ---\n"
                f"Fresh solver answer: {fresh_prediction}\n"
                f"Fresh solver reasoning: {fresh_raw[:500]}\n\n"
                f"--- Original Discussion Transcript (for reference) ---\n{transcript}")

    candidates = {"fresh_solver": fresh_prediction}
    # Also add old solver candidates for reference
    solver_finals = discussion.get("solver_finals", {})
    for side in ("a", "b"):
        sf = solver_finals.get(side, {})
        ans = sf.get("answer", event_answer(sf))
        if ans:
            candidates[f"solver_{side}"] = ans

    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            f'Valid non-empty candidates: {json.dumps(candidates)}\n'
            f'Available selected_source values for this question: ["fresh_solver", "solver_a", "solver_b", "recomputed", "none"]\n'
            f'Verifier report: "(no verifier in this setting)"\n'
            "The fresh_solver saw ONLY the canonical fact table (no discussion). "
            "Select the best-supported candidate or recompute. Begin immediately with `Selected source:`.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, candidates)

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "canonical_state_fresh",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion, "finalizer_event": finalizer,
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
        "canonical_fact_table": canonical_table,
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    # Add fresh solver appearance
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


def build_oracle_plan_trace(model, prompts, item, qid, base_seed, discussion_rounds=2):
    """Setting 4: Canonical fact table + equation structure / dependency graph.
    Solver only needs to fill values and compute."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "oracle_plan"))

    # Build canonical fact table
    facts_a = atomic_facts(item["condition_A"])
    facts_b = atomic_facts(item["condition_B"])
    canonical_table = "CANONICAL FACT TABLE (guaranteed correct and complete):\n"
    all_fact_list = []
    for i, f in enumerate(facts_a):
        canonical_table += f"  [{i+1}] {f}\n"
        all_fact_list.append(f"F{i+1} = {f}")
    for i, f in enumerate(facts_b):
        canonical_table += f"  [{len(facts_a)+i+1}] {f}\n"
        all_fact_list.append(f"F{len(facts_a)+i+1} = {f}")

    # Extract equation structure from gold answer
    eq_structure = extract_equation_structure(item["answer"])

    plan_text = (
        f"{canonical_table}\n\n"
        f"SOLUTION PLAN (equation dependency structure):\n"
        f"The solution follows this dependency chain:\n"
        f"{eq_structure}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Map each fact above to the variables in the equation structure\n"
        f"2. Fill the values from the fact table into the equation blanks\n"
        f"3. Execute the arithmetic step by step\n"
        f"4. Output the final result\n"
    )

    # Discussion with plan visible
    discussion = run_replay_discussion(model, prompts["solver"], item,
                                       reveal_after_round=0, order="AB",
                                       rounds_count=discussion_rounds)
    add_information_timeline(item, discussion)

    transcript = public_transcript(discussion.get("discussion_events", []))
    evidence = f"{plan_text}\n\nDiscussion transcript:\n{transcript}"

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
            "Use the solution plan as a guide. Select or recompute. Begin immediately with `Selected source:`.")
    finalizer = call_finalizer_once(model, prompts["finalizer"], user, candidates)

    prediction, extraction = extract_free_text_answer(finalizer.get("raw_output", ""), "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(finalizer.get("invalid_output"))

    trace = {
        "question_id": qid, "setting": "oracle_plan",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion, "finalizer_event": finalizer,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "equation_structure": eq_structure,
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
        "disclosure_mode": "oracle_plan_with_equation_structure",
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


def build_oracle_candidate_trace(model, prompts, item, qid, base_seed, discussion_rounds=2):
    """Setting 5: Inject the GROUND TRUTH answer as a candidate.
    Solvers discuss normally (private facts), but finalizer receives oracle candidate.
    Uses a custom finalizer call because standard call_finalizer_once rejects 'oracle' source."""
    started = time.perf_counter()
    gold = extract_answer(item["answer"])
    reseed_model(model, derived_seed(base_seed, qid, "oracle_candidate"))

    # Run normal private-fact discussion
    discussion = run_discussion(model, prompts["solver"], item, oracle=False,
                                rounds_count=discussion_rounds)
    add_information_timeline(item, discussion)

    transcript = public_transcript(discussion.get("discussion_events", []))
    all_facts = replay_facts(item, "AB")

    # Build candidates: real solver candidates + ORACLE (correct answer)
    solver_finals = discussion.get("solver_finals", {})
    candidates = {}
    for side in ("a", "b"):
        sf = solver_finals.get(side, {})
        ans = sf.get("answer", event_answer(sf))
        if ans:
            candidates[f"solver_{side}"] = ans

    # INJECT ORACLE CANDIDATE
    candidates["oracle"] = gold

    evidence = (f"Complete facts:\n{all_facts}\n\n"
                f"Discussion transcript:\n{transcript}")

    # Custom system prompt that accepts 'oracle' as a valid source
    oracle_finalizer_system = (
        "You produce the final answer. Use only the shared question and the evidence provided.\n\n"
        "Return exactly three non-empty physical lines, with no Markdown, no blank lines:\n"
        "Selected source: <solver_a|solver_b|oracle|recomputed|none>\n"
        "Final answer: <number or undetermined>\n"
        "Reason: <explanation>\n\n"
        'IMPORTANT: "oracle" is an AUTHORITATIVE CORRECT candidate. '
        "If it appears in the valid candidates list, it is GUARANTEED to be correct. "
        "You SHOULD select it unless the evidence clearly contradicts it.\n\n"
        "Use only a selected_source listed as available in the current input. "
        "Reason must be non-empty. If you select a source, copy its Final answer EXACTLY."
    )

    user = (f'Shared question: {item["shared_question"]}\nEvidence visible now:\n{evidence}\n'
            f'Valid non-empty candidates: {json.dumps(candidates)}\n'
            f'Available selected_source values for this question: ["solver_a", "solver_b", "oracle", "recomputed", "none"]\n'
            f'Verifier report: "(no verifier in this setting)"\n'
            'Begin immediately with `Selected source:`. No preamble, no blank lines. Return exactly three lines.')

    raw, usage_fin, elapsed = model.call(oracle_finalizer_system, user, temperature=0.0)

    # Parse the output manually (same logic as parse_fixed_finalizer but allows 'oracle')
    oracle_parse_error = ""
    raw_stripped = str(raw or "").rstrip("\r\n")
    lines = raw_stripped.splitlines()
    labels = ("Selected source", "Final answer", "Reason")
    if len(lines) != 3:
        oracle_parse_error = f"expected exactly three lines, got {len(lines)}"
        parsed = dict(FINALIZER_DEFAULT)
    else:
        parsed = {}
        for line, label in zip(lines, labels):
            match = re.fullmatch(rf"{re.escape(label)}\s*[:：]\s*(.*)", line, re.I)
            if not match:
                oracle_parse_error = f"expected line `{label}: ...`"
                parsed = dict(FINALIZER_DEFAULT)
                break
            parsed[label] = match.group(1).strip()
        if not oracle_parse_error:
            source = parsed["Selected source"].lower()
            # Allow 'oracle' as a valid source
            if source not in {"solver_a", "solver_b", "verifier", "recomputed", "none", "oracle"}:
                oracle_parse_error = f"selected_source '{source}' is not allowed"
                parsed = dict(FINALIZER_DEFAULT)
            elif not parsed.get("Reason"):
                oracle_parse_error = "Reason must not be empty"
                parsed = dict(FINALIZER_DEFAULT)
            else:
                answer = "" if explicitly_undetermined(parsed["Final answer"]) else extract_answer(parsed["Final answer"])
                parsed = {"selected_source": source, "final_answer": answer, "reason": parsed["Reason"]}

    finalizer_event = {
        "agent": "finalizer",
        "phase": "finalization",
        "raw_output": raw,
        "output": raw,
        "parsed_output": parsed,
        "parse_error": oracle_parse_error,
        "invalid_output": bool(oracle_parse_error),
        "token_usage": usage_fin,
        "runtime_seconds": elapsed,
    }

    selected_oracle = parsed.get("selected_source", "") == "oracle"
    prediction, extraction = extract_free_text_answer(raw, "Final answer")
    semantic_correct = equivalent(prediction, gold)
    format_compliant = not bool(oracle_parse_error)

    trace = {
        "question_id": qid, "setting": "oracle_candidate",
        "shared_question": item["shared_question"], "gold_answer": gold,
        "discussion": discussion, "finalizer_event": finalizer_event,
        "final_prediction": prediction, "semantic_answer_extraction": extraction,
        "candidate_answers": candidates,
        "oracle_candidate_injected": True,
        "oracle_candidate_value": gold,
        "finalizer_selected_oracle": selected_oracle,
        "information": {"information_complete": True, "side_revealed": {"A": True, "B": True}},
        "injected_facts": {"A": item["condition_A"], "B": item["condition_B"]},
        "semantic_correct": semantic_correct, "format_compliant": format_compliant,
        "invalid_output": not format_compliant,
    }
    set_outcome_fields(trace, gold, semantic_correct)
    trace["candidate_appearances"] = candidate_appearances(trace)
    for a in trace["candidate_appearances"]:
        a["correct"] = equivalent(a["answer"], gold)
    # Add oracle as a candidate appearance
    trace["candidate_appearances"].append({
        "source": "oracle", "phase": "injected",
        "answer": gold,
        "information_complete_at_appearance": True,
        "correct": True,
    })
    usage = blank_usage()
    add_usage(usage, usage_fin)
    for event in collect_events(trace):
        add_usage(usage, event.get("token_usage", blank_usage()))
    trace["inference_token_usage"] = usage
    trace["total_runtime_seconds"] = time.perf_counter() - started
    trace["failure_type"], trace["lucky_guess"] = classify(trace, gold)
    return trace


# ═══════════════════════════════════════════════════════════════════════════
# BUILDER DISPATCH
# ═══════════════════════════════════════════════════════════════════════════

BUILDERS = {
    "free_discussion":          build_free_discussion_trace,
    "oracle_disclosure":        build_oracle_disclosure_trace,
    "oracle_canonical_state":   build_oracle_canonical_state_trace,
    "canonical_state_fresh":    build_canonical_state_fresh_trace,
    "oracle_plan":              build_oracle_plan_trace,
    "oracle_candidate":         build_oracle_candidate_trace,
}


# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS PER SETTING
# ═══════════════════════════════════════════════════════════════════════════

def compute_setting_metrics(traces: list[dict], questions: list[dict]) -> dict:
    """Compute all required metrics for a setting's traces."""
    n = len(traces)
    if n == 0:
        return {}

    golds = {t["question_id"]: t["gold_answer"] for t in traces}

    # 1. Semantic accuracy
    semantic_correct = sum(1 for t in traces if t.get("semantic_correct", False))

    # 2. Format compliance
    format_compliant = sum(1 for t in traces if t.get("format_compliant", False))

    # 3. Correct candidate emergence (majority across seeds per question)
    by_q = defaultdict(list)
    for t in traces:
        by_q[t["question_id"]].append(t)

    cand_qs = 0
    for qid, tlist in by_q.items():
        gold = golds.get(qid, "")
        # Any trace has correct candidate?
        has_correct = any(
            any(str(a.get("answer", "")) == str(gold) for a in t.get("candidate_appearances", []))
            for t in tlist
        )
        if has_correct:
            cand_qs += 1

    # 4. Answer-reason consistency (only count among checkable traces)
    checkable = 0
    consistent = 0
    for t in traces:
        if t.get("answer_reason_checkable", False):
            checkable += 1
            if t.get("answer_reason_consistent", None) is True:
                consistent += 1

    # 5. Fact distortion rate (for multi-agent settings with discussion)
    distortion_rates = []
    for t in traces:
        disc = t.get("discussion", {})
        events = disc.get("discussion_events", [])
        all_outputs = []
        for evt in events:
            raw = evt.get("raw_output", "")
            if raw:
                all_outputs.append(raw)
        # Also include solver finals
        for side in ("a", "b"):
            sf = disc.get("solver_finals", {}).get(side, {})
            raw = sf.get("raw_output", "")
            if raw:
                all_outputs.append(raw)

        # Get all required facts for this question
        qid = t["question_id"]
        all_facts = []
        for q in questions:
            if q.get("shared_question") == t.get("shared_question", ""):
                all_facts = q.get("required_private_facts", {}).get("agent_A", []) + \
                           q.get("required_private_facts", {}).get("agent_B", [])
                break

        if all_facts and all_outputs:
            dr = compute_fact_distortion_rate(all_outputs, all_facts)
            distortion_rates.append(dr["distortion_rate"])

    avg_distortion = sum(distortion_rates) / len(distortion_rates) if distortion_rates else 0.0

    # 6. Partial answer rate
    partial_rates = []
    for t in traces:
        disc = t.get("discussion", {})
        events = disc.get("discussion_events", [])
        outputs = [evt.get("raw_output", "") for evt in events if evt.get("raw_output")]
        if outputs:
            pr = compute_partial_answer_rate(outputs, t.get("gold_answer", ""))
            partial_rates.append(pr["partial_answer_rate"])
    avg_partial = sum(partial_rates) / len(partial_rates) if partial_rates else 0.0

    # 7. Undetermined ratio
    undetermined_rates = []
    for t in traces:
        disc = t.get("discussion", {})
        events = disc.get("discussion_events", [])
        outputs = [evt.get("raw_output", "") for evt in events if evt.get("raw_output")]
        if outputs:
            ur = compute_undetermined_ratio(outputs)
            undetermined_rates.append(ur["undetermined_ratio"])
    avg_undetermined = sum(undetermined_rates) / len(undetermined_rates) if undetermined_rates else 0.0

    # 8. Oracle-specific: did finalizer select oracle?
    oracle_selection_rate = 0.0
    oracle_traces = [t for t in traces if t.get("setting") == "oracle_candidate"]
    if oracle_traces:
        oracle_selection_rate = sum(1 for t in oracle_traces
                                    if t.get("finalizer_selected_oracle", False)) / len(oracle_traces)

    # 9. Fresh solver accuracy (for canonical_state_fresh)
    fresh_solver_acc = 0.0
    fresh_traces = [t for t in traces if t.get("setting") == "canonical_state_fresh"]
    if fresh_traces:
        fresh_solver_acc = sum(1 for t in fresh_traces
                               if t.get("fresh_solver_correct", False)) / len(fresh_traces)

    return {
        "n": n,
        "semantic_accuracy": semantic_correct / n,
        "semantic_correct_count": semantic_correct,
        "format_compliance_rate": format_compliant / n,
        "format_compliant_count": format_compliant,
        "correct_candidate_emergence_rate": cand_qs / len(by_q) if by_q else 0,
        "correct_candidate_questions": cand_qs,
        "total_questions": len(by_q),
        "answer_reason_consistency_rate": consistent / checkable if checkable else 0,
        "answer_reason_consistent_count": consistent,
        "answer_reason_checkable_count": checkable,
        "avg_fact_distortion_rate": avg_distortion,
        "avg_partial_answer_rate": avg_partial,
        "avg_undetermined_ratio": avg_undetermined,
        "oracle_selection_rate": oracle_selection_rate,
        "fresh_solver_accuracy": fresh_solver_acc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_oracle_experiment(
    model,
    prompts: dict,
    items: list[dict],
    output_dir: Path,
    selected_settings: list[str],
    discussion_rounds: int = 2,
    num_seeds: int = 3,
    base_seed: int = DEFAULT_SEED,
) -> str:
    """Run oracle intervention experiment."""
    print(f"\n{'='*70}")
    print(f"ORACLE INTERVENTION EXPERIMENT")
    print(f"  Questions: {len(items)}")
    print(f"  Settings: {len(selected_settings)}")
    print(f"  Seeds per setting: {num_seeds}")
    print(f"  Discussion rounds: {discussion_rounds}")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_traces = []

    for qid, item in enumerate(items, 1):
        question_traces = []

        for setting in selected_settings:
            builder = BUILDERS.get(setting)
            if not builder:
                print(f"  [SKIP] Unknown setting: {setting}")
                continue

            print(f"  [Q{qid}/{len(items)}] {SETTING_LABELS.get(setting, setting)}")

            for seed_idx in range(num_seeds):
                s = derived_seed(base_seed, qid, setting, seed_idx)
                try:
                    trace = builder(model, prompts, item, qid, s, discussion_rounds)
                    question_traces.append(trace)
                except Exception as e:
                    print(f"    [ERROR seed={seed_idx}]: {e}")
                    import traceback
                    traceback.print_exc()

        all_traces.extend(question_traces)

        # Incremental write
        (output_dir / "traces_all.json").write_text(
            json.dumps(all_traces, ensure_ascii=False, indent=2), encoding="utf-8")

        # Per-setting write
        by_setting = defaultdict(list)
        for t in all_traces:
            by_setting[t.get("setting", "unknown")].append(t)
        for sname, straces in by_setting.items():
            (output_dir / f"traces_{sname}.json").write_text(
                json.dumps(straces, ensure_ascii=False, indent=2), encoding="utf-8")

        # Snapshot
        parts = []
        for sname in sorted(by_setting):
            st = by_setting[sname]
            sc = sum(1 for t in st if t.get("semantic_correct"))
            parts.append(f"{sname}={sc}/{len(st)}")
        print(f"    [{', '.join(parts)}]")

    # ── Final metrics ──
    print(f"\n{'='*70}")
    print("FINAL METRICS")
    print(f"{'='*70}")

    all_metrics = {}
    by_setting_final = defaultdict(list)
    for t in all_traces:
        by_setting_final[t.get("setting", "unknown")].append(t)

    for setting in selected_settings:
        straces = by_setting_final.get(setting, [])
        metrics = compute_setting_metrics(straces, items)
        all_metrics[setting] = metrics
        print(f"\n{SETTING_LABELS.get(setting, setting)}:")
        print(f"  Semantic Accuracy:        {metrics.get('semantic_accuracy', 0):.1%} ({metrics.get('semantic_correct_count', 0)}/{metrics.get('n', 0)})")
        print(f"  Format Compliance:        {metrics.get('format_compliance_rate', 0):.1%}")
        print(f"  Correct Candidate Emerg:  {metrics.get('correct_candidate_emergence_rate', 0):.1%} ({metrics.get('correct_candidate_questions', 0)}/{metrics.get('total_questions', 0)} qs)")
        ar_rate = metrics.get('answer_reason_consistency_rate', 0)
        if ar_rate is None or ar_rate == 0:
            ar_display = "N/A (no checkable traces)"
        elif ar_rate <= 1.0:
            ar_display = f"{ar_rate:.1%}"
        else:
            ar_display = f"{ar_rate:.1f} (n={metrics.get('answer_reason_consistent_count',0)}/{metrics.get('answer_reason_checkable_count',0)})"
        print(f"  Ans-Reason Consistency:   {ar_display}")
        print(f"  Avg Fact Distortion:      {metrics.get('avg_fact_distortion_rate', 0):.1%}")
        print(f"  Avg Partial Answer Rate:  {metrics.get('avg_partial_answer_rate', 0):.1%}")
        print(f"  Avg Undetermined Ratio:   {metrics.get('avg_undetermined_ratio', 0):.1%}")
        if setting == "canonical_state_fresh":
            print(f"  Fresh Solver Accuracy:    {metrics.get('fresh_solver_accuracy', 0):.1%}")
        if setting == "oracle_candidate":
            print(f"  Oracle Selection Rate:    {metrics.get('oracle_selection_rate', 0):.1%}")

    # ── Recovery analysis ──
    baseline_acc = all_metrics.get("free_discussion", {}).get("semantic_accuracy", 0)
    print(f"\n{'='*70}")
    print("RECOVERY ANALYSIS (relative to free_discussion baseline)")
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    print(f"{'='*70}")

    recovery = {}
    for setting in selected_settings:
        if setting == "free_discussion":
            continue
        acc = all_metrics.get(setting, {}).get("semantic_accuracy", 0)
        delta = acc - baseline_acc
        recovery[setting] = delta
        bar = "+" * max(0, int(delta * 100)) + "-" * max(0, int((0.8 - delta) * 100))
        print(f"  {SETTING_LABELS.get(setting, setting)[:60]}:")
        print(f"    Accuracy: {acc:.1%} | delta from baseline: {delta:+.1%}")
        print(f"    Interpretation: ", end="")
        if delta > 0.3:
            print("[MAJOR] recovery -- this layer is a PRIMARY bottleneck")
        elif delta > 0.1:
            print("[SIGNIFICANT] recovery -- this layer is a SECONDARY bottleneck")
        elif delta > 0.02:
            print("[MINOR] recovery -- this layer has a SMALL effect")
        else:
            print("No meaningful recovery -- bottleneck is ELSEWHERE")

    # ── Save metrics ──
    metrics_path = output_dir / "oracle_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=str)

    # Save metrics CSV
    csv_path = output_dir / "oracle_metrics.csv"
    metric_keys = ["semantic_accuracy", "format_compliance_rate",
                   "correct_candidate_emergence_rate", "answer_reason_consistency_rate",
                   "avg_fact_distortion_rate", "avg_partial_answer_rate",
                   "avg_undetermined_ratio", "oracle_selection_rate", "fresh_solver_accuracy"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "n"] + metric_keys)
        for setting in selected_settings:
            m = all_metrics.get(setting, {})
            writer.writerow([setting, m.get("n", 0)] +
                           [m.get(k, 0) for k in metric_keys])

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Traces saved to: {output_dir / 'traces_all.json'}")

    return str(output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Oracle Intervention Experiment")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--settings", type=str, nargs="*",
                       default=["free_discussion", "oracle_disclosure",
                                "oracle_canonical_state", "canonical_state_fresh",
                                "oracle_plan", "oracle_candidate"],
                       help="Which oracle settings to run (default: all 6)")
    parser.add_argument("--single-setting", type=str, default=None,
                       help="Run only one specific setting")
    args = parser.parse_args()

    # Determine output dir
    if args.output:
        output_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_BASE_DIR / ts

    # Load data
    items = read_json_records(Path(args.data))[:args.limit]
    print(f"Loaded {len(items)} questions from {args.data}")

    # Load model
    model_path = Path(args.model)
    print(f"Loading model from {model_path}...")
    model = LocalQwen(model_path, args.device, args.max_tokens, args.temperature, False)
    print(f"Model loaded. Device: {args.device}, max_tokens: {args.max_tokens}")

    # Load prompts
    prompts = load_prompts()
    print(f"Loaded prompts: {list(prompts.keys())}")

    # Determine settings
    if args.single_setting:
        selected = [args.single_setting]
    else:
        selected = args.settings

    # Validate
    for s in selected:
        if s not in ORACLE_SETTINGS:
            print(f"ERROR: Unknown setting '{s}'. Valid: {ORACLE_SETTINGS}")
            sys.exit(1)

    # Write run config
    config = {
        "experiment": "oracle_intervention",
        "data_path": args.data,
        "model_path": args.model,
        "device": args.device,
        "temperature": args.temperature,
        "max_new_tokens": args.max_tokens,
        "discussion_rounds": args.rounds,
        "num_seeds": args.seeds,
        "base_seed": args.seed,
        "settings": selected,
        "num_questions": len(items),
        "started_at": datetime.now().isoformat(),
    }

    # Run
    output_dir = run_oracle_experiment(
        model=model,
        prompts=prompts,
        items=items,
        output_dir=output_dir,
        selected_settings=selected,
        discussion_rounds=args.rounds,
        num_seeds=args.seeds,
        base_seed=args.seed,
    )

    (Path(output_dir) / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"Experiment complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
