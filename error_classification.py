#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 5: Unified Error Classification for ExecGround Experiment.

Classifies every error trace into at least the following categories:
  1. 私有事实没有披露         (Private facts not disclosed)
  2. 事实披露不完整           (Incomplete fact disclosure)
  3. 忘记自己的事实           (Forgets own facts)
  4. 忽略对方事实             (Ignores other's facts)
  5. 事实数值被修改           (Fact values modified)
  6. half/twice 等关系翻转    (Relation inversion)
  7. 实体或题意漂移           (Entity/task drift)
  8. 只完成局部计算           (Partial computation only)
  9. 依赖图缺失               (Missing dependency graph)
  10. 算术执行错误            (Arithmetic execution error)
  11. reasoning中正确但候选错误 (Correct reasoning, wrong candidate field)
  12. 正确候选存在但被丢失     (Correct candidate existed but lost)
  13. 信息足够但仍输出undetermined (Sufficient info, still undetermined)

Output: error_classification_report.md
"""

import json
import re
import math
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# ── Configuration ──
TRACES_PATH = Path("outputs_exec_ground/20260807_152225/traces_all.jsonl")
OUTPUT_DIR = Path("outputs_exec_ground/20260807_152225")
REPORT_PATH = OUTPUT_DIR / "error_classification_report.md"

SETTING_LABELS = {
    "free_discussion": "Setting 0: Free Discussion",
    "oracle_disclosure": "Setting 1: Oracle Disclosure",
    "canonical_ledger": "Setting 2: Canonical Ledger",
    "ledger_fresh_solver": "Setting 3: Ledger + Fresh Solver",
    "ledger_exec_plan": "Setting 4: Ledger + Executable Plan",
    "ledger_exec_plan_verify": "Setting 5: Ledger + Plan + Verify",
}

# ── Utility ──

def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text (including decimals)."""
    if not text:
        return []
    pattern = r'(?<![\w/])(\d+(?:\.\d+)?)(?![\w/])'
    return [float(m) for m in re.findall(pattern, text)]


def extract_numbers_with_context(text: str, window: int = 30) -> list[tuple[float, str]]:
    """Extract numbers with surrounding context."""
    if not text:
        return []
    results = []
    for m in re.finditer(r'(?<![\w/])(\d+(?:\.\d+)?)(?![\w/])', text):
        num = float(m.group(0))
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        ctx = text[start:end].lower()
        results.append((num, ctx))
    return results


def key_numbers_from_condition(condition: str) -> set[float]:
    """Extract key numbers from a condition text (excluding trivial ones like 1.0, 2.0)."""
    nums = extract_numbers(condition)
    # Filter out trivial ratios that appear standalone
    key = set()
    for n in nums:
        if n == int(n) and n in (1.0, 0.0):
            continue
        key.add(n)
    return key


def numbers_appearing_in_text(text: str, numbers: set[float]) -> set[float]:
    """Which of the given numbers appear in the text?"""
    if not text:
        return set()
    text_nums = set(extract_numbers(text))
    found = set()
    for n in numbers:
        # Exact match
        if n in text_nums:
            found.add(n)
        # Integer match (12.0 matches 12)
        elif n == int(n) and int(n) in text_nums:
            found.add(n)
        elif int(n) in text_nums and float(int(n)) == n:
            found.add(n)
    return found


def relation_keywords() -> dict[str, str]:
    """Map relation keywords to their canonical forms."""
    return {
        "half": "half_of",
        "twice": "twice_of",
        "double": "twice_of",
        "doubled": "twice_of",
        "triple": "triple_of",
        "tripled": "triple_of",
        "half of": "half_of",
        "twice as": "twice_of",
        "double of": "twice_of",
        "three times": "triple_of",
    }


def check_relation_direction(text: str, relation: str) -> str | None:
    """
    Check if a relation appears to be flipped in text.
    Returns 'flipped', 'correct', or None if can't determine.
    """
    text_lower = text.lower()
    if relation == "half_of":
        # "half of X" should mean X/2
        # If text says "X * 2" or "X times 2", it's flipped
        if re.search(r'(?:twice|double|multiply by 2|×\s*2|\*\s*2)', text_lower):
            return "flipped"
    elif relation in ("twice_of", "triple_of"):
        # "twice X" should mean X*2
        # If text says "X/2" or "half", it's flipped
        if re.search(r'(?:half|divide by 2|÷\s*2|/\s*2)', text_lower):
            return "flipped"
    return None


# ── Core Classification ──

class ErrorClassifier:
    """Classify errors in a single trace."""

    def __init__(self, trace: dict):
        self.t = trace
        self.setting = trace.get("setting", "")
        self.qid = trace.get("question_id", 0)
        self.seed = trace.get("seed", 0)
        self.gold = trace.get("gold_answer", "?")
        self.pred = trace.get("final_prediction", "")
        self.semantic_correct = trace.get("semantic_correct", False)
        self.errors: list[str] = []
        self.error_details: dict = {}
        self.discussion_rounds: list[dict] = []  # store extracted per-round info

        # Parse conditions
        injected = trace.get("injected_facts", {})
        self.cond_a = injected.get("A", "")
        self.cond_b = injected.get("B", "")
        self.nums_a = key_numbers_from_condition(self.cond_a)
        self.nums_b = key_numbers_from_condition(self.cond_b)
        self.all_nums = self.nums_a | self.nums_b

        # Discussion
        self.disc = trace.get("discussion", {})
        self.round_records = self.disc.get("round_records", [])
        self.solver_finals = self.disc.get("solver_finals", {})
        self.public_transcript = self.disc.get("public_transcript", "")

        # Candidates
        self.candidate_appearances = trace.get("candidate_appearances", [])

        # ExecGround
        self.eg = trace.get("exec_ground", {})

        # Failure type from existing classification
        self.failure_type = trace.get("failure_type", "")

    def classify_all(self) -> list[str]:
        """Run all classification checks. Returns list of error categories found."""
        if self.semantic_correct:
            return ["correct"]

        methods = [
            self._check_undetermined_despite_info,
            self._check_private_not_disclosed,
            self._check_incomplete_disclosure,
            self._check_forgets_own_facts,
            self._check_ignores_other_facts,
            self._check_fact_values_modified,
            self._check_relation_inversion,
            self._check_entity_drift,
            self._check_partial_computation,
            self._check_dependency_missing,
            self._check_arithmetic_error,
            self._check_correct_reasoning_wrong_field,
            self._check_correct_candidate_lost,
            self._check_fresh_solver_error,       # for settings 3-5 fresh solver output
            self._check_llm_extraction_error,     # LLM fact extraction quality
        ]

        for method in methods:
            result = method()
            if result:
                self.errors.append(result)

        if not self.errors:
            self.errors.append("unclassified_error")

        return self.errors

    def _check_undetermined_despite_info(self) -> str | None:
        """13. 信息足够但仍输出 undetermined"""
        # Check if key numbers from BOTH conditions appear in transcript
        a_disclosed = numbers_appearing_in_text(self.public_transcript, self.nums_a)
        b_disclosed = numbers_appearing_in_text(self.public_transcript, self.nums_b)

        all_disclosed = len(a_disclosed | b_disclosed) >= 0.6 * len(self.all_nums)

        # Check solver_finals for undetermined
        undetermined_phrases = [
            "undetermined", "cannot be determined", "cannot determine",
            "can't be determined", "can't determine", "impossible to determine",
            "not enough information", "insufficient information",
            "more information is needed",
        ]
        for side in ("a", "b"):
            sf = self.solver_finals.get(side, {})
            output = sf.get("output", "")
            answer = sf.get("answer", "")
            is_undetermined = not answer
            if not is_undetermined:
                # Also check if answer field is empty and output says undetermined
                pass
            has_undetermined_phrase = any(p in output.lower() for p in undetermined_phrases)
            if (is_undetermined or has_undetermined_phrase) and not answer:
                if all_disclosed or self.setting in ("oracle_disclosure", "canonical_ledger"):
                    self.error_details["undetermined_despite_info"] = {
                        "side": side,
                        "nums_disclosed": len(a_disclosed | b_disclosed),
                        "nums_total": len(self.all_nums),
                        "output_snippet": output[:200],
                    }
                    return "信息足够但仍输出undetermined"
        return None

    def _check_private_not_disclosed(self) -> str | None:
        """1. 私有事实没有披露"""
        if self.setting not in ("free_discussion",):
            return None  # only for private info setting

        # Check if agent A disclosed their numbers in any round
        a_all_output = ""
        b_all_output = ""
        for rr in self.round_records:
            st = rr.get("simultaneous_turn", {})
            a_all_output += st.get("a", {}).get("output", "") + " "
            b_all_output += st.get("b", {}).get("output", "") + " "

        a_disclosed = len(numbers_appearing_in_text(a_all_output, self.nums_a))
        b_disclosed = len(numbers_appearing_in_text(b_all_output, self.nums_b))
        a_total = len(self.nums_a)
        b_total = len(self.nums_b)

        non_disclosed = []
        if a_total > 0 and a_disclosed == 0:
            non_disclosed.append("A")
        if b_total > 0 and b_disclosed == 0:
            non_disclosed.append("B")

        if non_disclosed:
            self.error_details["private_not_disclosed"] = {
                "agents": non_disclosed,
                "a_disclosed": f"{a_disclosed}/{a_total}",
                "b_disclosed": f"{b_disclosed}/{b_total}",
            }
            return "私有事实没有披露"
        return None

    def _check_incomplete_disclosure(self) -> str | None:
        """2. 事实披露不完整"""
        if self.setting not in ("free_discussion",):
            return None

        a_all_output = ""
        b_all_output = ""
        for rr in self.round_records:
            st = rr.get("simultaneous_turn", {})
            a_all_output += st.get("a", {}).get("output", "") + " "
            b_all_output += st.get("b", {}).get("output", "") + " "

        a_disclosed = len(numbers_appearing_in_text(a_all_output, self.nums_a))
        b_disclosed = len(numbers_appearing_in_text(b_all_output, self.nums_b))
        a_total = max(len(self.nums_a), 1)
        b_total = max(len(self.nums_b), 1)

        if (a_disclosed > 0 and a_disclosed < a_total) or (b_disclosed > 0 and b_disclosed < b_total):
            # Only flag if some but not all numbers disclosed
            if a_disclosed / a_total < 1.0 or b_disclosed / b_total < 1.0:
                self.error_details["incomplete_disclosure"] = {
                    "a_coverage": f"{a_disclosed}/{a_total}",
                    "b_coverage": f"{b_disclosed}/{b_total}",
                    "nums_a": sorted(list(self.nums_a)),
                    "nums_b": sorted(list(self.nums_b)),
                }
                return "事实披露不完整"
        return None

    def _check_forgets_own_facts(self) -> str | None:
        """3. 忘记自己的事实"""
        for side, cond_nums in (("a", self.nums_a), ("b", self.nums_b)):
            sf = self.solver_finals.get(side, {})
            output = sf.get("output", "")
            if not output:
                continue
            found = numbers_appearing_in_text(output, cond_nums)
            if len(found) == 0 and len(cond_nums) > 0:
                self.error_details["forgets_own_facts"] = {
                    "side": side,
                    "own_numbers": sorted(list(cond_nums)),
                    "solver_output_snippet": output[:200],
                }
                return "忘记自己的事实"
        return None

    def _check_ignores_other_facts(self) -> str | None:
        """4. 忽略对方事实"""
        for side, other_nums in (("a", self.nums_b), ("b", self.nums_a)):
            sf = self.solver_finals.get(side, {})
            output = sf.get("output", "")
            if not output:
                continue
            found = numbers_appearing_in_text(output, other_nums)
            if len(found) == 0 and len(other_nums) > 0:
                # Only flag if the other numbers were disclosed in transcript
                other_disclosed = numbers_appearing_in_text(self.public_transcript, other_nums)
                if len(other_disclosed) > 0:
                    self.error_details["ignores_other_facts"] = {
                        "side": side,
                        "other_numbers_available": sorted(list(other_disclosed)),
                        "solver_output_snippet": output[:200],
                    }
                    return "忽略对方事实"
        return None

    def _check_fact_values_modified(self) -> str | None:
        """5. 事实数值被修改"""
        # For exec_ground: check if LLM-extracted facts differ from gold facts
        if self.setting in ("canonical_ledger", "ledger_fresh_solver",
                            "ledger_exec_plan", "ledger_exec_plan_verify"):
            typed_a = self.eg.get("typed_facts_A", [])
            typed_b = self.eg.get("typed_facts_B", [])

            # Check for obviously wrong values
            for fact_list, source in ((typed_a, "A"), (typed_b, "B")):
                for f in fact_list:
                    evidence = f.get("evidence", "")
                    orig_nums = extract_numbers(evidence)
                    fact_val = f.get("value")
                    if fact_val is not None and len(orig_nums) > 0:
                        # Check if the fact value appears in the evidence
                        if fact_val not in orig_nums:
                            # Could be a derived value (e.g., half of something)
                            # Check for derived relations
                            relation = f.get("relation", "")
                            if relation in ("half_of",):
                                # half_of should be ratio 0.5, not a dollar amount
                                if f.get("unit") not in ("ratio", ""):
                                    self.error_details["fact_values_modified"] = {
                                        "fact": f,
                                        "issue": f"half_of relation stored with unit={f.get('unit')} value={fact_val}",
                                    }
                                    return "事实数值被修改"
                            elif relation in ("equals",) and fact_val in orig_nums:
                                continue  # OK, value matches evidence
                            # Could also be intermediate computation

            # For free_discussion: check if numbers in discussion changed from originals
            if self.setting == "free_discussion":
                for rr in self.round_records:
                    st = rr.get("simultaneous_turn", {})
                    for side, cond_nums in (("a", self.nums_a), ("b", self.nums_b)):
                        output = st.get(side, {}).get("output", "")
                        out_nums = set(extract_numbers(output))
                        # Check for near-miss numbers (off by factor of 2, off by 1, etc.)
                        for orig in cond_nums:
                            for out_n in out_nums:
                                if out_n != orig and abs(out_n - orig) < 0.01:
                                    continue  # exact match
                                if out_n == orig * 2 or out_n == orig / 2:
                                    self.error_details["fact_values_modified"] = {
                                        "side": side,
                                        "original": orig,
                                        "modified_to": out_n,
                                        "relation": "factor_of_2",
                                    }
                                    return "事实数值被修改"

        return None

    def _check_relation_inversion(self) -> str | None:
        """6. half/twice 等关系翻转"""
        # Check solver_finals for relation inversion
        for side in ("a", "b"):
            sf = self.solver_finals.get(side, {})
            output = sf.get("output", "")
            if not output:
                continue
            output_lower = output.lower()

            # Check for "half" misuse
            if "half" in output_lower:
                # If output says "half of X" but should be "twice X" (or vice versa)
                nums_in_output = extract_numbers(output)
                # Compare with expected answer
                if self.gold:
                    gold_val = None
                    try:
                        gold_val = float(self.gold)
                    except (ValueError, TypeError):
                        pass
                    if gold_val and nums_in_output:
                        pred_nums = [n for n in nums_in_output if abs(n - gold_val) < 0.01]
                        if not pred_nums:
                            # Check if output with relation flipped would give correct answer
                            for n in nums_in_output:
                                if abs(n * 2 - gold_val) < 0.01 or abs(n / 2 - gold_val) < 0.01:
                                    self.error_details["relation_inversion"] = {
                                        "side": side,
                                        "computed": n,
                                        "gold": gold_val,
                                        "would_be_correct_if": "multiplied_by_2" if abs(n*2 - gold_val) < 0.01 else "divided_by_2",
                                        "snippet": output[:200],
                                    }
                                    return "half/twice等关系翻转"

            # Check for specific pattern: "twice as many" used where "half" should be
            if "twice" in output_lower:
                gold_val = None
                try:
                    gold_val = float(self.gold) if self.gold else None
                except (ValueError, TypeError):
                    pass
                if gold_val:
                    nums_in_output = extract_numbers(output)
                    for n in nums_in_output:
                        if abs(n / 2 - gold_val) < 0.01:
                            self.error_details["relation_inversion"] = {
                                "side": side,
                                "computed": n,
                                "gold": gold_val,
                                "would_be_correct_if": "divided_by_2",
                                "snippet": output[:200],
                            }
                            return "half/twice等关系翻转"

        return None

    def _check_entity_drift(self) -> str | None:
        """7. 实体或题意漂移"""
        # Check if the solver is solving for a different entity
        question = self.t.get("shared_question", "").lower()
        # Extract key entity from question (e.g., "how many pages", "how much money")
        question_entity_match = re.search(r'how (?:many|much) ([\w\s]+?)(?:\?|$|should|does|did|will)', question)
        question_entity = question_entity_match.group(1).strip() if question_entity_match else ""

        for side in ("a", "b"):
            sf = self.solver_finals.get(side, {})
            output = sf.get("output", "").lower()
            if not output:
                continue
            # Check if answer describes a different entity
            answer_matches = re.findall(r'final answer:?\s*(.+)', output)
            for ans in answer_matches:
                ans_clean = ans.strip().lower()
                if question_entity and question_entity not in ans_clean:
                    # Entity might have drifted
                    # Check common drift patterns
                    if any(w in ans_clean for w in ("today", "yesterday")) and "tomorrow" in question_entity:
                        self.error_details["entity_drift"] = {
                            "side": side,
                            "question_entity": question_entity,
                            "answer_entity": ans_clean[:100],
                            "drift_type": "time_reference_shift",
                        }
                        return "实体或题意漂移"
                    if "remaining" in question and "remaining" not in ans_clean and "total" in ans_clean:
                        self.error_details["entity_drift"] = {
                            "side": side,
                            "question_entity": question_entity,
                            "answer_entity": ans_clean[:100],
                            "drift_type": "total_vs_remaining",
                        }
                        return "实体或题意漂移"
        return None

    def _check_partial_computation(self) -> str | None:
        """8. 只完成局部计算"""
        for side in ("a", "b"):
            sf = self.solver_finals.get(side, {})
            output = sf.get("output", "")
            answer = sf.get("answer", "")
            if not output:
                continue

            # Check: output contains computation steps but answer is undetermined/empty
            has_computation = bool(re.search(
                r'(?:calculate|compute|subtract|add|multiply|divide|equals|=|\+|\-|\*|\/)',
                output.lower()
            ))
            has_intermediate = len(extract_numbers(output)) >= 2
            is_undetermined = not answer or "undetermined" in output.lower()

            if has_computation and has_intermediate and is_undetermined:
                self.error_details["partial_computation"] = {
                    "side": side,
                    "numbers_computed": extract_numbers(output)[:5],
                    "output_snippet": output[:300],
                }
                return "只完成局部计算"

            # Also check: partial computation that leads to wrong answer
            # (has intermediate results but doesn't reach final answer)
            if has_computation and has_intermediate and answer:
                # Check if answer only uses one side's facts
                answer_nums = set(extract_numbers(answer))
                all_key_nums = self.nums_a | self.nums_b
                if len(answer_nums & all_key_nums) < len(all_key_nums) * 0.5:
                    self.error_details["partial_computation"] = {
                        "side": side,
                        "answer": answer,
                        "only_used_facts_from": "single_side",
                        "output_snippet": output[:300],
                    }
                    return "只完成局部计算"
        return None

    def _check_dependency_missing(self) -> str | None:
        """9. 依赖图缺失"""
        if self.setting in ("ledger_exec_plan", "ledger_exec_plan_verify"):
            plan = (self.eg.get("final_plan_json") or self.eg.get("initial_plan_json")
                    or self.eg.get("plan_json"))
            if plan:
                steps = plan.get("steps", [])
                # Check if steps reference facts from only one source
                used_fact_ids = set()
                for step in steps:
                    for inp in step.get("inputs", []):
                        if inp.startswith("F"):
                            used_fact_ids.add(inp)

                # Check fact_id references against actual ledger facts
                typed_a = self.eg.get("typed_facts_A", [])
                typed_b = self.eg.get("typed_facts_B", [])
                a_ids = {f.get("fact_id", "") for f in typed_a}
                b_ids = {f.get("fact_id", "") for f in typed_b}

                # After ledger rebuild, fact_ids are renumbered
                # Check if plan uses facts from both sides by looking at referenced facts
                if len(steps) < 2:
                    self.error_details["dependency_missing"] = {
                        "reason": "too_few_steps",
                        "step_count": len(steps),
                    }
                    return "依赖图缺失"

                # Check for unexecutable plans
                executed_result = self.t.get("executed_plan_result") or self.t.get("verified_plan_result")
                if executed_result is None:
                    self.error_details["dependency_missing"] = {
                        "reason": "plan_execution_failed",
                        "step_count": len(steps),
                        "steps": steps[:3],
                    }
                    return "依赖图缺失"

        # For discussion settings: check if agents can't connect facts
        if self.setting in ("free_discussion",):
            for side in ("a", "b"):
                sf = self.solver_finals.get(side, {})
                output = sf.get("output", "")
                # Check if the output mentions needing info from the other side
                if re.search(r'(?:insufficient|missing|need more|don\'?t know|can\'?t determine)', output.lower()):
                    # Check if all needed numbers are actually in the transcript
                    all_nums_in_transcript = numbers_appearing_in_text(self.public_transcript, self.all_nums)
                    if len(all_nums_in_transcript) >= len(self.all_nums) * 0.8:
                        self.error_details["dependency_missing"] = {
                            "side": side,
                            "reason": "all_numbers_available_but_cant_connect",
                            "available": sorted(list(all_nums_in_transcript)),
                            "total": sorted(list(self.all_nums)),
                        }
                        return "依赖图缺失"
        return None

    def _check_arithmetic_error(self) -> str | None:
        """10. 算术执行错误"""
        # For exec_ground: plan executes but gives wrong result
        if self.setting in ("ledger_exec_plan", "ledger_exec_plan_verify"):
            plan_exec_correct = self.t.get("executed_plan_correct") or self.t.get("verified_plan_correct")
            executed_result = self.t.get("executed_plan_result") or self.t.get("verified_plan_result")
            if plan_exec_correct is False and executed_result is not None:
                # Plan executed (no structural failure) but result is wrong
                self.error_details["arithmetic_error"] = {
                    "computed": executed_result,
                    "expected": self.gold,
                    "setting": self.setting,
                }
                return "算术执行错误"
            # Also check verify_rounds for setting 5
            verify_rounds = self.eg.get("verify_rounds", [])
            for vr in verify_rounds:
                if vr.get("executable") and not vr.get("result_matches"):
                    self.error_details["arithmetic_error"] = {
                        "computed": vr.get("computed_result"),
                        "expected": vr.get("expected_result"),
                        "fix_attempted": True,
                        "round": vr.get("round"),
                    }
                    return "算术执行错误"

        # For discussion settings: check if approach is right but calculation wrong
        for side in ("a", "b"):
            sf = self.solver_finals.get(side, {})
            output = sf.get("output", "")
            answer = sf.get("answer", "")
            if not output:
                continue

            # Try to detect: has formula/approach but numeric result is wrong
            gold_val = None
            try:
                gold_val = float(self.gold) if self.gold else None
            except (ValueError, TypeError):
                pass

            if gold_val and answer:
                try:
                    ans_val = float(answer.replace("$", "").replace(",", ""))
                except (ValueError, TypeError):
                    continue

                if ans_val != gold_val:
                    # Check if the approach described would yield the right answer
                    # Look for arithmetic patterns
                    nums_in_reasoning = extract_numbers(output)
                    # Check if answer is off by a simple arithmetic error
                    for n in nums_in_reasoning:
                        if n != ans_val and abs(n - gold_val) < 0.01:
                            self.error_details["arithmetic_error"] = {
                                "side": side,
                                "correct_in_reasoning": n,
                                "wrong_in_answer": ans_val,
                                "gold": gold_val,
                            }
                            return "算术执行错误"
        return None

    def _check_correct_reasoning_wrong_field(self) -> str | None:
        """11. reasoning中正确但候选字段错误"""
        gold_val = None
        try:
            gold_val = float(self.gold) if self.gold else None
        except (ValueError, TypeError):
            return None

        for side in ("a", "b"):
            sf = self.solver_finals.get(side, {})
            output = sf.get("output", "")
            answer = sf.get("answer", "")
            if not output:
                continue

            # Extract all numbers from reasoning
            reasoning_nums = extract_numbers(output)
            if gold_val in reasoning_nums:
                # Correct number appears in reasoning...
                if answer:
                    try:
                        ans_val = float(answer.replace("$", "").replace(",", ""))
                    except (ValueError, TypeError):
                        ans_val = None
                    if ans_val != gold_val:
                        # ...but answer field is different
                        self.error_details["correct_reasoning_wrong_field"] = {
                            "side": side,
                            "correct_in_reasoning": gold_val,
                            "wrong_in_answer_field": ans_val,
                            "snippet": output[:300],
                        }
                        return "reasoning中正确但候选字段错误"
        return None

    def _check_correct_candidate_lost(self) -> str | None:
        """12. 正确候选存在但被verifier/finalizer丢失"""
        gold_val = None
        try:
            gold_val = float(self.gold) if self.gold else None
        except (ValueError, TypeError):
            return None

        # Check if any candidate appearance was correct
        correct_candidates = []
        for ca in self.candidate_appearances:
            if ca.get("correct"):
                correct_candidates.append(ca)

        if correct_candidates and not self.semantic_correct:
            # A correct answer appeared but wasn't selected
            self.error_details["correct_candidate_lost"] = {
                "correct_candidates": [
                    {"source": c.get("source"), "phase": c.get("phase"), "answer": c.get("answer")}
                    for c in correct_candidates
                ],
                "final_prediction": self.pred,
                "finalizer_selected": self.t.get("finalizer_event", {}).get("selected_source"),
            }
            return "正确候选存在但被verifier/finalizer丢失"
        return None

    def _check_fresh_solver_error(self) -> str | None:
        """For settings 3-5: classify errors in fresh solver / executable plan output."""
        if self.setting not in ("ledger_fresh_solver", "ledger_exec_plan",
                                 "ledger_exec_plan_verify", "canonical_ledger"):
            return None

        gold_val = None
        try:
            gold_val = float(self.gold) if self.gold else None
        except (ValueError, TypeError):
            pass

        # Check fresh_solver_event for settings 3-5
        fse = self.t.get("fresh_solver_event", {})
        fresh_raw = fse.get("raw_output", "")
        fresh_pred = fse.get("prediction", "")
        fresh_answer = fse.get("answer", "")

        # Also check for the fresh solver output in canonical_ledger discussion
        if not fresh_raw:
            # In canonical_ledger, the raw solver output is in solver_finals
            for side in ("a", "b"):
                sf = self.solver_finals.get(side, {})
                out = sf.get("output", "")
                if out and len(out) > 50:
                    fresh_raw = out
                    fresh_pred = sf.get("answer", "")
                    break
        if not fresh_raw:
            # For settings 4-5, check plan_raw from exec_ground
            plan_raw = self.eg.get("initial_plan_raw") or self.eg.get("plan_raw") or ""
            if plan_raw and len(str(plan_raw)) > 20:
                fresh_raw = str(plan_raw)
                # Plan execution result serves as the prediction
                executed = self.t.get("executed_plan_result") or self.t.get("verified_plan_result")
                if executed is not None:
                    fresh_pred = str(executed)

        if not fresh_raw and not fresh_pred:
            return None

        # 1. Check if the model completely ignores the question (computes wrong thing)
        question = self.t.get("shared_question", "").lower()
        if fresh_raw:
            fresh_lower = fresh_raw.lower()
            # Detect: model computes total when asked for remaining
            if "how many" in question and "remain" in question:
                if "remain" not in fresh_lower and "total" in fresh_lower:
                    self.error_details["entity_drift"] = {
                        "type": "computed_total_instead_of_remaining",
                        "question": question,
                        "reasoning": fresh_raw[:300],
                    }
                    return "实体或题意漂移"

            # Detect: model computes wrong entity entirely
            if "how much more" in question or "how many more" in question:
                if "total" in fresh_lower and "more" not in fresh_lower:
                    self.error_details["entity_drift"] = {
                        "type": "computed_total_instead_of_difference",
                        "question": question,
                        "reasoning": fresh_raw[:300],
                    }
                    return "实体或题意漂移"

        # 2. Check for arithmetic error: right approach but wrong numbers
        if gold_val and fresh_pred:
            try:
                pred_val = float(str(fresh_pred).replace("$", "").replace(",", "").replace("**", "").strip())
            except (ValueError, TypeError):
                pred_val = None

            if pred_val is not None and pred_val != gold_val:
                # Check if it's close (off by small amount vs off by factor)
                if pred_val != 0 and (abs(pred_val - gold_val) / gold_val < 0.02):
                    self.error_details["arithmetic_error"] = {
                        "type": "near_miss",
                        "predicted": pred_val,
                        "gold": gold_val,
                        "error_pct": abs(pred_val - gold_val) / gold_val * 100,
                    }
                    return "算术执行错误"

                # Check for factor-of errors (off by 2x, 3x, etc.)
                for factor in (2, 3, 0.5, 1/3):
                    if abs(pred_val * factor - gold_val) < 0.01 * gold_val:
                        self.error_details["arithmetic_error"] = {
                            "type": f"off_by_factor_{factor}",
                            "predicted": pred_val,
                            "gold": gold_val,
                        }
                        return "算术执行错误"

                # Check if the model included/excluded a component
                # (pred - gold equals one of the key numbers from conditions)
                diff = abs(pred_val - gold_val)
                all_fact_values = set()
                for f in self.eg.get("typed_facts_A", []) + self.eg.get("typed_facts_B", []):
                    v = f.get("value")
                    if v is not None:
                        all_fact_values.add(v)
                for v in all_fact_values:
                    if abs(diff - v) < 0.01:
                        self.error_details["arithmetic_error"] = {
                            "type": "missed_or_extra_component",
                            "predicted": pred_val,
                            "gold": gold_val,
                            "diff_matches_fact_value": v,
                        }
                        return "算术执行错误"

        # 3. Detect: facts available but model can't integrate (dependency failure)
        if fresh_raw and gold_val and (fresh_pred != str(gold_val)):
            nums_in_reasoning = extract_numbers(fresh_raw)
            if len(nums_in_reasoning) >= 2 and gold_val not in nums_in_reasoning:
                self.error_details["dependency_missing"] = {
                    "type": "fresh_solver_cant_integrate",
                    "numbers_used": nums_in_reasoning[:8],
                    "gold": gold_val,
                    "reasoning_snippet": fresh_raw[:300],
                }
                return "依赖图缺失"

        # 4. Default: model has all facts but computes wrong answer
        if fresh_raw and fresh_pred:
            self.error_details["integration_failure"] = {
                "fresh_prediction": fresh_pred,
                "gold": gold_val,
                "reasoning_snippet": fresh_raw[:300],
            }
            return "只完成局部计算"

        return None

    def _check_llm_extraction_error(self) -> str | None:
        """Check for LLM fact extraction errors (settings 2-5)."""
        if self.setting not in ("canonical_ledger", "ledger_fresh_solver",
                                 "ledger_exec_plan", "ledger_exec_plan_verify"):
            return None

        typed_a = self.eg.get("typed_facts_A", [])
        typed_b = self.eg.get("typed_facts_B", [])

        # Check if key information was lost during LLM extraction
        # Compare extracted facts vs. original conditions
        for fact_list, cond in ((typed_a, self.cond_a), (typed_b, self.cond_b)):
            if not cond:
                continue
            orig_nums = key_numbers_from_condition(cond)
            extracted_nums = set()
            for f in fact_list:
                v = f.get("value")
                if v is not None:
                    extracted_nums.add(v)
                # Also check evidence text
                ev = f.get("evidence", "")
                extracted_nums |= set(extract_numbers(ev))

            # Check if important numbers from condition are missing in extracted facts
            missing = orig_nums - extracted_nums
            if len(missing) >= len(orig_nums) * 0.5 and len(orig_nums) > 0:
                self.error_details["llm_extraction_error"] = {
                    "condition": cond[:100],
                    "original_numbers": sorted(list(orig_nums)),
                    "extracted_numbers": sorted(list(extracted_nums)),
                    "missing": sorted(list(missing)),
                }
                return "事实披露不完整"

        # Check for hallucinated values (extracted number not in condition)
        for fact_list, cond in ((typed_a, self.cond_a), (typed_b, self.cond_b)):
            if not cond:
                continue
            orig_nums = set(extract_numbers(cond))
            for f in fact_list:
                v = f.get("value")
                if v is not None and v not in orig_nums:
                    # Check if it's a relation-derived value
                    relation = f.get("relation", "")
                    if relation in ("half_of", "twice_of", "triple_of"):
                        # Derived values are expected
                        continue
                    # It might be a computed value from evidence text
                    ev = f.get("evidence", "")
                    ev_nums = set(extract_numbers(ev))
                    if v not in ev_nums:
                        self.error_details["fact_values_modified"] = {
                            "hallucinated_value": v,
                            "evidence": ev,
                        }
                        return "事实数值被修改"

        return None

    def earliest_error_round(self) -> int | None:
        """Determine the earliest round where an error became apparent."""
        for i, rr in enumerate(self.round_records):
            st = rr.get("simultaneous_turn", {})
            for side in ("a", "b"):
                output = st.get(side, {}).get("output", "")
                if output and "undetermined" in output.lower():
                    return i + 1
        # Check solver finals
        for side in ("a", "b"):
            sf = self.solver_finals.get(side, {})
            if sf.get("answer") == "" or "undetermined" in sf.get("output", "").lower():
                return len(self.round_records) + 1  # final stage
        return None

    def oracle_can_fix(self, error: str) -> str:
        """Determine which oracle intervention could fix this error."""
        oracle_map = {
            "私有事实没有披露": "oracle_disclosure (Setting 1) — 程序注入所有事实",
            "事实披露不完整": "oracle_disclosure (Setting 1) — 程序注入所有事实",
            "忘记自己的事实": "oracle_canonical_state (Setting 2) — 结构化事实表提醒",
            "忽略对方事实": "oracle_canonical_state (Setting 2) — 统一事实视图",
            "事实数值被修改": "oracle_canonical_state (Setting 2) — 规范化防止修改",
            "half/twice等关系翻转": "oracle_plan (Setting 4) — 方程结构明确关系方向",
            "实体或题意漂移": "canonical_state_fresh (Setting 3) — 清除讨论历史",
            "只完成局部计算": "oracle_plan (Setting 4) — 提供完整方程依赖图",
            "依赖图缺失": "oracle_plan (Setting 4) — 提供方程结构",
            "算术执行错误": "oracle_plan (Setting 4) — 明确计算步骤",
            "reasoning中正确但候选字段错误": "oracle_candidate (Setting 5) — 正确候选直接注入",
            "正确候选存在但被verifier/finalizer丢失": "oracle_candidate (Setting 5) — 标注权威候选",
            "信息足够但仍输出undetermined": "canonical_state_fresh (Setting 3) — 清除讨论历史，直接求解",
        }
        return oracle_map.get(error, "未知 — 可能需要模型能力提升")


def classify_all_traces(traces: list[dict]) -> dict:
    """Classify all traces and return structured results."""
    results = {
        "by_setting": defaultdict(lambda: defaultdict(list)),  # setting -> error -> [trace_info]
        "by_error": defaultdict(list),  # error -> [trace_info]
        "correct_count": 0,
        "total_count": len(traces),
        "all_classifications": [],  # (qid, setting, seed, errors, detail)
    }

    for t in traces:
        classifier = ErrorClassifier(t)
        errors = classifier.classify_all()
        setting = t.get("setting", "")
        qid = t.get("question_id", 0)
        seed = t.get("seed", 0)
        pred = t.get("final_prediction", "")
        gold = t.get("gold_answer", "")
        earliest = classifier.earliest_error_round()

        trace_info = {
            "qid": qid,
            "seed": seed,
            "pred": pred,
            "gold": gold,
            "errors": errors,
            "earliest_error_round": earliest,
            "details": dict(classifier.error_details),
            "failure_type": classifier.failure_type,
        }

        results["all_classifications"].append((qid, setting, seed, errors, classifier))

        if "correct" in errors:
            results["correct_count"] += 1
        else:
            for err in errors:
                results["by_setting"][setting][err].append(trace_info)
                results["by_error"][err].append(trace_info)

    return results


def generate_report(results: dict, output_path: Path):
    """Generate markdown analysis report."""
    lines = [
        "# ExecGround Error Classification Analysis",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Data**: {results['total_count']} traces",
        f"**Correct**: {results['correct_count']} ({results['correct_count']/max(results['total_count'],1):.1%})",
        f"**Errors**: {results['total_count'] - results['correct_count']}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # Per-setting summary
    lines.append("### Error Distribution by Setting")
    lines.append("")
    header = "| Setting | Total | Correct | " + " | ".join(
        [f"E{i}" for i in range(1, 14)]) + " |"
    lines.append(header)
    sep = "|---|" + "---|" * (2 + 13) + "|"
    # Actually simpler header
    lines = lines[:-1]  # remove complex header

    lines.append("| Setting | Traces | Correct | Top Errors |")
    lines.append("|---------|--------|---------|------------|")

    for setting in ["free_discussion", "oracle_disclosure", "canonical_ledger",
                     "ledger_fresh_solver", "ledger_exec_plan", "ledger_exec_plan_verify"]:
        by_err = results["by_setting"].get(setting, {})
        # Dedup by (qid, seed): a trace can have multiple errors
        seen = set()
        for t_info_list in by_err.values():
            for t in t_info_list:
                seen.add((t["qid"], t["seed"]))
        correct_in_setting = sum(
            1 for qid, s, seed, errs, _ in results["all_classifications"]
            if s == setting and "correct" in errs
        )
        total_in_setting = correct_in_setting + len(seen)
        if total_in_setting == 0:
            continue
        top_errors = sorted(by_err.items(), key=lambda x: len(x[1]), reverse=True)
        top_str = ", ".join(f"{e} ({len(v)})" for e, v in top_errors[:3])
        lines.append(
            f"| {SETTING_LABELS.get(setting, setting)} | {total_in_setting} | {correct_in_setting} | {top_str} |"
        )

    lines += ["", "---", "", "## Detailed Error Categories", ""]

    # Define expected order
    error_order = [
        "私有事实没有披露",
        "事实披露不完整",
        "忘记自己的事实",
        "忽略对方事实",
        "事实数值被修改",
        "half/twice等关系翻转",
        "实体或题意漂移",
        "只完成局部计算",
        "依赖图缺失",
        "算术执行错误",
        "reasoning中正确但候选字段错误",
        "正确候选存在但被verifier/finalizer丢失",
        "信息足够但仍输出undetermined",
        "unclassified_error",
    ]

    for i, error_name in enumerate(error_order, 1):
        error_traces = results["by_error"].get(error_name, [])
        if not error_traces:
            continue

        total_errors = results["total_count"] - results["correct_count"]
        pct = len(error_traces) / max(total_errors, 1)

        lines += [
            f"### Error {i}: {error_name}",
            "",
            f"- **Count**: {len(error_traces)} ({pct:.1%} of error traces)",
            f"- **Question IDs**: {sorted(set(t['qid'] for t in error_traces))}",
        ]

        # Setting distribution
        setting_dist = Counter(t.get("setting", t.get("_setting", "")) for t in error_traces)
        # Actually trace_info doesn't have setting directly, let me handle this differently
        # Count from the by_setting view
        setting_counts = {}
        for s_name in ["free_discussion", "oracle_disclosure", "canonical_ledger",
                        "ledger_fresh_solver", "ledger_exec_plan", "ledger_exec_plan_verify"]:
            s_traces = results["by_setting"].get(s_name, {}).get(error_name, [])
            if s_traces:
                setting_counts[s_name] = len(s_traces)

        if setting_counts:
            lines.append("- **Setting distribution**:")
            for s_name, cnt in sorted(setting_counts.items()):
                lines.append(f"  - {SETTING_LABELS.get(s_name, s_name)}: {cnt}")

        # Earliest error round
        rounds = [t.get("earliest_error_round") for t in error_traces if t.get("earliest_error_round")]
        if rounds:
            from statistics import mean
            lines.append(f"- **Avg earliest error round**: {mean(rounds):.1f}")

        # Oracle fix
        sample_classifier = None
        for qid, s_name, seed, errs, clf in results["all_classifications"]:
            if error_name in errs:
                sample_classifier = clf
                break
        if sample_classifier:
            lines.append(f"- **Oracle fix**: {sample_classifier.oracle_can_fix(error_name)}")

        # Typical traces (up to 3)
        lines += ["", "#### Typical Traces", ""]
        for t in error_traces[:3]:
            lines.append(f"- **Q{t['qid']} seed={t['seed']}**: pred={t['pred']!r} gold={t['gold']!r}")
            if t.get("details"):
                for k, v in t["details"].items():
                    if isinstance(v, dict):
                        flat = ", ".join(f"{dk}={dv!r}" for dk, dv in list(v.items())[:3])
                        lines.append(f"  - {k}: {flat}")
                    elif isinstance(v, str) and len(v) > 150:
                        lines.append(f"  - {k}: {v[:150]}...")
                    else:
                        lines.append(f"  - {k}: {v!r}")
            lines.append("")

        lines += ["---", ""]

    # Overall causal attribution
    lines += [
        "## Causal Attribution Summary",
        "",
        "Based on the error classification, the bottlenecks by prevalence:",
        "",
        "| Rank | Error Category | Count | % of Errors | Primary Oracle Fix |",
        "|------|---------------|-------|-------------|-------------------|",
    ]

    sorted_errors = sorted(
        [(e, len(v)) for e, v in results["by_error"].items()],
        key=lambda x: x[1], reverse=True
    )
    for rank, (err, cnt) in enumerate(sorted_errors, 1):
        pct = cnt / max(results["total_count"] - results["correct_count"], 1)
        sample = None
        for qid, s, seed, errs, clf in results["all_classifications"]:
            if err in errs:
                sample = clf
                break
        oracle = sample.oracle_can_fix(err) if sample else "—"
        oracle_short = oracle.split("—")[0].strip() if "—" in oracle else oracle[:60]
        lines.append(f"| {rank} | {err} | {cnt} | {pct:.1%} | {oracle_short} |")

    lines += [
        "",
        "---",
        "",
        "## Interpretation Guide",
        "",
        "1. **Setting 0-1 gap** reflects fact disclosure failures (Categories 1-5).",
        "   If oracle_disclosure recovers significantly, the problem is disclosure.",
        "",
        "2. **Setting 1-2 gap** reflects fact organization failures (Categories 5-7).",
        "   Note: canonical_ledger may underperform due to discussion protocol overhead.",
        "",
        "3. **Setting 2-3 gap** reflects discussion contamination (Categories 8, 9, 13).",
        "   Fresh solver bypasses contaminated discussion history.",
        "",
        "4. **Setting 3-4 gap** reflects plan generation failures (Categories 9, 10).",
        "   If the model can't generate executable plans, this layer adds no value.",
        "",
        "5. **Setting 4-5 gap** reflects fix capability (Categories 10, 12).",
        "   Coverage verification can detect but model may not be able to fix.",
        "",
        "6. **Category 11-12** (correct candidate loss) indicates Finalizer problems.",
        "   Even when the right answer appears, it may not survive to final output.",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main():
    print("Loading traces...")
    with open(TRACES_PATH, "r", encoding="utf-8") as f:
        traces = [json.loads(l) for l in f if l.strip()]
    print(f"  {len(traces)} traces loaded")

    print("Classifying errors...")
    results = classify_all_traces(traces)

    print(f"  Correct: {results['correct_count']}")
    print(f"  Errors: {results['total_count'] - results['correct_count']}")

    print("\nError distribution:")
    for err, trace_list in sorted(results["by_error"].items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {err}: {len(trace_list)}")

    print(f"\nGenerating report...")
    generate_report(results, REPORT_PATH)
    print(f"  Report saved to: {REPORT_PATH}")

    # Also save structured JSON for further analysis
    json_path = OUTPUT_DIR / "error_classification.json"
    json_output = {
        "summary": {
            "total_traces": results["total_count"],
            "correct": results["correct_count"],
            "errors": results["total_count"] - results["correct_count"],
        },
        "error_counts": {e: len(v) for e, v in results["by_error"].items()},
        "by_setting": {},
    }
    for setting, err_dict in results["by_setting"].items():
        json_output["by_setting"][setting] = {
            e: len(v) for e, v in err_dict.items()
        }

    json_path.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  JSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
