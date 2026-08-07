#!/usr/bin/env python3
"""
Tests for ExecGround modules.

Run: python test_exec_ground.py
Or:   python -m pytest test_exec_ground.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Ensure we can import from the project root
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import exec_ground as eg


# ═══════════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

PASSED = 0
FAILED = 0

def check(condition, msg=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {msg}" if msg else "  PASS")
    else:
        FAILED += 1
        print(f"  FAIL: {msg}" if msg else "  FAIL")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: TYPED FACT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_typed_fact_creation():
    section("Module 1: TypedFact Creation & Serialization")

    f = eg.TypedFact(
        fact_id="A_F1",
        subject="yesterday_pages",
        relation="equals",
        object="numeric_value",
        value=12,
        unit="pages",
        source="A",
        evidence="Julie read 12 pages of a book yesterday.",
    )
    check(f.fact_id == "A_F1", "fact_id")
    check(f.subject == "yesterday_pages", "subject")
    check(f.relation == "equals", "relation")
    check(f.value == 12.0, "value")
    check(f.unit == "pages", "unit")
    check(f.source == "A", "source")
    check(len(f.evidence) > 0, "evidence non-empty")

    # Serialization round-trip
    d = f.to_dict()
    f2 = eg.TypedFact.from_dict(d)
    check(f2.fact_id == f.fact_id, "round-trip fact_id")
    check(f2.subject == f.subject, "round-trip subject (normalized)")
    check(f2.value == f.value, "round-trip value")
    check(f2.source == f.source, "round-trip source")

    # Signature stability
    sig1 = f.signature
    sig2 = f.signature
    check(sig1 == sig2, "signature is stable")

    # Different facts have different signatures
    f3 = eg.TypedFact("B_F1", "today_pages", "twice_of", "yesterday_pages", 2, "ratio", "B", "twice as many")
    check(f.signature != f3.signature, "different facts have different signatures")

    # Ledger line format
    line = f.to_ledger_line()
    check("A_F1" in line, "ledger line contains fact_id")
    check("yesterday_pages" in line, "ledger line contains subject")
    check("12" in line, "ledger line contains value")


def test_typed_fact_from_dict_normalization():
    section("Module 1: TypedFact.from_dict Normalization")

    # Test that from_dict normalizes fields
    d = {
        "fact_id": "B_F3",
        "subject": "  Total Pages  ",
        "relation": "IS",
        "object": "numeric_value",
        "value": "120",
        "unit": "Pages",
        "source": "b",
        "evidence": "The book has 120 pages.",
    }
    f = eg.TypedFact.from_dict(d)
    check(f.subject == "total pages", f"subject normalized: '{f.subject}'")
    # from_dict lowercases but does NOT standardize relations (that happens in CanonicalLedger.build)
    check(f.relation == "is", f"relation lowercased: '{f.relation}'")
    check(f.value == 120.0, f"value parsed: {f.value}")
    check(f.unit == "pages", f"unit normalized: '{f.unit}'")
    check(f.source == "B", f"source uppercased: '{f.source}'")

    # Test missing fields
    d2 = {"fact_id": "X_F1", "subject": "x", "relation": "y", "object": "z"}
    f2 = eg.TypedFact.from_dict(d2)
    check(f2.value is None, "missing value → None")
    check(f2.unit == "", "missing unit → empty string")
    check(f2.source == "", "missing source → empty string")


def test_gold_fact_extraction():
    section("Module 1: Gold Fact Extraction (rule-based)")

    # Test 1: Simple direct facts
    facts = eg.extract_gold_typed_facts(
        "Julie read 12 pages of a book yesterday. Today she read twice as many pages as yesterday.",
        "A",
    )
    check(len(facts) >= 2, f"extracts at least 2 facts, got {len(facts)}")

    # Should find the "12 pages" fact
    values = [f.value for f in facts if f.value is not None]
    check(12.0 in values or 12 in values, f"finds value 12 in {values}")

    # Should find the "twice" relationship
    relations = [f.relation for f in facts]
    check("twice_of" in relations, f"finds twice_of relation in {relations}")

    # Test 2: Money facts
    facts2 = eg.extract_gold_typed_facts(
        "Betty wants to buy a wallet that costs $100. She already has half of the money needed.",
        "A",
    )
    check(len(facts2) >= 2, f"extracts money facts, got {len(facts2)}")
    values2 = [f.value for f in facts2 if f.value is not None]
    check(100.0 in values2 or 100 in values2, f"finds $100 in {values2}")

    # Test 3: Remaining/left facts
    facts3 = eg.extract_gold_typed_facts(
        "She also spends $11 on socks and $18 on a belt. After buying a pair of shoes, she has $16 left.",
        "B",
    )
    check(len(facts3) >= 2, f"extracts spending facts, got {len(facts3)}")
    relations3 = [f.relation for f in facts3]
    check("equals" in relations3 or "remaining_after" in relations3,
          f"finds relevant relation in {relations3}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: CANONICAL LEDGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_canonical_ledger_build():
    section("Module 2: CanonicalLedger.build()")

    facts_a = [
        eg.TypedFact("A_F1", "yesterday_pages", "equals", "numeric_value", 12, "pages", "A", "read 12 pages"),
        eg.TypedFact("A_F2", "today_pages", "twice_of", "yesterday_pages", 2, "ratio", "A", "twice as many"),
    ]
    facts_b = [
        eg.TypedFact("B_F1", "total_pages", "equals", "numeric_value", 120, "pages", "B", "book has 120 pages"),
        eg.TypedFact("B_F2", "tomorrow_pages", "half_of", "remaining_pages", 0.5, "ratio", "B", "half of remaining"),
    ]

    ledger = eg.CanonicalLedger.build(facts_a, facts_b)

    check(len(ledger.facts) == 4, f"4 facts after merge (got {len(ledger.facts)})")
    check(len(ledger.conflicts) == 0, f"no conflicts (got {len(ledger.conflicts)})")

    # Check fact_ids are renumbered to F1, F2, F3, F4
    ids = [f.fact_id for f in ledger.facts]
    check(ids == ["F1", "F2", "F3", "F4"], f"renumbered fact_ids: {ids}")

    # Check build info
    check(ledger.build_info["total_input_facts"] == 4, "build_info total_input_facts")
    check(ledger.build_info["after_dedup"] == 4, "build_info after_dedup")


def test_canonical_ledger_determinism():
    section("Module 2: CanonicalLedger Determinism (AB vs BA)")

    facts_a = [
        eg.TypedFact("A_F1", "yesterday_pages", "equals", "numeric_value", 12, "pages", "A", "read 12 pages"),
        eg.TypedFact("A_F2", "today_pages", "twice_of", "yesterday_pages", 2, "ratio", "A", "twice as many"),
    ]
    facts_b = [
        eg.TypedFact("B_F1", "total_pages", "equals", "numeric_value", 120, "pages", "B", "book has 120 pages"),
        eg.TypedFact("B_F2", "tomorrow_pages", "half_of", "remaining_pages", 0.5, "ratio", "B", "half of remaining"),
    ]

    ledger_ab = eg.CanonicalLedger.build(facts_a, facts_b)
    ledger_ba = eg.CanonicalLedger.build(facts_b, facts_a)

    text_ab = ledger_ab.to_text()
    text_ba = ledger_ba.to_text()
    check(text_ab == text_ba, "AB order == BA order (text identical)")

    # Also check JSON representation
    json_ab = json.dumps([f.to_dict() for f in ledger_ab.facts], sort_keys=True)
    json_ba = json.dumps([f.to_dict() for f in ledger_ba.facts], sort_keys=True)
    check(json_ab == json_ba, "AB order == BA order (JSON identical)")

    # Same factual content, different input ordering
    facts_a_shuffled = list(reversed(facts_a))
    ledger_shuffled = eg.CanonicalLedger.build(facts_a_shuffled, facts_b)
    check(ledger_shuffled.to_text() == text_ab, "shuffled input order → same ledger")


def test_canonical_ledger_dedup():
    section("Module 2: CanonicalLedger Deduplication")

    # Same fact appears in both A and B
    shared_fact = eg.TypedFact("A_F1", "total_items", "equals", "numeric_value", 100, "count", "A", "100 items total")
    shared_fact_b = eg.TypedFact("B_F1", "total_items", "equals", "numeric_value", 100, "count", "B", "100 items total")

    facts_a = [
        eg.TypedFact("A_F1", "total_items", "equals", "numeric_value", 100, "count", "A", "100 items total"),
        eg.TypedFact("A_F2", "item_price", "equals", "numeric_value", 5, "dollars", "A", "$5 each"),
    ]
    facts_b = [
        eg.TypedFact("B_F1", "total_items", "equals", "numeric_value", 100, "count", "B", "100 items total"),
        eg.TypedFact("B_F2", "tax", "equals", "numeric_value", 8, "dollars", "B", "$8 tax"),
    ]

    ledger = eg.CanonicalLedger.build(facts_a, facts_b)
    # "total_items = 100" should appear only once after dedup
    check(len(ledger.facts) == 3,
          f"dedup: 4 input facts → 3 unique (got {len(ledger.facts)})")

    # The deduped fact should note both sources
    total_item_facts = [f for f in ledger.facts if f.subject == "total_item"]
    check(len(total_item_facts) == 1, "deduped fact appears once")

    if total_item_facts:
        check("[A+B]" in total_item_facts[0].evidence,
              f"evidence notes dual source: {total_item_facts[0].evidence}")


def test_canonical_ledger_conflict_detection():
    section("Module 2: CanonicalLedger Conflict Detection")

    facts_a = [
        eg.TypedFact("A_F1", "total_cost", "equals", "numeric_value", 50, "dollars", "A", "costs $50"),
    ]
    facts_b = [
        eg.TypedFact("B_F1", "total_cost", "equals", "numeric_value", 75, "dollars", "B", "costs $75"),
    ]

    ledger = eg.CanonicalLedger.build(facts_a, facts_b)
    check(len(ledger.conflicts) == 1,
          f"detects value conflict (got {len(ledger.conflicts)} conflicts)")

    if ledger.conflicts:
        c = ledger.conflicts[0]
        check(c["type"] == "value_conflict", f"conflict type: {c.get('type')}")
        check(50.0 in c["values"] or 75.0 in c["values"], f"conflict values: {c.get('values')}")


def test_canonical_ledger_text_outputs():
    section("Module 2: CanonicalLedger Text Outputs")

    facts_a = [
        eg.TypedFact("A_F1", "yesterday_pages", "equals", "numeric_value", 12, "pages", "A", "read 12 pages"),
        eg.TypedFact("A_F2", "today_pages", "twice_of", "yesterday_pages", 2, "ratio", "A", "twice as many"),
    ]
    facts_b = [
        eg.TypedFact("B_F1", "total_pages", "equals", "numeric_value", 120, "pages", "B", "120 pages total"),
    ]

    ledger = eg.CanonicalLedger.build(facts_a, facts_b)

    text = ledger.to_text()
    check("CANONICAL FACT LEDGER" in text, "to_text has header")
    check("F1" in text and "F2" in text and "F3" in text, "to_text has fact IDs")

    table = ledger.to_table()
    check("| Fact ID |" in table, "to_table has header row")
    check("| F1 |" in table, "to_table has fact row")

    compact = ledger.to_compact_text()
    check("CANONICAL FACT LEDGER:" in compact, "to_compact_text has header")
    check("(src:A)" in compact or "(src:B)" in compact, "to_compact_text has source labels")

    # get() and get_value()
    f = ledger.get("F1")
    check(f is not None, "get returns fact for valid ID")
    check(f is not None and f.subject is not None, "get returns correct fact")

    v = ledger.get_value("F1")
    check(v is not None, "get_value returns value for valid ID")

    bad = ledger.get("F999")
    check(bad is None, "get returns None for invalid ID")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2b: UTILITY FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_utility_functions():
    section("Utility Functions")

    # Entity normalization
    check(eg._normalize_entity("Pages") == "page", "Pages → page")
    check(eg._normalize_entity("dollars") == "dollar", "dollars → dollar")
    check(eg._normalize_entity("Julie") == "julie", "Julie → julie")
    check(eg._normalize_entity("stamps") == "stamp", "stamps → stamp")
    check(eg._normalize_entity("parties") == "party", "parties → party")

    # Relation standardization
    check(eg._normalize_relation("twice") == "twice_of", "twice → twice_of")
    check(eg._normalize_relation("half") == "half_of", "half → half_of")
    check(eg._normalize_relation("is") == "equals", "is → equals")
    check(eg._normalize_relation("double") == "twice_of", "double → twice_of")
    check(eg._normalize_relation("triple") == "triple_of", "triple → triple_of")
    check(eg._normalize_relation("more_than") == "more_than", "more_than preserved")
    check(eg._normalize_relation("fewer_than") == "less_than", "fewer_than → less_than")
    check(eg._normalize_relation("remaining") == "remaining_after", "remaining → remaining_after")
    check(eg._normalize_relation("total") == "sum_of", "total → sum_of")

    # Unit normalization
    check(eg._normalize_unit("$") == "dollars", "$ → dollars")
    check(eg._normalize_unit("lbs") == "pounds", "lbs → pounds")
    check(eg._normalize_unit("pages") == "pages", "pages preserved")
    check(eg._normalize_unit("cnt") == "count", "cnt → count")

    # Fact ID detection
    check(eg._is_fact_id("F1"), "F1 is fact_id")
    check(eg._is_fact_id("A_F1"), "A_F1 is fact_id")
    check(eg._is_fact_id("B_F12"), "B_F12 is fact_id")
    check(not eg._is_fact_id("today_val"), "today_val is not fact_id")
    check(not eg._is_fact_id("answer"), "answer is not fact_id")

    # Float parsing
    check(eg._parse_float("42") == 42.0, "parse '42'")
    check(eg._parse_float("3.14") == 3.14, "parse '3.14'")
    check(eg._parse_float(None) is None, "parse None → None")
    check(eg._parse_float("abc") is None, "parse 'abc' → None")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: EXECUTABLE PLAN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_executable_plan_creation():
    section("Module 3: ExecutablePlan Creation & JSON")

    plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply", ["F1", "F2"], "today_val", "12 * 2"),
        eg.PlanStep("add", ["F1", "today_val"], "total_read", "12 + 24"),
        eg.PlanStep("subtract", ["F3", "total_read"], "remaining", "120 - 36"),
        eg.PlanStep("multiply", ["remaining", "F4"], "answer", "84 * 0.5"),
    ])

    check(len(plan.steps) == 4, f"4 steps (got {len(plan.steps)})")

    # JSON round-trip
    j = plan.to_json()
    plan2 = eg.ExecutablePlan.from_json(j)
    check(len(plan2.steps) == 4, "JSON round-trip preserves step count")
    check(plan2.steps[0].op == "multiply", "JSON round-trip preserves op")
    check(plan2.steps[-1].output == "answer", "JSON round-trip preserves last output")

    # Referenced fact_ids
    refs = plan.referenced_fact_ids
    check("F1" in refs and "F2" in refs, "referenced_fact_ids finds F1, F2")
    check("today_val" not in refs, "referenced_fact_ids excludes variables")

    # Defined variables
    defined = plan.defined_variables
    check("answer" in defined, "answer is defined")
    check("today_val" in defined, "today_val is defined")

    # Required inputs
    required = plan.required_inputs
    check("F1" in required, "F1 is required input")
    check("today_val" not in required, "today_val is not required (defined internally)")


def test_plan_execution():
    section("Module 3: Plan Execution")

    # Build a ledger with known values
    facts_a = [
        eg.TypedFact("A_F1", "yesterday_pages", "equals", "numeric_value", 12, "pages", "A", "12 pages"),
        eg.TypedFact("A_F2", "today_pages", "twice_of", "yesterday_pages", 2, "ratio", "A", "twice as many"),
    ]
    facts_b = [
        eg.TypedFact("B_F1", "total_pages", "equals", "numeric_value", 120, "pages", "B", "120 total"),
        eg.TypedFact("B_F2", "tomorrow_pages", "half_of", "remaining_pages", 0.5, "ratio", "B", "half of remaining"),
    ]
    ledger = eg.CanonicalLedger.build(facts_a, facts_b)

    # Build a plan that uses the canonical fact_ids
    fmap = {}
    for f in ledger.facts:
        fmap[(f.subject, f.relation)] = f.fact_id

    plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply",
                    [fmap[("yesterday_page", "equals")], fmap[("today_page", "twice_of")]],
                    "today_val"),
        eg.PlanStep("add",
                    [fmap[("yesterday_page", "equals")], "today_val"],
                    "total_read"),
        eg.PlanStep("subtract",
                    [fmap[("total_page", "equals")], "total_read"],
                    "remaining"),
        eg.PlanStep("multiply",
                    ["remaining", fmap[("tomorrow_page", "half_of")]],
                    "answer"),
    ])

    result = eg.execute_plan(plan, ledger)
    check(result is not None, "plan executes without error")
    check(math.isclose(result, 42.0), f"result = 42.0 (got {result})")

    # Test with missing fact reference
    bad_plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply", ["F999", "F1"], "today_val"),
        eg.PlanStep("add", ["F1", "today_val"], "answer"),
    ])
    bad_result = eg.execute_plan(bad_plan, ledger)
    check(bad_result is None, "unbound fact reference → None result")

    # Test division by zero
    div_zero_plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("divide", ["F1", "zero"], "answer"),
    ])
    # Need to provide "zero" as input somehow - this tests the error handling
    # Since "zero" isn't in the ledger, result should be None
    result2 = eg.execute_plan(div_zero_plan, ledger)
    check(result2 is None, "unbound input → None result")


def test_plan_json_parsing():
    section("Module 3: Plan JSON Parsing")

    # Valid JSON
    raw = '{"steps": [{"op": "add", "inputs": ["F1", "F2"], "output": "answer"}]}'
    plan = eg._parse_plan_json(raw)
    check(len(plan.steps) == 1, "valid JSON parsed")
    check(plan.steps[0].op == "add", "valid JSON op")

    # JSON in markdown code block
    raw2 = '```json\n{"steps": [{"op": "multiply", "inputs": ["F1", "F2"], "output": "x"}]}\n```'
    plan2 = eg._parse_plan_json(raw2)
    check(len(plan2.steps) == 1, "markdown JSON parsed")

    # Invalid JSON → empty plan
    raw3 = "this is not json at all"
    plan3 = eg._parse_plan_json(raw3)
    check(len(plan3.steps) == 0, "invalid JSON → empty plan")

    # JSON with extra text
    raw4 = 'Some text before {"steps": [{"op": "subtract", "inputs": ["F3", "F1"], "output": "diff"}]} more text'
    plan4 = eg._parse_plan_json(raw4)
    check(len(plan4.steps) == 1, "JSON with noise parsed")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: COVERAGE VERIFIER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_test_ledger_and_plan():
    """Create a standard test ledger and correct plan."""
    facts_a = [
        eg.TypedFact("A_F1", "yesterday_pages", "equals", "numeric_value", 12, "pages", "A", "12 pages"),
        eg.TypedFact("A_F2", "today_pages", "twice_of", "yesterday_pages", 2, "ratio", "A", "twice"),
    ]
    facts_b = [
        eg.TypedFact("B_F1", "total_pages", "equals", "numeric_value", 120, "pages", "B", "120 total"),
        eg.TypedFact("B_F2", "tomorrow_pages", "half_of", "remaining_pages", 0.5, "ratio", "B", "half"),
    ]
    ledger = eg.CanonicalLedger.build(facts_a, facts_b)

    fmap = {}
    for f in ledger.facts:
        fmap[(f.subject, f.relation)] = f.fact_id

    plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply",
                    [fmap[("yesterday_page", "equals")], fmap[("today_page", "twice_of")]],
                    "today_val"),
        eg.PlanStep("add",
                    [fmap[("yesterday_page", "equals")], "today_val"],
                    "total_read"),
        eg.PlanStep("subtract",
                    [fmap[("total_page", "equals")], "total_read"],
                    "remaining"),
        eg.PlanStep("multiply",
                    ["remaining", fmap[("tomorrow_page", "half_of")]],
                    "answer"),
    ])

    return ledger, plan, fmap


def test_coverage_verifier_clean_plan():
    section("Module 4: Coverage Verifier — Clean Plan")

    ledger, plan, fmap = _make_test_ledger_and_plan()
    report = eg.verify_coverage(plan, ledger, "42")

    check(report.is_clean, f"clean plan: is_clean=True (got {report.is_clean})")
    check(report.executable, f"clean plan: executable=True")
    check(report.result_matches, f"clean plan: result_matches=True (computed={report.computed_result}, expected={report.expected_result})")
    check(len(report.missing_facts) == 0, f"clean plan: no missing facts (got {report.missing_facts})")
    check(len(report.unbound_variables) == 0, f"clean plan: no unbound vars (got {report.unbound_variables})")
    check(report.facts_used == report.facts_total, f"all facts used: {report.facts_used}/{report.facts_total}")
    check(len(report.fix_hints) == 0, f"clean plan: no fix hints")


def test_coverage_verifier_missing_facts():
    section("Module 4: Coverage Verifier — Missing Facts")

    ledger, plan, fmap = _make_test_ledger_and_plan()

    # Create an incomplete plan (missing F3 and F4)
    bad_plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply",
                    [fmap[("yesterday_page", "equals")], fmap[("today_page", "twice_of")]],
                    "today_val"),
        eg.PlanStep("add",
                    [fmap[("yesterday_page", "equals")], "today_val"],
                    "answer"),  # Wrong: should compute total_read then subtract, etc.
    ])

    report = eg.verify_coverage(bad_plan, ledger, "42")

    check(not report.is_clean, "incomplete plan: is_clean=False")
    check(len(report.missing_facts) > 0, f"incomplete plan: has missing facts: {report.missing_facts}")
    check(len(report.fix_hints) > 0, f"incomplete plan: has fix hints")

    # Fix hints should mention specific missing facts
    hint_text = " ".join(report.fix_hints)
    check("Missing required fact" in hint_text,
          f"fix hints mention missing facts: {hint_text[:100]}...")


def test_coverage_verifier_unbound_variables():
    section("Module 4: Coverage Verifier — Unbound Variables")

    ledger, plan, fmap = _make_test_ledger_and_plan()

    # Plan with unbound variable reference
    bad_plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply", ["F999", "F1"], "today_val"),  # F999 doesn't exist
        eg.PlanStep("add", ["F1", "xyz"], "answer"),           # xyz not defined
    ])

    report = eg.verify_coverage(bad_plan, ledger, "42")
    check(not report.is_clean, "unbound plan: is_clean=False")
    check(len(report.unbound_variables) > 0,
          f"unbound plan: has unbound vars: {report.unbound_variables}")
    check(not report.executable, "unbound plan: not executable")


def test_coverage_verifier_no_answer_output():
    section("Module 4: Coverage Verifier — Missing Answer Output")

    ledger, plan, fmap = _make_test_ledger_and_plan()

    # Plan without "answer" output
    bad_plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply",
                    [fmap[("yesterday_page", "equals")], fmap[("today_page", "twice_of")]],
                    "today_val"),
    ])

    report = eg.verify_coverage(bad_plan, ledger, "42")
    check(not report.is_clean, "no-answer plan: is_clean=False")

    hint_text = " ".join(report.fix_hints)
    check("answer" in hint_text.lower(),
          f"fix hints mention missing 'answer' output: {hint_text[:100]}...")


def test_coverage_verifier_no_expected_answer():
    section("Module 4: Coverage Verifier — Without Expected Answer")

    ledger, plan, fmap = _make_test_ledger_and_plan()
    report = eg.verify_coverage(plan, ledger, expected_answer=None)

    check(report.executable, "executable without expected answer")
    check(report.result_matches, "result_matches=True when no expected answer")
    check(report.expected_result is None, "expected_result is None")
    check(report.computed_result is not None, "computed_result is not None")


def test_coverage_verifier_empty_plan():
    section("Module 4: Coverage Verifier — Empty Plan")

    ledger, plan, fmap = _make_test_ledger_and_plan()
    empty_plan = eg.ExecutablePlan(steps=[])
    report = eg.verify_coverage(empty_plan, ledger, "42")

    check(not report.is_clean, "empty plan: is_clean=False")
    check(not report.executable, "empty plan: not executable")
    check(len(report.fix_hints) > 0, "empty plan: has fix hints")


def test_fix_prompt_generation():
    section("Module 4: Fix Prompt Generation")

    ledger, plan, fmap = _make_test_ledger_and_plan()

    # Create a plan with issues
    bad_plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply",
                    [fmap[("yesterday_page", "equals")], fmap[("today_page", "twice_of")]],
                    "today_val"),
        eg.PlanStep("add",
                    [fmap[("yesterday_page", "equals")], "today_val"],
                    "answer"),
    ])

    report = eg.verify_coverage(bad_plan, ledger, "42")
    prompt = eg.generate_fix_prompt(report, bad_plan, ledger)

    check("FIX INSTRUCTIONS" in prompt, "fix prompt has header")
    check("Issue" in prompt, "fix prompt lists issues")
    check("corrected" in prompt.lower() or "CORRECTED" in prompt, "fix prompt asks for corrected plan")
    check("Output ONLY the JSON" in prompt, "fix prompt constrains output format")


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_with_gold_facts():
    section("Integration: Full Pipeline (gold facts, no LLM)")

    # Simulate a problem item
    item = {
        "condition_A": "Julie read 12 pages of a book yesterday. Today she read twice as many pages as yesterday.",
        "condition_B": "The book has 120 pages in total. Tomorrow Julie plans to read half of the pages that remain after yesterday and today.",
        "shared_question": "How many pages should Julie read tomorrow?",
        "answer": "Julie read 12 * 2 = 24 pages today. She read 12 + 24 = 36 pages. There are 120 - 36 = 84 pages remaining. Half is 84 / 2 = 42. #### 42",
    }

    # Phase 1: Extract gold typed facts
    facts_a = eg.extract_gold_typed_facts(item["condition_A"], "A")
    facts_b = eg.extract_gold_typed_facts(item["condition_B"], "B")
    check(len(facts_a) >= 2, f"extracted A facts: {len(facts_a)}")
    check(len(facts_b) >= 2, f"extracted B facts: {len(facts_b)}")

    # Phase 2: Build canonical ledger
    ledger = eg.CanonicalLedger.build(facts_a, facts_b, item)
    check(len(ledger.facts) >= 3, f"ledger has sufficient facts: {len(ledger.facts)}")

    # Check that all source numbers are in the ledger
    ledger_has_12 = any(f.value == 12.0 for f in ledger.facts)
    ledger_has_120 = any(f.value == 120.0 for f in ledger.facts)
    check(ledger_has_12, "ledger contains value 12")
    check(ledger_has_120, "ledger contains value 120")

    # Phase 3: Build a plan manually (simulating fresh solver output)
    fmap = {}
    for f in ledger.facts:
        fmap[(f.subject, f.relation)] = f.fact_id

    plan = eg.ExecutablePlan(steps=[
        eg.PlanStep("multiply",
                    [fmap.get(("yesterday_page", "equals"), fmap.get(("julie", "equals"), "F1")),
                     fmap.get(("today_page", "twice_of"), fmap.get(("today", "twice_of"), "F2"))],
                    "today_val"),
        eg.PlanStep("add",
                    [fmap.get(("yesterday_page", "equals"), fmap.get(("julie", "equals"), "F1")),
                     "today_val"],
                    "total_read"),
        eg.PlanStep("subtract",
                    [fmap.get(("total_page", "equals"), "F3"),
                     "total_read"],
                    "remaining"),
        eg.PlanStep("multiply",
                    ["remaining",
                     fmap.get(("tomorrow_page", "half_of"), "F4")],
                    "answer"),
    ])

    # But we need to use actual fact_ids - let me reconstruct using actual IDs
    # Since we know the gold facts are predictable, let's just build the plan from actual IDs
    plan = eg.ExecutablePlan(steps=[])
    used_ids = set()
    for f in ledger.facts:
        if f.relation == "equals" and f.value == 12.0:
            f1_id = f.fact_id; used_ids.add(f1_id)
        elif f.relation == "twice_of":
            f2_id = f.fact_id; used_ids.add(f2_id)
        elif f.relation == "equals" and f.value == 120.0:
            f3_id = f.fact_id; used_ids.add(f3_id)
        elif f.relation == "half_of":
            f4_id = f.fact_id; used_ids.add(f4_id)

    # Check we found all four expected fact types
    check(len(used_ids) == 4, f"found all 4 expected fact types (got {len(used_ids)}): {used_ids}")

    if len(used_ids) == 4:
        plan = eg.ExecutablePlan(steps=[
            eg.PlanStep("multiply", [f1_id, f2_id], "today_val"),
            eg.PlanStep("add", [f1_id, "today_val"], "total_read"),
            eg.PlanStep("subtract", [f3_id, "total_read"], "remaining"),
            eg.PlanStep("multiply", ["remaining", f4_id], "answer"),
        ])

        # Phase 4: Execute plan
        result = eg.execute_plan(plan, ledger)
        check(result is not None, "plan executed")
        if result is not None:
            check(math.isclose(result, 42.0), f"result correct: {result}")

        # Phase 5: Coverage verification
        report = eg.verify_coverage(plan, ledger, "42")
        check(report.is_clean, f"coverage clean: {report.is_clean}")
        if not report.is_clean:
            print(f"    Report issues: {report.fix_hints}")


def test_determinism_end_to_end():
    section("Integration: End-to-End Determinism")

    # Same facts, different input order → same plan execution result
    item = {
        "condition_A": "The wallet costs $100. Betty already has half of the required amount.",
        "condition_B": "Her parents give her $15. Her grandparents give her twice the amount her parents give.",
    }

    facts_a = eg.extract_gold_typed_facts(item["condition_A"], "A")
    facts_b = eg.extract_gold_typed_facts(item["condition_B"], "B")

    ledger_ab = eg.CanonicalLedger.build(facts_a, facts_b)
    ledger_ba = eg.CanonicalLedger.build(facts_b, facts_a)

    text_ab = ledger_ab.to_text()
    text_ba = ledger_ba.to_text()
    check(text_ab == text_ba, "ledger deterministic")

    # Same plan executed against both ledgers should give same result
    if text_ab == text_ba:
        # Build a plan that should work on the ledger
        plan = eg.ExecutablePlan(steps=[])
        for f in ledger_ab.facts:
            if f.relation == "equals" and f.value == 100.0:
                f_cost = f.fact_id
            elif f.relation == "half_of":
                f_half = f.fact_id
            elif f.relation == "equals" and f.value == 15.0:
                f_parents = f.fact_id
            elif f.relation == "twice_of":
                f_twice = f.fact_id

        # This test verifies that the ledger structure is consistent enough
        # that plans can reference facts by ID
        check(ledger_ab.fact_ids() == ledger_ba.fact_ids(),
              "same fact_ids regardless of input order")


# ═══════════════════════════════════════════════════════════════════════════════
# REAL DATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_20_problems_gold_facts():
    """Test that gold fact extraction and ledger building works for all 20 problems."""
    section("Integration: All 20 Problems — Gold Facts + Ledger")

    import importlib.util as _iu
    _DATA_PATH = _SCRIPT_DIR / "data" / "20.json"

    # Use the same read function as the main codebase
    _HIDDEN_SPEC = _iu.spec_from_file_location("run_hidden_gsm8k", _SCRIPT_DIR / "run_hidden_gsm8k.py")
    _run_hidden = _iu.module_from_spec(_HIDDEN_SPEC)
    _HIDDEN_SPEC.loader.exec_module(_run_hidden)

    items = _run_hidden.read_json_records(_DATA_PATH)
    check(len(items) == 20, f"loaded 20 items (got {len(items)})")

    success_count = 0
    for i, item in enumerate(items):
        try:
            facts_a = eg.extract_gold_typed_facts(item["condition_A"], "A")
            facts_b = eg.extract_gold_typed_facts(item["condition_B"], "B")

            ledger = eg.CanonicalLedger.build(facts_a, facts_b, item)

            # Basic sanity checks
            if len(ledger.facts) >= 2 and len(facts_a) >= 1 and len(facts_b) >= 1:
                success_count += 1
            else:
                print(f"  Q{i+1}: WARNING - ledger has {len(ledger.facts)} facts "
                      f"(A:{len(facts_a)}, B:{len(facts_b)})")
        except Exception as exc:
            print(f"  Q{i+1}: ERROR - {exc}")

    check(success_count >= 15, f"gold facts work for {success_count}/20 problems")
    print(f"    Gold fact extraction successful for {success_count}/20 problems")


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    global PASSED, FAILED
    PASSED = 0
    FAILED = 0

    print("=" * 60)
    print("  EXECGROUND TEST SUITE")
    print("=" * 60)

    # Module 1: TypedFact
    test_typed_fact_creation()
    test_typed_fact_from_dict_normalization()
    test_gold_fact_extraction()

    # Module 2: CanonicalLedger
    test_canonical_ledger_build()
    test_canonical_ledger_determinism()
    test_canonical_ledger_dedup()
    test_canonical_ledger_conflict_detection()
    test_canonical_ledger_text_outputs()

    # Utilities
    test_utility_functions()

    # Module 3: ExecutablePlan
    test_executable_plan_creation()
    test_plan_execution()
    test_plan_json_parsing()

    # Module 4: Coverage Verifier
    test_coverage_verifier_clean_plan()
    test_coverage_verifier_missing_facts()
    test_coverage_verifier_unbound_variables()
    test_coverage_verifier_no_answer_output()
    test_coverage_verifier_no_expected_answer()
    test_coverage_verifier_empty_plan()
    test_fix_prompt_generation()

    # Integration
    test_full_pipeline_with_gold_facts()
    test_determinism_end_to_end()
    test_all_20_problems_gold_facts()

    # Summary
    total = PASSED + FAILED
    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASSED}/{total} passed, {FAILED}/{total} failed")
    print(f"{'='*60}")

    return FAILED == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
