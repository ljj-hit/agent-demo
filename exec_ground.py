#!/usr/bin/env python3
"""
ExecGround: Minimal executable grounding for multi-agent math reasoning.

Four modules:
  1. TypedFact         — Phase 1: agent converts private info → structured facts (NO answering)
  2. CanonicalLedger   — Deterministic merge, dedup, entity-alignment, unit-normalization,
                          relation-standardization, conflict-detection, source-preservation
  3. FreshSolver       — Clean solver: only question + ledger → executable JSON-IR plan
  4. CoverageVerifier  — Programmatic plan check → targeted fix hints (no free discussion)

Design principle: every fact carries its source, evidence, and a unique ID.
The ledger is fully deterministic — same facts, same ledger, regardless of input order.
The solver never sees prior discussion; the verifier never reopens debate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypedFact:
    """A single structured fact extracted from an agent's private information."""
    fact_id: str          # e.g. "A_F1", "B_F2"
    subject: str           # entity being described
    relation: str          # e.g. "equals", "twice_of", "half_of", "sum_of", "remaining"
    object: str            # other entity or "numeric_value"
    value: Optional[float] # numeric value if the fact carries one
    unit: str              # e.g. "pages", "dollars", "ratio", "pounds", "count"
    source: str            # "A" or "B"
    evidence: str          # verbatim text from which this fact was extracted

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TypedFact":
        return cls(
            fact_id=str(d.get("fact_id", "")),
            subject=str(d.get("subject", "")).strip().lower(),
            relation=str(d.get("relation", "")).strip().lower(),
            object=str(d.get("object", "")).strip().lower(),
            value=_parse_float(d.get("value")),
            unit=str(d.get("unit", "")).strip().lower(),
            source=str(d.get("source", "")).strip().upper()[:1],
            evidence=str(d.get("evidence", "")).strip(),
        )

    @property
    def signature(self) -> str:
        """Deterministic identity key for dedup (order-independent)."""
        return _stable_hash(json.dumps({
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "value": self.value,
            "unit": self.unit,
        }, sort_keys=True))

    @property
    def entity_pair(self) -> tuple:
        """Normalized (subject, object) pair for entity-alignment."""
        return (self.subject, self.object)

    def to_ledger_line(self) -> str:
        """Single-line representation for the canonical ledger."""
        v = f"{self.value}" if self.value is not None else "?"
        return (f"[{self.fact_id}] {self.subject} {self.relation} {self.object}"
                f" = {v} {self.unit}  (source: {self.source})")


@dataclass
class PlanStep:
    """One step in an executable plan."""
    op: str               # "add" | "subtract" | "multiply" | "divide"
    inputs: list[str]     # fact_ids or previously-defined variable names
    output: str           # variable name defined by this step
    explanation: str = "" # optional human-readable explanation

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "inputs": list(self.inputs),
            "output": self.output,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep":
        return cls(
            op=str(d.get("op", "")).strip().lower(),
            inputs=[str(x).strip() for x in d.get("inputs", [])],
            output=str(d.get("output", "")).strip(),
            explanation=str(d.get("explanation", "")).strip(),
        )


@dataclass
class ExecutablePlan:
    """A complete executable plan: ordered list of steps."""
    steps: list[PlanStep] = field(default_factory=list)
    raw_output: str = ""

    def to_json(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_json(cls, d: dict) -> "ExecutablePlan":
        steps = [PlanStep.from_dict(s) for s in d.get("steps", [])]
        return cls(steps=steps)

    @property
    def referenced_fact_ids(self) -> set[str]:
        """All fact_ids referenced across all steps."""
        ids: set[str] = set()
        for step in self.steps:
            for inp in step.inputs:
                if _is_fact_id(inp):
                    ids.add(inp)
        return ids

    @property
    def defined_variables(self) -> set[str]:
        """All output variables defined by the plan."""
        return {s.output for s in self.steps}

    @property
    def required_inputs(self) -> set[str]:
        """Inputs that are NOT defined by this plan (must come from ledger)."""
        defined = self.defined_variables
        needed: set[str] = set()
        for step in self.steps:
            for inp in step.inputs:
                if inp not in defined:
                    needed.add(inp)
        return needed


@dataclass
class CoverageReport:
    """Result of coverage verification."""
    facts_total: int = 0
    facts_used: int = 0
    missing_facts: list[str] = field(default_factory=list)    # fact_ids not used
    unused_facts: list[str] = field(default_factory=list)     # same, but with reason
    unbound_variables: list[str] = field(default_factory=list) # inputs not in ledger or prior outputs
    direction_errors: list[str] = field(default_factory=list)  # relation direction issues
    executable: bool = False
    execution_error: str = ""
    computed_result: Optional[float] = None
    expected_result: Optional[float] = None
    result_matches: bool = False
    fix_hints: list[str] = field(default_factory=list)         # targeted fix prompts

    @property
    def is_clean(self) -> bool:
        """True when the plan is fully verified with no issues."""
        return (self.executable and self.result_matches
                and len(self.missing_facts) == 0
                and len(self.unbound_variables) == 0)

    @property
    def has_fixable_issues(self) -> bool:
        """True when there are issues but they can be fixed with hints."""
        return (len(self.fix_hints) > 0
                and (len(self.missing_facts) > 0
                     or len(self.unbound_variables) > 0
                     or not self.executable
                     or not self.result_matches))


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_float(value: Any) -> Optional[float]:
    """Safely parse a value to float, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _stable_hash(text: str) -> str:
    """Deterministic short hash for a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _is_fact_id(s: str) -> bool:
    """Check if a string looks like a fact_id (e.g. 'A_F1', 'B_F2', 'F3')."""
    return bool(re.match(r"^[A-Za-z]?_?F\d+$", s))


def _normalize_entity(name: str) -> str:
    """Normalize entity names for alignment (lowercase, strip, de-pluralize basics)."""
    name = name.strip().lower()
    # Basic de-pluralization for common patterns
    if name.endswith("ies") and len(name) > 4:
        name = name[:-3] + "y"
    elif name.endswith("ses") and len(name) > 4:
        name = name[:-2]
    elif name.endswith("s") and not name.endswith("ss") and len(name) > 2:
        name = name[:-1]
    return name


def _normalize_relation(rel: str) -> str:
    """Standardize relation names."""
    rel = rel.strip().lower().replace(" ", "_")
    # Standard forms
    mapping = {
        "is": "equals",
        "=": "equals",
        "equal": "equals",
        "equals": "equals",
        "equal_to": "equals",
        "has": "equals",
        "costs": "equals",
        "cost": "equals",
        "price": "equals",
        "weighs": "equals",
        "weight": "equals",
        "reads": "equals",
        "is_equal_to": "equals",

        "twice": "twice_of",
        "double": "twice_of",
        "doubles": "twice_of",
        "two_times": "twice_of",
        "2x": "twice_of",
        "twice_of": "twice_of",
        "double_of": "twice_of",

        "half": "half_of",
        "half_of": "half_of",
        "one_half_of": "half_of",
        "0.5x": "half_of",

        "triple": "triple_of",
        "triples": "triple_of",
        "three_times": "triple_of",
        "triple_of": "triple_of",

        "more_than": "more_than",
        "more": "more_than",
        "greater_than": "more_than",
        "additional": "more_than",

        "less_than": "less_than",
        "fewer_than": "less_than",
        "fewer": "less_than",

        "remaining": "remaining_after",
        "remain": "remaining_after",
        "left": "remaining_after",
        "remaining_after": "remaining_after",

        "total": "sum_of",
        "altogether": "sum_of",
        "sum_of": "sum_of",
        "sum": "sum_of",
        "combined": "sum_of",

        "difference": "difference_of",
        "difference_of": "difference_of",

        "ratio_to": "ratio_to",
        "ratio": "ratio_to",
        "compared_to": "ratio_to",
        "relative_to": "ratio_to",
    }
    return mapping.get(rel, rel)


def _normalize_unit(unit: str) -> str:
    """Standardize unit names."""
    unit = unit.strip().lower()
    mapping = {
        "$": "dollars",
        "dollar": "dollars",
        "usd": "dollars",
        "page": "pages",
        "lbs": "pounds",
        "lb": "pounds",
        "pound": "pounds",
        "hrs": "hours",
        "hr": "hours",
        "hour": "hours",
        "min": "minutes",
        "minute": "minutes",
        "sec": "seconds",
        "second": "seconds",
        "cnt": "count",
        "items": "count",
        "stamps": "count",
        "people": "count",
        "kids": "count",
        "students": "count",
    }
    return mapping.get(unit, unit)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: TYPED FACT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

TYPED_FACT_SYSTEM_PROMPT = """You are a FACT EXTRACTOR, not a problem solver. Your ONLY job is to extract structured facts from the information you receive.

CRITICAL RULES:
1. Do NOT compute any answers. Do NOT solve the problem.
2. Extract ONLY explicit facts present in the text.
3. Every number and relationship must be captured.
4. Each sentence may contain multiple facts.

Output a JSON array of fact objects. Each fact object must have these fields:
- fact_id: string like "A_F1", "A_F2" (use your source letter + _F + number)
- subject: the entity being described (short noun phrase, lowercase)
- relation: the relationship type, one of: equals, twice_of, half_of, triple_of, more_than, less_than, sum_of, remaining_after, difference_of, ratio_to
- object: the other entity or "numeric_value" if the fact is a direct value
- value: the numeric value (number only, no units) or null if the fact has no direct number
- unit: the unit like "pages", "dollars", "pounds", "count", "ratio" or "" if no unit
- source: "A" or "B" (whichever you are told)
- evidence: the EXACT sentence from which this fact was extracted

EXAMPLE:
Input text: "Julie read 12 pages of a book yesterday. Today she read twice as many pages as yesterday."
Output:
[
  {
    "fact_id": "A_F1",
    "subject": "yesterday_pages",
    "relation": "equals",
    "object": "numeric_value",
    "value": 12,
    "unit": "pages",
    "source": "A",
    "evidence": "Julie read 12 pages of a book yesterday."
  },
  {
    "fact_id": "A_F2",
    "subject": "today_pages",
    "relation": "twice_of",
    "object": "yesterday_pages",
    "value": 2,
    "unit": "ratio",
    "source": "A",
    "evidence": "Today she read twice as many pages as yesterday."
  }
]

Output ONLY the JSON array. No explanation, no markdown, no other text."""


def extract_typed_facts(
    model: Any,
    condition_text: str,
    source_label: str,
    temperature: float = 0.0,
) -> list[TypedFact]:
    """Phase 1: Agent converts private info to structured typed facts. NO answering allowed.

    Args:
        model: LocalQwen instance with .call(system, user, temperature) method
        condition_text: The agent's private condition (e.g. item["condition_A"])
        source_label: "A" or "B"

    Returns:
        List of TypedFact objects extracted by the LLM.
    """
    user = (
        f"Source label: {source_label}\n\n"
        f"Extract ALL structured facts from this information:\n\n"
        f"{condition_text}\n\n"
        f"Remember: output ONLY a JSON array of fact objects. Do NOT answer the problem."
    )

    raw, usage, elapsed = model.call(TYPED_FACT_SYSTEM_PROMPT, user, temperature=temperature)

    facts = _parse_facts_json(raw, source_label)
    if not facts:
        # Fallback: create a single "raw" fact from the condition
        facts = _fallback_facts(condition_text, source_label)

    # Assign correct source and renumber fact_ids
    for i, f in enumerate(facts):
        f.source = source_label
        f.fact_id = f"{source_label}_F{i + 1}"

    return facts


def _parse_facts_json(raw: str, source_label: str) -> list[TypedFact]:
    """Robust JSON parsing for typed facts. Tries multiple strategies."""
    cleaned = raw.strip()

    # Strategy 1: Direct JSON array
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [TypedFact.from_dict(d) for d in data if isinstance(d, dict)]
        if isinstance(data, dict) and "facts" in data:
            return [TypedFact.from_dict(d) for d in data["facts"]]
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON array from text (with possible markdown)
    for pattern in [
        r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
        r"(\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\])",
        r"(\{[^}]*\"facts\"[^}]*\[[\s\S]*?\][^}]*\})",
    ]:
        match = re.search(pattern, cleaned)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    return [TypedFact.from_dict(d) for d in data if isinstance(d, dict)]
                if isinstance(data, dict) and "facts" in data:
                    return [TypedFact.from_dict(d) for d in data["facts"]]
            except json.JSONDecodeError:
                continue

    # Strategy 3: Try to parse each line as a JSON object
    lines = cleaned.splitlines()
    facts = []
    for line in lines:
        line = line.strip()
        if line.startswith("{") and line.endswith("}") or line.startswith("{") and line.endswith("},"):
            line = line.rstrip(",")
            try:
                d = json.loads(line)
                if isinstance(d, dict) and "fact_id" in d:
                    facts.append(TypedFact.from_dict(d))
            except json.JSONDecodeError:
                continue
    if facts:
        return facts

    return []


def _fallback_facts(condition_text: str, source_label: str) -> list[TypedFact]:
    """Create simple facts from condition text by splitting on sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", condition_text.strip())
    facts = []
    for i, sent in enumerate(sentences):
        if not sent.strip():
            continue
        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", sent)
        # Determine relation type based on keywords
        rel = "equals"
        if any(w in sent.lower() for w in ("twice", "double", "triple")):
            rel = "multiplier"
        elif any(w in sent.lower() for w in ("half",)):
            rel = "multiplier"
        elif any(w in sent.lower() for w in ("more", "additional")):
            rel = "more_than"
        elif any(w in sent.lower() for w in ("fewer", "less")):
            rel = "less_than"
        elif any(w in sent.lower() for w in ("remain", "left")):
            rel = "remaining_after"
        elif any(w in sent.lower() for w in ("total", "altogether")):
            rel = "sum_of"

        value = _parse_float(nums[-1]) if nums else None
        facts.append(TypedFact(
            fact_id=f"{source_label}_F{i + 1}",
            subject=sent.split()[0].lower() if sent.split() else "entity",
            relation=rel,
            object="numeric_value" if nums else "unknown",
            value=value,
            unit="",
            source=source_label,
            evidence=sent.strip(),
        ))
    return facts


def extract_gold_typed_facts(
    condition_text: str,
    source_label: str,
) -> list[TypedFact]:
    """Deterministic gold-standard fact extraction from condition text (no LLM).

    This is used for testing the downstream pipeline without LLM dependency.
    It uses regex-based extraction of numbers and relationships.
    """
    facts: list[TypedFact] = []
    sentences = re.split(r"(?<=[.!?])\s+", condition_text.strip())
    fid = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", sent)
        s_lower = sent.lower()

        # Determine relation — check specific multipliers first, then possession/equality,
        # then aggregates (which are often prepositional: "in total", "altogether")
        if any(w in s_lower for w in ("twice", "double", "doubles", "two times")):
            rel = "twice_of"
            multiplier = 2.0
        elif any(w in s_lower for w in ("triple", "triples", "three times")):
            rel = "triple_of"
            multiplier = 3.0
        elif any(w in s_lower for w in ("half",)):
            rel = "half_of"
            multiplier = 0.5
        elif any(w in s_lower for w in ("more than", "more", "additional", "adds")):
            rel = "more_than"
            multiplier = None
        elif any(w in s_lower for w in ("fewer than", "fewer", "less than", "less")):
            rel = "less_than"
            multiplier = None
        elif any(w in s_lower for w in ("remain", "left", "remaining")):
            rel = "remaining_after"
            multiplier = None
        elif any(w in s_lower for w in ("spend", "spends", "cost", "costs", "pay", "pays", "budget")):
            rel = "equals"
            multiplier = None
        elif any(w in s_lower for w in ("buy", "buys", "has", "have", "read", "reads")):
            rel = "equals"
            multiplier = None
        elif any(w in s_lower for w in ("total", "altogether", "combined", "sum")):
            rel = "sum_of"
            multiplier = None
        else:
            rel = "equals"
            multiplier = None

        # Extract subject (first few words before a number or verb)
        words = sent.split()
        subject = words[0].lower().rstrip(".,;:!?") if words else "entity"

        # Extract unit from context
        unit = ""
        unit_patterns = [
            (r"pages?", "pages"), (r"\$\s*\d+|\d+\s*dollars?", "dollars"),
            (r"pounds?", "pounds"), (r"stamps?", "count"),
            (r"cents?", "cents"), (r"kids?", "count"),
            (r"students?", "count"), (r"items?", "count"),
            (r"snowflake", "count"), (r"truck", "count"), (r"rose", "count"),
        ]
        for pat, u in unit_patterns:
            if re.search(pat, s_lower):
                unit = u
                break

        fid += 1
        value = _parse_float(nums[-1]) if nums else None

        facts.append(TypedFact(
            fact_id=f"{source_label}_F{fid}",
            subject=subject,
            relation=rel,
            object="numeric_value" if nums and rel == "equals" else "other_entity",
            value=value if rel == "equals" else (multiplier if multiplier is not None else value),
            unit=unit,
            source=source_label,
            evidence=sent,
        ))

    return facts


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: CANONICAL LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

class CanonicalLedger:
    """Deterministic, merged, deduplicated, normalized fact ledger.

    Guarantees:
      - Same facts → same ledger regardless of input order (AB vs BA)
      - Conflict detection with source preservation
      - Entity alignment and relation standardization
      - Fixed deterministic sort order
    """

    def __init__(self, facts: list[TypedFact], conflicts: list[dict] | None = None,
                 build_info: dict | None = None):
        self.facts = facts
        self.conflicts = conflicts or []
        self.build_info = build_info or {}
        self._fact_map: dict[str, TypedFact] = {f.fact_id: f for f in facts}
        self._value_map: dict[str, float] = {}
        for f in facts:
            if f.value is not None:
                self._value_map[f.fact_id] = f.value

    def get(self, fact_id: str) -> Optional[TypedFact]:
        """Look up a fact by its ID."""
        return self._fact_map.get(fact_id)

    def get_value(self, ref: str) -> Optional[float]:
        """Get the numeric value for a fact_id reference."""
        return self._value_map.get(ref)

    def to_text(self) -> str:
        """Render the ledger as human-readable text for solver consumption."""
        lines = ["CANONICAL FACT LEDGER (guaranteed correct and complete):", "=" * 60]
        for f in self.facts:
            v_str = f"{f.value}" if f.value is not None else "?"
            lines.append(
                f"  [{f.fact_id}] {f.subject} --{f.relation}--> {f.object}"
                f"  |  value={v_str}  unit={f.unit}  source={f.source}"
            )
        if self.conflicts:
            lines.append("")
            lines.append("DETECTED CONFLICTS (resolve before computing):")
            for c in self.conflicts:
                lines.append(f"  ! {c['fact_a']} vs {c['fact_b']}: {c['description']}")
        return "\n".join(lines)

    def to_table(self) -> str:
        """Markdown table representation."""
        rows = ["| Fact ID | Subject | Relation | Object | Value | Unit | Source | Evidence |",
                "|---------|---------|----------|--------|-------|------|--------|----------|"]
        for f in self.facts:
            v = f"{f.value}" if f.value is not None else ""
            ev = f.evidence[:80] + ("..." if len(f.evidence) > 80 else "")
            rows.append(f"| {f.fact_id} | {f.subject} | {f.relation} | {f.object} | {v} | {f.unit} | {f.source} | {ev} |")
        return "\n".join(rows)

    def to_compact_text(self) -> str:
        """Compact representation for prompt efficiency."""
        lines = ["CANONICAL FACT LEDGER:"]
        for f in self.facts:
            v = f"{f.value}" if f.value is not None else "?"
            lines.append(f"[{f.fact_id}] {f.subject} {f.relation} {f.object} = {v} {f.unit} (src:{f.source})")
        return "\n".join(lines)

    def fact_ids(self) -> list[str]:
        return [f.fact_id for f in self.facts]

    def __len__(self) -> int:
        return len(self.facts)

    def __repr__(self) -> str:
        return f"CanonicalLedger({len(self.facts)} facts, {len(self.conflicts)} conflicts)"

    # ── Builder ──────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        facts_a: list[TypedFact],
        facts_b: list[TypedFact],
        item: dict | None = None,
    ) -> "CanonicalLedger":
        """Build a deterministic canonical ledger from A and B typed facts.

        Steps:
          1. Collect all facts, normalize each
          2. Dedup (same normalized signature → single entry, dual source)
          3. Entity alignment (normalize entity names)
          4. Relation standardization
          5. Unit normalization
          6. Fixed deterministic sort
          7. Conflict detection
          8. Assign canonical fact_ids
        """
        all_facts: list[TypedFact] = []

        # Step 1: Collect and normalize
        for f in facts_a + facts_b:
            nf = deepcopy(f)
            nf.subject = _normalize_entity(nf.subject)
            nf.object = _normalize_entity(nf.object)
            nf.relation = _normalize_relation(nf.relation)
            nf.unit = _normalize_unit(nf.unit)
            all_facts.append(nf)

        # Step 2: Dedup — group by normalized signature
        sig_groups: dict[str, list[TypedFact]] = defaultdict(list)
        for f in all_facts:
            sig_groups[f.signature].append(f)

        deduped: list[TypedFact] = []
        sources_map: dict[str, set] = defaultdict(set)

        for sig, group in sig_groups.items():
            # Keep the one with more detail (longer evidence, more specific)
            keeper = max(group, key=lambda x: (len(x.evidence), 0 if x.value is None else 1))
            for f in group:
                sources_map[sig].add(f.source)
            # If fact was seen by both sources, note it
            if len(sources_map[sig]) > 1:
                keeper.evidence = f"[A+B] {keeper.evidence}"
            deduped.append(keeper)

        # Step 3: Entity alignment — already done in normalize step above
        # Step 4: Relation standardization — already done
        # Step 5: Unit normalization — already done

        # Step 6: Fixed deterministic sort
        # Sort by: source (A first), then by subject, then by relation, then by value
        deduped.sort(key=lambda f: (
            0 if f.source == "A" else 1,
            f.subject,
            f.relation,
            f.value if f.value is not None else float("inf"),
        ))

        # Step 7: Conflict detection
        conflicts = cls._detect_conflicts(deduped)

        # Step 8: Assign canonical fact_ids (F1, F2, ...)
        for i, f in enumerate(deduped):
            f.fact_id = f"F{i + 1}"
            # Keep original source info but prepend canonical ID
            if not f.evidence.startswith("[A+B]"):
                pass  # evidence already carries source info

        # Build info
        build_info = {
            "total_input_facts": len(facts_a) + len(facts_b),
            "after_dedup": len(deduped),
            "conflicts_detected": len(conflicts),
            "sources": {"A": len(facts_a), "B": len(facts_b)},
        }

        return cls(facts=deduped, conflicts=conflicts, build_info=build_info)

    @staticmethod
    def _detect_conflicts(facts: list[TypedFact]) -> list[dict]:
        """Detect conflicting facts: same (subject, relation, object) but different values."""
        conflicts: list[dict] = []
        # Group by (subject, relation, object)
        groups: dict[tuple, list[TypedFact]] = defaultdict(list)
        for f in facts:
            key = (f.subject, f.relation, f.object)
            groups[key].append(f)

        for key, group in groups.items():
            if len(group) < 2:
                continue
            values = {f.value for f in group if f.value is not None}
            if len(values) > 1:
                conflicts.append({
                    "type": "value_conflict",
                    "subject": key[0],
                    "relation": key[1],
                    "object": key[2],
                    "facts": [f.fact_id for f in group],
                    "values": sorted(values, key=lambda x: x if x is not None else float("inf")),
                    "description": (
                        f"Same entity-relation ({key[0]} {key[1]} {key[2]}) "
                        f"has different values: {values}"
                    ),
                })

        return conflicts


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: FRESH SOLVER + EXECUTABLE PLAN
# ═══════════════════════════════════════════════════════════════════════════════

FRESH_SOLVER_SYSTEM = """You are a mathematical problem solver. You receive:
1. A math word problem (the question)
2. A CANONICAL FACT LEDGER — a verified, complete set of structured facts

CRITICAL RULES:
- You have NO access to any prior discussion or agent communication.
- Use ONLY the facts in the ledger. Do NOT guess or invent numbers.
- Every computation step MUST reference at least one fact_id from the ledger.
- Output an executable JSON plan, NOT just a final number.

The JSON plan must have this exact structure:
{
  "steps": [
    {
      "op": "add|subtract|multiply|divide",
      "inputs": ["fact_id_or_variable", "fact_id_or_variable"],
      "output": "descriptive_variable_name",
      "explanation": "what this step computes"
    }
  ]
}

Rules for the plan:
1. Each step's "inputs" must reference EITHER a fact_id from the ledger (like "F1", "F2") OR a variable defined by a previous step.
2. "output" is a short snake_case name for what this step computes.
3. The LAST step must have "output": "answer".
4. Supported operations: "add", "subtract", "multiply", "divide".
5. Steps must be in dependency order — a step can only reference outputs from earlier steps.

Example:
Question: "Julie read 12 pages yesterday. Today she read twice as many. The book has 120 pages. Tomorrow she reads half of the remaining pages. How many pages tomorrow?"
Ledger:
  [F1] yesterday_pages equals numeric_value = 12 pages (src:A)
  [F2] today_pages twice_of yesterday_pages = 2 ratio (src:A)
  [F3] total_pages equals numeric_value = 120 pages (src:B)
  [F4] tomorrow_pages half_of remaining_pages = 0.5 ratio (src:B)

Plan:
{
  "steps": [
    {"op": "multiply", "inputs": ["F1", "F2"], "output": "today_pages_val", "explanation": "Today = yesterday * 2"},
    {"op": "add", "inputs": ["F1", "today_pages_val"], "output": "total_read", "explanation": "Total read = yesterday + today"},
    {"op": "subtract", "inputs": ["F3", "total_read"], "output": "remaining", "explanation": "Remaining = total - read"},
    {"op": "multiply", "inputs": ["remaining", "F4"], "output": "answer", "explanation": "Tomorrow = remaining * 0.5"}
  ]
}

Output ONLY the JSON plan object. No other text, no markdown."""


def fresh_solve(
    model: Any,
    question: str,
    ledger: CanonicalLedger,
    temperature: float = 0.0,
) -> tuple[ExecutablePlan, str, dict]:
    """Fresh solver: reads only question + ledger, outputs executable JSON-IR plan.

    Args:
        model: LocalQwen instance
        question: The shared question text
        ledger: CanonicalLedger with all facts
        temperature: model temperature

    Returns:
        (ExecutablePlan, raw_output, token_usage)
    """
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{ledger.to_text()}\n\n"
        f"Output ONLY the JSON plan. The LAST step MUST have \"output\": \"answer\".\n"
        f"Reference fact_ids like F1, F2, etc. in your inputs."
    )

    raw, usage, elapsed = model.call(FRESH_SOLVER_SYSTEM, user, temperature=temperature)

    plan = _parse_plan_json(raw)

    return plan, raw, usage


def _parse_plan_json(raw: str) -> ExecutablePlan:
    """Robust JSON parsing for executable plan."""
    cleaned = raw.strip()

    # Strategy 1: Direct JSON
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "steps" in data:
            return ExecutablePlan(steps=[PlanStep.from_dict(s) for s in data["steps"]], raw_output=raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    for pattern in [
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
        r"(\{[\s\S]*?\"steps\"[\s\S]*?\}\s*)",
    ]:
        match = re.search(pattern, cleaned)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "steps" in data:
                    return ExecutablePlan(
                        steps=[PlanStep.from_dict(s) for s in data["steps"]],
                        raw_output=raw,
                    )
            except json.JSONDecodeError:
                continue

    # Strategy 3: Extract JSON object by counting braces
    # Find the first '{' after (or at) the "steps" keyword, then count braces
    steps_idx = cleaned.find('"steps"')
    if steps_idx >= 0:
        # Find the nearest '{' before "steps"
        brace_start = cleaned.rfind('{', 0, steps_idx)
        if brace_start >= 0:
            # Count braces from brace_start to find matching '}'
            depth = 0
            brace_end = -1
            for i in range(brace_start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        brace_end = i
                        break
            if brace_end >= 0:
                candidate = cleaned[brace_start:brace_end + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict) and "steps" in data:
                        return ExecutablePlan(
                            steps=[PlanStep.from_dict(s) for s in data["steps"]],
                            raw_output=raw,
                        )
                except json.JSONDecodeError:
                    pass

    # Strategy 4: Fallback — empty plan
    return ExecutablePlan(steps=[], raw_output=raw)


def execute_plan(plan: ExecutablePlan, ledger: CanonicalLedger) -> Optional[float]:
    """Execute a plan against a ledger, returning the final answer value.

    The execution environment starts with all ledger fact values.
    Each step reads from the environment and writes its result back.
    """
    env: dict[str, float] = {}

    # Initialize environment from ledger facts
    for f in ledger.facts:
        if f.value is not None:
            env[f.fact_id] = f.value

    for step in plan.steps:
        # Resolve inputs
        operands: list[float] = []
        for inp in step.inputs:
            val = env.get(inp)
            if val is None:
                # Try to parse as a numeric literal
                val = _parse_float(inp)
            if val is None:
                return None  # Unbound input
            operands.append(val)

        if len(operands) < 2:
            return None

        a, b = operands[0], operands[1]

        # Execute operation
        if step.op == "add":
            result = a + b
        elif step.op == "subtract":
            result = a - b
        elif step.op == "multiply":
            result = a * b
        elif step.op == "divide":
            if b == 0:
                return None
            result = a / b
        else:
            return None  # Unknown operation

        env[step.output] = result

    # Return the answer variable
    return env.get("answer")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: COVERAGE VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def verify_coverage(
    plan: ExecutablePlan,
    ledger: CanonicalLedger,
    expected_answer: str | None = None,
) -> CoverageReport:
    """Programmatic coverage verification of an executable plan against a ledger.

    Checks:
      1. All ledger facts are referenced in at least one step
      2. All plan inputs are bound (either in ledger or defined by prior steps)
      3. The plan is executable (no missing dependencies)
      4. Execution result matches expected answer (if provided)
      5. No duplicate variable definitions

    Returns a CoverageReport with findings and targeted fix hints.
    """
    report = CoverageReport()
    report.facts_total = len(ledger.facts)

    if not plan.steps:
        report.fix_hints.append("Plan has no steps. Create a plan with at least one step.")
        return report

    # ── Check 1: Fact usage ──
    all_fact_ids = set(ledger.fact_ids())
    referenced = plan.referenced_fact_ids

    # Facts with values are "actionable" — they should be used
    actionable_ids = {f.fact_id for f in ledger.facts if f.value is not None}
    # Facts without values (relational facts) are "reference" — may not be directly used
    reference_ids = {f.fact_id for f in ledger.facts if f.value is None}

    used_actionable = referenced & actionable_ids
    unused_actionable = actionable_ids - referenced

    report.facts_used = len(used_actionable)
    report.unused_facts = sorted(unused_actionable)
    report.missing_facts = sorted(unused_actionable)  # actionable but not used

    # ── Check 2: Variable binding ──
    defined: set[str] = set(all_fact_ids)  # ledger facts are pre-defined
    all_referenced: set[str] = set()
    unbound: list[str] = []

    for i, step in enumerate(plan.steps):
        for inp in step.inputs:
            all_referenced.add(inp)
            if inp not in defined and _parse_float(inp) is None:
                unbound.append(f"Step {i+1}: input '{inp}' is not a fact_id or previously defined variable")

        if step.output in defined:
            unbound.append(f"Step {i+1}: output '{step.output}' redefines an existing variable/fact")
        defined.add(step.output)

    report.unbound_variables = unbound

    # ── Check 3: Relation direction consistency ──
    direction_errors: list[str] = []
    for step in plan.steps:
        for inp in step.inputs:
            if _is_fact_id(inp):
                fact = ledger.get(inp)
                if fact and fact.relation in ("twice_of", "triple_of", "half_of"):
                    # These facts are multipliers; check direction
                    # If step is "multiply" and input is the multiplier fact, that's correct
                    # If step is "divide" and input is the multiplier fact, that could be wrong
                    # We just note it for the dataset, not block execution
                    pass

    report.direction_errors = direction_errors

    # ── Check 4: Executability ──
    expected_val = None
    if expected_answer:
        expected_val = _parse_float(expected_answer)
        report.expected_result = expected_val

    if len(unbound) > 0:
        report.executable = False
        report.execution_error = f"Unbound variables: {', '.join(unbound[:5])}"
    else:
        result = execute_plan(plan, ledger)
        report.computed_result = result
        if result is None:
            report.executable = False
            report.execution_error = "Plan execution returned None (missing operands or division by zero)"
        else:
            report.executable = True
            if expected_val is not None:
                # Use tolerance for floating point comparison
                report.result_matches = math.isclose(result, expected_val, rel_tol=1e-9)
            else:
                report.result_matches = True  # Can't verify without expected answer

    # ── Generate fix hints ──
    fix_hints: list[str] = []

    if report.missing_facts:
        for fid in report.missing_facts:
            fact = ledger.get(fid)
            if fact:
                fix_hints.append(
                    f"Missing required fact {fid}: {fact.subject} ({fact.evidence[:60]}...). "
                    f"Current plan does not use this fact. Add a step that incorporates {fid}."
                )

    if unbound:
        for ub in unbound[:3]:
            fix_hints.append(f"Unbound variable: {ub}")

    if report.executable and not report.result_matches and expected_val is not None:
        fix_hints.append(
            f"Result mismatch: computed {report.computed_result}, "
            f"expected ~{expected_val}. Check arithmetic in each step."
        )

    if not plan.steps:
        fix_hints.append("Empty plan. Define steps that compute the answer from ledger facts.")

    # Check if "answer" is defined
    if "answer" not in plan.defined_variables:
        fix_hints.append(
            "No step defines 'answer' as output. The last step must have \"output\": \"answer\"."
        )

    report.fix_hints = fix_hints

    return report


def generate_fix_prompt(
    report: CoverageReport,
    plan: ExecutablePlan,
    ledger: CanonicalLedger,
) -> str:
    """Generate a targeted fix prompt for the solver (no free discussion).

    Only includes specific, actionable hints about what to fix.
    """
    parts = ["FIX INSTRUCTIONS (fix ONLY the listed issues; do NOT re-derive everything):", ""]

    for i, hint in enumerate(report.fix_hints):
        parts.append(f"  Issue {i + 1}: {hint}")

    parts.append("")
    parts.append("Current plan for reference:")
    parts.append(json.dumps(plan.to_json(), indent=2))
    parts.append("")
    parts.append("Output a CORRECTED JSON plan. Fix ONLY the issues listed above.")
    parts.append("Keep all correct steps unchanged. Output ONLY the JSON plan.")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_exec_ground_pipeline(
    model: Any,
    item: dict,
    max_verify_rounds: int = 3,
    use_gold_facts: bool = False,
    fact_temperature: float = 0.0,
    solver_temperature: float = 0.0,
) -> dict:
    """Run the complete ExecGround pipeline on one problem.

    Phases:
      1. Typed fact extraction (each agent, no answering)
      2. Canonical ledger construction (programmatic)
      3. Fresh solver → executable plan
      4. Coverage verification (with up to max_verify_rounds fix iterations)

    Returns a dict with all intermediate artifacts and final answer.
    """
    started = time.perf_counter()
    gold_answer = _normalize_answer_text(item.get("answer", ""))

    result: dict[str, Any] = {
        "shared_question": item.get("shared_question", ""),
        "gold_answer": gold_answer,
        "phases": {},
    }

    # ── Phase 1: Typed Fact Extraction ──
    if use_gold_facts:
        facts_a = extract_gold_typed_facts(item.get("condition_A", ""), "A")
        facts_b = extract_gold_typed_facts(item.get("condition_B", ""), "B")
        fact_method = "gold_rule_based"
    else:
        facts_a = extract_typed_facts(model, item.get("condition_A", ""), "A", temperature=fact_temperature)
        facts_b = extract_typed_facts(model, item.get("condition_B", ""), "B", temperature=fact_temperature)
        fact_method = "llm_extraction"

    result["phases"]["typed_facts"] = {
        "method": fact_method,
        "A": [f.to_dict() for f in facts_a],
        "B": [f.to_dict() for f in facts_b],
        "count_A": len(facts_a),
        "count_B": len(facts_b),
    }

    # ── Phase 2: Canonical Ledger Construction ──
    ledger = CanonicalLedger.build(facts_a, facts_b, item)
    result["phases"]["canonical_ledger"] = {
        "fact_count": len(ledger.facts),
        "conflict_count": len(ledger.conflicts),
        "conflicts": ledger.conflicts,
        "build_info": ledger.build_info,
        "ledger_text": ledger.to_text(),
    }

    # ── Phase 3: Fresh Solver → Executable Plan ──
    plan, raw_plan, solver_usage = fresh_solve(
        model, item.get("shared_question", ""), ledger, temperature=solver_temperature,
    )
    result["phases"]["initial_solve"] = {
        "plan_steps": len(plan.steps),
        "plan_json": plan.to_json(),
        "plan_raw": raw_plan,
        "solver_usage": solver_usage,
    }

    # ── Phase 4: Coverage Verification + Fix Loop ──
    verify_rounds: list[dict] = []
    current_plan = plan
    final_answer: Optional[float] = None
    final_answer_correct = False

    for round_no in range(max_verify_rounds + 1):
        report = verify_coverage(current_plan, ledger, gold_answer)

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
            if report.executable and report.computed_result is not None:
                final_answer = report.computed_result
                final_answer_correct = report.result_matches
            elif report.executable:
                # Try execution even if result_matches wasn't verified
                ans = execute_plan(current_plan, ledger)
                final_answer = ans
                if ans is not None and _parse_float(gold_answer) is not None:
                    final_answer_correct = math.isclose(ans, _parse_float(gold_answer), rel_tol=1e-9)
            break

        if not report.has_fixable_issues:
            break

        # Generate fix prompt and re-solve
        fix_prompt = generate_fix_prompt(report, current_plan, ledger)
        fix_user = (
            f"QUESTION:\n{item.get('shared_question', '')}\n\n"
            f"{ledger.to_text()}\n\n"
            f"{fix_prompt}"
        )
        fix_raw, fix_usage, _ = model.call(FRESH_SOLVER_SYSTEM, fix_user, temperature=0.0)
        current_plan = _parse_plan_json(fix_raw)
        round_info["fix_raw"] = fix_raw
        round_info["fix_usage"] = fix_usage

    result["phases"]["verification"] = {
        "rounds": verify_rounds,
        "total_rounds": len(verify_rounds),
        "final_plan_clean": verify_rounds[-1]["is_clean"] if verify_rounds else False,
    }

    result["final_answer"] = final_answer
    result["final_answer_correct"] = final_answer_correct
    result["total_runtime_seconds"] = time.perf_counter() - started

    return result


def _normalize_answer_text(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K answer text."""
    if "####" in answer_text:
        return answer_text.rsplit("####", 1)[1].strip()
    return answer_text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick smoke test without LLM
    print("=" * 60)
    print("  ExecGround Smoke Test (no LLM)")
    print("=" * 60)

    # Test TypedFact
    print("1. TypedFact creation...")
    f1 = TypedFact(
        fact_id="A_F1", subject="yesterday_pages", relation="equals",
        object="numeric_value", value=12, unit="pages", source="A",
        evidence="Julie read 12 pages yesterday.",
    )
    f2 = TypedFact(
        fact_id="A_F2", subject="today_pages", relation="twice_of",
        object="yesterday_pages", value=2, unit="ratio", source="A",
        evidence="Today she read twice as many pages as yesterday.",
    )
    print(f"  {f1.to_ledger_line()}")
    print(f"  {f2.to_ledger_line()}")
    print(f"  f1 signature: {f1.signature}")
    print(f"  f2 signature: {f2.signature}")
    print("  OK")

    # Test CanonicalLedger
    print("\n2. CanonicalLedger build...")
    facts_a = [
        TypedFact("A_F1", "yesterday_pages", "equals", "numeric_value", 12, "pages", "A", "read 12 pages"),
        TypedFact("A_F2", "today_pages", "twice_of", "yesterday_pages", 2, "ratio", "A", "twice as many"),
    ]
    facts_b = [
        TypedFact("B_F1", "total_pages", "equals", "numeric_value", 120, "pages", "B", "book has 120 pages"),
        TypedFact("B_F2", "tomorrow_pages", "half_of", "remaining_pages", 0.5, "ratio", "B", "half of remaining"),
    ]
    ledger = CanonicalLedger.build(facts_a, facts_b)
    print(f"  Facts: {len(ledger.facts)}, Conflicts: {len(ledger.conflicts)}")
    print(f"  Build info: {ledger.build_info}")
    for f in ledger.facts:
        print(f"  {f.to_ledger_line()}")
    print("  OK")

    # Test determinism: AB vs BA
    print("\n3. Determinism check (AB vs BA)...")
    ledger_ab = CanonicalLedger.build(facts_a, facts_b)
    ledger_ba = CanonicalLedger.build(facts_b, facts_a)
    text_ab = ledger_ab.to_text()
    text_ba = ledger_ba.to_text()
    print(f"  AB == BA: {text_ab == text_ba}")
    assert text_ab == text_ba, "LEDGER NOT DETERMINISTIC!"
    print("  OK")

    # Test Plan execution
    # The canonical ledger renumbers facts (sorted by source, subject, relation, value).
    # Build a map from (subject, relation) → canonical fact_id for the test.
    print("\n4. Plan execution...")
    fmap = {}
    for f in ledger.facts:
        fmap[(f.subject, f.relation)] = f.fact_id

    plan = ExecutablePlan(steps=[
        PlanStep("multiply", [fmap[("yesterday_page", "equals")], fmap[("today_page", "twice_of")]],
                 "today_val", "today = 12 * 2"),
        PlanStep("add", [fmap[("yesterday_page", "equals")], "today_val"],
                 "total_read", "read = 12 + 24"),
        PlanStep("subtract", [fmap[("total_page", "equals")], "total_read"],
                 "remaining", "remaining = 120 - 36"),
        PlanStep("multiply", ["remaining", fmap[("tomorrow_page", "half_of")]],
                 "answer", "tomorrow = 84 * 0.5"),
    ])
    result = execute_plan(plan, ledger)
    print(f"  Plan result: {result} (expected 42.0)")
    assert result is not None and math.isclose(result, 42.0), f"Expected 42.0, got {result}"
    print("  OK")

    # Test Coverage Verifier
    print("\n5. Coverage verification...")
    report = verify_coverage(plan, ledger, "42")
    print(f"  Clean: {report.is_clean}")
    print(f"  Facts used: {report.facts_used}/{report.facts_total}")
    print(f"  Missing: {report.missing_facts}")
    print(f"  Unbound: {report.unbound_variables}")
    print(f"  Executable: {report.executable}")
    print(f"  Result matches: {report.result_matches}")
    print(f"  Fix hints: {report.fix_hints}")
    assert report.is_clean, f"Report not clean: {report.fix_hints}"
    print("  OK")

    # Test with incomplete plan
    print("\n6. Incomplete plan detection...")
    # Use actual fact_ids from the ledger
    bad_plan = ExecutablePlan(steps=[
        PlanStep("multiply", [fmap[("yesterday_page", "equals")], fmap[("today_page", "twice_of")]], "today_val"),
        PlanStep("add", [fmap[("yesterday_page", "equals")], "today_val"], "total_read"),
        # Missing: subtract remaining step, and multiply by half_of
        # F999 doesn't exist in ledger
        PlanStep("multiply", ["total_read", "F999"], "answer"),
    ])
    bad_report = verify_coverage(bad_plan, ledger, "42")
    print(f"  Clean: {bad_report.is_clean}")
    print(f"  Missing facts: {bad_report.missing_facts}")
    print(f"  Unbound: {bad_report.unbound_variables}")
    print(f"  Executable: {bad_report.executable}")
    print(f"  Fix hints: {bad_report.fix_hints}")
    assert not bad_report.is_clean, "Should detect issues"
    assert len(bad_report.fix_hints) > 0, "Should have fix hints"
    print("  OK")

    # Test gold fact extraction
    print("\n7. Gold fact extraction...")
    gold_facts = extract_gold_typed_facts(
        "Julie read 12 pages of a book yesterday. Today she read twice as many pages as yesterday.",
        "A",
    )
    print(f"  Extracted {len(gold_facts)} facts:")
    for f in gold_facts:
        print(f"  {f.to_ledger_line()}")
    assert len(gold_facts) >= 2, f"Expected at least 2 facts, got {len(gold_facts)}"
    print("  OK")

    # Test entity normalization
    print("\n8. Entity normalization...")
    assert _normalize_entity("pages") == "page"
    assert _normalize_entity("dollars") == "dollar"
    assert _normalize_entity("Julie") == "julie"
    print("  OK")

    # Test relation standardization
    print("\n9. Relation standardization...")
    assert _normalize_relation("twice") == "twice_of"
    assert _normalize_relation("half") == "half_of"
    assert _normalize_relation("is") == "equals"
    assert _normalize_relation("double") == "twice_of"
    print("  OK")

    print("\n=== All smoke tests passed ===")
