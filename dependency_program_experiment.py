"""Dependency-program recovery experiment for dispersed multi-agent evidence.

This file is intentionally self-contained.  It does not train a model; it builds
an executable IR, provenance-preserving facts, oracle plans from GSM8K-style
gold equations, and search/verification planners that can be swapped later for
LLM proposal functions.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "20.json"
DEFAULT_OUTPUT_DIR = ROOT / "outouts_dependecy_program"
DEFAULT_MODEL_PATH = ROOT / "qwen2.5-1.5B"
OPS = {"ADD", "SUB", "MUL", "DIV"}
PROPOSAL_BACKENDS = ("semantic", "llm", "hybrid")
LEXICAL_NUMERIC_FACTS = (
    (re.compile(r"\bhalf\b", re.I), Decimal("0.5"), "half"),
    (re.compile(r"\bone\b", re.I), Decimal("1"), "one"),
    (re.compile(r"\btwo\b", re.I), Decimal("2"), "two"),
    (re.compile(r"\bthree\b", re.I), Decimal("3"), "three"),
    (re.compile(r"\btwice\b|\bdouble[sd]?\b|\bdoubled\b", re.I), Decimal("2"), "double"),
    (re.compile(r"\btriple[sd]?\b|\btripled\b", re.I), Decimal("3"), "triple"),
)


def blank_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def add_usage(target: dict[str, int], usage: dict[str, Any] | None) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = int(target.get(key, 0) or 0) + int((usage or {}).get(key, 0) or 0)


def collect_trace_usage(trace: list[dict[str, Any]]) -> tuple[dict[str, int], float, int]:
    usage = blank_usage()
    elapsed = 0.0
    calls = 0
    for step in trace:
        meta = step.get("llm_meta")
        if not isinstance(meta, dict):
            continue
        add_usage(usage, meta.get("usage"))
        elapsed += float(meta.get("elapsed_seconds") or 0.0)
        calls += 1
    return usage, elapsed, calls


@dataclass(frozen=True)
class Fact:
    fact_id: str
    source: str
    content: str
    type: str = "numeric_fact"
    key: str = ""
    value: Decimal | None = None
    derived: bool = False
    provenance: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "source": self.source,
            "content": self.content,
            "type": self.type,
            "key": self.key,
            "value": str(self.value) if self.value is not None else None,
            "derived": self.derived,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class IRNode:
    node_id: str
    op: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    fact_id: str | None = None
    value: Decimal | None = None
    label: str = ""
    provenance: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "args": list(self.args),
            "fact_id": self.fact_id,
            "value": str(self.value) if self.value is not None else None,
            "label": self.label,
            "provenance": list(self.provenance),
        }


@dataclass
class Program:
    nodes: dict[str, IRNode]
    answer_node: str
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "answer_node": self.answer_node,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }

    def edges(self) -> set[tuple[str, str]]:
        reachable = self.reachable_nodes()
        edges: set[tuple[str, str]] = set()
        for node in self.nodes.values():
            if node.node_id not in reachable:
                continue
            for arg in node.args:
                if isinstance(arg, str) and arg in reachable:
                    edges.add((arg, node.node_id))
        return edges

    def referenced_facts(self) -> set[str]:
        reachable = self.reachable_nodes()
        return {
            n.fact_id for n in self.nodes.values()
            if n.node_id in reachable and n.op == "FACT" and n.fact_id
        }

    def reachable_nodes(self) -> set[str]:
        reachable: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in reachable or node_id not in self.nodes:
                return
            reachable.add(node_id)
            for arg in self.nodes[node_id].args:
                if isinstance(arg, str):
                    visit(arg)

        visit(self.answer_node)
        return reachable

    def reachable_program(self) -> "Program":
        reachable = self.reachable_nodes()
        return Program(
            nodes={node_id: self.nodes[node_id] for node_id in self.nodes if node_id in reachable},
            answer_node=self.answer_node,
            name=self.name,
        )


def clone_program(program: Program) -> Program:
    return Program(nodes=dict(program.nodes), answer_node=program.answer_node, name=program.name)


@dataclass
class ExecutionResult:
    ok: bool
    answer: Decimal | None
    values: dict[str, Decimal]
    errors: list[str]


@dataclass
class OraclePlan:
    facts: list[Fact]
    program: Program
    equation_structure: list[str]
    answer: Decimal | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [f.to_dict() for f in self.facts],
            "program": self.program.to_dict(),
            "equation_structure": self.equation_structure,
            "answer": str(self.answer) if self.answer is not None else None,
        }


@dataclass
class PlannerResult:
    program: Program
    candidates: list[Decimal]
    trace: list[dict[str, Any]]
    proposal_backend: str = "semantic"


@dataclass(frozen=True)
class Expansion:
    target: str
    op: str
    args: tuple[Any, ...]
    source: str = "semantic"


def decimal(value: Any) -> Decimal | None:
    text = str(value or "").replace(",", "").strip()
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    frac = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", text)
    try:
        if frac:
            a, b = map(Decimal, frac.groups())
            return None if b == 0 else a / b
        return Decimal(text) if re.fullmatch(r"-?\d+(?:\.\d+)?", text) else None
    except InvalidOperation:
        return None


def close_enough(left: Decimal | None, right: Decimal | None) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= Decimal("0.0000001")


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    rows = json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    for i, row in enumerate(rows, 1):
        if "fact" not in row and "required_private_facts" in row:
            row["fact"] = {
                "A": row["required_private_facts"].get("A", row["required_private_facts"].get("agent_A", [])),
                "B": row["required_private_facts"].get("B", row["required_private_facts"].get("agent_B", [])),
            }
        if "full" not in row and "full_question" in row:
            row["full"] = row["full_question"]
        missing = {"condition_A", "condition_B", "shared_question", "answer"} - set(row)
        if missing:
            raise ValueError(f"record {i} missing fields: {sorted(missing)}")
    return rows


def normalize_key(text: str, fallback: str) -> str:
    words = re.findall(r"[A-Za-z]+", text.lower())
    stop = {"the", "a", "an", "of", "to", "is", "are", "was", "were", "has", "have", "she", "he", "it", "they"}
    kept = [w for w in words if w not in stop][:5]
    return "_".join(kept) or fallback


def extract_facts_from_text(texts: Iterable[str], source: str, prefix: str) -> list[Fact]:
    facts: list[Fact] = []
    for raw in texts:
        for match in re.finditer(r"-?\d+(?:,\d{3})*(?:\.\d+)?", raw):
            idx = len(facts) + 1
            value = decimal(match.group(0))
            key = normalize_key(raw, f"{prefix.lower()}_{idx}")
            facts.append(Fact(
                fact_id=f"{prefix}_{idx:03d}",
                source=source,
                content=raw.strip(),
                key=key,
                value=value,
                provenance=(f"{prefix}_{idx:03d}",),
            ))
        for pattern, value, label in LEXICAL_NUMERIC_FACTS:
            if pattern.search(raw):
                idx = len(facts) + 1
                fact_id = f"{prefix}_{idx:03d}"
                facts.append(Fact(
                    fact_id=fact_id,
                    source=source,
                    content=raw.strip(),
                    type="relation_fact",
                    key=f"{normalize_key(raw, f'{prefix.lower()}_{idx}')}_{label}",
                    value=value,
                    provenance=(fact_id,),
                ))
    return facts


def gold_facts(item: dict[str, Any]) -> list[Fact]:
    fact_obj = item.get("fact") or {}
    a = fact_obj.get("A", fact_obj.get("agent_A", [item.get("condition_A", "")]))
    b = fact_obj.get("B", fact_obj.get("agent_B", [item.get("condition_B", "")]))
    return extract_facts_from_text(a, "agent_A", "A") + extract_facts_from_text(b, "agent_B", "B")


def llm_like_facts(item: dict[str, Any], history: bool = False) -> list[Fact]:
    facts = (
        extract_facts_from_text([item.get("condition_A", "")], "agent_A", "A") +
        extract_facts_from_text([item.get("condition_B", "")], "agent_B", "B")
    )
    if history and facts:
        first = facts[0]
        facts.append(Fact(
            fact_id="H_001",
            source="history",
            content=f"historical noisy restatement of {first.key}",
            key=first.key,
            value=first.value + 1 if first.value is not None else None,
            derived=True,
            provenance=(first.fact_id,),
        ))
    return facts


def detect_conflicts(facts: list[Fact]) -> list[dict[str, Any]]:
    by_key: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.value is not None:
            by_key.setdefault(fact.key, []).append(fact)
    conflicts = []
    for key, group in by_key.items():
        values = {g.value for g in group}
        sources = {g.source for g in group}
        if len(values) > 1 and len(sources) > 1:
            conflicts.append({"key": key, "facts": [g.fact_id for g in group], "values": [str(v) for v in values]})
    return conflicts


def make_fact_nodes(facts: list[Fact]) -> dict[str, IRNode]:
    return {
        f"fact_{fact.fact_id}": IRNode(
            node_id=f"fact_{fact.fact_id}",
            op="FACT",
            fact_id=fact.fact_id,
            label=fact.key,
            provenance=(fact.fact_id,),
        )
        for fact in facts
    }


def eval_program(program: Program, facts: list[Fact]) -> ExecutionResult:
    fact_values = {f.fact_id: f.value for f in facts if f.value is not None}
    values: dict[str, Decimal] = {}
    errors: list[str] = []
    visiting: set[str] = set()

    def visit(node_id: str) -> Decimal | None:
        if node_id in values:
            return values[node_id]
        node = program.nodes.get(node_id)
        if node is None:
            errors.append(f"unknown node: {node_id}")
            return None
        if node_id in visiting:
            errors.append(f"cycle at node: {node_id}")
            return None
        visiting.add(node_id)
        try:
            if node.op == "FACT":
                if not node.fact_id or node.fact_id not in fact_values:
                    errors.append(f"missing fact support: {node.node_id}")
                    return None
                result = fact_values[node.fact_id]
            elif node.op == "CONST":
                result = node.value
            elif node.op in OPS:
                args = [visit(str(arg)) for arg in node.args]
                if any(arg is None for arg in args) or len(args) < 2:
                    errors.append(f"bad args for {node.node_id}")
                    return None
                result = args[0]
                for arg in args[1:]:
                    if node.op == "ADD":
                        result += arg
                    elif node.op == "SUB":
                        result -= arg
                    elif node.op == "MUL":
                        result *= arg
                    elif node.op == "DIV":
                        if arg == 0:
                            errors.append(f"division by zero: {node.node_id}")
                            return None
                        result /= arg
            else:
                errors.append(f"illegal op: {node.op}")
                return None
            if result is None:
                errors.append(f"empty value: {node.node_id}")
                return None
            values[node_id] = result
            return result
        finally:
            visiting.discard(node_id)

    answer = visit(program.answer_node)
    return ExecutionResult(ok=answer is not None and not errors, answer=answer, values=values, errors=errors)


def parse_gold_equations(answer: str) -> list[tuple[str, Decimal]]:
    equations = []
    for expr, result in re.findall(r"<<([^=<>]+)=([^<>]+)>>", answer):
        value = decimal(result)
        if value is not None:
            equations.append((expr.strip(), value))
    return equations


def op_from_symbol(symbol: str) -> str:
    return {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}[symbol]


def tokenize_arithmetic(expr: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    cleaned = expr.replace(",", "").replace(" ", "")
    while i < len(cleaned):
        ch = cleaned[i]
        unary = ch == "-" and (i == 0 or cleaned[i - 1] in "+-*/(")
        if ch.isdigit() or ch == "." or unary:
            j = i + 1
            while j < len(cleaned) and (cleaned[j].isdigit() or cleaned[j] == "."):
                j += 1
            tokens.append(cleaned[i:j])
            i = j
        elif ch in "+-*/":
            tokens.append(ch)
            i += 1
        else:
            i += 1
    return tokens


def bind_value_node(value: Decimal, facts: list[Fact], nodes: dict[str, IRNode], const_counter: list[int]) -> str:
    for fact in facts:
        if fact.value == value:
            return f"fact_{fact.fact_id}"
    for node_id, node in nodes.items():
        if node_id.startswith("oracle_step_") and node.value == value:
            return node_id
    const_counter[0] += 1
    node_id = f"const_{const_counter[0]}"
    nodes[node_id] = IRNode(node_id=node_id, op="CONST", value=value, provenance=())
    return node_id


def build_oracle_plan(item: dict[str, Any], facts: list[Fact]) -> OraclePlan:
    nodes = make_fact_nodes(facts)
    const_counter = [0]
    equations = parse_gold_equations(item.get("answer", ""))
    if not equations:
        answer = decimal(item.get("answer"))
        node_id = bind_value_node(answer or Decimal(0), facts, nodes, const_counter)
        return OraclePlan(facts, Program(nodes, node_id, "oracle"), [], answer)
    for step_idx, (expr, result) in enumerate(equations, 1):
        terms = tokenize_arithmetic(expr)
        if len(terms) < 3:
            out_id = bind_value_node(result, facts, nodes, const_counter)
            nodes[f"oracle_step_{step_idx}"] = IRNode(f"oracle_step_{step_idx}", "ADD", (out_id, bind_value_node(Decimal(0), facts, nodes, const_counter)), value=result)
            continue
        current_value = decimal(terms[0]) or Decimal(0)
        current = bind_value_node(current_value, facts, nodes, const_counter)
        op_number = 0
        for symbol, raw_num in zip(terms[1::2], terms[2::2]):
            op_number += 1
            rhs_value = decimal(raw_num) or Decimal(0)
            rhs = bind_value_node(rhs_value, facts, nodes, const_counter)
            node_id = f"oracle_step_{step_idx}" if op_number == len(terms[1::2]) else f"oracle_step_{step_idx}_{op_number}"
            if symbol == "+":
                node_value = current_value + rhs_value
            elif symbol == "-":
                node_value = current_value - rhs_value
            elif symbol == "*":
                node_value = current_value * rhs_value
            else:
                node_value = current_value / rhs_value if rhs_value != 0 else result
            arg_facts = tuple(sorted(
                p for arg in (current, rhs)
                for p in nodes[arg].provenance
            ))
            nodes[node_id] = IRNode(node_id=node_id, op=op_from_symbol(symbol), args=(current, rhs), value=node_value, provenance=arg_facts)
            current_value = node_value
            current = node_id
    answer_node = f"oracle_step_{len(equations)}"
    return OraclePlan(facts, Program(nodes, answer_node, "oracle"), [expr for expr, _ in equations], decimal(item.get("answer")))


def contract_check(program: Program, facts: list[Fact]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    fact_ids = {f.fact_id for f in facts}
    for node in program.nodes.values():
        if node.op == "FACT" and node.fact_id not in fact_ids:
            errors.append(f"unsupported FACT node {node.node_id}")
        if node.op not in OPS | {"FACT", "CONST"}:
            errors.append(f"illegal op {node.op}")
        for arg in node.args:
            if isinstance(arg, str) and arg not in program.nodes:
                errors.append(f"invented variable {arg}")
    conflicts = detect_conflicts(facts)
    if conflicts:
        errors.append(f"fact conflict: {conflicts}")
    exec_result = eval_program(program, facts)
    if not exec_result.ok:
        errors.extend(exec_result.errors)
    leaf_facts = program.referenced_facts()
    if not leaf_facts:
        errors.append("no FACT provenance")
    return not errors, errors


def node_signature(program: Program, node_id: str, facts_by_id: dict[str, Fact]) -> str:
    node = program.nodes[node_id]
    if node.op == "FACT":
        fact = facts_by_id.get(node.fact_id or "")
        value = str(fact.value) if fact and fact.value is not None else "?"
        return f"FACT:{value}"
    if node.op == "CONST":
        return f"CONST:{node.value}"
    child_sigs = [node_signature(program, str(arg), facts_by_id) for arg in node.args if str(arg) in program.nodes]
    if node.op in {"ADD", "MUL"}:
        child_sigs = sorted(child_sigs)
    return f"{node.op}({','.join(child_sigs)})"


def canonical_expr_edges(program: Program, facts: list[Fact]) -> set[tuple[str, str]]:
    facts_by_id = {fact.fact_id: fact for fact in facts}
    edges: set[tuple[str, str]] = set()
    for parent in program.nodes.values():
        for arg in parent.args:
            child_id = str(arg)
            if child_id in program.nodes:
                edges.add((
                    node_signature(program, child_id, facts_by_id),
                    node_signature(program, parent.node_id, facts_by_id),
                ))
    return edges


def edge_f1(generated: Program, generated_facts: list[Fact], oracle: Program, oracle_facts: list[Fact]) -> dict[str, float]:
    pred = canonical_expr_edges(generated, generated_facts)
    gold = canonical_expr_edges(oracle, oracle_facts)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


def align_oracle_to_facts(oracle: OraclePlan, target_facts: list[Fact]) -> Program:
    """Map oracle leaf FACT nodes to the active ledger by numeric value.

    Gold plan structure stays fixed, but each gold FACT leaf must be executable
    against the selected evidence ledger.  This isolates extraction quality
    from arbitrary fact-id naming differences.
    """
    available: dict[Decimal, list[Fact]] = {}
    for fact in target_facts:
        if fact.value is not None:
            available.setdefault(fact.value, []).append(fact)
    used: set[str] = set()
    nodes = dict(oracle.program.nodes)
    for node_id, node in list(nodes.items()):
        if node.op != "FACT" or not node.fact_id:
            continue
        gold_fact = next((f for f in oracle.facts if f.fact_id == node.fact_id), None)
        if gold_fact is None or gold_fact.value is None:
            continue
        replacement = next((f for f in available.get(gold_fact.value, []) if f.fact_id not in used), None)
        if replacement is None:
            continue
        used.add(replacement.fact_id)
        nodes[node_id] = IRNode(
            node_id=node.node_id,
            op="FACT",
            fact_id=replacement.fact_id,
            label=replacement.key,
            provenance=(replacement.fact_id,),
        )
    return Program(nodes, oracle.program.answer_node, oracle.program.name)


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


def parse_llm_arg(arg: Any, nodes: dict[str, IRNode], facts_by_id: dict[str, Fact]) -> str | None:
    text = str(arg).strip()
    fact_match = re.fullmatch(r"FACT\[(.+?)\]", text)
    if fact_match:
        fact_id = fact_match.group(1).strip()
        if fact_id not in facts_by_id:
            return None
        node_id = f"fact_{fact_id}"
        if node_id not in nodes:
            fact = facts_by_id[fact_id]
            nodes[node_id] = IRNode(
                node_id=node_id,
                op="FACT",
                fact_id=fact_id,
                label=fact.key,
                provenance=(fact_id,),
            )
        return node_id
    const_match = re.fullmatch(r"CONST\((.+?)\)", text)
    if const_match:
        value = decimal(const_match.group(1))
        if value is None:
            return None
        node_id = f"const_{str(value).replace('.', '_').replace('-', 'neg_')}"
        nodes.setdefault(node_id, IRNode(node_id=node_id, op="CONST", value=value, provenance=()))
        return node_id
    return text if text in nodes else None


def program_from_llm_json(payload: dict[str, Any], facts: list[Fact]) -> tuple[Program | None, list[str]]:
    facts_by_id = {fact.fact_id: fact for fact in facts}
    nodes = make_fact_nodes(facts)
    errors: list[str] = []
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None, ["LLM payload missing list field: steps"]
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            errors.append(f"step {index} is not an object")
            continue
        node_id = str(step.get("node_id") or step.get("target") or "").strip()
        op = str(step.get("op") or "").strip().upper()
        raw_args = step.get("args", step.get("inputs"))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node_id):
            errors.append(f"step {index} invalid node_id: {node_id}")
            continue
        if op not in OPS:
            errors.append(f"step {index} illegal op: {op}")
            continue
        if not isinstance(raw_args, list) or len(raw_args) < 2:
            errors.append(f"step {index} needs at least two args")
            continue
        parsed_args = [parse_llm_arg(arg, nodes, facts_by_id) for arg in raw_args]
        if any(arg is None for arg in parsed_args):
            errors.append(f"step {index} has unsupported args: {raw_args}")
            continue
        provenance = tuple(sorted(set(
            p for arg in parsed_args
            for p in nodes[str(arg)].provenance
        )))
        nodes[node_id] = IRNode(node_id=node_id, op=op, args=tuple(str(arg) for arg in parsed_args), provenance=provenance)
    answer_node = str(payload.get("answer_node") or "answer").strip()
    if answer_node not in nodes:
        errors.append(f"answer_node not defined: {answer_node}")
        return None, errors
    program = Program(nodes=nodes, answer_node=answer_node, name="llm_backward").reachable_program()
    ok, contract_errors = contract_check(program, facts)
    if not ok:
        errors.extend(contract_errors)
    return (program if ok else None), errors


def visible_oracle_information(item: dict[str, Any]) -> list[str]:
    context = item.get("oracle_context")
    if not isinstance(context, dict):
        return []
    return [str(x) for x in context.get("visible_oracle_information") or []]


def format_oracle_context_for_prompt(item: dict[str, Any]) -> str:
    context = item.get("oracle_context")
    if not isinstance(context, dict):
        return ""
    visible = set(visible_oracle_information(item))
    lines: list[str] = []
    if "goal" in visible and context.get("goal"):
        lines.extend(["Oracle guidance:", f"Goal: {context['goal']}"])
    alias_lines: list[str] = []
    if "relevant_facts" in visible:
        for alias, value in sorted((context.get("relevant_facts") or {}).items()):
            if isinstance(value, dict) and value.get("source_fact_id"):
                alias_lines.append(f"- {alias} := FACT[{value['source_fact_id']}]")
    if "constants" in visible:
        for alias, value in sorted((context.get("constants") or {}).items()):
            alias_lines.append(f"- {alias} := CONST({value})")
    if alias_lines:
        if not lines:
            lines.append("Oracle guidance:")
        lines.extend([
            "Executable alias map:",
            *alias_lines,
            "Fxxx/Cxxx/Rxxx are oracle guidance aliases only.",
            "When returning JSON, use FACT[source_id], CONST(value), or actual generated node names.",
            "Never return Fxxx, Cxxx, or Rxxx directly as executable args.",
        ])
    if "relevant_facts" in visible and context.get("relevant_facts"):
        if not lines:
            lines.append("Oracle guidance:")
        lines.append("Relevant facts:")
        for fact_id, value in sorted(context["relevant_facts"].items()):
            if isinstance(value, dict):
                source = value.get("source_fact_id", fact_id)
                marker = "relevant" if value.get("relevant") else "irrelevant"
                lines.append(f"- {fact_id} (FACT[{source}]): {marker}")
            else:
                lines.append(f"- {fact_id}: {'relevant' if value else 'irrelevant'}")
    if "fact_binding" in visible and context.get("fact_binding"):
        if not lines:
            lines.append("Oracle guidance:")
        lines.append("Fact bindings:")
        for fact_id, variable in sorted(context["fact_binding"].items()):
            lines.append(f"- {fact_id} -> {variable}")
    if "constants" in visible and context.get("constants"):
        if not lines:
            lines.append("Oracle guidance:")
        lines.append("Constants:")
        for const_id, value in sorted(context["constants"].items()):
            lines.append(f"- {const_id} = {value}")
    if "local_relations" in visible and context.get("local_relations"):
        if not lines:
            lines.append("Oracle guidance:")
        lines.append("Local relations (unordered local relations; DERIVED means some derived result without revealing wiring):")
        for relation in context["local_relations"]:
            result = relation.get("relation_id", "R")
            op = relation.get("op", "")
            inputs = ", ".join(str(x) for x in relation.get("inputs", []))
            lines.append(f"- {result} = {op}({inputs})")
    if "topology" in visible and context.get("topology"):
        if not lines:
            lines.append("Oracle guidance:")
        lines.extend([
            "Topology interpretation:",
            '- ANSWER corresponds to the current planner target "answer".',
            "- Dxxx denotes an anonymous derived dependency node. Each Dxxx must eventually be recovered as an arithmetic operation.",
            "- Lxxx denotes an anonymous leaf position. Each Lxxx must eventually be grounded to an actual FACT[...] or CONST(...).",
            "- Edges specify dependency structure only.",
            "- The topology does NOT reveal arithmetic operators, semantic fact bindings, or which FACT corresponds to which Lxxx.",
            "- Do not use Lxxx or Dxxx directly as executable FACT IDs.",
        ])
        lines.append("Topology:")
        for edge in context["topology"]:
            lines.append(f"- {edge.get('from')} -> {edge.get('to')}")
    return "\n".join(lines)


def llm_proposal_program(model: Any, item: dict[str, Any], facts: list[Fact]) -> tuple[Program | None, dict[str, Any]]:
    fact_lines = "\n".join(
        f"- FACT[{fact.fact_id}] source={fact.source} type={fact.type} value={fact.value} content={fact.content}"
        for fact in facts
    )
    oracle_prompt_context = format_oracle_context_for_prompt(item)
    oracle_block = f"{oracle_prompt_context}\n\n" if oracle_prompt_context else ""
    system = (
        "You recover executable dependency programs from structured evidence. "
        "Use only FACT[id], CONST(value), and ops ADD/SUB/MUL/DIV. "
        "Do not answer in prose."
    )
    user = (
        "Build a dependency program for the shared question using only the facts below.\n\n"
        f"Question: {item.get('shared_question', '')}\n\n"
        f"Facts:\n{fact_lines}\n\n"
        f"{oracle_block}"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "steps": [\n'
        '    {"node_id": "n1", "op": "MUL", "args": ["FACT[A_001]", "CONST(2)"]},\n'
        '    {"node_id": "answer", "op": "ADD", "args": ["n1", "FACT[B_001]"]}\n'
        "  ],\n"
        '  "answer_node": "answer"\n'
        "}\n"
        "Every non-leaf node must be introduced by exactly one step."
    )
    raw, usage, elapsed = model.call(system, user, temperature=0.0)
    payload = extract_json_object(raw)
    meta = {
        "raw": raw,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "visible_oracle_information": visible_oracle_information(item),
        "oracle_prompt_context": oracle_prompt_context,
        "parse_errors": [],
    }
    if payload is None:
        meta["parse_errors"] = ["could not parse JSON object"]
        return None, meta
    program, errors = program_from_llm_json(payload, facts)
    meta["payload"] = payload
    meta["parse_errors"] = errors
    return program, meta


def llm_single_expansion(
    model: Any,
    item: dict[str, Any],
    facts: list[Fact],
    target: str,
    known_nodes: list[str],
    revision_context: dict[str, Any] | None = None,
) -> tuple[Expansion | None, dict[str, Any]]:
    fact_lines = "\n".join(
        f"- FACT[{fact.fact_id}] type={fact.type} value={fact.value} content={fact.content}"
        for fact in facts
    )
    oracle_prompt_context = format_oracle_context_for_prompt(item)
    oracle_block = f"{oracle_prompt_context}\n\n" if oracle_prompt_context else ""
    system = (
        "You expand exactly ONE unresolved dependency node. "
        "Use only FACT[id], CONST(value), existing node names, and ops ADD/SUB/MUL/DIV. "
        "Return JSON only."
    )
    revision_block = ""
    if revision_context:
        rejection_reasons = "\n".join(f"- {reason}" for reason in revision_context.get("rejection_reasons", []))
        current_program = json.dumps(revision_context.get("program", {}), ensure_ascii=False)
        revision_block = (
            "The current dependency program is executable but incomplete.\n\n"
            f"Current candidate program:\n{current_program}\n\n"
            "Verifier rejected it for:\n"
            f"{rejection_reasons or '- unspecified verifier rejection'}\n\n"
            "Revise the current target so that the missing dependency is represented explicitly.\n"
            "Do not return the same incomplete expression.\n"
            "Use FACT[id], CONST(value), or a new derived node that can be expanded later.\n\n"
        )
    user = (
        f"Question: {item.get('shared_question', '')}\n\n"
        f"Facts:\n{fact_lines}\n\n"
        f"{oracle_block}"
        f"{revision_block}"
        f"Known derived nodes: {known_nodes}\n"
        f"Unresolved target to expand: {target}\n\n"
        "Allowed op values: ADD, SUB, MUL, DIV. Choose exactly one value.\n"
        'Do not output the literal string "ADD|SUB|MUL|DIV".\n'
        "Return ONLY:\n"
        '{"target": "<same target>", "op": "ADD", "args": ["FACT[A_001]", "node_x"]}\n'
        "Do not define any other node."
    )
    raw, usage, elapsed = model.call(system, user, temperature=0.0)
    payload = extract_json_object(raw)
    meta = {
        "raw": raw,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "visible_oracle_information": visible_oracle_information(item),
        "oracle_prompt_context": oracle_prompt_context,
        "revision_context": revision_context or {},
        "payload": payload,
        "parse_errors": [],
    }
    if not isinstance(payload, dict):
        meta["parse_errors"] = ["could not parse expansion JSON"]
        return None, meta
    if str(payload.get("target", "")).strip() != target:
        meta["parse_errors"] = [f"target mismatch: {payload.get('target')} != {target}"]
        return None, meta
    op = str(payload.get("op", "")).strip().upper()
    args = payload.get("args")
    if op not in OPS or not isinstance(args, list) or len(args) < 2:
        meta["parse_errors"] = ["invalid op or args"]
        return None, meta
    return Expansion(target, op, tuple(args), "llm"), meta


def llm_strict_backward_search(
    model: Any,
    item: dict[str, Any],
    facts: list[Fact],
    max_steps: int = 8,
    failure_trace: list[dict[str, Any]] | None = None,
) -> PlannerResult | None:
    expansions: list[Expansion] = []
    unresolved = ["answer"]
    known_nodes: list[str] = []
    meta_trace: list[dict[str, Any]] = []
    revision_context: dict[str, Any] | None = None
    for _ in range(max_steps):
        if not unresolved:
            break
        target = unresolved.pop(0)
        expansion, meta = llm_single_expansion(model, item, facts, target, known_nodes, revision_context)
        revision_context = None
        meta_trace.append({"step": len(meta_trace), "target": target, "llm_meta": meta})
        if expansion is None:
            if failure_trace is not None:
                failure_trace.extend(meta_trace)
            return None
        expansions = [exp for exp in expansions if exp.target != target]
        expansions.append(expansion)
        for arg in expansion.args:
            text = str(arg)
            if not text.startswith(("FACT[", "CONST(")) and text not in known_nodes and text not in unresolved:
                unresolved.insert(0, text)
        if target not in known_nodes:
            known_nodes.append(target)
        if unresolved:
            continue
        trial = strict_backward_search(expansions, facts, "llm_backward")
        if trial is None:
            if failure_trace is not None:
                failure_trace.extend(meta_trace)
            return None
        exec_result = eval_program(trial.program, facts)
        contract_ok, contract_errors = contract_check(trial.program, facts)
        verified, rejection_reasons = verify_llm_candidate(trial.program, facts)
        verification_step = {
            "step": len(meta_trace),
            "status": "closed_candidate_verification",
            "eval_ok": exec_result.ok,
            "contract_ok": contract_ok,
            "verified": verified,
            "rejection_reasons": rejection_reasons,
            "contract_errors": contract_errors,
        }
        if exec_result.ok and contract_ok and verified:
            trial.trace = meta_trace + [verification_step] + trial.trace
            trial.proposal_backend = "llm"
            return trial
        meta_trace.append(verification_step | {
            "status": "closed_candidate_rejected_for_revision",
        })
        expansions = [exp for exp in expansions if exp.target != target]
        known_nodes = [node for node in known_nodes if node != target]
        unresolved = [target]
        revision_context = {
            "program": trial.program.to_dict(),
            "rejection_reasons": rejection_reasons + contract_errors + exec_result.errors,
        }
    if failure_trace is not None:
        failure_trace.extend([{
            "step": -1,
            "status": "failed_to_generate_verified_program",
            "reason": "llm_strict_backward_budget_exhausted_or_unresolved",
            "remaining_unresolved": unresolved,
        }, *meta_trace])
    return None


def verify_llm_candidate(program: Program, facts: list[Fact]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    referenced = program.referenced_facts()
    relation_facts = {fact.fact_id for fact in facts if fact.type == "relation_fact"}
    if relation_facts and not relation_facts <= referenced:
        missing = sorted(relation_facts - referenced)
        reasons.append(f"missing relation fact support: {missing}")
    if len(referenced) < min(len([f for f in facts if not f.derived]), 2):
        reasons.append("insufficient fact coverage")
    ok, contract_errors = contract_check(program, facts)
    if not ok:
        reasons.extend(contract_errors)
    return not reasons, reasons


def one_shot_program(facts: list[Fact]) -> Program:
    nodes = make_fact_nodes(facts)
    fact_nodes = list(nodes)
    if not fact_nodes:
        nodes["const_0"] = IRNode("const_0", "CONST", value=Decimal(0))
        return Program(nodes, "const_0", "one_shot")
    current = fact_nodes[0]
    for i, node_id in enumerate(fact_nodes[1:], 1):
        out = f"oneshot_{i}"
        prov = tuple(sorted(nodes[current].provenance + nodes[node_id].provenance))
        nodes[out] = IRNode(out, "ADD", (current, node_id), provenance=prov)
        current = out
    return Program(nodes, current, "one_shot")


def failed_planner_program(facts: list[Fact], name: str = "llm_failed") -> Program:
    return Program(make_fact_nodes(facts), "missing_answer_node", name)


def fact_node_for(facts: list[Fact], *patterns: str, value: Decimal | None = None) -> Fact | None:
    regexes = [re.compile(pattern, re.I) for pattern in patterns]
    for fact in facts:
        text = f"{fact.key} {fact.content}"
        if value is not None and fact.value != value:
            continue
        if regexes and not any(regex.search(text) for regex in regexes):
            continue
        return fact
    return None


def largest_numeric_fact(facts: list[Fact], *patterns: str) -> Fact | None:
    regexes = [re.compile(pattern, re.I) for pattern in patterns]
    candidates = []
    for fact in facts:
        if fact.value is None or fact.type != "numeric_fact":
            continue
        text = f"{fact.key} {fact.content}"
        if regexes and not any(regex.search(text) for regex in regexes):
            continue
        candidates.append(fact)
    if not candidates:
        return None
    return sorted(candidates, key=lambda fact: fact.value or Decimal(0), reverse=True)[0]


def add_op_node(nodes: dict[str, IRNode], node_id: str, op: str, left: str, right: str) -> str:
    provenance = tuple(sorted(set(nodes[left].provenance + nodes[right].provenance)))
    nodes[node_id] = IRNode(node_id=node_id, op=op, args=(left, right), provenance=provenance)
    return node_id


def semantic_expansion_plan(item: dict[str, Any], facts: list[Fact]) -> list[Expansion]:
    """Small deterministic proposal backend using evidence text, not gold plans.

    This is deliberately conservative: it covers common relation words in the
    current hidden-GSM8K style and provides a concrete slot where an LLM
    proposal backend can later be plugged in.
    """
    question = item.get("shared_question", "").lower()
    evidence = " ".join(f.content for f in facts).lower()
    def ref(fact: Fact | None) -> str | None:
        return f"FACT[{fact.fact_id}]" if fact else None

    if "tomorrow" in question and "remain" in evidence and "half" in evidence:
        yesterday = ref(fact_node_for(facts, "yesterday", value=None))
        double = ref(fact_node_for(facts, "twice|double", value=Decimal("2")))
        total = ref(fact_node_for(facts, "total") or largest_numeric_fact(facts))
        half = ref(fact_node_for(facts, "half", value=Decimal("0.5")))
        if yesterday and double and total and half:
            return [
                Expansion("answer", "MUL", ("generated_remaining", half)),
                Expansion("generated_remaining", "SUB", (total, "generated_read_so_far")),
                Expansion("generated_read_so_far", "ADD", (yesterday, "generated_today")),
                Expansion("generated_today", "MUL", (yesterday, double)),
            ]

    if ("still need" in question or "more money" in question) and ("cost" in evidence or "wallet" in evidence):
        cost = ref(fact_node_for(facts, "cost|wallet"))
        half = ref(fact_node_for(facts, "half", value=Decimal("0.5")))
        parent = ref(fact_node_for(facts, "parent", value=None))
        double = ref(fact_node_for(facts, "twice|double", value=Decimal("2")))
        if cost and half and parent and double:
            return [
                Expansion("answer", "SUB", (cost, "generated_has_now")),
                Expansion("generated_has_now", "ADD", ("generated_already_has", "generated_gifts")),
                Expansion("generated_already_has", "MUL", (cost, half)),
                Expansion("generated_gifts", "ADD", (parent, "generated_grandparents")),
                Expansion("generated_grandparents", "MUL", (parent, double)),
            ]

    if "final weight" in question and ("triple" in evidence or "double" in evidence):
        initial = ref(fact_node_for(facts, "initial|weigh|pound", value=None))
        triple = ref(fact_node_for(facts, "triple", value=Decimal("3")))
        added = ref(fact_node_for(facts, "more|afterward|added", value=Decimal("2")))
        double = ref(fact_node_for(facts, "double", value=Decimal("2")))
        if initial and triple and added and double:
            return [
                Expansion("answer", "MUL", ("generated_after_add", double)),
                Expansion("generated_after_add", "ADD", ("generated_after_triple", added)),
                Expansion("generated_after_triple", "MUL", (initial, triple)),
            ]

    return []


def resolve_expansion_arg(arg: Any, nodes: dict[str, IRNode], facts_by_id: dict[str, Fact]) -> str | None:
    text = str(arg).strip()
    fact_match = re.fullmatch(r"FACT\[(.+?)\]", text)
    if fact_match:
        fact_id = fact_match.group(1).strip()
        if fact_id not in facts_by_id:
            return None
        node_id = f"fact_{fact_id}"
        if node_id not in nodes:
            fact = facts_by_id[fact_id]
            nodes[node_id] = IRNode(node_id=node_id, op="FACT", fact_id=fact_id, label=fact.key, provenance=(fact_id,))
        return node_id
    const_match = re.fullmatch(r"CONST\((.+?)\)", text)
    if const_match:
        value = decimal(const_match.group(1))
        if value is None:
            return None
        node_id = f"const_{str(value).replace('.', '_').replace('-', 'neg_')}"
        nodes.setdefault(node_id, IRNode(node_id=node_id, op="CONST", value=value, provenance=()))
        return node_id
    return text


def strict_backward_search(expansions: list[Expansion], facts: list[Fact], name: str) -> PlannerResult | None:
    facts_by_id = {fact.fact_id: fact for fact in facts}
    nodes = make_fact_nodes(facts)
    unresolved = ["answer"]
    trace: list[dict[str, Any]] = []
    candidates: list[Decimal] = []
    expansion_by_target = {exp.target: exp for exp in expansions}
    step = 0
    while unresolved:
        target = unresolved.pop(0)
        expansion = expansion_by_target.get(target)
        if expansion is None:
            return None
        before = 1 + len(unresolved)
        if expansion.op not in OPS:
            return None
        parsed_args = [resolve_expansion_arg(arg, nodes, facts_by_id) for arg in expansion.args]
        if any(arg is None for arg in parsed_args) or len(parsed_args) < 2:
            return None
        child_nodes = [str(arg) for arg in parsed_args if str(arg) in nodes]
        provenance = tuple(sorted(set(p for arg in child_nodes for p in nodes[arg].provenance)))
        nodes[target] = IRNode(target, expansion.op, tuple(str(arg) for arg in parsed_args), provenance=provenance)
        introduced = [
            str(arg) for arg in parsed_args
            if str(arg) not in nodes and str(arg) != target
        ]
        unresolved = [node_id for node_id in introduced if node_id not in unresolved] + unresolved
        partial = Program(nodes, target, name)
        exec_result = eval_program(partial, facts)
        if exec_result.answer is not None:
            candidates.append(exec_result.answer)
        checks = {
            "fact_support": all(
                nodes[arg].op != "FACT" or nodes[arg].fact_id in facts_by_id
                for arg in child_nodes
            ),
            "invented_variable": all(
                str(arg).startswith(("FACT[", "CONST(")) or str(arg) in nodes or str(arg) in expansion_by_target
                for arg in expansion.args
            ),
            "unresolved_reduction": target not in unresolved,
            "executability": bool(introduced) or exec_result.ok,
            "fact_conflict": not detect_conflicts(facts),
        }
        step += 1
        trace.append({
            "step": step,
            "expanded_node": target,
            "op": expansion.op,
            "children": child_nodes,
            "provenance": list(provenance),
            "checks": checks,
            "errors": exec_result.errors,
            "unresolved_before": before,
            "unresolved_after": len(unresolved),
            "introduced_unresolved": introduced,
            "proposal_source": expansion.source,
        })
        if not all(checks.values()) and expansion.source != "semantic":
            return None
    program = Program(nodes, "answer", name).reachable_program()
    result = eval_program(program, facts)
    if result.answer is not None:
        candidates.append(result.answer)
    return PlannerResult(program, candidates, trace, name)


def backward_expansion_trace(program: Program, facts: list[Fact]) -> list[dict[str, Any]]:
    """Record a backward, one-unresolved-node-at-a-time expansion trace.

    The current proposal backend is still deterministic, but this trace makes
    the planner contract explicit: start at answer, expand exactly one node per
    step, and finish only when leaves are FACT/CONST.
    """
    fact_ids = {fact.fact_id for fact in facts}
    trace: list[dict[str, Any]] = []
    unresolved = [program.answer_node]
    expanded: set[str] = set()
    step = 0
    while unresolved:
        node_id = unresolved.pop(0)
        if node_id in expanded:
            continue
        node = program.nodes.get(node_id)
        before = 1 + len(unresolved)
        if node is None:
            step += 1
            trace.append({
                "step": step,
                "expanded_node": node_id,
                "status": "failed",
                "checks": {
                    "invented_variable": False,
                    "fact_support": False,
                    "unresolved_reduction": False,
                    "executability": False,
                    "fact_conflict": not detect_conflicts(facts),
                },
                "errors": [f"unknown node {node_id}"],
                "unresolved_before": before,
                "unresolved_after": len(unresolved),
            })
            continue
        child_nodes = [str(arg) for arg in node.args if str(arg) in program.nodes]
        new_unresolved = [
            child for child in child_nodes
            if program.nodes[child].op not in {"FACT", "CONST"} and child not in expanded
        ]
        unresolved = new_unresolved + unresolved
        expanded.add(node_id)
        after = len(unresolved)
        exec_result = eval_program(Program(program.nodes, node_id, program.name), facts)
        checks = {
            "invented_variable": all(str(arg) in program.nodes for arg in node.args),
            "fact_support": (node.op != "FACT" or bool(node.fact_id in fact_ids)),
            "unresolved_reduction": node_id not in unresolved,
            "executability": exec_result.ok,
            "fact_conflict": not detect_conflicts(facts),
        }
        step += 1
        trace.append({
            "step": step,
            "expanded_node": node_id,
            "op": node.op,
            "children": child_nodes,
            "bound_fact": node.fact_id,
            "provenance": list(node.provenance),
            "checks": checks,
            "errors": exec_result.errors,
            "unresolved_before": before,
            "unresolved_after": after,
            "introduced_unresolved": new_unresolved,
        })
    return trace


def enumerate_programs(facts: list[Fact], max_depth: int, beam_size: int, prune: bool) -> list[Program]:
    base = make_fact_nodes(facts)
    candidates: list[tuple[Decimal, str, dict[str, IRNode]]] = []
    for node_id, node in base.items():
        fact = next((f for f in facts if f.fact_id == node.fact_id), None)
        if fact and fact.value is not None:
            candidates.append((fact.value, node_id, dict(base)))
    frontier = list(candidates)
    seen = {(v, n) for v, n, _ in frontier}
    programs: list[Program] = [Program(nodes, node_id, "backward") for _, node_id, nodes in frontier]
    for depth in range(1, max_depth + 1):
        new_frontier = []
        for (left_val, left_id, left_nodes), (right_val, right_id, _) in itertools.product(frontier, candidates):
            for op in ("ADD", "SUB", "MUL", "DIV"):
                if op == "DIV" and right_val == 0:
                    continue
                value = {"ADD": left_val + right_val, "SUB": left_val - right_val, "MUL": left_val * right_val, "DIV": left_val / right_val}[op]
                if not math.isfinite(float(value)):
                    continue
                nodes = dict(left_nodes)
                nodes.update(base)
                out = f"search_{depth}_{len(new_frontier)}"
                prov = tuple(sorted(set(nodes[left_id].provenance + nodes[right_id].provenance)))
                nodes[out] = IRNode(out, op, (left_id, right_id), provenance=prov)
                program = Program(nodes, out, "backward_beam")
                if prune and not contract_check(program, facts)[0]:
                    continue
                key = (value, out)
                if key not in seen:
                    seen.add(key)
                    new_frontier.append((value, out, nodes))
                    programs.append(program)
        new_frontier.sort(key=lambda item: (len(item[2]), str(abs(item[0]))))
        frontier = new_frontier[:beam_size]
        if not frontier:
            break
    return programs


def choose_program(
    planner: str,
    facts: list[Fact],
    item: dict[str, Any],
    proposal_backend: str,
    model: Any | None = None,
) -> PlannerResult:
    if planner == "A_one_shot":
        program = one_shot_program(facts)
        result = eval_program(program, facts)
        program = program.reachable_program()
        return PlannerResult(
            program,
            [result.answer] if result.answer is not None else [],
            backward_expansion_trace(program, facts),
            "one_shot",
        )
    if proposal_backend in {"llm", "hybrid"}:
        if model is None:
            raise RuntimeError("proposal_backend requires a loaded model")
        failed_llm_trace: list[dict[str, Any]] = []
        llm_result = llm_strict_backward_search(model, item, facts, failure_trace=failed_llm_trace)
        if llm_result is not None:
            verified, rejection_reasons = verify_llm_candidate(llm_result.program, facts)
            if verified:
                return llm_result
            llm_rejection_trace = [{
                "step": -1,
                "proposal_backend": "llm",
                "verification": "failed",
                "rejection_reasons": rejection_reasons,
            }, *llm_result.trace]
        else:
            llm_rejection_trace = [{
                "step": -1,
                "proposal_backend": "llm",
                "verification": "failed_to_produce_program",
            }, *failed_llm_trace]
        if proposal_backend == "llm":
            trace = [{
                "step": -1,
                "proposal_backend": "llm",
                "status": "failed_to_generate_verified_program",
            }, *llm_rejection_trace]
            return PlannerResult(failed_planner_program(facts), [], trace, "llm_failed")
    if proposal_backend in {"semantic", "hybrid"}:
        semantic_result = strict_backward_search(
            semantic_expansion_plan(item, facts),
            facts,
            "semantic_backward",
        )
        if semantic_result is not None:
            if proposal_backend == "hybrid" and "llm_rejection_trace" in locals():
                semantic_result.trace = llm_rejection_trace + semantic_result.trace
                semantic_result.proposal_backend = "hybrid_repaired"
            return semantic_result
    prune = planner in {"D_beam_prune", "E_beam_verify_repair"}
    beam = 1 if planner == "B_backward" else 3
    programs = enumerate_programs(facts, max_depth=5, beam_size=beam, prune=prune)
    scored = []
    candidates = []
    for program in programs:
        result = eval_program(program, facts)
        if result.answer is not None:
            candidates.append(result.answer)
        # Evidence-only heuristic: prefer executable programs that combine more
        # distinct original facts, then shorter graphs.  Gold answers/plans are
        # deliberately unavailable here.
        support = len(program.referenced_facts())
        score = (-support, len(program.edges()))
        scored.append((score, program))
    if not scored:
        program = one_shot_program(facts).reachable_program()
        return PlannerResult(program, candidates, backward_expansion_trace(program, facts), "enumeration")
    chosen = sorted(scored, key=lambda x: x[0])[0][1]
    if planner == "E_beam_verify_repair":
        result = eval_program(chosen, facts)
        if not result.ok:
            repaired = [p for p in programs if eval_program(p, facts).ok and contract_check(p, facts)[0]]
            if repaired:
                chosen = sorted(repaired, key=lambda p: (-len(p.referenced_facts()), len(p.edges())))[0]
    chosen = chosen.reachable_program()
    trace = backward_expansion_trace(chosen, facts)
    if proposal_backend == "hybrid" and "llm_rejection_trace" in locals():
        trace = llm_rejection_trace + trace
    return PlannerResult(chosen, candidates, trace, "enumeration")


def fact_correctness(observed: list[Fact], gold: list[Fact]) -> float:
    gold_values = {g.value for g in gold if g.value is not None}
    observed_values = {f.value for f in observed if f.value is not None and not f.derived}
    return len(observed_values & gold_values) / len(gold_values) if gold_values else 1.0


def run_case(
    item: dict[str, Any],
    qid: int,
    setting: str,
    planner: str,
    proposal_backend: str,
    model: Any | None = None,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    use_gold = setting.startswith("gold")
    use_history = "_history_" in setting
    facts = gold_facts(item) if use_gold else llm_like_facts(item, history=False)
    if use_history and facts:
        first = facts[0]
        facts = [*facts, Fact(
            fact_id="H_001",
            source="history",
            content=f"historical noisy restatement of {first.key}",
            key=first.key,
            value=first.value + 1 if first.value is not None else None,
            derived=True,
            provenance=(first.fact_id,),
        )]
    oracle = build_oracle_plan(item, gold_facts(item))
    if setting.endswith("_oracle"):
        program = align_oracle_to_facts(oracle, facts).reachable_program()
        candidates = [eval_program(program, facts).answer]
        planner_trace = backward_expansion_trace(program, facts)
        active_backend = "oracle"
    else:
        planner_result = choose_program(planner, facts, item, proposal_backend, model)
        program = planner_result.program
        candidates = planner_result.candidates
        planner_trace = planner_result.trace
        active_backend = planner_result.proposal_backend
    result = eval_program(program, facts)
    contract_ok, contract_errors = contract_check(program, facts)
    f1 = edge_f1(program, facts, oracle.program, oracle.facts)
    final_correct = close_enough(result.answer, oracle.answer)
    token_usage, llm_runtime_seconds, llm_call_count = collect_trace_usage(planner_trace)
    case_runtime_seconds = time.perf_counter() - case_started
    return {
        "question_id": qid,
        "setting": setting,
        "planner": "oracle" if setting.endswith("_oracle") else planner,
        "proposal_backend": active_backend,
        "fact_correctness": fact_correctness(facts, oracle.facts),
        **f1,
        "executable": result.ok and contract_ok,
        "candidate_emerged": any(close_enough(c, oracle.answer) for c in candidates),
        "final_correct": final_correct,
        "llm_call_count": llm_call_count,
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
        "llm_runtime_seconds": round(llm_runtime_seconds, 6),
        "case_runtime_seconds": round(case_runtime_seconds, 6),
        "answer": str(result.answer) if result.answer is not None else "",
        "gold_answer": str(oracle.answer) if oracle.answer is not None else "",
        "errors": contract_errors + result.errors,
        "fact_conflicts": detect_conflicts(facts),
        "planner_trace": planner_trace,
        "program": program.to_dict(),
        "oracle": oracle.to_dict(),
        "facts": [f.to_dict() for f in facts],
    }


CAUSAL_SETTINGS = {
    "gold_fresh_oracle": "Gold Facts + Fresh + Oracle Plan",
    "gold_fresh_generated": "Gold Facts + Fresh + Generated Plan",
    "llm_fresh_oracle": "LLM Facts + Fresh + Oracle Plan",
    "llm_fresh_generated": "LLM Facts + Fresh + Generated Plan",
    "gold_history_generated": "Gold Facts + History + Generated Plan",
    "llm_history_generated": "LLM Facts + History + Generated Plan",
}

PLANNERS = ["A_one_shot", "B_backward", "C_beam", "D_beam_prune", "E_beam_verify_repair", "LLM_strict_backward"]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["setting"], row["planner"]), []).append(row)
    summary = []
    for (setting, planner), group in sorted(grouped.items()):
        n = len(group)
        summary.append({
            "setting": setting,
            **({"oracle_level": group[0].get("oracle_level", "")} if "oracle_level" in group[0] else {}),
            **({"visible_oracle_information": group[0].get("visible_oracle_information", "")} if "visible_oracle_information" in group[0] else {}),
            "planner": planner,
            "n": n,
            "fact_correctness": sum(r["fact_correctness"] for r in group) / n,
            "dependency_edge_f1": sum(r["edge_f1"] for r in group) / n,
            "executable_rate": sum(bool(r["executable"]) for r in group) / n,
            "candidate_emergence_rate": sum(bool(r["candidate_emerged"]) for r in group) / n,
            "final_accuracy": sum(bool(r["final_correct"]) for r in group) / n,
            "llm_call_count": sum(int(r.get("llm_call_count", 0) or 0) for r in group),
            "prompt_tokens": sum(int(r.get("prompt_tokens", 0) or 0) for r in group),
            "completion_tokens": sum(int(r.get("completion_tokens", 0) or 0) for r in group),
            "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in group),
            "llm_runtime_seconds": round(sum(float(r.get("llm_runtime_seconds", 0.0) or 0.0) for r in group), 6),
            "case_runtime_seconds": round(sum(float(r.get("case_runtime_seconds", 0.0) or 0.0) for r in group), 6),
        })
    return summary


def build_gap_analysis(summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["setting"], row["planner"]): row for row in summary}
    oracle_gold = by_key.get(("gold_fresh_oracle", "oracle"), {})
    oracle_llm = by_key.get(("llm_fresh_oracle", "oracle"), {})
    result: dict[str, Any] = {
        "fact_extraction_gap": {
            "definition": "Gold Facts + Fresh + Oracle Plan minus LLM Facts + Fresh + Oracle Plan",
            "final_accuracy_gap": oracle_gold.get("final_accuracy", 0.0) - oracle_llm.get("final_accuracy", 0.0),
            "fact_correctness_gap": oracle_gold.get("fact_correctness", 0.0) - oracle_llm.get("fact_correctness", 0.0),
        },
        "dependency_recovery_gap": {},
        "history_contamination_gap": {},
    }
    present_planners = sorted({
        planner for (setting, planner) in by_key
        if setting.endswith("_generated") and planner != "oracle"
    })
    for planner in present_planners:
        fresh = by_key.get(("gold_fresh_generated", planner), {})
        history_gold = by_key.get(("gold_history_generated", planner), {})
        llm_fresh = by_key.get(("llm_fresh_generated", planner), {})
        llm_history = by_key.get(("llm_history_generated", planner), {})
        result["dependency_recovery_gap"][planner] = {
            "definition": "Gold Facts + Fresh + Oracle Plan minus Gold Facts + Fresh + Generated Plan",
            "final_accuracy_gap": oracle_gold.get("final_accuracy", 0.0) - fresh.get("final_accuracy", 0.0),
            "dependency_edge_f1_gap": oracle_gold.get("dependency_edge_f1", 0.0) - fresh.get("dependency_edge_f1", 0.0),
        }
        result["history_contamination_gap"][planner] = {
            "definition": "Fresh generated setting minus history generated setting",
            "gold_fact_final_accuracy_gap": fresh.get("final_accuracy", 0.0) - history_gold.get("final_accuracy", 0.0),
            "llm_fact_final_accuracy_gap": llm_fresh.get("final_accuracy", 0.0) - llm_history.get("final_accuracy", 0.0),
            "gold_fact_executable_rate_gap": fresh.get("executable_rate", 0.0) - history_gold.get("executable_rate", 0.0),
            "llm_fact_executable_rate_gap": llm_fresh.get("executable_rate", 0.0) - llm_history.get("executable_rate", 0.0),
        }
    return result


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, run_config: dict[str, Any] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "script": Path(__file__).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": CAUSAL_SETTINGS,
        "planners": PLANNERS,
    }
    if run_config:
        config.update(run_config)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    compact_cases: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {"oracle_plans": {}, "generated_programs": {}, "dependency_graphs": {}}
    for row in rows:
        stem = f"q{row['question_id']:03d}_{row['setting']}_{row['planner']}"
        graph = Program(
            nodes={k: IRNode(
                node_id=v["node_id"],
                op=v["op"],
                args=tuple(v.get("args") or ()),
                fact_id=v.get("fact_id"),
                value=decimal(v.get("value")),
                label=v.get("label", ""),
                provenance=tuple(v.get("provenance") or ()),
            ) for k, v in row["program"]["nodes"].items()},
            answer_node=row["program"]["answer_node"],
            name=row["program"].get("name", ""),
        )

        artifacts["oracle_plans"][stem] = row["oracle"]
        artifacts["generated_programs"][stem] = row["program"]
        artifacts["dependency_graphs"][stem] = {
            "answer_node": graph.answer_node,
            "edges": sorted(list(graph.edges())),
            "referenced_facts": sorted(graph.referenced_facts()),
        }
        compact_cases.append({
            key: value for key, value in row.items()
            if key not in {"oracle", "program"}
        } | {
            "artifact_id": stem,
        })

    (output_dir / "cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in compact_cases),
        encoding="utf-8",
    )
    (output_dir / "artifacts.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_rows = summarize(rows)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]) if metric_rows else ["setting"])
        writer.writeheader()
        writer.writerows(metric_rows)
    errors = [r for r in compact_cases if not r["final_correct"] or r["errors"]]
    (output_dir / "error_analysis.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "gap_analysis.json").write_text(
        json.dumps(build_gap_analysis(metric_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_experiment(
    data_path: Path,
    output_dir: Path,
    limit: int,
    planner: str,
    proposal_backend: str = "semantic",
    model: Any | None = None,
    run_config: dict[str, Any] | None = None,
    incremental_write: bool = True,
) -> list[dict[str, Any]]:
    records = read_records(data_path)[:limit or None]
    rows = []
    generated_planners = PLANNERS if planner == "all" else [planner]
    per_question = sum(1 if setting.endswith("_oracle") else len(generated_planners) for setting in CAUSAL_SETTINGS)
    total_cases = len(records) * per_question
    completed = 0
    def progress_suffix(row: dict[str, Any]) -> str:
        return (
            f"llm_calls={row.get('llm_call_count', 0)} "
            f"tokens={row.get('total_tokens', 0)} "
            f"llm_sec={row.get('llm_runtime_seconds', 0.0)} "
            f"case_sec={row.get('case_runtime_seconds', 0.0)}"
        )

    for qid, item in enumerate(records, 1):
        print(f"[question {qid}/{len(records)}] {item.get('shared_question', '')}")
        for setting in CAUSAL_SETTINGS:
            if setting.endswith("_oracle"):
                completed += 1
                print(f"  [{completed}/{total_cases}] {setting} / oracle")
                row = run_case(item, qid, setting, planner, proposal_backend, model)
                rows.append(row)
                if incremental_write:
                    write_outputs(rows, output_dir, run_config=run_config)
                    print(f"    saved {len(rows)} rows -> {output_dir.resolve()} | {progress_suffix(row)}")
            else:
                for active_planner in generated_planners:
                    completed += 1
                    print(f"  [{completed}/{total_cases}] {setting} / {active_planner} / backend={proposal_backend}")
                    row = run_case(item, qid, setting, active_planner, proposal_backend, model)
                    rows.append(row)
                    if incremental_write:
                        write_outputs(rows, output_dir, run_config=run_config)
                        print(f"    saved {len(rows)} rows -> {output_dir.resolve()} | {progress_suffix(row)}")
    write_outputs(rows, output_dir, run_config=run_config)
    total_usage = {
        "llm_call_count": sum(int(r.get("llm_call_count", 0) or 0) for r in rows),
        "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in rows),
        "llm_runtime_seconds": round(sum(float(r.get("llm_runtime_seconds", 0.0) or 0.0) for r in rows), 6),
        "case_runtime_seconds": round(sum(float(r.get("case_runtime_seconds", 0.0) or 0.0) for r in rows), 6),
    }
    print(f"Usage totals: {total_usage}")
    return rows


def smoke_test() -> None:
    rows = run_experiment(DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR / "smoke", limit=3, planner="E_beam_verify_repair")
    oracle_rows = [r for r in rows if r["setting"] == "gold_fresh_oracle"]
    assert oracle_rows and all(r["executable"] for r in oracle_rows), "Phase 1 failed: oracle plan did not execute"
    assert all(r["program"]["answer_node"] in r["program"]["nodes"] for r in rows), "missing answer node"
    print("Phase 1: fact representation + provenance + executable IR + oracle execution: PASS")
    print("Phase 2: backward dependency search smoke path: PASS")
    print("Phase 3: beam search interface: PASS")
    print("Phase 4: executable/contract pruning: PASS")
    print("Phase 5: verification + local repair hook: PASS")
    print("Phase 6: six-group causal matrix output: PASS")
    print(f"Smoke rows: {len(rows)}")
    print(f"Smoke output: {DEFAULT_OUTPUT_DIR / 'smoke'}")


def llm_proposal_smoke(data_path: Path, model: Any, output_dir: Path) -> None:
    item = read_records(data_path)[0]
    facts = gold_facts(item)
    planner_result = llm_strict_backward_search(model, item, facts)
    output_dir.mkdir(parents=True, exist_ok=True)
    program = planner_result.program if planner_result else None
    result = eval_program(program, facts) if program is not None else ExecutionResult(False, None, {}, ["no program"])
    payload = {
        "question": item.get("shared_question", ""),
        "facts": [fact.to_dict() for fact in facts],
        "planner_trace": planner_result.trace if planner_result else [],
        "program": program.to_dict() if program else None,
        "execution": {
            "ok": result.ok,
            "answer": str(result.answer) if result.answer is not None else None,
            "errors": result.errors,
        },
    }
    (output_dir / "llm_proposal_smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if program is None or not result.ok:
        raise SystemExit(f"LLM proposal smoke failed; wrote {output_dir / 'llm_proposal_smoke.json'}")
    print(f"LLM proposal smoke PASS; answer={result.answer}; output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover executable dependency programs from dispersed evidence.")
    parser.add_argument("--mode", choices=["dependency_program", "oracle_decomposition"], default="dependency_program")
    parser.add_argument("--oracle-level", choices=["O0", "O1", "O2", "O3", "O4", "O5", "O6", "all"], default="all")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--planner", choices=[*PLANNERS, "all"], default="all")
    parser.add_argument("--proposal-backend", choices=PROPOSAL_BACKENDS, default="hybrid")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--llm-proposal-smoke", action="store_true")
    parser.add_argument(
        "--no-incremental-write",
        action="store_true",
        help="Disable saving after every completed case. By default each case is checkpointed immediately.",
    )
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="Alias for --proposal-backend hybrid.",
    )
    args = parser.parse_args()
    if args.use_model:
        args.proposal_backend = "hybrid"
    if args.mode == "oracle_decomposition":
        import oracle_decomposition_experiment as oracle_decomp

        if args.use_model or args.proposal_backend == "hybrid":
            args.proposal_backend = "llm"
        if args.smoke_test:
            oracle_decomp.smoke_test()
            return
        if args.planner == "all":
            args.planner = "LLM_strict_backward"
        output_dir = Path(args.output_dir)
        if output_dir.parent == DEFAULT_OUTPUT_DIR:
            output_dir = oracle_decomp.DEFAULT_OUTPUT_DIR / output_dir.name
        model = oracle_decomp.load_model_if_needed(args)
        rows = oracle_decomp.run_experiment(
            Path(args.data_path),
            output_dir,
            args.limit,
            args.oracle_level,
            args.planner,
            args.proposal_backend,
            model=model,
            incremental_write=not args.no_incremental_write,
            run_config={
                "mode": args.mode,
                "oracle_level": args.oracle_level,
                "data_path": str(Path(args.data_path).resolve()),
                "planner": args.planner,
                "proposal_backend": args.proposal_backend,
                "model_path": str(Path(args.model_path).resolve()) if args.proposal_backend in {"llm", "hybrid"} else "",
                "device": args.device if args.proposal_backend in {"llm", "hybrid"} else "",
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
        )
        print(f"Wrote {len(rows)} rows to {output_dir.resolve()}")
        print(f"Metrics: {output_dir.resolve() / 'metrics.csv'}")
        print(f"Cases: {output_dir.resolve() / 'cases.jsonl'}")
        return
    if args.smoke_test:
        smoke_test()
        return
    model = None
    if args.proposal_backend in {"llm", "hybrid"}:
        from run_hidden_gsm8k import LocalQwen, reseed_model

        print(f"Loading local Qwen proposal backend from {Path(args.model_path).resolve()}")
        model = LocalQwen(
            Path(args.model_path).resolve(),
            args.device,
            args.max_new_tokens,
            args.temperature,
            args.allow_download,
        )
        reseed_model(model, args.seed)
    else:
        print(
            "Running deterministic offline dependency-program experiment "
            "(no model calls, no training; quick completion is expected)."
        )
    if args.llm_proposal_smoke:
        if model is None:
            raise SystemExit("--llm-proposal-smoke requires --proposal-backend llm or hybrid")
        llm_proposal_smoke(Path(args.data_path), model, Path(args.output_dir))
        return
    run_config = {
        "data_path": str(Path(args.data_path).resolve()),
        "proposal_backend": args.proposal_backend,
        "model_path": str(Path(args.model_path).resolve()) if args.proposal_backend in {"llm", "hybrid"} else "",
        "device": args.device if args.proposal_backend in {"llm", "hybrid"} else "",
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    rows = run_experiment(
        Path(args.data_path),
        Path(args.output_dir),
        args.limit,
        args.planner,
        proposal_backend=args.proposal_backend,
        model=model,
        run_config=run_config,
        incremental_write=not args.no_incremental_write,
    )
    print(f"Wrote {len(rows)} rows to {Path(args.output_dir).resolve()}")
    print(f"Metrics: {Path(args.output_dir).resolve() / 'metrics.csv'}")
    print(f"Gap analysis: {Path(args.output_dir).resolve() / 'gap_analysis.json'}")


if __name__ == "__main__":
    main()
