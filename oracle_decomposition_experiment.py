"""Oracle-decomposition experiment for dependency-program recovery.

This module reuses the dependency-program experiment framework and varies only
the oracle information visible to the planner.  O6 is the upper-bound condition
that executes the full oracle program; O0-O5 still generate a dependency
program through planner paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dependency_program_experiment as dep


ROOT = Path(__file__).resolve().parent

# Question bank used by default. CLI --data-path can still override this.
QUESTION_BANK_PATH = ROOT / "data" / "20.json"
DEFAULT_DATA_PATH = QUESTION_BANK_PATH
DEFAULT_OUTPUT_DIR = ROOT / "outputs_oracle_decomposition"
ORACLE_LEVELS = ("O0", "O1", "O2", "O3", "O4", "O5", "O6")
ORACLE_LEVEL_DESCRIPTIONS = {
    "O0": "Gold Facts Only",
    "O1": "Oracle Goal",
    "O2": "Oracle Relevant Facts",
    "O3": "Oracle Fact Binding",
    "O4": "Oracle Operators / Local Relations",
    "O5": "Oracle Topology",
    "O6": "Full Oracle Program",
}
ORACLE_VISIBLE_INFORMATION = {
    "O0": [],
    "O1": ["goal"],
    "O2": ["relevant_facts"],
    "O3": ["fact_binding"],
    "O4": ["local_relations"],
    "O5": ["topology"],
    "O6": ["full_program"],
}
RECOVERY_CURVE_LABELS = {
    "O0": "O0 No Oracle",
    "O1": "O1 Goal",
    "O2": "O2 Leaves",
    "O3": "O3 Binding",
    "O4": "O4 Operator",
    "O5": "O5 Topology",
    "O6": "O6 Full Program",
}


@dataclass(frozen=True)
class OracleContext:
    level: str
    visible_oracle_information: list[str]
    goal: str | None
    relevant_facts: dict[str, Any]
    fact_binding: dict[str, str]
    constants: dict[str, str]
    local_relations: list[dict[str, Any]]
    topology: list[dict[str, Any]]
    full_program: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "visible_oracle_information": self.visible_oracle_information,
            "goal": self.goal,
            "relevant_facts": self.relevant_facts,
            "fact_binding": self.fact_binding,
            "constants": self.constants,
            "local_relations": self.local_relations,
            "topology": self.topology,
            "full_program": self.full_program,
        }


GOLD_SEMANTIC_GOALS = {
    "How many pages should Julie read tomorrow?": "tomorrow_pages",
    "How much more money does Betty still need?": "remaining_money",
    "What is the final weight of the box?": "final_weight",
    "How much did Alexis pay for the shoes?": "shoes_cost",
    "How many stamps does Bella buy altogether?": "total_stamps",
    "How much does each top cost?": "each_top_cost",
    "How much sales revenue does Noah make this month?": "this_month_sales_revenue",
    "How many minutes does Carolyn practice in 4 weeks?": "four_week_practice_minutes",
    "How many minutes does the third part take?": "third_part_duration_minutes",
    "How much does James earn each week from both jobs?": "weekly_total_earnings",
    "What balance remains after the 4 monthly payments?": "remaining_laptop_balance",
    "How many packs must Roger buy?": "required_pack_count",
    "How many kilograms does each of the last two people lose?": "last_two_loss_each",
    "How many vegetables does the garden produce altogether?": "total_vegetables",
    "How many cans does Jennifer take home?": "take_home_cans",
    "What was Irene's total income last week?": "last_week_total_income",
    "How much money does Winwin take home?": "take_home_money",
    "How much money remains in John's piggy bank?": "remaining_piggy_bank_money",
    "How many plates are needed?": "required_plate_count",
    "How many hard hats remain in the truck?": "remaining_hard_hats",
}


def meta(
    question: str,
    goal: str,
    facts: dict[str, str],
    relations: list[tuple[str, str, list[str]]],
    constants: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "goal": goal,
        "relevant_facts": list(facts),
        "fact_bindings": facts,
        "constants": constants or {},
        "relations": [
            {"relation_id": relation_id, "op": op, "inputs": inputs}
            for relation_id, op, inputs in relations
        ],
    }


GOLD_ORACLE_METADATA = {
    "How many pages should Julie read tomorrow?": meta(
        "How many pages should Julie read tomorrow?",
        "tomorrow_pages",
        {"A_001": "yesterday_pages", "A_002": "today_multiplier", "B_001": "total_pages", "B_002": "remaining_fraction"},
        [("R001", "MUL", ["A_001", "A_002"]), ("R002", "ADD", ["A_001", "R001"]), ("R003", "SUB", ["B_001", "R002"]), ("R004", "MUL", ["R003", "B_002"])],
    ),
    "How much more money does Betty still need?": meta(
        "How much more money does Betty still need?",
        "remaining_money",
        {"A_001": "wallet_cost", "A_002": "already_saved_fraction", "B_001": "parent_gift", "B_002": "grandparent_gift_multiplier"},
        [("R001", "MUL", ["A_001", "A_002"]), ("R002", "MUL", ["B_001", "B_002"]), ("R003", "ADD", ["R001", "B_001", "R002"]), ("R004", "SUB", ["A_001", "R003"])],
    ),
    "What is the final weight of the box?": meta(
        "What is the final weight of the box?",
        "final_weight",
        {"A_001": "initial_weight", "A_002": "brownie_weight_multiplier", "B_001": "added_weight", "B_002": "final_weight_multiplier"},
        [("R001", "MUL", ["A_001", "A_002"]), ("R002", "ADD", ["R001", "B_001"]), ("R003", "MUL", ["R002", "B_002"])],
    ),
    "How much did Alexis pay for the shoes?": meta(
        "How much did Alexis pay for the shoes?",
        "shoes_cost",
        {"A_001": "budget", "A_002": "item_cost_1", "A_003": "item_cost_2", "A_004": "item_cost_3", "B_001": "item_cost_4", "B_002": "item_cost_5", "B_003": "remaining_money"},
        [("R001", "ADD", ["A_002", "A_003", "A_004", "B_001", "B_002"]), ("R002", "SUB", ["A_001", "B_003"]), ("R003", "SUB", ["R002", "R001"])],
    ),
    "How many stamps does Bella buy altogether?": meta(
        "How many stamps does Bella buy altogether?",
        "total_stamps",
        {"A_001": "snowflake_stamps", "A_002": "truck_more_than_snowflake", "B_001": "rose_fewer_than_truck"},
        [("R001", "ADD", ["A_001", "A_002"]), ("R002", "SUB", ["R001", "B_001"]), ("R003", "ADD", ["A_001", "R001", "R002"])],
    ),
    "How much does each top cost?": meta(
        "How much does each top cost?",
        "each_top_cost",
        {"A_001": "total_spent", "A_002": "shorts_count", "A_003": "short_price", "B_001": "shoe_pair_count", "B_002": "shoe_pair_price", "B_003": "top_count"},
        [("R001", "MUL", ["A_002", "A_003"]), ("R002", "MUL", ["B_001", "B_002"]), ("R003", "SUB", ["A_001", "R001", "R002"]), ("R004", "DIV", ["R003", "B_003"])],
    ),
    "How much sales revenue does Noah make this month?": meta(
        "How much sales revenue does Noah make this month?",
        "this_month_sales_revenue",
        {"A_001": "large_painting_price", "A_002": "small_painting_price", "A_003": "large_paintings_sold_last_month", "B_001": "small_paintings_sold_last_month", "B_002": "this_month_revenue_multiplier"},
        [("R001", "MUL", ["A_003", "A_001"]), ("R002", "MUL", ["B_001", "A_002"]), ("R003", "ADD", ["R001", "R002"]), ("R004", "MUL", ["R003", "B_002"])],
    ),
    "How many minutes does Carolyn practice in 4 weeks?": meta(
        "How many minutes does Carolyn practice in 4 weeks?",
        "four_week_practice_minutes",
        {"A_001": "piano_minutes_per_day", "A_002": "violin_to_piano_multiplier", "B_001": "practice_days_per_week", "B_002": "week_count"},
        [("R001", "MUL", ["A_001", "A_002"]), ("R002", "ADD", ["A_001", "R001"]), ("R003", "MUL", ["R002", "B_001"]), ("R004", "MUL", ["R003", "B_002"])],
    ),
    "How many minutes does the third part take?": meta(
        "How many minutes does the third part take?",
        "third_part_duration_minutes",
        {"A_001": "first_part_minutes", "A_002": "second_part_multiplier", "B_001": "full_assignment_hours"},
        [("R001", "MUL", ["A_001", "A_002"]), ("R002", "ADD", ["A_001", "R001"]), ("R003", "MUL", ["B_001", "C001"]), ("R004", "SUB", ["R003", "R002"])],
        {"C001": "60"},
    ),
    "How much does James earn each week from both jobs?": meta(
        "How much does James earn each week from both jobs?",
        "weekly_total_earnings",
        {"A_001": "main_job_rate", "A_002": "main_job_hours", "B_001": "second_job_rate_reduction_percent", "B_002": "second_job_hours_ratio"},
        [("R001", "MUL", ["A_001", "A_002"]), ("R002", "DIV", ["B_001", "C001"]), ("R003", "SUB", ["C002", "R002"]), ("R004", "MUL", ["A_001", "R003"]), ("R005", "MUL", ["A_002", "B_002"]), ("R006", "MUL", ["R004", "R005"]), ("R007", "ADD", ["R001", "R006"])],
        {"C001": "100", "C002": "1"},
    ),
    "What balance remains after the 4 monthly payments?": meta(
        "What balance remains after the 4 monthly payments?",
        "remaining_laptop_balance",
        {"A_001": "laptop_cost", "A_002": "down_payment_percent", "A_003": "additional_down_payment", "B_001": "monthly_payment", "B_002": "payment_count"},
        [("R001", "DIV", ["A_002", "C001"]), ("R002", "MUL", ["A_001", "R001"]), ("R003", "ADD", ["R002", "A_003"]), ("R004", "SUB", ["A_001", "R003"]), ("R005", "MUL", ["B_001", "B_002"]), ("R006", "SUB", ["R004", "R005"])],
        {"C001": "100"},
    ),
    "How many packs must Roger buy?": meta(
        "How many packs must Roger buy?",
        "required_pack_count",
        {"A_001": "player_count", "A_002": "pouches_per_person", "B_001": "coach_count", "B_002": "helper_count", "B_003": "pouches_per_pack"},
        [("R001", "ADD", ["A_001", "B_001", "B_002"]), ("R002", "MUL", ["R001", "A_002"]), ("R003", "DIV", ["R002", "B_003"])],
    ),
    "How many kilograms does each of the last two people lose?": meta(
        "How many kilograms does each of the last two people lose?",
        "last_two_loss_each",
        {"A_001": "total_loss", "A_002": "first_person_loss", "B_001": "second_person_less_than_first", "B_002": "last_two_equal_count"},
        [("R001", "SUB", ["A_002", "B_001"]), ("R002", "SUB", ["A_001", "A_002", "R001"]), ("R003", "DIV", ["R002", "B_002"])],
    ),
    "How many vegetables does the garden produce altogether?": meta(
        "How many vegetables does the garden produce altogether?",
        "total_vegetables",
        {"A_001": "potato_count", "A_002": "cucumber_fewer_than_potatoes", "B_001": "pepper_to_cucumber_multiplier"},
        [("R001", "SUB", ["A_001", "A_002"]), ("R002", "MUL", ["R001", "B_001"]), ("R003", "ADD", ["A_001", "R001", "R002"])],
    ),
    "How many cans does Jennifer take home?": meta(
        "How many cans does Jennifer take home?",
        "take_home_cans",
        {"A_001": "initial_cans", "B_001": "jennifer_extra_cans_per_group", "B_002": "mark_group_size", "B_003": "mark_cans"},
        [("R001", "DIV", ["B_003", "B_002"]), ("R002", "MUL", ["R001", "B_001"]), ("R003", "ADD", ["A_001", "R002"])],
    ),
    "What was Irene's total income last week?": meta(
        "What was Irene's total income last week?",
        "last_week_total_income",
        {"A_001": "regular_pay", "A_002": "regular_hours", "B_001": "overtime_hourly_rate", "B_002": "hours_worked"},
        [("R001", "SUB", ["B_002", "A_002"]), ("R002", "MUL", ["R001", "B_001"]), ("R003", "ADD", ["A_001", "R002"])],
    ),
    "How much money does Winwin take home?": meta(
        "How much money does Winwin take home?",
        "take_home_money",
        {"A_001": "winnings", "A_002": "tax_percent", "B_001": "processing_fee"},
        [("R001", "DIV", ["A_002", "C001"]), ("R002", "MUL", ["A_001", "R001"]), ("R003", "SUB", ["A_001", "R002"]), ("R004", "SUB", ["R003", "B_001"])],
        {"C001": "100"},
    ),
    "How much money remains in John's piggy bank?": meta(
        "How much money remains in John's piggy bank?",
        "remaining_piggy_bank_money",
        {"A_001": "monthly_savings", "A_002": "saving_years", "B_001": "car_repair_cost"},
        [("R001", "MUL", ["A_002", "C001"]), ("R002", "MUL", ["A_001", "R001"]), ("R003", "SUB", ["R002", "B_001"])],
        {"C001": "12"},
    ),
    "How many plates are needed?": meta(
        "How many plates are needed?",
        "required_plate_count",
        {"A_001": "invited_people", "A_002": "additional_guest_fraction", "A_003": "additional_guest_per_person", "B_001": "course_count", "B_002": "plates_per_person_per_course"},
        [("R001", "MUL", ["A_001", "A_002", "A_003"]), ("R002", "ADD", ["A_001", "R001"]), ("R003", "MUL", ["R002", "B_001", "B_002"])],
    ),
    "How many hard hats remain in the truck?": meta(
        "How many hard hats remain in the truck?",
        "remaining_hard_hats",
        {"A_001": "initial_pink_hats", "A_002": "initial_green_hats", "A_003": "initial_yellow_hats", "B_001": "carl_removed_pink_hats", "B_002": "john_removed_pink_hats", "B_003": "john_green_hat_base_count", "B_005": "green_hat_removal_multiplier"},
        [("R001", "ADD", ["A_001", "A_002", "A_003"]), ("R002", "ADD", ["B_001", "B_002"]), ("R003", "MUL", ["B_003", "B_005"]), ("R004", "ADD", ["R002", "R003"]), ("R005", "SUB", ["R001", "R004"])],
    ),
}


def semantic_goal(item: dict[str, Any], oracle: dep.OraclePlan | None = None) -> str:
    shared_question = str(item.get("shared_question") or "").strip()
    if shared_question in GOLD_ORACLE_METADATA:
        return str(GOLD_ORACLE_METADATA[shared_question]["goal"])
    question = str(item.get("shared_question") or item.get("full") or "").lower()
    if "tomorrow" in question and "pages" in question:
        return "tomorrow_pages"
    if "still need" in question or "more money" in question:
        return "remaining_money"
    if "remain" in question or "left" in question:
        return "remaining_quantity"
    if "income" in question or "earn" in question:
        return "total_income"
    if "final weight" in question or "weight" in question:
        return "final_weight"
    if "shoes" in question and ("pay" in question or "cost" in question):
        return "shoes_cost"
    if "stamps" in question and ("altogether" in question or "total" in question):
        return "total_stamps"
    if "cans" in question:
        return "total_cans"
    if "total" in question or "altogether" in question:
        return "total_quantity"
    if "how many" in question:
        return "target_quantity"
    if "how much" in question:
        return "answer_amount"
    return "requested_value"


def build_oracle_node_namespace(program: dep.Program) -> dict[str, str]:
    reachable = program.reachable_nodes()
    mapping: dict[str, str] = {}
    fact_nodes = sorted(
        node_id for node_id in reachable
        if program.nodes[node_id].op == "FACT"
    )
    const_nodes = sorted(
        node_id for node_id in reachable
        if program.nodes[node_id].op == "CONST"
    )
    for index, node_id in enumerate(fact_nodes, 1):
        mapping[node_id] = f"F{index:03d}"
    for index, node_id in enumerate(const_nodes, 1):
        mapping[node_id] = f"C{index:03d}"

    ordered_derived: list[str] = []
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited or node_id not in reachable:
            return
        visited.add(node_id)
        node = program.nodes[node_id]
        for arg in node.args:
            child_id = str(arg)
            if child_id in program.nodes and program.nodes[child_id].op in dep.OPS:
                visit(child_id)
        if node.op in dep.OPS:
            ordered_derived.append(node_id)

    visit(program.answer_node)
    for index, node_id in enumerate(ordered_derived, 1):
        mapping[node_id] = f"D{index:03d}"
    return mapping


def canonical_fact_metadata(oracle: dep.OraclePlan, namespace: dict[str, str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    facts_by_id = {fact.fact_id: fact for fact in oracle.facts}
    for node_id, canonical_id in sorted(namespace.items(), key=lambda item: item[1]):
        node = oracle.program.nodes[node_id]
        if node.op != "FACT" or not node.fact_id:
            continue
        fact = facts_by_id.get(node.fact_id)
        rows[canonical_id] = {
            "source_fact_id": node.fact_id,
            "variable": fact.key if fact else node.label,
        }
    return rows


def canonical_constants(program: dep.Program, namespace: dict[str, str]) -> dict[str, str]:
    constants = {}
    for node_id, canonical_id in sorted(namespace.items(), key=lambda item: item[1]):
        node = program.nodes[node_id]
        if node.op == "CONST":
            constants[canonical_id] = str(node.value)
    return constants


def oracle_local_relations(program: dep.Program, namespace: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    derived_items = sorted(
        ((canonical_id, node_id) for node_id, canonical_id in namespace.items() if canonical_id.startswith("D")),
        key=lambda item: item[0],
    )
    for canonical_id, node_id in derived_items:
        node = program.nodes[node_id]
        rows.append({
            "relation_id": f"R{len(rows) + 1:03d}",
            "result_type": "DERIVED",
            "op": node.op,
            "inputs": [
                namespace[str(arg)] if namespace[str(arg)].startswith(("F", "C")) else "DERIVED"
                for arg in node.args
                if str(arg) in namespace
            ],
        })
    return rows


def oracle_topology(program: dep.Program, namespace: dict[str, str]) -> list[dict[str, Any]]:
    relation_by_derived = {
        node_id: f"R{index:03d}"
        for index, (canonical_id, node_id) in enumerate(sorted(
            ((canonical_id, node_id) for node_id, canonical_id in namespace.items() if canonical_id.startswith("D")),
            key=lambda item: item[0],
        ), 1)
    }
    def topo_id(node_id: str) -> str:
        canonical_id = namespace[node_id]
        if canonical_id.startswith("D"):
            return relation_by_derived[node_id]
        return canonical_id

    return [
        {"from": topo_id(source), "to": topo_id(target)}
        for source, target in sorted(program.edges(), key=lambda edge: (topo_id(edge[1]), topo_id(edge[0])))
        if source in namespace and target in namespace
    ]


def metadata_for_item(item: dict[str, Any]) -> dict[str, Any] | None:
    return GOLD_ORACLE_METADATA.get(str(item.get("shared_question") or "").strip())


def metadata_fact_namespace(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        source_fact_id: f"F{index:03d}"
        for index, source_fact_id in enumerate(metadata["relevant_facts"], 1)
    }


def metadata_relevant_facts(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fact_namespace = metadata_fact_namespace(metadata)
    return {
        canonical_id: {
            "source_fact_id": source_fact_id,
            "variable": metadata["fact_bindings"][source_fact_id],
            "relevant": True,
        }
        for source_fact_id, canonical_id in fact_namespace.items()
    }


def metadata_fact_binding(metadata: dict[str, Any]) -> dict[str, str]:
    fact_namespace = metadata_fact_namespace(metadata)
    return {
        fact_namespace[source_fact_id]: variable
        for source_fact_id, variable in metadata["fact_bindings"].items()
    }


def metadata_relevant_fact_ids(metadata: dict[str, Any]) -> dict[str, bool]:
    return {f"FACT[{source_fact_id}]": True for source_fact_id in metadata["relevant_facts"]}


def metadata_executable_fact_binding(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        f"FACT[{source_fact_id}]": variable
        for source_fact_id, variable in metadata["fact_bindings"].items()
    }


def metadata_relation_namespace(metadata: dict[str, Any]) -> dict[str, str]:
    keyed = []
    for relation in metadata["relations"]:
        content = json.dumps({
            "question": metadata["question"],
            "relation_id": relation["relation_id"],
            "op": relation["op"],
            "inputs": relation["inputs"],
        }, sort_keys=True)
        keyed.append((hashlib.sha256(content.encode("utf-8")).hexdigest(), relation["relation_id"]))
    return {
        original_id: f"R{index:03d}"
        for index, (_, original_id) in enumerate(sorted(keyed), 1)
    }


def metadata_relation_display_order(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = sorted(
        metadata["relations"],
        key=lambda relation: hashlib.sha256(json.dumps({
            "question": metadata["question"],
            "display": relation["relation_id"],
            "op": relation["op"],
            "inputs": relation["inputs"],
        }, sort_keys=True).encode("utf-8")).hexdigest(),
    )
    topo_ids = [relation["relation_id"] for relation in metadata["relations"]]
    display_ids = [relation["relation_id"] for relation in ordered]
    if len(ordered) >= 3 and display_ids == topo_ids:
        ordered = list(reversed(ordered))
    return ordered


def metadata_local_relations(metadata: dict[str, Any], mask_derived: bool) -> list[dict[str, Any]]:
    fact_namespace = metadata_fact_namespace(metadata)
    relation_ids = {relation["relation_id"] for relation in metadata["relations"]}
    relation_namespace = metadata_relation_namespace(metadata)
    constants = metadata.get("constants", {})
    rows = []
    for relation in metadata_relation_display_order(metadata):
        inputs = []
        for arg in relation["inputs"]:
            if arg in fact_namespace:
                inputs.append(f"FACT[{arg}]")
            elif arg in constants:
                inputs.append(f"CONST({constants[arg]})")
            elif arg in relation_ids:
                inputs.append("DERIVED" if mask_derived else relation_namespace[arg])
            else:
                inputs.append(arg)
        rows.append({
            "relation_id": relation_namespace[relation["relation_id"]],
            "result_type": "DERIVED",
            "op": relation["op"],
            "inputs": inputs,
        })
    return rows


def metadata_topology(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    fact_namespace = metadata_fact_namespace(metadata)
    relation_ids = {relation["relation_id"] for relation in metadata["relations"]}
    relation_namespace = metadata_relation_namespace(metadata)
    constants = set(metadata.get("constants", {}))
    edges = []
    for relation in metadata["relations"]:
        target = relation_namespace[relation["relation_id"]]
        for arg in relation["inputs"]:
            if arg in fact_namespace:
                source = fact_namespace[arg]
            elif arg in constants:
                source = arg
            elif arg in relation_ids:
                source = relation_namespace[arg]
            else:
                source = arg
            edges.append({"from": source, "to": target})
    return edges


def metadata_masked_topology(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    leaf_sources = []
    for source_fact_id in metadata["relevant_facts"]:
        leaf_sources.append(source_fact_id)
    for const_id in metadata.get("constants", {}):
        leaf_sources.append(const_id)
    leaf_namespace = {
        source_id: f"L{index:03d}"
        for index, source_id in enumerate(leaf_sources, 1)
    }
    final_relation = metadata["relations"][-1]["relation_id"]
    derived_namespace = {}
    next_derived = 1
    for relation in metadata["relations"]:
        original_id = relation["relation_id"]
        if original_id == final_relation:
            derived_namespace[original_id] = "ANSWER"
        else:
            derived_namespace[original_id] = f"D{next_derived:03d}"
            next_derived += 1

    relation_ids = set(derived_namespace)
    edges = []
    for relation in metadata["relations"]:
        target = derived_namespace[relation["relation_id"]]
        for arg in relation["inputs"]:
            if arg in leaf_namespace:
                source = leaf_namespace[arg]
            elif arg in relation_ids:
                source = derived_namespace[arg]
            else:
                source = arg
            edges.append({"from": source, "to": target})
    return edges


def build_semantic_reference_program(item: dict[str, Any], facts: list[dep.Fact]) -> dep.Program | None:
    metadata = metadata_for_item(item)
    if metadata is None:
        return None
    nodes = dep.make_fact_nodes(facts)
    for const_id, value in metadata.get("constants", {}).items():
        nodes[const_id] = dep.IRNode(const_id, "CONST", value=dep.decimal(value), provenance=())
    relation_namespace = metadata_relation_namespace(metadata)
    fact_node_by_source = {fact.fact_id: f"fact_{fact.fact_id}" for fact in facts}
    for relation in metadata["relations"]:
        args = []
        for arg in relation["inputs"]:
            if arg in fact_node_by_source:
                args.append(fact_node_by_source[arg])
            elif arg in relation_namespace:
                args.append(relation_namespace[arg])
            elif arg in metadata.get("constants", {}):
                args.append(arg)
            else:
                args.append(arg)
        child_nodes = [arg for arg in args if arg in nodes]
        provenance = tuple(sorted(set(p for arg in child_nodes for p in nodes[arg].provenance)))
        node_id = relation_namespace[relation["relation_id"]]
        nodes[node_id] = dep.IRNode(node_id, relation["op"], tuple(args), provenance=provenance)
    final_original = metadata["relations"][-1]["relation_id"]
    return dep.Program(nodes, relation_namespace[final_original], "semantic_oracle_reference").reachable_program()


def evaluate_semantic_metadata(item: dict[str, Any], facts: list[dep.Fact]) -> dep.ExecutionResult:
    program = build_semantic_reference_program(item, facts)
    if program is None:
        return dep.ExecutionResult(False, None, {}, ["missing semantic reference program"])
    return dep.eval_program(program, facts)


def build_oracle_context(level: str, item: dict[str, Any], oracle: dep.OraclePlan) -> OracleContext:
    explicit_metadata = metadata_for_item(item)
    if explicit_metadata is not None:
        empty = {
            "level": level,
            "visible_oracle_information": [],
            "goal": None,
            "relevant_facts": {},
            "fact_binding": {},
            "constants": {},
            "local_relations": [],
            "topology": [],
            "full_program": None,
        }
        if level == "O0":
            return OracleContext(**empty)
        if level == "O1":
            return OracleContext(**{**empty, "visible_oracle_information": ["goal"], "goal": explicit_metadata["goal"]})
        if level == "O2":
            return OracleContext(**{**empty, "visible_oracle_information": ["relevant_facts"], "relevant_facts": metadata_relevant_fact_ids(explicit_metadata)})
        if level == "O3":
            return OracleContext(**{**empty, "visible_oracle_information": ["fact_binding"], "fact_binding": metadata_executable_fact_binding(explicit_metadata)})
        if level == "O4":
            return OracleContext(**{
                **empty,
                "visible_oracle_information": ["local_relations"],
                "local_relations": metadata_local_relations(explicit_metadata, mask_derived=True),
            })
        if level == "O5":
            return OracleContext(**{**empty, "visible_oracle_information": ["topology"], "topology": metadata_masked_topology(explicit_metadata)})
        if level == "O6":
            return OracleContext(**{**empty, "visible_oracle_information": ["full_program"], "full_program": oracle.program.to_dict()})
        raise ValueError(f"unsupported oracle level: {level}")

    relevant = sorted(oracle.program.referenced_facts())
    namespace = build_oracle_node_namespace(oracle.program)
    fact_metadata = canonical_fact_metadata(oracle, namespace)
    constants = canonical_constants(oracle.program, namespace)
    relevant_source_ids = set(relevant)
    goal = semantic_goal(item, oracle)
    empty = {
        "level": level,
        "visible_oracle_information": ORACLE_VISIBLE_INFORMATION[level],
        "goal": None,
        "relevant_facts": {},
        "fact_binding": {},
        "constants": {},
        "local_relations": [],
        "topology": [],
        "full_program": None,
    }
    if level == "O0":
        return OracleContext(**empty)
    if level == "O1":
        return OracleContext(**{**empty, "goal": goal})
    if level == "O2":
        return OracleContext(**{
            **empty,
            "relevant_facts": {
                canonical_id: meta["source_fact_id"] in relevant_source_ids
                for canonical_id, meta in fact_metadata.items()
            },
        })
    if level == "O3":
        return OracleContext(**{
            **empty,
            "fact_binding": {
                canonical_id: meta["variable"]
                for canonical_id, meta in fact_metadata.items()
                if meta["source_fact_id"] in relevant_source_ids
            },
        })
    if level == "O4":
        return OracleContext(**{**empty, "local_relations": oracle_local_relations(oracle.program, namespace)})
    if level == "O5":
        return OracleContext(**{**empty, "topology": oracle_topology(oracle.program, namespace)})
    if level == "O6":
        return OracleContext(**{**empty, "full_program": oracle.program.to_dict()})
    raise ValueError(f"unsupported oracle level: {level}")


def choose_program_for_level(
    level: str,
    planner: str,
    facts: list[dep.Fact],
    item: dict[str, Any],
    oracle_context: OracleContext,
    proposal_backend: str,
    model: Any | None,
) -> dep.PlannerResult:
    planner_item = dict(item)
    planner_item["oracle_context"] = oracle_context.to_dict()
    result = dep.choose_program(planner, facts, planner_item, proposal_backend, model)
    result.trace.insert(0, {
        "step": -1,
        "oracle_level": level,
        "planner_received_oracle_context": oracle_context.to_dict(),
    })
    return result


def run_oracle_decomposition_case(
    item: dict[str, Any],
    qid: int,
    level: str,
    planner: str,
    proposal_backend: str,
    model: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    oracle = dep.build_oracle_plan(item, dep.gold_facts(item))
    oracle_context = build_oracle_context(level, item, oracle)
    facts = dep.gold_facts(item)

    if level == "O6":
        program = dep.align_oracle_to_facts(oracle, facts).reachable_program()
        result = dep.eval_program(program, facts)
        candidates = [result.answer] if result.answer is not None else []
        planner_trace = dep.backward_expansion_trace(program, facts)
        active_planner = "oracle"
        active_backend = "oracle"
    else:
        planner_result = choose_program_for_level(level, planner, facts, item, oracle_context, proposal_backend, model)
        program = planner_result.program
        candidates = planner_result.candidates
        planner_trace = planner_result.trace
        active_planner = planner
        active_backend = planner_result.proposal_backend
        result = dep.eval_program(program, facts)

    contract_ok, contract_errors = dep.contract_check(program, facts)
    semantic_reference = build_semantic_reference_program(item, facts)
    if semantic_reference is not None:
        f1 = dep.edge_f1(program, facts, semantic_reference, facts)
    else:
        f1 = dep.edge_f1(program, facts, oracle.program, oracle.facts)
    token_usage, llm_runtime_seconds, llm_call_count = dep.collect_trace_usage(planner_trace)
    return {
        "question_id": qid,
        "setting": f"oracle_decomposition_{level}",
        "oracle_level": level,
        "visible_oracle_information": ",".join(oracle_context.visible_oracle_information) or "none",
        "planner": active_planner,
        "proposal_backend": active_backend,
        "fact_correctness": dep.fact_correctness(facts, oracle.facts),
        **f1,
        "executable": result.ok and contract_ok,
        "candidate_emerged": any(dep.close_enough(c, oracle.answer) for c in candidates),
        "final_correct": dep.close_enough(result.answer, oracle.answer),
        "llm_call_count": llm_call_count,
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
        "llm_runtime_seconds": round(llm_runtime_seconds, 6),
        "case_runtime_seconds": round(time.perf_counter() - started, 6),
        "answer": str(result.answer) if result.answer is not None else "",
        "gold_answer": str(oracle.answer) if oracle.answer is not None else "",
        "errors": contract_errors + result.errors,
        "fact_conflicts": dep.detect_conflicts(facts),
        "planner_trace": planner_trace,
        "program": program.to_dict(),
        "oracle": oracle.to_dict(),
        "oracle_context": oracle_context.to_dict(),
        "planner_input": {
            "question": item.get("shared_question", ""),
            "facts": [fact.to_dict() for fact in facts],
            "oracle_context": oracle_context.to_dict(),
        },
        "facts": [fact.to_dict() for fact in facts],
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, run_config: dict[str, Any]) -> None:
    dep.write_outputs(rows, output_dir, run_config={
        **run_config,
        "script": Path(__file__).name,
        "settings": ORACLE_LEVEL_DESCRIPTIONS,
    })
    write_recovery_curve_outputs(rows, output_dir)


def build_recovery_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_rows = dep.summarize(rows)
    by_level = {row.get("oracle_level"): row for row in metric_rows if row.get("oracle_level") in ORACLE_LEVELS}
    recovery_rows = []
    for level in ORACLE_LEVELS:
        metric = by_level.get(level)
        if not metric:
            continue
        recovery_rows.append({
            "oracle_level": level,
            "label": RECOVERY_CURVE_LABELS[level],
            "visible_oracle_information": metric.get("visible_oracle_information", ""),
            "n": metric.get("n", 0),
            "final_accuracy": metric.get("final_accuracy", 0.0),
            "dependency_edge_f1": metric.get("dependency_edge_f1", 0.0),
            "executable_rate": metric.get("executable_rate", 0.0),
            "candidate_emergence_rate": metric.get("candidate_emergence_rate", 0.0),
        })
    return recovery_rows


def write_recovery_curve_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    recovery_rows = build_recovery_curve_rows(rows)
    if not recovery_rows:
        return
    csv_path = output_dir / "recovery_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(recovery_rows[0]))
        writer.writeheader()
        writer.writerows(recovery_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "recovery_curve.png.error.txt").write_text(
            f"matplotlib unavailable; recovery_curve.csv was written. Error: {exc}",
            encoding="utf-8",
        )
        return

    labels = [row["label"] for row in recovery_rows]
    accuracy = [float(row["final_accuracy"]) for row in recovery_rows]
    x_positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(x_positions, accuracy, color="#2563eb", marker="o", linewidth=2.0)
    for x, y in zip(x_positions, accuracy):
        ax.annotate(f"{y:.2%}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Oracle Decomposition Recovery Curve")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "recovery_curve.png", dpi=160)
    plt.close(fig)


def run_experiment(
    data_path: Path,
    output_dir: Path,
    limit: int,
    oracle_level: str,
    planner: str,
    proposal_backend: str,
    model: Any | None = None,
    incremental_write: bool = True,
    run_config: dict[str, Any] | None = None,
    question_whitelist: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = dep.read_records(data_path)
    if question_whitelist is not None:
        records = [
            item for item in records
            if str(item.get("shared_question") or "").strip() in question_whitelist
        ]
    records = records[:limit or None]
    levels = list(ORACLE_LEVELS) if oracle_level == "all" else [oracle_level]
    rows: list[dict[str, Any]] = []
    total = len(records) * len(levels)
    completed = 0
    config = run_config or {}
    for qid, item in enumerate(records, 1):
        print(f"[question {qid}/{len(records)}] {item.get('shared_question', '')}")
        for level in levels:
            completed += 1
            active_planner = "oracle" if level == "O6" else planner
            print(f"  [{completed}/{total}] oracle_level={level} / planner={active_planner} / backend={proposal_backend}")
            row = run_oracle_decomposition_case(item, qid, level, planner, proposal_backend, model)
            rows.append(row)
            if incremental_write:
                write_outputs(rows, output_dir, config)
                print(
                    f"    saved {len(rows)} rows -> {output_dir.resolve()} | "
                    f"final={row['final_correct']} edge_f1={row['edge_f1']:.3f} "
                    f"exec={row['executable']} emerged={row['candidate_emerged']}"
                )
    write_outputs(rows, output_dir, config)
    print(f"Usage totals: {{'llm_call_count': {sum(int(r.get('llm_call_count', 0) or 0) for r in rows)}, "
          f"'total_tokens': {sum(int(r.get('total_tokens', 0) or 0) for r in rows)}}}")
    return rows


class SmokeModel:
    def call(self, system: str, user: str, temperature: float = 0.0) -> tuple[str, dict[str, int], float]:
        fact_ids = re.findall(r"FACT\[([^\]]+)\]", user)
        fact_ids = list(dict.fromkeys(fact_ids))
        target_match = re.search(r"Unresolved target to expand:\s*([^\n]+)", user)
        target = target_match.group(1).strip() if target_match else "answer"
        prefix = "node"
        if "Local relations (unordered local relations" in user:
            prefix = "relation_node"
        elif "Topology interpretation:" in user:
            prefix = "topology_node"
        if not fact_ids:
            raw = "{}"
        else:
            index_match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", target)
            index = int(index_match.group(1)) if index_match else 0
            if target == "answer":
                next_node = f"{prefix}_1" if len(fact_ids) > 1 else f"FACT[{fact_ids[0]}]"
                raw = json.dumps({"target": target, "op": "ADD", "args": [next_node, f"FACT[{fact_ids[0]}]"]})
            elif index + 1 < len(fact_ids):
                raw = json.dumps({"target": target, "op": "ADD", "args": [f"{prefix}_{index + 1}", f"FACT[{fact_ids[index]}]"]})
            else:
                raw = json.dumps({"target": target, "op": "ADD", "args": [f"FACT[{fact_ids[-1]}]", "CONST(0)"]})
        tokens = max(1, len(user.split()))
        return raw, {
            "prompt_tokens": tokens,
            "completion_tokens": max(1, len(raw.split())),
            "total_tokens": tokens + max(1, len(raw.split())),
        }, 0.001


class ScriptedModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    def call(self, system: str, user: str, temperature: float = 0.0) -> tuple[str, dict[str, int], float]:
        self.calls.append({"system": system, "user": user})
        raw = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        usage = {
            "prompt_tokens": max(1, len(user.split())),
            "completion_tokens": max(1, len(raw.split())),
            "total_tokens": max(2, len(user.split()) + len(raw.split())),
        }
        return raw, usage, 0.001


def synthetic_protocol_item(include_relation_fact: bool = True) -> dict[str, Any]:
    b_facts = ["B has 3 tokens."]
    if include_relation_fact:
        b_facts.append("The multiplier is two.")
    return {
        "condition_A": "A has 2 tokens.",
        "condition_B": " ".join(b_facts),
        "shared_question": "How many tokens are counted?",
        "answer": "#### 7",
        "fact": {
            "A": ["A has 2 tokens."],
            "B": b_facts,
        },
    }


def protocol_regression_tests() -> None:
    source = Path("dependency_program_experiment.py").read_text(encoding="utf-8")
    assert '"op": "ADD|SUB|MUL|DIV"' not in source, "literal operator union remains in prompt"
    assert 'verified or proposal_backend == "llm"' not in source, "LLM verifier bypass remains"
    assert "failed_but_llm_only_accepts_no_fallback" not in source, "old verifier bypass trace remains"

    item = synthetic_protocol_item(include_relation_fact=True)
    facts = dep.gold_facts(item)
    premature_model = ScriptedModel([
        '{"target": "answer", "op": "ADD", "args": ["FACT[A_001]", "FACT[B_001]"]}',
        '{"target": "answer", "op": "ADD", "args": ["FACT[A_001]", "FACT[B_001]"]}',
    ])
    premature = dep.choose_program("LLM_strict_backward", facts, item, "llm", premature_model)
    assert not dep.eval_program(premature.program, facts).ok, "premature executable candidate was accepted"
    assert any(step.get("status") == "failed_to_generate_verified_program" for step in premature.trace), "premature failure status missing"
    assert any(
        isinstance(step.get("llm_meta"), dict) and step["llm_meta"].get("revision_context")
        for step in premature.trace
    ), "closed verifier failure did not enter revision"

    invalid_model = ScriptedModel(["not json"] * 4)
    failed = dep.choose_program("LLM_strict_backward", facts, item, "llm", invalid_model)
    assert not dep.eval_program(failed.program, facts).ok, "pure LLM failure returned executable fallback"
    assert failed.candidates == [], "pure LLM failure produced fake candidates"

    multistep_item = synthetic_protocol_item(include_relation_fact=False)
    multistep_facts = dep.gold_facts(multistep_item)
    multistep_model = ScriptedModel([
        '{"target": "answer", "op": "ADD", "args": ["node_1", "FACT[B_001]"]}',
        '{"target": "node_1", "op": "MUL", "args": ["FACT[A_001]", "CONST(2)"]}',
    ])
    multistep = dep.llm_strict_backward_search(multistep_model, multistep_item, multistep_facts, max_steps=4)
    assert multistep is not None, "multi-step backward expansion failed"
    assert dep.eval_program(multistep.program, multistep_facts).ok, "multi-step program is not executable"
    assert dep.contract_check(multistep.program, multistep_facts)[0], "multi-step program failed contract"
    assert dep.verify_llm_candidate(multistep.program, multistep_facts)[0], "multi-step program failed verifier"
    expanded = [step.get("target") for step in multistep.trace if "target" in step]
    assert expanded[:2] == ["answer", "node_1"], f"unexpected backward expansion order: {expanded}"

    records = dep.read_records(DEFAULT_DATA_PATH)
    item0 = records[0]
    oracle = dep.build_oracle_plan(item0, dep.gold_facts(item0))
    o5_context = build_oracle_context("O5", item0, oracle).to_dict()
    o5_prompt = dep.format_oracle_context_for_prompt({"oracle_context": o5_context})
    validate_oracle_context("O5", o5_context, o5_prompt)
    topology_text = "\n".join(f"{edge.get('from')} -> {edge.get('to')}" for edge in o5_context["topology"])
    assert not any(text in topology_text for text in ("ADD", "SUB", "MUL", "DIV", "FACT[", "CONST(")), "O5 topology data leaked semantics"
    print("Protocol regression tests: PASS")


def provided_information_labels(context: dict[str, Any]) -> list[str]:
    labels = ["facts"]
    for key, label in (
        ("goal", "goal"),
        ("relevant_facts", "relevant_facts"),
        ("fact_binding", "fact_binding"),
        ("constants", "constants"),
        ("local_relations", "local_relations"),
        ("topology", "topology"),
        ("full_program", "full_program"),
    ):
        if context.get(key):
            labels.append(label)
    return labels


def namespace_ids_from_context(context: dict[str, Any]) -> set[str]:
    ids = set(context.get("fact_binding") or {})
    ids.update(context.get("relevant_facts") or {})
    ids.update(context.get("constants") or {})
    for relation in context.get("local_relations") or []:
        if relation.get("relation_id"):
            ids.add(str(relation["relation_id"]))
        ids.update(str(arg) for arg in relation.get("inputs", []))
    for edge in context.get("topology") or []:
        ids.add(str(edge.get("from")))
        ids.add(str(edge.get("to")))
    ids.discard("")
    ids.discard("None")
    return ids


def assert_canonical_namespace(context: dict[str, Any], level: str) -> None:
    ids = namespace_ids_from_context(context)
    forbidden_fragments = ("oracle_step_", "derived_input_oracle_step_")
    assert not any(fragment in node_id for node_id in ids for fragment in forbidden_fragments), f"{level} leaked internal node id"
    assert all(
        node_id == "ANSWER"
        or node_id == "DERIVED"
        or node_id.startswith("FACT[")
        or node_id.startswith("CONST(")
        or (node_id.startswith(("F", "C", "D", "L", "R")) and node_id[1:].isdigit())
        for node_id in ids
    ), f"{level} has non-canonical ids: {sorted(ids)}"


def validate_oracle_context(level: str, context: dict[str, Any], prompt_context: str = "") -> None:
    empty_keys = ("goal", "relevant_facts", "fact_binding", "constants", "local_relations", "topology", "full_program")
    assert context.get("visible_oracle_information") == ORACLE_VISIBLE_INFORMATION[level], (
        f"{level} visible oracle information mismatch: {context.get('visible_oracle_information')}"
    )
    if level == "O0":
        assert not any(context[k] for k in empty_keys)
        assert not prompt_context
    if level == "O1":
        assert context["goal"]
        forbidden_goals = ("oracle", "oracle_step", "step_", "target_quantity", "answer_amount", "requested_value")
        assert not any(text in context["goal"] for text in forbidden_goals), f"{level} has non-gold semantic goal: {context['goal']}"
        assert not any(context[k] for k in ("relevant_facts", "fact_binding", "constants", "local_relations", "topology", "full_program"))
        assert "Goal:" in prompt_context and "Relevant facts:" not in prompt_context
        assert all(section not in prompt_context for section in ("Fact bindings:", "Local relations", "Topology:"))
    if level == "O2":
        assert context["relevant_facts"]
        assert not any(context[k] for k in ("goal", "fact_binding", "constants", "local_relations", "topology", "full_program"))
        assert all(fid.startswith("FACT[") for fid in context["relevant_facts"])
        assert "Relevant facts:" in prompt_context
        assert all(section not in prompt_context for section in ("Goal:", "Fact bindings:", "Local relations", "Topology:"))
    if level == "O3":
        assert context["fact_binding"]
        assert not any(context[k] for k in ("goal", "relevant_facts", "constants", "local_relations", "topology", "full_program"))
        assert all(fid.startswith("FACT[") for fid in context["fact_binding"])
        assert_canonical_namespace(context, level)
        assert "Fact bindings:" in prompt_context
        assert all(section not in prompt_context for section in ("Goal:", "Relevant facts:", "Local relations", "Topology:", "Executable alias map:"))
    if level == "O4":
        assert context["local_relations"]
        assert not any(context[k] for k in ("goal", "relevant_facts", "fact_binding", "constants", "topology", "full_program"))
        assert_canonical_namespace(context, level)
        assert "unordered local relations" in prompt_context.lower()
        assert any(relation["op"] in dep.OPS for relation in context["local_relations"])
        for relation in context["local_relations"]:
            assert str(relation.get("relation_id", "")).startswith("R")
            assert "result" not in relation
            for arg in relation.get("inputs", []):
                assert str(arg).startswith("FACT[") or str(arg).startswith("CONST(") or arg == "DERIVED"
                assert arg == "DERIVED" or not str(arg).startswith(("D", "R"))
        assert "Local relations" in prompt_context
        assert all(section not in prompt_context for section in ("Goal:", "Relevant facts:", "Fact bindings:", "Topology:", "Executable alias map:"))
    if level == "O5":
        assert context["topology"]
        assert not any(context[k] for k in ("goal", "relevant_facts", "fact_binding", "constants", "local_relations", "full_program"))
        assert_canonical_namespace(context, level)
        ids = namespace_ids_from_context(context)
        assert all(node_id == "ANSWER" or (node_id.startswith(("L", "D")) and node_id[1:].isdigit()) for node_id in ids)
        assert "Topology:" in prompt_context and "full_program" not in prompt_context
        topology_text = "\n".join(f"{edge.get('from')} -> {edge.get('to')}" for edge in context["topology"])
        assert not any(text in topology_text for text in ("ADD", "SUB", "MUL", "DIV", "FACT[", "CONST("))
        assert all(section not in prompt_context for section in ("Goal:", "Relevant facts:", "Fact bindings:", "Local relations", "Executable alias map:"))
    if level == "O6":
        assert context["full_program"]


def smoke_test() -> None:
    output_dir = DEFAULT_OUTPUT_DIR / "smoke"
    smoke_questions = {
        "How many pages should Julie read tomorrow?",
        "How many minutes does Carolyn practice in 4 weeks?",
        "How much does James earn each week from both jobs?",
    }
    rows = run_experiment(
        DEFAULT_DATA_PATH,
        output_dir,
        limit=0,
        oracle_level="all",
        planner="LLM_strict_backward",
        proposal_backend="llm",
        model=SmokeModel(),
        incremental_write=True,
        run_config={"mode": "oracle_decomposition", "smoke_test": True},
        question_whitelist=smoke_questions,
    )
    assert {row["planner_input"]["question"] for row in rows} == smoke_questions, "smoke question set mismatch"
    seen = {row["oracle_level"] for row in rows}
    assert seen == set(ORACLE_LEVELS), f"not all oracle levels ran: {sorted(seen)}"
    assert all(row["program"]["answer_node"] in row["program"]["nodes"] for row in rows if row["oracle_level"] == "O6"), "O6 missing answer node"
    assert all(row["planner"] != "oracle" for row in rows if row["oracle_level"] != "O6"), "O0-O5 must use planner"
    assert {row["planner"] for row in rows if row["oracle_level"] != "O6"} == {"LLM_strict_backward"}, "O0-O5 planner path changed"
    assert any(row["oracle_level"] == "O6" and row["executable"] for row in rows), "O6 oracle program did not execute"
    for qid in {row["question_id"] for row in rows}:
        generated_rows = [row for row in rows if row["question_id"] == qid and row["oracle_level"] != "O6"]
        questions = {row["planner_input"]["question"] for row in generated_rows}
        facts_json = {
            json.dumps(row["planner_input"]["facts"], sort_keys=True, ensure_ascii=False)
            for row in generated_rows
        }
        assert len(questions) == 1, f"q{qid} question changed across O0-O5"
        assert len(facts_json) == 1, f"q{qid} facts changed across O0-O5"
    for row in rows:
        context = row["planner_input"]["oracle_context"]
        level = row["oracle_level"]
        prompt_context = ""
        llm_steps = [step for step in row["planner_trace"] if isinstance(step.get("llm_meta"), dict)]
        if level != "O6":
            assert llm_steps, f"{level} has no LLM trace"
            assert row["llm_call_count"] == len(llm_steps), f"{level} lost failed LLM call count"
            assert row["total_tokens"] > 0 and row["llm_runtime_seconds"] > 0, f"{level} lost token/runtime usage"
            prompt_context = llm_steps[0]["llm_meta"].get("oracle_prompt_context", "")
            visible = llm_steps[0]["llm_meta"].get("visible_oracle_information", [])
            assert visible == context["visible_oracle_information"], f"{level} prompt audit mismatch"
            forbidden = ("full_program", "gold_answer", "Gold answer", "oracle program")
            assert not any(text in prompt_context for text in forbidden), f"{level} leaked oracle answer/program field"
        validate_oracle_context(level, context, prompt_context)
    first_by_level = {level: next(row for row in rows if row["oracle_level"] == level) for level in ORACLE_LEVELS}
    print("Oracle information by level:")
    for level in ORACLE_LEVELS:
        context = first_by_level[level]["planner_input"]["oracle_context"]
        labels = provided_information_labels(context)
        print(f"  {level}: {labels}")
    print("Oracle decomposition smoke: PASS")
    print(f"Smoke rows: {len(rows)}")
    print(f"Smoke output: {output_dir}")


def structural_audit_all_questions(data_path: Path = DEFAULT_DATA_PATH) -> None:
    records = dep.read_records(data_path)
    if data_path.resolve() == DEFAULT_DATA_PATH.resolve():
        missing = [
            (index, item.get("shared_question", ""))
            for index, item in enumerate(records, 1)
            if str(item.get("shared_question", "")).strip() not in GOLD_SEMANTIC_GOALS
        ]
        assert not missing, f"missing explicit gold semantic goals: {missing}"
    for qid, item in enumerate(records, 1):
        oracle = dep.build_oracle_plan(item, dep.gold_facts(item))
        for level in ORACLE_LEVELS:
            context = build_oracle_context(level, item, oracle).to_dict()
            prompt_context = dep.format_oracle_context_for_prompt({"oracle_context": context})
            validate_oracle_context(level, context, prompt_context)

            namespace = build_oracle_node_namespace(oracle.program)
            if level == "O5":
                explicit_metadata = metadata_for_item(item)
                expected_topology = (
                    metadata_masked_topology(explicit_metadata)
                    if explicit_metadata is not None
                    else oracle_topology(oracle.program, namespace)
                )
                assert context["topology"] == expected_topology, f"q{qid} O5 topology mismatch"
    print(f"Structural audit: PASS ({len(records)} questions x {len(ORACLE_LEVELS)} levels)")


def semantic_oracle_audit(data_path: Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    records = dep.read_records(data_path)
    missing_metadata = []
    numeric_collisions = []
    graph_execution_pass = 0
    relation_order_scrambled = 0
    relevant_usage_failures = []
    for qid, item in enumerate(records, 1):
        question = str(item.get("shared_question") or "").strip()
        metadata = metadata_for_item(item)
        if metadata is None:
            missing_metadata.append((qid, question))
            continue

        facts = dep.gold_facts(item)
        facts_by_id = {fact.fact_id: fact for fact in facts}
        source_ids = set(facts_by_id)
        relevant_ids = set(metadata["relevant_facts"])
        binding_ids = set(metadata["fact_bindings"])
        assert relevant_ids <= source_ids, f"q{qid} metadata references missing source facts: {sorted(relevant_ids - source_ids)}"
        assert relevant_ids == binding_ids, f"q{qid} relevant facts and bindings diverge"
        for source_fact_id, variable in metadata["fact_bindings"].items():
            fact = facts_by_id[source_fact_id]
            assert variable and isinstance(variable, str), f"q{qid} empty binding for {source_fact_id}"
            assert fact.content, f"q{qid} empty source content for {source_fact_id}"

        relation_ids = {relation["relation_id"] for relation in metadata["relations"]}
        constants = set(metadata.get("constants", {}))
        used_relevant_ids = set()
        for relation in metadata["relations"]:
            assert relation["relation_id"].startswith("R"), f"q{qid} non-canonical relation id"
            assert relation["op"] in dep.OPS, f"q{qid} illegal oracle operator: {relation['op']}"
            for arg in relation["inputs"]:
                assert arg in relevant_ids or arg in constants or arg in relation_ids, f"q{qid} relation arg has no semantic provenance: {arg}"
                if arg in relevant_ids:
                    assert arg in facts_by_id, f"q{qid} relation source fact missing: {arg}"
                    used_relevant_ids.add(arg)
        unused = sorted(relevant_ids - used_relevant_ids)
        if unused:
            relevant_usage_failures.append({"question_id": qid, "unused_relevant_facts": unused})
        assert not unused, f"q{qid} relevant facts unused by semantic graph: {unused}"

        topo_relation_order = [metadata_relation_namespace(metadata)[relation["relation_id"]] for relation in metadata["relations"]]
        o4_relation_order = [relation["relation_id"] for relation in metadata_local_relations(metadata, mask_derived=True)]
        if len(o4_relation_order) >= 3:
            assert o4_relation_order != topo_relation_order, f"q{qid} O4 relation order leaks topological order"
            relation_order_scrambled += 1
        for relation in metadata_local_relations(metadata, mask_derived=True):
            assert not any(str(arg).startswith("R") for arg in relation["inputs"]), f"q{qid} O4 leaks relation-to-relation wiring"

        execution = evaluate_semantic_metadata(item, facts)
        gold = dep.decimal(item.get("answer"))
        assert execution.ok, f"q{qid} semantic graph did not execute: {execution.errors}"
        assert dep.close_enough(execution.answer, gold), f"q{qid} semantic graph answer mismatch: {execution.answer} != {gold}"
        graph_execution_pass += 1

        by_value: dict[str, list[dep.Fact]] = {}
        for fact in facts:
            if fact.value is not None:
                by_value.setdefault(str(fact.value), []).append(fact)
        for value, group in sorted(by_value.items()):
            variables = {metadata["fact_bindings"].get(fact.fact_id) for fact in group if fact.fact_id in metadata["fact_bindings"]}
            if len(group) > 1:
                relations_using = {}
                for fact in group:
                    relations_using[fact.fact_id] = [
                        relation["relation_id"]
                        for relation in metadata["relations"]
                        if fact.fact_id in relation["inputs"]
                    ]
                numeric_collisions.append({
                    "question_id": qid,
                    "value": value,
                    "facts": [
                        {
                            "fact_id": fact.fact_id,
                            "type": fact.type,
                            "source": fact.source,
                            "variable": metadata["fact_bindings"].get(fact.fact_id, ""),
                            "content": fact.content,
                            "relations_using_this_fact": relations_using.get(fact.fact_id, []),
                        }
                        for fact in group
                    ],
                    "bound_variables": sorted(v for v in variables if v),
                })

    assert not missing_metadata, f"missing explicit gold oracle metadata: {missing_metadata}"

    julie = GOLD_ORACLE_METADATA["How many pages should Julie read tomorrow?"]
    assert julie["fact_bindings"]["A_002"] == "today_multiplier"
    assert julie["fact_bindings"]["B_002"] == "remaining_fraction"
    julie_r4 = next(relation for relation in julie["relations"] if relation["relation_id"] == "R004")
    assert "B_002" in julie_r4["inputs"] and "A_002" not in julie_r4["inputs"]

    james = GOLD_ORACLE_METADATA["How much does James earn each week from both jobs?"]
    assert james["fact_bindings"]["B_001"] == "second_job_rate_reduction_percent"
    assert james["fact_bindings"]["B_002"] == "second_job_hours_ratio"
    james_rate = next(relation for relation in james["relations"] if relation["relation_id"] == "R002")
    james_hours = next(relation for relation in james["relations"] if relation["relation_id"] == "R005")
    assert "B_001" in james_rate["inputs"]
    assert "B_002" in james_hours["inputs"]

    tara = GOLD_ORACLE_METADATA["What balance remains after the 4 monthly payments?"]
    assert tara["fact_bindings"]["A_002"] == "down_payment_percent"
    assert tara["fact_bindings"]["A_003"] == "additional_down_payment"
    tara_percent = next(relation for relation in tara["relations"] if relation["relation_id"] == "R001")
    tara_absolute = next(relation for relation in tara["relations"] if relation["relation_id"] == "R003")
    assert "A_002" in tara_percent["inputs"]
    assert "A_003" in tara_absolute["inputs"]

    result = {
        "metadata_coverage": f"{len(records)}/{len(records)}",
        "numeric_collision_count": len(numeric_collisions),
        "numeric_collisions": numeric_collisions,
        "semantic_graph_execution": f"{graph_execution_pass}/{len(records)}",
        "operator_vocab": "PASS",
        "o4_topology_leakage_audit": f"PASS ({relation_order_scrambled} order checks)",
        "relevant_fact_usage_audit": "PASS",
        "julie_twice_half": {
            "today_multiplier": julie["fact_bindings"]["A_002"],
            "remaining_fraction": julie["fact_bindings"]["B_002"],
            "final_relation_inputs": julie_r4["inputs"],
        },
        "james_relation_provenance": {
            "rate_reduction_relation": james_rate,
            "hours_ratio_relation": james_hours,
        },
        "tara_numeric_collision_regression": {
            "down_payment_percent": tara["fact_bindings"]["A_002"],
            "additional_down_payment": tara["fact_bindings"]["A_003"],
            "percentage_relation": tara_percent,
            "absolute_addition_relation": tara_absolute,
        },
    }
    print(f"Semantic oracle audit: PASS ({len(records)} questions, {len(numeric_collisions)} numeric collision cases)")
    print(f"SEMANTIC_GRAPH_EXECUTION={graph_execution_pass}/{len(records)} PASS")
    print("ORACLE_OPERATOR_VOCAB=PASS")
    print("EXECUTABLE_ALIAS_MAP=PASS")
    print("O4_TOPOLOGY_LEAKAGE_AUDIT=PASS")
    print("NUMERIC_COLLISION_AUDIT=PASS")
    print("RELEVANT_FACT_USAGE_AUDIT=PASS")
    return result


def load_model_if_needed(args: argparse.Namespace) -> Any | None:
    if args.proposal_backend not in {"llm", "hybrid"}:
        print("Running deterministic offline oracle-decomposition experiment.")
        return None
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
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle decomposition for dependency-program recovery.")
    parser.add_argument("--mode", choices=["oracle_decomposition"], default="oracle_decomposition")
    parser.add_argument("--oracle-level", choices=[*ORACLE_LEVELS, "all"], default="all")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--planner", choices=[*dep.PLANNERS, "all"], default="LLM_strict_backward")
    parser.add_argument("--proposal-backend", choices=dep.PROPOSAL_BACKENDS, default="llm")
    parser.add_argument("--model-path", default=str(dep.DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--protocol-regression-test", action="store_true")
    parser.add_argument("--structural-audit", action="store_true")
    parser.add_argument("--semantic-audit", action="store_true")
    parser.add_argument(
        "--no-incremental-write",
        action="store_true",
        help="Disable checkpointing after every case. Default saves metrics/cases immediately after each case.",
    )
    args = parser.parse_args()
    if args.planner == "all":
        raise SystemExit("oracle_decomposition expects one planner at a time; pass one of: " + ", ".join(dep.PLANNERS))
    if args.smoke_test:
        smoke_test()
        return
    if args.protocol_regression_test:
        protocol_regression_tests()
        return
    if args.structural_audit:
        structural_audit_all_questions(Path(args.data_path))
        return
    if args.semantic_audit:
        semantic_oracle_audit(Path(args.data_path))
        return

    model = load_model_if_needed(args)
    run_config = {
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
    }
    rows = run_experiment(
        Path(args.data_path),
        Path(args.output_dir),
        args.limit,
        args.oracle_level,
        args.planner,
        args.proposal_backend,
        model=model,
        incremental_write=not args.no_incremental_write,
        run_config=run_config,
    )
    print(f"Wrote {len(rows)} rows to {Path(args.output_dir).resolve()}")
    print(f"Metrics: {Path(args.output_dir).resolve() / 'metrics.csv'}")
    print(f"Cases: {Path(args.output_dir).resolve() / 'cases.jsonl'}")


if __name__ == "__main__":
    main()
