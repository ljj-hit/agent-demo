"""Oracle-decomposition experiment for dependency-program recovery.

This module reuses the dependency-program experiment framework and varies only
the oracle information visible to the planner.  O6 is the upper-bound condition
that executes the full oracle program; O0-O5 still generate a dependency
program through planner paths.
"""
from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class OracleContext:
    level: str
    visible_oracle_information: list[str]
    goal: str | None
    relevant_facts: dict[str, bool]
    fact_binding: dict[str, str]
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
            "local_relations": self.local_relations,
            "topology": self.topology,
            "full_program": self.full_program,
        }


def semantic_goal(item: dict[str, Any], oracle: dep.OraclePlan) -> str:
    question = str(item.get("shared_question") or item.get("full") or "").lower()
    if "remain" in question or "left" in question or "still need" in question:
        return "remaining_amount"
    if "total" in question:
        return "total_amount"
    if "income" in question or "earn" in question:
        return "total_income"
    if "weight" in question:
        return "final_weight"
    if "cans" in question:
        return "total_cans"
    return oracle.program.answer_node


def oracle_local_relations(program: dep.Program) -> list[dict[str, Any]]:
    rows = []
    facts_by_node = {
        node_id: node.label or (node.fact_id or node_id)
        for node_id, node in program.nodes.items()
        if node.op == "FACT"
    }
    for index, node_id in enumerate(sorted(program.reachable_nodes()), 1):
        node = program.nodes[node_id]
        if node.op not in dep.OPS:
            continue
        rows.append({
            "relation_id": f"R{index:03d}",
            "result": node.label or f"derived_{index:03d}",
            "op": node.op,
            "inputs": [
                facts_by_node.get(str(arg), f"derived_input_{arg}")
                for arg in node.args
                if str(arg) in program.nodes
            ],
        })
    return rows


def oracle_topology(program: dep.Program) -> list[dict[str, Any]]:
    reachable = program.reachable_nodes()
    derived = sorted(node_id for node_id in reachable if program.nodes[node_id].op in dep.OPS)
    leaves = sorted(node_id for node_id in reachable if program.nodes[node_id].op in {"FACT", "CONST"})
    node_map = {node_id: f"D{idx:03d}" for idx, node_id in enumerate(derived, 1)}
    node_map.update({node_id: f"L{idx:03d}" for idx, node_id in enumerate(leaves, 1)})
    return [{"from": node_map[source], "to": node_map[target]} for source, target in sorted(program.edges())]


def build_oracle_context(level: str, item: dict[str, Any], oracle: dep.OraclePlan) -> OracleContext:
    relevant = sorted(oracle.program.referenced_facts())
    facts_by_id = {fact.fact_id: fact for fact in oracle.facts}
    all_fact_ids = sorted(fact.fact_id for fact in oracle.facts)
    goal = semantic_goal(item, oracle)
    if level == "O6":
        return OracleContext(
            level=level,
            visible_oracle_information=["goal", "relevant_facts", "fact_binding", "local_relations", "topology", "full_program"],
            goal=goal,
            relevant_facts={fid: fid in relevant for fid in all_fact_ids},
            fact_binding={fid: facts_by_id[fid].key for fid in relevant if fid in facts_by_id},
            local_relations=oracle_local_relations(oracle.program),
            topology=oracle_topology(oracle.program),
            full_program=oracle.program.to_dict(),
        )
    level_index = ORACLE_LEVELS.index(level)
    return OracleContext(
        level=level,
        visible_oracle_information=[
            name for cutoff, name in (
                (1, "goal"),
                (2, "relevant_facts"),
                (3, "fact_binding"),
                (4, "local_relations"),
                (5, "topology"),
            )
            if level_index >= cutoff
        ],
        goal=semantic_goal(item, oracle) if level_index >= 1 else None,
        relevant_facts={fid: fid in relevant for fid in all_fact_ids} if level_index >= 2 else {},
        fact_binding={fid: facts_by_id[fid].key for fid in relevant if fid in facts_by_id} if level_index >= 3 else {},
        local_relations=oracle_local_relations(oracle.program) if level_index >= 4 else [],
        topology=oracle_topology(oracle.program) if level_index >= 5 else [],
        full_program=None,
    )


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
) -> list[dict[str, Any]]:
    records = dep.read_records(data_path)[:limit or None]
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
        tokens = max(1, len(user.split()))
        return "{}", {
            "prompt_tokens": tokens,
            "completion_tokens": 1,
            "total_tokens": tokens + 1,
        }, 0.001


def provided_information_labels(context: dict[str, Any]) -> list[str]:
    labels = ["facts"]
    for key, label in (
        ("goal", "goal"),
        ("relevant_facts", "relevant_facts"),
        ("fact_binding", "fact_binding"),
        ("local_relations", "local_relations"),
        ("topology", "topology"),
        ("full_program", "full_program"),
    ):
        if context.get(key):
            labels.append(label)
    return labels


def smoke_test() -> None:
    output_dir = DEFAULT_OUTPUT_DIR / "smoke"
    rows = run_experiment(
        DEFAULT_DATA_PATH,
        output_dir,
        limit=3,
        oracle_level="all",
        planner="LLM_strict_backward",
        proposal_backend="llm",
        model=SmokeModel(),
        incremental_write=True,
        run_config={"mode": "oracle_decomposition", "smoke_test": True},
    )
    seen = {row["oracle_level"] for row in rows}
    assert seen == set(ORACLE_LEVELS), f"not all oracle levels ran: {sorted(seen)}"
    assert all(row["program"]["answer_node"] in row["program"]["nodes"] for row in rows), "missing answer node"
    assert all(row["planner"] != "oracle" for row in rows if row["oracle_level"] != "O6"), "O0-O5 must use planner"
    assert {row["planner"] for row in rows if row["oracle_level"] != "O6"} == {"LLM_strict_backward"}, "O0-O5 planner path changed"
    assert any(row["oracle_level"] == "O6" and row["executable"] for row in rows), "O6 oracle program did not execute"
    for row in rows:
        context = row["planner_input"]["oracle_context"]
        level = row["oracle_level"]
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
        if level == "O0":
            assert not any(context[k] for k in ("goal", "relevant_facts", "fact_binding", "local_relations", "topology", "full_program"))
            assert all(not step["llm_meta"].get("oracle_prompt_context") for step in llm_steps)
        if level == "O1":
            assert context["goal"] and not any(context[k] for k in ("relevant_facts", "fact_binding", "local_relations", "topology", "full_program"))
            assert "Goal:" in prompt_context and "Relevant facts:" not in prompt_context
        if level == "O2":
            assert context["goal"] and context["relevant_facts"] and not any(context[k] for k in ("fact_binding", "local_relations", "topology", "full_program"))
            assert "Relevant facts:" in prompt_context and "Fact bindings:" not in prompt_context
        if level == "O3":
            assert context["goal"] and context["relevant_facts"] and context["fact_binding"]
            assert not any(context[k] for k in ("local_relations", "topology", "full_program"))
            assert "Fact bindings:" in prompt_context and "Local relations:" not in prompt_context
        if level == "O4":
            assert context["goal"] and context["relevant_facts"] and context["fact_binding"] and context["local_relations"]
            assert not any(context[k] for k in ("topology", "full_program"))
            assert "Local relations:" in prompt_context and "Topology:" not in prompt_context
        if level == "O5":
            assert context["goal"] and context["relevant_facts"] and context["fact_binding"]
            assert context["local_relations"] and context["topology"] and not context["full_program"]
            assert "Topology:" in prompt_context and "full_program" not in prompt_context
        if level == "O6":
            assert context["full_program"]
    first_by_level = {level: next(row for row in rows if row["oracle_level"] == level) for level in ORACLE_LEVELS}
    previous: set[str] = set()
    print("Oracle information by level:")
    for level in ORACLE_LEVELS:
        context = first_by_level[level]["planner_input"]["oracle_context"]
        labels = provided_information_labels(context)
        current = set(labels)
        print(f"  {level}: {labels}")
        assert previous < current, f"{level} does not strictly add oracle information"
        previous = current
    print("Oracle decomposition smoke: PASS")
    print(f"Smoke rows: {len(rows)}")
    print(f"Smoke output: {output_dir}")


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
