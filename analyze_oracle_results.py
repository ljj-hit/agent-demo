#!/usr/bin/env python3
"""
Analysis wrapper for Oracle Intervention Experiment.
Reads traces and generates:
  1. Per-setting metrics table
  2. Recovery delta analysis
  3. Causal bottleneck interpretation
  4. Comprehensive markdown report

Usage:
  python analyze_oracle_results.py <output_dir>
  python analyze_oracle_results.py outputs_oracle_intervention/20260806_120000
"""

import json, csv, os, re, sys
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

# ── Reuse core utilities ──
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from run_hidden_gsm8k import (
    extract_answer, decimal, equivalent, candidate_appearances,
    atomic_facts, fact_is_public, collect_events, blank_usage, add_usage
)

SETTING_LABELS = {
    "free_discussion":          "0. Free Discussion (baseline: private facts)",
    "oracle_disclosure":        "1. Oracle Disclosure (raw facts injected)",
    "oracle_canonical_state":   "2. Oracle Canonical State (normalized table)",
    "canonical_state_fresh":    "3. Canonical State + Fresh Solver (no history)",
    "oracle_plan":              "4. Oracle Plan (equation structure)",
    "oracle_candidate":         "5. Oracle Candidate (correct answer as candidate)",
}

SETTING_SHORT = {
    "free_discussion":          "Free Discussion",
    "oracle_disclosure":        "Oracle Disclosure",
    "oracle_canonical_state":   "Oracle Canonical State",
    "canonical_state_fresh":    "Canonical State + Fresh",
    "oracle_plan":              "Oracle Plan",
    "oracle_candidate":         "Oracle Candidate",
}

# Detection order of the 6 settings in recovery order
SETTING_ORDER = [
    "free_discussion",
    "oracle_disclosure",
    "oracle_canonical_state",
    "canonical_state_fresh",
    "oracle_plan",
    "oracle_candidate",
]

def load_traces(traces_path: Path) -> list[dict]:
    """Load traces from JSON file."""
    with open(traces_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_fact_distortion_rate(trace: dict, all_required_facts: list[str]) -> float:
    """Check fraction of required fact numbers missing from agent outputs."""
    disc = trace.get("discussion", {})
    events = disc.get("discussion_events", [])
    all_output = " ".join(evt.get("raw_output", "") for evt in events)
    # Also include solver finals
    for side in ("a", "b"):
        sf = disc.get("solver_finals", {}).get(side, {})
        all_output += " " + sf.get("raw_output", "")

    if not all_required_facts or not all_output.strip():
        return 0.0

    total_nums = set()
    found_nums = set()
    for fact in all_required_facts:
        nums = set(re.findall(r"\b(\d+(?:\.\d+)?)\b", fact))
        total_nums.update(nums)
        for n in nums:
            if n in all_output:
                found_nums.add(n)

    if not total_nums:
        return 0.0
    return len(total_nums - found_nums) / len(total_nums)

def compute_partial_answer_rate(trace: dict) -> float:
    """Check if agents produced calculations with intermediate numbers."""
    disc = trace.get("discussion", {})
    events = disc.get("discussion_events", [])
    count_with_calcs = 0
    total = 0
    for evt in events:
        raw = evt.get("raw_output", "")
        if raw:
            total += 1
            # Check for calculation patterns: =, equals, gives, yields, <<...>>
            if re.search(r"(?:=\s*\d+|equals?\s*\d+|<<[\d\s+\-*/.=]+>>|\d+\s*[\+\-\*/]\s*\d+)", raw):
                count_with_calcs += 1
    return count_with_calcs / max(1, total)

def compute_undetermined_ratio(trace: dict) -> float:
    """Check ratio of undetermined answers across discussion events."""
    disc = trace.get("discussion", {})
    events = disc.get("discussion_events", [])
    undetermined = 0
    total = 0
    for evt in events:
        raw = evt.get("raw_output", "")
        if not raw:
            continue
        total += 1
        # Check Current answer / Final answer
        for prefix in ["Current answer", "Final answer"]:
            m = re.search(rf"(?im)^{re.escape(prefix)}\s*[:：]\s*(.+)", raw)
            if m:
                ans = m.group(1).strip().lower()
                if ans in ("undetermined", "undetermined.", "unknown", "insufficient", "n/a", ""):
                    undetermined += 1
                    break
    return undetermined / max(1, total)

def compute_setting_metrics(traces: list[dict]) -> dict:
    """Compute all 8 metrics for a setting."""
    n = len(traces)
    if n == 0:
        return {"n": 0}

    # Group by question
    by_q = defaultdict(list)
    for t in traces:
        by_q[t["question_id"]].append(t)

    # 1. Semantic accuracy (per-trace)
    semantic_correct = sum(1 for t in traces if t.get("semantic_correct", False))

    # 2. Correct candidate emergence (per-question, any seed)
    cand_qs = 0
    for qid, tlist in by_q.items():
        gold = tlist[0].get("gold_answer", "")
        if any(
            any(str(a.get("answer", "")) == str(gold) for a in t.get("candidate_appearances", []))
            for t in tlist
        ):
            cand_qs += 1

    # 3. Format compliance
    format_ok = sum(1 for t in traces if t.get("format_compliant", False))

    # 4. Answer-reason consistency (only among checkable traces)
    checkable = 0
    consistent = 0
    for t in traces:
        if t.get("answer_reason_checkable"):
            checkable += 1
            if t.get("answer_reason_consistent") is True:
                consistent += 1

    # 5. Fact distortion rate (average across traces)
    # Need to load questions data for required facts
    questions_path = Path("data/20.json")
    questions = json.loads(questions_path.read_text(encoding="utf-8")) if questions_path.exists() else []
    q_facts = {}
    for q in questions:
        sq = q.get("shared_question", "")
        if sq:
            q_facts[sq] = q.get("required_private_facts", {}).get("agent_A", []) + \
                          q.get("required_private_facts", {}).get("agent_B", [])

    distortion_vals = []
    for t in traces:
        sq = t.get("shared_question", "")
        facts = q_facts.get(sq, [])
        if facts:
            distortion_vals.append(compute_fact_distortion_rate(t, facts))
    avg_distortion = sum(distortion_vals) / len(distortion_vals) if distortion_vals else 0.0

    # 6. Partial answer rate
    partial_vals = [compute_partial_answer_rate(t) for t in traces]
    avg_partial = sum(partial_vals) / len(partial_vals) if partial_vals else 0.0

    # 7. Undetermined ratio
    undet_vals = [compute_undetermined_ratio(t) for t in traces]
    avg_undetermined = sum(undet_vals) / len(undet_vals) if undet_vals else 0.0

    # 8. Oracle-specific metrics
    oracle_selection_rate = 0.0
    fresh_solver_acc = 0.0
    oracle_traces = [t for t in traces if t.get("setting") == "oracle_candidate"]
    if oracle_traces:
        oracle_selection_rate = sum(1 for t in oracle_traces
                                    if t.get("finalizer_selected_oracle", False)) / len(oracle_traces)
    fresh_traces = [t for t in traces if t.get("setting") == "canonical_state_fresh"]
    if fresh_traces:
        fresh_solver_acc = sum(1 for t in fresh_traces
                               if t.get("fresh_solver_correct", False)) / len(fresh_traces)

    # Failure type distribution
    failure_types = Counter()
    for t in traces:
        ft = t.get("failure_type", "unknown")
        failure_types[ft] += 1

    return {
        "n": n,
        "num_questions": len(by_q),
        "semantic_accuracy": semantic_correct / n,
        "semantic_correct_count": semantic_correct,
        "format_compliance_rate": format_ok / n,
        "format_compliant_count": format_ok,
        "correct_candidate_emergence_rate": cand_qs / len(by_q) if by_q else 0,
        "correct_candidate_questions": cand_qs,
        "answer_reason_consistency_rate": consistent / checkable if checkable > 0 else None,
        "answer_reason_consistent_count": consistent,
        "answer_reason_checkable_count": checkable,
        "avg_fact_distortion_rate": avg_distortion,
        "avg_partial_answer_rate": avg_partial,
        "avg_undetermined_ratio": avg_undetermined,
        "oracle_selection_rate": oracle_selection_rate,
        "fresh_solver_accuracy": fresh_solver_acc,
        "failure_type_distribution": dict(failure_types),
    }


def generate_report(output_dir: str) -> str:
    """Generate comprehensive analysis report."""
    output_path = Path(output_dir)
    traces_path = output_path / "traces_all.json"
    if not traces_path.exists():
        print(f"ERROR: traces_all.json not found in {output_dir}")
        sys.exit(1)

    traces = load_traces(traces_path)
    print(f"Loaded {len(traces)} traces from {traces_path}")

    # Group by setting
    by_setting = defaultdict(list)
    for t in traces:
        by_setting[t.get("setting", "unknown")].append(t)

    print(f"Settings found: {sorted(by_setting.keys())}")

    # Compute metrics
    all_metrics = {}
    for setting in SETTING_ORDER:
        if setting in by_setting:
            all_metrics[setting] = compute_setting_metrics(by_setting[setting])
            print(f"  {setting}: {all_metrics[setting]['semantic_accuracy']:.1%} ({all_metrics[setting]['n']} traces)")

    # ── Build report ──
    report = f"""# Oracle Intervention Experiment — Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data source**: `{output_dir}`
**Model**: Qwen2.5-1.5B
**Questions**: 20

---

## 1. Per-Setting Metrics

| # | Setting | n | Semantic Acc | Format | Cand Emerge | Ans-Reason Consist | Fact Distortion | Partial Ans | Undetermined | Oracle Select | Fresh Solver |
|---|---------|---|-------------|--------|-------------|-------------------|----------------|-------------|--------------|---------------|---------------|
"""

    for setting in SETTING_ORDER:
        m = all_metrics.get(setting, {})
        if not m:
            continue
        label = SETTING_SHORT.get(setting, setting)
        def fmt(val):
            if val is None:
                return "N/A"
            return f"{val:.1%}"

        report += (f"| {setting[:4]} | {label} | {m.get('n', 0)} | "
                   f"**{fmt(m.get('semantic_accuracy', 0))}** | "
                   f"{fmt(m.get('format_compliance_rate', 0))} | "
                   f"{fmt(m.get('correct_candidate_emergence_rate', 0))} | "
                   f"{fmt(m.get('answer_reason_consistency_rate'))} | "
                   f"{fmt(m.get('avg_fact_distortion_rate', 0))} | "
                   f"{fmt(m.get('avg_partial_answer_rate', 0))} | "
                   f"{fmt(m.get('avg_undetermined_ratio', 0))} | "
                   f"{fmt(m.get('oracle_selection_rate', 0))} | "
                   f"{fmt(m.get('fresh_solver_accuracy', 0))} |\n")

    # ── Recovery Analysis ──
    baseline = all_metrics.get("free_discussion", {})
    baseline_acc = baseline.get("semantic_accuracy", 0)

    report += f"""
## 2. Recovery Analysis (Causal Bottleneck Identification)

**Baseline**: `free_discussion` — semantic accuracy = **{baseline_acc:.1%}**

| Oracle Intervention | Accuracy | Δ from Baseline | Interpretation |
|---------------------|----------|-----------------|----------------|
"""

    interpretations = []
    for setting in SETTING_ORDER:
        if setting == "free_discussion":
            report += f"| {SETTING_SHORT[setting]} | **{baseline_acc:.1%}** | — | (baseline: private facts, agents must communicate) |\n"
            continue

        m = all_metrics.get(setting, {})
        if not m:
            continue
        acc = m.get("semantic_accuracy", 0)
        delta = acc - baseline_acc

        if delta > 0.3:
            interp = "★★★ MAJOR — this layer is a PRIMARY bottleneck"
        elif delta > 0.15:
            interp = "★★  Significant — this layer is a SECONDARY bottleneck"
        elif delta > 0.05:
            interp = "★   Minor — this layer has a SMALL effect"
        elif delta < -0.05:
            interp = "⚠️  Degradation — intervention made things WORSE"
        else:
            interp = "—   No meaningful effect — bottleneck is ELSEWHERE"

        interpretations.append((setting, delta, interp))
        report += f"| {SETTING_SHORT[setting]} | **{acc:.1%}** | {delta:+.1%} | {interp} |\n"

    # ── Causal Chain Analysis ──
    report += f"""
## 3. Causal Chain Interpretation

```
Free Discussion ({baseline_acc:.0%})
    │
    ├──[Oracle Disclosure]──→ {"Recovers to " + f"{all_metrics.get('oracle_disclosure', {}).get('semantic_accuracy', 0):.0%}" if 'oracle_disclosure' in all_metrics else "N/A"}
    │   Interpretation: If large recovery → fact DISCLOSURE (agents not saying what they know) is the bottleneck
    │
    ├──[Oracle Canonical State]──→ {"Recovers to " + f"{all_metrics.get('oracle_canonical_state', {}).get('semantic_accuracy', 0):.0%}" if 'oracle_canonical_state' in all_metrics else "N/A"}
    │   Interpretation: If recovery beyond Disclosure → fact ORGANIZATION/NORMALIZATION is the bottleneck
    │
    ├──[Canonical State + Fresh Solver]──→ {"Recovers to " + f"{all_metrics.get('canonical_state_fresh', {}).get('semantic_accuracy', 0):.0%}" if 'canonical_state_fresh' in all_metrics else "N/A"}
    │   Interpretation: If recovery beyond Canonical → DISCUSSION CONTAMINATION (old errors pollute reasoning)
    │
    ├──[Oracle Plan]──→ {"Recovers to " + f"{all_metrics.get('oracle_plan', {}).get('semantic_accuracy', 0):.0%}" if 'oracle_plan' in all_metrics else "N/A"}
    │   Interpretation: If recovery beyond Fresh → PLAN GENERATION (cross-fact dependency tracking) is the bottleneck
    │
    └──[Oracle Candidate]──→ {"Recovers to " + f"{all_metrics.get('oracle_candidate', {}).get('semantic_accuracy', 0):.0%}" if 'oracle_candidate' in all_metrics else "N/A"}
        Interpretation: If recovery incomplete → FINALIZER/VERIFIER still has problems (can't retain correct answer even when given)
        Oracle Selection Rate: {all_metrics.get('oracle_candidate', {}).get('oracle_selection_rate', 0):.1%}
```

## 4. Key Diagnostic Questions

"""

    # Determine primary bottleneck
    max_delta = -1.0
    max_setting = None
    for setting, delta, interp in interpretations:
        if delta > max_delta:
            max_delta = delta
            max_setting = setting

    report += f"**Largest recovery**: `{max_setting}` (+{max_delta:.1%}) — {SETTING_LABELS.get(max_setting, max_setting)}\n\n"

    report += "### Bottleneck Attribution\n\n"

    m_disclosure = all_metrics.get("oracle_disclosure", {}).get("semantic_accuracy", 0)
    m_canonical = all_metrics.get("oracle_canonical_state", {}).get("semantic_accuracy", 0)
    m_fresh = all_metrics.get("canonical_state_fresh", {}).get("semantic_accuracy", 0)
    m_plan = all_metrics.get("oracle_plan", {}).get("semantic_accuracy", 0)
    m_candidate = all_metrics.get("oracle_candidate", {}).get("semantic_accuracy", 0)

    # Attribution logic
    fact_disclosure_loss = baseline_acc - m_disclosure
    fact_organization_loss = m_disclosure - m_canonical
    discussion_contamination_loss = m_canonical - m_fresh
    plan_generation_loss = m_fresh - m_plan
    finalizer_loss = m_plan - m_candidate  # if plan recovers but candidate doesn't fully

    report += f"""| Bottleneck Layer | Accuracy Loss | Cumulative |
|------------------|---------------|------------|
| Fact Disclosure (agents don't say what they know) | {fact_disclosure_loss:.1%} | {fact_disclosure_loss:.1%} |
| Fact Organization (shared state poorly structured) | {fact_organization_loss:.1%} | {fact_disclosure_loss + fact_organization_loss:.1%} |
| Discussion Contamination (old errors corrupt reasoning) | {discussion_contamination_loss:.1%} | {fact_disclosure_loss + fact_organization_loss + discussion_contamination_loss:.1%} |
| Plan Generation (can't build cross-fact dependencies) | {plan_generation_loss:.1%} | {fact_disclosure_loss + fact_organization_loss + discussion_contamination_loss + plan_generation_loss:.1%} |
| Finalizer/Verifier (can't retain correct answer) | {finalizer_loss:.1%} | {fact_disclosure_loss + fact_organization_loss + discussion_contamination_loss + plan_generation_loss + finalizer_loss:.1%} |
| **Residual (model arithmetic capability ceiling)** | {m_plan:.1%} (remaining error after full Oracle Plan) | — |
"""

    # ── Detailed metric comparison ──
    report += """
## 5. Detailed Metric Comparison

### 5.1 Fact Distortion Rate (lower is better)

| Setting | Fact Distortion Rate |
|---------|---------------------|
"""
    for setting in SETTING_ORDER:
        m = all_metrics.get(setting, {})
        if not m:
            continue
        report += f"| {SETTING_SHORT[setting]} | **{m.get('avg_fact_distortion_rate', 0):.1%}** |\n"

    report += """
### 5.2 Undetermined Answer Ratio (lower is better)

| Setting | Undetermined Ratio |
|---------|-------------------|
"""
    for setting in SETTING_ORDER:
        m = all_metrics.get(setting, {})
        if not m:
            continue
        report += f"| {SETTING_SHORT[setting]} | **{m.get('avg_undetermined_ratio', 0):.1%}** |\n"

    report += """
### 5.3 Partial Answer Rate (higher is better — agents are trying)

| Setting | Partial Answer Rate |
|---------|-------------------|
"""
    for setting in SETTING_ORDER:
        m = all_metrics.get(setting, {})
        if not m:
            continue
        report += f"| {SETTING_SHORT[setting]} | **{m.get('avg_partial_answer_rate', 0):.1%}** |\n"

    # ── Failure type analysis ──
    report += """
## 6. Failure Type Distribution

"""
    for setting in SETTING_ORDER:
        m = all_metrics.get(setting, {})
        if not m:
            continue
        ftypes = m.get("failure_type_distribution", {})
        if not ftypes:
            continue
        report += f"### {SETTING_SHORT.get(setting, setting)}\n\n"
        report += "| Failure Type | Count |\n|---|---|\n"
        for ft, count in sorted(ftypes.items(), key=lambda x: -x[1]):
            report += f"| {ft} | {count} |\n"
        report += "\n"

    # ── Save report ──
    report_path = output_path / "oracle_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # ── Save metrics CSV ──
    csv_path = output_path / "oracle_metrics_analysis.csv"
    metric_keys = [
        "semantic_accuracy", "format_compliance_rate",
        "correct_candidate_emergence_rate", "answer_reason_consistency_rate",
        "avg_fact_distortion_rate", "avg_partial_answer_rate",
        "avg_undetermined_ratio",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "n"] + metric_keys)
        for setting in SETTING_ORDER:
            m = all_metrics.get(setting, {})
            if not m:
                continue
            writer.writerow([setting, m.get("n", 0)] +
                           [m.get(k, 0) for k in metric_keys])
    print(f"CSV saved to: {csv_path}")

    return str(report_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_oracle_results.py <output_dir>")
        print("Example: python analyze_oracle_results.py outputs_oracle_intervention/20260806_120000")
        sys.exit(1)

    output_dir = sys.argv[1]
    report_path = generate_report(output_dir)
    print(f"\nDone! Report: {report_path}")
