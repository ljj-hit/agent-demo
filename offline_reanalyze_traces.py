"""Offline re-scoring for the five Hidden-GSM8K result sets.

Semantic answer correctness is intentionally independent from presentation
format and selected-source consistency.  No model or network call is made.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


DEFAULT_RUNS = (
    "20260721_145822_single_full",
    "20260721_145822_single_partial",
    "20260721_172014_multi_partial",
    "20260721_172014_multi_partial_verifier",
    "20260721_172014_oracle_broadcast",
)
UNKNOWN = re.compile(
    r"\b(?:undetermined|unknown|none|n/?a|insufficient)\b|"
    r"cannot (?:be )?(?:determine|calculate)|impossible to (?:determine|calculate)",
    re.I,
)
NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+)?")


def numeric(value: object) -> Decimal | None:
    text = str(value or "").replace(",", "").strip()
    match = NUMBER.fullmatch(text)
    if not match:
        return None
    try:
        if "/" in text:
            left, right = (Decimal(x.strip()) for x in text.split("/", 1))
            return None if right == 0 else left / right
        return Decimal(text)
    except InvalidOperation:
        return None


def declared_value(raw: str, label: str) -> str:
    clean = re.sub(r"[*`]", "", raw)
    matches = re.findall(rf"(?im)^\s*{re.escape(label)}\s*[:：=]\s*(.*?)\s*$", clean)
    return matches[-1].strip() if matches else ""


def number_from_value(value: str) -> str:
    if not value or UNKNOWN.search(value):
        return ""
    match = NUMBER.search(value)
    return match.group(0).replace(",", "") if match else ""


def recover_final_answer(trace: dict) -> tuple[str, str]:
    """Recover only an explicit/previously-audited final numeric answer."""
    event = trace.get("single_event") or trace.get("finalizer_event") or {}
    raw = str(event.get("raw_output") or event.get("output") or "")
    declared = declared_value(raw, "Final answer")
    answer = number_from_value(declared)
    if answer:
        return answer, "explicit_final_answer_label"
    # A strict parser may have erased a clear numeric answer for formatting.
    # The trace's earlier free-text extraction is safe to retain when numeric.
    for key, value in (("event.answer", event.get("answer")),
                       ("trace.final_prediction", trace.get("final_prediction"))):
        candidate = number_from_value(str(value or ""))
        if candidate:
            return candidate, key
    return "", "explicitly_undetermined" if UNKNOWN.search(declared or raw) else "no_explicit_numeric_answer"


def sentence_count(lines: list[str]) -> int:
    count = 0
    for line in (x.strip() for x in lines if x.strip()):
        protected = re.sub(r"(?<=\d)\.(?=\d)", "\ue000", line)
        count += len([x for x in re.split(r"[.!?。！？]+", protected) if x.strip()])
    return count


def format_issues(trace: dict) -> list[str]:
    event = trace.get("single_event") or trace.get("finalizer_event") or {}
    raw = str(event.get("raw_output") or event.get("output") or "").rstrip("\r\n")
    lines = raw.splitlines()
    issues: list[str] = []
    if trace.get("single_event"):
        if not lines or not re.fullmatch(r"Final answer\s*[:：]\s*.+", lines[0], re.I):
            issues.append("answer_not_on_first_line")
        if sentence_count(lines[1:]) > 3:
            issues.append("more_than_three_reasoning_sentences")
    else:
        source = declared_value(raw, "Selected source").lower()
        if source == "computed":
            issues.append("computed_used_instead_of_recomputed")
        nonempty = [x for x in lines if x.strip()]
        expected = ("Selected source", "Final answer", "Reason")
        labels_ok = len(nonempty) == 3 and all(
            re.match(rf"^\s*{re.escape(label)}\s*[:：]", line, re.I)
            for line, label in zip(nonempty, expected)
        )
        if not labels_ok:
            issues.append("finalizer_not_three_labeled_lines")
        if len(lines) != 3:
            issues.append("finalizer_physical_line_count_not_three")
    return issues


def equivalent(left: object, right: object) -> bool:
    a, b = numeric(left), numeric(right)
    return a is not None and b is not None and a == b


def source_issues(trace: dict, answer: str) -> tuple[str, list[str]]:
    event = trace.get("finalizer_event")
    if not event:
        return "", []
    raw = str(event.get("raw_output") or event.get("output") or "")
    original = declared_value(raw, "Selected source").lower()
    source = "recomputed" if original == "computed" else original
    issues: list[str] = []
    candidates = trace.get("candidate_answers", {})
    if not source:
        issues.append("selected_source_missing")
    elif source in {"solver_a", "solver_b", "verifier"}:
        candidate = candidates.get(source, "")
        if numeric(candidate) is None:
            issues.append(f"selected_source_{source}_unavailable")
        elif not equivalent(answer, candidate):
            issues.append(f"answer_mismatch_with_{source}")
    elif source == "none":
        if answer:
            issues.append("numeric_answer_with_source_none")
    elif source == "recomputed":
        if not answer:
            issues.append("recomputed_without_numeric_answer")
    else:
        issues.append(f"unsupported_selected_source:{original}")
    return source, issues


def analyze(trace: dict, run: str) -> dict:
    answer, method = recover_final_answer(trace)
    source, source_problem = source_issues(trace, answer)
    gold = str(trace.get("gold_answer", ""))
    return {
        "run": run,
        "setting": trace.get("setting", ""),
        "agent_variant": trace.get("agent_variant", ""),
        "question_id": trace.get("question_id"),
        "gold_answer": gold,
        "offline_answer": answer,
        "answer_extraction": method,
        "answer_correct": equivalent(answer, gold),
        "format_issue": bool(problems := format_issues(trace)),
        "format_issues": problems,
        "selected_source": source,
        "source_inconsistency": bool(source_problem),
        "source_consistency_issues": source_problem,
        "original_final_prediction": trace.get("final_prediction", ""),
        "original_correct": bool(trace.get("correct")),
        "original_invalid_output": bool(trace.get("invalid_output")),
    }


def summaries(rows: list[dict], include_variant: bool = False) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        variant = row["agent_variant"] if include_variant else ""
        groups[(row["setting"], variant)].append(row)
    result = []
    for (setting, variant), items in groups.items():
        result.append({
            "setting": setting,
            "agent_variant": variant,
            "n": len(items),
            "answer_correct": sum(x["answer_correct"] for x in items),
            "answer_accuracy": sum(x["answer_correct"] for x in items) / len(items),
            "format_issue": sum(x["format_issue"] for x in items),
            "source_inconsistency": sum(x["source_inconsistency"] for x in items),
            "original_correct": sum(x["original_correct"] for x in items),
            "recovered_correct_due_to_decoupling": sum(
                x["answer_correct"] and not x["original_correct"] for x in items
            ),
        })
    return result


def report_text(rows: list[dict], summary: list[dict], input_root: Path) -> str:
    total = len(rows)
    fmt = Counter(issue for row in rows for issue in row["format_issues"])
    src = Counter(issue for row in rows for issue in row["source_consistency_issues"])
    out = [
        "# Hidden-GSM8K 五种设置 trace 离线重算报告", "",
        "## 口径", "",
        f"本报告从 `{input_root}` 下五份 `traces_all.json` 离线生成，共 {total} 条 trace；没有调用模型或在线裁判。数值答案正确性、格式合规性、来源一致性是三个互不覆盖的维度。明确写出的数值答案即使格式违规也会保留并参与正确性比较。`computed` 按 `recomputed` 的语义读取，但另记格式问题。", "",
        "## 汇总", "",
        "| 设置 | 变体 | N | 答案正确 | 正确率 | 格式问题 | 来源不一致 | 原始 correct | 因解耦恢复的正确项 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summary:
        out.append(f"| {x['setting']} | {x['agent_variant'] or '-'} | {x['n']} | {x['answer_correct']} | {x['answer_accuracy']:.1%} | {x['format_issue']} | {x['source_inconsistency']} | {x['original_correct']} | {x['recovered_correct_due_to_decoupling']} |")
    variant_summary = [x for x in summaries(rows, include_variant=True) if x["agent_variant"]]
    out += ["", "### single_partial 变体明细", "",
            "| 变体 | N | 答案正确 | 正确率 | 格式问题 |", "|---|---:|---:|---:|---:|"]
    for x in variant_summary:
        out.append(f"| {x['agent_variant']} | {x['n']} | {x['answer_correct']} | {x['answer_accuracy']:.1%} | {x['format_issue']} |")
    overall_correct = sum(x["answer_correct"] for x in rows)
    out += [
        "", "## 总体结论", "",
        f"离线数值正确 {overall_correct}/{total}（{overall_correct/total:.1%}）；有格式问题 {sum(x['format_issue'] for x in rows)}/{total}；有来源不一致 {sum(x['source_inconsistency'] for x in rows)}/{total}。原始 `correct` 为真 {sum(x['original_correct'] for x in rows)}/{total}，格式与正确性解耦后恢复 {sum(x['answer_correct'] and not x['original_correct'] for x in rows)} 个正确答案。", "",
        "格式问题分布：" + ("；".join(f"`{k}` {v} 次" for k, v in fmt.most_common()) or "无") + "。", "",
        "来源一致性问题分布：" + ("；".join(f"`{k}` {v} 次" for k, v in src.most_common()) or "无") + "。", "",
        "## 被原逻辑因格式清空、但离线恢复为正确的条目", "",
    ]
    recovered = [x for x in rows if x["answer_correct"] and not x["original_correct"]]
    if recovered:
        for x in recovered:
            out.append(f"- `{x['setting']}` q{x['question_id']}：恢复答案 `{x['offline_answer']}`；格式问题：{', '.join(x['format_issues']) or '无'}；来源问题：{', '.join(x['source_consistency_issues']) or '无'}。")
    else:
        out.append("- 无。")
    out += [
        "", "## 解读限制", "",
        "`single_partial` 的 A/B 是同一设置下两个独立变体，因此该目录有 40 条而非 20 条；报告既分变体展示，也保留全部逐题记录。离线判断只认可明确标签或 trace 中已经保存的数值预测，不会从长推理过程里挑一个碰巧等于 gold 的数字。来源一致性使用 trace 保存的候选答案；若所选候选因上游格式错误而不可用，会记为来源问题，而不会反过来清空最终数值答案。", "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs_hidden_gsm8k"))
    parser.add_argument("--output-dir", type=Path, default=Path("offline_reanalysis"))
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        path = args.input_root / run / "traces_all.json"
        traces = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(analyze(trace, run) for trace in traces)
    summary = summaries(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "offline_trace_scores.json").write_text(
        json.dumps({"runs": args.runs, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = [k for k in rows[0] if k not in {"format_issues", "source_consistency_issues"}]
    with (args.output_dir / "offline_trace_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["format_issues", "source_consistency_issues"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "format_issues": ";".join(row["format_issues"]),
                             "source_consistency_issues": ";".join(row["source_consistency_issues"])})
    (args.output_dir / "analysis_report.md").write_text(
        report_text(rows, summary, args.input_root), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} offline trace scores to {args.output_dir}")


if __name__ == "__main__":
    main()
