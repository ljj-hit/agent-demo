# ExecGround Six-Group Ablation Experiment — Analysis Report

**Date**: 2026-08-07 20:09:43
**Settings**: 6

## Layer-by-Layer Correct Candidate Emergence

The key question: at which layer does the correct answer first emerge as a candidate?

| # | Setting | Accuracy | Candidate Emergence | Emerged w/ Complete Info | Format OK |
|---|---------|----------|--------------------|-------------------------|----------|
| 0 | free_discussion | 0.0% | 0.0% | 0.0% | 10.0% |
| 1 | reveal_all | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | canonical_ledger | 0.0% | 0.0% | 0.0% | 0.0% |
| 3 | ledger_fresh_solver | 5.6% | 5.6% | 5.6% | 100.0% |
| 4 | ledger_exec_plan | 5.6% | 0.0% | 0.0% | 100.0% |
| 5 | ledger_exec_plan_verify | 5.6% | 0.0% | 0.0% | 100.0% |

## Causal Attribution

Each layer adds one capability. The Δ in emergence rate reveals the bottleneck:

- **free_discussion** (baseline): emergence = 0.0%
- **reveal_all**: emergence = 0.0% (Δ = +0.0%)
  - Negligible effect: +0.0%
- **canonical_ledger**: emergence = 0.0% (Δ = +0.0%)
  - Negligible effect: +0.0%
- **ledger_fresh_solver**: emergence = 5.6% (Δ = +5.6%)
  - ★ MAJOR BOTTLENECK: 3. Ledger + Fresh Solver (no discussion history) contributes 6% improvement
- **ledger_exec_plan**: emergence = 0.0% (Δ = -5.6%)
  - Negligible effect: -5.6%
- **ledger_exec_plan_verify**: emergence = 0.0% (Δ = +0.0%)
  - Negligible effect: +0.0%

## Interpretation Guide

1. If **ledger + fresh solver** already recovers most accuracy:
   → Main problem is discussion history contamination and state confusion.
   → Keep the architecture simple.

2. If **executable plan** is needed for significant recovery:
   → Core problem is cross-fact dependency construction failure.
   → Focus on dependency graph and programmatic reasoning.

3. If **even oracle state + oracle plan** cannot recover:
   → Re-examine problem splitting, data labeling, and model base capability.
   → Do NOT rush to write paper conclusions.

## ExecGround-Specific Metrics

### canonical_ledger
- **avg_ledger_fact_count**: 4.111
- **avg_fact_quality_A**: 0.852
- **avg_fact_quality_B**: 0.769

### ledger_fresh_solver
- **avg_ledger_fact_count**: 4.111
- **avg_fact_quality_A**: 0.852
- **avg_fact_quality_B**: 0.769

### ledger_exec_plan
- **avg_ledger_fact_count**: 4.111
- **avg_fact_quality_A**: 0.852
- **avg_fact_quality_B**: 0.769
- **plan_execution_correct**: 0
- **plan_execution_correct_rate**: 0.000
- **avg_plan_steps**: 3.000

### ledger_exec_plan_verify
- **avg_ledger_fact_count**: 4.111
- **avg_fact_quality_A**: 0.852
- **avg_fact_quality_B**: 0.769
- **plan_execution_correct**: 0
- **plan_execution_correct_rate**: 0.000
- **avg_plan_steps**: 0.000
- **final_plan_clean_rate**: 0.000
- **avg_verify_rounds**: 2.889
