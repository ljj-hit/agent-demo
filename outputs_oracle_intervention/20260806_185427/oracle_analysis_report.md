# Oracle Intervention Experiment — Analysis Report

**Generated**: 2026-08-06 19:10:08
**Data source**: `outputs_oracle_intervention/20260806_185427`
**Model**: Qwen2.5-1.5B
**Questions**: 20

---

## 1. Per-Setting Metrics

| # | Setting | n | Semantic Acc | Format | Cand Emerge | Ans-Reason Consist | Fact Distortion | Partial Ans | Undetermined | Oracle Select | Fresh Solver |
|---|---------|---|-------------|--------|-------------|-------------------|----------------|-------------|--------------|---------------|---------------|
| free | Free Discussion | 6 | **16.7%** | 0.0% | 0.0% | 100.0% | 0.0% | 29.2% | 100.0% | 0.0% | 0.0% |
| orac | Oracle Disclosure | 6 | **66.7%** | 66.7% | 66.7% | 100.0% | 0.0% | 50.0% | 100.0% | 0.0% | 0.0% |
| orac | Oracle Canonical State | 6 | **66.7%** | 66.7% | 66.7% | 100.0% | 0.0% | 50.0% | 100.0% | 0.0% | 0.0% |
| cano | Canonical State + Fresh | 6 | **100.0%** | 0.0% | 100.0% | N/A | 0.0% | 50.0% | 100.0% | 0.0% | 100.0% |
| orac | Oracle Plan | 6 | **100.0%** | 66.7% | 66.7% | 100.0% | 0.0% | 50.0% | 100.0% | 0.0% | 0.0% |
| orac | Oracle Candidate | 6 | **66.7%** | 100.0% | 100.0% | 33.3% | 8.3% | 16.7% | 100.0% | 66.7% | 0.0% |

## 2. Recovery Analysis (Causal Bottleneck Identification)

**Baseline**: `free_discussion` — semantic accuracy = **16.7%**

| Oracle Intervention | Accuracy | Δ from Baseline | Interpretation |
|---------------------|----------|-----------------|----------------|
| Free Discussion | **16.7%** | — | (baseline: private facts, agents must communicate) |
| Oracle Disclosure | **66.7%** | +50.0% | ★★★ MAJOR — this layer is a PRIMARY bottleneck |
| Oracle Canonical State | **66.7%** | +50.0% | ★★★ MAJOR — this layer is a PRIMARY bottleneck |
| Canonical State + Fresh | **100.0%** | +83.3% | ★★★ MAJOR — this layer is a PRIMARY bottleneck |
| Oracle Plan | **100.0%** | +83.3% | ★★★ MAJOR — this layer is a PRIMARY bottleneck |
| Oracle Candidate | **66.7%** | +50.0% | ★★★ MAJOR — this layer is a PRIMARY bottleneck |

## 3. Causal Chain Interpretation

```
Free Discussion (17%)
    │
    ├──[Oracle Disclosure]──→ Recovers to 67%
    │   Interpretation: If large recovery → fact DISCLOSURE (agents not saying what they know) is the bottleneck
    │
    ├──[Oracle Canonical State]──→ Recovers to 67%
    │   Interpretation: If recovery beyond Disclosure → fact ORGANIZATION/NORMALIZATION is the bottleneck
    │
    ├──[Canonical State + Fresh Solver]──→ Recovers to 100%
    │   Interpretation: If recovery beyond Canonical → DISCUSSION CONTAMINATION (old errors pollute reasoning)
    │
    ├──[Oracle Plan]──→ Recovers to 100%
    │   Interpretation: If recovery beyond Fresh → PLAN GENERATION (cross-fact dependency tracking) is the bottleneck
    │
    └──[Oracle Candidate]──→ Recovers to 67%
        Interpretation: If recovery incomplete → FINALIZER/VERIFIER still has problems (can't retain correct answer even when given)
        Oracle Selection Rate: 66.7%
```

## 4. Key Diagnostic Questions

**Largest recovery**: `canonical_state_fresh` (+83.3%) — 3. Canonical State + Fresh Solver (no history)

### Bottleneck Attribution

| Bottleneck Layer | Accuracy Loss | Cumulative |
|------------------|---------------|------------|
| Fact Disclosure (agents don't say what they know) | -50.0% | -50.0% |
| Fact Organization (shared state poorly structured) | 0.0% | -50.0% |
| Discussion Contamination (old errors corrupt reasoning) | -33.3% | -83.3% |
| Plan Generation (can't build cross-fact dependencies) | 0.0% | -83.3% |
| Finalizer/Verifier (can't retain correct answer) | 33.3% | -50.0% |
| **Residual (model arithmetic capability ceiling)** | 100.0% (remaining error after full Oracle Plan) | — |

## 5. Detailed Metric Comparison

### 5.1 Fact Distortion Rate (lower is better)

| Setting | Fact Distortion Rate |
|---------|---------------------|
| Free Discussion | **0.0%** |
| Oracle Disclosure | **0.0%** |
| Oracle Canonical State | **0.0%** |
| Canonical State + Fresh | **0.0%** |
| Oracle Plan | **0.0%** |
| Oracle Candidate | **8.3%** |

### 5.2 Undetermined Answer Ratio (lower is better)

| Setting | Undetermined Ratio |
|---------|-------------------|
| Free Discussion | **100.0%** |
| Oracle Disclosure | **100.0%** |
| Oracle Canonical State | **100.0%** |
| Canonical State + Fresh | **100.0%** |
| Oracle Plan | **100.0%** |
| Oracle Candidate | **100.0%** |

### 5.3 Partial Answer Rate (higher is better — agents are trying)

| Setting | Partial Answer Rate |
|---------|-------------------|
| Free Discussion | **29.2%** |
| Oracle Disclosure | **50.0%** |
| Oracle Canonical State | **50.0%** |
| Canonical State + Fresh | **50.0%** |
| Oracle Plan | **50.0%** |
| Oracle Candidate | **16.7%** |

## 6. Failure Type Distribution

### Free Discussion

| Failure Type | Count |
|---|---|
| invalid_output | 6 |

### Oracle Disclosure

| Failure Type | Count |
|---|---|
| None | 4 |
| invalid_output | 2 |

### Oracle Canonical State

| Failure Type | Count |
|---|---|
| None | 4 |
| invalid_output | 2 |

### Canonical State + Fresh

| Failure Type | Count |
|---|---|
| invalid_output | 6 |

### Oracle Plan

| Failure Type | Count |
|---|---|
| None | 4 |
| invalid_output | 2 |

### Oracle Candidate

| Failure Type | Count |
|---|---|
| None | 3 |
| answer_reason_inconsistency | 2 |
| answer_selection_failure | 1 |

