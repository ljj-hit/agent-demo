# Oracle Intervention Experiment — Analysis Report

**Generated**: 2026-08-06 22:57:15
**Data source**: `outputs_oracle_intervention/20260806_191645`
**Model**: Qwen2.5-1.5B
**Questions**: 20

---

## 1. Per-Setting Metrics

| # | Setting | n | Semantic Acc | Format | Cand Emerge | Ans-Reason Consist | Fact Distortion | Partial Ans | Undetermined | Oracle Select | Fresh Solver |
|---|---------|---|-------------|--------|-------------|-------------------|----------------|-------------|--------------|---------------|---------------|
| free | Free Discussion | 60 | **5.0%** | 15.0% | 5.0% | 50.0% | 10.0% | 19.2% | 99.6% | 0.0% | 0.0% |
| orac | Oracle Disclosure | 60 | **25.0%** | 25.0% | 20.0% | 100.0% | 22.5% | 27.5% | 90.0% | 0.0% | 0.0% |
| orac | Oracle Canonical State | 60 | **20.0%** | 25.0% | 20.0% | 100.0% | 22.5% | 27.5% | 90.0% | 0.0% | 0.0% |
| cano | Canonical State + Fresh | 60 | **60.0%** | 0.0% | 25.0% | 50.0% | 22.5% | 27.5% | 90.0% | 0.0% | 60.0% |
| orac | Oracle Plan | 60 | **70.0%** | 30.0% | 20.0% | 42.9% | 22.5% | 27.5% | 90.0% | 0.0% | 0.0% |
| orac | Oracle Candidate | 60 | **53.3%** | 100.0% | 100.0% | 76.5% | 7.6% | 22.5% | 99.2% | 51.7% | 0.0% |

## 2. Recovery Analysis (Causal Bottleneck Identification)

**Baseline**: `free_discussion` — semantic accuracy = **5.0%**

| Oracle Intervention | Accuracy | Δ from Baseline | Interpretation |
|---------------------|----------|-----------------|----------------|
| Free Discussion | **5.0%** | — | (baseline: private facts, agents must communicate) |
| Oracle Disclosure | **25.0%** | +20.0% | ★★  Significant — this layer is a SECONDARY bottleneck |
| Oracle Canonical State | **20.0%** | +15.0% | ★★  Significant — this layer is a SECONDARY bottleneck |
| Canonical State + Fresh | **60.0%** | +55.0% | ★★★ MAJOR — this layer is a PRIMARY bottleneck |
| Oracle Plan | **70.0%** | +65.0% | ★★★ MAJOR — this layer is a PRIMARY bottleneck |
| Oracle Candidate | **53.3%** | +48.3% | ★★★ MAJOR — this layer is a PRIMARY bottleneck |

## 3. Causal Chain Interpretation

```
Free Discussion (5%)
    │
    ├──[Oracle Disclosure]──→ Recovers to 25%
    │   Interpretation: If large recovery → fact DISCLOSURE (agents not saying what they know) is the bottleneck
    │
    ├──[Oracle Canonical State]──→ Recovers to 20%
    │   Interpretation: If recovery beyond Disclosure → fact ORGANIZATION/NORMALIZATION is the bottleneck
    │
    ├──[Canonical State + Fresh Solver]──→ Recovers to 60%
    │   Interpretation: If recovery beyond Canonical → DISCUSSION CONTAMINATION (old errors pollute reasoning)
    │
    ├──[Oracle Plan]──→ Recovers to 70%
    │   Interpretation: If recovery beyond Fresh → PLAN GENERATION (cross-fact dependency tracking) is the bottleneck
    │
    └──[Oracle Candidate]──→ Recovers to 53%
        Interpretation: If recovery incomplete → FINALIZER/VERIFIER still has problems (can't retain correct answer even when given)
        Oracle Selection Rate: 51.7%
```

## 4. Key Diagnostic Questions

**Largest recovery**: `oracle_plan` (+65.0%) — 4. Oracle Plan (equation structure)

### Bottleneck Attribution

| Bottleneck Layer | Accuracy Loss | Cumulative |
|------------------|---------------|------------|
| Fact Disclosure (agents don't say what they know) | -20.0% | -20.0% |
| Fact Organization (shared state poorly structured) | 5.0% | -15.0% |
| Discussion Contamination (old errors corrupt reasoning) | -40.0% | -55.0% |
| Plan Generation (can't build cross-fact dependencies) | -10.0% | -65.0% |
| Finalizer/Verifier (can't retain correct answer) | 16.7% | -48.3% |
| **Residual (model arithmetic capability ceiling)** | 70.0% (remaining error after full Oracle Plan) | — |

## 5. Detailed Metric Comparison

### 5.1 Fact Distortion Rate (lower is better)

| Setting | Fact Distortion Rate |
|---------|---------------------|
| Free Discussion | **10.0%** |
| Oracle Disclosure | **22.5%** |
| Oracle Canonical State | **22.5%** |
| Canonical State + Fresh | **22.5%** |
| Oracle Plan | **22.5%** |
| Oracle Candidate | **7.6%** |

### 5.2 Undetermined Answer Ratio (lower is better)

| Setting | Undetermined Ratio |
|---------|-------------------|
| Free Discussion | **99.6%** |
| Oracle Disclosure | **90.0%** |
| Oracle Canonical State | **90.0%** |
| Canonical State + Fresh | **90.0%** |
| Oracle Plan | **90.0%** |
| Oracle Candidate | **99.2%** |

### 5.3 Partial Answer Rate (higher is better — agents are trying)

| Setting | Partial Answer Rate |
|---------|-------------------|
| Free Discussion | **19.2%** |
| Oracle Disclosure | **27.5%** |
| Oracle Canonical State | **27.5%** |
| Canonical State + Fresh | **27.5%** |
| Oracle Plan | **27.5%** |
| Oracle Candidate | **22.5%** |

## 6. Failure Type Distribution

### Free Discussion

| Failure Type | Count |
|---|---|
| invalid_output | 51 |
| information_integration_failure | 5 |
| answer_reason_inconsistency | 3 |
| None | 1 |

### Oracle Disclosure

| Failure Type | Count |
|---|---|
| invalid_output | 45 |
| None | 12 |
| information_integration_failure | 3 |

### Oracle Canonical State

| Failure Type | Count |
|---|---|
| invalid_output | 45 |
| None | 12 |
| information_integration_failure | 3 |

### Canonical State + Fresh

| Failure Type | Count |
|---|---|
| invalid_output | 60 |

### Oracle Plan

| Failure Type | Count |
|---|---|
| invalid_output | 42 |
| None | 15 |
| information_integration_failure | 3 |

### Oracle Candidate

| Failure Type | Count |
|---|---|
| None | 30 |
| answer_reason_inconsistency | 24 |
| answer_selection_failure | 6 |

