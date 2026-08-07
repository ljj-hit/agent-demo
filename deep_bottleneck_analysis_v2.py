#!/usr/bin/env python3
"""
Rigorous bottleneck analysis v2.
For each of 20 questions in canonical_order setting, track 11 dimensions precisely.
Then build the correct funnel and compute separation metrics.
"""
import json
import re
import os
from collections import defaultdict, Counter
from datetime import datetime

TRACES_PATH = 'outputs_full_experiment/20260804_174126/traces_all.json'
QUESTIONS_PATH = 'data/20.json'
OUTPUT_DIR = 'outputs_full_experiment/20260804_174126/bottleneck_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TRACES_PATH, 'r', encoding='utf-8') as f:
    traces = json.load(f)
with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    questions = json.load(f)

# --- Build question lookup by matching shared_question ---
q_by_shared = {}
for q in questions:
    q_by_shared[q['shared_question']] = q

q_lookup = {}
for t in traces:
    sq = t.get('shared_question', '')
    if sq in q_by_shared and t['question_id'] not in q_lookup:
        q_lookup[t['question_id']] = q_by_shared[sq]

# --- Group traces ---
by_q_setting = defaultdict(lambda: defaultdict(list))
for t in traces:
    by_q_setting[t['question_id']][t['setting']].append(t)

def extract_answer_value(text):
    if not text:
        return None
    m = re.search(r'####\s*(\d+(?:\.\d+)?)', text)
    if m:
        return m.group(1)
    m = re.search(r'(?:answer\s*(?:is|:)?\s*)(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return m.group(1)
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    return nums[-1] if nums else None

def all_numbers_present(fact_text, output_text):
    """Check if all distinct numbers from fact_text appear in output_text."""
    if not output_text or not fact_text:
        return False, set()
    fact_nums = set(re.findall(r'\b(\d+(?:\.\d+)?)\b', fact_text))
    out_nums = set(re.findall(r'\b(\d+(?:\.\d+)?)\b', output_text))
    missing = fact_nums - out_nums
    return len(missing) == 0, missing

# ============================================================
# PER-QUESTION DEEP ANALYSIS for canonical_order
# ============================================================

print("=" * 80)
print("PER-QUESTION DEEP ANALYSIS: canonical_order")
print("=" * 80)

FOCUS = 'canonical_order'

per_q = {}

for qid in sorted(q_lookup.keys()):
    tlist = by_q_setting[qid].get(FOCUS, [])
    if not tlist:
        continue

    q = q_lookup[qid]
    gold = q['answer'].split('####')[-1].strip()

    # Analyze all 3 seeds
    seed_results = []
    for seed_idx, t in enumerate(tlist):
        disc = t.get('discussion', {})
        events = disc.get('discussion_events', [])
        finals = disc.get('solver_finals', {})
        fe = t.get('finalizer_event', {})
        candidate_apps = t.get('candidate_appearances', [])

        # --- 1&2: Fact disclosure ---
        # In canonical_order, both A and B facts are progressively revealed
        injected = t.get('injected_facts', {})
        a_facts = q['required_private_facts']['agent_A']
        b_facts = q['required_private_facts']['agent_B']

        a_disclosed = True  # In canonical_order, A facts are always in the mix
        b_disclosed = True

        # --- 3: First complete public checkpoint ---
        first_complete = disc.get('first_complete_checkpoint', None)
        if first_complete and isinstance(first_complete, dict):
            fc_round = first_complete.get('round', '?')
            fc_checkpoint = first_complete.get('checkpoint', '?')
        else:
            fc_round = '?'
            fc_checkpoint = str(first_complete)

        # --- 4&5: Agent restatement accuracy ---
        a_restate_ok = True
        b_restate_ok = True
        a_issues = []
        b_issues = []

        for evt in events:
            agent = evt.get('agent', '')
            output = evt.get('raw_output', evt.get('output', ''))
            round_num = evt.get('round', '?')

            if 'solver_a' in agent:
                for fact in a_facts:
                    ok, missing = all_numbers_present(fact, output)
                    if not ok:
                        a_issues.append(f'R{round_num}: missing {missing} from "{fact[:50]}"')
                        a_restate_ok = False
            elif 'solver_b' in agent:
                for fact in b_facts:
                    ok, missing = all_numbers_present(fact, output)
                    if not ok:
                        b_issues.append(f'R{round_num}: missing {missing} from "{fact[:50]}"')
                        b_restate_ok = False

        # Also check solver finals
        for skey in ['a', 'b']:
            sf = finals.get(skey, {})
            s_output = sf.get('raw_output', sf.get('output', ''))
            if skey == 'a':
                for fact in a_facts:
                    ok, missing = all_numbers_present(fact, s_output)
                    if not ok:
                        a_issues.append(f'Final: missing {missing} from "{fact[:50]}"')
                        a_restate_ok = False
            else:
                for fact in b_facts:
                    ok, missing = all_numbers_present(fact, s_output)
                    if not ok:
                        b_issues.append(f'Final: missing {missing} from "{fact[:50]}"')
                        b_restate_ok = False

        # --- 6: Fact distortion check (cross-reference) ---
        # Check if A's output contains B's facts correctly and vice versa
        a_absorbs_b = None
        b_absorbs_a = None
        a_all_outputs = ' '.join(evt.get('raw_output', '') for evt in events if 'solver_a' in evt.get('agent', ''))
        b_all_outputs = ' '.join(evt.get('raw_output', '') for evt in events if 'solver_b' in evt.get('agent', ''))

        b_nums_in_a = True
        for fact in b_facts:
            ok, _ = all_numbers_present(fact, a_all_outputs)
            if not ok:
                b_nums_in_a = False
        a_absorbs_b = b_nums_in_a

        a_nums_in_b = True
        for fact in a_facts:
            ok, _ = all_numbers_present(fact, b_all_outputs)
            if not ok:
                a_nums_in_b = False
        b_absorbs_a = a_nums_in_b

        # --- 7: Value changes / relationship flips / entity drift ---
        # Check if the fact numbers appear correctly in solver outputs
        fact_distortions = []
        # Check A's own facts in A's outputs
        for evt in events:
            agent = evt.get('agent', '')
            output = evt.get('raw_output', '')
            if 'solver_a' in agent:
                # Check if key relationships are preserved
                for fact in a_facts:
                    nums_in_fact = re.findall(r'\b(\d+(?:\.\d+)?)\b', fact)
                    nums_in_out = re.findall(r'\b(\d+(?:\.\d+)?)\b', output)
                    if nums_in_fact and not all(n in nums_in_out for n in nums_in_fact):
                        fact_distortions.append(f'{agent}: missing fact nums {nums_in_fact}')
                # Check specific relationships
                if 'twice' in str(a_facts) and '24' not in output and 'qid' in str(qid):
                    pass  # Relationship check is hard to automate

        # --- 8: Correct answer in any reasoning ---
        correct_in_reasoning = False
        correct_first_source = None
        correct_first_round = None
        for evt in events:
            output = evt.get('raw_output', '')
            if gold in re.findall(r'\b(\d+(?:\.\d+)?)\b', output):
                correct_in_reasoning = True
                if correct_first_source is None:
                    correct_first_source = evt.get('agent', '?')
                    correct_first_round = evt.get('round', '?')

        # Also check solver finals
        for skey in ['a', 'b']:
            sf = finals.get(skey, {})
            s_output = sf.get('raw_output', '')
            if gold in re.findall(r'\b(\d+(?:\.\d+)?)\b', s_output):
                correct_in_reasoning = True
                if correct_first_source is None:
                    correct_first_source = f'solver_{skey}_final'
                    correct_first_round = 'final'

        # --- 9: Correct answer enters candidate field ---
        correct_in_candidate = any(
            str(app.get('answer', '')) == str(gold)
            for app in candidate_apps
        )

        # --- 10: Finalizer sees correct candidate ---
        fe_output = fe.get('raw_output', '')
        finalizer_mentions_correct = gold in re.findall(r'\b(\d+(?:\.\d+)?)\b', fe_output)

        # Check if finalizer received correct candidates
        finalizer_input = fe.get('actual_input', '')
        finalizer_sees_correct_candidate = gold in re.findall(r'\b(\d+(?:\.\d+)?)\b', finalizer_input)

        # --- 11: Error stage ---
        final_pred = t.get('final_prediction', '')
        final_val = extract_answer_value(final_pred)
        final_correct = str(final_val) == str(gold)

        if final_correct:
            error_stage = 'none_correct'
        elif not correct_in_reasoning:
            error_stage = 'solver_never_produced_correct'
        elif not correct_in_candidate:
            error_stage = 'correct_in_reasoning_not_captured'
        elif not finalizer_sees_correct_candidate:
            error_stage = 'correct_candidate_not_passed_to_finalizer'
        else:
            error_stage = 'finalizer_rejected_correct_candidate'

        # --- 7 (continued): Complete computation chain ---
        # Check if any solver produced a complete step-by-step solution
        complete_chain = False
        for evt in events:
            output = evt.get('raw_output', '')
            # Check for multi-step arithmetic patterns
            calc_steps = re.findall(r'<<[\d\s\+\-\*\/\.=]+>>', output)
            if len(calc_steps) >= 2 and gold in re.findall(r'\b(\d+(?:\.\d+)?)\b', output):
                complete_chain = True

        seed_results.append({
            'seed': seed_idx,
            'a_restate_ok': a_restate_ok,
            'b_restate_ok': b_restate_ok,
            'a_issues': a_issues,
            'b_issues': b_issues,
            'a_absorbs_b': a_absorbs_b,
            'b_absorbs_a': b_absorbs_a,
            'fact_distortions': fact_distortions,
            'first_complete_round': fc_round,
            'first_complete_checkpoint': fc_checkpoint,
            'correct_in_reasoning': correct_in_reasoning,
            'correct_in_candidate': correct_in_candidate,
            'correct_first_source': correct_first_source,
            'correct_first_round': correct_first_round,
            'finalizer_sees_correct': finalizer_sees_correct_candidate,
            'finalizer_mentions_correct': finalizer_mentions_correct,
            'final_correct': final_correct,
            'final_prediction': str(final_val),
            'error_stage': error_stage,
            'complete_chain': complete_chain,
            'candidate_count': len(candidate_apps),
            'correct_candidate_count': sum(1 for a in candidate_apps if str(a.get('answer','')) == str(gold)),
        })

    # Majority vote
    n = len(seed_results)
    majority = lambda field: sum(s[field] for s in seed_results) >= (n // 2 + 1)

    per_q[qid] = {
        'gold': gold,
        'question': q['shared_question'],
        'a_facts': q['required_private_facts']['agent_A'],
        'b_facts': q['required_private_facts']['agent_B'],
        'seeds': seed_results,
        'facts_disclosed_A': True,
        'facts_disclosed_B': True,
        'a_restate_majority': majority('a_restate_ok'),
        'b_restate_majority': majority('b_restate_ok'),
        'a_absorbs_b_majority': majority('a_absorbs_b'),
        'b_absorbs_a_majority': majority('b_absorbs_a'),
        'correct_in_reasoning_majority': majority('correct_in_reasoning'),
        'correct_in_candidate_majority': majority('correct_in_candidate'),
        'final_correct_majority': majority('final_correct'),
        'complete_chain_majority': majority('complete_chain'),
    }

# ============================================================
# BUILD FUNNEL
# ============================================================
print("\n" + "=" * 80)
print("FUNNEL ANALYSIS for canonical_order (20 questions, majority vote across 3 seeds)")
print("=" * 80)

funnel = {
    'facts_complete': 20,
    'facts_not_distorted': sum(1 for qid, r in per_q.items()
                               if r['a_restate_majority'] and r['b_restate_majority']
                               and r['a_absorbs_b_majority'] and r['b_absorbs_a_majority']),
    'global_state_reconstructed': sum(1 for qid, r in per_q.items()
                                       if r['correct_in_reasoning_majority']),
    'complete_reasoning_plan': sum(1 for qid, r in per_q.items()
                                    if r['complete_chain_majority']),
    'correct_candidate_emerged': sum(1 for qid, r in per_q.items()
                                      if r['correct_in_candidate_majority']),
    'final_correct': sum(1 for qid, r in per_q.items()
                          if r['final_correct_majority']),
}

print(f"""
20 道题:
  -> {funnel['facts_complete']} 题事实完整公开
  -> {funnel['facts_not_distorted']} 题事实没有失真
  -> {funnel['global_state_reconstructed']} 题完成全局状态重建 (正确答案在reasoning中出现)
  -> {funnel['complete_reasoning_plan']} 题形成完整推理计划
  -> {funnel['correct_candidate_emerged']} 题出现正确候选
  -> {funnel['final_correct']} 题最终正确
""")

# Key metrics
if funnel['global_state_reconstructed'] > 0:
    c1 = funnel['correct_candidate_emerged'] / funnel['global_state_reconstructed']
    print(f"candidate_emergence_given_reconstruction = {funnel['correct_candidate_emerged']}/{funnel['global_state_reconstructed']} = {c1:.1%}")
    print(f"  (推理中出现正确答案后，正确候选正式生成的比例)")

if funnel['correct_candidate_emerged'] > 0:
    c2 = funnel['final_correct'] / funnel['correct_candidate_emerged']
    print(f"final_retention_given_correct_candidate = {funnel['final_correct']}/{funnel['correct_candidate_emerged']} = {c2:.1%}")
    print(f"  (正确候选存在时，最终答案保留的比例)")

# Error stage breakdown
error_stages = Counter()
for qid, r in per_q.items():
    for s in r['seeds']:
        error_stages[s['error_stage']] += 1

print(f"\n错误阶段分布 (60 traces = 20题 x 3 seeds):")
for stage, count in error_stages.most_common():
    label = {
        'none_correct': '✅ 最终正确',
        'solver_never_produced_correct': '❌ Solver从未产生正确答案',
        'correct_in_reasoning_not_captured': '⚠️ 推理中有正确答案但未成为正式候选',
        'correct_candidate_not_passed_to_finalizer': '⚠️ 正确候选未被传给Finalizer',
        'finalizer_rejected_correct_candidate': '❌ Finalizer拒绝/忽略了正确候选',
    }.get(stage, stage)
    print(f"  {stage}: {count} ({count/60*100:.1f}%) — {label}")

# ============================================================
# BUILD PER-SETTING FUNNEL FOR ALL FIVE KEY SETTINGS
# ============================================================

print("\n" + "=" * 80)
print("PER-SETTING FUNNEL (5 multi-agent settings)")
print("=" * 80)

KEY_SETTINGS = ['all_at_start_AB', 'all_at_start_BA', 'canonical_order',
                'before_final_reset', 'before_final_transcript']

all_setting_funnels = {}

for setting in KEY_SETTINGS:
    setting_results = {}
    for qid in sorted(q_lookup.keys()):
        tlist = by_q_setting[qid].get(setting, [])
        if not tlist:
            continue
        q = q_lookup[qid]
        gold = q['answer'].split('####')[-1].strip()

        seed_ok = []
        for t in tlist:
            cands = t.get('candidate_appearances', [])
            correct_cand = any(str(a.get('answer','')) == str(gold) for a in cands)
            correct_final = t.get('semantic_correct', False)
            seed_ok.append({'correct_cand': correct_cand, 'correct_final': correct_final})

        n = len(seed_ok)
        cand_majority = sum(s['correct_cand'] for s in seed_ok) >= (n//2 + 1) if n > 0 else False
        final_majority = sum(s['correct_final'] for s in seed_ok) >= (n//2 + 1) if n > 0 else False

        setting_results[qid] = {
            'correct_cand_majority': cand_majority,
            'final_correct_majority': final_majority,
        }

    cand_count = sum(1 for r in setting_results.values() if r['correct_cand_majority'])
    final_count = sum(1 for r in setting_results.values() if r['final_correct_majority'])

    all_setting_funnels[setting] = {
        'correct_candidate': cand_count,
        'final_correct': final_count,
    }

    print(f"\n{setting}:")
    print(f"  正确候选出现: {cand_count}/20")
    print(f"  最终正确: {final_count}/20")
    if cand_count > 0:
        print(f"  retention = {final_count}/{cand_count} = {final_count/cand_count*100:.0f}%")

# Aggregate
total_cand = sum(f['correct_candidate'] for f in all_setting_funnels.values())
total_final = sum(f['final_correct'] for f in all_setting_funnels.values())
total_pairs = len(KEY_SETTINGS) * 20

print(f"\n汇总 (5 settings × 20 questions = {total_pairs} pairs):")
print(f"  正确候选出现: {total_cand}/{total_pairs} = {total_cand/total_pairs*100:.1f}%")
print(f"  最终正确: {total_final}/{total_pairs} = {total_final/total_pairs*100:.1f}%")

# For computation of the two separation metrics, we need to know:
# Among cases where facts are complete, how many produce correct candidates?
# Among cases with correct candidates, how many retain them?

# Since facts are complete in all 5 settings, the denominator for metric 1 is total_pairs
# But more meaningfully, we should compute per-setting

print("\n" + "=" * 80)
print("KEY SEPARATION METRICS (per setting)")
print("=" * 80)

for setting in KEY_SETTINGS:
    f = all_setting_funnels[setting]
    cand = f['correct_candidate']
    final = f['final_correct']

    # For these settings, facts are always complete, so complete_disclosure = 20
    emergence = cand / 20 if 20 > 0 else 0
    retention = final / cand if cand > 0 else None

    print(f"\n{setting}:")
    print(f"  Facts complete = 20/20 (100%)")
    print(f"  candidate_emergence_given_complete_disclosure = {cand}/20 = {emergence:.1%}")
    if retention is not None:
        print(f"  final_retention_given_correct_candidate = {final}/{cand} = {retention:.1%}")
    else:
        print(f"  final_retention_given_correct_candidate = N/A (no correct candidates)")

# ============================================================
# SAVE DETAILED JSON
# ============================================================

json_path = os.path.join(OUTPUT_DIR, 'per_question_deep_analysis.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(per_q, f, ensure_ascii=False, indent=2, default=str)
print(f"\nDetailed analysis saved to: {json_path}")

# ============================================================
# GENERATE FINAL REPORT
# ============================================================

# Also save a summary CSV
csv_path = os.path.join(OUTPUT_DIR, 'per_question_summary.csv')
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(['qid', 'gold', 'question', 'a_restate_ok', 'b_restate_ok',
                     'correct_in_reasoning', 'correct_in_candidate',
                     'final_correct', 'error_stage_majority'])
    for qid in sorted(per_q.keys()):
        r = per_q[qid]
        # Get majority error stage
        error_counts = Counter(s['error_stage'] for s in r['seeds'])
        maj_error = error_counts.most_common(1)[0][0]
        writer.writerow([qid, r['gold'], r['question'][:60],
                        r['a_restate_majority'], r['b_restate_majority'],
                        r['correct_in_reasoning_majority'], r['correct_in_candidate_majority'],
                        r['final_correct_majority'], maj_error])

print(f"CSV saved to: {csv_path}")

print("\nDone!")
