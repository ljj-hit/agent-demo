# ExecGround Error Classification Analysis

**Date**: 2026-08-07 20:45:04
**Data**: 336 traces
**Correct**: 24 (7.1%)
**Errors**: 312

---

## Executive Summary

### Error Distribution by Setting

| Setting | Traces | Correct | Top Errors |
|---------|--------|---------|------------|
| Setting 0: Free Discussion | 60 | 0 | 信息足够但仍输出undetermined (48), 忘记自己的事实 (42), 忽略对方事实 (42) |
| Setting 1: Oracle Disclosure | 60 | 15 | 信息足够但仍输出undetermined (42), 忘记自己的事实 (30), 忽略对方事实 (30) |
| Setting 2: Canonical Ledger | 54 | 0 | 信息足够但仍输出undetermined (54), 忘记自己的事实 (48), 忽略对方事实 (24) |
| Setting 3: Ledger + Fresh Solver | 54 | 3 | 依赖图缺失 (48), 事实数值被修改 (15), 事实披露不完整 (6) |
| Setting 4: Ledger + Executable Plan | 54 | 3 | 依赖图缺失 (45), 事实数值被修改 (15), 只完成局部计算 (6) |
| Setting 5: Ledger + Plan + Verify | 54 | 3 | 依赖图缺失 (45), 事实数值被修改 (15), 算术执行错误 (6) |

---

## Detailed Error Categories

### Error 1: 私有事实没有披露

- **Count**: 21 (6.7% of error traces)
- **Question IDs**: [3, 7, 12, 16, 17, 19, 20]
- **Setting distribution**:
  - Setting 0: Free Discussion: 21
- **Avg earliest error round**: 1.0
- **Oracle fix**: oracle_disclosure (Setting 1) — 程序注入所有事实

#### Typical Traces

- **Q3 seed=1**: pred='' gold='16'
  - undetermined_despite_info: side='a', nums_disclosed=1, nums_total=1
  - private_not_disclosed: agents=['B'], a_disclosed='1/1', b_disclosed='0/1'
  - forgets_own_facts: side='a', own_numbers=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - ignores_other_facts: side='a', other_numbers_available=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - dependency_missing: side='b', reason='all_numbers_available_but_cant_connect', available=[2.0]

- **Q7 seed=1**: pred='960' gold='1200'
  - undetermined_despite_info: side='a', nums_disclosed=3, nums_total=4
  - private_not_disclosed: agents=['B'], a_disclosed='3/3', b_disclosed='0/1'
  - forgets_own_facts: side='a', own_numbers=[8.0, 30.0, 60.0], solver_output_snippet="Final answer: Undetermined\n\nTo determine Noah's sales revenue for this month, we need additional information about the number of small paintings sold last month. Without this detail, it's impossible t"
  - ignores_other_facts: side='b', other_numbers_available=[8.0, 30.0, 60.0], solver_output_snippet="Current answer: Undetermined\n\nTo solve this problem, we need additional information about Noah's sales performance last month. Specifically, we require the number of small paintings sold last month. O"

- **Q12 seed=1**: pred='' gold='3'
  - private_not_disclosed: agents=['B'], a_disclosed='1/1', b_disclosed='0/3'
  - forgets_own_facts: side='a', own_numbers=[13.0], solver_output_snippet='Final answer: Undetermined\n\nTo determine the number of packs Roger needs, we would need to know either the total number of trail-mix pouches required (one for each player plus possibly coaches and hel'
  - ignores_other_facts: side='a', other_numbers_available=[2.0], solver_output_snippet='Final answer: Undetermined\n\nTo determine the number of packs Roger needs, we would need to know either the total number of trail-mix pouches required (one for each player plus possibly coaches and hel'

---

### Error 2: 事实披露不完整

- **Count**: 27 (8.7% of error traces)
- **Question IDs**: [7, 11]
- **Setting distribution**:
  - Setting 2: Canonical Ledger: 6
  - Setting 0: Free Discussion: 3
  - Setting 4: Ledger + Executable Plan: 6
  - Setting 5: Ledger + Plan + Verify: 6
  - Setting 3: Ledger + Fresh Solver: 6
- **Avg earliest error round**: 1.0
- **Oracle fix**: oracle_disclosure (Setting 1) — 程序注入所有事实

#### Typical Traces

- **Q11 seed=1**: pred='780' gold='520'
  - undetermined_despite_info: side='a', nums_disclosed=3, nums_total=4
  - incomplete_disclosure: a_coverage='2/2', b_coverage='1/2', nums_a=[20.0, 1000.0]
  - forgets_own_facts: side='a', own_numbers=[20.0, 1000.0], solver_output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far'
  - ignores_other_facts: side='b', other_numbers_available=[20.0, 1000.0], solver_output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far'
  - partial_computation: side='a', numbers_computed=[4.0, 4.0], output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far does not include details about the initial balance or the nature of the payments (whether they cove'

- **Q11 seed=2**: pred='780' gold='520'
  - undetermined_despite_info: side='a', nums_disclosed=3, nums_total=4
  - incomplete_disclosure: a_coverage='2/2', b_coverage='1/2', nums_a=[20.0, 1000.0]
  - forgets_own_facts: side='a', own_numbers=[20.0, 1000.0], solver_output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far'
  - ignores_other_facts: side='b', other_numbers_available=[20.0, 1000.0], solver_output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far'
  - partial_computation: side='a', numbers_computed=[4.0, 4.0], output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far does not include details about the initial balance or the nature of the payments (whether they cove'

- **Q11 seed=3**: pred='780' gold='520'
  - undetermined_despite_info: side='a', nums_disclosed=3, nums_total=4
  - incomplete_disclosure: a_coverage='2/2', b_coverage='1/2', nums_a=[20.0, 1000.0]
  - forgets_own_facts: side='a', own_numbers=[20.0, 1000.0], solver_output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far'
  - ignores_other_facts: side='b', other_numbers_available=[20.0, 1000.0], solver_output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far'
  - partial_computation: side='a', numbers_computed=[4.0, 4.0], output_snippet='Current answer: undetermined\n\nTo accurately determine the remaining balance after 4 monthly payments, we need to know the initial balance before any payments were made. The information provided so far does not include details about the initial balance or the nature of the payments (whether they cove'

---

### Error 3: 忘记自己的事实

- **Count**: 120 (38.5% of error traces)
- **Question IDs**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- **Setting distribution**:
  - Setting 2: Canonical Ledger: 48
  - Setting 0: Free Discussion: 42
  - Setting 1: Oracle Disclosure: 30
- **Avg earliest error round**: 1.0
- **Oracle fix**: oracle_canonical_state (Setting 2) — 结构化事实表提醒

#### Typical Traces

- **Q1 seed=1**: pred='' gold='42'
  - undetermined_despite_info: side='a', nums_disclosed=2, nums_total=2
  - forgets_own_facts: side='a', own_numbers=[12.0], solver_output_snippet='Final answer: The number of pages Julie should read tomorrow cannot be determined without knowing the number of pages she has already read today and yesterday.'
  - ignores_other_facts: side='a', other_numbers_available=[120.0], solver_output_snippet='Final answer: The number of pages Julie should read tomorrow cannot be determined without knowing the number of pages she has already read today and yesterday.'

- **Q3 seed=1**: pred='' gold='16'
  - undetermined_despite_info: side='a', nums_disclosed=1, nums_total=1
  - private_not_disclosed: agents=['B'], a_disclosed='1/1', b_disclosed='0/1'
  - forgets_own_facts: side='a', own_numbers=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - ignores_other_facts: side='a', other_numbers_available=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - dependency_missing: side='b', reason='all_numbers_available_but_cant_connect', available=[2.0]

- **Q4 seed=1**: pred='155' gold='41'
  - undetermined_despite_info: side='a', nums_disclosed=7, nums_total=7
  - forgets_own_facts: side='b', own_numbers=[11.0, 16.0, 18.0], solver_output_snippet='Final answer: Alexis paid $155 for the shoes.'
  - ignores_other_facts: side='b', other_numbers_available=[30.0, 38.0, 46.0, 200.0], solver_output_snippet='Final answer: Alexis paid $155 for the shoes.'
  - partial_computation: side='a', numbers_computed=[155.0, 200.0, 30.0, 46.0, 38.0], output_snippet='Final answer: Alexis paid $155 for the shoes.\n\nExplanation: \nFrom the shared information, we know Alexis had a $200 clothing budget. Her expenses include a $30 shirt, a $46 pair of pants, and a $38 coat. To find out how much she paid for the shoes, we subtract the known amounts spent on shirts, pant'

---

### Error 4: 忽略对方事实

- **Count**: 96 (30.8% of error traces)
- **Question IDs**: [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
- **Setting distribution**:
  - Setting 2: Canonical Ledger: 24
  - Setting 0: Free Discussion: 42
  - Setting 1: Oracle Disclosure: 30
- **Avg earliest error round**: 1.0
- **Oracle fix**: oracle_canonical_state (Setting 2) — 统一事实视图

#### Typical Traces

- **Q1 seed=1**: pred='' gold='42'
  - undetermined_despite_info: side='a', nums_disclosed=2, nums_total=2
  - forgets_own_facts: side='a', own_numbers=[12.0], solver_output_snippet='Final answer: The number of pages Julie should read tomorrow cannot be determined without knowing the number of pages she has already read today and yesterday.'
  - ignores_other_facts: side='a', other_numbers_available=[120.0], solver_output_snippet='Final answer: The number of pages Julie should read tomorrow cannot be determined without knowing the number of pages she has already read today and yesterday.'

- **Q3 seed=1**: pred='' gold='16'
  - undetermined_despite_info: side='a', nums_disclosed=1, nums_total=1
  - private_not_disclosed: agents=['B'], a_disclosed='1/1', b_disclosed='0/1'
  - forgets_own_facts: side='a', own_numbers=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - ignores_other_facts: side='a', other_numbers_available=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - dependency_missing: side='b', reason='all_numbers_available_but_cant_connect', available=[2.0]

- **Q4 seed=1**: pred='155' gold='41'
  - undetermined_despite_info: side='a', nums_disclosed=7, nums_total=7
  - forgets_own_facts: side='b', own_numbers=[11.0, 16.0, 18.0], solver_output_snippet='Final answer: Alexis paid $155 for the shoes.'
  - ignores_other_facts: side='b', other_numbers_available=[30.0, 38.0, 46.0, 200.0], solver_output_snippet='Final answer: Alexis paid $155 for the shoes.'
  - partial_computation: side='a', numbers_computed=[155.0, 200.0, 30.0, 46.0, 38.0], output_snippet='Final answer: Alexis paid $155 for the shoes.\n\nExplanation: \nFrom the shared information, we know Alexis had a $200 clothing budget. Her expenses include a $30 shirt, a $46 pair of pants, and a $38 coat. To find out how much she paid for the shoes, we subtract the known amounts spent on shirts, pant'

---

### Error 5: 事实数值被修改

- **Count**: 60 (19.2% of error traces)
- **Question IDs**: [4, 8, 10, 13, 20]
- **Setting distribution**:
  - Setting 2: Canonical Ledger: 15
  - Setting 4: Ledger + Executable Plan: 15
  - Setting 5: Ledger + Plan + Verify: 15
  - Setting 3: Ledger + Fresh Solver: 15
- **Avg earliest error round**: 1.0
- **Oracle fix**: oracle_canonical_state (Setting 2) — 规范化防止修改

#### Typical Traces

- **Q4 seed=1**: pred='' gold='41'
  - undetermined_despite_info: side='a', nums_disclosed=0, nums_total=7
  - forgets_own_facts: side='a', own_numbers=[30.0, 38.0, 46.0, 200.0], solver_output_snippet='Final answer: Undetermined\n\nThe provided information does not include the cost of the shoes that Alexis bought. To determine the amount Alexis paid for the shoes, we would need more specific informati'
  - fact_values_modified: hallucinated_value=-7.0, evidence='After buying a pair of shoes, she has $16 left.'

- **Q8 seed=1**: pred='' gold='1920'
  - undetermined_despite_info: side='a', nums_disclosed=1, nums_total=4
  - forgets_own_facts: side='a', own_numbers=[3.0, 20.0], solver_output_snippet='Final answer: Undetermined\n\nInsufficient information to calculate the total number of minutes Carolyn practices in 4 weeks. More details about her daily practice time and whether she alternates betwee'
  - fact_values_modified: fact={'fact_id': 'A_F2', 'subject': 'violin_practice_time', 'relation': 'half_of', 'object': 'piano_practice_time', 'value': 10.0, 'unit': 'minutes', 'source': 'A', 'evidence': 'She practices violin for 3 times as long as piano.'}, issue='half_of relation stored with unit=minutes value=10.0'

- **Q10 seed=1**: pred='' gold='840'
  - undetermined_despite_info: side='a', nums_disclosed=2, nums_total=2
  - forgets_own_facts: side='a', own_numbers=[20.0, 30.0], solver_output_snippet="Final answer: Undetermined\n\nWithout knowing the distribution of hours between the primary and secondary jobs, it's impossible to calculate James' weekly earnings from both jobs accurately."
  - ignores_other_facts: side='a', other_numbers_available=[20.0], solver_output_snippet="Final answer: Undetermined\n\nWithout knowing the distribution of hours between the primary and secondary jobs, it's impossible to calculate James' weekly earnings from both jobs accurately."
  - fact_values_modified: hallucinated_value=0.2, evidence='At his second job, he earns 20% less per hour'

---

### Error 8: 只完成局部计算

- **Count**: 81 (26.0% of error traces)
- **Question IDs**: [2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- **Setting distribution**:
  - Setting 2: Canonical Ledger: 9
  - Setting 0: Free Discussion: 42
  - Setting 4: Ledger + Executable Plan: 6
  - Setting 5: Ledger + Plan + Verify: 6
  - Setting 3: Ledger + Fresh Solver: 3
  - Setting 1: Oracle Disclosure: 15
- **Avg earliest error round**: 1.0
- **Oracle fix**: oracle_plan (Setting 4) — 提供完整方程依赖图

#### Typical Traces

- **Q2 seed=1**: pred='' gold='5'
  - undetermined_despite_info: side='a', nums_disclosed=2, nums_total=2
  - partial_computation: side='a', numbers_computed=[100.0, 50.0, 15.0, 30.0, 45.0], output_snippet='Final answer: Undetermined\n\nExplanation: \nFrom the public transcript, we have the following information:\n- Betty needs $100 to buy a wallet.\n- She already has half of the money needed, which is $50.\n- Betty has received $15 from her parents and $30 from her grandparents, totaling $45.\n\nHowever, the '
  - dependency_missing: side='a', reason='all_numbers_available_but_cant_connect', available=[15.0, 100.0]

- **Q4 seed=1**: pred='155' gold='41'
  - undetermined_despite_info: side='a', nums_disclosed=7, nums_total=7
  - forgets_own_facts: side='b', own_numbers=[11.0, 16.0, 18.0], solver_output_snippet='Final answer: Alexis paid $155 for the shoes.'
  - ignores_other_facts: side='b', other_numbers_available=[30.0, 38.0, 46.0, 200.0], solver_output_snippet='Final answer: Alexis paid $155 for the shoes.'
  - partial_computation: side='a', numbers_computed=[155.0, 200.0, 30.0, 46.0, 38.0], output_snippet='Final answer: Alexis paid $155 for the shoes.\n\nExplanation: \nFrom the shared information, we know Alexis had a $200 clothing budget. Her expenses include a $30 shirt, a $46 pair of pants, and a $38 coat. To find out how much she paid for the shoes, we subtract the known amounts spent on shirts, pant'

- **Q5 seed=1**: pred='' gold='38'
  - undetermined_despite_info: side='a', nums_disclosed=3, nums_total=3
  - partial_computation: side='a', numbers_computed=[34.0, 11.0, 20.0, 13.0, 13.0], output_snippet="Final answer: Bella buys 34 stamps altogether.\n\nExplanation: \n- From my private fact, we know Bella buys 11 snowflake stamps and 20 truck stamps.\n- Using the relationship from solver B's message, if we let \\( T \\) represent the number of truck stamps, then the number of rose stamps is \\( T - 13 \\).\n"

---

### Error 9: 依赖图缺失

- **Count**: 162 (51.9% of error traces)
- **Question IDs**: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- **Setting distribution**:
  - Setting 2: Canonical Ledger: 9
  - Setting 0: Free Discussion: 15
  - Setting 4: Ledger + Executable Plan: 45
  - Setting 5: Ledger + Plan + Verify: 45
  - Setting 3: Ledger + Fresh Solver: 48
- **Avg earliest error round**: 1.0
- **Oracle fix**: oracle_plan (Setting 4) — 提供方程结构

#### Typical Traces

- **Q2 seed=1**: pred='' gold='5'
  - undetermined_despite_info: side='a', nums_disclosed=2, nums_total=2
  - partial_computation: side='a', numbers_computed=[100.0, 50.0, 15.0, 30.0, 45.0], output_snippet='Final answer: Undetermined\n\nExplanation: \nFrom the public transcript, we have the following information:\n- Betty needs $100 to buy a wallet.\n- She already has half of the money needed, which is $50.\n- Betty has received $15 from her parents and $30 from her grandparents, totaling $45.\n\nHowever, the '
  - dependency_missing: side='a', reason='all_numbers_available_but_cant_connect', available=[15.0, 100.0]

- **Q3 seed=1**: pred='' gold='16'
  - undetermined_despite_info: side='a', nums_disclosed=1, nums_total=1
  - private_not_disclosed: agents=['B'], a_disclosed='1/1', b_disclosed='0/1'
  - forgets_own_facts: side='a', own_numbers=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - ignores_other_facts: side='a', other_numbers_available=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - dependency_missing: side='b', reason='all_numbers_available_but_cant_connect', available=[2.0]

- **Q9 seed=1**: pred='75' gold='45'
  - undetermined_despite_info: side='a', nums_disclosed=2, nums_total=2
  - partial_computation: side='a', numbers_computed=[1.0, 25.0, 2.0, 1.0, 2.0], output_snippet='Final answer: Undetermined\n\nExplanation: \nWe have the following known values:\n- Part 1 takes 25 minutes.\n- Part 2 takes twice as long as Part 1, which means it takes \\(2 \\times 25 = 50\\) minutes.\n- The total time for the entire assignment is 2 hours (which is 120 minutes).\n\nHowever, we do not have a'
  - dependency_missing: side='b', reason='all_numbers_available_but_cant_connect', available=[2.0, 25.0]

---

### Error 10: 算术执行错误

- **Count**: 6 (1.9% of error traces)
- **Question IDs**: [2, 8]
- **Setting distribution**:
  - Setting 5: Ledger + Plan + Verify: 6
- **Oracle fix**: oracle_plan (Setting 4) — 明确计算步骤

#### Typical Traces

- **Q2 seed=1**: pred='-115.0' gold='5'
  - arithmetic_error: computed=-115.0, expected='5', setting='ledger_exec_plan_verify'
  - integration_failure: fresh_prediction='-115.0', gold=5.0, reasoning_snippet='{\n  "steps": [\n    {\n      "op": "subtract",\n      "inputs": ["F1", "F2"],\n      "output": "wallet_balance",\n      "explanation": "Wallet balance = wallet cost - money needed"\n    },\n    {\n      "op": "multiply",\n      "inputs": ["wallet_balance", "F3"],\n      "output": "additional_money",\n      "ex'

- **Q8 seed=1**: pred='144.0' gold='1920'
  - fact_values_modified: fact={'fact_id': 'A_F2', 'subject': 'violin_practice_time', 'relation': 'half_of', 'object': 'piano_practice_time', 'value': 10.0, 'unit': 'minutes', 'source': 'A', 'evidence': 'She practices violin for 3 times as long as piano.'}, issue='half_of relation stored with unit=minutes value=10.0'
  - arithmetic_error: computed=144.0, expected='1920', setting='ledger_exec_plan_verify'
  - integration_failure: fresh_prediction='144.0', gold=1920.0, reasoning_snippet='{\n  "steps": [\n    {\n      "op": "multiply",\n      "inputs": ["F1", "F3"],\n      "output": "piano_daily_practice",\n      "explanation": "Piano daily practice time per day = piano_practice_time"\n    },\n    {\n      "op": "multiply",\n      "inputs": ["F2", "F3"],\n      "output": "violin_daily_practice"'

- **Q2 seed=2**: pred='-115.0' gold='5'
  - arithmetic_error: computed=-115.0, expected='5', setting='ledger_exec_plan_verify'
  - integration_failure: fresh_prediction='-115.0', gold=5.0, reasoning_snippet='{\n  "steps": [\n    {\n      "op": "subtract",\n      "inputs": ["F1", "F2"],\n      "output": "wallet_balance",\n      "explanation": "Wallet balance = wallet cost - money needed"\n    },\n    {\n      "op": "multiply",\n      "inputs": ["wallet_balance", "F3"],\n      "output": "additional_money",\n      "ex'

---

### Error 13: 信息足够但仍输出undetermined

- **Count**: 144 (46.2% of error traces)
- **Question IDs**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- **Setting distribution**:
  - Setting 2: Canonical Ledger: 54
  - Setting 0: Free Discussion: 48
  - Setting 1: Oracle Disclosure: 42
- **Avg earliest error round**: 1.0
- **Oracle fix**: canonical_state_fresh (Setting 3) — 清除讨论历史，直接求解

#### Typical Traces

- **Q1 seed=1**: pred='' gold='42'
  - undetermined_despite_info: side='a', nums_disclosed=2, nums_total=2
  - forgets_own_facts: side='a', own_numbers=[12.0], solver_output_snippet='Final answer: The number of pages Julie should read tomorrow cannot be determined without knowing the number of pages she has already read today and yesterday.'
  - ignores_other_facts: side='a', other_numbers_available=[120.0], solver_output_snippet='Final answer: The number of pages Julie should read tomorrow cannot be determined without knowing the number of pages she has already read today and yesterday.'

- **Q2 seed=1**: pred='' gold='5'
  - undetermined_despite_info: side='a', nums_disclosed=2, nums_total=2
  - partial_computation: side='a', numbers_computed=[100.0, 50.0, 15.0, 30.0, 45.0], output_snippet='Final answer: Undetermined\n\nExplanation: \nFrom the public transcript, we have the following information:\n- Betty needs $100 to buy a wallet.\n- She already has half of the money needed, which is $50.\n- Betty has received $15 from her parents and $30 from her grandparents, totaling $45.\n\nHowever, the '
  - dependency_missing: side='a', reason='all_numbers_available_but_cant_connect', available=[15.0, 100.0]

- **Q3 seed=1**: pred='' gold='16'
  - undetermined_despite_info: side='a', nums_disclosed=1, nums_total=1
  - private_not_disclosed: agents=['B'], a_disclosed='1/1', b_disclosed='0/1'
  - forgets_own_facts: side='a', own_numbers=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - ignores_other_facts: side='a', other_numbers_available=[2.0], solver_output_snippet='Final answer: The final weight of the box cannot be determined with the given information. More details about the number of jelly beans added or the number of gummy worms added after each additional p'
  - dependency_missing: side='b', reason='all_numbers_available_but_cant_connect', available=[2.0]

---

---

### Error 6: half/twice等关系翻转

- **Count**: 0 (0.0% of error traces)
- **Question IDs**: N/A
- **Note**: 在本次20题 Qwen2.5-1.5B 实验中未观察到该错误模式。关系翻转需要模型在推理中明确使用 half/twice 等关系词并产生可识别的反向计算，但此模型倾向于直接跳过关系推理或完全无法完成多步计算，因而该类别未被触发。更大模型或包含显式关系推理的题目中可能会出现。
- **Oracle fix**: oracle_plan (Setting 4) — 方程结构明确关系方向

---

### Error 7: 实体或题意漂移

- **Count**: 0 (0.0% of error traces)
- **Question IDs**: N/A
- **Note**: 在本次实验中未检测到显著的题意漂移。虽然模型频繁输出错误答案，但通常仍在尝试回答原始问题——错误源于无法正确组合事实，而非理解错误的问题。Qwen2.5-1.5B 在保持问题理解方面表现稳定。
- **Oracle fix**: canonical_state_fresh (Setting 3) — 清除讨论历史

---

### Error 11: reasoning中正确但候选字段错误

- **Count**: 0 (0.0% of error traces)
- **Question IDs**: N/A
- **Note**: 未发现推理文本中包含正确答案但候选字段输出不同数字的情况。模型的行为模式是：要么整个推理过程都错误（输出错误答案），要么因无法完成推理而输出 undetermined。不存在正确答案在 reasoning 中出现后被 answer 字段覆盖的模式。
- **Oracle fix**: oracle_candidate (Setting 5) — 正确候选直接注入

---

### Error 12: 正确候选存在但被verifier/finalizer丢失

- **Count**: 0 (0.0% of error traces)
- **Question IDs**: N/A
- **Note**: 当正确答案在 candidate_appearances 中出现时（即 solver 输出了正确数字），它总是被 finalizer 正确选出。24 条正确 trace 无一例外。这说明对于 Qwen2.5-1.5B，Finalizer 不是一个独立的瓶颈——问题完全出在 Solver 侧无法产生正确候选。
- **Oracle fix**: oracle_candidate (Setting 5) — 标注权威候选

---

## Causal Attribution Summary

Based on the error classification, the bottlenecks by prevalence:

| Rank | Error Category | Count | % of Errors | Primary Oracle Fix |
|------|---------------|-------|-------------|-------------------|
| 1 | 依赖图缺失 | 162 | 51.9% | oracle_plan (Setting 4) |
| 2 | 信息足够但仍输出undetermined | 144 | 46.2% | canonical_state_fresh (Setting 3) |
| 3 | 忘记自己的事实 | 120 | 38.5% | oracle_canonical_state (Setting 2) |
| 4 | 忽略对方事实 | 96 | 30.8% | oracle_canonical_state (Setting 2) |
| 5 | 只完成局部计算 | 81 | 26.0% | oracle_plan (Setting 4) |
| 6 | 事实数值被修改 | 60 | 19.2% | oracle_canonical_state (Setting 2) |
| 7 | 事实披露不完整 | 27 | 8.7% | oracle_disclosure (Setting 1) |
| 8 | 私有事实没有披露 | 21 | 6.7% | oracle_disclosure (Setting 1) |
| 9 | 算术执行错误 | 6 | 1.9% | oracle_plan (Setting 4) |
| 10 | half/twice等关系翻转 | 0 | 0.0% | 模型未达到产生此错误的复杂度 |
| 11 | 实体或题意漂移 | 0 | 0.0% | 模型保持问题理解稳定 |
| 12 | reasoning正确但候选字段错误 | 0 | 0.0% | 正确数字不出现在reasoning中 |
| 13 | 正确候选存在但被丢失 | 0 | 0.0% | Finalizer非独立瓶颈 |

---

## Interpretation Guide

1. **Setting 0→1 gap** (free_discussion → oracle_disclosure): +25% accuracy.
   Reflects fact disclosure failures (私有事实未披露, 披露不完整).
   21 traces (6.7%) had completely undisclosed private facts — oracle recovery = 25%.

2. **Setting 1→2 gap** (oracle_disclosure → canonical_ledger): -25% accuracy.
   The canonical ledger with discussion protocol actually performs WORSE.
   Root cause: discussion format causes all solvers to output "undetermined" even with all facts.
   144 traces (46.2%) had sufficient information but still said undetermined.

3. **Setting 2→3 gap** (canonical_ledger → ledger_fresh_solver): +5.6% accuracy.
   Bypassing discussion history helps marginally. Discussion contamination confirmed.
   But 48/54 setting-3 traces still fail due to dependency graph missing (依赖图缺失).

4. **Setting 3→4 gap** (fresh_solver → executable_plan): no improvement.
   Model cannot generate semantically correct JSON-IR plans. 45/54 plans fail execution.
   Executable plan layer adds zero value for this model.

5. **Setting 4→5 gap** (executable_plan → plan+verify): no improvement.
   Coverage verification detects errors but the model cannot fix them.
   Fix loop produces identical broken plans each round.

6. **Categories 6-7, 11-12 are absent** (0 instances):
   Relation inversion, entity drift, correct-reasoning-wrong-field, and correct-candidate-lost
   do NOT occur in this dataset. The bottlenecks are earlier in the pipeline:
   the model never reaches the stage where these error types can manifest.
   For Qwen2.5-1.5B, the dominant failure is: can't connect facts → undetermined.
