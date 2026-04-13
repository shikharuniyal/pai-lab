# lab9


# LAB 9: CNF Conversion & Resolution-Based Inference

**How it works:**
1. Rule `A AND B -> C` becomes CNF clause `[NOT_A, NOT_B, C]`
2. Each known fact is its own clause e.g. `[MultipleFailedLogins]`
3. To prove query Q: add `[NOT_Q]` to KB
4. Keep resolving pairs: cancel opposites (A vs NOT_A), merge the rest
5. Empty clause `[]` appears -> contradiction -> Q is TRUE
6. No new clauses can be made -> Q is FALSE

```python
# Rule to CNF: A AND B -> C  becomes  [NOT_A, NOT_B, C]
def rule_to_cnf(conditions, conclusion):
    clause = ['NOT_' + c for c in conditions]
    clause.append(conclusion)
    return clause

# Flip a literal: NOT_A -> A  and  A -> NOT_A
def negate(lit):
    if lit.startswith('NOT_'):
        return lit[4:]
    return 'NOT_' + lit

# Try to resolve two clauses
# If c1 has A and c2 has NOT_A -> cancel them, merge the rest
def resolve(c1, c2):
    for lit in c1:
        if negate(lit) in c2:
            new = [x for x in c1 if x != lit] + [x for x in c2 if x != negate(lit)]
            return list(dict.fromkeys(new))  # remove duplicates
    return None

# Full resolution engine
def check(clauses, query):
    all_clauses = clauses + [['NOT_' + query]]

    print('CNF Clauses:')
    for i, c in enumerate(all_clauses):
        print(f'  {i+1}. {" OR ".join(c)}')

    print('\nResolution Steps:')
    while True:
        new_clauses = []
        n = len(all_clauses)
        for i in range(n):
            for j in range(i+1, n):
                result = resolve(all_clauses[i], all_clauses[j])
                if result is not None:
                    if result == []:
                        print('  -> Empty clause! Query is TRUE.')
                        return True
                    if result not in all_clauses and result not in new_clauses:
                        print(f'  Derived: {" OR ".join(result)}')
                        new_clauses.append(result)
        if not new_clauses:
            return False
        all_clauses += new_clauses
```

## Q1: Cybersecurity Intrusion Detection

**Rules:**
1. MultipleFailedLogins AND LoginFromUnknownIP -> SuspiciousLogin
2. SuspiciousLogin AND AdminPrivileges -> HighRiskIntrusion
3. HighRiskIntrusion -> IntrusionDetected

```python
cyber_rules = [
    rule_to_cnf(['MultipleFailedLogins', 'LoginFromUnknownIP'], 'SuspiciousLogin'),
    rule_to_cnf(['SuspiciousLogin', 'AdminPrivileges'],          'HighRiskIntrusion'),
    rule_to_cnf(['HighRiskIntrusion'],                           'IntrusionDetected'),
]

# Scenario 1: Only MultipleFailedLogins -> Expected: FALSE
print('=== Scenario 1: Low Risk ===')
result = check(cyber_rules + [['MultipleFailedLogins']], 'IntrusionDetected')
print(f'IntrusionDetected = {result}\n')

# Scenario 2: MultipleFailedLogins + LoginFromUnknownIP -> Expected: FALSE
print('=== Scenario 2: Suspicious Activity ===')
result = check(cyber_rules + [['MultipleFailedLogins'], ['LoginFromUnknownIP']], 'IntrusionDetected')
print(f'IntrusionDetected = {result}\n')

# Scenario 3: All flags raised -> Expected: TRUE
print('=== Scenario 3: High Risk Intrusion ===')
facts = [['MultipleFailedLogins'], ['LoginFromUnknownIP'], ['AdminPrivileges']]
result = check(cyber_rules + facts, 'IntrusionDetected')
print(f'IntrusionDetected = {result}\n')
```

## Q2: Wastewater Treatment Decision

**Rules:**
1. HighBOD AND LowDO -> OrganicPollution
2. HighTurbidity -> Contamination
3. ToxicChemicals -> Contamination
4. OrganicPollution AND Contamination -> SeverePollution
5. SeverePollution -> TreatmentRequired
6. pHImbalance -> TreatmentRequired

```python
water_rules = [
    rule_to_cnf(['HighBOD', 'LowDO'],                 'OrganicPollution'),
    rule_to_cnf(['HighTurbidity'],                     'Contamination'),
    rule_to_cnf(['ToxicChemicals'],                    'Contamination'),
    rule_to_cnf(['OrganicPollution', 'Contamination'], 'SeverePollution'),
    rule_to_cnf(['SeverePollution'],                   'TreatmentRequired'),
    rule_to_cnf(['pHImbalance'],                       'TreatmentRequired'),
]

# Scenario 1: Only HighBOD -> Expected: FALSE
print('=== Scenario 1: HighBOD only ===')
result = check(water_rules + [['HighBOD']], 'TreatmentRequired')
print(f'TreatmentRequired = {result}\n')

# Scenario 2: HighBOD + LowDO + HighTurbidity -> Expected: TRUE
print('=== Scenario 2: HighBOD + LowDO + HighTurbidity ===')
result = check(water_rules + [['HighBOD'], ['LowDO'], ['HighTurbidity']], 'TreatmentRequired')
print(f'TreatmentRequired = {result}\n')

# Scenario 3: ToxicChemicals -> Expected: TRUE
print('=== Scenario 3: ToxicChemicals ===')
result = check(water_rules + [['ToxicChemicals']], 'TreatmentRequired')
print(f'TreatmentRequired = {result}\n')
```