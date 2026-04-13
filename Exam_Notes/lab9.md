# lab9


# LAB 9: CNF Conversion & Resolution-Based Inference

**Concept:** Convert rules to CNF clauses, then apply resolution to prove/disprove a query.

**Key idea:**
- Implication A→B becomes clause {¬A, B}
- To prove Q: add ¬Q to KB, if resolution derives empty clause → Q is TRUE


## Helper: CNF Conversion & Resolution Engine


```python
def implication_to_cnf(conditions, conclusion):
    """A∧B→C  becomes  {¬A, ¬B, C}"""
    negated = ['¬' + c if not c.startswith('¬') else c[1:] for c in conditions]
    return frozenset(negated + [conclusion])

def resolve(c1, c2):
    """Try to resolve two clauses. Returns new clause or None."""
    for lit in c1:
        neg = lit[1:] if lit.startswith('¬') else '¬' + lit
        if neg in c2:
            new_clause = (c1 - {lit}) | (c2 - {neg})
            return frozenset(new_clause)
    return None

def resolution(clauses, query):
    """Add ¬query, apply resolution. Empty clause = query is TRUE."""
    neg_query = frozenset(['¬' + query if not query.startswith('¬') else query[1:]])
    all_clauses = set(clauses) | {neg_query}

    print("CNF Clauses:")
    for i, c in enumerate(all_clauses):
        print(f"  {i+1}. {' ∨ '.join(c)}")

    print("\nResolution Steps:")
    clause_list = list(all_clauses)
    while True:
        new_clauses = set()
        for i in range(len(clause_list)):
            for j in range(i+1, len(clause_list)):
                resolvent = resolve(clause_list[i], clause_list[j])
                if resolvent is not None:
                    if len(resolvent) == 0:
                        print("  → Empty clause derived!")
                        return True
                    if resolvent not in all_clauses:
                        print(f"  Derived: {' ∨ '.join(resolvent) if resolvent else '∅'}")
                        new_clauses.add(resolvent)
        if not new_clauses:
            return False  # no new clauses = can't prove
        all_clauses |= new_clauses
        clause_list = list(all_clauses)

```

## Q1: Cybersecurity Intrusion Detection

**Rules:**
1. MultipleFailedLogins ∧ LoginFromUnknownIP → SuspiciousLogin
2. SuspiciousLogin ∧ AdminPrivileges → HighRiskIntrusion
3. HighRiskIntrusion → IntrusionDetected
4. AccessToSensitiveFiles ∧ ¬AdminPrivileges → IntrusionDetected


```python
# Build CNF clauses from rules
cyber_clauses = [
    implication_to_cnf(['MultipleFailedLogins', 'LoginFromUnknownIP'], 'SuspiciousLogin'),
    implication_to_cnf(['SuspiciousLogin', 'AdminPrivileges'], 'HighRiskIntrusion'),
    implication_to_cnf(['HighRiskIntrusion'], 'IntrusionDetected'),
]

# Scenario 1: Low Risk - only MultipleFailedLogins
print("=== Scenario 1: Low Risk ===")
facts1 = [frozenset(['MultipleFailedLogins'])]
result = resolution(cyber_clauses + facts1, 'IntrusionDetected')
print(f"IntrusionDetected = {result}\n")

```

```python
# Scenario 2: Suspicious but not confirmed
print("=== Scenario 2: Suspicious Activity ===")
facts2 = [frozenset(['MultipleFailedLogins']), frozenset(['LoginFromUnknownIP'])]
result = resolution(cyber_clauses + facts2, 'IntrusionDetected')
print(f"IntrusionDetected = {result}\n")

```

```python
# Scenario 3: High Risk - all flags raised
print("=== Scenario 3: High Risk Intrusion ===")
facts3 = [
    frozenset(['MultipleFailedLogins']),
    frozenset(['LoginFromUnknownIP']),
    frozenset(['AdminPrivileges'])
]
result = resolution(cyber_clauses + facts3, 'IntrusionDetected')
print(f"IntrusionDetected = {result}\n")

```

## Q2: Wastewater Treatment Decision

**Rules:**
1. HighBOD ∧ LowDO → OrganicPollution
2. HighTurbidity → Contamination
3. ToxicChemicals → Contamination
4. OrganicPollution ∧ Contamination → SeverePollution
5. SeverePollution → TreatmentRequired
6. pHImbalance → TreatmentRequired


```python
water_clauses = [
    implication_to_cnf(['HighBOD', 'LowDO'], 'OrganicPollution'),
    implication_to_cnf(['HighTurbidity'], 'Contamination'),
    implication_to_cnf(['ToxicChemicals'], 'Contamination'),
    implication_to_cnf(['OrganicPollution', 'Contamination'], 'SeverePollution'),
    implication_to_cnf(['SeverePollution'], 'TreatmentRequired'),
    implication_to_cnf(['pHImbalance'], 'TreatmentRequired'),
]

# Scenario 1: Only HighBOD → FALSE
print("=== Scenario 1: HighBOD only ===")
result = resolution(water_clauses + [frozenset(['HighBOD'])], 'TreatmentRequired')
print(f"TreatmentRequired = {result}\n")

# Scenario 2: HighBOD, LowDO, HighTurbidity → TRUE
print("=== Scenario 2: HighBOD + LowDO + HighTurbidity ===")
facts2 = [frozenset(['HighBOD']), frozenset(['LowDO']), frozenset(['HighTurbidity'])]
result = resolution(water_clauses + facts2, 'TreatmentRequired')
print(f"TreatmentRequired = {result}\n")

# Scenario 3: ToxicChemicals → TRUE
print("=== Scenario 3: ToxicChemicals ===")
result = resolution(water_clauses + [frozenset(['ToxicChemicals'])], 'TreatmentRequired')
print(f"TreatmentRequired = {result}\n")

```