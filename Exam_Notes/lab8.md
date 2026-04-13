# lab8


# LAB 8: Logical Agents

**Concept:** Rules stored in a knowledge base. Agent infers actions using forward chaining (if A and B then C).


## Q1: Rule-Based Logical Agent for Smart Lighting

Rules:
- IsDark AND MotionDetected → TurnLightON
- NOT MotionDetected → TurnLightOFF
- ManualOverride → ignore automatic rules


```python
def smart_light(MotionDetected, IsDark, ManualOverride):
    if ManualOverride:
        return "Manual Override: Automatic rules ignored"
    if not MotionDetected:
        return "Action: TurnLightOFF"
    if IsDark and MotionDetected:
        return "Action: TurnLightON"
    return "Action: TurnLightOFF"

# Test scenarios
tests = [
    (True, True, False),
    (False, True, False),
    (True, False, False),
    (True, True, True),
]
for m, d, o in tests:
    print(f"Motion={m}, Dark={d}, Override={o} → {smart_light(m, d, o)}")

```

## Q2: Medical Diagnosis Logical Agent

Rules:
- Fever AND Cough → Flu
- Fever AND Cough AND ChestPain → Pneumonia
- Pneumonia → HospitalizationRequired


```python
def medical_agent(facts):
    kb = set(facts)

    # Forward chaining rules
    rules = [
        ({'Fever', 'Cough'}, 'Flu'),
        ({'Fever', 'Cough', 'ChestPain'}, 'Pneumonia'),
        ({'Pneumonia'}, 'HospitalizationRequired'),
    ]

    changed = True
    while changed:
        changed = False
        for conditions, conclusion in rules:
            if conditions.issubset(kb) and conclusion not in kb:
                kb.add(conclusion)
                changed = True

    return kb

# Test
symptoms1 = ['Fever', 'Cough']
symptoms2 = ['Fever', 'Cough', 'ChestPain']

print("Case 1:", medical_agent(symptoms1))
print("Case 2:", medical_agent(symptoms2))

print("""
Difference from ML classifier:
- Logical agent uses explicit rules → decisions are explainable
- ML classifier learns patterns from data → black box
""")

```

## Q3: Traffic Violation Detection

Rules:
- SpeedAboveLimit → SpeedViolation
- SignalRed AND VehicleCrossedLine → SignalViolation
- EmergencyVehicle → no violation (exception)


```python
def traffic_agent(percepts):
    kb = set(percepts)
    violations = []

    if 'EmergencyVehicle' in kb:
        return "No violation (Emergency Vehicle exception)"

    if 'SpeedAboveLimit' in kb:
        kb.add('SpeedViolation')
        violations.append('SpeedViolation')

    if 'SignalRed' in kb and 'VehicleCrossedLine' in kb:
        kb.add('SignalViolation')
        violations.append('SignalViolation')

    return f"Violations: {violations}" if violations else "No violations"

# Scenarios over time
scenarios = [
    ['SpeedAboveLimit'],
    ['SignalRed', 'VehicleCrossedLine'],
    ['SpeedAboveLimit', 'SignalRed', 'VehicleCrossedLine'],
    ['SignalRed', 'EmergencyVehicle'],
]
for s in scenarios:
    print(f"Percepts: {s} → {traffic_agent(s)}")

```

## Q4: Cybersecurity Intrusion Detection

Rules:
- MultipleFailedLogins AND LoginFromUnknownIP → SuspiciousLogin
- SuspiciousLogin AND AdminPrivileges → HighRiskIntrusion
- HighRiskIntrusion → TriggerAlert


```python
def cyber_agent(facts):
    kb = set(facts)
    rules = [
        ({'MultipleFailedLogins', 'LoginFromUnknownIP'}, 'SuspiciousLogin'),
        ({'SuspiciousLogin', 'AdminPrivileges'}, 'HighRiskIntrusion'),
        ({'HighRiskIntrusion'}, 'TriggerAlert'),
    ]
    changed = True
    while changed:
        changed = False
        for conds, conc in rules:
            if conds.issubset(kb) and conc not in kb:
                kb.add(conc)
                changed = True
    return kb

events = ['MultipleFailedLogins', 'LoginFromUnknownIP', 'AdminPrivileges']
result = cyber_agent(events)
print("Inferred facts:", result)
print("Alert triggered:", 'TriggerAlert' in result)

```

## Q5: Autonomous Warehouse Robot

Rules:
- ObstacleAhead → Stop
- CarryingFragileItem AND ObstacleAhead → SlowDown
- HumanNearby → ReduceSpeed
- PathClear AND NOT HumanNearby → MoveForward


```python
def warehouse_robot(percepts):
    kb = set(percepts)
    actions = []

    if 'ObstacleAhead' in kb:
        if 'CarryingFragileItem' in kb:
            actions.append('SlowDown')
        else:
            actions.append('Stop')

    if 'HumanNearby' in kb:
        actions.append('ReduceSpeed')

    if 'PathClear' in kb and 'HumanNearby' not in kb:
        actions.append('MoveForward')

    return actions if actions else ['Wait']

scenarios = [
    ['PathClear'],
    ['ObstacleAhead'],
    ['ObstacleAhead', 'CarryingFragileItem'],
    ['PathClear', 'HumanNearby'],
]
for s in scenarios:
    print(f"Percepts: {s} → Actions: {warehouse_robot(s)}")

```