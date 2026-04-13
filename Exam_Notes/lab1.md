# lab1


# LAB 1: Intelligent Agents and Environments

**Aim:** Understand intelligent agents, PEAS, and agent architectures.


## Experiment 1 (Solved): Simple Vacuum Cleaner Agent (2 rooms)

**PEAS:**
- Performance: Cleanliness, minimal movement
- Environment: Two rooms A, B
- Actuators: Left, Right, Suck
- Sensors: Location, Dirt status


```python
class VacuumAgent:
    def __init__(self):
        self.location = 'A'

    def act(self, percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        elif location == 'A':
            return 'Right'
        else:
            return 'Left'

agent = VacuumAgent()
# Test all combinations
for loc in ['A', 'B']:
    for status in ['Dirty', 'Clean']:
        print(f"Location={loc}, Status={status} → Action={agent.act((loc, status))}")

```

## Experiment 2: Simple Reflex Agent in 4-Room Grid

4 rooms arranged as 2x2 grid: (0,0), (0,1), (1,0), (1,1)
- If dirty → Suck
- If clean → move randomly (Up/Down/Left/Right within bounds)


```python
import random

# Setup: random dirt in each room
rooms = {(0,0): random.choice(['Clean','Dirty']),
         (0,1): random.choice(['Clean','Dirty']),
         (1,0): random.choice(['Clean','Dirty']),
         (1,1): random.choice(['Clean','Dirty'])}

print("Initial room state:", rooms)

def get_valid_moves(pos):
    x, y = pos
    moves = []
    if x > 0: moves.append((-1, 0))  # Up
    if x < 1: moves.append((1, 0))   # Down
    if y > 0: moves.append((0, -1))  # Left
    if y < 1: moves.append((0, 1))   # Right
    return moves

def reflex_agent(pos, status):
    if status == 'Dirty':
        return 'Suck'
    else:
        dx, dy = random.choice(get_valid_moves(pos))
        return (pos[0]+dx, pos[1]+dy)  # new position

# Simulate
pos = (0, 0)
rooms_cleaned = 0
movements = 0
steps = 20

for step in range(steps):
    status = rooms[pos]
    action = reflex_agent(pos, status)
    if action == 'Suck':
        if rooms[pos] == 'Dirty':
            rooms[pos] = 'Clean'
            rooms_cleaned += 1
            print(f"Step {step+1}: Cleaned {pos}")
    else:
        pos = action
        movements += 1

print(f"\nRooms cleaned: {rooms_cleaned}, Movements: {movements}")
print("Final room state:", rooms)

```

## Experiment 3: Model-Based Vacuum Cleaner Agent

Keeps an internal map of which rooms are clean/dirty.
- Prefer unvisited or dirty rooms
- Stop when all rooms known to be clean


```python
import random

# Environment
rooms = {(0,0): 'Dirty', (0,1): 'Dirty', (1,0): 'Clean', (1,1): 'Dirty'}
print("Initial:", rooms)

# Agent with internal model
model = {}  # agent's belief of room states
pos = (0, 0)
movements = 0
all_rooms = [(0,0), (0,1), (1,0), (1,1)]

def get_neighbors(pos):
    x, y = pos
    candidates = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
    return [p for p in candidates if p in all_rooms]

for step in range(30):
    status = rooms[pos]
    model[pos] = status  # update model

    if status == 'Dirty':
        rooms[pos] = 'Clean'
        model[pos] = 'Clean'
        print(f"Step {step+1}: Cleaned {pos}")
    else:
        # Check if all rooms known clean
        if all(model.get(r) == 'Clean' for r in all_rooms):
            print(f"All rooms clean! Done in {step+1} steps.")
            break
        # Move to unvisited or dirty neighbor
        neighbors = get_neighbors(pos)
        unvisited = [n for n in neighbors if n not in model]
        dirty = [n for n in neighbors if model.get(n) == 'Dirty']
        if unvisited:
            pos = unvisited[0]
        elif dirty:
            pos = dirty[0]
        else:
            pos = random.choice(neighbors)
        movements += 1

print(f"Movements: {movements}")

```

## Analysis Questions

1. **Why does simple reflex agent do redundant actions?** → No memory, so it revisits clean rooms.

2. **How does model improve performance?** → Tracks visited/clean rooms, avoids revisiting.

3. **Comparison:**
   - Rationality: Model-based is more rational (smarter decisions)
   - Efficiency: Model-based uses fewer moves
   - Scalability: Model-based scales better (fewer redundant moves)

4. **Would model-based work in partially observable env?** → Partially. It tracks what it has seen, but can't know unvisited rooms, so it might miss dirt in rooms it hasn't reached yet.
