# lab7


# LAB 7: Adversarial Search — Minimax & Alpha-Beta


## Exercise 1: Tic-Tac-Toe AI

- Board: list of 9 cells (`' '`, `'X'`, `'O'`)
- AI = X (MAX), Human = O (MIN)
- Minimax: pick best move by exploring all outcomes
- Alpha-Beta: same result but skips branches that can't help


```python
# Core game functions
def print_board(state):
    for i in range(0, 9, 3):
        print(state[i], '|', state[i+1], '|', state[i+2])
    print()

def actions(state):
    return [i for i, s in enumerate(state) if s == ' ']

def result(state, action, player):
    s = state[:]
    s[action] = player
    return s

def player(state):
    return 'X' if state.count('X') == state.count('O') else 'O'

def terminal(state):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if state[a] == state[b] == state[c] != ' ':
            return True
    return ' ' not in state

def utility(state):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if state[a] == state[b] == state[c] == 'X': return 1
        if state[a] == state[b] == state[c] == 'O': return -1
    return 0

```

```python
# Minimax
nodes_minimax = 0

def max_value(state):
    global nodes_minimax
    nodes_minimax += 1
    if terminal(state): return utility(state)
    v = -2
    for a in actions(state):
        v = max(v, min_value(result(state, a, 'X')))
    return v

def min_value(state):
    global nodes_minimax
    nodes_minimax += 1
    if terminal(state): return utility(state)
    v = 2
    for a in actions(state):
        v = min(v, max_value(result(state, a, 'O')))
    return v

def minimax_decision(state):
    return max(actions(state), key=lambda a: min_value(result(state, a, 'X')))

```

```python
# Alpha-Beta Pruning
nodes_ab = 0

def max_ab(state, alpha, beta):
    global nodes_ab
    nodes_ab += 1
    if terminal(state): return utility(state)
    v = -2
    for a in actions(state):
        v = max(v, min_ab(result(state, a, 'X'), alpha, beta))
        if v >= beta: return v  # prune
        alpha = max(alpha, v)
    return v

def min_ab(state, alpha, beta):
    global nodes_ab
    nodes_ab += 1
    if terminal(state): return utility(state)
    v = 2
    for a in actions(state):
        v = min(v, max_ab(result(state, a, 'O'), alpha, beta))
        if v <= alpha: return v  # prune
        beta = min(beta, v)
    return v

def ab_decision(state):
    return max(actions(state), key=lambda a: min_ab(result(state, a, 'X'), -2, 2))

```

```python
# Play AI vs AI: X uses Minimax, O uses Alpha-Beta
nodes_minimax = 0
nodes_ab = 0

state = [' '] * 9
print("=== Tic-Tac-Toe: AI(X/Minimax) vs AI(O/AlphaBeta) ===")

while not terminal(state):
    p = player(state)
    if p == 'X':
        move = minimax_decision(state)
    else:
        move = ab_decision(state)
    state = result(state, move, p)
    print(f"{p} plays at position {move}:")
    print_board(state)

u = utility(state)
if u == 1: print("X wins!")
elif u == -1: print("O wins!")
else: print("Draw!")

print(f"\nNodes evaluated — Minimax: {nodes_minimax}, Alpha-Beta: {nodes_ab}")
print("Alpha-Beta evaluates fewer nodes by pruning branches that can't change the outcome.")

```

## Exercise 2: Car Overtaking Game

- 7 cells (0–6), X starts at 0, O starts at 1
- Each turn: move forward 1 or 2 cells
- Can't land on opponent's cell or go past cell 6
- First to reach cell 6 wins


```python
car_nodes_mm = 0
car_nodes_ab = 0

def car_terminal(state):
    x, o = state
    return x == 6 or o == 6

def car_utility(state):
    x, o = state
    if x == 6: return 1
    if o == 6: return -1
    return 0

def car_actions(state, is_max):
    x, o = state
    pos = x if is_max else o
    other = o if is_max else x
    moves = []
    for step in [1, 2]:
        new_pos = pos + step
        if new_pos <= 6 and new_pos != other:
            moves.append(step)
    return moves

def car_result(state, step, is_max):
    x, o = state
    if is_max:
        return (x + step, o)
    else:
        return (x, o + step)

def car_max(state):
    global car_nodes_mm
    car_nodes_mm += 1
    if car_terminal(state): return car_utility(state)
    return max(car_min(car_result(state, s, True)) for s in car_actions(state, True)) if car_actions(state, True) else -1

def car_min(state):
    global car_nodes_mm
    car_nodes_mm += 1
    if car_terminal(state): return car_utility(state)
    return min(car_max(car_result(state, s, False)) for s in car_actions(state, False)) if car_actions(state, False) else 1

def car_max_ab(state, alpha, beta):
    global car_nodes_ab
    car_nodes_ab += 1
    if car_terminal(state): return car_utility(state)
    v = -2
    for s in car_actions(state, True):
        v = max(v, car_min_ab(car_result(state, s, True), alpha, beta))
        if v >= beta: return v
        alpha = max(alpha, v)
    return v if car_actions(state, True) else -1

def car_min_ab(state, alpha, beta):
    global car_nodes_ab
    car_nodes_ab += 1
    if car_terminal(state): return car_utility(state)
    v = 2
    for s in car_actions(state, False):
        v = min(v, car_max_ab(car_result(state, s, False), alpha, beta))
        if v <= alpha: return v
        beta = min(beta, v)
    return v if car_actions(state, False) else 1

# Play game
state = (0, 1)  # (X_pos, O_pos)
is_max_turn = True
print("=== Car Overtaking Game ===")

while not car_terminal(state):
    x, o = state
    print(f"X at {x}, O at {o}")
    mvs = car_actions(state, is_max_turn)
    if not mvs:
        print("No moves!")
        break
    if is_max_turn:
        best = max(mvs, key=lambda s: car_min(car_result(state, s, True)))
    else:
        best = min(mvs, key=lambda s: car_max(car_result(state, s, False)))
    state = car_result(state, best, is_max_turn)
    print(f"  {'X' if is_max_turn else 'O'} moves +{best} → {state}")
    is_max_turn = not is_max_turn

u = car_utility(state)
print("Result:", "X wins!" if u==1 else "O wins!" if u==-1 else "Draw")
print(f"Nodes — Minimax: {car_nodes_mm}, Alpha-Beta: {car_nodes_ab}")

```