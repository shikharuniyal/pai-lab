# lab10


# LAB 10: Reinforcement Learning

**Concept:** Agent learns by trial and error, getting rewards/penalties.
- Q-Learning update: Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]


## Exercise 1: Q-Learning for Grid Navigation (Delivery Robot)

- Grid: 4×4, some cells are obstacles
- Robot starts at (0,0), goal at (3,3)
- Reward: +100 at goal, -10 for obstacle, -1 per step


```python
import random

ROWS, COLS = 4, 4
START = (0, 0)
GOAL  = (3, 3)
OBSTACLES = {(1,1), (2,2), (1,3)}
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

# Q-table: state → [Q for each action]
Q = {}
def get_q(s):
    if s not in Q:
        Q[s] = [0.0] * 4
    return Q[s]

alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.2 # exploration rate

def step(state, action_idx):
    dr, dc = ACTIONS[action_idx]
    r, c = state[0]+dr, state[1]+dc
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return state, -1  # wall: stay, small penalty
    new_state = (r, c)
    if new_state in OBSTACLES:
        return state, -10
    if new_state == GOAL:
        return new_state, 100
    return new_state, -1

# Train
for episode in range(500):
    state = START
    for _ in range(100):  # max steps per episode
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = get_q(state).index(max(get_q(state)))

        new_state, reward = step(state, action)

        # Q update
        old_q = get_q(state)[action]
        best_next = max(get_q(new_state))
        get_q(state)[action] = old_q + alpha * (reward + gamma * best_next - old_q)

        state = new_state
        if state == GOAL:
            break

# Test learned policy
state = START
path = [state]
for _ in range(20):
    action = get_q(state).index(max(get_q(state)))
    state, _ = step(state, action)
    path.append(state)
    if state == GOAL:
        break

print("Learned path:", path)
print("Reached goal:", path[-1] == GOAL)

```

## Exercise 2: k-Armed Bandit (Ad Selection)

- 5 ads with unknown click probabilities
- Two strategies: ε-greedy and UCB


```python
import random, math

# True click probabilities (unknown to agent)
true_probs = [0.1, 0.3, 0.5, 0.2, 0.4]
k = len(true_probs)
N = 1000  # rounds

def pull(arm):
    return 1 if random.random() < true_probs[arm] else 0

# Epsilon-Greedy
epsilon = 0.1
counts = [0] * k
rewards = [0.0] * k
total_eg = 0

for t in range(N):
    if random.random() < epsilon or 0 in counts:
        arm = random.randint(0, k-1)
    else:
        arm = rewards.index(max(r/c for r,c in zip(rewards,counts)))
    r = pull(arm)
    counts[arm] += 1
    rewards[arm] += r
    total_eg += r

print("=== Epsilon-Greedy ===")
print("Total reward:", total_eg)
print("Times each ad selected:", counts)

# UCB
counts_ucb = [0] * k
rewards_ucb = [0.0] * k
total_ucb = 0

for t in range(1, N+1):
    if 0 in counts_ucb:
        arm = counts_ucb.index(0)  # try each arm once first
    else:
        ucb = [rewards_ucb[a]/counts_ucb[a] + math.sqrt(2*math.log(t)/counts_ucb[a]) for a in range(k)]
        arm = ucb.index(max(ucb))
    r = pull(arm)
    counts_ucb[arm] += 1
    rewards_ucb[arm] += r
    total_ucb += r

print("\n=== UCB ===")
print("Total reward:", total_ucb)
print("Times each ad selected:", counts_ucb)
print(f"\nBest ad is Ad 3 (prob={true_probs[2]}). UCB focuses more on it.")

```

## Exercise 3: CartPole — Custom Simulation (no gym needed)

Simplified: pole has angle. Action: push left (0) or right (1).
We discretize angle and use Q-learning.


```python
import random

# Simple cartpole simulation: just track pole angle
def cartpole_step(angle, vel, action):
    force = 1.0 if action == 1 else -1.0
    vel = vel + 0.1 * angle - 0.01 * force
    angle = angle + vel
    done = abs(angle) > 1.0  # fell over
    reward = 1 if not done else -10
    return angle, vel, reward, done

def discretize(angle, vel):
    a = min(max(int(angle * 5 + 5), 0), 9)
    v = min(max(int(vel * 5 + 5), 0), 9)
    return (a, v)

Q = {}
def get_q(s):
    if s not in Q: Q[s] = [0.0, 0.0]
    return Q[s]

alpha, gamma, epsilon = 0.1, 0.95, 0.2
episode_rewards = []

for ep in range(300):
    angle = random.uniform(-0.1, 0.1)
    vel = 0.0
    total = 0
    for _ in range(200):
        s = discretize(angle, vel)
        action = random.randint(0,1) if random.random() < epsilon else get_q(s).index(max(get_q(s)))
        angle, vel, r, done = cartpole_step(angle, vel, action)
        s2 = discretize(angle, vel)
        get_q(s)[action] += alpha * (r + gamma * max(get_q(s2)) - get_q(s)[action])
        total += r
        if done: break
    episode_rewards.append(total)

print("Early avg reward (ep 1-50):", sum(episode_rewards[:50])//50)
print("Late avg reward (ep 250-300):", sum(episode_rewards[250:])//50)
print("Agent improved over time (higher = pole stayed up longer)")

```

## Exercise 4: Traffic Signal Q-Learning

- State: vehicles waiting in each lane (discretized)
- Action: which signal to turn green (0=North-South, 1=East-West)
- Reward: negative of total waiting time


```python
import random

# State: (cars_NS, cars_EW) discretized 0-4
Q = {}
def get_q(s):
    if s not in Q: Q[s] = [0.0, 0.0]
    return Q[s]

alpha, gamma, epsilon = 0.1, 0.9, 0.2

def simulate_step(state, action):
    ns, ew = state
    # Green light clears ~2 cars, red adds ~1
    if action == 0:  # NS green
        ns = max(0, ns - 2) + random.randint(0, 2)
        ew = min(4, ew + random.randint(0, 2))
    else:            # EW green
        ew = max(0, ew - 2) + random.randint(0, 2)
        ns = min(4, ns + random.randint(0, 2))
    ns = min(4, ns)
    ew = min(4, ew)
    reward = -(ns + ew)  # minimize total waiting
    return (ns, ew), reward

total_waits = []
for ep in range(200):
    state = (random.randint(0,4), random.randint(0,4))
    ep_wait = 0
    for _ in range(50):
        s = state
        action = random.randint(0,1) if random.random() < epsilon else get_q(s).index(max(get_q(s)))
        new_state, reward = simulate_step(state, action)
        get_q(s)[action] += alpha * (reward + gamma * max(get_q(new_state)) - get_q(s)[action])
        state = new_state
        ep_wait -= reward  # reward is negative, so negate to get wait time
    total_waits.append(ep_wait)

print("Early avg waiting (ep 1-50):", round(sum(total_waits[:50])/50, 1))
print("Late avg waiting (ep 150-200):", round(sum(total_waits[150:])/50, 1))
print("Lower waiting = better signal control learned")

```

## Exercise 5: Naive Bayes Medical Diagnosis

P(Disease | Symptoms) using Bayes' Theorem:
P(D|S) = P(S|D) * P(D) / P(S)


```python
# P(Disease)
P_disease = 0.01  # 1% of population has disease

# P(Symptom | Disease) and P(Symptom | No Disease)
symptoms = {
    'Fever':     {'yes': 0.9, 'no': 0.1},
    'Cough':     {'yes': 0.8, 'no': 0.2},
    'ChestPain': {'yes': 0.6, 'no': 0.05},
}

def diagnose(observed_symptoms, threshold=0.5):
    # P(symptoms | disease)
    p_s_given_d = 1.0
    p_s_given_no_d = 1.0

    for s in symptoms:
        if s in observed_symptoms:
            p_s_given_d    *= symptoms[s]['yes']
            p_s_given_no_d *= symptoms[s]['no']
        # (ignoring symptoms not observed for simplicity)

    # Bayes theorem (unnormalized)
    p_d_given_s    = p_s_given_d    * P_disease
    p_nod_given_s  = p_s_given_no_d * (1 - P_disease)

    # Normalize
    total = p_d_given_s + p_nod_given_s
    posterior = p_d_given_s / total

    print(f"Symptoms: {observed_symptoms}")
    print(f"P(Disease | Symptoms) = {posterior:.4f}")
    print(f"Diagnosis: {'Disease PRESENT' if posterior > threshold else 'Disease ABSENT'}\n")

diagnose(['Fever'])
diagnose(['Fever', 'Cough'])
diagnose(['Fever', 'Cough', 'ChestPain'])

```