# lab6


# LAB 6: Local Search Algorithms

**Algorithms:** Hill Climbing, Simulated Annealing, Genetic Algorithm


## Exercise 1: Hill-Climbing for Feature Selection

- State: binary vector (which features are selected)
- Neighbor: flip one bit
- Score: cross-validation accuracy of Logistic Regression


```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import random

data = load_iris()
X, y = data.data, data.target
n_features = X.shape[1]

def evaluate(state):
    selected = [i for i, s in enumerate(state) if s == 1]
    if not selected:
        return 0
    X_sel = X[:, selected]
    return cross_val_score(LogisticRegression(max_iter=200), X_sel, y, cv=3).mean()

def get_neighbors(state):
    neighbors = []
    for i in range(len(state)):
        n = state[:]
        n[i] = 1 - n[i]  # flip bit
        neighbors.append(n)
    return neighbors

for run in range(3):
    state = [random.randint(0,1) for _ in range(n_features)]
    score = evaluate(state)

    while True:
        best_n = max(get_neighbors(state), key=evaluate)
        best_s = evaluate(best_n)
        if best_s <= score:
            break
        state, score = best_n, best_s

    print(f"Run {run+1}: Features={state}, Accuracy={score:.4f}")

```

## Exercise 2: Simulated Annealing for Vehicle Routing

- State: visiting order of delivery locations
- Neighbor: swap two locations
- Accept worse with probability P = e^(-ΔE/T)


```python
import random, math

random.seed(42)
locations = [(random.randint(0,100), random.randint(0,100)) for _ in range(8)]
depot = (50, 50)

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def total_cost(route):
    pts = [depot] + [locations[i] for i in route] + [depot]
    return sum(dist(pts[i], pts[i+1]) for i in range(len(pts)-1))

state = list(range(len(locations)))
random.shuffle(state)
cost = total_cost(state)
print("Initial cost:", round(cost, 2))

T, T_min, cooling = 1000, 1, 0.95

while T > T_min:
    neighbor = state[:]
    i, j = random.sample(range(len(neighbor)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # swap

    delta = total_cost(neighbor) - cost
    if delta < 0 or random.random() < math.exp(-delta / T):
        state, cost = neighbor, total_cost(neighbor)
    T *= cooling

print("Best route:", state)
print("Best cost:", round(cost, 2))

```

## Exercise 3: Genetic Algorithm for Drone Path Planning

- Chromosome: sequence of (x,y) waypoints
- Fitness: path length + obstacle collision penalty
- Operators: tournament selection, one-point crossover, random mutation


```python
import random, math

START, END = (0,0), (10,10)
OBSTACLES = [(3,3), (5,5), (7,4)]
N_WP = 3  # waypoints

def path_len(chrom):
    full = [START] + chrom + [END]
    return sum(math.dist(full[i], full[i+1]) for i in range(len(full)-1))

def penalty(chrom):
    p = 0
    for wp in chrom:
        for obs in OBSTACLES:
            if math.dist(wp, obs) < 1.5:
                p += 100
    return p

def fitness(chrom):
    return path_len(chrom) + penalty(chrom)

def rand_chrom():
    return [(random.uniform(0,10), random.uniform(0,10)) for _ in range(N_WP)]

def tournament(pop):
    return min(random.sample(pop, 3), key=fitness)

def crossover(a, b):
    pt = random.randint(1, N_WP-1)
    return a[:pt] + b[pt:]

def mutate(chrom):
    return [(x+random.uniform(-1,1), y+random.uniform(-1,1))
            if random.random() < 0.2 else (x,y) for x,y in chrom]

pop = [rand_chrom() for _ in range(20)]

for gen in range(50):
    pop = [mutate(crossover(tournament(pop), tournament(pop))) for _ in range(20)]

best = min(pop, key=fitness)
print("Best waypoints:", [(round(x,2), round(y,2)) for x,y in best])
print("Path length:", round(path_len(best), 2))
print("Fitness score:", round(fitness(best), 2))

```