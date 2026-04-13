# lab4


# LAB 4: Informed Search Algorithms

**Aim:** Implement Greedy Best-First Search and A* using heuristics.

**Maze as a grid** — nodes are (row, col) pairs, heuristic = Manhattan distance.


## Setup: Maze as a graph with (row,col) positions


```python
# Maze graph: node → list of neighbors
# Positions are (row, col) tuples
maze = {
    (0,0): [(0,1),(1,0)],
    (0,1): [(0,0),(0,2)],
    (0,2): [(0,1),(1,2)],
    (1,0): [(0,0),(2,0)],
    (1,2): [(0,2),(2,2)],
    (2,0): [(1,0),(2,1)],
    (2,1): [(2,0),(2,2)],
    (2,2): [(1,2),(2,1)]
}

start = (0, 0)
goal  = (2, 2)

def heuristic(node, goal):
    # Manhattan distance
    return abs(node[0]-goal[0]) + abs(node[1]-goal[1])

print("Start:", start, "Goal:", goal)
print("h(start):", heuristic(start, goal))

```

## Experiment 2: Greedy Best-First Search

Always expands the node with **lowest h(n)** (closest to goal by heuristic).
Not guaranteed optimal, but fast.


```python
import heapq

def greedy_bfs(graph, start, goal):
    # Priority queue: (h_value, path)
    pq = [(heuristic(start, goal), [start])]
    visited = set()
    expanded = 0

    while pq:
        h, path = heapq.heappop(pq)
        node = path[-1]

        if node == goal:
            return path, expanded

        if node not in visited:
            visited.add(node)
            expanded += 1
            for neighbor in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(pq, (heuristic(neighbor, goal), path + [neighbor]))

    return None, expanded

path, expanded = greedy_bfs(maze, start, goal)
print("Greedy BFS Path:", path)
print("Path length:", len(path)-1)
print("Nodes expanded:", expanded)

```

## Experiment 3: A* Search

Expands node with lowest **f(n) = g(n) + h(n)**
- g(n) = actual cost from start
- h(n) = Manhattan distance to goal

A* is **optimal** because h(n) never overestimates (admissible).


```python
import heapq

def astar(graph, start, goal):
    # Priority queue: (f, g, path)
    pq = [(heuristic(start,goal), 0, [start])]
    visited = set()
    expanded = 0

    while pq:
        f, g, path = heapq.heappop(pq)
        node = path[-1]

        if node == goal:
            return path, expanded

        if node not in visited:
            visited.add(node)
            expanded += 1
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_g = g + 1  # each step costs 1
                    new_f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(pq, (new_f, new_g, path + [neighbor]))

    return None, expanded

path, expanded = astar(maze, start, goal)
print("A* Path:", path)
print("Path length:", len(path)-1)
print("Nodes expanded:", expanded)

```

## Comparison: BFS vs Greedy vs A*


```python
from collections import deque
import heapq

def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = set()
    expanded = 0
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path, expanded
        if node not in visited:
            visited.add(node); expanded += 1
            for n in graph[node]:
                queue.append(path + [n])
    return None, expanded

bfs_path,  bfs_exp  = bfs(maze, start, goal)
gbfs_path, gbfs_exp = greedy_bfs(maze, start, goal)
astar_path,astar_exp= astar(maze, start, goal)

print(f"{'Algorithm':<12} {'Path Length':>12} {'Nodes Expanded':>15} {'Optimal?':>10}")
print(f"{'BFS':<12} {len(bfs_path)-1:>12} {bfs_exp:>15} {'Yes':>10}")
print(f"{'Greedy BFS':<12} {len(gbfs_path)-1:>12} {gbfs_exp:>15} {'No':>10}")
print(f"{'A*':<12} {len(astar_path)-1:>12} {astar_exp:>15} {'Yes':>10}")

```