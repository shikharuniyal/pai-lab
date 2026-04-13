# lab2


# LAB 2: Uninformed Search Algorithms

**Aim:** Implement BFS, DFS and apply them to maze/graph problems.


## Experiment 1: BFS for Maze Navigation

**Search Problem:**
- State space: All nodes in graph
- Initial state: start node
- Actions: Move to a neighbor
- Transition model: Follow edges
- Goal test: current node == goal
- Path cost: number of edges


```python
from collections import deque

def bfs(graph, start, goal):
    queue = deque([[start]])  # queue of paths
    visited = set()
    nodes_expanded = 0

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path, nodes_expanded

        if node not in visited:
            visited.add(node)
            nodes_expanded += 1
            for neighbor in graph[node]:
                queue.append(path + [neighbor])

    return None, nodes_expanded

# Maze as adjacency list
maze = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': []
}

path, expanded = bfs(maze, 'A', 'G')
print("Path found:", path)
print("Nodes expanded:", expanded)

```

## Experiment 2: DFS for Graph Traversal

**Problem:** Find a path from start to goal using depth-first (recursive).
- Not guaranteed to find shortest path
- Uses less memory than BFS


```python
def dfs(graph, node, goal, path=[], visited=set()):
    path = path + [node]
    visited = visited | {node}

    if node == goal:
        return path

    for neighbor in graph[node]:
        if neighbor not in visited:
            result = dfs(graph, neighbor, goal, path, visited)
            if result:
                return result

    return None

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': []
}

path = dfs(graph, 'A', 'G')
print("DFS Path found:", path)

```

## Experiment 3: 8-Puzzle Problem using BFS

3x3 grid with tiles 1-8 and a blank (0). Move blank to reach goal.
- State: tuple of 9 numbers
- Goal: (1,2,3,4,5,6,7,8,0)


```python
from collections import deque

def get_neighbors_puzzle(state):
    state = list(state)
    i = state.index(0)  # find blank
    row, col = i // 3, i % 3
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:  # up,down,left,right
        r, c = row+dr, col+dc
        if 0 <= r < 3 and 0 <= c < 3:
            j = r*3 + c
            new = state[:]
            new[i], new[j] = new[j], new[i]  # swap blank
            neighbors.append(tuple(new))
    return neighbors

def bfs_puzzle(start, goal):
    queue = deque([[start]])
    visited = {start}
    expanded = 0

    while queue:
        path = queue.popleft()
        state = path[-1]
        expanded += 1

        if state == goal:
            return path, expanded

        for neighbor in get_neighbors_puzzle(state):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return None, expanded

start = (1, 2, 3, 4, 0, 5, 7, 8, 6)  # solvable
goal  = (1, 2, 3, 4, 5, 6, 7, 8, 0)

path, expanded = bfs_puzzle(start, goal)
print(f"Steps to solve: {len(path)-1}")
print(f"States expanded: {expanded}")
for state in path:
    print(state[:3])
    print(state[3:6])
    print(state[6:])
    print('---')

```

## Experiment 4: BFS vs DFS Performance Comparison


```python
import time
from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': []
}

# BFS with tracking
def bfs_track(graph, start, goal):
    queue = deque([[start]])
    visited = set()
    expanded = 0
    max_queue = 0
    while queue:
        max_queue = max(max_queue, len(queue))
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path, expanded, max_queue
        if node not in visited:
            visited.add(node)
            expanded += 1
            for n in graph[node]:
                queue.append(path + [n])
    return None, expanded, max_queue

# DFS with tracking
def dfs_track(graph, node, goal, path=[], visited=set(), depth=0):
    path = path + [node]
    visited = visited | {node}
    if node == goal:
        return path, depth
    for neighbor in graph[node]:
        if neighbor not in visited:
            result = dfs_track(graph, neighbor, goal, path, visited, depth+1)
            if result[0]:
                return result
    return None, depth

t1 = time.time()
bfs_path, bfs_exp, bfs_mem = bfs_track(graph, 'A', 'G')
bfs_time = time.time() - t1

t1 = time.time()
dfs_path, dfs_depth = dfs_track(graph, 'A', 'G')
dfs_time = time.time() - t1

print("=== Results ===")
print(f"{'Metric':<25} {'BFS':>10} {'DFS':>10}")
print(f"{'Path':<25} {str(bfs_path):>10} {str(dfs_path):>10}")
print(f"{'Path length':<25} {len(bfs_path)-1:>10} {len(dfs_path)-1:>10}")
print(f"{'Nodes expanded':<25} {bfs_exp:>10} {'N/A':>10}")
print(f"{'Max memory (queue/depth)':<25} {bfs_mem:>10} {dfs_depth:>10}")
print(f"{'Time (s)':<25} {bfs_time:.6f} {dfs_time:.6f}")

print("""
Analysis:
- BFS guarantees shortest path; DFS may find longer path
- BFS uses more memory (stores all paths at current level)
- DFS uses less memory (only current path in recursion)
""")

```

## Post-Exercise Analysis

1. **Why does BFS guarantee optimal?** → Explores level by level, so first path found is always shortest.

2. **When is DFS preferred?** → When memory is limited, or solution is deep in the tree.

3. **How branching factor affects BFS?** → Higher branching = exponentially more nodes at each level → more memory.

4. **Why BFS impractical for large spaces?** → Memory grows as O(b^d) where b=branching, d=depth → too much RAM.
