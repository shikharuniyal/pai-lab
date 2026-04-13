# lab5


# LAB 5 (4a): Memory-Bound Heuristic Search

**Algorithms:** RBFS, IDA*, SMA*

Same maze/heuristic from Lab 4 — Manhattan distance on a grid graph.


```python
# Shared setup
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

start = (0,0)
goal  = (2,2)

def h(node):
    return abs(node[0]-goal[0]) + abs(node[1]-goal[1])

print("Maze ready. Start:", start, "Goal:", goal)

```

## Experiment 4: Recursive Best-First Search (RBFS)

Like A* but uses **linear memory** via recursion.
Tracks best alternative so it knows when to backtrack.


```python
expanded_count = 0

def rbfs(graph, node, g, f_limit, path, visited):
    global expanded_count

    if node == goal:
        return path[:], g

    # Get successors with f values
    successors = []
    for neighbor in graph[node]:
        if neighbor not in visited:
            f = g + 1 + h(neighbor)
            successors.append((f, neighbor))

    if not successors:
        return None, float('inf')

    successors.sort()

    while True:
        best_f, best_node = successors[0]

        if best_f > f_limit:
            return None, best_f  # tell parent to raise limit

        alt_f = successors[1][0] if len(successors) > 1 else float('inf')

        expanded_count += 1
        visited.add(best_node)
        result, new_f = rbfs(graph, best_node, g+1, min(f_limit, alt_f), path+[best_node], visited)
        visited.discard(best_node)

        successors[0] = (new_f, best_node)  # update f
        successors.sort()

        if result:
            return result, new_f

expanded_count = 0
path, cost = rbfs(maze, start, 0, float('inf'), [start], {start})
print("RBFS Path:", path)
print("Path length:", len(path)-1)
print("Nodes expanded:", expanded_count)

```

## Experiment 5: Iterative Deepening A* (IDA*)

DFS with a cost threshold. Increases threshold each iteration.
Very memory efficient — only stores current path.


```python
ida_expanded = 0

def ida_search(graph, node, g, threshold, path, visited):
    global ida_expanded
    f = g + h(node)

    if f > threshold:
        return f, None  # exceeded limit

    if node == goal:
        return -1, path[:]  # found!

    min_exceeded = float('inf')

    for neighbor in graph[node]:
        if neighbor not in visited:
            ida_expanded += 1
            visited.add(neighbor)
            result, found_path = ida_search(graph, neighbor, g+1, threshold, path+[neighbor], visited)
            visited.discard(neighbor)

            if found_path:  # goal found
                return -1, found_path

            min_exceeded = min(min_exceeded, result)

    return min_exceeded, None

def idastar(graph, start, goal_node):
    global ida_expanded
    ida_expanded = 0
    threshold = h(start)

    while True:
        result, path = ida_search(graph, start, 0, threshold, [start], {start})
        if path:
            return path
        if result == float('inf'):
            return None
        threshold = result  # raise limit

path = idastar(maze, start, goal)
print("IDA* Path:", path)
print("Path length:", len(path)-1)
print("Nodes expanded:", ida_expanded)

```

## Experiment 6: Simplified Memory-Bounded A* (SMA*)

Like A* but limits memory. When full, drops worst node and backs up its cost to parent.


```python
import heapq

def sma_star(graph, start, goal, mem_limit=10):
    # Each entry: (f, g, path)
    pq = [(h(start), 0, [start])]
    visited = {}
    expanded = 0

    while pq:
        # Drop worst if over memory limit
        if len(pq) > mem_limit:
            pq.sort(reverse=True)
            pq.pop()  # remove worst (highest f)
            pq.sort()

        f, g, path = heapq.heappop(pq)
        node = path[-1]

        if node == goal:
            return path, expanded

        # Skip if we've seen this node with a better cost
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        expanded += 1

        for neighbor in graph[node]:
            new_g = g + 1
            new_f = new_g + h(neighbor)
            heapq.heappush(pq, (new_f, new_g, path + [neighbor]))

    return None, expanded

path, expanded = sma_star(maze, start, goal, mem_limit=8)
print("SMA* Path:", path)
print("Path length:", len(path)-1)
print("Nodes expanded:", expanded)

```