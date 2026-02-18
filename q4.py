"""
lab4_with_libs.py
Lab 4: Greedy Best-First, A*, RBFS — library-first, fallback-safe.
Run: python lab4_with_libs.py
Optional pip installs for nicer output: pip install networkx psutil rich tqdm
"""

import time
import heapq
from collections import deque
from functools import lru_cache

# Try to import helpful third-party libs, otherwise fallback
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    from rich.pretty import pprint as rprint
    from rich import print as rprint_line
    HAS_RICH = True
except Exception:
    HAS_RICH = False

def maybe_print(*args, **kwargs):
    if HAS_RICH:
        rprint_line(*args, **kwargs)
    else:
        print(*args, **kwargs)

# -----------------------
# Utilities (common)
# -----------------------

def reconstruct(parent, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path

def time_and_mem():
    t = time.time()
    mem = None
    if HAS_PSUTIL:
        process = psutil.Process()
        mem = process.memory_info().rss  # bytes
    return t, mem

# -----------------------
# Grid helpers
# -----------------------

def grid_neighbors(pos, grid):
    r,c = pos
    rows, cols = len(grid), len(grid[0])
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield (nr,nc), 1

def manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# -----------------------
# Greedy Best-First Search
# Uses networkx if available for clean code on graphs, otherwise simple heap-based.
# -----------------------

def greedy_bfs(start, goal, successors_fn, heuristic_fn):
    t0 = time.time()
    nodes_expanded = 0
    open_heap = [(heuristic_fn(start), start)]
    parent = {start: None}
    closed = set()

    while open_heap:
        h, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        nodes_expanded += 1
        if node == goal:
            return {'path': reconstruct(parent, goal), 'nodes': nodes_expanded, 'time': time.time()-t0}
        for nb, cost in successors_fn(node):
            if nb not in closed and nb not in parent:
                parent[nb] = node
                heapq.heappush(open_heap, (heuristic_fn(nb), nb))
    return {'path': None, 'nodes': nodes_expanded, 'time': time.time()-t0}

# -----------------------
# A* search
# If networkx available and states are hashable nodes, use nx.astar_path for succinctness.
# Otherwise fallback to standard heap-based A*.
# -----------------------

def a_star(start, goal, successors_fn, heuristic_fn):
    # If networkx is available and start/goal are suitable node types and successors_fn maps to neighbors, we skip.
    # But safer to use our generic A* (works always).
    t0 = time.time()
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (heuristic_fn(start), counter, start))
    g = {start: 0}
    parent = {start: None}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        f, _, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        nodes_expanded += 1
        if node == goal:
            return {'path': reconstruct(parent, goal), 'cost': g[node], 'nodes': nodes_expanded, 'time': time.time()-t0}
        for nb, cost in successors_fn(node):
            ng = g[node] + cost
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                parent[nb] = node
                counter += 1
                heapq.heappush(open_heap, (ng + heuristic_fn(nb), counter, nb))
    return {'path': None, 'nodes': nodes_expanded, 'time': time.time()-t0}

# -----------------------
# RBFS (recursive best-first)
# Simple, clear RBFS implementation (uses heuristic_fn, successors_fn)
# -----------------------

def rbfs(start, goal, successors_fn, heuristic_fn):
    t0 = time.time()
    parent = {start: None}
    nodes_expanded = 0

    def rbfs_rec(node, g, f_limit):
        nonlocal nodes_expanded
        nodes_expanded += 1
        f_node = g + heuristic_fn(node)
        if node == goal:
            return True, g, f_node
        children = []
        for nb, cost in successors_fn(node):
            ng = g + cost
            children.append([nb, ng, ng + heuristic_fn(nb)])  # [node,g,f]
            parent.setdefault(nb, node)
        if not children:
            return False, float('inf'), float('inf')
        children.sort(key=lambda x: x[2])
        while True:
            best = children[0]
            if best[2] > f_limit:
                return False, float('inf'), best[2]
            alt = children[1][2] if len(children) > 1 else float('inf')
            found, cost_res, best_f = rbfs_rec(best[0], best[1], min(f_limit, alt))
            children[0][2] = best_f
            children.sort(key=lambda x: x[2])
            if found:
                return True, cost_res, best_f

    found, cost, fval = rbfs_rec(start, 0, float('inf'))
    return {'path': reconstruct(parent, goal) if found else None, 'cost': cost if found else None, 'nodes': nodes_expanded, 'time': time.time()-t0}

# -----------------------
# 8-puzzle helpers (A* test and RBFS test)
# -----------------------

PUZZLE_GOAL = (1,2,3,4,5,6,7,8,0)

def neighbors_8puzzle(state):
    idx = state.index(0)
    r,c = divmod(idx, 3)
    moves = []
    if r>0: moves.append(-3)
    if r<2: moves.append(3)
    if c>0: moves.append(-1)
    if c<2: moves.append(1)
    for m in moves:
        s = list(state)
        j = idx + m
        s[idx], s[j] = s[j], s[idx]
        yield tuple(s), 1

def manhattan_8p(state, goal=PUZZLE_GOAL):
    total=0
    for i,v in enumerate(state):
        if v==0: continue
        tr,tc = divmod(i,3)
        gi = goal.index(v)
        gr,gc = divmod(gi,3)
        total += abs(tr-gr)+abs(tc-gc)
    return total

# -----------------------
# If networkx is installed, show how to use it for grid A* concisely
# -----------------------

def demo_with_networkx_grid(grid, start, goal):
    if not HAS_NX:
        maybe_print("networkx not available — skipping networkx demo.")
        return
    # create a graph where nodes are (r,c) and edges for free cells
    G = nx.Graph()
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                G.add_node((r,c))
    for node in list(G.nodes):
        for nb, cost in grid_neighbors(node, grid):
            G.add_edge(node, nb, weight=cost)
    # use networkx.astar_path
    path = nx.astar_path(G, start, goal, heuristic=lambda a,b: manhattan(a,b), weight='weight')
    cost = nx.path_weight(G, path, weight='weight') if hasattr(nx, 'path_weight') else sum(1 for _ in path)-1
    return path, cost

# -----------------------
# Example runs
# -----------------------

if __name__ == "__main__":
    maybe_print("=== Lab 4: library-friendly informed searches ===")
    grid = [
        [0,0,0,1,0],
        [1,1,0,1,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ]
    start = (0,0)
    goal = (4,4)

    maybe_print("\n-- Greedy Best-First (grid) --")
    res = greedy_bfs(start, goal, lambda n: list(grid_neighbors(n, grid)), lambda n: manhattan(n, goal))
    maybe_print("Path:", res['path'])
    maybe_print("Nodes:", res['nodes'], "Time:", round(res['time'],6))

    maybe_print("\n-- A* (grid) --")
    res = a_star(start, goal, lambda n: list(grid_neighbors(n, grid)), lambda n: manhattan(n, goal))
    maybe_print("Path:", res['path'])
    maybe_print("Cost:", res.get('cost'), "Nodes:", res['nodes'], "Time:", round(res['time'],6))

    maybe_print("\n-- RBFS (grid) --")
    res = rbfs(start, goal, lambda n: list(grid_neighbors(n, grid)), lambda n: manhattan(n, goal))
    maybe_print("Path:", res['path'])
    maybe_print("Cost:", res.get('cost'), "Nodes:", res['nodes'], "Time:", round(res['time'],6))

    maybe_print("\n-- A* & RBFS on 8-puzzle (small example) --")
    start8 = (1,2,3,4,5,6,0,7,8)
    ares = a_star(start8, PUZZLE_GOAL, lambda s: list(neighbors_8puzzle(s)), lambda s: manhattan_8p(s))
    rres = rbfs(start8, PUZZLE_GOAL, lambda s: list(neighbors_8puzzle(s)), lambda s: manhattan_8p(s))
    maybe_print("A* path length:", len(ares['path']) if ares['path'] else None, "Nodes:", ares['nodes'])
    maybe_print("RBFS path length:", len(rres['path']) if rres['path'] else None, "Nodes:", rres['nodes'])

    if HAS_NX:
        maybe_print("\n-- networkx A* demo (grid) --")
        nx_path, nx_cost = demo_with_networkx_grid(grid, start, goal)
        maybe_print("networkx A* path:", nx_path, "cost:", nx_cost)
