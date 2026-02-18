"""
lab5_with_libs.py
Lab 5: IDA*, simplified SMA* — library-first with graceful fallbacks.
Run: python lab5_with_libs.py
Optional: pip install psutil rich tqdm
"""

import time
import heapq
from functools import lru_cache

# optional niceties
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    from rich import print as rprint
    HAS_RICH = True
except Exception:
    HAS_RICH = False

def maybe_print(*args, **kwargs):
    if HAS_RICH:
        rprint(*args, **kwargs)
    else:
        print(*args, **kwargs)

# -----------------------
# Helpers (grid + 8-puzzle)
# -----------------------

def reconstruct(parent, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path

def grid_neighbors(pos, grid):
    r,c = pos
    rows, cols = len(grid), len(grid[0])
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield (nr,nc), 1

def manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

PUZZLE_GOAL = (1,2,3,4,5,6,7,8,0)
def neighbors_8puzzle(state):
    idx = state.index(0)
    r,c = divmod(idx,3)
    moves=[]
    if r>0: moves.append(-3)
    if r<2: moves.append(3)
    if c>0: moves.append(-1)
    if c<2: moves.append(1)
    for m in moves:
        s=list(state); j=idx+m
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
# IDA* (iterative deepening on f = g + h)
# -----------------------

def ida_star(start, goal, successors_fn, heuristic_fn):
    t0 = time.time()
    nodes_expanded = 0

    def dfs(node, g, threshold, parent, onpath):
        nonlocal nodes_expanded
        f = g + heuristic_fn(node)
        nodes_expanded += 1
        if f > threshold:
            return None, f
        if node == goal:
            return True, g
        min_over = float('inf')
        for nb, cost in successors_fn(node):
            if nb in onpath:
                continue
            parent[nb] = node
            onpath.add(nb)
            found, val = dfs(nb, g + cost, threshold, parent, onpath)
            onpath.remove(nb)
            if found:
                return True, val
            if val < min_over:
                min_over = val
        return None, min_over

    threshold = heuristic_fn(start)
    parent = {start: None}
    while True:
        onpath = set([start])
        found, val = dfs(start, 0, threshold, parent, onpath)
        if found:
            return {'path': reconstruct(parent, goal), 'cost': val, 'nodes': nodes_expanded, 'time': time.time()-t0}
        if val == float('inf'):
            return {'path': None, 'nodes': nodes_expanded, 'time': time.time()-t0}
        threshold = val

# -----------------------
# SMA* simplified
# - keeps nodes dict and min-heap by f
# - when memory limit exceeded, drops worst-leaf and continues
# NOTE: educational simplified version
# -----------------------

def sma_star(start, goal, successors_fn, heuristic_fn, memory_limit=1000):
    t0 = time.time()
    nodes = {}  # node -> {parent,g,f,children,expanded}
    def add_node(n, parent, g):
        nodes[n] = {'parent': parent, 'g': g, 'f': g + heuristic_fn(n), 'children': set(), 'expanded': False}
        if parent in nodes:
            nodes[parent]['children'].add(n)
    add_node(start, None, 0)
    open_heap = [(nodes[start]['f'], start)]
    nodes_expanded = 0

    def drop_worst_leaf():
        leaf=None; maxf=-1
        for n,info in nodes.items():
            if len(info['children'])==0 and n != start:
                if info['f'] > maxf:
                    maxf = info['f']; leaf = n
        if leaf is None: return False
        p = nodes[leaf]['parent']
        if p is not None and leaf in nodes[p]['children']:
            nodes[p]['children'].remove(leaf)
        del nodes[leaf]
        return True

    while open_heap:
        # pop valid entry
        while open_heap and (open_heap[0][1] not in nodes or nodes[open_heap[0][1]]['expanded']):
            heapq.heappop(open_heap)
        if not open_heap:
            break
        f,node = heapq.heappop(open_heap)
        if node not in nodes:
            continue
        nodes[node]['expanded'] = True
        nodes_expanded += 1
        if node == goal:
            parent_map = {n: nodes[n]['parent'] for n in nodes}
            return {'path': reconstruct(parent_map, goal), 'cost': nodes[node]['g'], 'nodes': nodes_expanded, 'time': time.time()-t0}
        for nb, cost in successors_fn(node):
            ng = nodes[node]['g'] + cost
            if nb in nodes:
                if ng < nodes[nb]['g']:
                    nodes[nb]['g'] = ng
                    nodes[nb]['f'] = ng + heuristic_fn(nb)
                    nodes[nb]['parent'] = node
            else:
                add_node(nb, node, ng)
            heapq.heappush(open_heap, (nodes[nb]['f'], nb))
            if len(nodes) > memory_limit:
                dropped = drop_worst_leaf()
                if not dropped:
                    return {'path': None, 'nodes': nodes_expanded, 'time': time.time()-t0, 'note': 'memory too small'}
    return {'path': None, 'nodes': nodes_expanded, 'time': time.time()-t0}

# -----------------------
# Example runs
# -----------------------

if __name__ == "__main__":
    maybe_print("=== Lab 5: IDA* and simplified SMA* ===")
    grid = [
        [0,0,0,1,0],
        [1,1,0,1,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ]
    start = (0,0); goal = (4,4)

    maybe_print("\n-- IDA* (grid) --")
    r = ida_star(start, goal, lambda n: list(grid_neighbors(n, grid)), lambda n: manhattan(n, goal))
    maybe_print("Path:", r['path']); maybe_print("Nodes:", r['nodes'], "Time:", round(r['time'],6))

    maybe_print("\n-- SMA* (grid, mem limit=30) --")
    r = sma_star(start, goal, lambda n: list(grid_neighbors(n, grid)), lambda n: manhattan(n, goal), memory_limit=30)
    maybe_print("Path:", r.get('path')); maybe_print("Nodes:", r['nodes'], "Time:", round(r['time'],6), "Note:", r.get('note'))

    maybe_print("\n-- IDA* (8-puzzle) --")
    start8 = (1,2,3,4,5,6,0,7,8)
    r = ida_star(start8, PUZZLE_GOAL, lambda s: list(neighbors_8puzzle(s)), lambda s: manhattan_8p(s))
    maybe_print("Path length:", len(r['path']) if r['path'] else None, "Nodes:", r['nodes'])

    maybe_print("\n-- SMA* (8-puzzle, mem limit=1000) --")
    r = sma_star(start8, PUZZLE_GOAL, lambda s: list(neighbors_8puzzle(s)), lambda s: manhattan_8p(s), memory_limit=1000)
    maybe_print("Path length:", len(r.get('path')) if r.get('path') else None, "Nodes:", r['nodes'], "Note:", r.get('note'))
