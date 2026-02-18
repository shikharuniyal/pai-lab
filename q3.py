# ai_lab_uninformed_search.py
# All Uninformed Search Algorithms in ONE file
# BFS, DFS, 8-Puzzle BFS, Uniform Cost Search, Comparison

from collections import deque
import heapq
import time
from pprint import pprint

# -------------------------------------------------
# PATH RECONSTRUCTION
# -------------------------------------------------

def build_path(parent, goal):
    path = []
    while goal is not None:
        path.append(goal)
        goal = parent.get(goal)
    return path[::-1]


# -------------------------------------------------
# BFS (Unweighted Graph / Maze)
# -------------------------------------------------

def bfs(graph, start, goal):
    q = deque([start])
    parent = {start: None}
    nodes = 0

    while q:
        node = q.popleft()
        nodes += 1

        if node == goal:
            return build_path(parent, goal), nodes

        for nb in graph[node]:
            if nb not in parent:
                parent[nb] = node
                q.append(nb)

    return None, nodes


# -------------------------------------------------
# DFS (Recursive)
# -------------------------------------------------

def dfs(graph, node, goal, visited, path):
    visited.add(node)
    path.append(node)

    if node == goal:
        return True

    for nb in graph[node]:
        if nb not in visited:
            if dfs(graph, nb, goal, visited, path):
                return True

    path.pop()
    return False


def run_dfs(graph, start, goal):
    visited = set()
    path = []
    dfs(graph, start, goal, visited, path)
    return path


# -------------------------------------------------
# 8 PUZZLE BFS
# -------------------------------------------------

def neighbors_8puzzle(state):
    res = []
    i = state.index(0)
    r, c = divmod(i, 3)

    moves = []
    if r > 0: moves.append(-3)
    if r < 2: moves.append(3)
    if c > 0: moves.append(-1)
    if c < 2: moves.append(1)

    for m in moves:
        s = list(state)
        j = i + m
        s[i], s[j] = s[j], s[i]
        res.append(tuple(s))

    return res


def bfs_8puzzle(start, goal):
    q = deque([start])
    parent = {start: None}

    while q:
        s = q.popleft()

        if s == goal:
            return build_path(parent, goal)

        for nb in neighbors_8puzzle(s):
            if nb not in parent:
                parent[nb] = s
                q.append(nb)

    return None


# -------------------------------------------------
# UNIFORM COST SEARCH
# -------------------------------------------------

def ucs(graph, start, goal):
    pq = [(0, start)]
    parent = {start: None}
    cost = {start: 0}

    while pq:
        g, node = heapq.heappop(pq)

        if node == goal:
            return build_path(parent, goal), g

        for nb, w in graph[node]:
            new_cost = g + w

            if nb not in cost or new_cost < cost[nb]:
                cost[nb] = new_cost
                parent[nb] = node
                heapq.heappush(pq, (new_cost, nb))

    return None, float("inf")


# -------------------------------------------------
# PERFORMANCE COMPARISON
# -------------------------------------------------

def compare(graph, start, goal):
    print("\n--- PERFORMANCE COMPARISON ---")

    t1 = time.time()
    p1, n1 = bfs(graph, start, goal)
    t2 = time.time()

    t3 = time.time()
    p2 = run_dfs(graph, start, goal)
    t4 = time.time()

    print("BFS Path:", p1)
    print("BFS Nodes:", n1, "Time:", round(t2-t1,6))

    print("DFS Path:", p2)
    print("DFS Time:", round(t4-t3,6))


# -------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------

if __name__ == "__main__":

    graph = {
        'A':['B','C'],
        'B':['D','E'],
        'C':['F'],
        'D':['G'],
        'E':['G'],
        'F':[],
        'G':[]
    }

    weighted_graph = {
        'A':[('B',50),('C',10)],
        'B':[('D',20),('G',100)],
        'C':[('F',15)],
        'D':[('G',10)],
        'F':[('G',20)],
        'G':[]
    }

    print("\n=== BFS ===")
    print(bfs(graph,'A','G'))

    print("\n=== DFS ===")
    print(run_dfs(graph,'A','G'))

    print("\n=== UCS ===")
    print(ucs(weighted_graph,'A','G'))

    print("\n=== 8 PUZZLE ===")
    start = (1,2,3,4,5,6,0,7,8)
    goal  = (1,2,3,4,5,6,7,8,0)
    pprint(bfs_8puzzle(start,goal))

    compare(graph,'A','G')
