# -*- coding: utf-8 -*-
"""
PRM for 2D motion planning in the OSR environment.

"""
import sys, time, math, random
import numpy as np
import pylab as pl

# find environment_2d.py
if 'osr_examples/scripts/' not in sys.path:
    sys.path.append('osr_examples/scripts/')

import environment_2d  # from the course repo

# ---------- general tools ----------
def line_collision_free(p, q, env, step=0.01):
    """
    以固定步长在 [p,q] 上采样，调用 env.check_collision(x,y) 判定是否碰撞。
    返回 True 表示整段无碰撞。
    """
    px, py = p; qx, qy = q
    dx, dy = qx - px, qy - py
    dist = math.hypot(dx, dy)
    if dist == 0:
        return not env.check_collision(px, py)
    n = max(2, int(dist / step))
    for i in range(n + 1):
        t = float(i) / n
        x = px + t * dx
        y = py + t * dy
        if env.check_collision(x, y):
            return False
    return True

def euclid(a, b):
    ax, ay = a; bx, by = b
    return math.hypot(ax - bx, ay - by)

def dijkstra(adj, start_idx, goal_idx):
    """
    简易 Dijkstra，在小图上足够好用。
    adj: 邻接表 list[ list[(v, w)] ] ，v 为邻居索引，w 为边长
    返回：节点索引路径(list)；若不可达返回 None
    """
    n = len(adj)
    INF = 1e100
    dist = [INF]*n
    prev = [-1]*n
    used = [False]*n
    dist[start_idx] = 0.0

    for _ in range(n):
        # Select the undetermined minimum dist
        u = -1
        best = INF
        for i in range(n):
            if (not used[i]) and dist[i] < best:
                best = dist[i]; u = i
        if u < 0 or u == goal_idx: break
        used[u] = True
        for v, w in adj[u]:
            if used[v]: continue
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u

    if dist[goal_idx] >= INF/2:
        return None

    # Backtracking Path
    path_idx = []
    cur = goal_idx
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    return path_idx

# ---------- PRM Main Process ----------
def prm_plan(env, x_start, y_start, x_goal, y_goal,
             n_samples=500, radius=0.6, step=0.01, seed=4, verbose=True):
    """
    Generate a PRM and use Dijkstra search on the graph。
    - n_samples: Number of free space sampling points (excluding start and end points)
    - radius:    Try edge radius
    - step:      Sampling step size for line collision detection
    """
    rng = random.Random(seed)
    pl.clf(); env.plot(); env.plot_query(x_start, y_start, x_goal, y_goal)

    # Collecting free-space samples
    nodes = [(x_start, y_start), (x_goal, y_goal)]
    tried = 0
    while len(nodes) < n_samples + 2:
        x = rng.uniform(0.0, env.size_x)
        y = rng.uniform(0.0, env.size_y)
        if not env.check_collision(x, y):
            nodes.append((x, y))
        tried += 1
        # Draw a small number of visualization points
        if tried % 50 == 0:
            pl.plot([x], [y], ".", alpha=0.2); pl.pause(0.001)

    n = len(nodes)
    adj = [[] for _ in range(n)]

    # Try connecting edges within the radius
    t0 = time.time()
    for i in range(n):
        for j in range(i+1, n):
            if euclid(nodes[i], nodes[j]) <= radius:
                if line_collision_free(nodes[i], nodes[j], env, step=step):
                    w = euclid(nodes[i], nodes[j])
                    adj[i].append((j, w))
                    adj[j].append((i, w))
                    # Visualize edges (draw them sparsely to avoid being too slow)
                    if (i % 25 == 0) and (j % 25 == 0):
                        pl.plot([nodes[i][0], nodes[j][0]],
                                [nodes[i][1], nodes[j][1]], "k-", alpha=0.15)
    t_build = time.time() - t0

    # Start and end index
    s_idx, g_idx = 0, 1

    # search
    t1 = time.time()
    path_idx = dijkstra(adj, s_idx, g_idx)
    t_search = time.time() - t1
    if path_idx is None:
        if verbose:
            print("No path found. build: %.3fs, search: %.3fs" % (t_build, t_search))
        return None, nodes, adj, (t_build, t_search)

    path_xy = [nodes[i] for i in path_idx]
    if verbose:
        print("Path found. #nodes=%d  build: %.3fs  search: %.3fs  length=%.3f"
              % (n, t_build, t_search,
                 sum(euclid(path_xy[i], path_xy[i+1]) for i in range(len(path_xy)-1))))

    # draw final path
    xs = [p[0] for p in path_xy]; ys = [p[1] for p in path_xy]
    pl.plot(xs, ys, "r-", linewidth=2)
    pl.title("PRM path")
    pl.pause(0.001)
    return path_xy, nodes, adj, (t_build, t_search)

def run_single(seed=4, n_samples=500, radius=0.6):
    np.random.seed(seed)
    env = environment_2d.Environment(10, 6, 5)
    pl.clf(); env.plot()
    q = env.random_query()
    if q is None:
        print("No query generated"); return
    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)
    path_xy, nodes, adj, times = prm_plan(env, x_start, y_start, x_goal, y_goal,
                                          n_samples=n_samples, radius=radius, seed=seed)
    return path_xy

# ------- Batch experiments: multiple endpoints/multiple environments -------
def benchmark(num_env=3, num_queries=5, n_samples=600, radius=0.6):
    lens, succ = [], 0
    t_build_all, t_search_all = 0.0, 0.0
    for e in range(num_env):
        env = environment_2d.Environment(10, 6, 5)
        for qid in range(num_queries):
            q = env.random_query()
            if q is None: continue
            x_start, y_start, x_goal, y_goal = q
            path_xy, _, _, (tb, ts) = prm_plan(env, x_start, y_start, x_goal, y_goal,
                                               n_samples=n_samples, radius=radius,
                                               seed=(e*100+qid), verbose=False)
            t_build_all += tb; t_search_all += ts
            if path_xy is not None:
                succ += 1
                lens.append(sum(euclid(path_xy[i], path_xy[i+1]) for i in range(len(path_xy)-1)))
    print("PRM benchmark: env=%d queries/env=%d  success=%d  avg_len=%.3f  avg_build=%.3fs  avg_search=%.3fs"
          % (num_env, num_queries, succ,
             (np.mean(lens) if lens else float('nan')),
             t_build_all/(num_env*num_queries), t_search_all/(num_env*num_queries)))

if __name__ == "__main__":
    # single-run
    # run_single(seed=4, n_samples=600, radius=0.6)

    # benchmark
    benchmark()
