# -*- coding: utf-8 -*-
"""
Post-processing a 2D path (Path shortcutting + optional smoothing) for OSR 2D env.

"""

import sys, math, time, random
import numpy as np
import pylab as pl

# ensure we can import environment_2d and your prm_2d
if 'osr_examples/scripts/' not in sys.path:
    sys.path.append('osr_examples/scripts/')
import environment_2d

# ======================
# Basic geometry helpers
# ======================
def seg_collision_free(p, q, env, step=0.01):
    """Sample along [p,q]; return True if all samples are collision-free."""
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

def path_length(P):
    return sum(math.hypot(P[i+1][0]-P[i][0], P[i+1][1]-P[i][1]) for i in range(len(P)-1))

# ======================
# PATH SHORTCUTTING
# ======================
def path_shortcutting(path_xy, env, maxrep=300, step=0.01, seed=0):
    """
    Implements PATH_SHORTCUTTING:
      repeat:
        pick two random points along the polyline;
        if the straight-line shortcut is collision-free -> replace middle segment(s).
    We pick random vertex indices i<j and attempt a straight segment [Pi,Pj].
    """
    if path_xy is None or len(path_xy) < 3:
        return path_xy

    rng = random.Random(seed)
    P = list(path_xy)

    for _ in range(maxrep):
        if len(P) <= 2: break
        # pick vertex indices far enough apart to create a non-trivial shortcut
        i = rng.randrange(0, len(P) - 2)
        j = rng.randrange(i + 2, len(P))
        if seg_collision_free(P[i], P[j], env, step=step):
            # replace P[i+1 ... j-1] by direct segment
            P = P[:i+1] + P[j:]
    return P

# ======================
# OPTIONAL simple smoothing (moving average on vertices)
# ======================
def smooth_path_moving_average(path_xy, window=5, repeat=2):
    if path_xy is None or len(path_xy) < 3:
        return path_xy
    P = np.asarray(path_xy, dtype=float)
    for _ in range(repeat):
        Q = P.copy()
        for i in range(1, len(P)-1):
            i0 = max(0, i - window//2)
            i1 = min(len(P), i + window//2 + 1)
            Q[i] = P[i0:i1].mean(axis=0)
        P = Q
    return [tuple(x) for x in P]

# ======================
# Try to import your PRM; else provide a tiny fallback PRM
# ======================
def _euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _dijkstra(adj, s, g):
    INF = 1e100
    n = len(adj)
    dist = [INF]*n; prev = [-1]*n; used = [False]*n
    dist[s] = 0.0
    for _ in range(n):
        u = -1; best = INF
        for i in range(n):
            if (not used[i]) and dist[i] < best:
                best = dist[i]; u = i
        if u < 0 or u == g: break
        used[u] = True
        for v, w in adj[u]:
            if used[v]: continue
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd; prev[v] = u
    if dist[g] >= INF/2: return None
    path = []; cur = g
    while cur != -1:
        path.append(cur); cur = prev[cur]
    path.reverse(); return path

def _fallback_prm(env, start, goal, n_samples=600, radius=0.6, step=0.01, seed=4, viz=True):
    rng = random.Random(seed)
    nodes = [start, goal]
    # sample free-space points
    while len(nodes) < n_samples + 2:
        x = rng.uniform(0.0, env.size_x)
        y = rng.uniform(0.0, env.size_y)
        if not env.check_collision(x, y):
            nodes.append((x, y))
    n = len(nodes)
    adj = [[] for _ in range(n)]
    # connect radius neighbors
    for i in range(n):
        for j in range(i+1, n):
            if _euclid(nodes[i], nodes[j]) <= radius:
                if seg_collision_free(nodes[i], nodes[j], env, step=step):
                    w = _euclid(nodes[i], nodes[j])
                    adj[i].append((j, w)); adj[j].append((i, w))
                    if viz and (i%30==0) and (j%30==0):
                        pl.plot([nodes[i][0], nodes[j][0]],
                                [nodes[i][1], nodes[j][1]], "k-", alpha=0.15)
    s, g = 0, 1
    idx = _dijkstra(adj, s, g)
    return ([nodes[i] for i in idx] if idx is not None else None)

def find_path_with_prm(env, seed=4, n_samples=600, radius=0.6):
    """Try to call user's prm_2d; fallback to minimal PRM if not found."""
    try:
        import prm_2d
        np = __import__("numpy")
        np.random.seed(seed)
        q = env.random_query()
        if q is None: return None
        xs, ys, xg, yg = q
        env.plot_query(xs, ys, xg, yg)
        path, *_ = prm_2d.prm_plan(env, xs, ys, xg, yg,
                                   n_samples=n_samples, radius=radius,
                                   seed=seed, verbose=True)
        return path
    except Exception as e:
        print("[WARN] Failed to import/use prm_2d:", e)
        q = env.random_query()
        if q is None: return None
        xs, ys, xg, yg = q
        env.plot_query(xs, ys, xg, yg)
        return _fallback_prm(env, (xs,ys), (xg,yg),
                             n_samples=n_samples, radius=radius,
                             step=0.01, seed=seed, viz=True)

# ======================
# Demo + Benchmark
# ======================
def demo_once(seed=4, n_samples=600, radius=0.6,
              maxrep=400, step=0.01, smooth=True):
    np.random.seed(seed)
    pl.ion()
    env = environment_2d.Environment(10, 6, 5)
    pl.clf(); env.plot()

    raw = find_path_with_prm(env, seed=seed, n_samples=n_samples, radius=radius)
    if raw is None:
        print("No path found by PRM."); return None

    L0 = path_length(raw)
    pl.plot([p[0] for p in raw], [p[1] for p in raw], "r-", lw=2, label="raw")

    t0 = time.time()
    cut = path_shortcutting(raw, env, maxrep=maxrep, step=step, seed=seed)
    t1 = time.time()
    L1 = path_length(cut)
    pl.plot([p[0] for p in cut], [p[1] for p in cut], "g--", lw=2, label="shortcut")

    if smooth:
        s = smooth_path_moving_average(cut, window=5, repeat=3)
        L2 = path_length(s)
        pl.plot([p[0] for p in s], [p[1] for p in s], "b-", lw=2, label="smoothed")
        title = f"Post-processing: L0={L0:.3f}  L1={L1:.3f}  L2={L2:.3f}  (shortcut {t1-t0:.3f}s)"
        out = (raw, cut, s, (L0, L1, L2))
    else:
        title = f"Post-processing: L0={L0:.3f}  L1={L1:.3f}  (shortcut {t1-t0:.3f}s)"
        out = (raw, cut, None, (L0, L1, None))

    pl.legend(loc="best"); pl.title(title); pl.pause(0.001)
    print(title)
    return out

def benchmark(num_env=3, num_queries=5, n_samples=600, radius=0.6,
              maxrep=400, step=0.01, seed=42):
    rng = random.Random(seed)
    lengths = []
    gains = []
    t_total = 0.0
    succ = 0

    for e in range(num_env):
        env = environment_2d.Environment(10, 6, 5)
        for qid in range(num_queries):
            pl.clf(); env.plot()
            # deterministic-ish varied seeds
            s = seed + e*100 + qid
            np.random.seed(s)
            raw = find_path_with_prm(env, seed=s, n_samples=n_samples, radius=radius)
            if raw is None: continue
            L0 = path_length(raw)
            t0 = time.time()
            cut = path_shortcutting(raw, env, maxrep=maxrep, step=step, seed=s)
            t1 = time.time()
            L1 = path_length(cut)
            succ += 1
            lengths.append((L0, L1))
            gains.append((L0 - L1) / max(L0, 1e-9))
            t_total += (t1 - t0)

    if succ == 0:
        print("Benchmark: no successful paths.")
        return

    print("Benchmark results:")
    print(f"  total cases       : {num_env*num_queries}")
    print(f"  successful paths  : {succ}")
    print(f"  avg shortcut time : {t_total/succ:.4f} s")
    L0s = [x for x,_ in lengths]; L1s = [y for _,y in lengths]
    print(f"  avg raw length    : {np.mean(L0s):.3f}")
    print(f"  avg cut length    : {np.mean(L1s):.3f}")
    print(f"  avg relative gain : {np.mean(gains)*100:.2f}%")

if __name__ == "__main__":
    # Single demo
    # demo_once(seed=4, n_samples=600, radius=0.6, maxrep=400, smooth=True)
    # Or run a small benchmark:
    benchmark()
