# -*- coding: utf-8 -*-
"""Probabilistic roadmap (PRM) planner for the 2D OSR environments."""
from __future__ import annotations

import math
import random
import time
import heapq
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pylab as pl

import sys

# Ensure the environment module can be imported when the script is run directly.
if 'osr_examples/scripts/' not in sys.path:
    sys.path.append('osr_examples/scripts/')

import environment_2d  # type: ignore

try:  # Optional acceleration for neighbor search
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - SciPy may be unavailable on some setups
    cKDTree = None

Point = Tuple[float, float]


def euclid(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two planar points."""
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)


def path_length(path: Sequence[Point]) -> float:
    return sum(euclid(path[i], path[i + 1]) for i in range(len(path) - 1))


def line_collision_free(p: Point, q: Point, env: environment_2d.Environment, step: float = 0.015) -> bool:
    """Check whether the segment ``p-q`` stays collision free in ``env``.

    The segment is discretised with a step that scales with the segment length, so
    small features are sampled densely enough while keeping long-range checks
    inexpensive.
    """
    px, py = p
    qx, qy = q
    dx, dy = qx - px, qy - py
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        return not env.check_collision(px, py)

    # Sample proportionally to distance, but cap the number of tests to avoid
    # excessive evaluations on long straight segments.
    step = max(1e-3, step)
    n = max(2, min(int(dist / step) + 1, 500))
    for i in range(n + 1):
        t = float(i) / n
        x = px + t * dx
        y = py + t * dy
        if env.check_collision(x, y):
            return False
    return True


def _arc_lengths(path: Sequence[Point]) -> List[float]:
    cum = [0.0]
    for i in range(len(path) - 1):
        cum.append(cum[-1] + euclid(path[i], path[i + 1]))
    return cum


def _locate_point(
    path: Sequence[Point],
    cum_lengths: Sequence[float],
    s: float,
) -> Tuple[int, float, Point]:
    if s <= 0.0:
        return 0, 0.0, path[0]
    total = cum_lengths[-1]
    if s >= total:
        idx = len(path) - 2
        return idx, 1.0, path[-1]
    for idx in range(len(path) - 1):
        if s <= cum_lengths[idx + 1]:
            seg_len = cum_lengths[idx + 1] - cum_lengths[idx]
            if seg_len <= 1e-9:
                return idx, 0.0, path[idx]
            alpha = (s - cum_lengths[idx]) / seg_len
            px = path[idx][0] + alpha * (path[idx + 1][0] - path[idx][0])
            py = path[idx][1] + alpha * (path[idx + 1][1] - path[idx][1])
            return idx, alpha, (float(px), float(py))
    idx = len(path) - 2
    return idx, 1.0, path[-1]


def _insert_point(
    points: List[Point],
    seg_index: int,
    alpha: float,
    point: Point,
) -> Tuple[List[Point], int, bool]:
    eps = 1e-6
    if alpha <= eps:
        return points, seg_index, False
    if alpha >= 1.0 - eps:
        return points, seg_index + 1, False
    points.insert(seg_index + 1, point)
    return points, seg_index + 1, True


def path_shortcutting(
    path_xy: Sequence[Point],
    env: environment_2d.Environment,
    *,
    maxrep: int = 300,
    step: float = 0.015,
    seed: Optional[int] = None,
) -> List[Point]:
    if path_xy is None:
        return []
    if len(path_xy) < 3:
        return list(path_xy)

    rng = random.Random(seed)
    path = [tuple(p) for p in path_xy]

    for _ in range(maxrep):
        if len(path) <= 2:
            break
        cum = _arc_lengths(path)
        total_len = cum[-1]
        if total_len <= 1e-9:
            break

        s1, s2 = sorted(rng.uniform(0.0, total_len) for _ in range(2))
        if s2 - s1 < step:
            continue

        idx2, alpha2, point2 = _locate_point(path, cum, s2)
        idx1, alpha1, point1 = _locate_point(path, cum, s1)

        path, idx2_new, inserted2 = _insert_point(path, idx2, alpha2, point2)
        path, idx1_new, inserted1 = _insert_point(path, idx1, alpha1, point1)
        if inserted1 and idx1_new <= idx2_new:
            idx2_new += 1

        if idx2_new - idx1_new < 1:
            continue

        if not line_collision_free(path[idx1_new], path[idx2_new], env, step=step):
            continue

        path = path[: idx1_new + 1] + path[idx2_new:]

    return path


def _dijkstra(adj: Sequence[Sequence[Tuple[int, float]]], start_idx: int, goal_idx: int) -> Tuple[Optional[List[int]], int]:
    """Shortest path using Dijkstra with a binary heap.

    Returns the path as a list of node indices and the number of nodes expanded.
    """
    n = len(adj)
    dist = [float('inf')] * n
    prev = [-1] * n
    dist[start_idx] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, start_idx)]
    expanded = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        expanded += 1
        if u == goal_idx:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if not math.isfinite(dist[goal_idx]):
        return None, expanded

    path_idx: List[int] = []
    cur = goal_idx
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    return path_idx, expanded


def _astar(
    adj: Sequence[Sequence[Tuple[int, float]]],
    coords: Sequence[Point],
    start_idx: int,
    goal_idx: int,
) -> Tuple[Optional[List[int]], int]:
    """A* search using the Euclidean distance as an admissible heuristic."""

    n = len(adj)
    dist = [float('inf')] * n
    prev = [-1] * n
    dist[start_idx] = 0.0
    goal = coords[goal_idx]
    pq: List[Tuple[float, int]] = [(euclid(coords[start_idx], goal), start_idx)]
    expanded = 0

    while pq:
        f, u = heapq.heappop(pq)
        if f > dist[u] + euclid(coords[u], goal):
            continue
        expanded += 1
        if u == goal_idx:
            break
        for v, w in adj[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                priority = nd + euclid(coords[v], goal)
                heapq.heappush(pq, (priority, v))

    if not math.isfinite(dist[goal_idx]):
        return None, expanded

    path_idx: List[int] = []
    cur = goal_idx
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    return path_idx, expanded


@dataclass
class PRMStats:
    build_time: float
    search_time: float
    total_samples: int
    total_nodes: int
    algorithm: str = "dijkstra"
    path_length: Optional[float] = None
    expanded_nodes: Optional[int] = None


class ProbabilisticRoadmap2D:
    """Reusable PRM planner that separates roadmap construction from queries."""

    def __init__(
        self,
        n_samples: int = 600,
        connection_radius: float = 0.6,
        max_neighbors: Optional[int] = 15,
        collision_check_step: float = 0.015,
        max_sample_attempts: int = 20,
        rng: Optional[random.Random] = None,
    ) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if connection_radius <= 0.0:
            raise ValueError("connection_radius must be positive")
        self.n_samples = n_samples
        self.connection_radius = connection_radius
        self.max_neighbors = max_neighbors
        self.collision_check_step = collision_check_step
        self.max_sample_attempts = max_sample_attempts
        self._rng = rng or random.Random()

        self._nodes: List[Point] = []
        self._points: Optional[np.ndarray] = None
        self._adj: List[List[Tuple[int, float]]] = []
        self._tree: Optional["cKDTree"] = None
        self._env: Optional[environment_2d.Environment] = None
        self._last_build_time: float = 0.0
        self.last_path: Optional[List[Point]] = None
        self.last_shortcut_path: Optional[List[Point]] = None

    @property
    def nodes(self) -> List[Point]:
        return self._nodes

    @property
    def adjacency(self) -> List[List[Tuple[int, float]]]:
        return self._adj

    @property
    def environment(self) -> Optional[environment_2d.Environment]:
        return self._env

    def build(self, env: environment_2d.Environment, seed: Optional[int] = None) -> PRMStats:
        """Sample ``n_samples`` configurations and build a roadmap for ``env``."""
        if seed is not None:
            self._rng.seed(seed)

        self._env = env
        samples: List[Point] = []
        attempts_per_sample = max(1, self.max_sample_attempts)
        max_failures = attempts_per_sample * self.n_samples
        failures = 0

        while len(samples) < self.n_samples:
            x = self._rng.uniform(0.0, env.size_x)
            y = self._rng.uniform(0.0, env.size_y)
            if env.check_collision(x, y):
                failures += 1
                if failures > max_failures:
                    raise RuntimeError("Sampling stalled – increase workspace size or radius")
                continue
            samples.append((x, y))

        points = np.asarray(samples)
        t0 = time.time()
        candidate_lists = self._candidate_neighbors(points)
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(len(samples))]
        for i, neighbors in enumerate(candidate_lists):
            pi = samples[i]
            for j in neighbors:
                if j <= i:
                    continue
                pj = samples[j]
                if line_collision_free(pi, pj, env, step=self.collision_check_step):
                    weight = euclid(pi, pj)
                    adj[i].append((j, weight))
                    adj[j].append((i, weight))
        build_time = time.time() - t0

        self._nodes = samples
        self._points = points
        self._adj = adj
        self._last_build_time = build_time
        self._tree = cKDTree(points) if cKDTree is not None else None

        return PRMStats(build_time=build_time, search_time=0.0, total_samples=len(samples), total_nodes=len(samples))

    def query(
        self,
        start: Point,
        goal: Point,
        direct_connection: bool = True,
        algorithm: str = "dijkstra",
    ) -> Tuple[Optional[List[Point]], PRMStats]:
        """Plan from ``start`` to ``goal`` using the pre-built roadmap."""
        if self._env is None:
            raise RuntimeError("Roadmap not built – call build() first")
        env = self._env
        if env.check_collision(*start):
            raise ValueError("Start lies in collision")
        if env.check_collision(*goal):
            raise ValueError("Goal lies in collision")

        base_nodes = list(self._nodes)
        base_adj = [list(neigh) for neigh in self._adj]

        start_idx = len(base_nodes)
        goal_idx = len(base_nodes) + 1
        base_nodes.extend([start, goal])
        base_adj.extend([[], []])

        t0 = time.time()
        start_neighbors = self._connect_to_graph(start)
        goal_neighbors = self._connect_to_graph(goal)

        if direct_connection and line_collision_free(start, goal, env, step=self.collision_check_step):
            dist_start_goal = euclid(start, goal)
            start_neighbors.append((goal_idx, dist_start_goal))
            goal_neighbors.append((start_idx, dist_start_goal))

        for j, w in start_neighbors:
            base_adj[start_idx].append((j, w))
            base_adj[j].append((start_idx, w))
        for j, w in goal_neighbors:
            base_adj[goal_idx].append((j, w))
            base_adj[j].append((goal_idx, w))

        algo_name = algorithm.lower()
        if algo_name not in {"dijkstra", "astar"}:
            raise ValueError("Unsupported search algorithm: %s" % algorithm)

        if algo_name == "astar":
            path_idx, expanded = _astar(base_adj, base_nodes, start_idx, goal_idx)
        else:
            path_idx, expanded = _dijkstra(base_adj, start_idx, goal_idx)

        search_time = time.time() - t0
        stats = PRMStats(
            build_time=self._last_build_time,
            search_time=search_time,
            total_samples=self.n_samples,
            total_nodes=len(base_nodes),
            algorithm=algo_name,
            expanded_nodes=expanded,
        )
        self.last_path = None
        self.last_shortcut_path = None
        if path_idx is None:
            return None, stats
        path_xy = [base_nodes[i] for i in path_idx]
        stats.path_length = path_length(path_xy)
        self.last_path = path_xy
        return path_xy, stats

    # --- internals -----------------------------------------------------
    def _candidate_neighbors(self, points: np.ndarray) -> List[List[int]]:
        n = len(points)
        if n == 0:
            return []
        radius_sq = self.connection_radius ** 2
        max_neighbors = self.max_neighbors

        if cKDTree is not None:
            tree = cKDTree(points)
            neighbor_lists = []
            for i in range(n):
                idxs = tree.query_ball_point(points[i], self.connection_radius)
                filtered = [j for j in idxs if j != i]
                if filtered and max_neighbors is not None:
                    filtered = self._select_nearest(points, i, filtered, max_neighbors)
                neighbor_lists.append(filtered)
            return neighbor_lists

        # Fallback: brute-force search
        neighbor_lists = [[] for _ in range(n)]
        for i in range(n):
            pi = points[i]
            for j in range(i + 1, n):
                pj = points[j]
                if np.sum((pi - pj) ** 2) <= radius_sq:
                    neighbor_lists[i].append(j)
                    neighbor_lists[j].append(i)

        if max_neighbors is not None:
            for i in range(n):
                if len(neighbor_lists[i]) > max_neighbors:
                    neighbor_lists[i] = self._select_nearest(points, i, neighbor_lists[i], max_neighbors)
        return neighbor_lists

    def _select_nearest(self, points: np.ndarray, index: int, candidates: Iterable[int], k: int) -> List[int]:
        cand = list(candidates)
        if len(cand) <= k:
            return cand
        pi = points[index]
        dists = [(float(np.linalg.norm(points[j] - pi)), j) for j in cand]
        dists.sort()
        return [j for _, j in dists[:k]]

    def _connect_to_graph(self, point: Point) -> List[Tuple[int, float]]:
        if self._env is None:
            raise RuntimeError("Roadmap not built – call build() first")
        env = self._env
        if self._points is None or not len(self._points):
            return []

        if self._tree is not None:
            idxs = [i for i in self._tree.query_ball_point(point, self.connection_radius) if i < len(self._nodes)]
        else:
            diff = self._points - point
            d2 = np.einsum('ij,ij->i', diff, diff)
            idxs = [int(i) for i in np.where(d2 <= self.connection_radius ** 2)[0]]

        if not idxs:
            return []

        distances = [float(np.linalg.norm(self._points[idx] - point)) for idx in idxs]
        order = list(range(len(idxs)))
        order.sort(key=lambda k: distances[k])
        if self.max_neighbors is not None:
            order = order[: self.max_neighbors]

        neighbors: List[Tuple[int, float]] = []
        for k in order:
            idx = idxs[k]
            dist = distances[k]
            node = self._nodes[idx]
            if line_collision_free(point, node, env, step=self.collision_check_step):
                neighbors.append((idx, dist))
        return neighbors


# ---------------------------------------------------------------------------
# Convenience helpers mirroring the original script API
# ---------------------------------------------------------------------------

def prm_plan(
    env: environment_2d.Environment,
    x_start: float,
    y_start: float,
    x_goal: float,
    y_goal: float,
    n_samples: int = 600,
    radius: float = 0.6,
    step: float = 0.015,
    seed: Optional[int] = None,
    verbose: bool = True,
    max_neighbors: Optional[int] = 15,
    direct_connection: bool = True,
    visualize: bool = True,
    apply_shortcut: bool = True,
    shortcut_maxrep: int = 400,
    shortcut_seed: Optional[int] = None,
    search_algorithm: str = "dijkstra",
):
    """Build a roadmap and attempt to solve a single query, preserving the
    interface of the original educational script."""

    planner = ProbabilisticRoadmap2D(
        n_samples=n_samples,
        connection_radius=radius,
        max_neighbors=max_neighbors,
        collision_check_step=step,
    )
    stats_build = planner.build(env, seed=seed)
    path_xy, stats_query = planner.query(
        (x_start, y_start),
        (x_goal, y_goal),
        direct_connection=direct_connection,
        algorithm=search_algorithm,
    )

    shortcut_path: Optional[List[Point]] = None
    if path_xy is not None and apply_shortcut:
        shortcut_seed_eff = shortcut_seed if shortcut_seed is not None else seed
        shortcut_path = path_shortcutting(
            path_xy,
            env,
            maxrep=shortcut_maxrep,
            step=step,
            seed=shortcut_seed_eff,
        )
        planner.last_shortcut_path = shortcut_path
    else:
        planner.last_shortcut_path = None

    if verbose:
        if path_xy is None:
            print(
                "No path found. build: %.3fs  search: %.3fs  nodes=%d"
                % (stats_build.build_time, stats_query.search_time, stats_query.total_nodes)
            )
        else:
            raw_len = stats_query.path_length or path_length(path_xy)
            msgs = [
                f"Path found using {stats_query.algorithm.upper()}:",
                "  roadmap nodes : %d" % len(planner.nodes),
                "  build time    : %.3fs" % stats_build.build_time,
                "  search time   : %.3fs" % stats_query.search_time,
                "  raw length    : %.3f" % raw_len,
            ]
            if shortcut_path is not None and len(shortcut_path) >= 2:
                short_len = path_length(shortcut_path)
                gain_abs = raw_len - short_len
                gain_rel = (gain_abs / raw_len) * 100.0 if raw_len > 1e-9 else float("nan")
                msgs.extend(
                    [
                        "  shortcut len : %.3f" % short_len,
                        "  gain         : %.3f (%.2f%% shorter)" % (gain_abs, gain_rel),
                    ]
                )
            print("\n".join(msgs))

    if visualize:
        pl.clf()
        env.plot()
        env.plot_query(x_start, y_start, x_goal, y_goal)
        _draw_roadmap(planner)
        if path_xy is not None:
            xs = [p[0] for p in path_xy]
            ys = [p[1] for p in path_xy]
            pl.plot(xs, ys, "g-", linewidth=2, label="PRM path")
            if shortcut_path is not None and len(shortcut_path) >= 2:
                xs_s = [p[0] for p in shortcut_path]
                ys_s = [p[1] for p in shortcut_path]
                pl.plot(xs_s, ys_s, linestyle="--", linewidth=2, color="#720ab8", label="Shortcut")
                pl.legend(loc="best")
            pl.title("PRM path")
        pl.pause(0.001)

    return path_xy, planner, stats_query


def _draw_roadmap(planner: ProbabilisticRoadmap2D) -> None:
    # Light-weight visualisation helper; keeps plotting optional for production use.
    nodes = planner.nodes
    adj = planner.adjacency
    if not nodes:
        return
    xs, ys = zip(*nodes)
    pl.plot(xs, ys, ".", alpha=0.3)
    for i, neighbors in enumerate(adj):
        xi, yi = nodes[i]
        for j, _ in neighbors:
            if j <= i:
                continue
            xj, yj = nodes[j]
            pl.plot([xi, xj], [yi, yj], "k-", alpha=0.1)


def run_single(
    seed: int = 4,
    n_samples: int = 600,
    radius: float = 0.6,
    algorithms: Sequence[str] = ("dijkstra", "astar"),
    visualize_path: bool = True,
    show_hist: bool = True,
) -> Optional[dict]:
    print("\n=== Visualize single case ===")
    np.random.seed(seed)
    env = environment_2d.Environment(10, 6, 5)
    q = env.random_query()
    if q is None:
        print("No query generated")
        return None

    x_start, y_start, x_goal, y_goal = q
    planner = ProbabilisticRoadmap2D(
        n_samples=n_samples,
        connection_radius=radius,
    )
    stats_build = planner.build(env, seed=seed)

    results = {}
    for idx, algo in enumerate(algorithms):
        path_xy, stats_query = planner.query(
            (x_start, y_start),
            (x_goal, y_goal),
            algorithm=algo,
        )
        shortcut_path: Optional[List[Point]] = None
        if path_xy is not None:
            shortcut_candidate = path_shortcutting(
                path_xy,
                env,
                maxrep=400,
                step=planner.collision_check_step,
                seed=(seed if seed is not None else 0) + idx,
            )
            if shortcut_candidate and len(shortcut_candidate) >= 2:
                shortcut_path = shortcut_candidate
        results[algo.lower()] = {
            "path": path_xy,
            "stats": stats_query,
            "shortcut": shortcut_path,
        }

    if visualize_path:
        # primary_algo = algorithms[0].lower()
        color_map = {
            "dijkstra": "#4daf4a",
            "astar": "#377eb8",
        }
        shortcut_map = {
            "dijkstra": "#814131",
            "astar": "#b937a7",
        }
        pl.figure("prm_run_single_path")
        pl.clf()
        env.plot()
        env.plot_query(x_start, y_start, x_goal, y_goal)
        _draw_roadmap(planner)
        for algo in algorithms:
            key = algo.lower()
            res = results.get(key)
            if not res:
                continue
            path_raw = res.get("path")
            path_short = res.get("shortcut")
            if path_raw is not None:
                xs = [p[0] for p in path_raw]
                ys = [p[1] for p in path_raw]
                pl.plot(
                    xs,
                    ys,
                    color=color_map.get(key, None),
                    linewidth=2,
                    label=f"{key.upper()} raw",
                )
            if path_short is not None and len(path_short) >= 2:
                xs_s = [p[0] for p in path_short]
                ys_s = [p[1] for p in path_short]
                pl.plot(
                    xs_s,
                    ys_s,
                    linestyle="--",
                    linewidth=2,
                    color=shortcut_map.get(key, "#555555"),
                    label=f"{key.upper()} shortcut",
                )
        pl.title("PRM path comparison")
        pl.legend(loc="best")
        pl.pause(0.001)

    if show_hist:
        labels = []
        search_times = []
        path_lengths = []
        shortcut_lengths = []
        expansions = []
        gains = []
        for algo in algorithms:
            key = algo.lower()
            res = results.get(key)
            if res is None or res["path"] is None:
                continue
            stats = res["stats"]
            labels.append(key.upper())
            search_times.append(stats.search_time)
            raw_len = stats.path_length if stats.path_length is not None else path_length(res["path"])
            path_lengths.append(raw_len)
            shortcut = res.get("shortcut")
            if shortcut and len(shortcut) >= 2:
                sc_len = path_length(shortcut)
                shortcut_lengths.append(sc_len)
                gains.append(raw_len - sc_len)
            else:
                shortcut_lengths.append(float("nan"))
                gains.append(float("nan"))
            expansions.append(stats.expanded_nodes or 0)

        if labels:
            fig, axes = pl.subplots(1, 3, figsize=(14, 4))
            colors = ["#1b9e77", "#d95f02", "#7570b3"]
            axes[0].bar(labels, search_times, color=colors[: len(labels)])
            axes[0].set_title("Search Time (s)")
            axes[0].set_ylabel("seconds")

            width = 0.35
            x = np.arange(len(labels))
            ax1 = axes[1]
            ax1.bar(x - width / 2, path_lengths, width=width, color="#377eb8", label="raw")
            ax1.bar(
                x + width / 2,
                [v if not math.isnan(v) else 0.0 for v in shortcut_lengths],
                width=width,
                color="#4daf4a",
                label="shortcut",
            )
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.set_title("Path Length vs Shortcut")
            ax1.set_ylabel("length")
            ax1.legend()

            axes[2].bar(labels, expansions, color=colors[: len(labels)])
            axes[2].set_title("Expanded Nodes")
            axes[2].set_ylabel("count")

            for ax in axes:
                ax.set_ylim(bottom=0)
            fig.suptitle("Algorithm Comparison")
            fig.tight_layout()

    print(f"Roadmap build time: {stats_build.build_time:.3f}s (nodes={stats_build.total_nodes})")
    for algo in algorithms:
        key = algo.lower()
        res = results.get(key)
        if res is None or res["path"] is None:
            print(f"  {algo.upper()}: failed to find a path")
            continue
        stats = res["stats"]
        raw_len = stats.path_length if stats.path_length is not None else path_length(res["path"])
        short = res.get("shortcut")
        short_len = path_length(short) if short is not None else float("nan")
        gain = raw_len - short_len if short is not None else float("nan")
        gain_pct = (gain / raw_len * 100.0) if short is not None and raw_len > 1e-9 else float("nan")
        print(
            "  %s -> time: %.3fs | length: %.3f | shortcut: %s | gain: %s (%.2f%%) | expanded: %s"
            % (
                algo.upper(),
                stats.search_time,
                raw_len,
                f"{short_len:.3f}" if short else "n/a",
                f"{gain:.3f}" if short else "n/a",
                gain_pct if not math.isnan(gain_pct) else float("nan"),
                stats.expanded_nodes if stats.expanded_nodes is not None else "n/a",
            )
        )

    return results


def benchmark(
    num_env: int = 3,
    num_queries: int = 5,
    n_samples: int = 600,
    radius: float = 0.6,
    max_neighbors: Optional[int] = 15,
    algorithms: Sequence[str] = ("dijkstra", "astar"),
    show_hist: bool = True,
) -> None:
    total_build = 0.0
    roadmap_nodes_total = 0
    metrics = {
        algo.lower(): {
            "times": [],
            "lengths": [],
            "shortcut_lengths": [],
            "shortcut_times": [],
            "expanded": [],
            "gains_abs": [],
            "gains_rel": [],
            "success": 0,
        }
        for algo in algorithms
    }

    for env_idx in range(num_env):
        env = environment_2d.Environment(10, 6, 5)
        planner = ProbabilisticRoadmap2D(
            n_samples=n_samples,
            connection_radius=radius,
            max_neighbors=max_neighbors,
        )
        stats_build = planner.build(env, seed=env_idx)
        total_build += stats_build.build_time
        roadmap_nodes_total += stats_build.total_nodes

        for q_idx in range(num_queries):
            q = env.random_query()
            if q is None:
                continue
            x_start, y_start, x_goal, y_goal = q
            for algo in algorithms:
                key = algo.lower()
                path_xy, stats_query = planner.query(
                    (x_start, y_start),
                    (x_goal, y_goal),
                    algorithm=algo,
                )
                if path_xy is None:
                    continue
                metrics[key]["success"] += 1
                metrics[key]["times"].append(stats_query.search_time)
                raw_len = stats_query.path_length or path_length(path_xy)
                metrics[key]["lengths"].append(raw_len)

                sc_start = time.time()
                shortcut = path_shortcutting(
                    path_xy,
                    env,
                    maxrep=400,
                    step=planner.collision_check_step,
                    seed=env_idx * 10_000 + q_idx,
                )
                sc_duration = time.time() - sc_start
                metrics[key]["shortcut_times"].append(sc_duration)
                if shortcut and len(shortcut) >= 2:
                    sc_len = path_length(shortcut)
                    gain_abs = raw_len - sc_len
                    gain_rel = gain_abs / raw_len if raw_len > 1e-9 else 0.0
                    metrics[key]["shortcut_lengths"].append(sc_len)
                    metrics[key]["gains_abs"].append(gain_abs)
                    metrics[key]["gains_rel"].append(gain_rel)
                else:
                    metrics[key]["shortcut_lengths"].append(float('nan'))
                    metrics[key]["gains_abs"].append(float('nan'))
                    metrics[key]["gains_rel"].append(float('nan'))

                if stats_query.expanded_nodes is not None:
                    metrics[key]["expanded"].append(stats_query.expanded_nodes)

    total_cases = num_env * num_queries

    print("\n=== PRM Benchmark (algorithm comparison) ===")
    print(f"  Environments x Queries : {num_env} x {num_queries} (total {total_cases})")
    print(f"  Avg roadmap build      : {total_build / max(1, num_env):.3f} s  | avg nodes: {roadmap_nodes_total / max(1, num_env):.1f}")

    labels: List[str] = []
    avg_times: List[float] = []
    avg_lengths: List[float] = []
    avg_shortcuts: List[float] = []
    avg_shortcut_times: List[float] = []
    avg_expanded: List[float] = []
    success_rates: List[float] = []

    for algo in algorithms:
        key = algo.lower()
        data = metrics[key]
        succ = data["success"]
        labels.append(key.upper())
        if succ == 0:
            print(f"  {algo.upper()}: no successful paths")
            avg_times.append(0.0)
            avg_lengths.append(float('nan'))
            avg_shortcuts.append(float('nan'))
            avg_shortcut_times.append(0.0)
            avg_expanded.append(0.0)
            success_rates.append(0.0)
            continue

        avg_time = float(np.mean(data["times"]))
        avg_len = float(np.mean(data["lengths"])) if data["lengths"] else float('nan')
        sc_vals = [v for v in data["shortcut_lengths"] if not math.isnan(v)]
        avg_short_len = float(np.mean(sc_vals)) if sc_vals else float('nan')
        avg_sc_time = float(np.mean(data["shortcut_times"])) if data["shortcut_times"] else 0.0
        avg_exp = float(np.mean(data["expanded"])) if data["expanded"] else 0.0
        gains_rel = [g for g in data["gains_rel"] if not math.isnan(g)]
        avg_gain_pct = float(np.mean(gains_rel) * 100.0) if gains_rel else float('nan')
        best_gain = max(gains_rel) * 100.0 if gains_rel else float('nan')
        worst_gain = min(gains_rel) * 100.0 if gains_rel else float('nan')

        avg_times.append(avg_time)
        avg_lengths.append(avg_len)
        avg_shortcuts.append(avg_short_len)
        avg_shortcut_times.append(avg_sc_time)
        avg_expanded.append(avg_exp)
        success_rates.append(succ / total_cases)

        print(
            f"  {algo.upper():8s} -> success: {succ}/{total_cases}  avg_time: {avg_time:.4f}s  avg_length: {avg_len:.3f} "
            f"avg_shortcut: {avg_short_len:.3f}  shortcut_time: {avg_sc_time:.4f}s  avg_expanded: {avg_exp:.1f}"
        )
        if gains_rel:
            print(
                f"    shortcut gain avg: {avg_gain_pct:.2f}%  best: {best_gain:.2f}%  worst: {worst_gain:.2f}%"
            )

    if show_hist and any(metrics[a.lower()]["success"] > 0 for a in algorithms):
        colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
        fig, axes = pl.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        axes[0].bar(labels, avg_times, color=colors[: len(labels)])
        axes[0].set_title("Avg Search Time")
        axes[0].set_ylabel("seconds")

        axes[1].bar(labels, avg_shortcut_times, color=colors[: len(labels)])
        axes[1].set_title("Avg Shortcut Time")
        axes[1].set_ylabel("seconds")

        axes[2].bar(labels, avg_lengths, color=colors[: len(labels)])
        axes[2].set_title("Avg Path Length")
        axes[2].set_ylabel("length")

        axes[3].bar(labels, avg_shortcuts, color=colors[: len(labels)])
        axes[3].set_title("Avg Shortcut Length")
        axes[3].set_ylabel("length")

        axes[4].bar(labels, avg_expanded, color=colors[: len(labels)])
        axes[4].set_title("Avg Expanded Nodes")
        axes[4].set_ylabel("count")

        axes[5].bar(labels, [rate * 100.0 for rate in success_rates], color=colors[: len(labels)])
        axes[5].set_title("Success Rate")
        axes[5].set_ylabel("percent")

        for ax in axes:
            ax.set_ylim(bottom=0)

        fig.suptitle("Algorithm Benchmark Comparison")
        fig.tight_layout(rect=(0, 0, 1, 0.97))



if __name__ == "__main__":
    pl.ion()
    # benchmark()
    run_single(seed=4, n_samples=800, radius=0.6)
    pl.show()
