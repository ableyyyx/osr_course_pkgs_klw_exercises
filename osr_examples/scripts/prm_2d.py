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


def _dijkstra(adj: Sequence[Sequence[Tuple[int, float]]], start_idx: int, goal_idx: int) -> Optional[List[int]]:
    """Shortest path using Dijkstra with a binary heap."""
    n = len(adj)
    dist = [float('inf')] * n
    prev = [-1] * n
    dist[start_idx] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, start_idx)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == goal_idx:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if not math.isfinite(dist[goal_idx]):
        return None

    path_idx: List[int] = []
    cur = goal_idx
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    return path_idx


@dataclass
class PRMStats:
    build_time: float
    search_time: float
    total_samples: int
    total_nodes: int


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

        path_idx = _dijkstra(base_adj, start_idx, goal_idx)
        search_time = time.time() - t0
        stats = PRMStats(
            build_time=self._last_build_time,
            search_time=search_time,
            total_samples=self.n_samples,
            total_nodes=len(base_nodes),
        )
        self.last_path = None
        self.last_shortcut_path = None
        if path_idx is None:
            return None, stats
        path_xy = [base_nodes[i] for i in path_idx]
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
    path_xy, stats_query = planner.query((x_start, y_start), (x_goal, y_goal), direct_connection=direct_connection)

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
                "No path found. build: %.3fs search: %.3fs nodes=%d"
                % (stats_build.build_time, stats_query.search_time, stats_query.total_nodes)
            )
        else:
            length = path_length(path_xy)
            msg = (
                "Path found. #roadmap_nodes=%d build: %.3fs search: %.3fs length=%.3f"
                % (len(planner.nodes), stats_build.build_time, stats_query.search_time, length)
            )
            if shortcut_path is not None and len(shortcut_path) >= 2:
                msg += " shortcut=%.3f" % path_length(shortcut_path)
            print(msg)

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

    return path_xy, planner


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


def run_single(seed: int = 4, n_samples: int = 600, radius: float = 0.6) -> Optional[List[Point]]:
    np.random.seed(seed)
    env = environment_2d.Environment(10, 6, 5)
    q = env.random_query()
    if q is None:
        print("No query generated")
        return None
    x_start, y_start, x_goal, y_goal = q
    path_xy, _ = prm_plan(
        env,
        x_start,
        y_start,
        x_goal,
        y_goal,
        n_samples=n_samples,
        radius=radius,
        seed=seed,
        visualize=True,
    )
    return path_xy


def benchmark(
    num_env: int = 3,
    num_queries: int = 5,
    n_samples: int = 600,
    radius: float = 0.6,
    max_neighbors: Optional[int] = 15,
) -> None:
    lengths: List[Tuple[float, float]] = []
    success = 0
    total_build = 0.0
    total_search = 0.0

    for env_idx in range(num_env):
        env = environment_2d.Environment(10, 6, 5)
        planner = ProbabilisticRoadmap2D(
            n_samples=n_samples,
            connection_radius=radius,
            max_neighbors=max_neighbors,
        )
        stats_build = planner.build(env, seed=env_idx)
        total_build += stats_build.build_time

        for q_idx in range(num_queries):
            q = env.random_query()
            if q is None:
                continue
            x_start, y_start, x_goal, y_goal = q
            path_xy, stats_query = planner.query((x_start, y_start), (x_goal, y_goal))
            total_search += stats_query.search_time
            if path_xy is not None:
                success += 1
                raw_len = path_length(path_xy)
                shortcut = path_shortcutting(
                    path_xy,
                    env,
                    maxrep=400,
                    step=planner.collision_check_step,
                    seed=env_idx * 10_000 + q_idx,
                )
                lengths.append((raw_len, path_length(shortcut)))

    denom = max(1, num_env * num_queries)
    avg_raw = float(np.mean([x for x, _ in lengths])) if lengths else float('nan')
    avg_short = float(np.mean([y for _, y in lengths])) if lengths else float('nan')
    print(
        "PRM benchmark: env=%d queries/env=%d success=%d avg_len=%.3f avg_short=%.3f avg_build=%.3fs avg_search=%.3fs"
        % (
            num_env,
            num_queries,
            success,
            avg_raw,
            avg_short,
            total_build / num_env,
            total_search / denom,
        )
    )


if __name__ == "__main__":
    benchmark()
    pl.ion()
    run_single(seed=4, n_samples=800, radius=0.6)
    pl.show()
