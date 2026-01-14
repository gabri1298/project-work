from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from tqdm.auto import tqdm

from Problem import Problem


SHOW_PROGRESS = True
MAX_ITERS = 1000
LNS_SEED: int | None = None

Visit = Tuple[int, float]
Trip = List[Visit]


def _tqdm(it: Iterable, **kwargs):
    return tqdm(it, **kwargs) if SHOW_PROGRESS else it


@dataclass(frozen=True)
class _PathStats:
    dist_sum: float
    dist_beta_sum: float


# Edge-aggregated cost for a shortest path summary.
# Matches expanding the path edge-by-edge:
# sum(d_e + (alpha * d_e * carried)^beta) = dist_sum + (alpha*carried)^beta * sum(d_e^beta)
def _leg_cost(alpha: float, beta: float, st: _PathStats, carried: float) -> float:

    if st.dist_sum == 0.0:
        return 0.0
    if carried <= 0.0 or alpha <= 0.0:
        return st.dist_sum
    return st.dist_sum + (alpha * carried) ** beta * st.dist_beta_sum



# On-demand shortest paths cache (path nodes + summary stats).
class _ShortestPathCache:
 

    def __init__(self, g: nx.Graph, beta: float):
        self.g = g
        self.beta = float(beta)
        self._cache: Dict[Tuple[int, int], Tuple[List[int], _PathStats]] = {}
        self._has_negative_weights = any(float(ed.get("dist", 0.0)) < 0.0 for _, _, ed in g.edges(data=True))
       
        self._g_no0 = nx.subgraph_view(g, filter_node=lambda n: int(n) != 0)

        
        self._node_all: List[int] = sorted(int(n) for n in self.g.nodes)
        self._idx_all: Dict[int, int] = {node: i for i, node in enumerate(self._node_all)}
        self._dist0_all, self._pred0_all = self._precompute_from_source0()


        self._node_no0: List[int] = sorted(int(n) for n in self.g.nodes if int(n) != 0)
        self._idx_no0: Dict[int, int] = {node: i for i, node in enumerate(self._node_no0)}
        self._dist_no0, self._pred_no0 = self._precompute_no0_all_pairs()

    def _precompute_from_source0(self) -> Tuple[np.ndarray, np.ndarray]:
        nodes_all = self._node_all
        k = len(nodes_all)
        if k <= 1:
            dist0 = np.zeros((k,), dtype=float)
            pred0 = np.full((k,), -9999, dtype=np.int32)
            return dist0, pred0

        i0 = self._idx_all.get(0)
        if i0 is None:
            raise KeyError("node 0 not found")

        mat = nx.to_scipy_sparse_array(
            self.g,
            nodelist=nodes_all,
            weight="dist",
            dtype=float,
            format="csr",
        )

        method = "BF" if self._has_negative_weights else "auto"
        dist0, pred0 = shortest_path(
            mat,
            method=method,
            directed=False,
            unweighted=False,
            indices=int(i0),
            return_predecessors=True,
        )

        dist0 = np.asarray(dist0, dtype=float).reshape(-1)
        pred0 = np.asarray(pred0, dtype=np.int32).reshape(-1)
        return dist0, pred0

    def _precompute_no0_all_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        nodes_no0 = self._node_no0
        k = len(nodes_no0)
        if k <= 1:
            dist = np.zeros((k, k), dtype=float)
            pred = np.full((k, k), -9999, dtype=np.int32)
            return dist, pred

        
        mat = nx.to_scipy_sparse_array(
            self._g_no0,
            nodelist=nodes_no0,
            weight="dist",
            dtype=float,
            format="csr",
        )

        method = "BF" if self._has_negative_weights else "auto"
        dist, pred = shortest_path(
            mat,
            method=method,
            directed=False,
            unweighted=False,
            return_predecessors=True,
        )
        return dist, pred.astype(np.int32, copy=False)

    def _reconstruct_no0_path_and_beta_sum(self, u: int, v: int) -> Tuple[List[int], float]:
        iu = self._idx_no0.get(int(u))
        iv = self._idx_no0.get(int(v))
        if iu is None or iv is None:
            raise KeyError("node not in non-zero index")
        return self._reconstruct_path_and_beta_sum(self._pred_no0, self._node_no0, iu, iv, err="unreachable pair in no-0 graph")

    def _reconstruct_0_to_path_and_beta_sum(self, v: int) -> Tuple[List[int], float]:
        iv = self._idx_all.get(int(v))
        if iv is None:
            raise KeyError("node not in full index")
        i0 = self._idx_all.get(0)
        if i0 is None:
            raise KeyError("node 0 not found")
        return self._reconstruct_from_pred0_and_beta_sum(self._pred0_all, self._node_all, int(i0), int(iv), err="unreachable from 0")

    def _reconstruct_path_and_beta_sum(
        self,
        pred: np.ndarray,
        nodes: List[int],
        iu: int,
        iv: int,
        err: str,
    ) -> Tuple[List[int], float]:
        if iu == iv:
            return [int(nodes[iu])], 0.0

        
        cur = int(iv)
        path_idx = [cur]
        while cur != iu:
            prev = int(pred[iu, cur])
            if prev < 0:
                raise ValueError(err)
            path_idx.append(prev)
            cur = prev

        path_idx.reverse()
        path_nodes = [int(nodes[i]) for i in path_idx]

        dist_beta_sum = 0.0
        for a, b in zip(path_nodes, path_nodes[1:]):
            dist_beta_sum += float(self.g.edges[int(a), int(b)]["dist"]) ** self.beta

        return path_nodes, float(dist_beta_sum)

    def _reconstruct_from_pred0_and_beta_sum(
        self,
        pred0: np.ndarray,
        nodes: List[int],
        i0: int,
        iv: int,
        err: str,
    ) -> Tuple[List[int], float]:
        if i0 == iv:
            return [int(nodes[i0])], 0.0

        cur = int(iv)
        path_idx = [cur]
        while cur != i0:
            prev = int(pred0[cur])
            if prev < 0:
                raise ValueError(err)
            path_idx.append(prev)
            cur = prev

        path_idx.reverse()
        path_nodes = [int(nodes[i]) for i in path_idx]

        dist_beta_sum = 0.0
        for a, b in zip(path_nodes, path_nodes[1:]):
            dist_beta_sum += float(self.g.edges[int(a), int(b)]["dist"]) ** self.beta

        return path_nodes, float(dist_beta_sum)

    def get(self, u: int, v: int) -> Tuple[List[int], _PathStats]:
        if u == v:
            return [u], _PathStats(0.0, 0.0)
        key = (u, v)
        got = self._cache.get(key)
        if got is not None:
            return got

        if u != 0 and v != 0:
            iu = self._idx_no0.get(int(u))
            iv = self._idx_no0.get(int(v))
            if iu is None or iv is None:
                raise KeyError("node not in non-zero index")
            dist_sum = float(self._dist_no0[iu, iv])
            if not np.isfinite(dist_sum):
                raise ValueError("unreachable pair in no-0 graph")
            path_nodes, dist_beta_sum = self._reconstruct_no0_path_and_beta_sum(int(u), int(v))
            st = _PathStats(dist_sum=dist_sum, dist_beta_sum=dist_beta_sum)
            self._cache[key] = (path_nodes, st)
            self._cache[(v, u)] = (list(reversed(path_nodes)), st)
            return path_nodes, st

        if u == 0:
            dist_sum = float(self._dist0_all[self._idx_all[int(v)]])
            if not np.isfinite(dist_sum):
                raise ValueError("unreachable from 0")
            path_nodes, dist_beta_sum = self._reconstruct_0_to_path_and_beta_sum(int(v))
        else:
            dist_sum = float(self._dist0_all[self._idx_all[int(u)]])
            if not np.isfinite(dist_sum):
                raise ValueError("unreachable from 0")
            path_nodes, dist_beta_sum = self._reconstruct_0_to_path_and_beta_sum(int(u))
            path_nodes = list(reversed(path_nodes))
        st = _PathStats(dist_sum=dist_sum, dist_beta_sum=dist_beta_sum)
        self._cache[key] = (path_nodes, st)
        self._cache[(v, u)] = (list(reversed(path_nodes)), st)
        return path_nodes, st

    def _stats_from_path(self, path_nodes: List[int], dist_sum: float) -> _PathStats:
        if len(path_nodes) < 2:
            return _PathStats(0.0, 0.0)
        dist_beta_sum = 0.0
        for a, b in zip(path_nodes, path_nodes[1:]):
            dist_beta_sum += float(self.g.edges[a, b]["dist"]) ** self.beta
        return _PathStats(dist_sum=dist_sum, dist_beta_sum=dist_beta_sum)


def _trip_taken_sum(trip: Sequence[Visit]) -> float:
    return float(sum(take for _, take in trip))

# Computes the cost of a single trip interpreted as: 0 -> visits -> 0.
# The carried load is cumulative within the trip: before moving to the next city we pay the leg cost
# using the current carried amount, then we increase carried by the pickup `take` at that city.
# Movement costs are evaluated on shortest paths provided by `sp` (with the no-0 constraint for
# inter-city legs and a dedicated 0->* precompute for depot legs).
def _trip_cost(trip: Sequence[Visit], alpha: float, beta: float, sp: _ShortestPathCache) -> float:
    
    carried = 0.0
    pos = 0
    total = 0.0
    for city, take in trip:
        _, st = sp.get(pos, city)
        total += _leg_cost(alpha, beta, st, carried)
        carried += float(take)
        pos = city
    _, st_back = sp.get(pos, 0)
    total += _leg_cost(alpha, beta, st_back, carried)
    return float(total)

# Cheap intra-trip local search: greedy adjacent swaps.
# Tries swapping each neighboring pair and keeps the swap only if it strictly improves the trip cost.
# This can reorder visits (including repeated cities). It does NOT merge consecutive equal-city visits;
# consecutive-visit merging is handled elsewhere (recreate normalization / final route `push()`).
def _local_improve_adjacent_swaps_trip(trip: Trip, alpha: float, beta: float, sp: _ShortestPathCache) -> None:

    if len(trip) < 2:
        return
    best_cost = _trip_cost(trip, alpha, beta, sp)

    max_rounds = 2
    for _ in range(max_rounds):
        improved = False
        for i in range(len(trip) - 1):
            trip[i], trip[i + 1] = trip[i + 1], trip[i]
            cand = _trip_cost(trip, alpha, beta, sp)
            if cand + 1e-12 < best_cost:
                best_cost = cand
                improved = True
            else:
                trip[i], trip[i + 1] = trip[i + 1], trip[i]
        if not improved:
            break


# Delta-cost model for the RECREATE step.
# We consider appending a new visit (city, take) at the END of an existing trip.
# The trip is interpreted as 0 -> ... -> last -> 0.
# Appending replaces the old return leg (last -> 0) with two legs: (last -> city) and (city -> 0).
#
# Inputs:
# - trip_taken_sum: precomputed sum of takes already in the trip (i.e., carried load upon leaving `last`).
# - take: requested pickup for the appended visit.
#
# Return value:
# - incremental cost: new_cost(trip + [(city,take)]) - old_cost(trip)
#
# Notes:
# - If trip is empty, this returns the full cost of a new single-visit trip 0->city->0.
# - Shortest-path legs come from `sp` (which enforces the no-0 constraint for inter-city paths).
def _append_delta_visit(
    trip: Sequence[Visit],
    trip_taken_sum: float,
    city: int,
    take: float,
    alpha: float,
    beta: float,
    sp: _ShortestPathCache,
) -> float:


    if take <= 0.0:
        return 0.0

    if not trip:
        _, st1 = sp.get(0, city)
        _, st2 = sp.get(city, 0)
        return _leg_cost(alpha, beta, st1, 0.0) + _leg_cost(alpha, beta, st2, float(take))

    last = int(trip[-1][0])
    carried = float(trip_taken_sum)

    _, st_old = sp.get(last, 0)
    old_back = _leg_cost(alpha, beta, st_old, carried)

    _, st_a = sp.get(last, city)
    _, st_b = sp.get(city, 0)
    new_part = _leg_cost(alpha, beta, st_a, carried) + _leg_cost(alpha, beta, st_b, carried + float(take))
    return float(new_part - old_back)

# Heuristic "soft" target load used to decide chunk sizes during RECREATE.
# Intuition: for beta > 1, carrying large loads is increasingly penalized, so we prefer splitting
# city gold into smaller chunks and distributing them across trips; for beta ~ 1 the penalty is mild
# and larger chunks are acceptable.
#
# Implementation: estimate a characteristic gold amount (mean of positive golds) and scale it by a
# beta-dependent factor, also damped by alpha (higher alpha -> smaller target).
# This is not a constraint, only a guideline for chunk sizing.
def _auto_soft_target_load(gold: Sequence[float], alpha: float, beta: float) -> float:

    golds = [float(g) for i, g in enumerate(gold) if i != 0 and g > 0]
    if not golds:
        return 0.0
    mean_g = float(statistics.mean(golds))
    if beta <= 1.05:
        scale = 6.0
    elif beta <= 1.5:
        scale = 3.0
    elif beta <= 2.2:
        scale = 1.4
    else:
        scale = 1.0
    scale = scale / max(1.0, float(alpha))
    return max(0.0, mean_g * scale)


def _initial_trips_singletons_full(gold: Sequence[float]) -> List[Trip]:
    trips: List[Trip] = []
    for city in range(1, len(gold)):
        take = float(gold[city])
        if take > 1e-12:
            trips.append([(city, take)])
    return trips


# Ruin operator (trip-level removal).
# Randomly removes `remove_trip_count` whole trips from the current solution.
# Returns a dict `removed` mapping city -> total amount of gold that must be reinserted during RECREATE.
#
# We remove whole trips (instead of individual visits) because it's cheap, keeps bookkeeping simple,
# and guarantees `trips` and `trip_costs` stay aligned after deletions.
def _ruin_remove_trips(
    trips: List[Trip],
    trip_costs: List[float],
    rng: random.Random,
    remove_trip_count: int,
) -> Dict[int, float]:
    if not trips:
        return {}
    remove_trip_count = min(remove_trip_count, max(1, len(trips) - 1))
    idxs = sorted(rng.sample(range(len(trips)), k=remove_trip_count), reverse=True)
    removed: Dict[int, float] = {}
    for idx in idxs:
        for city, take in trips[idx]:
            if take > 0.0 and city != 0:
                removed[city] = float(removed.get(city, 0.0) + float(take))
        del trips[idx]
        del trip_costs[idx]
    return removed


def _merge_consecutive_same_city(trip: Trip) -> None:
    if not trip:
        return
    out: Trip = [trip[0]]
    for city, take in trip[1:]:
        last_city, last_take = out[-1]
        if city == last_city:
            out[-1] = (last_city, float(last_take) + float(take))
        else:
            out.append((city, float(take)))
    trip[:] = out



# Recreate operator: greedy reinsertion by appending to trip tails.
#
# Given `removed[city] = amount` (gold removed during RUIN), we reinsert each city's missing gold
# as one or more chunks (partial pickups). Chunk size is guided by `_auto_soft_target_load`.
#
# For each chunk we choose where to append it:
# - baseline option: start a new trip [ (city, chunk) ]
# - otherwise: sample up to `sample_k` existing trips and estimate the incremental cost of appending
#   this chunk at the end of that trip using `_append_delta_visit`.
# - we select the best option by minimizing (delta_cost / chunk), i.e. cost per unit of gold.
#
# Bookkeeping:
# - `trip_taken[i]` tracks the total gold already assigned to trip i (used as carried-load proxy).
# - After appending we normalize consecutive same-city visits and recompute that trip's exact cost.
def _recreate_append_greedy_sampled(
    trips: List[Trip],
    trip_costs: List[float],
    gold: Sequence[float],
    alpha: float,
    beta: float,
    sp: _ShortestPathCache,
    removed: Dict[int, float],
    rng: random.Random,
    sample_k: int,
) -> None:

    trip_taken = [_trip_taken_sum(t) for t in trips]

    soft_target = _auto_soft_target_load(gold, alpha, beta)

    
    order = sorted(removed.items(), key=lambda kv: kv[1], reverse=True)

    for city, amt_total in order:
        remaining_amt = float(amt_total)
        while remaining_amt > 1e-12:
           
            if soft_target > 0.0:
                chunk = min(remaining_amt, max(soft_target * 0.35, remaining_amt * 0.15))
            else:
                chunk = remaining_amt
            chunk = float(chunk)

            best_idx = -1
            best_score = _append_delta_visit([], 0.0, city, chunk, alpha, beta, sp) / max(1e-12, chunk)

            if trips:
                cand_idxs = rng.sample(range(len(trips)), k=min(sample_k, len(trips)))
                for idx in cand_idxs:
                    d = _append_delta_visit(trips[idx], trip_taken[idx], city, chunk, alpha, beta, sp)
                    score = float(d) / max(1e-12, chunk)
                    if score < best_score:
                        best_score = score
                        best_idx = idx

            if best_idx < 0:
                trips.append([(city, chunk)])
                trip_taken.append(chunk)
                trip_costs.append(_trip_cost(trips[-1], alpha, beta, sp))
            else:
                if trips[best_idx] and trips[best_idx][-1][0] == city:
                    lc, lt = trips[best_idx][-1]
                    trips[best_idx][-1] = (lc, float(lt) + float(chunk))
                else:
                    trips[best_idx].append((city, chunk))
                trip_taken[best_idx] += chunk
                _merge_consecutive_same_city(trips[best_idx])
                trip_costs[best_idx] = _trip_cost(trips[best_idx], alpha, beta, sp)

            remaining_amt -= chunk





# Main LNS solver (multi-visit / partial pickups).
#
# Representation:
# - A solution is a list of trips; each trip is a list of visits (city, take).
# - Each trip is implicitly executed as 0 -> (visits) -> 0.
# - A city may appear multiple times across (and within) trips; the total taken per city is enforced
#   only when building the final route (clipped by remaining gold).
#
# Iteration (Ruin & Recreate):
# 1) RUIN: remove a small random subset of whole trips and aggregate their removed gold in `removed`.
# 2) RECREATE: reinsert `removed` by splitting city amounts into chunks and appending each chunk to
#    the best sampled trip tail (minimizing estimated delta_cost / chunk), creating new trips if needed.
# 3) LOCAL: cheap intra-trip refinement via greedy adjacent swaps.
#
# Acceptance:
# - Greedy: accept the candidate as current only if it improves the current cost; track the best seen.
#
# Parameters:
# - `remove_trip_count` controls ruin aggressiveness.
# - `sample_k` controls how many existing trips are evaluated per chunk during recreate.
def _lns_multivisit(
    g: nx.Graph,
    gold: Sequence[float],
    alpha: float,
    beta: float,
    seed: int,
    max_iters: int | None,
) -> List[Trip]:
    rng = random.Random(seed)
    sp = _ShortestPathCache(g, beta)

    n = g.number_of_nodes()

    
    trips = _initial_trips_singletons_full(gold)
    trip_costs: List[float] = []
    for t in trips:
        city, take = t[0]
        city = int(city)
        take = float(take)
        _, st_out = sp.get(0, city)
        _, st_back = sp.get(city, 0)
        
        trip_costs.append(float(_leg_cost(alpha, beta, st_out, 0.0) + _leg_cost(alpha, beta, st_back, take)))

    best_trips = [t[:] for t in trips]
    best_cost = float(sum(trip_costs))

    current_trips = [t[:] for t in trips]
    current_costs = list(trip_costs)
    current_cost = best_cost

    remove_trip_count = min(6, max(2, len(current_trips) // 16))
    sample_k = 25

    it = 0
    total = max_iters if max_iters is not None else None
    pbar = _tqdm(range(10**18), desc="lns-iters", total=total, leave=False)

    for _ in pbar:
        if max_iters is not None and it >= max_iters:
            break
        it += 1

        cand_trips = [t[:] for t in current_trips]
        cand_costs = list(current_costs)

        removed = _ruin_remove_trips(cand_trips, cand_costs, rng, remove_trip_count)
        _recreate_append_greedy_sampled(cand_trips, cand_costs, gold, alpha, beta, sp, removed, rng, sample_k)

        
        for i, t in enumerate(cand_trips):
            _local_improve_adjacent_swaps_trip(t, alpha, beta, sp)
            cand_costs[i] = _trip_cost(t, alpha, beta, sp)

        cand_cost = float(sum(cand_costs))

        if cand_cost + 1e-12 < current_cost:
            current_trips = cand_trips
            current_costs = cand_costs
            current_cost = cand_cost

        if cand_cost + 1e-12 < best_cost:
            best_trips = [t[:] for t in cand_trips]
            best_cost = cand_cost

    return best_trips


def _build_route_from_trips(trips: Sequence[Sequence[Visit]], gold: Sequence[float], sp: _ShortestPathCache) -> List[Tuple[int, float]]:
    remaining = [float(x) for x in gold]
    remaining[0] = 0.0

    out: List[Tuple[int, float]] = []

    def push(node: int, take: float) -> None:
       
        if out and out[-1][0] == node:
            if node == 0:
                out[-1] = (0, 0.0)
            else:
                out[-1] = (node, float(out[-1][1]) + float(take))
            return
        out.append((node, float(take)))

    push(0, 0.0)

    for trip in trips:
        pos = 0
        for city, take_req in trip:
            take_req = float(take_req)
            if take_req <= 0.0:
                
                if pos != city:
                    path_nodes, _ = sp.get(pos, city)
                    for node in path_nodes[1:]:
                        push(node, 0.0)
                    pos = city
                continue

            if pos != city:
                path_nodes, _ = sp.get(pos, city)
                for node in path_nodes[1:]:
                    push(node, 0.0)
                pos = city

            
            take = min(remaining[city], take_req)
            remaining[city] -= float(take)
            push(city, float(take))

        back_nodes, _ = sp.get(pos, 0)
        for node in back_nodes[1:]:
            push(node, 0.0)

    if not out or out[-1][0] != 0:
        push(0, 0.0)
    else:
        out[-1] = (0, 0.0)
    return out


def solution(p: Problem):
    g = p.graph
    n = g.number_of_nodes()

    gold = [0.0] * n
    for i in range(n):
        gold[i] = float(g.nodes[i].get("gold", 0.0))

    seed = random.randrange(2**31) if LNS_SEED is None else int(LNS_SEED)
    best_trips = _lns_multivisit(g, gold, p.alpha, p.beta, seed=seed, max_iters=MAX_ITERS)

    sp = _ShortestPathCache(g, p.beta)
    route = _build_route_from_trips(best_trips, gold, sp)

    
    _simulate_and_cost(p, route)
    return route


def _simulate_and_cost(p: Problem, route: Sequence[Tuple[int, float]]) -> Tuple[float, Dict[int, float]]:
    g = p.graph
    remaining = {n: float(g.nodes[n].get("gold", 0.0)) for n in g.nodes}
    remaining[0] = 0.0
    carried = 0.0
    total_cost = 0.0

    current = 0
    for idx, (city, take) in enumerate(route):
        if not g.has_edge(current, city) and current != city:
            raise ValueError(f"Route step {idx}: edge ({current},{city}) not in graph")

        if current != city:
            total_cost += p.cost([current, city], carried)
            current = city

        if take < -1e-9:
            raise ValueError(f"Negative gold collected at step {idx}")
        if city != 0:
            can_take = min(remaining[city], float(take))
            remaining[city] -= can_take
            carried += can_take
        else:
            carried = 0.0

    if current != 0:
        raise ValueError("Route does not end at city 0")
    if carried != 0.0:
        raise ValueError("Route ends at base with non-zero carried weight")
    return float(total_cost), remaining


