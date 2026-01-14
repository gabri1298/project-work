# Final project — Gold Thief Problem solver (LNS multi-visit) 💰 — s347915

This project implements a **Large Neighborhood Search (LNS)** heuristic for the *Gold Thief Problem*: collect gold on a weighted graph with a **depot** at node `0`.

The solver returns a **route** (sequence of steps) in the simulator format:

```text
[(city_0, take_0), (city_1, take_1), ...]
```

where:
- `city` is the visited node;
- `take` is the amount of gold collected at that step (can be `0`).

The *multi-visit* variant allows:
- visiting the same city multiple times;
- **partial pickups** (a city's gold can be split into multiple “chunks”).

The main algorithm is in [s347915.py](s347915.py) (entry-point: `solution(p)`).


## Cost model (operational view)

The cost of traveling grows with the carried load. For a leg that follows a shortest path made of edges `e` with distance `d_e`, the implementation uses an edge-aggregated decomposition:

```text
cost(leg, carried) = dist_sum + (alpha * carried)^beta * dist_beta_sum
```

where:
- `dist_sum = Σ_e d_e`
- `dist_beta_sum = Σ_e (d_e^beta)`

This makes leg evaluation fast for a given `carried` without re-expanding the path edge-by-edge every time (the path is reconstructed only to compute `dist_beta_sum`).


## Key constraint: no “passing through 0” between non-zero cities

In the problem model, visiting the depot `0` implies **unloading** (carried load resets to zero). To avoid invalid shortcuts:

- for any pair `u != 0` and `v != 0`, the shortest path used **must not traverse node `0`**.

In practice:
- all shortest paths/distances between non-zero nodes are computed on the subgraph that excludes node `0`.


## Shortest path cache (SciPy + predecessors)

For efficiency, shortest paths are precomputed with `scipy.sparse.csgraph.shortest_path` on sparse matrices from `nx.to_scipy_sparse_array`:

1) **All-pairs on the graph without 0** (for all pairs `u,v != 0`):
	 - stores distance matrix `dist_no0[u,v]` and predecessors `pred_no0[u,v]`.

2) **Single-source from node 0** on the full graph (for depot legs `0 ↔ v`):
	 - stores `dist0_all[v]` and predecessors `pred0_all`.
	 - the graph is treated as undirected (`directed=False`), so `v -> 0` is the reverse of `0 -> v`.

Reconstruction:
- given a predecessor array/matrix, the cache reconstructs the node sequence of the path and computes `dist_beta_sum` by summing `d_e^beta` along the edges of the path.

Negative weights:
- if any edge has `dist < 0`, SciPy is forced to use `method="BF"` (Bellman–Ford); otherwise `method="auto"` is used.


## Internal solution representation

The LNS operates on a list of **trips**:

- `Visit = (city: int, take: float)`
- `Trip = List[Visit]`
- `trips = List[Trip]`

Each trip is implicitly executed as:

```text
0 -> (visits) -> 0
```

The carried load `carried` within a trip is the cumulative sum of `take` collected earlier in that trip.


## Initialization

The initial state is a simple baseline:

- one trip per city with positive gold;
- collect the entire city's gold in a single visit.

This keeps startup cheap and lets LNS do the splitting (chunking) and rearrangement.


## LNS (Ruin & Recreate)

For `MAX_ITERS` iterations:

1) **Ruin (removal)**
	 - randomly removes a small number of whole trips.
	 - accumulates removed gold into a dictionary `removed[city] += take`.

2) **Recreate (reinsertion with chunking)**
	 - for each `city` in `removed`, reinserts gold in multiple steps by choosing a `chunk` size.
	 - `chunk` depends on:
		 - `remaining_amt` (gold still to reinsert for that city);
		 - a soft target `soft_target` estimated by `_auto_soft_target_load(gold, alpha, beta)`.

	 Assigning a chunk to a trip:
	 - the chunk is **appended** to the end of the trip that minimizes estimated incremental cost per unit of gold:
		 - evaluate `_append_delta_visit(trip, trip_taken_sum, city, chunk, ...) / chunk` on a sample of trips.
	 - if no existing trip is better than starting fresh, create a new trip `[(city, chunk)]`.

3) **Local refinement (intra-trip)**
	 - each trip is refined with a tiny local search based on **adjacent swaps** to reduce trip cost.

Acceptance:
- current version is greedy: accept the candidate only if it improves the current cost; keep a separate global best.


## Building the final route

Once the best `trips` are found, the final route is expanded:

- for each visit `(city, take_req)`:
	- move along the shortest path (inserting intermediate nodes with `take=0`);
	- collect `take = min(remaining_gold[city], take_req)`.
- between trips the route always returns to depot `0`.

Finally, `_simulate_and_cost(p, route)` is run as a validation step.


## Tunable parameters

Main constants (top of [s347915.py](s347915.py)):

- `SHOW_PROGRESS`: enable/disable the progress bar (`tqdm`)
- `MAX_ITERS`: maximum number of LNS iterations
- `LNS_SEED`: if `None`, uses a random seed; if set, makes runs reproducible

Other parameters (in code):

- `remove_trip_count`: ruin aggressiveness (how many trips to remove)
- `sample_k`: how many existing trips are evaluated per chunk during recreate


## Practical notes

- Chunking is decided **per city** during recreate; trips influence it indirectly because the delta cost depends on the carried load already present in the trip.
- For `beta > 1`, splitting gold into chunks and distributing them across trips often reduces cost because it keeps carried load lower for part of the travel.



