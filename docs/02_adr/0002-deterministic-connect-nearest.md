
# ADR 0002: Deterministic "connect to nearest"

- **Status:** Accepted
- **Date:** 2025-08-31
- **Owners:** @roma-goodok

## Context
The operation `TryToConnectWithNearest` previously used a “first-found” BFS choice, which made results depend on node enumeration order and implementation details (Python dict order, iteration patterns). This led to ambiguity across environments and runs. The operation must be reproducible and configurable.

## Decision
We implement nearest connection as:

1. **BFS** from the current node up to a configurable `max_depth` (default 2).
2. Let `d*` be the first depth with **eligible** nodes (not the start, not already adjacent, not marked new/deleted, and when the rule has an **operand**, match by **saved** state; fall back to current state if saved is absent).
3. If `connect_all: true` → connect to **all** nodes at depth `d*`.  
   Otherwise choose **one** via `tie_breaker`:
   - `stable` / `by_id`: deterministic ascending **id** order (reproducible)
   - `by_creation`: creation order (aliases to id for now)
   - `random`: uniform random using the engine RNG
4. Randomness is driven by a machine‑level `rng_seed` for reproducibility.

We add global engine options under `machine.nearest_search`:
- `max_depth` (int ≥1, default 2)
- `tie_breaker` (`stable` | `random` | `by_id` | `by_creation`; default `stable`)
- `connect_all` (bool, default `false`)

We also add `machine.rng_seed`.

We retain the existing rule operand semantics: when present, the operand filters by **saved** state (fallback to current state).

## Consequences
- **Pros**
  - Deterministic and reproducible runs (`stable` and `by_id`).
  - Configurable selection behavior, including **connect all**.
  - Backward‑compatible with operand semantics (uses saved state).
- **Cons**
  - Slight overhead from BFS and sorting per step (bounded by `max_depth`).
  - More config surface to document and test.

## Alternatives considered
- Global randomized node order (still non‑deterministic without a seed).
- Geometry‑based nearest (requires coordinates; not always available).
- Dijkstra/A* weighting (unnecessary complexity; BFS adequate here).
- Per‑rule overrides (deferred; engine‑level options suffice now).

## Rollout
- Implemented in `guca/core/graph.py` (BFS + tie‑breakers) and called from `guca/core/machine.py`.
- YAML schema extended (`machine.nearest_search`, `machine.rng_seed`, `init_graph`).
- Tests cover depth, tie‑breakers, `connect_all`, and seeded randomness.
- Docs updated (this ADR + `docs/reference/genome_yaml_spec.md`).
