# Genome YAML Spec (single‑run)

This document defines the **single‑run** YAML that `guca.cli.run_gum` consumes.

Top‑level keys:

- `machine` — engine defaults & limits
- `init_graph` — optional seed graph (nodes/edges)
- `rules` — GUM rules table

> **Compatibility:** If `init_graph.nodes` is **present and non‑empty**, `start_state` is used **only as a fallback** for nodes that omit `state`. If `init_graph.nodes` is **empty/missing**, the engine starts from **one node** with `start_state` (default `"A"`).

---

## `machine` (engine defaults)

```yaml
machine:
  max_steps: 120                # int ≥1
  max_vertices: 2000            # 0 means "unlimited"
  start_state: A                # fallback only; see note above
  transcription: resettable     # implementation-defined
  count_compare: range          # implementation-defined

  # Reproducibility
  rng_seed: 42                  # optional, int

  # Nearest-connection behavior (global)
  nearest_search:
    max_depth: 2                # int ≥1 (default 2)
    tie_breaker: stable         # stable | random | by_id | by_creation (default stable)
    connect_all: false          # false => pick 1; true => connect to all at minimal distance
```

### Nearest semantics

When a rule with `op.kind: TryToConnectWithNearest` fires for node `u`:

1. Run **BFS** from `u` up to `max_depth`.
2. Let `d*` be the **smallest** depth at which **eligible** nodes appear.
3. **Eligible** means: not `u`, not already adjacent to `u`, not `marked_new/marked_deleted`, and (if the rule has an **operand**) the node’s **saved state** equals that operand. If saved state is absent, fall back to current `state`.
4. If `connect_all: true` → connect `u` to **all** eligible nodes at depth `d*`.  
   Otherwise → pick **one** using `tie_breaker`:
   - `stable` / `by_id`: deterministic ascending **id** order (reproducible across runs)
   - `by_creation`: creation order (aliases to id unless a separate creation index is added)
   - `random`: uniform at random from candidates, using the engine **RNG** (seeded by `rng_seed`)

---

## `init_graph` (optional seed graph)

```yaml
init_graph:
  nodes:
    - { id: 0, state: A }
    - { id: 1, state: B }   # if "state" omitted, falls back to machine.start_state
  edges:
    - [0, 1]                # undirected edge between node ids 0 and 1
```

- If `id` is omitted, ids are assigned in **listed order** (0..N-1).
- Edges reference **these ids**.
- If `nodes` is missing/empty, the engine creates **one node** with `start_state`.

---

## `rules` (GUM rules table)

Each rule has:

```yaml
- condition:
    current: A          # match current node state "A"
  op:
    kind: <Operation>   # e.g., TurnToState / GiveBirth / GiveBirthConnected /
                        #       TryToConnectWith / TryToConnectWithNearest / DisconnectFrom / Die
    operand: C          # optional; meaning depends on the operation
```

Common operations:

- `TurnToState` (operand = target state)
- `GiveBirth` (operand = state of new node)
- `GiveBirthConnected` (operand = state of new node; also connects parent→child)
- `TryToConnectWith` (operand = state; connect to **all** nodes with that **saved** state)
- `TryToConnectWithNearest` (operand = state; nearest search per **Nearest semantics** above)
- `DisconnectFrom` (operand = state; remove edges to nodes with that **saved** state)
- `Die` (mark current node for deletion)

---

## Minimal example

```yaml
machine:
  max_steps: 1
  max_vertices: 100
  rng_seed: 42
  nearest_search:
    max_depth: 2
    tie_breaker: stable
    connect_all: false

init_graph:
  nodes:
    - { id: 0, state: A }
    - { id: 1, state: B }
    - { id: 2, state: B }
    - { id: 3, state: C }
    - { id: 4, state: C }
  edges:
    - [0, 1]
    - [0, 2]
    - [1, 3]
    - [2, 4]

rules:
  - condition: { current: A }
    op: { kind: TryToConnectWithNearest, operand: C }
```

**Effect:** Node `0` connects to the **nearest** `C` nodes at minimal distance. With `connect_all: false` and `stable`, it deterministically picks the **lowest id** among the candidates at that depth.
