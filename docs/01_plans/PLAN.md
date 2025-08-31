# GUCA Evolution Lab — Short-Term Plan (Weeks 1–3)

## Week 1 — Core GUM + M1 golden tests ✅ Completed
- [x] #1 Core: Node/Graph/Rules/Machine (parity with C# ops)  
- [x] #2 CLI: run genome YAML and assert stats  
- [x] #3 Golden tests: M1 genomes (dummy_bell → dumb_belly_genom, fractal7genom → fractal7_genom, plus others)  
- [x] #4 Project bootstrap: pyproject + pytest cfg (+ruff/black optional)  

**Deliverables:**  
- Core GUM engine with resettable/continuable transcription modes.  
- CLI `run_gum` executable from YAML.  
- Converter + freezer pipeline from GUCA JSON → YAML.  
- Golden tests passing (`pytest -q`).  

---
## Week 1.5 — 2D visualization of GUM results (new)
- [ ] Simple 2D renderer for final graph snapshot (PNG)
- [ ] CLI flag to save image on demand (no UI)
- [ ] Output-path conventions for tests vs ad-hoc runs

**Deliverables**
- `viz/plot2d.py` with a small `save_graph_png(graph, path, **opts)` function
- Optionally `cli/plot_graph.py` (or `--save-png` in `run_gum`)
- Minimal styling (node color by state, thin edges), layout via NetworkX:
  - default: `spring_layout`; fallback to `kamada_kawai_layout` if tiny graphs
- Works without a display (Agg backend)

**Output conventions**
- Ad-hoc CLI runs: `runs/vis/{genome_stem}/{timestamp}.png`
- Tests/golden: `tests/artifacts/{genome_stem}.png` (overwritten deterministically)
- Batch snapshots (future): `runs/snapshots/{run_id}/step_{k:04d}.png`

**Open choices (we can decide quickly when implementing)**
- Filename scheme for CLI: `{genome_stem}__steps{N}.png`
- Max nodes threshold for labels (e.g., label when V ≤ 50)
- Node size scaling with degree (small range, e.g., 80–200)

---

## Week 2 — Deterministic “connect nearest” & engine config

**Goal.** Remove order-dependent ambiguity from the “connect to nearest” operation and make single-run engine behavior reproducible and configurable.

### Deliverables
- Global engine options (machine-level defaults):
  - `nearest_search.max_depth` (default: **2**)
  - `nearest_search.tie_breaker` ∈ {**stable**, random, by_id, by_creation} (default: **stable**)
  - `nearest_search.connect_all` (boolean, default: **false**)
  - `rng_seed` (optional; use machine/global seed if omitted)
- BFS-based nearest search with **early exit** at minimal distance `d*`.
- Deterministic enqueueing in **stable** mode (sort neighbors by `id`/creation order).
- Optional random tie-break using engine RNG (seeded).
- Unit tests (tie-breakers, multi-connect, max-depth) + one CLI integration example.
- Docs (YAML spec snippet + ADR link).

### Out of scope (now)
- Per-rule overrides (we use **global** engine options only).
- Population-level evolution configs (future milestone).

---

## Week 3 — Fitness v0 + scoring CLI (in progress)
- [ ] Fitness v0: planarity filter + facet proxy + BySample & Mesh heuristics  
- [ ] CLI: score graph/genome with fitness; YAML presets  
- [ ] Tests: fitness smoke + monotonicity sanity  
- [ ] Hydra configs for machine/fitness settings  

**Target Deliverables:**  
- `fitness/planar_basic.py` with basic fitness functions.  
- CLI `score_graph` to evaluate genomes.  
- Example YAML configs (`triangle`, `quad`, `hex`).  

---

## Week 4 — DEAP integration + configs + GA run (planned)
- [ ] Encoding/toolbox/evaluate for GA  
- [ ] Hydra configs for GA population/steps/etc.  
- [ ] GA example script with CSV logs  
- [ ] Tests: GA smoke test, reproducibility with seed  

**Target Deliverables:**  
- Working GA loop with DEAP.  
- Population fitness log, reproducibility with seeds.  
- Config-driven runs (`python -m guca.examples.run_ga`). 