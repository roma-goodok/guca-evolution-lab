# GUCA Evolution Lab

Evolutionary search for **Graph Unfolding Cellular Automata (GUCA)** rules (“change tables”) with compact, interpretable fitness, a simple CLI to run genomes, and a growing base for GA‑driven evolution.

> Current milestone: **M2 — Evolution Lab**. See plans under `docs/01_plans/`.

---

## Overview

- **GUM Machine (Graph Unfolding Machine)** — executes a genome (rules) to grow a graph from an initial seed.
- **Fitness & evaluation** — reference fitness functions to score produced graphs (triangle/quad/hex mesh heuristics, and a distribution‑based “BySample”).
- **Experimentation** — a CLI to run genomes and inspect results; evolution configs and GA operators are planned next.

For details of the machine CLI and genome format, see the links in the **Documentation** section below.

---

## Install

Python **3.10+** recommended.

```bash
pip install -r requirements.txt
# (dev) tests & previews
pip install -r requirements-dev.txt
```

---

## Quick Start

### 1) Run a genome with the GUM machine

Use the CLI to execute a genome YAML, print a short JSON summary, and (optionally) save a PNG visualization.

```bash
# Show help
python -m guca.cli.run_gum --help

# Minimal run (no PNG)
python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_genom.yaml \
  --steps 120

# Save a PNG next to run logs
python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_genom.yaml \
  --steps 120 \
  --save-png
```

**Default output layout**

```
runs/
  <genome_name>/
    vis/   step<EXEC_STEPS>.png
    logs/  run-YYYYMMDD-HHMMSS.log
```

> `<EXEC_STEPS>` is the actual number of steps the machine executed; it can be less than `--steps` if the run stops earlier.

### 2) Genome YAML at a glance

A single‑run genome YAML has three top‑level sections:

```yaml
machine:
  max_steps: 120
  max_vertices: 2000
  start_state: A
  rng_seed: 42
  nearest_search:
    max_depth: 2
    tie_breaker: stable
    connect_all: false

init_graph:
  nodes:
    - { id: 0, state: A }
    - { id: 1, state: B }
  edges:
    - [0, 1]

rules:
  - condition: { current: A }
    op: { kind: TryToConnectWithNearest, operand: B }
```

- If `init_graph.nodes` is missing or empty, the engine starts from a single node with `machine.start_state` (default "A").
- `nearest_search` controls BFS depth and deterministic tie‑breaking for nearest‑neighbor connect operations.

---

## Documentation

- **How to run the GUM machine (CLI):** `docs/how-to/cli_run_gum.md`
- **Genome YAML spec (single‑run):** `docs/reference/genome_yaml_spec.md`
- **Fitness functions (reference):** `docs/reference/fitness.md`
- **Plans & vision:**  
  - `docs/01_plans/PLAN.md`  
  - `docs/01_plans/vision_and_high_level_plan.md`  
  - Milestone M2 plan: `docs/01_plans/milestones/m2_evolution_lab/04_milestone_M2_plan.md`
- **Architecture decisions (ADRs):** `docs/02_adr/`

---

## Repository Layout

```
guca-evolution-lab/
├─ src/                  # GUCA core, fitness, CLI
├─ tests/                # pytest suite
├─ tools/                # development scripts
├─ docs/                 # plans, ADRs, how‑tos, references
│  ├─ 01_plans/
│  ├─ 02_adr/
│  ├─ how-to/
│  └─ reference/
├─ artifacts/            # generated previews (git‑ignored)
├─ configs/              # (planned) GA/experiment configs
└─ README.md
```

---

## Next Steps

- **GA integration (DEAP):** variable‑length chromosome (64‑bit gene), crossover and mutation operators, structural edits (insert/duplicate/delete), active/passive regimes.
- **Experiment runner & configs:** Hydra‑based experiments, reproducibility tests, logging.

---

## Contributing

- Create a feature branch (`feat/<topic>`), add tests, and open a PR.
- Generated images should live under `artifacts/` (git‑ignored).
