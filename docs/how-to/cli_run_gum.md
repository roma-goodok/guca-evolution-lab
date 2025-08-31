# `guca.cli.run_gum` — run genomes & export PNG visualizations

This CLI runs a GUCA genome (YAML), prints result stats as JSON, and can save a PNG
visualization of the resulting graph.

## Key features (Week 1.5)

- `--save-png` export to `runs/<genome_name>/vis/step<EXEC_STEPS>.png`
- Logs to `runs/<genome_name>/logs/run-YYYYMMDD-HHMMSS.log`
  - includes **graph stats** (nodes/edges/states)
  - **timings** (engine, plot, total)
  - **steps executed**
- Visualization:
  - black background, centered labels on nodes
  - generalized color mapping for **any state** (16‑tone wheel; text color per M1 logic)
  - **auto‑scales** canvas / node size / font / edge width to graph size
  - **render modes**:
    - `full` (default): colored nodes + optional labels
    - `dots`: tiny dots (no labels)
    - `none`: edges only (no nodes)

## Requirements

Base (add to `requirements.txt`):
```
matplotlib>=3.8
networkx>=3.2
```
Optional (faster spring layout / better quality for big graphs):
```
scipy>=1.10
```
If SciPy is not installed, the renderer falls back to SciPy‑free layouts automatically.

## Directory layout

```csharp
runs/
<genome_name>/
vis/
step<EXEC_STEPS>.png
logs/
run-YYYYMMDD-HHMMSS.log
```
> `<EXEC_STEPS>` is the **actual** steps the machine executed (`m.passed_steps`), which
can be less than `--steps` if the machine stopped earlier.

## Usage

### Show CLI help
```bash
python -m guca.cli.run_gum --help
```
Minimal run (no PNG)

```bash
python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_genom.yaml \
  --steps 120
```

Save PNG (default, full nodes)
```bash
python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_genom.yaml \
  --steps 120 \
  --save-png
```

Dense graph (tiny dots, no labels)
```bash
python -m guca.cli.run_gum \
  --genome examples/genomes/fractal7_genom.yaml \
  --steps 120 \
  --save-png \
  --vis-node-render dots \
  --vis-dot-size 10
```

Edges only (no nodes)
```bash
python -m guca.cli.run_gum \
  --genome examples/genomes/fractal7_genom.yaml \
  --steps 120 \
  --save-png \
  --vis-node-render none
```

Custom output base directory
```bash
python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_genom.yaml \
  --steps 120 \
  --save-png \
  --run-dir runs
```

### Notes on visualization
Auto‑scaling: canvas grows with √N; node size / font size / edge width shrink with √N

Labels: automatically disabled when the graph is very large (defaults inside png.py)

Colors:

A 16‑tone wheel is used; text color is white for indices {2,3,5,7,0}, black otherwise

For numeric states we use that number; for string states we map the first letter;
for anything else we use a stable hash

Edge color is a mix of its endpoints’ node colors

### Layouts:

Prefers NetworkX spring_layout (SciPy‑accelerated if available)

Falls back to SciPy‑free layouts (fruchterman_reingold, kamada_kawai, shell, circular)

### Example log (INFO)
```yaml
Copy code
INFO GUM run: genome=dumb_belly_genom steps_param=120 genome_path=... run_dir=runs
INFO GUM result graph stats: {"edges": 13, "nodes": 14, "states_count": {"C": 2, "G": 2, "H": 10}}
INFO Timing: engine=0.003s, plot=0.122s, total=0.145s
INFO Steps executed: 120
INFO Saved PNG: runs/dumb_belly_genom/vis/step120.png
INFO Run log: runs/dumb_belly_genom/logs/run-2025....
```

### Troubleshooting

“No module named 'scipy'” during layout: install SciPy or rely on fallbacks

Overlapping labels on big graphs: try --vis-node-render dots or none

Slow layout on huge graphs: reduce --steps, or use vis-node-render none to check structure quickly

