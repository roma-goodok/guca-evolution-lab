# CLI — score_graph

A small CLI to **generate** a tiny lattice graph and **score** it with one of the fitness functions.

**Module:** `guca.cli.score_graph`

```bash
python -m guca.cli.score_graph --fitness {triangle|quad|hex|by-sample} \
                               --family  {tri|quad|hex} [graph args] [target args] [--debug]
```

## Graph generation options

- `--family tri`  
  - `--faces INT` (default 4) — approximate number of triangular faces  
  - `--tri-kind {block|strip|compact}` (default `block`)

- `--family hex`  
  - `--faces INT` (default 4) — exact number of hex faces for strips/blocks used here  
  - `--hex-kind {block|strip|compact}` (default `block`)

- `--family quad`  
  - `--rows INT` (default 2), `--cols INT` (default 2) — grid faces = rows * cols

> Generators attach a `pos` attribute so the shell/face finder is geometric and stable.

## Fitness selection

- `--fitness triangle` → `TriangleMesh`  
- `--fitness quad`     → `QuadMesh`  
- `--fitness hex`      → `HexMesh` (has a default `target_presence_bonus=1.6`)  
- `--fitness by-sample` → `BySample` (requires **target** spec)

## Target spec (BySample only)

A reference graph is generated using **the same** options, prefixed with `--target-`:

- `--target-family {tri|quad|hex}` (default `tri`)
- `--target-faces INT` (default 6)
- `--target-tri-kind {block|strip|compact}` (default `block`)
- `--target-hex-kind {block|strip|compact}` (default `block`)
- `--target-rows INT` (default 2), `--target-cols INT` (default 2)

## Debug output

Add `--debug` to print a JSON block with:
- `face_lengths` (histogram across **all** faces)
- `shell_len`
- `interior_nodes`
- `nodes`, `edges`

## Examples

**Triangle fitness on a 6‑face tri patch**
```bash
python -m guca.cli.score_graph --fitness triangle --family tri --faces 6
```

**Hex fitness on a hex strip of 10 faces**
```bash
python -m guca.cli.score_graph --fitness hex --family hex --faces 10 --hex-kind strip
```

**BySample: target = quad 2×2; candidate = quad 3×2 (with debug)**
```bash
python -m guca.cli.score_graph --fitness by-sample \
  --family quad --rows 3 --cols 2 \
  --target-family quad --target-rows 2 --target-cols 2 \
  --debug
```

**Inspect faces/shell in debug**
```bash
python -m guca.cli.score_graph --fitness hex --family hex --faces 4 --debug
```

## Rendering PNGs (optional, for sanity checks)

Use `tools/debug_draw_png.py` to render quick previews (shell highlighted):

```bash
# default out: artifacts/mesh_previews
python tools/debug_draw_png.py

# only hex strips to a custom folder
python tools/debug_draw_png.py --families hex --hex-kind strip --out-dir artifacts/hex_strips
```

> PNGs are ignored by git under `/artifacts/**` (see repo .gitignore).
