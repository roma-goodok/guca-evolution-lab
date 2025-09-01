# Fitness (Week 3)

This document describes the **graph fitness** components delivered in Week 3:

- shared scaffolding: `PlanarBasic`
- mesh heuristics: `TriangleMesh`, `QuadMesh`, `HexMesh`
- distribution-based: `BySample`

It also lists **knobs** you can tune during evolution experiments (Week 4), and explains the **formulas** and **terminology** so results are interpretable and replicable.

---

## Glossary

- **Shell / outer face:** the unbounded face of a planar embedding. We choose it by **maximal area** using node positions (`pos`) when available; otherwise we use a planar layout.
- **Interior faces:** all faces *except* the shell. We measure their lengths (3, 4, 6, …).
- **Interior nodes:** nodes not on the shell polygon.
- **Interior edges:** edges **not** on the shell cycle.
- **Planarity:** we require planarity (configurable) for v0 fitness.

---

## PlanarBasic (shared scaffolding)

**Module:** `guca.fitness.planar_basic.PlanarBasic`

Responsibilities:

1) **Viability filter** – cheap early exits (returns a base score if not viable):

   - `one_node` (≤1 node) → `one_node_penalty` *(default 0.00)*  
   - `oversize` (> `max_vertices`) → `oversize_penalty` *(default 0.10)*  
   - `diverged` (meta: `diverged=True` or `steps >= max_steps`) → `diverged_penalty` *(default 0.90)*  
   - `nonplanar` (if `require_planarity=True`) → `nonplanar_penalty` *(default 0.30)*  
   - if viable: `base_score = 1.0`

2) **Face extraction & shell detection**

   - **Geometry-first** face walk if nodes have 2D `pos` (clockwise DCEL traversal).  
   - Otherwise use `networkx` planar embedding traversal.  
   - Canonicalize face cycles (ignore rotation/direction) to deduplicate.  
   - **Shell** = face with the **largest polygon area** (shoelace) in the current coordinates.

3) **EmbeddingInfo** (features for scoring)

   - `faces`: list of node-cycles (including shell)  
   - `shell`: nodes on the outer face  
   - `face_lengths`: histogram of `len(face)` over all faces  
   - `interior_nodes`: `V \ shell`  
   - `interior_degree_hist`: histogram of `deg(v)` for interior nodes

**Config knobs (constructor):**
```python
PlanarBasic(
  max_vertices=2000,
  oversize_penalty=0.10,
  one_node_penalty=0.00,
  nonplanar_penalty=0.30,
  diverged_penalty=0.90,
  take_lcc=True,           # evaluate largest connected component
  require_planarity=True
)
```

---

## Mesh heuristics: Triangle, Quad, Hex

**Module:** `guca.fitness.meshes`  
Classes: `TriangleMesh`, `QuadMesh`, `HexMesh` (all inherit `PlanarBasic`)

### Shared definitions

Let:

- `F_int_total` = number of **interior faces** (exclude shell by identity)  
- `F_int_target` = interior faces whose length equals the target face length (3, 4, or 6)  
- `face_ratio = F_int_target / F_int_total` (0 if `F_int_total = 0`)  
- `shell_pen = |shell| / |V|`  
- `deg_reward = 1 - mean_v_interior( |deg(v) - d_target| / d_target )` (neutral 1.0 if no interior nodes)  
- `non_target_pen = (F_int_total - F_int_target) / F_int_total` (0 if no interior faces)  
- `internal_edge_ratio = |E \ E_shell| / |E|`  
- `size_bonus = min(1, F_int_total / size_cap)`  
- `presence_bonus = target_presence_bonus` iff **any** face (including shell) has the target length  
- `genome_len_bonus = 1 / len(genome)` if enabled and length provided via `meta`

**Scoring formula (all meshes):**
```
score = base
      + w_face      * face_ratio
      + w_deg       * deg_reward
      - w_shell     * shell_pen
      - w_nontarget * non_target_pen
      - forbidden_pen
      + w_internal  * internal_edge_ratio
      + w_size      * size_bonus
      + presence_bonus
      + genome_len_bonus
```

**Forbidden faces (optional):**
```
forbidden_pen = Σ_k  w_forbidden_faces[k] * ( count_int_faces_len_k / F_int_total )
```
> Only **interior** faces are considered here. Use this to **discourage unwanted cycles** (e.g., triangles) when optimizing hex meshes.

### Defaults (can be overridden per family)

```python
MeshWeights(
  w_face=1.0,
  w_deg=0.6,
  w_shell=0.4,
  w_nontarget=0.5,
  w_internal=0.30,
  w_size=0.20,
  size_cap=10,
  target_presence_bonus=0.0,     # overridden in HexMesh default
  w_forbidden_faces={},          # e.g., {3: 0.8} for HexMesh to penalize triangles
  genome_len_bonus=False
)
```

**Family targets:**

- `TriangleMesh`: target face length=3, interior degree≈6  
- `QuadMesh`:     target face length=4, interior degree≈4  
- `HexMesh`:      target face length=6, interior degree≈3

**HexMesh default bias**  
`HexMesh()` sets `target_presence_bonus=1.6` so that even a tiny hex **outranks** large triangle patches (helps steer evolution toward hexes when the population is mixed). Tune in experiments as needed.

**Example (intuition):**  
A strip that adds faces but keeps all nodes on the boundary will have `deg_reward≈1.0` and `face_ratio≈1.0`. The **internal-edge ratio** and **size bonus** provide gentle growth to avoid score plateaus.

---

## BySample (distribution-based)

**Module:** `guca.fitness.by_sample.BySample`

Compares a candidate graph against either:

- a **reference graph** (we compute target distributions from it), or  
- **explicit** target distributions for **interior face lengths** and **degrees**.

**Distributions**

- `face_dist`: normalized counts of interior face lengths, e.g. `{3: 0.8, 4: 0.2}`  
- `degree_dist`: normalized counts of (interior) vertex degrees, e.g. `{6: 0.9, 5: 0.1}`  
- (If no interior nodes, we fallback to all nodes to avoid empties.)

**Similarity metric:**  
`similarity(p, q) = 1 - 0.5 * L1(p, q)` in [0, 1] over the union of keys.  
Optional **Laplace smoothing** avoids zeros on tiny samples.

**Scoring:**
```
score = base
      + w_faces   * face_similarity
      + w_degrees * degree_similarity
      - w_shell   * shell_pen
      + w_internal* internal_edge_ratio
      + w_size    * size_bonus
      + genome_len_bonus
```

**Config:**
```python
BySample(
  reference_graph=G_target,   # or target_face_dist=..., target_degree_dist=...
  weights=BySampleWeights(
    w_faces=1.0, w_degrees=1.0, w_shell=0.4,
    w_internal=0.30, w_size=0.20, size_cap=10,
    smoothing=0.0, genome_len_bonus=False
  ),
  # PlanarBasic kwargs...
)
```

---

## Debug tips

- Use the CLI `--debug` (see `docs/cli_score_graph.md`) to print:
  - `face_lengths` histogram
  - `shell_len`, `interior_nodes`, `|V|`, `|E|`
- Render PNGs with `tools/debug_draw_png.py` (shell edges are highlighted).
- If face counts look off, ensure your graph has `pos`; otherwise we fall back to a planar layout.

---

## Known limitations (v0)

- Face enumeration is **topological**; for highly irregular graphs without positions, the outer face chosen by planar layout might differ from an intuitive geometric shell.
- `size_bonus` saturates at `size_cap` (default 10) to keep values bounded.
- `genome_len_bonus` is off by default; enable only when you want a weak bias toward shorter genomes.

---

## Example knobs for Week‑4 experiments (Hydra-style, illustrative)

```yaml
# configs/fitness/hex.yaml
fitness: hex
weights:
  target_presence_bonus: 1.6
  w_shell: 0.4
  w_internal: 0.30
  w_size: 0.20
  size_cap: 10
  w_forbidden_faces:
    3: 0.8   # penalize triangles during hex optimization

# configs/fitness/triangle.yaml
fitness: triangle
weights:
  w_shell: 0.35
  w_internal: 0.25
  w_size: 0.20
  size_cap: 10
```
