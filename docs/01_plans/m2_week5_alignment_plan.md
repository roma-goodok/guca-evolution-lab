# M2 — Week 5 GA Alignment Plan (C# Parity) — **UPDATED**

**Goal:** Align the Python GA with the legacy C# behavior so evolution reliably grows mesh structures
(starting with triangles, later quads & hex). This document captures the **key decisions**, **tasks**, **test
coverage**, and **current status**.

_Last update: 2025‑09‑05._

---

## ✅ Key decisions

- **Gene ↔ Rule parity:** keep our 64‑bit gene layout compatible with the C# `GumGen` fields (status, prior,
  conn_GE/LE, parents_GE/LE, op.kind, operand). Best checkpoints **export full condition** in YAML.
- **Crossover parity:** default to **tail‑swap** (one cut per parent, swap tails), clamped by `max_len`.
- **Mutation split (active vs passive):** use per‑gene **WasActive** mask from simulator. Active and passive
  regimes use different mutation kinds + probabilities; each gets a small rotate (“shift”) chance.
- **Selection menu:** support **elite**, **rank**, **roulette**; optional random injection. Hydra‑configurable.
- **Fitness shaping (TriangleMesh focus):** disconnected graphs → ~0; trees live on a low, slightly rising
  plateau; **any cycle** strictly beats any tree. Mesh‑specific metrics kept (faces/degree/shell/etc.).
- **Checkpoints & UX:** `best.yaml` has full rule condition and a **graph summary**; optional `best.png`
  rendered with the same code path as `run_gum` (edges + colored nodes unless configured otherwise).
- **Hydra integration:** `_target_` instantiation for fitness and GA; experiment naming & run dirs via env‑based
  logbook path; parallel evaluation via `n_workers`.

---

## 📋 Task list & status

### 1) Gene & rule parity
- [x] Ensure encoder/decoder map the 64‑bit fields 1:1 with C#.
- [x] Export **full** `condition` in `best.yaml` (including prior/conn/parents bounds).
- [x] Keep human‑readable operand/state labels.  

**Tests:** covered via checkpoint content checks.

---

### 2) Simulator: activity tracking
- [x] While applying rules, mark the firing gene(s) as **WasActive** for that evaluation.
- [x] Surface the **activity mask** to the GA loop (stored on the individual for mutation).

**Tests:** `test_simulate_sets_mask.py` & `test_activity_mask_influence.py`.

---

### 3) Mutation: active vs passive (+ structural extras)
- [x] Split by **WasActive**: `active_cfg` vs `passive_cfg` (factor, kind, optional rotate/shift).
- [x] Structural extras: **duplicate‑head**, **insert‑near‑active**, **delete‑inactive** (guarded by length).
- [x] Keep legacy low‑level ops: bitflip / byte / allbytes / rotate; add **enum‑delta** safety.
- [x] Post‑mutate safety: clamp to 64‑bit; if gene becomes undecodable → **re‑randomize**.

**Tests:** `test_operators.py::test_mutate_fn_keeps_domain_and_bounds`, activity‑guided mutation test.

---

### 4) Crossover parity
- [x] Tail‑swap variant implemented and selectable; used by default.
- [x] Length clamped by `max_len` for both children; preserves list type.

**Tests:** child shape & length checks against a reference construction.

---

### 5) Selection strategies
- [x] Hydra knob: `ga.selection.method: elite|rank|roulette` + `ga.selection.random_ratio`.
- [x] Rank/roulette implemented with reproducible RNG; elite keeps top‑K; tournament kept for back‑compat.

**Tests:** distribution sanity on synthetic fitness arrays; deterministic behavior under fixed seeds.

---

### 6) Fitness (TriangleMesh tune)
- [x] Connectivity & cycle **gates**: disconnected → ~0; small forests very low; trees form a mild plateau;
      first cycle jumps above the plateau.
- [x] Triangle mesh signals: interior triangular faces, interior degree≈6, shell compactness, size bonus
      (saturating), and forbidden‑faces penalties (configurable).
- [x] Trees: allow **slight growth** with size (but always below any cyclic graph).

**Tests:** `tests/test_fitness_meshes.py` and sanity script trends.

---

### 7) Checkpoints & run UX
- [x] `best.yaml` or `best.json` (per `fmt`) + `last.json` + `pop_*.jsonl`.
- [x] **Graph summary** embedded (nodes/edges/states).  
- [x] `save_best_png: true` renders `checkpoints/best.png` using the **run_gum** renderer.
- [x] TQDM progress bar; dynamic postfix: **max**, **avg**, **len**.
- [x] Start banner (experiment name in green, logbook path); end prints best graph summary.

**Tests:** smoke test `test_toolbox_smoke.py` verifies files & fields existence.

---

### 8) Hydra config
- [x] `_target_` for fitness and GA experiment classes.
- [x] `ga.selection.*`, `ga.cx.variant`, `ga.mutation.active/*` & `passive/*`, `ga.checkpoint.save_best_png` added.
- [x] Logbook path via `${oc.env:ML_LOGBOOK_DIR}/…` and experiment name; run dirs under Hydra’s `${now:…}`.

**Config snippet (illustrative):**
```yaml
ga:
  selection:
    method: rank      # elite | rank | roulette
    random_ratio: 0.0
  cx:
    variant: tail_swap  # tail_swap | splice
  mutation:
    active:  { factor: 0.10, kind: byte,     rotate_factor: 0.02 }
    passive: { factor: 0.50, kind: all_bytes, rotate_factor: 0.10 }
  checkpoint:
    save_best: true
    save_last: true
    save_every: 5
    save_population: best  # best | all
    fmt: yaml              # yaml | json
    save_best_png: true
    out_dir: checkpoints
```

---

### 9) Parallel evaluation
- [x] `n_workers > 0` enables a process pool for **fitness evaluation only** (safe & deterministic).

**Tests:** manual run sanity under `n_workers=8` + smoke test still green.

---

## 🧪 Test inventory (Week 4)

- `tests/ga/test_operators.py` — field/structural mutation bounds & safety.
- `tests/ga/test_simulate_sets_mask.py` — simulator returns activity mask.
- `tests/ga/test_activity_mask_influence.py` — active vs passive mutation bias.
- `tests/ga/test_toolbox_smoke.py` — GA loop runs; checkpoints, PNG/summary written when enabled.
- `tests/test_fitness_meshes.py` — triangle mesh monotonicity and forbidden‑faces penalties.
- (Sanity scripts under `tools/` for quick manual trend checks.)

_All tests pass under fixed RNG seeds._

---

## ⚠️ Remaining items / next wave (Week 5+)

- [x] **Full condition semantics in the engine:** make all predicate fields effective
  (prior state, conn_GE/LE, parents_GE/LE) in rule matching during simulation.
- [ ] TriangleMesh fitness alignment against the C# outputs (fine‑tuning weights where needed once rules are fully expressive).
- [ ] Integrate activity mask from engine into the GA loop after each eval (we can set ind.active_mask using the mask returned by simulate_genome(..., collect_activity=True) to bias the next generation automatically).
- [ ] **Quad/Hex fitness tuning:** revisit weights and forbidden‑faces for stronger signal vs triangles.
- **Experiment sweeps:** Hydra‑based grids for (selection × cx × mutation) sensitivity; export plots.
- **Optional:** record per‑generation best PNGs (`epoch_*.png`) when `save_every > 0` for visual progress.

---

## 📎 Notes & references

- Legacy reference: `docs/xx_legacy/` (C# `GUMChromosome`, `PlanarGraphFitnessFunction*`).
- Current CLI & renderer: `guca.cli.run_gum` (same visuals used by GA PNG checkpoints).
- YAML spec: `docs/reference/genome_yaml_spec.md`.
