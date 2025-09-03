# M2 — Week 4 GA alignment plan (with C# reference)

**Goal:** align our Python GA to the successful behavior observed in the legacy C# implementation so that the search can reliably grow mesh structures (triangles first, later quads & hex).

This document records **key decisions**, **implementation steps**, **tests**, and **acceptance criteria** for Week 4.

---

## 1) Scope & decisions

- **Gene format / rule shape**
  - Adopt the full 64‑bit gene layout from the C# code (status, prior, conn\_GE/LE, parents\_GE/LE, op.kind, op.operand), and expose the **full condition** in YAML for best checkpoints.
  - Keep our `Rule` structure but ensure it maps 1:1 to the C# fields.

- **Crossover**
  - Use **one‑cut per parent** (tail swap) like C#: pick `cp1` in parent A and `cp2` in parent B → child1 = `A[:cp1] + B[cp2:]`, child2 = `B[:cp2] + A[cp1:]`. Clamp by `max_len`. (Matches `GUMChromosome.Crossover`.)

- **Mutation (active vs passive)**
  - Implement **conditional mutation** driven by per‑gene activity flags (WasActive) collected during simulation:
    - For **active genes**: mutate with `active_gen_mutation_factor` using `active_gen_mutation_kind`.
    - For **passive genes**: mutate with `passive_gen_mutation_factor` using `passive_gen_mutation_kind`.
    - Also include a small **byte‑rotate** (shift) chance for both, as in C# (×0.2 of the factor).
  - Add **structural tweaks** echoing the legacy:
    - With small probability, **duplicate the first gene** (if below `max_len`).
    - With small probability, **insert near a random active gene** (if any and below `max_len`).
    - With small probability, **delete a random inactive gene** (if long enough).

- **Selection**
  - Provide **ELITE / RANK / ROULETTE** strategies selectable via Hydra (`ga.selection.method`), plus `ga.selection.random_ratio` for injecting random picks — just like the UI. Default: **ELITE** + **tournament** for survivors, but allow swapping to **RANK** or **ROULETTE** easily.

- **Fitness (Week 4 focus = TriangleMesh)**
  - Keep our new **connectivity/cycle gating**: disconnected ⇒ ~0, trees ⇒ low plateau (~1.0..1.1), any cycle ⇒ jump above tree plateau.
  - Ensure **triangle signal** dominates triangle‑like graphs: reward interior triangular faces, correct shell pressure, degree target (=6) for interior.
  - Preserve **hex/quad knobs** but keep them out of scope for Week 4 fine‑tuning.

- **Checkpoints & run UX**
  - `best.yaml` includes **full rule condition** and a **graph summary**.
  - `best.png` is rendered in‑process with the same renderer as `run_gum` (edges + colored nodes, controllable via knob).
  - Progress bar shows **max, avg, len**. At the end print **best graph summary** (nodes/edges/state counts).


> C# reference for crossover/mutation/fitness filters: `GUMChromosome` and `PlanarGraphFitnessFunction` family. See provided legacy sources in `docs/xx_legacy`.


---

## 2) Implementation steps (tasks)

1. **Gene & rule parity**
   - [ ] Confirm decoder/encoder map all 64‑bit fields (status, prior, conn\_GE/LE, parents\_GE/LE, op.kind, operand).
   - [ ] When writing `best.yaml`, include the full condition block (already supported).

2. **Simulator: activity tracking**
   - [ ] While applying rules, **record which rule fired** for each step; mark those genes as **WasActive** for the evaluated genome.
   - [ ] Return this activity mask to the GA loop (store on the individual temporarily during evaluation).

3. **Mutation: active vs passive**
   - [ ] In `operators.make_mutate_fn`, branch mutation parameters by **WasActive** flag.
   - [ ] Add byte‑**rotate** sub‑mutation (×0.2 of factor) for both active & passive (per legacy).
   - [ ] Implement structural tweaks: **duplicate‑first**, **insert‑near‑active**, **delete‑inactive** with small probabilities and length guards.

4. **Crossover parity**
   - [ ] Ensure our `splice_cx` behaves as **one‑cut per parent** tail swap, or add `legacy_tail_swap_cx` and switch to it by default.

5. **Selection menu**
   - [ ] Add `ga.selection.method: elite|rank|roulette` + `ga.selection.random_ratio`.
   - [ ] Wire **tournament** to be used only if `method=elite` (or keep as internal survivor selector), and support alternative selectors via DEAP stock ops.

6. **Fitness (TriangleMesh tune)**
   - [ ] Keep connectivity/cycle gates; align triangle weights to prefer **interior triangles** and **degree=6**.
   - [ ] Make sure trees can **grow slightly with size**, but **any cycle** still beats any tree.

7. **Checkpoints & UX polish**
   - [ ] Write `best.yaml` (or `.json`) + `last.json` + `pop_*.jsonl`; include **graph summary**.
   - [ ] If `save_best_png: true`, write `checkpoints/best.png` (edges + colored nodes by default).

8. **Hydra config**
   - [ ] Add new keys: `ga.selection.method`, `ga.selection.random_ratio`, `ga.mutation.active.*`, `ga.mutation.passive.*`, switches for CX variant, and renderer flags.
   - [ ] Keep experiment block: `experiment.name`, `description`, `inherits` with `${oc.env:ML_LOGBOOK_DIR}` pathing.

9. **Parallel evaluation**
   - [ ] (Already supported) Confirm pool is applied to **fitness eval** only; mutation/crossover remain in‑process.


---

## 3) Tests (pytest)

- **Crossover parity**
  - [ ] With two known parents, assert children equal legacy tail‑swap construction; lengths respect `max_len`.
- **Active vs passive mutation**
  - [ ] Craft a genome where some rules certainly fire; check higher mutation rate / different kind for **active** vs **passive** slices.
- **Structural tweaks**
  - [ ] Probabilistic tests with seeded RNG: when probabilities set to 1.0, duplicate/insert/delete effects occur as specified.
- **Selection strategies**
  - [ ] Rank and roulette produce distributions consistent with expectations on synthetic fitness arrays; `random_ratio` injects uniform randomness.
- **Triangle fitness trend**
  - [ ] Disconnected ⇒ ~0; single edge ⇒ low (~1.0); larger trees slightly higher; any single cycle ⇒ strictly greater than any tree (sanity scripts already cover this trend).
- **Checkpoint content**
  - [ ] `best.yaml` contains full condition; `best.png` exists when knob is on; both include graph summary.


---

## 4) Risks & caveats

- **Saved vs current state** semantics in neighbor‑search: keep using **saved state** to match legacy behavior for `TryToConnectWith` and “nearest” rules.
- **BFS tie‑breakers**: ensure deterministic `stable` order to improve reproducibility.
- **Plateaus**: even with better fitness shaping, evolution may stall if initial genomes can only create trees; keeping cycle bonus and hex/quad penalties consistent helps escape.
- **Performance**: activity tracking and PNG rendering are off the hot path; parallel eval mitigates added cost.


---

## 5) Effort estimate

Assuming ~2–3 h/day on weekdays & 6–10 h on weekend:

- Core parity (decoder/encoder, activity tracking, mutation split, CX parity): **1.5–2 days**
- Selection strategies & Hydra wiring: **0.5 day**
- Fitness tune (triangle) & tests: **0.5–1 day**
- Checkpoints/UX polish & docs: **0.5 day**

**Total:** ~**3–4 days** of focused work.

---

## 6) Hydra knobs (sketch)

```yaml
ga:
  selection:
    method: elite   # elite | rank | roulette
    random_ratio: 0.0
  cx:
    variant: tail_swap   # tail_swap | splice
  mutation:
    active:
      factor: 0.10
      kind: byte        # bit | byte | all_bytes
      rotate_factor: 0.02  # optional (applied as extra rotate chance)
    passive:
      factor: 0.50
      kind: all_bytes
      rotate_factor: 0.10
```

---

## 7) References

- Legacy crossover/mutation and fitness gate logic — **C#**: `GUMChromosome` & `PlanarGraphFitnessFunction` family (docs/xx_legacy).  
- Current Hydra config & CLI docs — `config.yaml`, `docs/how-to/cli_run_gum.md`, `docs/reference/genome_yaml_spec.md`.
