# GUCA Evolution Lab — Codebook & Working Agreements

> Living guide for contributors (humans & AIs) on how we plan, track, code, and review in **guca-evolution-lab**.

---

## 0) Purpose & Scope
- Keep a **single, clear way** to run the project day‑to‑day.
- Capture our **time cadence** and convert it into practical workflows.
- Standardize **issues, labels, branches, PRs, and tests**.
- Add friendly **AI-collaboration rules** (what context to supply, what to expect).

---

## 1) Planning Artifacts — when to use what

### Single Source of Truth
- **`docs/PLAN.md`** — current short‑term plan (weeks 1–3 or similar).
  - Narrative + checklists.
  - Links out to Issues (`#123`) and back from Issues to this plan.

### Execution
- **GitHub Issues** — small, actionable tasks that can be closed.
  - One clear **Goal**, **Deliverables**, **Acceptance criteria**, **Timebox**.
- **GitHub Project** — live board to visualize status by week and focus.
  - Status: `Todo → Doing → PR → Done`.
  - Fields: `Week (W1/W2/W3)`, `Type (core/fitness/evo/cli/tests/infra)`, `Focus (weekday-1h | weekend-focus)`.

### Cadence (matches owner’s time budget)
- **Weekdays:** 1–2 hours per day → pick **one** Issue labeled `weekday-1h`.
- **Weekend:** 6–10 hours for focused activity → 1–2 broader Issues labeled `weekend-focus`.

**Rule of thumb**
- Change the plan? Update `docs/PLAN.md` and add/close Issues accordingly.
- Big ideas / decisions → record an ADR (see §7).

---

## 2) Creating a good Issue (step-by-step)
1. **Title:** imperative, 6–10 words (e.g., “Implement ChangeTable with continuable scan”).
2. **Template:** use the Task template below (Goal / Deliverables / Acceptance / Timebox / Links).
3. **Labels:** choose from
   - Domain: `core`, `fitness`, `evo`, `cli`, `tests`, `infra`, `docs`.
   - Focus: `weekday-1h`, `weekend-focus`.
   - Size (optional): `size-S`, `size-M`, `size-L`.
4. **Milestone:** Week N (dates), so the board groups correctly.
5. **Cross-link:** add references to `docs/PLAN.md` section or ADR.
6. **Exit condition:** ensure the Acceptance criteria are **objective** and **executable**.

### Issue Template (copy into description)
```
**Goal**
One-sentence outcome (user-value).

**Deliverables**
- [ ] code / files / CLI / docs updated

**Acceptance criteria**
- [ ] deterministic tests pass (list)
- [ ] CLI example runs (command)
- [ ] matches spec / parity notes (link)

**Timebox**
weekday-1h | weekend-focus

**Links**
Plan: docs/PLAN.md (Week N, section …)
Refs: (M1 genome, C# file, ADR, etc.)
```

---

## 3) Labels, Milestones, and Project Board
- **Labels**
  - Domain: `core`, `fitness`, `evo`, `cli`, `tests`, `infra`, `docs`.
  - Focus: `weekday-1h`, `weekend-focus`.
  - Size (opt): `size-S/M/L`.
  - Priority (opt): `P1`, `P2`.
- **Milestones**
  - `Week 1`, `Week 2`, `Week 3` with date ranges.
- **Project**
  - View: table or board; auto-add new issues.
  - Automations: move to `PR` when a PR opens, to `Done` when merged.

---

## 4) Branches, Commits, and PRs

### Branch naming
- `feature/<short-topic>` (e.g., `feature/core-gum-machine`)
- `fix/<short-topic>`
- `docs/<short-topic>`

Optionally append `-#123` to tie to an issue.

### Conventional commits (recommended)
- `feat(core): add ChangeTable resettable & continuable`
- `fix(fitness): handle empty graph in planarity filter`
- `test(cli): golden run for dummy_bell`
- `docs(plan): add W2 fitness checklist`

### Pull Requests
- Keep small and reviewable (≤ ~400 lines diff ideally).
- Description should include:
  - **What & Why** (1–2 paragraphs)
  - **How** (bullet list of key changes)
  - **Testing** (commands, test names)
  - **Links** (`Closes #123`, Plan section)
- Merge strategy: **Squash & merge** (default), to keep main history clean.
- CI must pass; at least one review approval (self-review allowed for solo work, but still run CI).

**PR Template (optional)**
```
## What & Why

## How
- change A
- change B

## Tests
- [ ] pytest -q
- [ ] CLI: python -m guca.cli.run_gum --genome ... --assert

Closes #123
```

---

## 5) Testing & Quality Gates
- **Unit tests** for each core module (node/graph/rules/machine).
- **Golden tests** for M1 genomes (e.g., `dummy_bell`, `fractal7genom`):
  - run with fixed steps, assert `{nodes, edges, states_count?}`.
  - expectations live in the same YAML as the genome.
- **Fitness tests**: planarity filter, facet proxy, monotonic sanity checks.
- **GA smoke**: tiny pop/gens completes fast; deterministic with fixed seed.
- **Markers**: use `fast`/`slow` if needed; CI runs `fast` subset by default.

**Reproducibility**
- Always set a `seed` in configs; log it in outputs.
- Store Hydra configs / CLI args snapshot with run artifacts.

---

## 6) Configs & Runs
- Configs in `configs/` (Hydra style) and small YAMLs in `examples/`.
- CLI entry points (suggested):
  - `python -m guca.cli.run_gum --genome <yaml> [--steps N] [--assert]`
  - `python -m guca.cli.score_graph --genome <yaml> --fitness <name> [--steps N]`
  - `python -m guca.examples.run_ga [overrides...]`
- Logs: print JSON summary for quick copy-paste to Issues/PRs.

---

## 7) Decisions & ADRs
- Non-trivial design choices → add an ADR in `docs/adr/ADR-YYYYMMDD-<slug>.md`.
- Keep it short: Context → Decision → Consequences (+ Alternatives if helpful).

**ADR Template**
```
# <Title>
Date: YYYY-MM-DD

## Context

## Decision

## Consequences

## Alternatives considered
```

---

## 8) AI Collaboration Guidelines
- **Context you provide:**
  - File paths and minimal snippets (avoid huge diffs unless asked).
  - Goals, constraints, expected outputs; if there’s a golden expectation, state it.
  - Hardware/runtime limits (so solutions stay practical).
- **What AI can do here:**
  - Draft code files, tests, YAMLs, CLI stubs, Issues/PR texts.
  - Cross-check parity vs. legacy C# logic (based on provided snippets).
- **What AI cannot do:**
  - Push to your repo, or access private resources by itself.
  - Keep secrets; never paste tokens/credentials.
- **Determinism:**
  - Ask for seeded examples; prefer testable outputs over screenshots.
- **Security/Privacy:**
  - No personal data; avoid proprietary datasets; sanitize logs.

---

## 9) Style & Tooling
- **Python**: black + ruff (recommended), type hints where helpful.
- **Dependencies**: keep `pyproject.toml` minimal; pin versions for CI stability.
- **Directory layout** (suggested):
```
guca-evolution-lab/
  src/guca/
    core/        # node, graph, rules, machine
    fitness/     # fitness functions
    evo/         # DEAP integration
    cli/         # CLI entry points
    examples/    # sample GA scripts
  configs/       # hydra configs
  examples/      # genomes/graphs YAMLs
  tests/
  docs/
    adr/
    PLAN.md
    CODEBOOK.md
```

---

## 10) Tips & Agreements (quick-reference)
- Keep Issues small; weekend-focus items can be broader, but still testable.
- Always link PRs to Issues and Plan; use `Closes #N` where appropriate.
- Prefer adding a small test with each change (even if smoke level).
- Use the **raw GitHub host** if web UI/CDN glitches prevent browsing.
- If plan changes, update `docs/PLAN.md` first; then create/close Issues.

---

**Last updated:** <fill on edit>

