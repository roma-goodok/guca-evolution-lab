# Contributing to GUCA Evolution Lab

Thanks for your interest in improving the project!  
This document explains how to get a local dev setup and the basic workflow.

---

## 1. Development setup

We recommend using a fresh **Conda** or **virtualenv**.

```bash
# create and activate a conda env
conda create -n guca-lab python=3.11 -y
conda activate guca-lab

# install package (editable mode) + dev deps
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

See [docs/INSTALL.md](docs/INSTALL.md) for details.

---

## 2. Running tests

```bash
python -m pytest -q
```

- Fast unit tests must always pass before opening a PR.
- Golden tests (M1 genomes) will assert against known expected graph sizes once filled in.

---

## 3. Git & Repo hygiene

- Never commit build artifacts:
  - `__pycache__/`
  - `*.egg-info/`
  - `dist/`, `build/`
- See `.gitignore` in the repo root.
- Work on feature branches:
  - `feature/<topic>` (e.g., `feature/core-machine`)
  - `fix/<topic>`, `docs/<topic>`

---

## 4. Issues & Planning

- Small, actionable tasks are tracked as **Issues**.
- Short-term roadmap lives in [docs/PLAN.md](docs/PLAN.md).
- Guidelines, labels, and workflow are described in [docs/CODEBOOK.md](docs/CODEBOOK.md).

---

## 5. Pull Requests

- Keep PRs small and focused (≤ ~400 lines).
- Fill in:
  - **What & Why**
  - **How** (bullet list of changes)
  - **Testing** (commands run)
  - **Links** (Issues, Plan)
- Use `Closes #<issue>` to auto-close Issues.

---

## 6. Style & Tools

- Python ≥3.10
- [Black](https://black.readthedocs.io/) + [Ruff](https://github.com/astral-sh/ruff) formatting (see `pyproject.toml`).
- Type hints where practical.
- Deterministic runs: always set a random seed when applicable.

---

## 7. Questions?

Open an Issue or check with project maintainers.  
For bigger design choices, add an ADR in `docs/adr/`.

---

Happy hacking 🚀
