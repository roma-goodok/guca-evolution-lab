# Install & Run

## Prereqs
- Python ≥ 3.10 (3.11 recommended)
- pip (and optionally Conda/Miniconda)

## Option A — Quick run without install
From the repo root:
```bash
PYTHONPATH=src python -m guca.cli.run_gum --genome examples/genomes/dummy_bell.yaml
```

## Option B — Editable install (recommended)
```bash
python -m pip install -U pip
python -m pip install -e .
pytest -q
python -m guca.cli.run_gum --genome examples/genomes/dummy_bell.yaml
```

## Option C — Conda environment
```bash
conda create -n guca-lab python=3.11 -y
conda activate guca-lab
python -m pip install -U pip
python -m pip install -e .
pytest -q
python -m guca.cli.run_gum --genome examples/genomes/dummy_bell.yaml
```

## Troubleshooting
- `ModuleNotFoundError: No module named 'guca'`
  - Ensure you’re in the repo root
  - Did you install with `pip install -e .`?
  - Or prefix your command with `PYTHONPATH=src`
- Different Python picks up the package?
  - Print interpreter: `python -c "import sys; print(sys.executable)"`
  - Use the same interpreter for install and run.
