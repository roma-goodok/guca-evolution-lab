# Install & Run

## Prereqs
- Python ≥ 3.10 (3.11 recommended)
- pip (and optionally Conda/Miniconda)

---

## Option A — Quick run without install
From the repo root:
```bash
PYTHONPATH=src python -m guca.cli.run_gum --genome examples/genomes/dummy_bell.yaml
```

---

## Option B — Editable install (recommended)

Install runtime and dev dependencies:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Run tests and CLI:

```bash
python -m pytest -q
python -m guca.cli.run_gum --genome examples/genomes/dummy_bell.yaml
```

---

## Option C — Conda environment

```bash
conda create -n guca-lab python=3.11 -y
conda activate guca-lab
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Test run:

```bash
python -m pytest -q
python -m guca.cli.run_gum --genome examples/genomes/dummy_bell.yaml
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'guca'`**
  - Ensure you’re in the repo root
  - Did you install with `pip install -e .`?
  - Or prefix your command with `PYTHONPATH=src`

- **Different Python picks up the package?**
  - Print interpreter:
    ```bash
    python -c "import sys; print(sys.executable)"
    ```
  - Use the same interpreter for install and run.


## Jupyter on a headless server (Ubuntu) + SSH tunnel

> Assumes you already installed runtime deps (`requirements.txt`).
> For notebooks, install dev tools in your virtualenv/conda env:

```bash
# (on server)
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name guca-lab --display-name "Python (guca-lab)"
```

Run Jupyter without opening a browser, bound to localhost:

```bash
# (on server, in the repo root)
jupyter lab --no-browser --port 8888
# or:
jupyter notebook --no-browser --port 8888
```
Create an SSH tunnel from your local machine:

# (on your laptop)
ssh -N -L 8888:127.0.0.1:8888 roma@<server_ip>


Open the printed URL in your local browser (it will look like http://127.0.0.1:8888/lab?token=...).

If you prefer a fixed token/password:

jupyter lab --no-browser --port 8888 --ServerApp.token='' --ServerApp.password=''
# (empty token; for trusted environments only)
