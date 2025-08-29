#!/usr/bin/env bash
set -euo pipefail

# Quick memo: pull GUCA M1 genes → convert → freeze → verify
# Assumes the converter and freezer scripts already exist:
#   scripts/convert_m1_json_to_yaml.py
#   scripts/freeze_expected.py
#
# Usage:
#   scripts/m1_genomes_quick_update.sh

echo "== Step 1) Download the JSON if not yet =="
curl -L -o data_demo_2010_dict_genes.json \
  https://raw.githubusercontent.com/roma-goodok/guca/main/data/demo_2010_dict_genes.json

echo
echo "== Step 2) Convert to YAML genomes =="
python scripts/convert_m1_json_to_yaml.py data_demo_2010_dict_genes.json --outdir examples/genomes

echo
echo "== Step 3) Freeze expected values at 120 steps (DRY RUN) =="
python scripts/freeze_expected.py --steps 120

echo
echo "== Step 3b) Apply expected values (WRITE) =="
python scripts/freeze_expected.py --write --steps 120

echo
echo "== Step 4) Verify one genome + run tests =="
# adjust the filename if needed based on your converter's naming (snake_case)
python -m guca.cli.run_gum --genome examples/genomes/dumb_belly_genom.yaml --assert
python -m pytest -q

echo
echo "All done."
