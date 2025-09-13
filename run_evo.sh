export ML_LOGBOOK_DIR="_logbook"

python -m guca.cli.run_ga_hydra --config-path=../../../configs/phase01 --config-name=exp003.05 \
  experiment.name="exp003.05.03_shellpenalty_gen100K" \
  ga.generations=100000 ga.pop_size=200 \
  ga.checkpoint.save_every=5000