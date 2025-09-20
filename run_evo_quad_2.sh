export ML_LOGBOOK_DIR="_logbook"

python -m guca.cli.run_ga_hydra --config-path=../../../configs/quadmesh_phase01 --config-name=exp004 \
  experiment.name="exp004.04_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-path=../../../configs/quadmesh_phase01 --config-name=exp004 \
  experiment.name="exp004.05_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000
