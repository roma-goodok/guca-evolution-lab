export ML_LOGBOOK_DIR="_logbook"

python -m guca.cli.run_ga_hydra --config-path=../../../configs/quadmesh_phase01 --config-name=exp001 \
  experiment.name="exp001.01_tregen1K" \
  ga.generations=1000 ga.pop_size=200 \
  ga.checkpoint.save_every=100

python -m guca.cli.run_ga_hydra --config-path=../../../configs/quadmesh_phase01 --config-name=exp001 \
  experiment.name="exp001.02_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-path=../../../configs/quadmesh_phase01 --config-name=exp001 \
  experiment.name="exp001.03_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-path=../../../configs/quadmesh_phase01 --config-name=exp001 \
  experiment.name="exp001.04_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-path=../../../configs/quadmesh_phase01 --config-name=exp001 \
  experiment.name="exp001.05_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000


python -m guca.cli.run_ga_hydra --config-path=../../../configs/quadmesh_phase01 --config-name=exp001 \
  experiment.name="exp001.06_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000