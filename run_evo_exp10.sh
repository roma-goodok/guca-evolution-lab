export ML_LOGBOOK_DIR="_logbook"

python -m guca.cli.run_ga_hydra --config-path=../../../configs/phase01 --config-name=exp003.10 \
  experiment.name="exp003.10.01_tregen5K" \
  ga.generations=5000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-path=../../../configs/phase01 --config-name=exp003.10 \
  experiment.name="exp003.10.02_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-path=../../../configs/phase01 --config-name=exp003.10 \
  experiment.name="exp003.10.03_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-path=../../../configs/phase01 --config-name=exp003.10 \
  experiment.name="exp003.10.04_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-path=../../../configs/phase01 --config-name=exp003.10 \
  experiment.name="exp003.10.05_tregen10K" \
  ga.generations=10000 ga.pop_size=200 \
  ga.checkpoint.save_every=1000


python -m guca.cli.run_ga_hydra --config-path=../../../configs/phase01 --config-name=exp003.10 \
  experiment.name="exp003.10.06_tregen100K" \
  ga.generations=100000 ga.pop_size=200 \
  ga.checkpoint.save_every=10000