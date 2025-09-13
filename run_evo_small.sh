export ML_LOGBOOK_DIR="_logbook"

# python -m guca.cli.run_ga_hydra ga.checkpoint.fmt=yaml

# grow more structure (more births/connections)
python -m guca.cli.run_ga_hydra \
  experiment.name="ga.yaml" \
  ga.generations=50 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=20 \
  ga.checkpoint.save_population=all \
  ga.checkpoint.save_every=10 \
  ga.checkpoint.fmt=yaml \
  n_workers=8
