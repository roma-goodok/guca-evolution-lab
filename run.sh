export ML_LOGBOOK_DIR="logbook"

python -m guca.cli.run_ga_hydra ga.checkpoint.fmt=yaml

# grow more structure (more births/connections)
python -m guca.cli.run_ga_hydra \
  ga.generations=60 ga.pop_size=60 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=200 \
  fitness.weights.target_presence_bonus=2.0 \
  ga.checkpoint.save_population=all \



