export ML_LOGBOOK_DIR="logbook"

# python -m guca.cli.run_ga_hydra ga.checkpoint.fmt=yaml

# grow more structure (more births/connections)
python -m guca.cli.run_ga_hydra \
  ga.generations=10 ga.pop_size=10 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=100 \
  ga.checkpoint.save_population=all \
  ga.checkpoint.fmt=yaml \
  n_workers=8

python -m guca.cli.run_ga_hydra \
  ga.generations=1000 ga.pop_size=500 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=100 \
  ga.checkpoint.save_population=all \
  ga.checkpoint.fmt=yaml \
  n_workers=12
  
# # grow more structure (more births/connections)
# python -m guca.cli.run_ga_hydra \
#   ga.generations=1000 ga.pop_size=100 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=100  \
#   ga.checkpoint.save_population=all \
#   ga.checkpoint.fmt=yaml 



