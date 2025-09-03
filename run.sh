export ML_LOGBOOK_DIR="logbook"

# python -m guca.cli.run_ga_hydra ga.checkpoint.fmt=yaml

# grow more structure (more births/connections)
python -m guca.cli.run_ga_hydra \
  ga.generations=50 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=20 \
  ga.checkpoint.save_population=all \
  ga.checkpoint.fmt=yaml \
  n_workers=8

python -m guca.cli.run_ga_hydra \
  ga.generations=1000 ga.pop_size=100 \
  ga.structural.insert_pb=0.5 ga.structural.duplicate_pb=0.5 \
  ga.field.enum_delta_pb=0.25 \
  ga.tournament_k=2 \
  ga.elitism=0 \
  ga.init_len=1 \
  machine.max_steps=20 \
  ga.checkpoint.save_population=all \
  ga.checkpoint.save_every=100 \
  n_workers=12

python -m guca.cli.run_ga_hydra \
  ga.generations=1000 ga.pop_size=100 \
  ga.structural.insert_pb=0.25 ga.structural.duplicate_pb=0.5 \
  ga.field.enum_delta_pb=0.05 \
  ga.tournament_k=10 \
  ga.elitism=0 \
  ga.init_len=16 \
  machine.max_steps=20 \
  ga.checkpoint.save_population=all \
  ga.checkpoint.save_every=100 \
  n_workers=12
  
# # grow more structure (more births/connections)
# python -m guca.cli.run_ga_hydra \
#   ga.generations=1000 ga.pop_size=100 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=100  \
#   ga.checkpoint.save_population=all \
#   ga.checkpoint.fmt=yaml 



