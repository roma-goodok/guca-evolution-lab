export ML_LOGBOOK_DIR="_logbook"

# python -m guca.cli.run_ga_hydra ga.checkpoint.fmt=yaml

# grow more structure (more births/connections)
python -m guca.cli.run_ga_hydra \
  experiment.name="exp000.00_gen50_small" \
  ga.generations=50 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=20 \
  ga.checkpoint.save_population=all \
  ga.checkpoint.save_every=10 \
  ga.checkpoint.fmt=yaml \
  n_workers=8


python -m guca.cli.run_ga_hydra \
  experiment.name="exp003.02_gen5000_roulette_test" \
  ga.selection.method=roulette \
  ga.generations=5000 ga.pop_size=200 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=1000 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8

# python -m guca.cli.run_ga_hydra \
#   experiment.name="exp001.01_gen500" \
#   ga.selection.method=rank \
#   ga.generations=500 ga.pop_size=100 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=50 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp001.02_gen5000" \
#   ga.selection.method=rank \
#   ga.generations=5000 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=1000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp001.03_gen5000_nosanitize" \
#   ga.encoding.sanitize_on_decode=false \
#   ga.selection.method=rank \
#   ga.generations=5000 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=1000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp002.01_gen500_tournament" \
#   ga.selection.method=tournament ga.tournament_k=3 \
#   ga.generations=500 ga.pop_size=100 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=100 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp002.02_gen5000_tournament" \
#   ga.selection.method=tournament ga.tournament_k=3 \
#   ga.generations=5000 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=1000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp002.03_gen5000_tournament_nosanitize" \
#   ga.encoding.sanitize_on_decode=false \
#   ga.selection.method=tournament ga.tournament_k=3 \
#   ga.generations=5000 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=1000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8

  # python -m guca.cli.run_ga_hydra \
  # experiment.name="exp003.01_gen500_roulette" \
  # ga.selection.method=roulette \
  # ga.generations=500 ga.pop_size=200 \
  # ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  # ga.field.enum_delta_pb=0.25 \
  # machine.max_steps=50 \
  # ga.checkpoint.save_population=best \
  # ga.checkpoint.save_every=100 \
  # ga.checkpoint.fmt=yaml \
  # ga.checkpoint.save_best_png=true \
  # n_workers=8

  # python -m guca.cli.run_ga_hydra \
  # experiment.name="exp003.02_gen5000_roulette" \
  # ga.selection.method=roulette \
  # ga.generations=5000 ga.pop_size=200 \
  # ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  # ga.field.enum_delta_pb=0.25 \
  # machine.max_steps=50 \
  # ga.checkpoint.save_population=best \
  # ga.checkpoint.save_every=1000 \
  # ga.checkpoint.fmt=yaml \
  # ga.checkpoint.save_best_png=true \
  # n_workers=8


#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp003.03_gen5000_roulette_nosanitize" \
#   ga.encoding.sanitize_on_decode=false \
#   ga.selection.method=roulette \
#   ga.generations=5000 ga.pop_size=100 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=1000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8
  
# 50000

  # python -m guca.cli.run_ga_hydra \
  # experiment.name="exp003.04_gen50K_roulette" \
  # ga.selection.method=roulette \
  # ga.generations=50000 ga.pop_size=200 \
  # ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  # ga.field.enum_delta_pb=0.25 \
  # machine.max_steps=50 \
  # ga.checkpoint.save_population=best \
  # ga.checkpoint.save_every=10000 \
  # ga.checkpoint.fmt=yaml \
  # ga.checkpoint.save_best_png=true \
  # n_workers=6

  # python -m guca.cli.run_ga_hydra \
  # experiment.name="exp001.04_gen50K" \
  # ga.selection.method=rank \
  # ga.generations=50000 ga.pop_size=200 \
  # ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  # ga.field.enum_delta_pb=0.25 \
  # machine.max_steps=50 \
  # ga.checkpoint.save_population=best \
  # ga.checkpoint.save_every=10000 \
  # ga.checkpoint.fmt=yaml \
  # ga.checkpoint.save_best_png=true \
  # n_workers=6

  # python -m guca.cli.run_ga_hydra \
  # experiment.name="exp002.04_gen50K_tournament" \
  # ga.selection.method=tournament ga.tournament_k=3 \
  # ga.generations=50000 ga.pop_size=200 \
  # ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  # ga.field.enum_delta_pb=0.25 \
  # machine.max_steps=50 \
  # ga.checkpoint.save_population=best \
  # ga.checkpoint.save_every=10000 \
  # ga.checkpoint.fmt=yaml \
  # ga.checkpoint.save_best_png=true \
  # n_workers=6
