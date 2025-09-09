export ML_LOGBOOK_DIR="logbook"

# python -m guca.cli.run_ga_hydra ga.checkpoint.fmt=yaml

grow more structure (more births/connections)
python -m guca.cli.run_ga_hydra \
  experiment.name="exp001_gen50" \
  ga.generations=50 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=20 \
  ga.checkpoint.save_population=all \
  ga.checkpoint.fmt=yaml \
  n_workers=8

python -m guca.cli.run_ga_hydra \
  experiment.name="exp001_gen500" \
  ga.selection.method=rank \
  ga.generations=500 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=50 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp001_gen5000" \
  ga.selection.method=rank \
  ga.generations=5000 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=200 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp003_gen5000_nosanitize" \
  ga.encoding.sanitize_on_decode=false \
  ga.selection.method=rank \
  ga.generations=5000 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=200 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp002_gen500_tournament" \
  ga.selection.method=tournament ga.tournament_k=3 \
  ga.generations=500 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=50 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp002_gen5000_tournament" \
  ga.selection.method=tournament ga.tournament_k=3 \
  ga.generations=5000 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=200 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp003_gen5000_tournament_nosanitize" \
  ga.encoding.sanitize_on_decode=false \
  ga.selection.method=tournament ga.tournament_k=3 \
  ga.generations=5000 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=200 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp003_gen500_roulette" \
  ga.selection.method=roulette \
  ga.generations=500 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=50 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp003_gen5000_roulette" \
  ga.selection.method=roulette \
  ga.generations=5000 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=200 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8


  python -m guca.cli.run_ga_hydra \
  experiment.name="exp003_gen5000_roulette_nosanitize" \
  ga.encoding.sanitize_on_decode=false \
  ga.selection.method=roulette \
  ga.generations=5000 ga.pop_size=100 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=200 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=8
  
# 50000

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp001_gen50K" \
  ga.selection.method=rank \
  ga.generations=50000 ga.pop_size=200 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=200 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=6

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp002_gen50K_tournament" \
  ga.selection.method=tournament ga.tournament_k=3 \
  ga.generations=50000 ga.pop_size=200 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=10000 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=6

  python -m guca.cli.run_ga_hydra \
  experiment.name="exp003_gen50K_roulette" \
  ga.selection.method=roulette \
  ga.generations=50000 ga.pop_size=200 \
  ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
  ga.field.enum_delta_pb=0.25 \
  machine.max_steps=50 \
  ga.checkpoint.save_population=best \
  ga.checkpoint.save_every=10000 \
  ga.checkpoint.fmt=yaml \
  ga.checkpoint.save_best_png=true \
  n_workers=6


