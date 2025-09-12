export ML_LOGBOOK_DIR="_logbook"

# python -m guca.cli.run_ga_hydra ga.checkpoint.fmt=yaml


# python -m guca.cli.run_ga_hydra --config-name=exp003.01 \
#   experiment.name="exp003.01.01_roulette_gen05K" \
#   ga.generations=500 ga.pop_size=200 \
#   ga.checkpoint.save_every=100

# python -m guca.cli.run_ga_hydra --config-name=exp003.01 \
#   experiment.name="exp003.01.02_roulette_gen10K" \
#   ga.generations=10000 ga.pop_size=200 \
#   ga.checkpoint.save_every=1000


# python -m guca.cli.run_ga_hydra --config-name=exp003.02 \
#   experiment.name="exp003.02.01_roulette_gen05K" \
#   ga.generations=500 ga.pop_size=200 \
#   ga.checkpoint.save_every=100

# python -m guca.cli.run_ga_hydra --config-name=exp003.02 \
#   experiment.name="exp003.02.02_roulette_gen10K" \
#   ga.generations=10000 ga.pop_size=200 \
#   ga.checkpoint.save_every=1000

# python -m guca.cli.run_ga_hydra --config-name=exp003.03 \
#   experiment.name="exp003.03.01_roulette_gen05K" \
#   ga.generations=500 ga.pop_size=200 \
#   ga.checkpoint.save_every=100

# python -m guca.cli.run_ga_hydra --config-name=exp003.03 \
#   experiment.name="exp003.03.02_roulette_gen10K" \
#   ga.generations=10000 ga.pop_size=200 \
#   ga.checkpoint.save_every=1000

# python -m guca.cli.run_ga_hydra --config-name=exp003.01 \
#   experiment.name="exp003.01.03_roulette_gen50K" \
#   ga.generations=50000 ga.pop_size=200 \
#   ga.checkpoint.save_every=1000

# python -m guca.cli.run_ga_hydra --config-name=exp003.02 \
#   experiment.name="exp003.02.03_roulette_gen50K" \
#   ga.generations=50000 ga.pop_size=200 \
#   ga.checkpoint.save_every=1000

# python -m guca.cli.run_ga_hydra --config-name=exp003.03 \
#   experiment.name="exp003.03.03_roulette_gen50K" \
#   ga.generations=50000 ga.pop_size=200 \
#   ga.checkpoint.save_every=1000
  

# python -m guca.cli.run_ga_hydra --config-name=exp003.04 \
#   experiment.name="exp003.04.01_continuable_gen05K" \
#   ga.generations=500 ga.pop_size=200 \
#   ga.checkpoint.save_every=100

# python -m guca.cli.run_ga_hydra --config-name=exp003.04 \
#   experiment.name="exp003.04.02_continuable_gen10K" \
#   ga.generations=10000 ga.pop_size=200 \
#   ga.checkpoint.save_every=1000

# python -m guca.cli.run_ga_hydra --config-name=exp003.05 \
#   experiment.name="exp003.05.01_shellpenalty_gen05K" \
#   ga.generations=500 ga.pop_size=200 \
#   ga.checkpoint.save_every=100

# python -m guca.cli.run_ga_hydra --config-name=exp003.05 \
#   experiment.name="exp003.05.02_shellpenalty_gen10K" \
#   ga.generations=10000 ga.pop_size=200 \
#   ga.checkpoint.save_every=1000

python -m guca.cli.run_ga_hydra --config-name=exp003.05 \
  experiment.name="exp003.05.03_shellpenalty_gen100K" \
  ga.generations=100000 ga.pop_size=200 \
  ga.checkpoint.save_every=5000

#   python -m guca.cli.run_ga_hydra --config-name=exp003.07 \
#   experiment.name="exp003.07.01_less_mut_gen05K" \
#   ga.generations=500 ga.pop_size=200 \
#   ga.checkpoint.save_every=100

# python -m guca.cli.run_ga_hydra --config-name=exp003.07 \
#   experiment.name="exp003.07.02_less_mut_gen10K" \
#   ga.generations=10000 ga.pop_size=200 \
#   ga.checkpoint.save_every=100



# # grow more structure (more births/connections)
# python -m guca.cli.run_ga_hydra \
#   experiment.name="exp001.00_gen50" \
#   ga.generations=50 ga.pop_size=100 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=20 \
#   ga.checkpoint.save_population=all \
#   ga.checkpoint.fmt=yaml \
#   n_workers=8

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

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp003.01_gen500_roulette" \
#   ga.selection.method=roulette \
#   ga.generations=500 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=100 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp003.02_gen5000_roulette" \
#   ga.selection.method=roulette \
#   ga.generations=5000 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=1000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=8


# #   python -m guca.cli.run_ga_hydra \
# #   experiment.name="exp003.03_gen5000_roulette_nosanitize" \
# #   ga.encoding.sanitize_on_decode=false \
# #   ga.selection.method=roulette \
# #   ga.generations=5000 ga.pop_size=100 \
# #   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
# #   ga.field.enum_delta_pb=0.25 \
# #   machine.max_steps=50 \
# #   ga.checkpoint.save_population=best \
# #   ga.checkpoint.save_every=1000 \
# #   ga.checkpoint.fmt=yaml \
# #   ga.checkpoint.save_best_png=true \
# #   n_workers=8
  
# # 50000

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp003.04_gen50K_roulette" \
#   ga.selection.method=roulette \
#   ga.generations=50000 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=10000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=6

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp001.04_gen50K" \
#   ga.selection.method=rank \
#   ga.generations=50000 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=10000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=6

#   python -m guca.cli.run_ga_hydra \
#   experiment.name="exp002.04_gen50K_tournament" \
#   ga.selection.method=tournament ga.tournament_k=3 \
#   ga.generations=50000 ga.pop_size=200 \
#   ga.structural.insert_pb=0.35 ga.structural.duplicate_pb=0.20 \
#   ga.field.enum_delta_pb=0.25 \
#   machine.max_steps=50 \
#   ga.checkpoint.save_population=best \
#   ga.checkpoint.save_every=10000 \
#   ga.checkpoint.fmt=yaml \
#   ga.checkpoint.save_best_png=true \
#   n_workers=6
