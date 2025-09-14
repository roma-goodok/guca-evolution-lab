# python -m guca.cli.run_gum \
#   --genome examples/genomes/dumb_belly_genom.yaml \
#   --steps 121 \
#   --save-png


#   python -m guca.cli.run_gum \
#   --genome examples/genomes/fractal7_genom.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render none

# python -m guca.cli.run_gum \
#   --genome examples/genomes/dumb_belly_and_hirsute_circle_hybrid_genom.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render dots


# python -m guca.cli.run_gum \
#   --genome examples/genomes/hirsute_circle_genom.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render dots


python -m guca.cli.run_gum \
  --genome _logbook/trimesh/baseline/exp003.05.03_shellpenalty_gen100K/ga_experiment_20250914_020852/checkpoints/last_00246/genome.yaml \
  --run-dir _logbook/trimesh/baseline/exp003.05.03_shellpenalty_gen100K/ga_experiment_20250914_020852/checkpoints/last_00246/rerun_gum \
  --save-png \
  --vis-node-render ids


# visualize directly from a checkpoint genome.yaml
python -m guca.cli.draw_edge_list --yaml _logbook/trimesh/baseline/exp003.05.03_shellpenalty_gen100K/ga_experiment_20250914_020852/checkpoints/last_00246/genome.yaml \
  --run-dir _logbook/trimesh/baseline/exp003.05.03_shellpenalty_gen100K/ga_experiment_20250914_020852/checkpoints/last_00246/graph_vis \
  --vis-node-render ids






# python -m guca.cli.run_gum \
#   --genome examples/genomes/primitive_fractal_genom.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render dots


# python -m guca.cli.run_gum \
#   --genome examples/genomes/strange_figure1_genom.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render dots

# python -m guca.cli.run_gum \
#   --genome examples/genomes/strange_figure2_genom.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render dots

# python -m guca.cli.run_gum \
#   --genome examples/genomes/HexMesh_64.13_short_continuable.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render dots


# python -m guca.cli.run_gum \
#   --genome examples/genomes/HexMesh_64.13_short_resettable.yaml \
#   --steps 121 \
#   --save-png \
#   --vis-node-render dots

# python -m guca.cli.run_gum \
#   --genome examples/genomes/HexMesh_55.0_short_43gens_continuable.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render dots

# python -m guca.cli.run_gum \
#   --genome examples/genomes/HexMesh_55.0_short_43gens_resettable.yaml \
#   --steps 121 \
#   --save-png \
#   --vis-node-render dots

