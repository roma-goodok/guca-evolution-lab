python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_genom.yaml \
  --steps 120 \
  --save-png


  python -m guca.cli.run_gum \
  --genome examples/genomes/fractal7_genom.yaml \
  --steps 120 \
  --save-png \
  --vis-node-render none

python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_and_hirsute_circle_hybrid_genom.yaml \
  --steps 120 \
  --save-png \
  --vis-node-render dots


python -m guca.cli.run_gum \
  --genome examples/genomes/hirsute_circle_genom.yaml \
  --steps 120 \
  --save-png \
  --vis-node-render dots

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

python -m guca.cli.run_gum \
  --genome examples/genomes/HexMesh_64.13_short_continuable.yaml \
  --steps 120 \
  --save-png \
  --vis-node-render dots


python -m guca.cli.run_gum \
  --genome examples/genomes/HexMesh_64.13_short_resettable.yaml \
  --steps 121 \
  --save-png \
  --vis-node-render dots

python -m guca.cli.run_gum \
  --genome examples/genomes/HexMesh_55.0_short_43gens_continuable.yaml \
  --steps 120 \
  --save-png \
  --vis-node-render dots

python -m guca.cli.run_gum \
  --genome examples/genomes/HexMesh_55.0_short_43gens_resettable.yaml \
  --steps 121 \
  --save-png \
  --vis-node-render dots

