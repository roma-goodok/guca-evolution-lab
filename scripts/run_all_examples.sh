python -m guca.cli.run_gum --genome examples/single_run_nearest.yaml --save-png

python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_genom.yaml \
  --steps 120 \
  --save-png


python -m guca.cli.run_gum \
  --genome examples/genomes/fractal7_genom.yaml \
  --steps 120 \
  --save-png \
  --vis-node-render dots \
  --vis-dot-size 10


# python -m guca.cli.run_gum \
#   --genome examples/genomes/fractal7_genom.yaml \
#   --steps 120 \
#   --save-png \
#   --vis-node-render none

python -m guca.cli.run_gum \
  --genome examples/genomes/dumb_belly_and_hirsute_circle_hybrid_genom.yaml \
  --steps 150 \
  --save-png \
  --vis-node-render none

python -m guca.cli.run_gum \
  --genome examples/genomes/hirsute_circle_genom.yaml \
  --steps 150 \
  --save-png \
  --vis-node-render none

python -m guca.cli.run_gum \
  --genome examples/genomes/primitive_fractal_genom.yaml \
  --steps 50 \
  --save-png \
  --vis-node-render none

python -m guca.cli.run_gum \
  --genome examples/genomes/strange_figure1_genom.yaml \
  --steps 100 \
  --save-png \
  --vis-node-render none

python -m guca.cli.run_gum \
  --genome examples/genomes/strange_figure2_genom.yaml \
  --steps 101 \
  --save-png \
  --vis-node-render none


# python -m guca.cli.run_gum \
#   --genome examples/genomes/HexMesh_55.0_short_43gens.yaml \
#   --steps 200 \
#   --save-png \
#   --vis-node-render none

# python -m guca.cli.run_gum \
#   --genome examples/genomes/HexMesh_64.13_short.yaml \
#   --steps 200 \
#   --save-png \
#   --vis-node-render none