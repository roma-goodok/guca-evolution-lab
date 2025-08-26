# GUCA Evolution Lab

This repository contains the **Graph Evolution Lab** for exploring evolutionary algorithms applied to **Graph Unfolding Cellular Automata (GUCA)**.

## Project Overview

1. **Objective**  
   - Implement an evolutionary framework to evolve “change tables” (rules) that drive graph unfolding.
   - Reproduce and extend old experiments from the original C# codebase, now in modern Python with parallelization support.

2. **Main Components**  
   - **Python GUM Machine** (Graph Unfolding Machine):  
     - Core classes for graph nodes, edges, and unfolding logic.
   - **Genetic Algorithm**:
     - Based on **DEAP** to handle variable-length chromosomes representing GUCA rules.
   - **Experiment Management**:
     - CLI-based approach (optionally using **Hydra-Core**) for easy parameter configuration.
     - Optional integration with **ClearML** or **Weights & Biases** for tracking experiment results.
   - **Analytics**:
     - Jupyter notebooks for visualization and analysis of evolved graphs.

3. **Repository Structure**

guca-evolution-lab/
├─ src/
├─ tests/
├─ docs/
├─ examples/
├─ configs/
└─ README.md

- **src/**: Core Python code for the Graph Unfolding Machine (GUM), genetic algorithm setup, and experiment runners.  
- **tests/**: Unit tests (and possibly integration tests) to ensure each module works correctly.  
- **docs/**: Project documentation, user guides, and design notes.  
- **examples/**: Sample scripts or notebooks demonstrating how to run GUCA experiments or visualize results.  
- **configs/**: YAML or JSON configuration files defining experiment parameters (population size, mutation rates, etc.).  
- **README.md**: The top-level project overview (this file).

4. **How to Get Started**  
- Clone the repository:  
  ```
  git clone git@github.com:YOUR_USERNAME/guca-evolution-lab.git
  cd guca-evolution-lab
  ```
- Install required packages (Python 3.10+ recommended). For example:
  ```
  pip install -r requirements.txt
  ```
- Run a sample experiment:
  ```
  python src/run_experiment.py +experiment=default
  ```

5. **References**  
- Original **GUCA Web Demo**: [github.com/roma-goodok/guca/](https://github.com/roma-goodok/guca/)  
- Detailed plan in `04_milestone_M2_plan.md` for milestone tasks and deliverables.


