# Vision
I have an idea to implement a genetic programming approach based on graph cellular automata. Graph unfolding cellular automata (GUCA) involve a program that executes commands on a graph (such as connecting nodes and generating new ones) depending on the state of a node or its connections. In other words, I want to perform an evolutionary search for such programs (which can be considered as genes) to find, for example, a program that generates a graph with specified properties (determined by a fitness function). Examples include a ring graph, a grid graph, or a 2D graph that resembles a butterfly. I already have some work done on graph unfolding cellular automata and some old C# code for an evolutionary algorithm. However, I want to rewrite it in Python using modern genetic algorithm frameworks that support parallelism on CPU cores. Obviously, I don't want to program everything from scratch since there are surely ready-made frameworks available, preferably in Python.

# High-Level Plan
The current high-level plan includes the following milestones:

M1. Web Demo of GUCA (almost DONE):
Re-implement the web demo application (in TypeScript) to show how the growing graph cellular automata works (here is an implementation: https://github.com/roma-goodok/guca/; I can also provide the source code). During the re-implementation, I found that some operations could be adjusted. Also, the aesthetics need some improvement, but for now it's good enough.

M2. Graph Evolution Lab:
Re-implement the first version of the evolutionary algorithm and reproduce some old experiments using CPU parallelization. I also have old C# code for this. I plan to re-implement it using Python and modern, advanced frameworks for genetic algorithms and evolutionary programming, such as DEAP.

M3. Advanced Genetic Algorithms and Experiments:
Develop advanced genetic algorithms and run sophisticated experiments to collect results with insights and findings. During this milestone, the web demo application and Graph Evolution Lab are expected to be enhanced. For example, improve the Graph Evolution Lab to conduct experiments faster and more conveniently, and in a reproducible manner (using ClearML for experiment tracking and visualization).

M4. Public Project Web Page:

Prepare a public project web page (e.g., GitHub Pages)
Write an article
Publish the source code
Provide visualization examples

M5. Replicators and interactions in the 2D

# Further possible research directions: 
 In the **current GUCA approach**, reproduction (i.e., generating a separate graph seed) does not lead to **competition** because the newly formed graphs remain **isolated**, and their evolution is governed solely by their **internal conditions** (local node connections and states within the same graph).

However, if **GUCA were extended to operate in a 2D or 3D spatial environment**, it could introduce **weak connections** or **proximity-based interactions** between separate graphs. Some possible enhancements:

1. **Spatial Interaction Rules**  
   - Introduce **distance-based conditions** for interactions between different graph instances. For example, a node could attempt to **connect with a node from another graph** if it is within a certain spatial radius.

2. **Weak Connections ("Virtual Edges")**  
   - Even if two graphs are not explicitly connected via edges, nodes **close enough in space** could be considered **weakly connected**. This could allow:
     - Influence from nearby graphs (e.g., diffusion-like effects).
     - Emergent **swarming** or **aggregation** behaviors.

3. **Environmental Influence on Evolution**  
   - In a **spatialized version of GUCA**, evolution could be influenced not just by **internal graph conditions**, but also by the **local environment**. For instance:
     - Certain areas of space could provide **favorable conditions** for growth.
     - External **fields or gradients** could attract or repel graph structures.
     - "Nutrients" or **resource zones** could promote expansion in a given direction.

4. **Competition Between Graphs**  
   - If separate graphs can interact, they could:
     - **Compete** for resources (if a node from one graph consumes something that another needs).
     - **Fuse** into hybrid structures if their conditions align.
     - **Attack** each other by disconnecting edges or occupying shared space.

5. **Colony-Like Growth & Evolution**  
   - A spatial extension could allow for **cooperative behaviors**, where clusters of nodes from different graphs **coordinate** or **specialize** to achieve higher-order structure.

This would transform **GUCA from a purely graph-based model to a hybrid "spatialized graph unfolding system"**, bridging **graph evolution, cellular automata, and swarm intelligence**.

Definitely worth thinking about! 🚀