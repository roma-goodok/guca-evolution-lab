**Framework Selection and Decision Documentation**

## 1) User Interface and Modular Concept of Application

### **Decision:**

- Start with a **Command-Line Interface (CLI)** implementation.
- Use **Hydra-Core** for flexible configuration management.
- Integrate **ClearML** for experiment tracking and logging.
- If a richer UI is needed in later stages, adopt **PyQt** for a **single-user desktop tool** with real-time graph visualization.

### **Rationale:**

- **Modularity**: A CLI-based approach allows for easy headless execution, automation, and remote experimentation without the complexity of GUI dependencies.
- **Reproducibility**: Using Hydra-Core ensures consistent and versioned experiment configurations, making it easier to rerun and tweak parameter settings.
- **Experiment Tracking**: ClearML provides a robust way to log, compare, and visualize experiment results without requiring custom-built visualization.
- **Future Scalability**: PyQt is a suitable choice if interactive visualization becomes necessary, as it provides a responsive and native-feeling UI without requiring a web-based deployment.
- **Comparison with Alternatives:**
  - **Flask/Dash (Web UI):** Not selected due to the complexity of setting up a web server and maintaining client-server communication for real-time updates.
  - **Other GUI frameworks (Tkinter, wxWidgets, Kivy, etc.):** PyQt offers the best balance of **performance, flexibility, and integration with Python scientific computing libraries**.

---

## 2) Genetic Algorithm and Evolutionary Programming Framework

### **Decision:**

- Use **DEAP (Distributed Evolutionary Algorithms in Python)** as the primary genetic algorithm framework.

### **Rationale:**

- **Flexibility:** DEAP supports arbitrary-length genomes, which aligns well with evolving **graph change tables**.
- **Parallel Execution:** Built-in support for multiprocessing allows efficient evaluation of populations across CPU cores, leveraging modern hardware capabilities.
- **Customizability:** The DEAP toolbox architecture allows for defining unique mutation, crossover, and selection strategies tailored for **graph evolution**.
- **Community and Documentation:** DEAP has extensive documentation and a strong user base, making it easier to troubleshoot and extend.
- **Comparison with Alternatives:**
  - **LEAP:** While offering modern design and scalable parallelism, LEAP is relatively new with fewer community resources and examples.
  - **Inspyred:** Simpler than DEAP but lacks the same level of parallelism support and flexibility for arbitrary-length genomes.
  - **PonyGE2 (Grammatical Evolution):** More suited for evolving symbolic rule representations rather than direct genetic algorithms on change tables.

### **Conclusion:**

Using **DEAP** provides the best combination of **flexibility, performance, and integration** with existing Python-based scientific computing tools, while the CLI-based approach ensures **modularity and reproducibility**, with the potential for PyQt-based real-time visualization if needed in the future.

