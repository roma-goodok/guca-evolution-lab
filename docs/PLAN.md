# GUCA Evolution Lab — Short-Term Plan (Weeks 1–3)

## Week 1 — Core GUM + M1 golden tests ✅ Completed
- [x] #1 Core: Node/Graph/Rules/Machine (parity with C# ops)  
- [x] #2 CLI: run genome YAML and assert stats  
- [x] #3 Golden tests: M1 genomes (dummy_bell → dumb_belly_genom, fractal7genom → fractal7_genom, plus others)  
- [x] #4 Project bootstrap: pyproject + pytest cfg (+ruff/black optional)  

**Deliverables:**  
- Core GUM engine with resettable/continuable transcription modes.  
- CLI `run_gum` executable from YAML.  
- Converter + freezer pipeline from GUCA JSON → YAML.  
- Golden tests passing (`pytest -q`).  

---


## Week 2 — Fitness v0 + scoring CLI (in progress)
- [ ] Fitness v0: planarity filter + facet proxy + BySample & Mesh heuristics  
- [ ] CLI: score graph/genome with fitness; YAML presets  
- [ ] Tests: fitness smoke + monotonicity sanity  
- [ ] Hydra configs for machine/fitness settings  

**Target Deliverables:**  
- `fitness/planar_basic.py` with basic fitness functions.  
- CLI `score_graph` to evaluate genomes.  
- Example YAML configs (`triangle`, `quad`, `hex`).  

---

## Week 3 — DEAP integration + configs + GA run (planned)
- [ ] Encoding/toolbox/evaluate for GA  
- [ ] Hydra configs for GA population/steps/etc.  
- [ ] GA example script with CSV logs  
- [ ] Tests: GA smoke test, reproducibility with seed  

**Target Deliverables:**  
- Working GA loop with DEAP.  
- Population fitness log, reproducibility with seeds.  
- Config-driven runs (`python -m guca.examples.run_ga`).  
