# tests/ga/test_gaexperiment_normalizes.py
from guca.ga.experiment import GAExperiment, ActiveCfg

def test_active_cfg_is_dataclass():
    exp = GAExperiment(active={"factor": 0.2, "kind": "byte"})
    assert isinstance(exp.active, ActiveCfg)
