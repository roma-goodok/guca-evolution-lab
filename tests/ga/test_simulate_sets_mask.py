from guca.ga.toolbox import simulate_genome
from guca.ga.encoding import Rule, OpKind

def test_simulate_sets_activity_mask_simple():
    # one rule: when A, GiveBirthConnected B; should fire at least once
    states = ["A", "B", "C"]
    # Minimal rule object (operand=1 -> "B")
    r = Rule(cond_current=0, op_kind=OpKind.GiveBirthConnected, operand=1)

    G, mask = simulate_genome([int(r)], states=states, machine_cfg={"start_state": "A", "max_steps": 2}, collect_activity=True)
    assert isinstance(mask, list) and len(mask) == 1
    assert mask[0] is True  # the rule should have effected changes
    assert G.number_of_nodes() >= 2 and G.number_of_edges() >= 1
