from bloom_policy import compose_instruction, response_instruction
from uncertainty import UncertaintyEngine

def test_each_bloom_level_has_distinct_policy():
    policies = [response_instruction(level) for level in ("remember", "understand", "apply", "analyze", "evaluate", "create")]
    assert len(set(policies)) == 6
    assert "Bloom-aware" in compose_instruction("analyze", task="question answering")

def test_uncertainty_gate_falls_back_for_uniform_distribution():
    gate = UncertaintyEngine(K=6).gate([1/6] * 6, threshold=0.4)
    assert not gate["accepted"]
