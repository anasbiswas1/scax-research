
from scax.metrics.validity import validity_verdict


def test_plan_example_is_inconclusive():
    v = validity_verdict(25, 500, 0.05)
    assert v["verdict"] == "inconclusive"
    assert abs(v["upper"] - 0.0691) < 5e-4


def test_low_and_excess():
    assert validity_verdict(5, 500, 0.05)["verdict"] == "low enough"
    assert validity_verdict(45, 500, 0.05)["verdict"] == "excess"


def test_multiplicity_softens_excess():
    single = validity_verdict(36, 500, 0.05)
    many = validity_verdict(36, 500, 0.05, k_configs=200)
    assert many["lower"] < single["lower"]
