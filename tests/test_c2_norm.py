from scripts.c2_norm_report import verdict


def test_verdict_matches_the_predeclared_condition() -> None:
    assert verdict(5.0) == "SUPPORTED"
    assert verdict(3.0) == "PARTIAL"
    assert verdict(1.9) == "REFUTED"
