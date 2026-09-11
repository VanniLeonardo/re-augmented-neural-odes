from scripts.d3_long_report import verdict


def test_verdict_matches_the_predeclared_condition() -> None:
    assert verdict(96.0, 95.0) == "SUPPORTED"
    assert verdict(96.0, 94.0) == "PARTIAL"  # test accuracy alone is not enough
    assert verdict(95.2, 95.0) == "PARTIAL"
    assert verdict(94.7, 95.0) == "REFUTED"
