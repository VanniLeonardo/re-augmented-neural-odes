from scripts.c1_reference_check import verdict

SMALLEST = 2.678e-5  # smallest relative error the committed claim-5 sweep reports


def test_verdict_matches_the_predeclared_condition() -> None:
    assert verdict(1e-6, SMALLEST) == "ADEQUATE"            # inside a tenth of it
    assert verdict(5.96e-6, SMALLEST) == "PARTIAL"          # what we measured
    assert verdict(3e-5, SMALLEST) == "REFERENCE-LIMITED"   # as large as the smallest error
