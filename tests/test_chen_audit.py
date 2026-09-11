from scripts.chen_fig3c_audit import blocks, export, inferred_forward


def test_logs_and_extracted_csv_give_the_same_counts(tmp_path) -> None:
    log = tmp_path / "nfe_logs"
    log.write_text("source\nNamespace(x=1)\n1e-05\n10\nNumber of NFE in backward: 4\n9\n"
                   "Number of NFE in backward: 5\nTol 1e-05 | Acc 0.9\n0.0001\n6\n"
                   "Number of NFE in backward: 2\nTol 1e-04 | Acc 0.9\n")
    raw = list(blocks(str(log)))
    assert raw == [("1e-05", [10, 9], [4, 5]), ("1e-04", [6], [2])]
    export(raw, tmp_path / "c.csv")
    assert list(blocks(str(tmp_path / "c.csv"))) == raw
    assert inferred_forward([10, 9], [4, 5]) == [5.0, 4]  # 9 - 4 - 1
