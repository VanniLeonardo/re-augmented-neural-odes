"""The rule that selects the tolerance for the reported image results.

scripts.image_table quotes accuracy and cost at the loosest tolerance where both models pass
the reconstruction check on every seed, and refuses input files that count a seed twice. If
either behaviour breaks, a reported number silently moves to a tolerance at which the solver
is not integrating.
"""
import pandas as pd
import pytest

from scripts.image_table import faithful_table, load_shards


def _row(model: str, seed: int, tol: float, ok: int, acc: float = 0.5) -> dict:
    return {"model": model, "seed": seed, "epoch": 1, "eval_tol": tol, "train_tol": 1e-3,
            "recon_ok": ok, "test_acc": acc, "test_acc_at_tol": acc, "eval_fwd_nfe": 10,
            "hardware": "test"}


def test_load_shards_refuses_duplicated_cells(tmp_path) -> None:
    shard = pd.DataFrame([_row("NODE", 0, 1e-5, 1)])
    shard.to_csv(tmp_path / "t_s0.csv", index=False)
    shard.to_csv(tmp_path / "t_s1.csv", index=False)  # same seed written twice
    with pytest.raises(SystemExit):
        load_shards(tmp_path, "t")


def test_headline_is_loosest_tol_where_both_arms_pass_every_seed(tmp_path, capsys) -> None:
    """1e-5 passes for ANODE and for NODE seed 0, but NOT NODE seed 1 -- so it must not be
    chosen; the headline has to fall back to 1e-6."""
    rows = []
    for s in (0, 1):
        rows += [_row("NODE", s, 1e-3, 0), _row("NODE", s, 1e-5, int(s == 0)),
                 _row("NODE", s, 1e-6, 1),
                 _row("ANODE-p5", s, 1e-3, 0, 0.9), _row("ANODE-p5", s, 1e-5, 1, 0.9),
                 _row("ANODE-p5", s, 1e-6, 1, 0.9)]
    faithful_table(pd.DataFrame(rows), "eval_fwd_nfe",
                   {"NODE": (0.0, 0.0), "ANODE-p5": (0.0, 0.0)}, tmp_path)
    assert "BOTH arms recon_ok on every seed: 1e-06" in capsys.readouterr().out
