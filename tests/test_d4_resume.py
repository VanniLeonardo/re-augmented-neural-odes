"""D4 resume/idempotency invariants.

The D4 CIFAR-10 run died twice mid-flight. The harness used to `unlink()` its
trajectory on startup, so restarting it destroyed every finished seed. Resuming is
therefore a correctness requirement, and the *key* it resumes on is the load-bearing
detail:

  training is sequential and no checkpoint is saved, so a seed stopped at epoch 9
  cannot be resumed at epoch 10 -- there is no model state to resume from.

Keying resume on (model, seed, epoch) would train a fresh random init for the missing
epoch and record it as the continuation of a 9-epoch run, silently corrupting the
result. Resume must key on the COMPLETE (model, seed). These tests pin that.
"""

import csv

from scripts.run_d4_anode_cifar import FIELDS, complete_pairs, resume_trajectory

LADDER = [1e-5, 1e-6, 1e-7]
EPOCHS = 10


def _rows(model: str, seed: int, epochs, ladder=LADDER, hardware: str = ""):
    return [{"model": model, "seed": str(seed), "epoch": str(e), "eval_tol": repr(t),
             "test_acc": "0.5", "recon_ok": "1", "hardware": hardware}
            for e in epochs for t in ladder]


def _write(path, rows, fields=FIELDS):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def test_complete_seed_is_recognised() -> None:
    rows = _rows("NODE", 0, range(1, EPOCHS + 1))
    assert complete_pairs(rows, EPOCHS, LADDER) == {("NODE", 0)}


def test_partial_seed_is_not_resumable() -> None:
    """THE bug this guards: 9 of 10 epochs must NOT count as complete."""
    rows = _rows("NODE", 3, range(1, EPOCHS))  # epochs 1..9
    assert complete_pairs(rows, EPOCHS, LADDER) == set()


def test_seed_missing_one_tolerance_is_not_complete() -> None:
    """A missing ladder rung means the faithful comparison cannot be made for that
    seed, so the seed is not done."""
    rows = _rows("NODE", 0, range(1, EPOCHS + 1), ladder=[1e-5, 1e-6])
    assert complete_pairs(rows, EPOCHS, LADDER) == set()


def test_arms_are_tracked_independently() -> None:
    rows = _rows("NODE", 0, range(1, EPOCHS + 1)) + _rows("ANODE-p10", 0, range(1, 4))
    assert complete_pairs(rows, EPOCHS, LADDER) == {("NODE", 0)}


def test_resume_keeps_complete_seeds_and_drops_orphans(tmp_path) -> None:
    """End-to-end on the real file path: the shape D4 actually died in -- NODE seeds
    0-2 complete, seed 3 stopped at epoch 9, zero ANODE rows."""
    traj = tmp_path / "d4_trajectory.csv"
    rows = (_rows("NODE", 0, range(1, EPOCHS + 1)) + _rows("NODE", 1, range(1, EPOCHS + 1))
            + _rows("NODE", 2, range(1, EPOCHS + 1)) + _rows("NODE", 3, range(1, EPOCHS)))
    _write(traj, rows)

    done, n_prior, n_kept = resume_trajectory(traj, EPOCHS, LADDER, "TEST-GPU")

    assert done == {("NODE", 0), ("NODE", 1), ("NODE", 2)}
    assert n_prior == 117 and n_kept == 90  # the 27 orphan seed-3 rows are dropped
    kept = list(csv.DictReader(traj.open()))
    assert len(kept) == 90
    assert {int(r["seed"]) for r in kept} == {0, 1, 2}  # seed 3 will be re-run fresh


def test_resume_is_idempotent(tmp_path) -> None:
    """Resuming twice must not duplicate or drop rows -- the harness is restarted by
    hand after a death, so this happens for real."""
    traj = tmp_path / "d4_trajectory.csv"
    _write(traj, _rows("NODE", 0, range(1, EPOCHS + 1)))

    first = resume_trajectory(traj, EPOCHS, LADDER, "TEST-GPU")
    after_one = traj.read_text()
    second = resume_trajectory(traj, EPOCHS, LADDER, "TEST-GPU")

    assert first[0] == second[0] == {("NODE", 0)}
    assert traj.read_text() == after_one  # byte-identical: no duplication, no loss


def test_resume_backfills_hardware_on_pre_resume_rows(tmp_path) -> None:
    """The committed 117 rows predate the `hardware` column; upgrading the file must
    fill it rather than shifting every column."""
    traj = tmp_path / "d4_trajectory.csv"
    old_fields = [f for f in FIELDS if f != "hardware"]
    _write(traj, _rows("NODE", 0, range(1, EPOCHS + 1)), fields=old_fields)

    resume_trajectory(traj, EPOCHS, LADDER, "RTX 3090")

    kept = list(csv.DictReader(traj.open()))
    assert all(r["hardware"] == "RTX 3090" for r in kept)
    assert all(r["model"] == "NODE" for r in kept)  # columns did not shift


def test_existing_hardware_is_not_overwritten(tmp_path) -> None:
    """Rows that already record their GPU keep it -- seeds may be run on more than one
    machine and the row must say which."""
    traj = tmp_path / "d4_trajectory.csv"
    _write(traj, _rows("NODE", 0, range(1, EPOCHS + 1), hardware="A100-40GB"))

    resume_trajectory(traj, EPOCHS, LADDER, "RTX 3090")

    assert {r["hardware"] for r in csv.DictReader(traj.open())} == {"A100-40GB"}


def test_fresh_discards_everything(tmp_path) -> None:
    traj = tmp_path / "d4_trajectory.csv"
    _write(traj, _rows("NODE", 0, range(1, EPOCHS + 1)))

    done, n_prior, n_kept = resume_trajectory(traj, EPOCHS, LADDER, "TEST-GPU", fresh=True)

    assert done == set() and n_prior == 0 and n_kept == 0
    assert list(csv.DictReader(traj.open())) == []
