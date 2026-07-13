"""Pluggable experiment-logging backend.

ReScience reviewers must be able to run every experiment with **zero external
accounts**. The original code hard-imported ``wandb`` at module top in every
entry point (including ``training/utils.py``, which sits on the core training
path via ``NFEStats``) and called ``wandb.init`` online-by-default in several
scripts, so a fresh clone without a W&B login would prompt/hang before any work.

This module replaces that with a tiny logger interface and three backends:

* ``csv``  (DEFAULT) -- writes per-step metrics to ``results/logs/<run>.csv`` and
  the run config to ``results/logs/<run>.config.json``. No network, no account.
* ``none`` / ``noop`` -- discards everything (useful for smoke/CI).
* ``wandb`` -- opt-in; ``wandb`` is imported lazily so it need not even be
  installed unless this backend is selected.

Backend selection precedence: explicit ``backend=`` argument > ``$NODE_LOGGER``
environment variable > ``"csv"``.

Nothing here depends on ``wandb`` at import time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["BaseLogger", "CSVLogger", "NoOpLogger", "WandbLogger", "get_logger"]

DEFAULT_LOG_DIR = "results/logs"


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of config values to JSON-serialisable form."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


class BaseLogger:
    """Minimal logging interface shared by all backends."""

    def __init__(self, run_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.run_name = run_name
        # ``config`` is exposed so callers can read back (possibly sweep-injected)
        # hyperparameters uniformly across backends.
        self.config: Dict[str, Any] = dict(config or {})

    def log(self, metrics: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def log_image(self, key: str, path: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def finish(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    # Context-manager sugar so callers can ``with get_logger(...) as log:``.
    def __enter__(self) -> "BaseLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()


class NoOpLogger(BaseLogger):
    """Discards all metrics. Selected with backend ``none``/``noop``."""

    def log(self, metrics: Dict[str, Any]) -> None:
        return None

    def log_image(self, key: str, path: str) -> None:
        return None

    def finish(self) -> None:
        return None


class CSVLogger(BaseLogger):
    """Writes per-step metrics to a CSV and the config to a sidecar JSON.

    Rows are buffered and flushed on ``finish()`` (and opportunistically) so the
    header is the union of all logged keys in first-seen order -- callers may log
    different key sets per step without corrupting the file.
    """

    def __init__(
        self,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        out_dir: str = DEFAULT_LOG_DIR,
    ) -> None:
        super().__init__(run_name, config)
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        # Sanitise run_name for use as a filename.
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
        self._csv_path = self._out_dir / f"{safe}.csv"
        self._config_path = self._out_dir / f"{safe}.config.json"
        self._rows: List[Dict[str, Any]] = []
        self._columns: List[str] = []
        self._images: List[Dict[str, str]] = []
        self._write_config()

    def _write_config(self) -> None:
        payload = {"run_name": self.run_name, "config": _json_safe(self.config)}
        self._config_path.write_text(json.dumps(payload, indent=2))

    def log(self, metrics: Dict[str, Any]) -> None:
        row = {str(k): _json_safe(v) for k, v in metrics.items()}
        for key in row:
            if key not in self._columns:
                self._columns.append(key)
        self._rows.append(row)
        self._flush()

    def log_image(self, key: str, path: str) -> None:
        # Images are already written to disk by callers; just record the mapping.
        self._images.append({"key": key, "path": str(path)})
        if self._images:
            (self._out_dir / f"{self._csv_path.stem}.images.json").write_text(
                json.dumps(self._images, indent=2)
            )

    def _flush(self) -> None:
        import csv  # stdlib, local import keeps module import cost minimal

        with self._csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({col: row.get(col, "") for col in self._columns})

    def finish(self) -> None:
        self._flush()


class WandbLogger(BaseLogger):
    """Opt-in Weights & Biases backend. ``wandb`` is imported lazily."""

    def __init__(
        self,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        super().__init__(run_name, config)
        import wandb  # lazy: only required when this backend is selected

        self._wandb = wandb
        # Default to offline unless the caller/env asks otherwise, so even the
        # wandb backend never blocks on a login prompt by default.
        resolved_mode = mode or os.environ.get("WANDB_MODE", "offline")
        self._run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", "neural-odes-30562"),
            name=run_name,
            config=self.config,
            mode=resolved_mode,
            reinit="finish_previous",
        )
        # Expose the effective (possibly sweep-injected) config.
        try:
            self.config = dict(wandb.config)
        except Exception:
            pass

    def log(self, metrics: Dict[str, Any]) -> None:
        self._wandb.log(metrics)

    def log_image(self, key: str, path: str) -> None:
        self._wandb.log({key: self._wandb.Image(str(path))}, commit=False)

    def finish(self) -> None:
        self._wandb.finish()


def get_logger(
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    out_dir: str = DEFAULT_LOG_DIR,
) -> BaseLogger:
    """Return a logger for the selected backend.

    Args:
        run_name: Unique name for this run (used for CSV filenames / W&B run name).
        config: Hyperparameter dict; stored and exposed as ``logger.config``.
        project: W&B project (ignored by csv/noop).
        backend: ``"csv"`` (default) / ``"none"`` / ``"wandb"``. Falls back to the
            ``$NODE_LOGGER`` env var, then ``"csv"``.
        mode: W&B mode (e.g. ``"offline"``/``"online"``); ignored by csv/noop.
        out_dir: Directory for CSV logs.
    """
    chosen = (backend or os.environ.get("NODE_LOGGER", "csv")).lower()
    if chosen in {"none", "noop", "off", "disabled"}:
        return NoOpLogger(run_name, config)
    if chosen == "wandb":
        return WandbLogger(run_name, config, project=project, mode=mode)
    if chosen == "csv":
        return CSVLogger(run_name, config, out_dir=out_dir)
    raise ValueError(
        f"Unknown NODE_LOGGER backend {chosen!r}; expected 'csv', 'none', or 'wandb'."
    )
