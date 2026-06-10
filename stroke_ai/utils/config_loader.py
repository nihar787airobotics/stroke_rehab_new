"""
utils/config_loader.py
----------------------
Loads the project YAML configuration as a dot-accessible object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _DotDict(dict):
    """Dict subclass that allows attribute-style access (cfg.key)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None
        if isinstance(value, dict):
            return _DotDict(value)
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(config_path: str | Path | None = None) -> _DotDict:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    config_path : Path to config.yaml. Defaults to project configs/config.yaml.

    Returns
    -------
    _DotDict with attribute access.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    return _DotDict(raw)
