"""Fail fast if Hydra config does not compose (used in CI)."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def main() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")
    assert cfg.hpo is not None
    assert hasattr(cfg.hpo, "oversample")
    print("Hydra config OK:", CONFIG_DIR)


if __name__ == "__main__":
    main()
