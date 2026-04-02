"""CI: load Airflow DagBag and fail on import/syntax errors in airflow/dags."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAGS = ROOT / "airflow" / "dags"

# Minimal Airflow home (no DB required for DagBag parsing)
AF_HOME = ROOT / ".airflow_ci"
AF_HOME.mkdir(exist_ok=True)

os.environ.setdefault("AIRFLOW_HOME", str(AF_HOME))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
os.environ.setdefault("AIRFLOW__CORE__DAGS_FOLDER", str(DAGS))
os.environ.setdefault("AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS", "False")
os.environ.setdefault("MLOPS_PROJECT_ROOT", str(ROOT))


def main() -> None:
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    if bag.import_errors:
        for path, err in bag.import_errors.items():
            print(f"--- {path} ---\n{err}", file=sys.stderr)
        raise SystemExit(1)
    if not bag.dag_ids:
        print("No DAGs found.", file=sys.stderr)
        raise SystemExit(1)
    print("OK:", sorted(bag.dag_ids))


if __name__ == "__main__":
    main()
