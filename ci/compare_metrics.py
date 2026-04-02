import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "baseline" / "metrics.json"
CURRENT_PATH = ROOT / "data" / "models" / "metrics.json"
REPORT_PATH = ROOT / "report.md"
QUALITY_GATE_METRIC = "f1_score"
MAX_DEGRADATION = 0.02


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def metric_row(name: str, baseline: dict, current: dict) -> str:
    old = baseline.get(name)
    new = current.get(name)
    if old is None or new is None:
        return f"| {name} | {old} | {new} | n/a |"
    delta = new - old
    return f"| {name} | {old:.4f} | {new:.4f} | {delta:+.4f} |"


def main() -> None:
    baseline = load_json(BASELINE_PATH)
    current = load_json(CURRENT_PATH)

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    rows = [metric_row(name, baseline, current) for name in metrics]

    quality_status = "PASS"
    reason = ""
    if QUALITY_GATE_METRIC in baseline and QUALITY_GATE_METRIC in current:
        drop = baseline[QUALITY_GATE_METRIC] - current[QUALITY_GATE_METRIC]
        if drop > MAX_DEGRADATION:
            quality_status = "FAIL"
            reason = (
                f"{QUALITY_GATE_METRIC} dropped by {drop:.4f}, "
                f"allowed degradation is {MAX_DEGRADATION:.4f}"
            )
    else:
        reason = "Baseline or current metric is missing for strict gate check."

    report = "\n".join(
        [
            "# CML Report",
            "",
            "## Metrics Comparison (baseline vs current)",
            "",
            "| Metric | Baseline | Current | Delta |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            f"**Quality Gate:** {quality_status}",
            "",
            f"- Rule: `{QUALITY_GATE_METRIC}` should not degrade by more than {MAX_DEGRADATION:.2f}",
            f"- Details: {reason or 'Within allowed threshold.'}",
            "",
            "## Confusion Matrix",
            "",
            "![confusion_matrix](./data/models/confusion_matrix.png)",
        ]
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    if quality_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
