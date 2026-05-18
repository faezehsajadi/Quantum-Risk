"""
Sensitivity analysis for the Quantum Risk Index (Section 4.5).

Reproduces Tables 7 and 8: the KEV vs non-KEV separation is recomputed
under five Quantum Factor (QF) parameterisations and six HNDL Score (HS)
functional forms, then compared against plain CVSS for binary KEV
classification by ROC-AUC.

Input   (in ../raw/):       hndl_dataset.csv
Output  (in ../results/):   sensitivity_report.txt

Usage:
    python sensitivity_analysis.py
"""

from __future__ import annotations

import csv
import io
import math
import statistics
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = SCRIPT_DIR.parent / "raw" / "hndl_dataset.csv"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
REPORT_PATH = RESULTS_DIR / "sensitivity_report.txt"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
QF_BASELINE = {1.0: 1.0, 1.1: 1.1, 1.2: 1.2, 1.5: 1.5}

QF_SCENARIOS: dict[str, dict[float, float]] = {
    "Baseline (Shor=1.5, Grover=1.2)": QF_BASELINE,
    "Scenario A (Shor=1.3, Grover=1.1)": {1.0: 1.0, 1.1: 1.05, 1.2: 1.1, 1.5: 1.3},
    "Scenario B (Shor=1.7, Grover=1.3)": {1.0: 1.0, 1.1: 1.15, 1.2: 1.3, 1.5: 1.7},
    "Scenario C (Shor=2.0, Grover=1.5)": {1.0: 1.0, 1.1: 1.20, 1.2: 1.5, 1.5: 2.0},
    "Scenario D (no QF adjustment)":     {1.0: 1.0, 1.1: 1.00, 1.2: 1.0, 1.5: 1.0},
}

HS_BASELINE: Callable[[int], float] = lambda hs: 1.0 + hs ** 2 / 10.0

HS_FORMULAS: dict[str, Callable[[int], float]] = {
    "Quadratic   1 + HS^2/10  (baseline)": HS_BASELINE,
    "Linear      1 + HS/2":                lambda hs: 1.0 + hs / 2.0,
    "Cubic       1 + HS^3/50":             lambda hs: 1.0 + hs ** 3 / 50.0,
    "Binary      2 if HS >= 2 else 1":     lambda hs: 2.0 if hs >= 2 else 1.0,
    "Logarithmic log(HS + 2)":             lambda hs: math.log(hs + 2),
    "No HS adjustment (HS term = 1)":      lambda hs: 1.0,
}


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def load_records() -> list[dict]:
    rows: list[dict] = []
    with open(DATASET, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "cvss": float(row["cvss_base"]),
                    "qf": float(row["qf"]),
                    "hs": int(row["hndl_score"]),
                    "in_kev": int(row["in_kev"]),
                }
            )
    return rows


def compute_qri(
    records: list[dict],
    qf_map: dict[float, float],
    hs_formula: Callable[[int], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return QRI, CVSS-base and KEV-label vectors aligned by record index."""
    qri = np.empty(len(records), dtype=float)
    cvss = np.empty(len(records), dtype=float)
    in_kev = np.empty(len(records), dtype=int)
    for i, record in enumerate(records):
        qri[i] = record["cvss"] * qf_map[record["qf"]] * hs_formula(record["hs"])
        cvss[i] = record["cvss"]
        in_kev[i] = record["in_kev"]
    return qri, cvss, in_kev


def separation_test(scores: np.ndarray, in_kev: np.ndarray) -> dict:
    """One-sided Mann-Whitney U comparing KEV vs non-KEV scores."""
    kev = scores[in_kev == 1]
    non = scores[in_kev == 0]
    u_stat, p_value = stats.mannwhitneyu(kev, non, alternative="greater")
    return {
        "n_kev": len(kev),
        "n_non": len(non),
        "mean_kev": float(np.mean(kev)),
        "mean_non": float(np.mean(non)),
        "diff_pct": 100 * (np.mean(kev) - np.mean(non)) / np.mean(non),
        "u": float(u_stat),
        "p": float(p_value),
        "auc": float(u_stat / (len(kev) * len(non))),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.buffer = io.StringIO()

    def __call__(self, *args, **kwargs) -> None:
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.buffer)

    def section(self, title: str) -> None:
        self("\n" + "=" * 65)
        self(f"  {title}")
        self("=" * 65)


def report_qf_sensitivity(records: list[dict], log) -> None:
    log.section("Sensitivity to QF parameterisation (Table 7)")
    log(
        f"\n  {'Scenario':42} {'KEV mean':>9} {'non-KEV':>9} "
        f"{'Diff %':>8} {'p-value':>12}"
    )
    log("  " + "-" * 86)
    for name, qf_map in QF_SCENARIOS.items():
        qri, _, in_kev = compute_qri(records, qf_map, HS_BASELINE)
        stats_dict = separation_test(qri, in_kev)
        p_str = f"{stats_dict['p']:.2e}"
        log(
            f"  {name:42} {stats_dict['mean_kev']:>9.3f} "
            f"{stats_dict['mean_non']:>9.3f} "
            f"{stats_dict['diff_pct']:>+7.1f}% {p_str:>12}"
        )


def report_hs_sensitivity(records: list[dict], log) -> None:
    log.section("Sensitivity to HS functional form (Table 8)")
    log(
        f"\n  {'HS formula':42} {'KEV mean':>9} {'non-KEV':>9} "
        f"{'Diff %':>8} {'p-value':>12}"
    )
    log("  " + "-" * 86)
    for name, formula in HS_FORMULAS.items():
        qri, _, in_kev = compute_qri(records, QF_BASELINE, formula)
        stats_dict = separation_test(qri, in_kev)
        p_str = f"{stats_dict['p']:.2e}"
        log(
            f"  {name:42} {stats_dict['mean_kev']:>9.3f} "
            f"{stats_dict['mean_non']:>9.3f} "
            f"{stats_dict['diff_pct']:>+7.1f}% {p_str:>12}"
        )


def report_cvss_versus_qri(records: list[dict], log) -> None:
    log.section("QRI versus plain CVSS for KEV classification (Table 9)")
    qri, cvss, in_kev = compute_qri(records, QF_BASELINE, HS_BASELINE)

    qri_stats = separation_test(qri, in_kev)
    cvss_stats = separation_test(cvss, in_kev)

    log(
        f"\n  {'Metric':14} {'KEV mean':>10} {'non-KEV':>10} "
        f"{'Diff %':>8} {'p-value':>13} {'ROC-AUC':>9}"
    )
    log("  " + "-" * 70)
    for name, s in (("Plain CVSS", cvss_stats), ("QRI (baseline)", qri_stats)):
        log(
            f"  {name:14} {s['mean_kev']:>10.3f} {s['mean_non']:>10.3f} "
            f"{s['diff_pct']:>+7.1f}% {s['p']:>13.3e} {s['auc']:>9.4f}"
        )

    log(
        f"\n  Note: QRI raises the mean separation by "
        f"{qri_stats['diff_pct'] - cvss_stats['diff_pct']:+.1f} percentage "
        f"points relative to CVSS. The ROC-AUC values are univariate; "
        f"Section 4.5.4 reports the incremental contribution of QRI over "
        f"CVSS via multivariable logistic regression."
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    print(f"Loaded {len(records):,} records from {DATASET.name}.\n")

    log = Report()
    log("=" * 65)
    log("  Sensitivity Analysis - QRI Formula Parameters")
    log("  Dataset: NVD 2016-2026, n = 78,587, CISA KEV = 423")
    log("=" * 65)

    report_qf_sensitivity(records, log)
    report_hs_sensitivity(records, log)
    report_cvss_versus_qri(records, log)

    REPORT_PATH.write_text(log.buffer.getvalue(), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH.relative_to(SCRIPT_DIR.parent)}")


if __name__ == "__main__":
    main()
