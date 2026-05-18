"""
Reproduce the EPSS benchmarking of Table 9 (Section 4.5.3) and the
convergent-validity analysis of the keyword classifier against
cryptography-related CWE tags (Section 3.3.1).

The EPSS scores are read from the locally cached bulk feed downloaded
from FIRST.org; no network access is performed. The output dataset
joins per-record EPSS scores to the corpus by CVE identifier and the
convergent-validity sample of n = 200 (50 records per QF tier, seed 42)
is exported for independent inspection.

Inputs   (in ../raw/):
    hndl_dataset.csv                EPSS-augmented input
    epss_scores-YYYY-MM-DD.csv.gz   bulk feed from first.org/epss

Outputs  (in ../raw/):
    hndl_dataset_with_epss.csv      input + EPSS columns
    validation_sample.csv           stratified sample for inspection
Outputs  (in ../results/):
    epss_auc_results.txt            metrics reported in Table 9

Usage:
    python epss_and_validation.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "raw"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

DATASET_IN = DATA_DIR / "hndl_dataset.csv"
DATASET_OUT = DATA_DIR / "hndl_dataset_with_epss.csv"
VALIDATION_SAMPLE = DATA_DIR / "validation_sample.csv"
AUC_REPORT = RESULTS_DIR / "epss_auc_results.txt"
RANDOM_SEED = 42

CRYPTO_CWE_REFERENCE: tuple[str, ...] = (
    "CWE-310", "CWE-311", "CWE-312", "CWE-321", "CWE-322", "CWE-323",
    "CWE-324", "CWE-325", "CWE-326", "CWE-327", "CWE-328", "CWE-329",
    "CWE-330", "CWE-331", "CWE-332", "CWE-333", "CWE-335", "CWE-336",
    "CWE-337", "CWE-338", "CWE-339", "CWE-340", "CWE-345", "CWE-346",
    "CWE-347", "CWE-295", "CWE-296", "CWE-297", "CWE-298", "CWE-299",
)


def find_epss_file(directory: Path) -> Path:
    """Return the most recent epss_scores-*.csv.gz file present locally."""
    candidates = sorted(directory.glob("epss_scores-*.csv.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"No EPSS bulk feed found in {directory}. Download a snapshot "
            f"from https://first.org/epss/data_stats and save it as "
            f"epss_scores-YYYY-MM-DD.csv.gz."
        )
    return candidates[-1]


def load_epss(path: Path) -> pd.DataFrame:
    """Read the FIRST.org bulk feed, skipping the model-info header line."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        first_line = handle.readline()
        if first_line.startswith("#"):
            epss = pd.read_csv(handle)
        else:
            handle.seek(0)
            epss = pd.read_csv(handle, comment="#")
    return epss.rename(columns={"cve": "cve_id", "percentile": "epss_percentile"})


def join_epss(dataset_path: Path, epss_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df["qri"] = df["cvss_base"] * df["qf"] * (1.0 + df["hndl_score"] ** 2 / 10.0)
    epss = load_epss(epss_path)[["cve_id", "epss", "epss_percentile"]]
    merged = df.merge(epss, on="cve_id", how="left")
    return merged


def benchmark_against_kev(df: pd.DataFrame) -> dict:
    """Compute the per-metric KEV separation statistics used in Table 9."""
    subset = df[df["epss"].notna()].copy()
    y = subset["in_kev"].astype(int).to_numpy()

    metrics: dict[str, dict] = {}
    for name, scores in (
        ("CVSS", subset["cvss_base"].to_numpy()),
        ("QRI", subset["qri"].to_numpy()),
        ("EPSS", subset["epss"].to_numpy()),
    ):
        u_stat, p_value = stats.mannwhitneyu(
            scores[y == 1], scores[y == 0], alternative="greater"
        )
        kev_mean = float(scores[y == 1].mean())
        non_mean = float(scores[y == 0].mean())
        metrics[name] = {
            "n": int(len(subset)),
            "n_kev": int(y.sum()),
            "mean_kev": kev_mean,
            "mean_non": non_mean,
            "diff_pct": 100 * (kev_mean - non_mean) / non_mean,
            "u": float(u_stat),
            "p": float(p_value),
            "auc": float(roc_auc_score(y, scores)),
        }
    return metrics


def write_epss_report(metrics: dict, path: Path) -> None:
    lines = [
        "EPSS Empirical Comparison - Table 9",
        "=" * 60,
        f"n CVEs with EPSS join : {metrics['CVSS']['n']:,}",
        f"n CISA KEV among join : {metrics['CVSS']['n_kev']}",
        "",
        f"{'Metric':6} {'KEV mean':>12} {'non-KEV mean':>14} "
        f"{'Diff %':>9} {'p-value':>13} {'ROC-AUC':>9}",
        "-" * 64,
    ]
    for name in ("CVSS", "QRI", "EPSS"):
        m = metrics[name]
        lines.append(
            f"{name:6} {m['mean_kev']:>12.4f} {m['mean_non']:>14.4f} "
            f"{m['diff_pct']:>+8.1f}% {m['p']:>13.3e} {m['auc']:>9.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convergent_validity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reproduce Section 3.3.1: stratified n=200 sample and concordance metrics.

    A crypto-related CWE tag (Section 3.3.1 reference list) serves as a
    sparse but independent annotation signal. The classifier under test
    is the QF gating rule (QF > 1.0 means the description was matched
    against at least one cryptographic keyword set).
    """
    df = df.copy()
    df["has_crypto_cwe"] = (
        df["cwes"]
        .fillna("")
        .apply(lambda s: any(cwe in str(s) for cwe in CRYPTO_CWE_REFERENCE))
    )

    rng = np.random.default_rng(RANDOM_SEED)
    chunks = []
    for qf_value in (1.0, 1.1, 1.2, 1.5):
        tier = df[df["qf"] == qf_value]
        chunks.append(tier.sample(min(50, len(tier)), random_state=RANDOM_SEED))
    sample = pd.concat(chunks, ignore_index=True)

    y_true = sample["has_crypto_cwe"].astype(int)
    y_pred = (sample["qf"] > 1.0).astype(int)

    metrics = {
        "n": int(len(sample)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }
    return sample, metrics


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_IN.exists():
        print(f"ERROR: {DATASET_IN} not found. Run parse_all_nvd_hndl.py first.",
              file=sys.stderr)
        sys.exit(1)

    epss_path = find_epss_file(DATA_DIR)
    print(f"EPSS bulk feed: {epss_path.name}")

    merged = join_epss(DATASET_IN, epss_path)
    coverage = merged["epss"].notna().sum()
    print(f"EPSS join coverage: {coverage:,}/{len(merged):,} CVEs "
          f"({100 * coverage / len(merged):.1f}%).")
    merged.to_csv(DATASET_OUT, index=False)

    metrics = benchmark_against_kev(merged)
    write_epss_report(metrics, AUC_REPORT)
    print("\nTable 9 - KEV separation:")
    for name, m in metrics.items():
        print(f"  {name:6}  mean KEV = {m['mean_kev']:.4f}  "
              f"mean non = {m['mean_non']:.4f}  "
              f"diff = {m['diff_pct']:+.1f}%  AUC = {m['auc']:.4f}")

    sample, validity = convergent_validity(merged)
    print(f"\nConvergent validity (Section 3.3.1), n = {validity['n']}:")
    for key in ("precision", "recall", "f1", "kappa"):
        print(f"  {key:9} = {validity[key]:.3f}")

    sample_export_cols = [
        "cve_id", "year", "qf", "hndl_score", "qri", "in_kev",
        "cwes", "has_crypto_cwe", "desc_snippet",
    ]
    sample[sample_export_cols].to_csv(VALIDATION_SAMPLE, index=False)
    print(f"\nWrote: {DATASET_OUT.relative_to(SCRIPT_DIR.parent)}")
    print(f"       {AUC_REPORT.relative_to(SCRIPT_DIR.parent)}")
    print(f"       {VALIDATION_SAMPLE.relative_to(SCRIPT_DIR.parent)}")


if __name__ == "__main__":
    main()
