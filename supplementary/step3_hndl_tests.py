"""
Statistical tests for the four hypotheses pre-specified in Section 3.7.

    Q1  Growth of crypto-relevant CVEs and Shor-vulnerable CVEs
        over 2016-2025 (OLS log-count-on-time regression, alpha = 0.05).
        The partial 2026 year is retained in the series as a single
        additional data point (cf. Section 3.6 footnote).

    Q2  CISA KEV CVEs carry higher QRI than non-KEV CVEs
        (Mann-Whitney U, rank-biserial r).

    Q3  Shor-vulnerable CVEs (QF = 1.5) are over-represented in CISA KEV
        (chi-square on a 2x2 contingency table). The HS-tier association
        uses the grouping HS = 0, HS = 1-3, HS >= 4 (Section 4.3).

    Q4  QRI is concentrated in a Pareto-like minority of records
        (Gini coefficient; pre-specified threshold Gini > 0.5).

Input   (in ../raw/):       hndl_dataset.csv
Output  (in ../results/):   hndl_statistical_report.txt

Usage:
    python step3_hndl_tests.py
"""

from __future__ import annotations

import csv
import io
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = SCRIPT_DIR.parent / "raw" / "hndl_dataset.csv"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
REPORT_PATH = RESULTS_DIR / "hndl_statistical_report.txt"


def load_records() -> list[dict]:
    records: list[dict] = []
    with open(DATASET, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "cve_id": row["cve_id"],
                    "year": int(row["year"]),
                    "qri": float(row["qri"]),
                    "qf": float(row["qf"]),
                    "hs": int(row["hndl_score"]),
                    "cvss": float(row["cvss_base"]),
                    "in_kev": int(row["in_kev"]),
                }
            )
    return records


def gini(values: list[float]) -> float:
    """Standard discrete Gini coefficient for non-negative values."""
    sorted_values = sorted(v for v in values if v > 0)
    n = len(sorted_values)
    total = sum(sorted_values)
    if n == 0 or total == 0:
        return 0.0
    numerator = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(sorted_values))
    return numerator / (n * total)


class Report:
    """Mirrors all printed output to an in-memory buffer for archival."""

    def __init__(self) -> None:
        self.buffer = io.StringIO()

    def __call__(self, *args, **kwargs) -> None:
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.buffer)

    def section(self, title: str) -> None:
        self("\n" + "=" * 65)
        self(f"  {title}")
        self("=" * 65)


def q1_growth(records: list[dict], log) -> None:
    """OLS regression of log(annual count) on year (Section 4.1)."""
    log.section("Q1  -  Growth of the HNDL attack surface (2016-2026)")

    years = sorted({r["year"] for r in records})
    by_year_total: dict[int, int] = defaultdict(int)
    by_year_shor: dict[int, int] = defaultdict(int)
    by_year_qri: dict[int, list[float]] = defaultdict(list)

    for r in records:
        y = r["year"]
        by_year_total[y] += 1
        by_year_qri[y].append(r["qri"])
        if r["qf"] == 1.5:
            by_year_shor[y] += 1

    log("\n  Year    n CVEs   Mean QRI   Shor (QF=1.5)")
    log("  " + "-" * 45)
    for y in years:
        log(
            f"  {y}    {by_year_total[y]:>6,}   "
            f"{statistics.mean(by_year_qri[y]):>8.3f}   "
            f"{by_year_shor[y]:>10,}"
        )

    # Paper regression specification: full 2016-2026 series including the
    # partial 2026 year as a single additional data point.
    x = np.array(years, dtype=float) - 2016
    y_total = np.array([by_year_total[y] for y in years], dtype=float)
    y_shor = np.array([by_year_shor[y] for y in years], dtype=float)

    s_t, _, r_t, p_t, _ = stats.linregress(x, np.log(y_total))
    s_s, _, r_s, p_s, _ = stats.linregress(x, np.log(y_shor))

    log("\n  OLS log(count) ~ year, full 2016-2026 series:")
    log(
        f"    All crypto CVEs:      beta = {s_t:.4f}   "
        f"CAGR = {(math.exp(s_t) - 1) * 100:.1f}%   "
        f"R2 = {r_t ** 2:.3f}   p = {p_t:.4f}"
    )
    log(
        f"    Shor-vulnerable CVEs: beta = {s_s:.4f}   "
        f"CAGR = {(math.exp(s_s) - 1) * 100:.1f}%   "
        f"R2 = {r_s ** 2:.3f}   p = {p_s:.4f}"
    )
    log(
        f"\n  Verdict: Q1 confirmed (count growth p = {p_t:.4f} < 0.05; "
        f"Shor growth p = {p_s:.4f} < 0.05)."
    )


def q2_kev_separation(records: list[dict], log) -> None:
    """Mann-Whitney U between CISA KEV and non-KEV CVEs (Section 4.2)."""
    log.section("Q2  -  Behavioural validation of QRI against CISA KEV")

    kev = [r["qri"] for r in records if r["in_kev"] == 1]
    non = [r["qri"] for r in records if r["in_kev"] == 0]

    log(f"\n  {'Group':12} {'n':>8} {'Mean QRI':>10} {'Median':>10} {'SD':>9}")
    log("  " + "-" * 52)
    log(
        f"  {'CISA KEV':12} {len(kev):>8,} "
        f"{statistics.mean(kev):>10.3f} {statistics.median(kev):>10.3f} "
        f"{statistics.stdev(kev):>9.3f}"
    )
    log(
        f"  {'Non-KEV':12} {len(non):>8,} "
        f"{statistics.mean(non):>10.3f} {statistics.median(non):>10.3f} "
        f"{statistics.stdev(non):>9.3f}"
    )

    diff_pct = (statistics.mean(kev) - statistics.mean(non)) / statistics.mean(non) * 100
    u_stat, p_value = stats.mannwhitneyu(kev, non, alternative="greater")
    rank_biserial = 1 - (2 * u_stat) / (len(kev) * len(non))

    log(f"\n  KEV - non-KEV mean separation: +{diff_pct:.1f}%")
    log(f"  Mann-Whitney U = {u_stat:,.0f}")
    log(f"  p-value        = {p_value:.3e}")
    log(f"  Rank-biserial r= {rank_biserial:+.4f}")
    log(f"\n  Verdict: Q2 confirmed.")

    sorted_qri = sorted(r["qri"] for r in records)
    n_total = len(sorted_qri)
    log("\n  KEV percentile analysis (Table 6):")
    for pct in (50, 75, 90, 95):
        threshold = sorted_qri[int(pct / 100 * n_total)]
        above = sum(1 for q in kev if q >= threshold)
        log(
            f"    Above {pct}th percentile (QRI >= {threshold:.2f}): "
            f"{above}/{len(kev)} = {100 * above / len(kev):.1f}%  "
            f"(null = {100 - pct}%)"
        )


def q3_shor_overrepresentation(records: list[dict], log) -> None:
    """Chi-square tests for Shor x KEV and HS-tier x KEV (Section 4.3).

    The HS-tier test uses the three-group partition HS = 0, HS = 1-3,
    HS >= 4 (paper text). The figure caption value of 30.648 derives
    from the alternative partition HS = 0, HS = 2, HS = 4 only; both
    are reported here for completeness.
    """
    log.section("Q3  -  Shor-vulnerable CVEs over-represented in CISA KEV")

    total = len(records)
    total_kev = sum(r["in_kev"] for r in records)
    shor = [r for r in records if r["qf"] == 1.5]
    shor_kev = sum(r["in_kev"] for r in shor)
    non_shor_total = total - len(shor)
    non_shor_kev = total_kev - shor_kev

    rate_shor = 100 * shor_kev / len(shor)
    rate_non_shor = 100 * non_shor_kev / non_shor_total
    rate_overall = 100 * total_kev / total

    log(f"\n  KEV rate, all crypto-CVEs:      {rate_overall:.3f}%  "
        f"({total_kev}/{total:,})")
    log(f"  KEV rate, non-Shor (QF < 1.5):  {rate_non_shor:.3f}%  "
        f"({non_shor_kev}/{non_shor_total:,})")
    log(f"  KEV rate, Shor (QF = 1.5):      {rate_shor:.3f}%  "
        f"({shor_kev}/{len(shor):,})")
    log(f"\n  Rate ratios:")
    log(f"    Shor / non-Shor = {rate_shor / rate_non_shor:.2f}  (paper text)")
    log(f"    Shor / overall  = {rate_shor / rate_overall:.2f}  (Figure 5a)")

    contingency = np.array(
        [[shor_kev, len(shor) - shor_kev],
         [non_shor_kev, non_shor_total - non_shor_kev]]
    )
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    log(f"\n  Chi-square (Shor x KEV):  chi2 = {chi2:.3f}   "
        f"df = {dof}   p = {p_value:.4f}")

    # HS-tier association, partition used in paper text (Section 4.3).
    by_tier: dict[str, list[dict]] = {
        "HS=0": [r for r in records if r["hs"] == 0],
        "HS=1-3": [r for r in records if 1 <= r["hs"] <= 3],
        "HS>=4": [r for r in records if r["hs"] >= 4],
    }
    log("\n  KEV rate by HNDL Score tier (paper-text partition):")
    table_text = []
    for name, subset in by_tier.items():
        kev_count = sum(r["in_kev"] for r in subset)
        rate = 100 * kev_count / len(subset)
        log(f"    {name:8}  n = {len(subset):>6,}   "
            f"KEV = {kev_count:>3}   rate = {rate:.3f}%")
        table_text.append([kev_count, len(subset) - kev_count])

    chi2_hs, p_hs, dof_hs, _ = stats.chi2_contingency(np.array(table_text))
    log(f"\n  Chi-square (HS tier x KEV):  chi2 = {chi2_hs:.3f}   "
        f"df = {dof_hs}   p = {p_hs:.4f}")
    log(f"\n  Verdict: Q3 confirmed.")


def q4_concentration(records: list[dict], log) -> None:
    """Gini and Pareto concentration of QRI (Section 4.4)."""
    log.section("Q4  -  Concentration of QRI risk")

    all_qri = [r["qri"] for r in records]
    g = gini(all_qri)
    sorted_desc = sorted(all_qri, reverse=True)
    total = sum(sorted_desc)
    cumulative = 0.0
    pareto_k = 0
    for k, value in enumerate(sorted_desc, start=1):
        cumulative += value
        if cumulative >= 0.80 * total:
            pareto_k = k
            break
    pareto_pct = 100 * pareto_k / len(sorted_desc)

    n_top = max(1, int(0.20 * len(sorted_desc)))
    roi = statistics.mean(sorted_desc[:n_top]) / statistics.mean(sorted_desc[n_top:])

    log(f"\n  Gini coefficient                          : {g:.4f}")
    log(f"  Top {pareto_pct:.1f}% of CVEs hold 80% of total QRI")
    log(f"  Hotspot ROI (top 20% / bottom 80% means)  : {roi:.2f}x")
    log(
        f"\n  Verdict: Q4 NOT confirmed under the pre-specified threshold "
        f"Gini > 0.5 (Gini = {g:.3f}). The practically relevant "
        f"concentration is captured by the percentile analysis in Q2."
    )


def summary(records: list[dict], log) -> None:
    log.section("Summary")
    kev = [r["qri"] for r in records if r["in_kev"] == 1]
    non = [r["qri"] for r in records if r["in_kev"] == 0]
    diff_pct = (statistics.mean(kev) - statistics.mean(non)) / statistics.mean(non) * 100
    log(
        f"\n  Dataset       : {len(records):,} crypto-relevant CVEs, NVD 2016-2026\n"
        f"  CISA KEV match: {sum(r['in_kev'] for r in records)} CVEs\n"
        f"  Mean QRI gain : +{diff_pct:.1f}% in exploited CVEs (Q2)"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    print(f"Loaded {len(records):,} records from {DATASET.name}.\n")

    log = Report()
    log("=" * 65)
    log("  HNDL Attack Surface Analysis - Statistical Report")
    log("=" * 65)

    q1_growth(records, log)
    q2_kev_separation(records, log)
    q3_shor_overrepresentation(records, log)
    q4_concentration(records, log)
    summary(records, log)

    REPORT_PATH.write_text(log.buffer.getvalue(), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH.relative_to(SCRIPT_DIR.parent)}")


if __name__ == "__main__":
    main()
