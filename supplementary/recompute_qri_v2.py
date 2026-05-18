"""
Recompute the Quantum Risk Index (QRI) from the per-record base columns
in ``hndl_dataset.csv`` and write the result back as the ``qri`` column.

QRI formula (Section 3.5):
    QRI = CVSS_base x QF x (1 + HS^2 / 10)

This utility exists so the QRI column can be regenerated from
``cvss_base``, ``qf`` and ``hndl_score`` alone, without re-running the
full NVD-parsing pipeline (``parse_all_nvd_hndl.py``). It is provided
for transparency: a reviewer who wants to audit only the scoring step
can do so by reading this file and ``hndl_dataset.csv``.

Input  (in ../raw/):  hndl_dataset.csv
Output (in ../raw/):  hndl_dataset.csv  (qri column overwritten in place)

Usage:
    python recompute_qri_v2.py
"""

from __future__ import annotations

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = SCRIPT_DIR.parent / "raw" / "hndl_dataset.csv"


def quantum_risk_index(cvss: float, qf: float, hs: int) -> float:
    """QRI = CVSS_base x QF x (1 + HS^2 / 10)."""
    return round(cvss * qf * (1.0 + hs ** 2 / 10.0), 4)


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"{DATASET} not found. Run parse_all_nvd_hndl.py first to "
            f"build the corpus."
        )

    with open(DATASET, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for row in rows:
        row["qri"] = f"{quantum_risk_index(float(row['cvss_base']), float(row['qf']), int(row['hndl_score'])):.4f}"

    with open(DATASET, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Recomputed QRI for {len(rows):,} records in {DATASET.name}.")


if __name__ == "__main__":
    main()
