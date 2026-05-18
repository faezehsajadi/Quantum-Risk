"""
Build the cryptographically relevant CVE corpus from NVD JSON feeds
and the CISA Known Exploited Vulnerabilities (KEV) catalogue, then
compute per-record Quantum Factor (QF), HNDL Score (HS), and Quantum
Risk Index (QRI) following the definitions in Sections 3.3-3.5.

Inputs (in ../raw/):
    nvdcve-2.0-YYYY.json                  NVD bulk feeds, 2016-2026
    known_exploited_vulnerabilities.csv   CISA KEV catalogue

Output (in ../raw/):
    hndl_dataset.csv                      one row per crypto-relevant CVE

QRI formula (Section 3.5):
    QRI = CVSS_base x QF x (1 + HS^2 / 10)

Usage:
    python parse_all_nvd_hndl.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "raw"
OUTPUT_CSV = DATA_DIR / "hndl_dataset.csv"
KEV_CSV = DATA_DIR / "known_exploited_vulnerabilities.csv"
YEARS = range(2016, 2027)

# ---------------------------------------------------------------------------
# Quantum Factor (QF) keyword lexicons (Section 3.3, Table 1)
# ---------------------------------------------------------------------------
# Shor's algorithm provides a polynomial-time break of integer factorisation
# and discrete logarithm problems, so any asymmetric primitive whose security
# rests on those problems is flagged with QF = 1.5.
SHOR_KEYWORDS: tuple[str, ...] = (
    "ecdsa", "elliptic curve", "secp256k1", "ecc", "rsa", "rsa-oaep",
    "dsa", "diffie-hellman", "dh key", "ecdh", "ed25519", "curve25519",
    "public key cryptograph", "asymmetric", "pkcs#1", "pkcs1",
    "private key", "key pair", "digital signature", "x.509", "pki",
)

# Grover's algorithm halves the effective key length of symmetric primitives
# and hash functions; QF = 1.2 reflects this partial reduction.
GROVER_KEYWORDS: tuple[str, ...] = (
    "sha-1", "sha1", "sha-256", "sha256", "sha-512", "sha512",
    "md5", "keccak", "blake2", "hash function", "hmac",
    "aes-128", "aes-128-", "128-bit", "symmetric key", "block cipher",
)

# Generic cryptographic terms where the primitive cannot be identified from
# the CVE description (QF = 1.1 — residual quantum exposure).
GENERIC_CRYPTO_KEYWORDS: tuple[str, ...] = (
    "cryptograph", "cipher", "encrypt", "decrypt", "tls ", "ssl ",
    "openssl", "bouncycastle", "libsodium", "nss ", "gnutls",
    "certificate", "x.509", "pkix", "handshake", "key exchange",
)

# ---------------------------------------------------------------------------
# HNDL Score (HS) keyword sets (Section 3.4, Table 2)
# ---------------------------------------------------------------------------
HNDL_STORAGE_KEYWORDS: tuple[str, ...] = (
    "database", "storage", "archive", "backup", "log", "record",
    "persistent", "disk", "file system", "s3", "blob", "vault",
    "healthcare", "medical", "patient", "hospital", "ehr",
    "financial", "banking", "payment", "transaction", "ledger",
    "blockchain", "ethereum", "bitcoin", "cryptocurrency", "wallet",
    "government", "military", "classified", "sensitive",
    "identity", "authentication", "credential", "token",
)

HNDL_CHANNEL_KEYWORDS: tuple[str, ...] = (
    "tls", "ssl", "https", "ssh", "vpn", "ipsec", "starttls",
    "openvpn", "wireguard", "kerberos", "ldap", "smtps", "imaps",
    "key exchange", "key agreement", "key establishment",
)

HIGH_VALUE_KEYWORDS: tuple[str, ...] = (
    "blockchain", "financial", "bank", "healthcare", "government",
    "military", "critical infrastructure", "smart grid", "scada",
)

KEY_EXPOSURE_KEYWORDS: tuple[str, ...] = (
    "private key", "session key", "master key",
    "key disclosure", "key exposure", "key leak",
)

HS_CAP = 6


def quantum_factor(description: str) -> float:
    """Return QF in {1.0, 1.1, 1.2, 1.5} per the rules in Table 1."""
    text = description.lower()
    if any(k in text for k in SHOR_KEYWORDS):
        return 1.5
    if any(k in text for k in GROVER_KEYWORDS):
        return 1.2
    if any(k in text for k in GENERIC_CRYPTO_KEYWORDS):
        return 1.1
    return 1.0


def hndl_score(description: str) -> int:
    """Return HS in {0..6} per the additive rubric in Table 2."""
    text = description.lower()
    score = 0
    if any(k in text for k in HNDL_STORAGE_KEYWORDS):
        score += 2
    if any(k in text for k in HNDL_CHANNEL_KEYWORDS):
        score += 2
    if any(k in text for k in HIGH_VALUE_KEYWORDS):
        score += 1
    if any(k in text for k in KEY_EXPOSURE_KEYWORDS):
        score += 1
    return min(score, HS_CAP)


def quantum_risk_index(cvss: float, qf: float, hs: int) -> float:
    """QRI = CVSS_base x QF x (1 + HS^2 / 10)  (Section 3.5)."""
    return round(cvss * qf * (1.0 + hs ** 2 / 10.0), 4)


# ---------------------------------------------------------------------------
# NVD JSON field extraction
# ---------------------------------------------------------------------------
def extract_cvss(metrics: dict) -> tuple[float, str]:
    """Return (base_score, version_used).

    NVD records may carry CVSS v4.0, v3.1, v3.0, or v2 metrics. The highest
    version present is preferred. Primary-source entries are preferred over
    secondary contributions. When no metric is published, NVD's documented
    default of 5.0 is used.
    """
    for key, version in (
        ("cvssMetricV40", "4.0"),
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV30", "3.0"),
        ("cvssMetricV2", "2.0"),
    ):
        entries = metrics.get(key, [])
        if not entries:
            continue
        primary = next((e for e in entries if e.get("type") == "Primary"), entries[0])
        score = primary.get("cvssData", {}).get("baseScore")
        if score is not None:
            return float(score), version
    return 5.0, "default"


def extract_english_description(descriptions: list) -> str:
    for item in descriptions:
        if item.get("lang") == "en":
            return item.get("value", "")
    return ""


def extract_cwes(weaknesses: list) -> list[str]:
    cwes: list[str] = []
    for w in weaknesses:
        for d in w.get("description", []):
            if d.get("lang") == "en":
                value = d.get("value", "")
                if value.startswith("CWE-"):
                    cwes.append(value)
    return cwes


# ---------------------------------------------------------------------------
# CISA KEV loader
# ---------------------------------------------------------------------------
def load_kev_ids(path: Path) -> set[str]:
    if not path.exists():
        print(f"WARNING: CISA KEV file not found at {path}; "
              f"in_kev column will be 0 for all records.")
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cve_id = (row.get("cveID") or "").strip()
            if cve_id:
                ids.add(cve_id)
    return ids


# ---------------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------------
def iter_year_records(year: int, kev_ids: set[str]) -> Iterable[dict]:
    """Yield one record per crypto-relevant CVE in the NVD feed for `year`."""
    path = DATA_DIR / f"nvdcve-2.0-{year}.json"
    if not path.exists():
        return

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    for entry in payload.get("vulnerabilities", []):
        cve = entry.get("cve", {})
        description = extract_english_description(cve.get("descriptions", []))
        if not description:
            continue

        qf = quantum_factor(description)
        hs = hndl_score(description)
        # Filter to the crypto-relevant corpus (Section 3.2).
        if qf == 1.0 and hs == 0:
            continue

        cvss, version = extract_cvss(cve.get("metrics", {}))
        cwes = extract_cwes(cve.get("weaknesses", []))
        cve_id = cve.get("id", "")

        yield {
            "cve_id": cve_id,
            "year": year,
            "published": cve.get("published", "")[:10],
            "cvss_base": cvss,
            "cvss_version": version,
            "qf": qf,
            "hndl_score": hs,
            "qri": quantum_risk_index(cvss, qf, hs),
            "in_kev": 1 if cve_id in kev_ids else 0,
            "vuln_status": cve.get("vulnStatus", ""),
            "cwes": "|".join(cwes),
            "desc_snippet": description[:150].replace("\n", " "),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    kev_ids = load_kev_ids(KEV_CSV)
    print(f"CISA KEV entries loaded: {len(kev_ids):,}")

    records: list[dict] = []
    by_year: dict[int, int] = defaultdict(int)
    for year in YEARS:
        year_records = list(iter_year_records(year, kev_ids))
        records.extend(year_records)
        by_year[year] = len(year_records)
        print(f"  {year}: {by_year[year]:>6,} crypto-relevant CVEs")

    if not records:
        raise RuntimeError(
            f"No NVD JSON feeds were processed. Expected files like "
            f"nvdcve-2.0-2016.json in {DATA_DIR}."
        )

    fieldnames = list(records[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    n_kev = sum(r["in_kev"] for r in records)
    print(
        f"\nWrote {len(records):,} records ({n_kev} CISA KEV) "
        f"to {OUTPUT_CSV.relative_to(SCRIPT_DIR.parent)}"
    )


if __name__ == "__main__":
    main()
