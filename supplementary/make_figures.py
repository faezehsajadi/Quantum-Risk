"""
Generate Figures 1 through 8 of the manuscript in a single pass.

    Figure 1 - QRI computation pipeline (Sections 3.1-3.6)
    Figure 2 - Quantum Factor and HNDL Score corpus composition
    Figure 3 - Annual distribution of crypto-relevant CVEs
    Figure 4 - KEV vs non-KEV QRI distribution (violin + box)
    Figure 5 - KEV percentile analysis (Table 6)
    Figure 6 - KEV rates by Shor vulnerability and HS tier
    Figure 7 - Sensitivity of KEV separation to QF and HS specification
    Figure 8 - QRI versus CVSS and Lorenz curve of QRI

Input   (in ../raw/):       hndl_dataset.csv
Output  (in ../figures/):   fig1_*.png ... fig8_*.png

The CSV is parsed once into a Records object that exposes every shared
derived view (KEV/non-KEV splits, sorted QRI, the global Mann-Whitney
test, yearly aggregates) as a cached property, so each figure pays the
data-loading cost only once.

Usage:
    python make_figures.py
"""

from __future__ import annotations

import csv
from functools import cached_property
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy import stats

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = SCRIPT_DIR.parent / "raw" / "hndl_dataset.csv"
OUT_DIR = SCRIPT_DIR.parent / "figures"
DPI = 180
RNG_SEED = 42

# Palette held constant across the figure set.
C_BLUE = "#2E74B5"
C_ORANGE = "#C55A11"
C_GREEN = "#1E8449"
C_RED = "#C0392B"
C_GREY = "#7F8C8D"
C_LIGHT = "#EEF4FB"
C_DARK = "#1F3864"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


# ===========================================================================
# Data layer
# ===========================================================================
class Records:
    """Per-record corpus loaded from ``hndl_dataset.csv``.

    Derived views are exposed as cached properties so each figure pays
    the cost of loading and slicing exactly once.
    """

    def __init__(self, path: Path) -> None:
        cvss: list[float] = []
        qf: list[float] = []
        hs: list[int] = []
        year: list[int] = []
        in_kev: list[int] = []
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cvss.append(float(row["cvss_base"]))
                qf.append(float(row["qf"]))
                hs.append(int(row["hndl_score"]))
                year.append(int(row["year"]))
                in_kev.append(int(row["in_kev"]))
        self.cvss = np.asarray(cvss, dtype=float)
        self.qf = np.asarray(qf, dtype=float)
        self.hs = np.asarray(hs, dtype=int)
        self.year = np.asarray(year, dtype=int)
        self.in_kev = np.asarray(in_kev, dtype=int)
        # QRI v2 (Section 3.5).
        self.qri = self.cvss * self.qf * (1.0 + self.hs ** 2 / 10.0)

    @property
    def n(self) -> int:
        return int(self.qri.size)

    @cached_property
    def kev_mask(self) -> np.ndarray:
        return self.in_kev == 1

    @cached_property
    def kev_qri(self) -> np.ndarray:
        return self.qri[self.kev_mask]

    @cached_property
    def non_qri(self) -> np.ndarray:
        return self.qri[~self.kev_mask]

    @cached_property
    def kev_cvss(self) -> np.ndarray:
        return self.cvss[self.kev_mask]

    @cached_property
    def non_cvss(self) -> np.ndarray:
        return self.cvss[~self.kev_mask]

    @cached_property
    def qri_sorted(self) -> np.ndarray:
        return np.sort(self.qri)

    @cached_property
    def mw_kev(self) -> tuple[float, float, float]:
        """Mann-Whitney U, one-sided p-value, rank-biserial r."""
        u, p = stats.mannwhitneyu(
            self.kev_qri, self.non_qri, alternative="greater"
        )
        r = 1 - (2 * u) / (len(self.kev_qri) * len(self.non_qri))
        return float(u), float(p), float(r)

    @cached_property
    def years(self) -> np.ndarray:
        return np.arange(2016, 2027)

    @cached_property
    def annual_total(self) -> np.ndarray:
        return np.array([(self.year == y).sum() for y in self.years])

    @cached_property
    def annual_shor(self) -> np.ndarray:
        return np.array(
            [((self.year == y) & (self.qf == 1.5)).sum() for y in self.years]
        )

    @cached_property
    def annual_kev(self) -> np.ndarray:
        return np.array(
            [((self.year == y) & (self.in_kev == 1)).sum() for y in self.years]
        )


# ===========================================================================
# Figure 1 - QRI computation pipeline
# ===========================================================================
def figure_qri_pipeline(out_path: Path) -> None:
    """End-to-end pipeline that turns the NVD JSON feeds into the
    per-record corpus ``hndl_dataset.csv``. The four numbered phases
    (INPUT, FILTER, SCORE, OUTPUT) align with Sections 3.1-3.6. The
    dashed connectors on each side show that the CISA KEV catalogue
    and the EPSS bulk feed are joined to the corpus by CVE identifier
    rather than entering the scoring computation."""
    # Picture2 reference: pale pastels, sans-serif, circled digit phase
    # markers, L-shaped cross-reference arrows that run around the
    # diagram (not through it) and enter the OUTPUT box from the side.
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 9)
        ax.axis("off")
        fig.patch.set_facecolor("white")

        # (fill, border, dark-text)
        palette = {
            "input":  ("#E8F0F8", "#7AAAD0", "#2A4F7A"),
            "filter": ("#FBE6D0", "#D9985A", "#7A4F1E"),
            "score":  ("#DDEDC9", "#85B265", "#2A4F2A"),
            "qri":    ("#FBEC8C", "#C2A02A", "#6B5510"),
            "output": ("#FBD5DA", "#C9818A", "#6B2020"),
        }

        def rbox(x, y, w, h, style, lw=1.1, radius=0.10):
            face, edge, _ = palette[style]
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle=f"round,pad={radius}",
                facecolor=face, edgecolor=edge, linewidth=lw, zorder=3,
            ))

        def head(x, y, text, style, size=10, weight="normal"):
            _, _, color = palette[style]
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=size, fontweight=weight, color=color,
                    zorder=4, linespacing=1.30)

        def sub(x, y, text, size=9):
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=size, color="#4F4F4F",
                    zorder=4, linespacing=1.30)

        def arrow(x1, y1, x2, y2, color="#3A3A3A", lw=1.0):
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=lw,
                    mutation_scale=12,
                ), zorder=2,
            )

        # ---------- layout grid ----------
        BW = 3.3
        X1, X2, X3 = 1.7, 5.2, 8.7
        C1, C2, C3 = X1 + BW/2, X2 + BW/2, X3 + BW/2
        WIDE_X, WIDE_W = 1.7, 9.6
        WIDE_C = WIDE_X + WIDE_W / 2

        Y_INPUT,  H_INPUT  = 7.75, 0.85
        Y_FILT_1, H_FILT_1 = 6.25, 0.75
        Y_FILT_2, H_FILT_2 = 4.75, 0.85
        Y_SCORE,  H_SCORE  = 2.60, 1.25
        Y_QRI,    H_QRI    = 1.30, 0.70
        Y_OUTPUT, H_OUTPUT = 0.20, 0.60

        # ---------- phase labels with circled digits ----------
        for y, digit, text, style in (
            (Y_INPUT  + H_INPUT  / 2, "\u2460", "INPUT",  "input"),
            (Y_FILT_1 + H_FILT_1 / 2, "\u2461", "FILTER", "filter"),
            (Y_SCORE  + H_SCORE  / 2, "\u2462", "SCORE",  "score"),
            (Y_OUTPUT + H_OUTPUT / 2, "\u2463", "OUTPUT", "output"),
        ):
            _, _, color = palette[style]
            ax.text(0.55, y, digit, ha="center", va="center",
                    fontsize=13, color=color, zorder=4)
            ax.text(0.85, y, text, ha="left", va="center",
                    fontsize=10.5, fontweight="bold", color=color, zorder=4)

        # ---------- Row 1: INPUT ----------
        for x, c, title, sub_text in (
            (X1, C1, "NVD CVE API 2.0",
             "annual JSON feeds (2016 \u2013 2026)"),
            (X2, C2, "CISA KEV Catalog",
             "(n = 1,558 IDs;\nground-truth exploits)"),
            (X3, C3, "EPSS bulk feed",
             "(FIRST.org;\nexploitation likelihood)"),
        ):
            rbox(x, Y_INPUT, BW, H_INPUT, "input")
            head(c, Y_INPUT + 0.58, title, "input", size=10.5)
            sub (c, Y_INPUT + 0.24, sub_text, size=9)

        # NVD -> Raw NVD corpus
        arrow(C1, Y_INPUT, C1, Y_FILT_1 + H_FILT_1 + 0.10)

        # ---------- Row 2a: Raw NVD + Crypto-relevance ----------
        rbox(X1, Y_FILT_1, BW, H_FILT_1, "input")
        head(C1, Y_FILT_1 + 0.50, "Raw NVD corpus", "input", size=10.5)
        sub (C1, Y_FILT_1 + 0.22, "(n = 260,253 CVEs)", size=9)

        rbox(X2, Y_FILT_1, BW, H_FILT_1, "filter")
        head(C2, Y_FILT_1 + 0.50, "Crypto-relevance", "filter", size=10.5)
        sub (C2, Y_FILT_1 + 0.22, "keyword filter", size=9)

        # Raw -> Crypto-relevance
        arrow(X1 + BW + 0.05, Y_FILT_1 + H_FILT_1 / 2,
              X2 - 0.05,      Y_FILT_1 + H_FILT_1 / 2)

        # ---------- Row 2b: Filtered corpus (wide) ----------
        rbox(WIDE_X, Y_FILT_2, WIDE_W, H_FILT_2, "filter", lw=1.3)
        head(WIDE_C, Y_FILT_2 + 0.56,
             "Filtered corpus \u2014 n = 78,587",
             "filter", size=11.5, weight="bold")
        head(WIDE_C, Y_FILT_2 + 0.26,
             "crypto-relevant CVEs (30.2% of raw)",
             "filter", size=11, weight="bold")

        # Crypto-relevance -> Filtered corpus
        arrow(C2, Y_FILT_1, C2, Y_FILT_2 + H_FILT_2 + 0.10)

        # ---------- Row 3: SCORE (3 boxes) ----------
        for x, c, title, line1, line2 in (
            (X1, C1, "Quantum Factor (QF)",
             "\u2208 {1.0, 1.1, 1.2, 1.5}",   "(Shor / Grover / generic)"),
            (X2, C2, "CVSS base severity",
             "v2.0 / v3.0 / v3.1 / v4.0",      "(from NVD)"),
            (X3, C3, "HNDL Score (HS) \u2208 0\u20266",
             "storage + channel +",            "sector + key exposure"),
        ):
            rbox(x, Y_SCORE, BW, H_SCORE, "score")
            head(c, Y_SCORE + 0.92, title, "score", size=10)
            sub (c, Y_SCORE + 0.55, line1, size=9.5)
            sub (c, Y_SCORE + 0.28, line2, size=9.5)

        # Filtered corpus -> 3 SCORE boxes (three arrows fanning out)
        arrow(WIDE_X + 1.6,            Y_FILT_2, C1, Y_SCORE + H_SCORE + 0.10)
        arrow(WIDE_C,                  Y_FILT_2, C2, Y_SCORE + H_SCORE + 0.10)
        arrow(WIDE_X + WIDE_W - 1.6,   Y_FILT_2, C3, Y_SCORE + H_SCORE + 0.10)

        # ---------- Row 4: QRI formula ----------
        rbox(WIDE_X, Y_QRI, WIDE_W, H_QRI, "qri", lw=1.3)
        head(WIDE_C, Y_QRI + 0.35,
             "QRI  =  CVSS_base  \u00d7  QF  \u00d7  (1 + HS\u00b2 / 10)",
             "qri", size=12.5, weight="bold")

        # 3 SCORE -> QRI converging arrows
        arrow(C1, Y_SCORE, WIDE_X + 1.6,           Y_QRI + H_QRI + 0.10)
        arrow(C2, Y_SCORE, WIDE_C,                 Y_QRI + H_QRI + 0.10)
        arrow(C3, Y_SCORE, WIDE_X + WIDE_W - 1.6,  Y_QRI + H_QRI + 0.10)

        # ---------- Row 5: OUTPUT ----------
        rbox(WIDE_X, Y_OUTPUT, WIDE_W, H_OUTPUT, "output", lw=1.2)
        head(WIDE_C, Y_OUTPUT + 0.30,
             "hndl_dataset.csv  (n = 78,587 \u00d7 12)  +  EPSS  +  KEV",
             "output", size=11, weight="bold")

        # QRI -> OUTPUT
        arrow(WIDE_C, Y_QRI, WIDE_C, Y_OUTPUT + H_OUTPUT + 0.10)

        # ---------- Cross-references: L-shaped, outside the diagram ----------
        # Arrows enter the OUTPUT box near its top edge so the horizontal
        # segment of the L runs ABOVE the "(4) OUTPUT" phase label (which
        # sits at the vertical centre of the OUTPUT row).
        kev_color  = "#9B4F9F"
        epss_color = "#3E9B8A"

        # KEV: top-left margin -> down -> right INTO top-left of OUTPUT.
        ax.annotate(
            "", xy=(WIDE_X, Y_OUTPUT + H_OUTPUT - 0.05),
            xytext=(0.20, Y_INPUT + H_INPUT / 2),
            arrowprops=dict(
                arrowstyle="->", color=kev_color, lw=1.2,
                linestyle=(0, (5, 4)),
                connectionstyle="angle,angleA=-90,angleB=0,rad=8",
                mutation_scale=14,
            ), zorder=2,
        )
        ax.text(0.10, (Y_INPUT + Y_OUTPUT) / 2 + 0.4,
                "KEV cross-ref", rotation=90,
                fontsize=9.5, color=kev_color, fontstyle="italic",
                ha="center", va="center")

        # EPSS: top-right margin -> down -> left INTO top-right of OUTPUT.
        ax.annotate(
            "", xy=(WIDE_X + WIDE_W, Y_OUTPUT + H_OUTPUT - 0.05),
            xytext=(12.80, Y_INPUT + H_INPUT / 2),
            arrowprops=dict(
                arrowstyle="->", color=epss_color, lw=1.2,
                linestyle=(0, (5, 4)),
                connectionstyle="angle,angleA=-90,angleB=180,rad=8",
                mutation_scale=14,
            ), zorder=2,
        )
        ax.text(12.90, (Y_INPUT + Y_OUTPUT) / 2 + 0.4,
                "EPSS join", rotation=270,
                fontsize=9.5, color=epss_color, fontstyle="italic",
                ha="center", va="center")

        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ===========================================================================
# Figure 2 - QF and HS corpus composition
# ===========================================================================
def figure_qf_hs_distribution(r: Records, out_path: Path) -> None:
    """Composition of the filtered corpus by Quantum Factor (panel a)
    and HNDL Score (panel b). Horizontal bars handle the long-tailed
    QF and HS distributions more legibly than a pie chart, which
    crowds the small Shor / Grover / generic-crypto slices."""
    fig, (ax_qf, ax_hs) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel (a): Quantum Factor ---
    qf_values = (1.0, 1.1, 1.2, 1.5)
    qf_counts = [int((r.qf == v).sum()) for v in qf_values]
    qf_labels = (
        "QF = 1.0  Classical",
        "QF = 1.1  Generic crypto",
        "QF = 1.2  Grover-vulnerable",
        "QF = 1.5  Shor-vulnerable",
    )
    qf_colors = ["#ADB5BD", "#74C0FC", "#F08C00", C_BLUE]

    y_pos = np.arange(len(qf_values))
    bars_qf = ax_qf.barh(y_pos, qf_counts, color=qf_colors,
                         edgecolor="white", linewidth=1.2, zorder=2)
    for bar, count in zip(bars_qf, qf_counts):
        pct = 100 * count / r.n
        ax_qf.text(bar.get_width() + max(qf_counts) * 0.012,
                   bar.get_y() + bar.get_height() / 2,
                   f"{count:,}  ({pct:.1f}%)",
                   va="center", ha="left",
                   fontsize=9.5, color=C_DARK)

    ax_qf.set_yticks(y_pos)
    ax_qf.set_yticklabels(qf_labels, fontsize=10)
    ax_qf.invert_yaxis()
    ax_qf.set_xlabel("Number of CVEs", fontsize=11)
    ax_qf.set_xlim(0, max(qf_counts) * 1.20)
    ax_qf.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_qf.set_title(f"(a) Quantum Factor (QF) Distribution\nn = {r.n:,}",
                    fontsize=11, color=C_DARK, pad=10)

    # --- Panel (b): HNDL Score ---
    hs_values = range(6)
    hs_counts = [int((r.hs == v).sum()) for v in hs_values]
    hs_labels = ["HS=0\nNo HNDL", "HS=1\nMinor", "HS=2\nChannel",
                 "HS=3\nCh+Sector", "HS=4\nHigh-Value", "HS=5\nCritical"]
    hs_colors = ["#ADB5BD", "#74C0FC", "#4DABF7", "#339AF0", C_ORANGE, C_RED]

    bars_hs = ax_hs.bar(range(6), hs_counts, color=hs_colors,
                        edgecolor="white", linewidth=1.2, zorder=2)
    for bar, count in zip(bars_hs, hs_counts):
        pct = 100 * count / r.n
        ax_hs.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + max(hs_counts) * 0.012,
                   f"{pct:.1f}%",
                   ha="center", va="bottom",
                   fontsize=9, color=C_DARK)

    ax_hs.set_xticks(range(6))
    ax_hs.set_xticklabels(hs_labels, fontsize=9)
    ax_hs.set_ylabel("Number of CVEs", fontsize=11)
    ax_hs.set_ylim(0, max(hs_counts) * 1.18)
    ax_hs.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_hs.set_title(f"(b) HNDL Score (HS) Distribution\nn = {r.n:,}",
                    fontsize=11, color=C_DARK, pad=10)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Figure 3 - Annual CVE growth and KEV trend
# ===========================================================================
def figure_annual_growth(r: Records, out_path: Path) -> None:
    """Per-year crypto-CVE counts and overlay of CISA KEV matches.
    The 2026 column reflects the partial year covered by the snapshot."""
    fig, ax_left = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(r.years))
    width = 0.52

    ax_left.bar(x, r.annual_total, width, color=C_BLUE, alpha=0.72,
                label="All Crypto CVEs", zorder=2)
    ax_left.bar(x, r.annual_shor, width, color=C_ORANGE, alpha=0.90,
                label="Shor-Vulnerable (QF=1.5)", zorder=3)
    ax_left.set_xlabel("Year", fontsize=12)
    ax_left.set_ylabel("Number of CVEs", fontsize=12, color=C_DARK)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(
        [f"{y}{'*' if y == 2026 else ''}" for y in r.years], fontsize=10
    )
    ax_left.set_ylim(0, r.annual_total.max() * 1.25)
    ax_left.tick_params(axis="y", labelcolor=C_DARK)
    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    ax_right = ax_left.twinx()
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_visible(False)
    ax_right.plot(x, r.annual_kev, "o-", color=C_RED,
                  linewidth=2.2, markersize=6,
                  label="CISA KEV Matches", zorder=5)
    ax_right.set_ylabel("CISA KEV Matches per Year", fontsize=12, color=C_RED)
    ax_right.tick_params(axis="y", labelcolor=C_RED)
    ax_right.set_ylim(0, r.annual_kev.max() * 1.45)

    ax_left.annotate(
        "CAGR = 11.0%\n(p = 0.025)",
        xy=(8, r.annual_total[8]),
        xytext=(6.2, r.annual_total[8] + 2400),
        fontsize=9.5, color=C_BLUE,
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.4),
    )
    ax_left.annotate(
        "Shor CAGR = 8.8%\n(p = 0.031)",
        xy=(9, r.annual_shor[9]),
        xytext=(7.0, r.annual_shor[9] + 2400),
        fontsize=9, color=C_ORANGE,
        arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.4),
    )

    handles, labels = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(handles + handles_r, labels + labels_r,
                   loc="upper left", fontsize=9.5,
                   framealpha=0.92, edgecolor="#CCCCCC")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Figure 4 - KEV vs non-KEV QRI distribution
# ===========================================================================
def figure_kev_violin(r: Records, out_path: Path) -> None:
    """Distribution of QRI in CISA KEV vs non-KEV CVEs, with the
    Mann-Whitney U test result annotated."""
    _, p_value, rank_biserial = r.mw_kev
    diff_pct = 100 * (r.kev_qri.mean() - r.non_qri.mean()) / r.non_qri.mean()

    rng = np.random.default_rng(RNG_SEED)
    non_sample = rng.choice(r.non_qri, size=5000, replace=False).tolist()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    parts = ax.violinplot([non_sample, r.kev_qri.tolist()],
                          positions=[1, 2],
                          showmedians=False, showextrema=False, widths=0.65)
    for body, color in zip(parts["bodies"], [C_BLUE, C_ORANGE]):
        body.set_facecolor(color)
        body.set_alpha(0.30)
        body.set_edgecolor(color)
        body.set_linewidth(1.5)

    box = ax.boxplot(
        [r.non_qri.tolist(), r.kev_qri.tolist()],
        positions=[1, 2], widths=0.18,
        patch_artist=True, notch=False,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=1.5, alpha=0.15),
    )
    for patch, color in zip(box["boxes"], [C_BLUE, C_ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    ax.scatter([1, 2], [r.non_qri.mean(), r.kev_qri.mean()],
               marker="D", s=55, color="white",
               edgecolors=C_DARK, linewidth=1.5, zorder=5)
    ax.annotate(
        f"Mean = {r.non_qri.mean():.2f}",
        xy=(1, r.non_qri.mean()), xytext=(1.22, 17.5),
        fontsize=9.5, color=C_BLUE,
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2),
    )
    ax.annotate(
        f"Mean = {r.kev_qri.mean():.2f}",
        xy=(2, r.kev_qri.mean()), xytext=(1.52, 21.5),
        fontsize=9.5, color=C_ORANGE,
        arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.2),
    )

    ax.text(
        1.5, 26.5,
        f"+{diff_pct:.1f}% higher QRI\n"
        f"Mann\u2013Whitney  p = {p_value:.2e}\n"
        f"Effect size r = {rank_biserial:+.2f}",
        ha="center", va="center", fontsize=10, color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=C_LIGHT,
                  edgecolor=C_BLUE, linewidth=1.5),
    )

    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(-0.5, 31)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"Non-KEV\n(n = {len(r.non_qri):,})",
         f"CISA KEV\n(n = {len(r.kev_qri):,})"],
        fontsize=12,
    )
    ax.set_ylabel("Quantum Risk Index (QRI)", fontsize=12)
    ax.legend(
        handles=[
            mpatches.Patch(color=C_BLUE, alpha=0.7, label="Non-KEV CVEs"),
            mpatches.Patch(color=C_ORANGE, alpha=0.7, label="CISA KEV CVEs"),
        ],
        fontsize=10, loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Figure 5 - KEV percentile analysis
# ===========================================================================
def figure_kev_percentile(r: Records, out_path: Path) -> None:
    """Share of CISA KEV CVEs above each percentile of the corpus QRI
    distribution, compared with the rate expected under uniform risk
    (the null model used in Table 6)."""
    percentiles = (50, 75, 90, 95)
    thresholds = [r.qri_sorted[int(p / 100 * r.n)] for p in percentiles]
    observed = [
        100 * (r.kev_qri >= t).sum() / len(r.kev_qri) for t in thresholds
    ]
    expected = [100 - p for p in percentiles]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(percentiles))
    width = 0.34

    bars_obs = ax.bar(x - width / 2, observed, width, color=C_ORANGE,
                      label="Observed (KEV CVEs)", zorder=2)
    bars_exp = ax.bar(x + width / 2, expected, width, color=C_GREY,
                      alpha=0.55, label="Expected (null model)", zorder=2)
    for bar, value in zip(bars_obs, observed):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
                f"{value:.1f}%", ha="center", va="bottom",
                fontsize=10, color=C_ORANGE, fontweight="bold")
    for bar, value in zip(bars_exp, expected):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
                f"{value:.0f}%", ha="center", va="bottom",
                fontsize=10, color=C_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}th percentile\n(QRI \u2265 {t:.1f})"
         for p, t in zip(percentiles, thresholds)],
        fontsize=9.5,
    )
    ax.set_ylabel("% of CISA KEV CVEs Above Threshold", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Figure 6 - KEV rates by Shor vulnerability and HS tier
# ===========================================================================
def figure_kev_rates(r: Records, out_path: Path) -> None:
    """Two-panel chi-square visualisation (Section 4.3).
    Panel (a) compares the CISA KEV rate of Shor-vulnerable (QF = 1.5)
    CVEs with that of all other crypto CVEs; panel (b) breaks the rate
    down by HNDL Score tier (HS = 0, HS = 1-3, HS >= 4), the partition
    used in the body of Section 4.3."""
    is_shor = r.qf == 1.5
    n_shor = int(is_shor.sum())
    n_non_shor = r.n - n_shor
    kev_shor = int(r.in_kev[is_shor].sum())
    kev_non_shor = int(r.in_kev[~is_shor].sum())

    rate_shor = 100 * kev_shor / n_shor
    rate_non_shor = 100 * kev_non_shor / n_non_shor

    chi_shor = stats.chi2_contingency(np.array([
        [kev_shor, n_shor - kev_shor],
        [kev_non_shor, n_non_shor - kev_non_shor],
    ]))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.5))

    bars_a = ax_a.bar(
        ["Non-Shor\nCrypto CVEs\n(QF < 1.5)",
         "Shor-Vulnerable\nCVEs\n(QF = 1.5)"],
        [rate_non_shor, rate_shor],
        color=[C_BLUE, C_ORANGE], width=0.42,
        edgecolor="white", linewidth=1.5, zorder=2,
    )
    for bar, rate in zip(bars_a, [rate_non_shor, rate_shor]):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.018,
                  f"{rate:.3f}%", ha="center", va="bottom",
                  fontsize=11.5, fontweight="bold", color=C_DARK)
    ax_a.set_ylabel("CISA KEV Rate (%)", fontsize=11)
    ax_a.set_ylim(0, max(rate_non_shor, rate_shor) * 1.55)
    ax_a.annotate(
        f"{rate_shor / rate_non_shor:.2f}\u00d7 higher",
        xy=(1, rate_shor), xytext=(0.45, rate_shor + 0.25),
        fontsize=10, color=C_ORANGE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.5),
    )
    ax_a.set_title(
        f"(a) KEV Rate: Shor-Vulnerable vs. Other\n"
        f"\u03c7\u00b2(1) = {chi_shor.statistic:.3f},  "
        f"p = {chi_shor.pvalue:.3f}",
        fontsize=10.5, color=C_DARK, pad=8,
    )

    hs_groups = (
        ("HS = 0\n(No HNDL)",         r.hs == 0,                C_GREY),
        ("HS = 1\u20133\n(Partial)",  (r.hs >= 1) & (r.hs <= 3), C_BLUE),
        ("HS \u2265 4\n(High-Value)", r.hs >= 4,                C_ORANGE),
    )
    table = []
    rates_b = []
    labels_b = []
    colors_b = []
    for label, mask, color in hs_groups:
        n_group = int(mask.sum())
        kev_group = int(r.in_kev[mask].sum())
        rate = 100 * kev_group / n_group
        rates_b.append(rate)
        labels_b.append(f"{label}\nn = {n_group:,}")
        colors_b.append(color)
        table.append([kev_group, n_group - kev_group])

    chi_hs = stats.chi2_contingency(np.array(table))

    bars_b = ax_b.bar(range(len(hs_groups)), rates_b, color=colors_b,
                      width=0.42, edgecolor="white", linewidth=1.5, zorder=2)
    for bar, rate in zip(bars_b, rates_b):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.025,
                  f"{rate:.3f}%", ha="center", va="bottom",
                  fontsize=11.5, fontweight="bold", color=C_DARK)
    ax_b.set_xticks(range(len(hs_groups)))
    ax_b.set_xticklabels(labels_b, fontsize=9.5)
    ax_b.set_ylabel("CISA KEV Rate (%)", fontsize=11)
    ax_b.set_ylim(0, max(rates_b) * 1.45)

    ratio_extremes = rates_b[2] / rates_b[0]
    ax_b.annotate(
        "", xy=(2, rates_b[2]), xytext=(0, rates_b[0]),
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2.2,
                        connectionstyle="arc3,rad=-0.18"),
    )
    ax_b.text(
        1.55, max(rates_b) * 1.18,
        f"HS \u2265 4 is {ratio_extremes:.2f}\u00d7\nthe HS = 0 rate",
        fontsize=9, color=C_GREEN, fontstyle="italic", ha="center",
    )
    p_str = "< 0.001" if chi_hs.pvalue < 1e-3 else f"= {chi_hs.pvalue:.3f}"
    ax_b.set_title(
        f"(b) KEV Rate by HNDL Score Tier\n"
        f"\u03c7\u00b2(2) = {chi_hs.statistic:.3f},  p {p_str}",
        fontsize=10.5, color=C_DARK, pad=8,
    )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Figure 7 - Sensitivity analysis (Tables 7 and 8)
# ===========================================================================
QF_SENSITIVITY: tuple[tuple[str, dict[float, float]], ...] = (
    ("Baseline\n(Shor=1.5,\nGrover=1.2)", {1.0: 1.0, 1.1: 1.1, 1.2: 1.2, 1.5: 1.5}),
    ("Scenario A\n(Shor=1.3,\nGrover=1.1)", {1.0: 1.0, 1.1: 1.05, 1.2: 1.1, 1.5: 1.3}),
    ("Scenario B\n(Shor=1.7,\nGrover=1.3)", {1.0: 1.0, 1.1: 1.15, 1.2: 1.3, 1.5: 1.7}),
    ("Scenario C\n(Shor=2.0,\nGrover=1.5)", {1.0: 1.0, 1.1: 1.20, 1.2: 1.5, 1.5: 2.0}),
    ("Scenario D\n(No QF\nadjustment)",     {1.0: 1.0, 1.1: 1.00, 1.2: 1.0, 1.5: 1.0}),
)

HS_SENSITIVITY: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("Quadratic\n(baseline)\n1+HS\u00b2/10", lambda hs: 1.0 + hs ** 2 / 10.0),
    ("Linear\n1+HS/2",                       lambda hs: 1.0 + hs / 2.0),
    ("Cubic\n1+HS\u00b3/50",                 lambda hs: 1.0 + hs ** 3 / 50.0),
    ("Binary\n2 if HS\u22652",               lambda hs: np.where(hs >= 2, 2.0, 1.0)),
    ("Logarithmic\nlog(HS+2)",               lambda hs: np.log(hs + 2)),
    ("No HS\nterm",                          lambda hs: np.ones_like(hs, dtype=float)),
)


def _kev_diff_pct(qri: np.ndarray, kev_mask: np.ndarray) -> float:
    mean_kev = qri[kev_mask].mean()
    mean_non = qri[~kev_mask].mean()
    return 100 * (mean_kev - mean_non) / mean_non


def figure_sensitivity(r: Records, out_path: Path) -> None:
    """KEV vs non-KEV mean-QRI separation under alternative QF
    parameterisations (panel a) and HS functional forms (panel b).
    Each bar reports the percentage difference in mean QRI between
    CISA KEV and non-KEV CVEs for one configuration."""
    diffs_a = []
    for _, qf_map in QF_SENSITIVITY:
        qf_remapped = np.array([qf_map[v] for v in r.qf])
        qri = r.cvss * qf_remapped * (1.0 + r.hs ** 2 / 10.0)
        diffs_a.append(_kev_diff_pct(qri, r.kev_mask))

    diffs_b = []
    for _, hs_func in HS_SENSITIVITY:
        qri = r.cvss * r.qf * hs_func(r.hs)
        diffs_b.append(_kev_diff_pct(qri, r.kev_mask))

    baseline_diff = diffs_a[0]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.8))

    for ax, diffs, scenarios in ((ax_a, diffs_a, QF_SENSITIVITY),
                                  (ax_b, diffs_b, HS_SENSITIVITY)):
        colors = [C_ORANGE] + [C_BLUE] * (len(diffs) - 2) + [C_GREY]
        bars = ax.bar(range(len(diffs)), diffs, color=colors,
                      edgecolor="white", linewidth=1.2,
                      zorder=2, width=0.55)
        ax.axhline(baseline_diff, color=C_ORANGE, linestyle="--",
                   linewidth=1.5, alpha=0.65,
                   label=f"Baseline: {baseline_diff:.1f}%")
        for bar, value in zip(bars, diffs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.25,
                    f"{value:.1f}%", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=C_DARK)
        ax.set_xticks(range(len(diffs)))
        ax.set_xticklabels([s[0] for s in scenarios], fontsize=8.8)
        ax.set_ylabel("KEV vs. Non-KEV Mean QRI Difference (%)", fontsize=10.5)
        ax.set_ylim(min(diffs) - 3, max(diffs) + 4)
        ax.legend(fontsize=9, loc="upper right")

    ax_a.set_title(
        "(a) Sensitivity to QF Parameterisation\n"
        "All 5 scenarios: p < 10\u207b\u2075\u2070",
        fontsize=10.5, color=C_DARK, pad=8,
    )
    ax_b.set_title(
        "(b) Sensitivity to HS Formula Choice\n"
        "All 6 formulas: p < 10\u207b\u2075\u00b3",
        fontsize=10.5, color=C_DARK, pad=8,
    )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Figure 8 - QRI vs CVSS bar comparison and Lorenz curve
# ===========================================================================
def figure_cvss_versus_qri(r: Records, out_path: Path) -> None:
    """KEV separation under plain CVSS and under QRI (panel a) and the
    Lorenz curve of QRI showing the Pareto concentration (panel b)."""
    u_q, _, _ = r.mw_kev
    auc_qri = u_q / (len(r.kev_qri) * len(r.non_qri))
    u_c, _ = stats.mannwhitneyu(r.kev_cvss, r.non_cvss, alternative="greater")
    auc_cvss = u_c / (len(r.kev_cvss) * len(r.non_cvss))

    kev_means = [r.kev_cvss.mean(), r.kev_qri.mean()]
    non_means = [r.non_cvss.mean(), r.non_qri.mean()]
    diffs = [100 * (k - n) / n for k, n in zip(kev_means, non_means)]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.5))

    x = np.arange(2)
    width = 0.30
    bars_k = ax_a.bar(x - width / 2, kev_means, width, color=C_ORANGE,
                      label="CISA KEV mean", zorder=2)
    bars_n = ax_a.bar(x + width / 2, non_means, width, color=C_BLUE,
                      alpha=0.72, label="Non-KEV mean", zorder=2)
    for bar, value in zip(bars_k, kev_means):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                  f"{value:.2f}", ha="center", fontsize=9.5,
                  color=C_ORANGE, fontweight="bold")
    for bar, value in zip(bars_n, non_means):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                  f"{value:.2f}", ha="center", fontsize=9.5,
                  color=C_BLUE, fontweight="bold")
    for i, diff in enumerate(diffs):
        y_pos = max(kev_means[i], non_means[i]) + 1.5
        ax_a.annotate(
            f"+{diff:.1f}%", xy=(i, y_pos - 0.8), xytext=(i + 0.28, y_pos),
            fontsize=10, color=C_GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.3),
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Plain CVSS", "QRI"], fontsize=11)
    ax_a.set_ylabel("Mean Score Value", fontsize=11)
    ax_a.set_ylim(0, max(kev_means) * 1.4)
    ax_a.legend(fontsize=10, loc="upper left")
    ax_a.set_title(
        f"(a) KEV vs. Non-KEV Score Comparison\n"
        f"ROC-AUC:  CVSS = {auc_cvss:.3f}  |  QRI = {auc_qri:.3f}",
        fontsize=10.5, color=C_DARK, pad=8,
    )

    cum_pop = np.linspace(0, 1, r.n)
    cum_risk = np.cumsum(r.qri_sorted) / r.qri_sorted.sum()
    gini = 1 - 2 * np.trapezoid(cum_risk, cum_pop)

    ax_b.plot(cum_pop, cum_risk, color=C_BLUE, linewidth=2.2,
              label=f"QRI Lorenz curve (Gini = {gini:.3f})")
    ax_b.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.45,
              label="Perfect equality")
    ax_b.fill_between(cum_pop, cum_pop, cum_risk, alpha=0.12, color=C_BLUE)

    idx_80 = int(np.searchsorted(cum_risk, 0.80))
    pareto_x = cum_pop[idx_80]
    ax_b.axvline(pareto_x, color=C_ORANGE, linestyle=":",
                 linewidth=1.6, alpha=0.8)
    ax_b.axhline(0.80, color=C_ORANGE, linestyle=":",
                 linewidth=1.6, alpha=0.8)
    ax_b.scatter([pareto_x], [0.80], color=C_ORANGE, s=60, zorder=5)

    n_top = max(1, int(0.20 * r.n))
    roi = r.qri_sorted[-n_top:].mean() / r.qri_sorted[:-n_top].mean()
    ax_b.text(
        pareto_x + 0.02, 0.68,
        f"Top {pareto_x * 100:.0f}% of CVEs\n"
        f"\u2192 80% of total QRI\n(ROI = {roi:.2f}\u00d7)",
        fontsize=9, color=C_ORANGE,
    )

    ax_b.set_xlabel("Cumulative proportion of CVEs", fontsize=11)
    ax_b.set_ylabel("Cumulative proportion of QRI", fontsize=11)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.legend(fontsize=10, loc="upper left")
    ax_b.set_title(
        f"(b) Lorenz Curve of QRI Distribution\n"
        f"Gini = {gini:.3f};  top 20% score {roi:.2f}\u00d7 higher",
        fontsize=10.5, color=C_DARK, pad=8,
    )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Driver
# ===========================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1 is structural - it does not read the dataset.
    figure_qri_pipeline(OUT_DIR / "fig1_qri_pipeline.png")
    print("Wrote fig1_qri_pipeline.png")

    # Load the corpus once; every subsequent figure shares the same view.
    r = Records(DATASET)
    print(f"Loaded {r.n:,} records from {DATASET.name}.")

    for fn, name in (
        (figure_qf_hs_distribution, "fig2_qf_hs_distribution.png"),
        (figure_annual_growth,      "fig3_annual_growth.png"),
        (figure_kev_violin,         "fig4_kev_vs_nonkev_violin.png"),
        (figure_kev_percentile,     "fig5_kev_percentile.png"),
        (figure_kev_rates,          "fig6_kev_rates_by_tier.png"),
        (figure_sensitivity,        "fig7_sensitivity_analysis.png"),
        (figure_cvss_versus_qri,    "fig8_cvss_vs_qri_lorenz.png"),
    ):
        fn(r, OUT_DIR / name)
        print(f"Wrote {name}")


if __name__ == "__main__":
    main()
