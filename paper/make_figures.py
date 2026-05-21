"""
make_figures.py
===============
Assemble publication composite figures for the JanusDiscover manuscript.

Generates:
  paper/figures/lbdd_results.pdf   -- 2x3 grid: PKA + ROCK2 + FAK1 ROC/enrich
  paper/figures/sbdd_results.pdf   -- 1x3 grid: ABL1 SBDD figures
  paper/figures/workflow_overview.pdf  -- text placeholder panel
  paper/figures/md_results.pdf     -- placeholder (MD not yet run)

Run from repo root:
    python paper/make_figures.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

REPO     = Path(__file__).resolve().parent.parent
FIGS_IN  = REPO / "paper" / "results" / "figures"
FIGS_OUT = REPO / "paper" / "figures"
FIGS_OUT.mkdir(exist_ok=True)


def load_img(path: Path):
    if not path.exists():
        return None
    if PIL_OK:
        return np.array(Image.open(path))
    return None


def panel(ax, img, title, fontsize=10):
    if img is not None:
        ax.imshow(img)
    else:
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, f"{title}\n(figure not generated)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="gray")
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=4)
    ax.axis("off")


# ── Figure 2: LBDD results (PKA / ROCK2 / FAK1, 2 rows × 3 cols) ─────────────

def make_lbdd_figure():
    targets = [
        ("PKA",   "MUV-548", "AUC=0.926, BEDROC=0.743"),
        ("ROCK2", "MUV-644", "AUC=0.923, BEDROC=0.741"),
        ("FAK1",  "MUV-810", "AUC=0.776, BEDROC=0.376"),
    ]
    panels = ["lbdd_roc_curve", "lbdd_enrichment_curve"]
    labels = ["ROC curve", "Enrichment curve"]

    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

    for col, (short, task, stats) in enumerate(targets):
        for row, (fname, label) in enumerate(zip(panels, labels)):
            ax  = fig.add_subplot(gs[row, col])
            img = load_img(FIGS_IN / short / f"{fname}.png")
            title = f"({chr(97+row*3+col)}) {short} ({task}) — {label}"
            panel(ax, img, title, fontsize=9)

    fig.suptitle(
        "JanusDiscover LBDD Performance on MUV Kinase Benchmark\n"
        "(5-fold RF/ECFP4 cross-validation; MUV property-matched decoys)",
        fontsize=11, fontweight="bold", y=1.01
    )
    out = FIGS_OUT / "lbdd_results.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ── Figure 3: SBDD results (ABL1, 1 row × 3 cols) ────────────────────────────

def make_sbdd_figure():
    sbdd_panels = [
        ("sbdd_roc_curve",          "ABL1 SBDD ROC Curve"),
        ("sbdd_score_distribution", "ABL1 Docking Score Distributions"),
        ("sbdd_enrichment_curve",   "ABL1 SBDD Enrichment Curve"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3)

    for ax, (fname, title) in zip(axes, sbdd_panels):
        img = load_img(FIGS_IN / "ABL1_sbdd" / f"{fname}.png")
        panel(ax, img, title, fontsize=10)

    # Annotate with metrics
    fig.text(0.5, -0.02,
             "ABL1 (PDB: 2HYY, 2.20 Å)  |  AUC-ROC = 0.770  |  BEDROC = 0.445  |  "
             "EF₁% = 5.4×  |  Imatinib docking score = −9.65 kcal/mol",
             ha="center", fontsize=9, style="italic")
    fig.suptitle("JanusDiscover SBDD Performance — ABL1 Kinase",
                 fontsize=11, fontweight="bold")
    out = FIGS_OUT / "sbdd_results.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ── Figure 1: Workflow overview (text placeholder) ────────────────────────────

def make_workflow_figure():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    steps = [
        ("1. Target Input",      "User enters protein name\nand disease context"),
        ("2. AI Lit Review",     "Claude summarises literature,\nrecommends mode"),
        ("3. LBDD / SBDD",       "ECFP4+RF (LBDD) or\nAutoDock Vina (SBDD)"),
        ("4. Hybrid Scoring",    "z-score fusion\n0.4·LBDD + 0.6·SBDD"),
        ("5. MD Simulation",     "OpenMM 8 + ff14SB\n10 ns all-atom MD"),
    ]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    n = len(steps)
    for i, ((title, desc), color) in enumerate(zip(steps, colors)):
        x = i / (n - 1)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.08, 0.25), 0.16, 0.50,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=2,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(x, 0.62, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.text(x, 0.40, desc, ha="center", va="center",
                fontsize=8, color="white", transform=ax.transAxes)
        if i < n - 1:
            ax.annotate("", xy=((i+1)/(n-1) - 0.09, 0.50),
                        xytext=(x + 0.09, 0.50),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=2))

    ax.text(0.5, 0.10,
            "Browser Interface  ·  Python 3.12 / FastAPI / React  ·  "
            "MIT License  ·  github.com/patriotsbreeze/JanusDiscover5",
            ha="center", fontsize=8, color="#555555", transform=ax.transAxes)

    fig.suptitle("JanusDiscover — Unified Virtual Screening Workflow",
                 fontsize=13, fontweight="bold", y=0.98)
    out = FIGS_OUT / "workflow_overview.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ── Figure 4: MD results placeholder ─────────────────────────────────────────

def make_md_figure():
    md_panels = [
        "RMSD vs time",
        "Per-residue RMSF",
        "Radius of gyration",
        "Potential energy",
        "SASA vs time",
        "Score distribution",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    for ax, title in zip(axes.flat, md_panels):
        ax.set_facecolor("#f8f8f8")
        ax.text(0.5, 0.5, f"{title}\n(MD simulation pending)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
    fig.suptitle("JanusDiscover MD Simulation — ABL1 Top Hit (10 ns, OpenMM 8)",
                 fontsize=11, fontweight="bold")
    out = FIGS_OUT / "md_results.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating manuscript figures ...")
    make_workflow_figure()
    make_lbdd_figure()
    make_sbdd_figure()
    make_md_figure()
    print("\nAll figures written to", FIGS_OUT)
