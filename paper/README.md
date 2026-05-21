# JanusDiscover — Paper Directory

This directory contains the complete JCIM submission package.

---

## Files

| File | Purpose |
|---|---|
| `manuscript.tex` | Main LaTeX source (achemso, journal=jcisd8) |
| `manuscript.bib` | BibTeX bibliography (37 real peer-reviewed references) |
| `figures/` | Place generated PDFs/PNGs here before compiling |
| `README.md` | This file |

---

## Compile

```bash
pdflatex manuscript
bibtex   manuscript
pdflatex manuscript
pdflatex manuscript
```

Requires:
- TeX Live 2022+ or MiKTeX 22+
- `achemso` package (CTAN, included in TeX Live)
- `soul`, `siunitx`, `booktabs`, `algorithm`, `algpseudocode` (standard)

---

## How to Fill Every `\placeholder{...}`

Every yellow-highlighted `\placeholder{...}` in the compiled PDF must be
replaced with a real value before submission.  The table below maps each
placeholder to the JanusDiscover command that generates it.

### Step 1 — Run the platform in real mode on three targets

```bash
# Start the backend in real mode
export ANTHROPIC_API_KEY=<your_key>
uvicorn backend.app.main:app --reload --port 8000

# In a second terminal
cd frontend && npm run dev
```

Then, in the browser at http://localhost:3000, run the full workflow
(lit review → approve → discover → MD) for each target:

| Target | Recommended PDB | Dataset |
|---|---|---|
| ABL1 kinase | 2HYY (imatinib, 2.20 Å) | zinc_250k |
| EGFR | 1IVO (erlotinib, 2.60 Å) | zinc_250k |
| CDK2 | 1JST (roscovitine, 2.00 Å) | zinc_250k |

### Step 2 — Collect LBDD metrics → Table 1

From the JSON returned by `GET /api/session/{id}` → `discovery_result.validation_metrics`:

```
auc_roc           → \placeholder{0.XX} in Table 1 row
bedroc            → \placeholder{0.XX} (BEDROC value)
bedroc_ci_lo/hi   → compute via bootstrap (see below)
ef1_percent       → \placeholder{X.X}× (EF1%)
ef5_percent       → \placeholder{X.X}× (EF5%)
```

**BEDROC 95% CI** — run after collecting out-of-fold scores:
```python
from backend.app.core.lbdd import bedroc_score
import numpy as np
rng = np.random.default_rng(0)
ci = [bedroc_score(y[rng.choice(len(y), len(y))],
                   s[rng.choice(len(s), len(s))]) for _ in range(1000)]
lo, hi = np.percentile(ci, [2.5, 97.5])
```

### Step 3 — Collect SBDD metrics → Table 2

From `discovery_result.validation_metrics`:
```
pose_rmsd_mean          → redocking RMSD (Å)
pose_success_rate_2A    → success rate (multiply by 100 for %)
auc_roc                 → SBDD AUC-ROC
ef1_percent, ef5_percent
```

PDB ID and resolution: read from `lit_review.pdb_ids[0]` and look up at
https://www.rcsb.org.

### Step 4 — MD metrics (Section 3.5)

From `md_result`:
```
rmsd_mean_nm * 10    → RMSD mean in Å
rmsd_std_nm  * 10    → RMSD std  in Å
```

For per-residue RMSF peaks, open the `md_rmsf.png` figure and read off
the residue ranges for the P-loop and activation loop annotations.

### Step 5 — Timing (Table 3)

Record wall-clock time for each stage from `run_time_seconds` fields in
the session JSON, plus GPU runtime for the MD simulation (OpenMM logs it).

### Step 6 — Figures

Copy the generated PNG files from `static/figures/<session_id>/` into
`paper/figures/` and convert to PDF for vector quality:

```bash
# Using ImageMagick
convert lbdd_roc_curve.png -density 300 lbdd_roc_curve.pdf

# Or use matplotlib's savefig with format='pdf' in the source
```

Required figure files (match the \includegraphics paths in manuscript.tex):
- `figures/workflow_overview.pdf` — create a schematic diagram manually
  (PowerPoint → Export as PDF, or use draw.io / Inkscape)
- `figures/lbdd_results.pdf` — composite of the 4 LBDD figures for ABL1
- `figures/sbdd_results.pdf` — composite of the 5 SBDD figures for ABL1
- `figures/md_results.pdf`   — composite of the 6 MD figures for ABL1

**Compositing with Python:**
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

# Example for lbdd_results.pdf (2×2 grid)
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig)
panels = [
    ("lbdd_roc_curve.png",        "(a) ROC Curve"),
    ("lbdd_enrichment_curve.png", "(b) Enrichment Curve"),
    ("lbdd_score_distribution.png","(c) Score Distributions"),
    ("lbdd_metrics_summary.png",  "(d) Metrics Summary"),
]
for idx, (fname, label) in enumerate(panels):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    img = np.array(Image.open(f"static/figures/<SESSION_ID>/{fname}"))
    ax.imshow(img)
    ax.set_title(label, fontweight="bold")
    ax.axis("off")
fig.savefig("paper/figures/lbdd_results.pdf", bbox_inches="tight", dpi=300)
```

---

## JCIM Submission Checklist

Before uploading to ACS Paragon Plus:

- [ ] All `\placeholder{...}` replaced with real values
- [ ] `\placeholder` command and `\usepackage{soul}` removed from manuscript.tex
- [ ] Figures at ≥ 300 DPI, EPS or TIFF preferred by ACS (PDF accepted)
- [ ] `workflow_overview.pdf` created (hand-drawn schematic of the UI workflow)
- [ ] Supporting Information PDF prepared (Figures S1–S8, Tables S1–S2)
- [ ] Cover letter written explaining novelty vs. AutoDock-GPU, PyRMD, etc.
- [ ] Author ORCID added to achemso author block
- [ ] Funding statement filled in `\begin{acknowledgement}`
- [ ] Competing interests statement confirmed (none if applicable)
- [ ] Code availability: GitHub URL + version tag (create a GitHub Release)
- [ ] Data availability: ChEMBL version, PDB accession codes listed
- [ ] `manuscript.bib` deduplicated (harvey2019htmd and doerr2017htmd are
      the same paper — remove one before submission)

---

## JCIM Author Guidelines (key points)

- Word limit: ~8000 words for a Full Paper (Methods + Results + Discussion)
- Abstract: ≤ 150 words
- References: ACS numbered style, auto-formatted by achemso
- Graphical abstract: 8.5 × 4.75 cm image required at submission
- TOC graphic: same image used for Table of Contents
- Data and code must be publicly archived (GitHub is accepted)

---

## Quick Citation

```
Loalt, B. JanusDiscover: A Unified Web Platform for Ligand-Based and
Structure-Based Virtual Screening with Integrated Molecular Dynamics
Validation. J. Chem. Inf. Model. 2025, XX, XXXX–XXXX.
https://doi.org/10.1021/acs.jcim.XXXXXXX
```
