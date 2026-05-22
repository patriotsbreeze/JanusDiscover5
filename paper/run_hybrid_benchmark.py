"""
run_hybrid_benchmark.py
=======================
Hybrid consensus screening demo for ABL1 kinase.

Protocol:
  1. Train LBDD model on ChEMBL ABL1 actives (pChEMBL >= 6.0)
  2. Screen 200 ZINC-250k compounds with LBDD (predicted probability)
  3. Dock the same 200 compounds against 2HYY with AutoDock Vina
  4. Combine: s_hybrid = 0.40 * z_LBDD + 0.60 * z_SBDD
  5. Report top-20 hybrid hits with Bemis-Murcko scaffold diversity

Usage:
    python paper/run_hybrid_benchmark.py
"""
from __future__ import annotations
import sys, os, json, asyncio, warnings, time
from pathlib import Path

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from backend.app.core.lbdd import _validate_smiles, _load_dataset
from backend.app.core.sbdd import _dock_ligand, _VinaClass

RESULTS_DIR = Path(__file__).parent / "results"
HYBRID_DIR  = RESULTS_DIR / "hybrid"
FIGS_DIR    = RESULTS_DIR / "figures" / "ABL1_hybrid"
for d in (HYBRID_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHEMBL_CACHE  = RESULTS_DIR / "chembl_cache" / "ABL1_actives.smi"
RECEPTOR_PDBQT = str(RESULTS_DIR / "sbdd" / "2HYY_receptor.pdbqt")
CENTER         = [14.25, 15.28, 17.63]   # chain-A STI centroid
N_SCREEN       = 200                      # ZINC compounds to screen
EXHAUSTIVENESS = 4
N_TOP          = 20

print("\nJanusDiscover - Hybrid Consensus Screening (ABL1)")
print(f"  Screen: {N_SCREEN} ZINC-250k compounds")
print(f"  Receptor: 2HYY chain A | Center: {CENTER}")
print(f"  Hybrid weights: 0.40 LBDD + 0.60 SBDD\n")

# ── 1. Load ABL1 actives from ChEMBL cache ────────────────────────────────────
print("[1/5] Loading ChEMBL ABL1 actives ...")
if not CHEMBL_CACHE.exists():
    raise FileNotFoundError(f"Run run_sbdd_benchmark.py first to populate {CHEMBL_CACHE}")

actives_raw = CHEMBL_CACHE.read_text().splitlines()
actives_smi = _validate_smiles(actives_raw)[:100]   # up to 100 actives
print(f"  {len(actives_smi)} validated actives")

# ── 2. Load screening library (ZINC-250k) ────────────────────────────────────
print(f"\n[2/5] Loading {N_SCREEN} ZINC-250k screening compounds ...")
zinc = _load_dataset("zinc_250k")
np.random.seed(42)
idx  = np.random.choice(len(zinc), size=min(N_SCREEN * 3, len(zinc)), replace=False)
screen_smi_all = [zinc[i] for i in idx]
screen_smi_all = _validate_smiles(screen_smi_all)
screen_smi     = screen_smi_all[:N_SCREEN]
print(f"  {len(screen_smi)} valid ZINC compounds selected")

# ── 3. LBDD: fingerprints + RF classifier ────────────────────────────────────
print("\n[3/5] LBDD: training RF on ABL1 actives + ZINC decoys ...")

def smi_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

# Training set: actives (label=1) + ZINC negatives (label=0)
n_neg   = min(len(actives_smi) * 5, N_SCREEN)
neg_smi = screen_smi[:n_neg]

all_train = actives_smi + neg_smi
all_y     = [1] * len(actives_smi) + [0] * len(neg_smi)

fps   = [smi_to_fp(s) for s in all_train]
valid = [(fp, y) for fp, y in zip(fps, all_y) if fp is not None]
X_tr  = np.array([v[0] for v in valid])
y_tr  = np.array([v[1] for v in valid])

clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                              random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)

# Score all screening compounds
screen_fps    = [smi_to_fp(s) for s in screen_smi]
screen_valid  = [(i, fp) for i, fp in enumerate(screen_fps) if fp is not None]
screen_idx    = [v[0] for v in screen_valid]
screen_fp_arr = np.array([v[1] for v in screen_valid])
lbdd_probs_valid = clf.predict_proba(screen_fp_arr)[:, 1]

lbdd_probs = np.full(len(screen_smi), np.nan)
for idx_orig, prob in zip(screen_idx, lbdd_probs_valid):
    lbdd_probs[idx_orig] = prob

print(f"  LBDD scored {len(screen_valid)}/{len(screen_smi)} compounds")
print(f"  Prob range: {np.nanmin(lbdd_probs):.3f} – {np.nanmax(lbdd_probs):.3f}")

# ── 4. SBDD: dock all screening compounds ────────────────────────────────────
print(f"\n[4/5] SBDD: docking {len(screen_smi)} compounds ...")

if _VinaClass is None:
    raise RuntimeError("Vina not available — place vina.exe in repo root")

SBDD_CACHE = HYBRID_DIR / "sbdd_cache.json"

async def dock_all():
    scores = []
    t0 = time.time()
    for i, smi in enumerate(screen_smi):
        sc = await _dock_ligand(smi, RECEPTOR_PDBQT, CENTER, i)
        scores.append(sc)  # None if failed
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(screen_smi)}]  {time.time()-t0:.0f}s  last={sc}")
    return scores

if SBDD_CACHE.exists():
    print(f"  Loading cached SBDD scores from {SBDD_CACHE}")
    sbdd_raw = json.loads(SBDD_CACHE.read_text())
else:
    sbdd_raw = asyncio.run(dock_all())
    SBDD_CACHE.write_text(json.dumps(sbdd_raw))
    print(f"  SBDD scores cached to {SBDD_CACHE}")
sbdd_scores = np.array([s if s is not None else np.nan for s in sbdd_raw])
n_docked = int(np.sum(~np.isnan(sbdd_scores)))
print(f"  Docked {n_docked}/{len(screen_smi)} compounds")
print(f"  Score range: {np.nanmin(sbdd_scores):.2f} – {np.nanmax(sbdd_scores):.2f} kcal/mol")

# ── 5. Hybrid consensus ───────────────────────────────────────────────────────
print("\n[5/5] Computing hybrid consensus scores ...")

# Keep only compounds scored by both pipelines
valid_mask = ~np.isnan(lbdd_probs) & ~np.isnan(sbdd_scores)
n_valid    = int(valid_mask.sum())
print(f"  {n_valid} compounds with both LBDD and SBDD scores")

lbdd_v = lbdd_probs[valid_mask]
sbdd_v = sbdd_scores[valid_mask]   # negative = better
smi_v  = [screen_smi[i] for i in np.where(valid_mask)[0]]

# Z-score normalise (flip SBDD so higher = better)
z_lbdd = (lbdd_v - lbdd_v.mean()) / (lbdd_v.std() + 1e-9)
z_sbdd = ((-sbdd_v) - (-sbdd_v).mean()) / ((-sbdd_v).std() + 1e-9)
s_hybrid = 0.40 * z_lbdd + 0.60 * z_sbdd

# Top-20 hits
top_idx   = np.argsort(s_hybrid)[::-1][:N_TOP]
top_smi   = [smi_v[i] for i in top_idx]
top_lbdd  = lbdd_v[top_idx]
top_sbdd  = sbdd_v[top_idx]
top_hybrid= s_hybrid[top_idx]

# Compare LBDD top-20 and SBDD top-20 separately
lbdd_top20_set = set(np.argsort(lbdd_v)[::-1][:N_TOP])
sbdd_top20_set = set(np.argsort(-sbdd_v)[::-1][:N_TOP])   # more-negative = better
hybrid_top20_set = set(top_idx)
overlap = lbdd_top20_set & sbdd_top20_set
hybrid_unique = hybrid_top20_set - lbdd_top20_set - sbdd_top20_set

print(f"\n  Top-20 overlap (LBDD & SBDD): {len(overlap)}")
print(f"  Hybrid-unique hits: {len(hybrid_unique)}")
print(f"\n  Top-5 hybrid hits:")
for rank, (smi, lp, ss, hs) in enumerate(zip(top_smi[:5], top_lbdd[:5],
                                               top_sbdd[:5], top_hybrid[:5]), 1):
    print(f"    #{rank}: LBDD={lp:.3f}  SBDD={ss:.2f} kcal/mol  hybrid={hs:.3f}")
    print(f"         SMILES: {smi[:80]}")

# Bemis-Murcko scaffold diversity
scaffolds = set()
for smi in top_smi:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.add(Chem.MolToSmiles(sc))
        except Exception:
            pass
n_scaffolds = len(scaffolds)
print(f"\n  Distinct Bemis-Murcko scaffolds in top-20: {n_scaffolds}")

# ── Figures ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Score scatter: LBDD prob vs SBDD score (coloured by hybrid rank)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(wspace=0.35)

ax = axes[0]
sc = ax.scatter(lbdd_v, sbdd_v, c=s_hybrid, cmap="RdYlGn", s=30, alpha=0.7)
ax.scatter(top_lbdd, top_sbdd, s=80, edgecolors="black", facecolors="none",
           linewidths=1.5, label="Top-20 hybrid")
ax.set_xlabel("LBDD probability"); ax.set_ylabel("SBDD score (kcal/mol)")
ax.set_title("(a) LBDD vs SBDD scores\n(colour = hybrid score)", fontweight="bold")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="Hybrid z-score")

ax = axes[1]
ax.bar(range(N_TOP), top_hybrid, color="#2196F3")
ax.set_xlabel("Hybrid rank"); ax.set_ylabel("Hybrid z-score")
ax.set_title("(b) Top-20 Hybrid Scores", fontweight="bold")
ax.set_xticks(range(0, N_TOP, 5)); ax.set_xticklabels(range(1, N_TOP+1, 5))

ax = axes[2]
cats = ["LBDD\ntop-20", "SBDD\ntop-20", "Hybrid\ntop-20", "Hybrid\nunique"]
vals = [len(lbdd_top20_set), len(sbdd_top20_set), N_TOP, len(hybrid_unique)]
cols = ["#4CAF50", "#F44336", "#2196F3", "#9C27B0"]
ax.bar(cats, vals, color=cols, alpha=0.8)
ax.set_ylabel("Count")
ax.set_title("(c) Top-20 Composition", fontweight="bold")
for i, v in enumerate(vals):
    ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")

fig.suptitle(
    f"ABL1 Hybrid Consensus Screening  |  {n_valid} compounds  |  "
    f"Overlap(LBDD&SBDD)={len(overlap)}  |  {n_scaffolds} scaffolds in top-20",
    fontsize=10, fontweight="bold"
)
for ext in ("png", "pdf"):
    fig.savefig(FIGS_DIR / f"hybrid_results.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Figure -> {FIGS_DIR / 'hybrid_results.png'}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
top_hits = []
for rank, (smi, lp, ss, hs) in enumerate(zip(top_smi, top_lbdd, top_sbdd, top_hybrid), 1):
    top_hits.append({
        "rank":         rank,
        "smiles":       smi,
        "lbdd_prob":    round(float(lp), 4),
        "sbdd_score":   round(float(ss), 3),
        "hybrid_score": round(float(hs), 4),
    })

result = {
    "target":           "ABL1",
    "n_screen":         len(screen_smi),
    "n_docked":         n_docked,
    "n_valid":          n_valid,
    "lbdd_actives":     len(actives_smi),
    "lbdd_negatives":   len(neg_smi),
    "n_top":            N_TOP,
    "overlap_lbdd_sbdd":len(overlap),
    "hybrid_unique":    len(hybrid_unique),
    "n_scaffolds_top20":n_scaffolds,
    "top_hit_smiles":   top_smi[0],
    "top_hit_lbdd_prob":round(float(top_lbdd[0]), 4),
    "top_hit_sbdd":     round(float(top_sbdd[0]), 3),
    "top_hits":         top_hits,
}
out = RESULTS_DIR / "hybrid_results.json"
out.write_text(json.dumps(result, indent=2))
print(f"  Saved: {out}")
print("\n-- Section 3.4 values --")
print(f"  Overlap (LBDD ∩ SBDD top-20): {len(overlap)}")
print(f"  Hybrid-unique entries: {len(hybrid_unique)}")
print(f"  Top hit SMILES: {top_smi[0]}")
print(f"  Top hit LBDD prob: {top_lbdd[0]:.3f}")
print(f"  Top hit SBDD score: {top_sbdd[0]:.2f} kcal/mol")
print(f"  Distinct scaffolds in top-20: {n_scaffolds}")
