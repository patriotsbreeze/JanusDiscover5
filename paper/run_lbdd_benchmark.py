"""
run_lbdd_benchmark.py  (v3 — ChEMBL with canonical target IDs)
===============================================================
Standalone LBDD benchmark for the JanusDiscover JCIM paper.

Uses ChEMBL directly, queried by canonical target IDs and with a
relaxed pChEMBL threshold (>= 5.0, IC50 <= 10 uM) so that 400-600
actives are obtained per target.  Decoys are drawn property-matched
from the ZINC-250k library.

Targets and ChEMBL IDs:
  ABL1  → CHEMBL1862  (BCR-ABL tyrosine kinase, human)
  EGFR  → CHEMBL203   (Epidermal growth factor receptor, human)
  CDK2  → CHEMBL301   (Cyclin-dependent kinase 2, human)

Usage (from repo root):
    python paper/run_lbdd_benchmark.py

Requirements:
    pip install rdkit chembl-webresource-client scikit-learn numpy matplotlib seaborn
"""

import sys, os, json, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from backend.app.core.lbdd import (
    _smiles_to_fp,
    bedroc_score,
    enrichment_factor,
    _generate_lbdd_figures,
    _validate_smiles,
    _load_dataset,
    _generate_property_matched_decoys,
)

RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR    = RESULTS_DIR / "figures"
CACHE_DIR   = RESULTS_DIR / "chembl_cache"
for d in (RESULTS_DIR, FIGS_DIR, CACHE_DIR):
    d.mkdir(exist_ok=True)

# ── Target definitions ────────────────────────────────────────────────────────

TARGETS = [
    {
        "name":       "ABL1 kinase",
        "short":      "ABL1",
        "chembl_id":  "CHEMBL1862",          # BCR-ABL, human
        "uniprot":    "P00519",
    },
    {
        "name":       "EGFR",
        "short":      "EGFR",
        "chembl_id":  "CHEMBL203",           # EGFR, human
        "uniprot":    "P00533",
    },
    {
        "name":       "CDK2",
        "short":      "CDK2",
        "chembl_id":  "CHEMBL301",           # CDK2, human
        "uniprot":    "P24941",
    },
]

MAX_ACTIVES = 500          # cap per target
PCHEMBL_MIN = 5.0          # IC50 / Ki <= 10 uM
DECOY_RATIO = 5            # decoys per active (DUD-E convention)


# ── ChEMBL active fetcher ─────────────────────────────────────────────────────

def fetch_actives(chembl_id: str, short: str, max_n: int = MAX_ACTIVES) -> list[str]:
    """Fetch unique, validated SMILES for actives with pChEMBL >= PCHEMBL_MIN."""
    cache = CACHE_DIR / f"{short}_actives.smi"
    if cache.exists():
        smiles = cache.read_text().splitlines()
        print(f"    Loaded {len(smiles)} actives from cache")
        return _validate_smiles(smiles)

    from chembl_webresource_client.new_client import new_client
    print(f"    Querying ChEMBL ({chembl_id}, pChEMBL >= {PCHEMBL_MIN}) ...", end="", flush=True)

    activities = new_client.activity.filter(
        target_chembl_id=chembl_id,
        pchembl_value__gte=PCHEMBL_MIN,
    ).only(["canonical_smiles", "pchembl_value"])

    seen: set[str] = set()
    smiles: list[str] = []
    for act in activities:
        smi = (act.get("canonical_smiles") or "").strip()
        if smi and smi not in seen:
            seen.add(smi)
            smiles.append(smi)
        if len(smiles) >= max_n:
            break

    valid = _validate_smiles(smiles)
    cache.write_text("\n".join(valid))
    print(f" {len(valid)} valid SMILES")
    return valid


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bedroc_bootstrap_ci(y, scores, n_boot: int = 1000, alpha: float = 20.0):
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            vals.append(bedroc_score(y[idx], scores[idx], alpha))
        except Exception:
            pass
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return float(np.mean(vals)), lo, hi


# ── 5-fold CV ─────────────────────────────────────────────────────────────────

def run_cv(X, y, n_splits: int = 5):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    clf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]


# ── Per-target run ────────────────────────────────────────────────────────────

def run_target(target: dict, screen_smiles: list[str]) -> dict:
    short = target["short"]
    t0    = time.time()

    print(f"\n{'='*60}")
    print(f"  {short}  (ChEMBL: {target['chembl_id']})")
    print(f"{'='*60}")

    # 1. Actives
    actives = fetch_actives(target["chembl_id"], short)
    if len(actives) < 20:
        return {
            "target": short,
            "error": f"Only {len(actives)} actives — ChEMBL query may have failed",
        }
    print(f"  Actives : {len(actives)}")

    # 2. Property-matched decoys from ZINC
    decoys = _generate_property_matched_decoys(
        actives, screen_smiles, n_decoys_per_active=DECOY_RATIO
    )
    print(f"  Decoys  : {len(decoys)} (ratio {len(decoys)/len(actives):.1f}:1)")

    # 3. Fingerprints
    print("  ECFP4 fingerprints ... ", end="", flush=True)
    X_act   = _smiles_to_fp(actives)
    X_inact = _smiles_to_fp(decoys)
    X = np.vstack([X_act, X_inact])
    y = np.array([1] * len(actives) + [0] * len(decoys))
    print(f"done  ({X.shape[0]} x {X.shape[1]})")

    # 4. 5-fold CV
    print("  5-fold stratified CV ... ", end="", flush=True)
    y_prob = run_cv(X, y)
    print("done")

    # 5. Metrics
    auc         = float(roc_auc_score(y, y_prob))
    bd, lo, hi  = bedroc_bootstrap_ci(y, y_prob)
    ef1         = enrichment_factor(y, y_prob, 0.01)
    ef5         = enrichment_factor(y, y_prob, 0.05)

    n_tot = len(y)
    top1  = max(1, int(n_tot * 0.01))

    print(f"\n  AUC-ROC : {auc:.4f}")
    print(f"  BEDROC  : {bd:.4f}  (95% CI {lo:.4f} - {hi:.4f})")
    print(f"  EF  1%  : {ef1:.2f}x  (top-{top1} of {n_tot})")
    print(f"  EF  5%  : {ef5:.2f}x")

    # 6. Sanity check
    if auc > 0.98:
        print(f"  NOTE    : AUC={auc:.4f} is high; inspect score distributions")
        print(f"            This may indicate decoys are not structurally challenging.")
        print(f"            Consider DUD-E decoys for external-validation benchmarking.")

    # 7. Figures
    fig_dir = FIGS_DIR / short
    fig_dir.mkdir(exist_ok=True)
    figs = _generate_lbdd_figures(y, y_prob, roc_curve, precision_recall_curve, fig_dir)
    print(f"  Figures : {len(figs)} -> {fig_dir}")

    elapsed = round(time.time() - t0, 1)
    print(f"  Time    : {elapsed}s")

    return {
        "target":       short,
        "target_full":  target["name"],
        "chembl_id":    target["chembl_id"],
        "n_actives":    len(actives),
        "n_decoys":     len(decoys),
        "n_total":      n_tot,
        "auc_roc":      round(auc, 4),
        "bedroc":       round(bd, 4),
        "bedroc_ci_lo": round(lo, 4),
        "bedroc_ci_hi": round(hi, 4),
        "ef1_percent":  round(ef1, 2),
        "ef5_percent":  round(ef5, 2),
        "figures_dir":  str(fig_dir),
        "runtime_s":    elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nJanusDiscover -- LBDD Benchmark (ChEMBL + ZINC-250k decoys)")
    print("  Targets: ABL1 (CHEMBL1862), EGFR (CHEMBL203), CDK2 (CHEMBL301)")
    print("  pChEMBL >= 5.0 | 500-tree RF | 5-fold CV | 1000 BEDROC bootstraps\n")

    # Load ZINC-250k once (shared decoy pool)
    print("Loading ZINC-250k screening library ...")
    screen_smiles = _load_dataset("zinc_250k")
    print(f"Library: {len(screen_smiles)} valid SMILES\n")

    results = []
    for target in TARGETS:
        try:
            r = run_target(target, screen_smiles)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR ({target['short']}): {exc}")
            results.append({"target": target["short"], "error": str(exc)})

    # Save
    out = RESULTS_DIR / "lbdd_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    # LaTeX table rows
    print("\n-- Table 1 rows (paste into manuscript.tex) --")
    for r in results:
        if "error" in r:
            print(f"  {r['target']}: ERROR -- {r['error']}")
            continue
        print(
            f"    {r['target']:6s} & {r['auc_roc']:.4f} & "
            f"{r['bedroc']:.4f} ({r['bedroc_ci_lo']:.4f}--{r['bedroc_ci_hi']:.4f}) & "
            f"{r['ef1_percent']:.1f}$\\times$ & "
            f"{r['ef5_percent']:.1f}$\\times$ \\\\"
        )
    print("---")

    # Summary
    good = [r for r in results if "error" not in r]
    if good:
        sizes = ", ".join("%s %da/%dd" % (r['target'], r['n_actives'], r['n_decoys']) for r in good)
        print("\nDataset sizes: " + sizes)
        mean_auc = sum(r['auc_roc'] for r in good) / len(good)
        print(f"Mean AUC-ROC: {mean_auc:.4f}")


if __name__ == "__main__":
    main()
