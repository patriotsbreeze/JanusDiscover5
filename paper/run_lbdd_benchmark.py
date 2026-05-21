"""
run_lbdd_benchmark.py  (v4 — MUV kinase benchmark)
===================================================
Standalone LBDD benchmark for the JanusDiscover JCIM paper.

Uses the MUV (Maximum Unbiased Validation) dataset [Rohrer & Baumann, 2009]
downloaded from DeepChem's public S3 bucket.  MUV decoys are deliberately
designed to be topologically similar to actives (same property profile,
different scaffold) so the benchmark is *not* trivially solved by structural
filters — unlike random ZINC decoys.

Kinase targets selected:
  MUV-548  →  PKA  (cAMP-dependent protein kinase catalytic subunit)
  MUV-644  →  ROCK2 (Rho-associated protein kinase 2)
  MUV-810  →  FAK1 (Focal adhesion kinase 1 / PTK2)

Usage (from repo root):
    python paper/run_lbdd_benchmark.py

Requirements:
    pip install rdkit scikit-learn numpy matplotlib seaborn requests
"""

import sys, os, json, time, gzip, io, warnings
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
)

RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR    = RESULTS_DIR / "figures"
CACHE_DIR   = RESULTS_DIR / "muv_cache"
for d in (RESULTS_DIR, FIGS_DIR, CACHE_DIR):
    d.mkdir(exist_ok=True)

MUV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"
MUV_CACHE = CACHE_DIR / "muv.csv.gz"

# MUV kinase tasks  (column_name, short, full_name)
TARGETS = [
    ("MUV-548", "PKA",   "cAMP-dependent protein kinase (PKA)"),
    ("MUV-644", "ROCK2", "Rho-associated protein kinase 2 (ROCK2)"),
    ("MUV-810", "FAK1",  "Focal adhesion kinase 1 (FAK1/PTK2)"),
]

# cap decoys per target (MUV has ~15k decoys per task; cap keeps runtime fast
# while remaining statistically solid)
MAX_DECOYS = 1000


# ── MUV downloader ────────────────────────────────────────────────────────────

def load_muv_dataframe():
    """Download (once) and return MUV as a pandas DataFrame."""
    import pandas as pd

    if not MUV_CACHE.exists():
        import requests
        print("Downloading MUV dataset from DeepChem S3 ...", end="", flush=True)
        r = requests.get(MUV_URL, timeout=120)
        r.raise_for_status()
        MUV_CACHE.write_bytes(r.content)
        print(f" {len(r.content)//1024} KB")
    else:
        print("MUV dataset: loaded from cache")

    with gzip.open(str(MUV_CACHE), "rb") as fh:
        df = pd.read_csv(fh)
    print(f"MUV shape: {df.shape}  (rows x cols)")
    return df


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
    if not vals:
        return 0.0, 0.0, 0.0
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

def run_target(col: str, short: str, full_name: str, df) -> dict:
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  {short}  ({col})  --  {full_name}")
    print(f"{'='*60}")

    # --- extract rows where label is known (not NaN) ---
    sub = df[df[col].notna()].copy()
    sub[col] = sub[col].astype(int)

    actives_df  = sub[sub[col] == 1]
    decoys_df   = sub[sub[col] == 0]

    print(f"  MUV actives  : {len(actives_df)}")
    print(f"  MUV decoys   : {len(decoys_df)} (capped at {MAX_DECOYS})")

    if len(actives_df) < 10:
        return {"target": short, "error": f"Only {len(actives_df)} actives in MUV-{col}"}

    # SMILES extraction & validation
    act_smi_raw = actives_df["smiles"].dropna().tolist()
    dec_smi_raw = decoys_df["smiles"].dropna().sample(
        n=min(MAX_DECOYS, len(decoys_df)), random_state=42
    ).tolist()

    act_smi = _validate_smiles(act_smi_raw)
    dec_smi = _validate_smiles(dec_smi_raw)

    if len(act_smi) < 10:
        return {"target": short, "error": f"Only {len(act_smi)} valid active SMILES"}

    print(f"  Valid actives: {len(act_smi)}")
    print(f"  Valid decoys : {len(dec_smi)}")

    # cache per-target validated SMILES for reproducibility
    (CACHE_DIR / f"{short}_actives.smi").write_text("\n".join(act_smi))
    (CACHE_DIR / f"{short}_decoys.smi").write_text("\n".join(dec_smi))

    # fingerprints
    print("  ECFP4 fingerprints ... ", end="", flush=True)
    X_act   = _smiles_to_fp(act_smi)
    X_inact = _smiles_to_fp(dec_smi)
    X = np.vstack([X_act, X_inact])
    y = np.array([1] * len(act_smi) + [0] * len(dec_smi))
    print(f"done  ({X.shape[0]} x {X.shape[1]})")

    # 5-fold stratified CV
    print("  5-fold stratified CV ... ", end="", flush=True)
    y_prob = run_cv(X, y)
    print("done")

    # metrics
    auc        = float(roc_auc_score(y, y_prob))
    bd, lo, hi = bedroc_bootstrap_ci(y, y_prob)
    ef1        = enrichment_factor(y, y_prob, 0.01)
    ef5        = enrichment_factor(y, y_prob, 0.05)

    n_tot = len(y)
    top1  = max(1, int(n_tot * 0.01))

    print(f"\n  AUC-ROC : {auc:.4f}")
    print(f"  BEDROC  : {bd:.4f}  (95% CI {lo:.4f} - {hi:.4f})")
    print(f"  EF  1%  : {ef1:.2f}x  (top-{top1} of {n_tot})")
    print(f"  EF  5%  : {ef5:.2f}x")

    # figures
    fig_dir = FIGS_DIR / short
    fig_dir.mkdir(exist_ok=True)
    figs = _generate_lbdd_figures(y, y_prob, roc_curve, precision_recall_curve, fig_dir)
    print(f"  Figures : {len(figs)} -> {fig_dir}")

    elapsed = round(time.time() - t0, 1)
    print(f"  Time    : {elapsed}s")

    return {
        "target":        short,
        "target_full":   full_name,
        "muv_col":       col,
        "n_actives":     len(act_smi),
        "n_decoys":      len(dec_smi),
        "n_total":       n_tot,
        "auc_roc":       round(auc, 4),
        "bedroc":        round(bd, 4),
        "bedroc_ci_lo":  round(lo, 4),
        "bedroc_ci_hi":  round(hi, 4),
        "ef1_percent":   round(ef1, 2),
        "ef5_percent":   round(ef5, 2),
        "figures_dir":   str(fig_dir),
        "runtime_s":     elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nJanusDiscover -- LBDD Benchmark (MUV kinase tasks)")
    print("  Targets : PKA (MUV-548), ROCK2 (MUV-644), FAK1 (MUV-810)")
    print("  Decoys  : MUV property-matched topological decoys (cap %d)" % MAX_DECOYS)
    print("  Model   : 500-tree RF | 5-fold stratified CV | 1000 BEDROC bootstraps\n")

    # Note: DUD-E (dude.docking.org) returned HTTP 502 on all download
    # attempts in May 2025 (server backend crash). MUV is used instead;
    # it provides similarly rigorous property-matched decoys.
    print("NOTE: DUD-E server returned HTTP 502 (unavailable); MUV used instead.")
    print("      MUV decoys are property-matched and topologically diverse,")
    print("      providing a rigorous benchmark comparable to DUD-E.\n")

    df = load_muv_dataframe()

    results = []
    for col, short, full_name in TARGETS:
        if col not in df.columns:
            print(f"  SKIP: column {col} not found in MUV CSV")
            results.append({"target": short, "error": "column not found"})
            continue
        try:
            r = run_target(col, short, full_name, df)
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"  ERROR ({short}): {exc}")
            traceback.print_exc()
            results.append({"target": short, "error": str(exc)})

    # save JSON
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
            "    %-6s & %s & %.4f & %.4f (%.4f--%.4f) & %.1f$\\times$ & %.1f$\\times$ \\\\" % (
                r["target"],
                r.get("muv_col", ""),
                r["auc_roc"],
                r["bedroc"], r["bedroc_ci_lo"], r["bedroc_ci_hi"],
                r["ef1_percent"],
                r["ef5_percent"],
            )
        )
    print("---")

    # summary
    good = [r for r in results if "error" not in r]
    if good:
        sizes = ", ".join(
            "%s %da/%dd" % (r["target"], r["n_actives"], r["n_decoys"])
            for r in good
        )
        print("\nDataset sizes: " + sizes)
        mean_auc = sum(r["auc_roc"] for r in good) / len(good)
        mean_bd  = sum(r["bedroc"]  for r in good) / len(good)
        print("Mean AUC-ROC : %.4f" % mean_auc)
        print("Mean BEDROC  : %.4f" % mean_bd)


if __name__ == "__main__":
    main()
