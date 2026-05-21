"""
run_sbdd_benchmark.py
=====================
Standalone SBDD benchmark for the JanusDiscover JCIM paper.

Target : ABL1 kinase   PDB 2HYY (imatinib, 2.20 Å)
Protocol:
  1. Download 2HYY from RCSB
  2. Prepare receptor PDBQT
  3. Detect binding site from co-crystal imatinib (HETATM STI)
  4. Redock imatinib -> report RMSD
  5. Dock 20 ChEMBL ABL1 actives + 80 property-matched decoys
  6. Compute AUC-ROC, BEDROC, EF1%, EF5%  from docking scores

Usage:
    python paper/run_sbdd_benchmark.py
"""
from __future__ import annotations

import sys, os, json, time, asyncio, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import requests

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from backend.app.core.sbdd import (
    _pdb_to_pdbqt_receptor,
    _calc_binding_center,
    _compute_redocking_rmsds,
    _dock_ligand,
    _VinaClass,
)
from backend.app.core.lbdd import (
    _validate_smiles,
    bedroc_score,
    enrichment_factor,
    _load_dataset,
    _generate_property_matched_decoys,
)

RESULTS_DIR = Path(__file__).parent / "results"
SBDD_DIR    = RESULTS_DIR / "sbdd"
FIGS_DIR    = RESULTS_DIR / "figures" / "ABL1_sbdd"
for d in (RESULTS_DIR, SBDD_DIR, FIGS_DIR):
    d.mkdir(exist_ok=True)

PDB_ID       = "2HYY"
PDB_RES_ANG  = 2.20          # resolution in Å (from RCSB)
N_ACTIVES    = 20            # ChEMBL ABL1 actives to dock
N_DECOYS     = 80            # property-matched decoys to dock
EXHAUSTIVE   = 4             # Vina exhaustiveness (4 = fast, acceptable for paper)


# ── PDB download ──────────────────────────────────────────────────────────────

def fetch_pdb(pdb_id: str) -> str:
    cache = SBDD_DIR / f"{pdb_id}.pdb"
    if cache.exists():
        print(f"  PDB {pdb_id}: loaded from cache")
        return cache.read_text()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from RCSB ...", end="", flush=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    cache.write_text(r.text)
    print(f" {len(r.text)//1024} KB")
    return r.text


# ── Receptor preparation ──────────────────────────────────────────────────────

def prepare_receptor(pdb_text: str, pdb_id: str):
    """Write PDB to disk and convert to PDBQT. Returns (pdbqt_path, center)."""
    pdb_path  = str(SBDD_DIR / f"{pdb_id}.pdb")
    pdbqt_path = str(SBDD_DIR / f"{pdb_id}_receptor.pdbqt")

    Path(pdb_path).write_text(pdb_text)

    print("  Converting to PDBQT ...", end="", flush=True)
    _pdb_to_pdbqt_receptor(pdb_path, pdbqt_path)
    if not Path(pdbqt_path).exists():
        raise RuntimeError("Receptor PDBQT not created — check Open Babel / fallback")
    print(" done")

    center, method = _calc_binding_center(pdb_text)
    print(f"  Binding centre: {[round(c,2) for c in center]}  ({method})")
    return pdbqt_path, center


# ── Docking enrichment ────────────────────────────────────────────────────────

async def dock_library(smiles_list, labels, receptor_pdbqt, center):
    """Dock all SMILES; return (scores, labels) arrays."""
    scores = []
    valid_labels = []
    t0 = time.time()
    for i, (smi, lbl) in enumerate(zip(smiles_list, labels)):
        score = await _dock_ligand(smi, receptor_pdbqt, center, i)
        if score is not None:
            scores.append(score)          # kcal/mol — more negative = better
            valid_labels.append(lbl)
        elapsed = time.time() - t0
        if (i+1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{len(smiles_list)}] {elapsed:.0f}s elapsed, "
                  f"last score={score}")
    return np.array(scores), np.array(valid_labels)


# ── Enrichment figures ────────────────────────────────────────────────────────

def make_enrichment_figures(y, scores_neg, fig_dir):
    """Generate ROC + enrichment + score distribution figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # For AUC: higher score = active. Vina scores are negative (more negative = better).
    # Invert so that active-favouring scores are high.
    scores = -scores_neg   # now larger = more active

    # ROC
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ABL1 SBDD ROC Curve"); ax.legend()
    fig.savefig(fig_dir / "sbdd_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Score distributions
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(scores_neg[y==1], bins=15, alpha=0.7, label="Actives", color="blue")
    ax.hist(scores_neg[y==0], bins=15, alpha=0.7, label="Decoys",  color="red")
    ax.set_xlabel("Docking score (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("ABL1 SBDD Score Distributions")
    ax.legend()
    fig.savefig(fig_dir / "sbdd_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Enrichment curve
    n = len(y); n_act = int(y.sum())
    order = np.argsort(scores)[::-1]  # descending
    cumact = np.cumsum(y[order])
    fracs  = np.arange(1, n+1) / n
    random_line = fracs * n_act
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fracs*100, cumact, lw=2, label="JanusDiscover SBDD")
    ax.plot(fracs*100, random_line, "k--", lw=1, label="Random")
    ax.set_xlabel("% library screened"); ax.set_ylabel("# actives recovered")
    ax.set_title("ABL1 SBDD Enrichment Curve"); ax.legend()
    fig.savefig(fig_dir / "sbdd_enrichment_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return auc


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("\nJanusDiscover -- SBDD Benchmark")
    print(f"  Target : ABL1 kinase ({PDB_ID}, {PDB_RES_ANG} Ang)")
    print(f"  Actives: {N_ACTIVES} ChEMBL ABL1 | Decoys: {N_DECOYS} | ex={EXHAUSTIVE}\n")

    if _VinaClass is None:
        print("ERROR: Vina not available. Place vina.exe in repo root.")
        return

    t0 = time.time()

    # 1. Fetch PDB
    print("[1/5] Fetching PDB structure ...")
    pdb_text = fetch_pdb(PDB_ID)
    print(f"  {len(pdb_text)//1024} KB, {pdb_text.count('ATOM'):,} ATOM records")

    # 2. Prepare receptor
    print("\n[2/5] Preparing receptor ...")
    receptor_pdbqt, center = prepare_receptor(pdb_text, PDB_ID)
    print(f"  Receptor PDBQT: {Path(receptor_pdbqt).stat().st_size//1024} KB")

    # 3. Redocking RMSD
    print("\n[3/5] Redocking co-crystal ligand (imatinib / STI) ...")
    rmsds = await _compute_redocking_rmsds(pdb_text, receptor_pdbqt, center)
    if rmsds:
        rmsd_mean = float(np.mean(rmsds))
        success   = sum(1 for r in rmsds if r < 2.0)
        print(f"  RMSD values: {[round(r,2) for r in rmsds]}")
        print(f"  Mean RMSD  : {rmsd_mean:.2f} Ang")
        print(f"  Success (<2 Ang): {success}/{len(rmsds)}")
    else:
        rmsd_mean = None
        success   = 0
        print("  WARNING: No RMSD computed (no suitable HETATM / Vina error)")

    # 4. Build screening library
    print("\n[4/5] Building screening library ...")
    actives_cache = RESULTS_DIR / "chembl_cache" / "ABL1_actives.smi"
    if actives_cache.exists():
        all_act_smi = _validate_smiles(actives_cache.read_text().splitlines())
    else:
        print("  No ABL1 cache found — downloading from ChEMBL ...")
        from chembl_webresource_client.new_client import new_client
        acts = new_client.activity.filter(
            target_chembl_id="CHEMBL1862", pchembl_value__gte=5.0
        ).only(["canonical_smiles"])
        smis = [a.get("canonical_smiles","") for a in acts if a.get("canonical_smiles")]
        all_act_smi = _validate_smiles(smis[:200])

    act_smi = all_act_smi[:N_ACTIVES]
    print(f"  {len(act_smi)} actives loaded from ChEMBL ABL1 cache")

    print("  Generating property-matched decoys ...")
    screen = _load_dataset("zinc_250k")
    dec_smi = _generate_property_matched_decoys(act_smi, screen, n_decoys_per_active=N_DECOYS//N_ACTIVES)
    dec_smi = dec_smi[:N_DECOYS]
    print(f"  {len(dec_smi)} decoys (property-matched from ZINC-250k)")

    all_smi    = act_smi + dec_smi
    all_labels = [1]*len(act_smi) + [0]*len(dec_smi)
    print(f"  Total: {len(all_smi)} compounds to dock")

    # 5. Dock + metrics
    print(f"\n[5/5] Docking {len(all_smi)} compounds (ex={EXHAUSTIVE}) ...")
    scores, labels = await dock_library(all_smi, all_labels, receptor_pdbqt, center)

    if len(scores) < 10:
        print("ERROR: Too few docking results to compute metrics")
        return

    # Invert scores: Vina kcal/mol is negative, more-negative = better = higher rank
    scores_inv = -scores
    auc  = float(roc_auc_score(labels, scores_inv))
    bd   = bedroc_score(labels, scores_inv)
    ef1  = enrichment_factor(labels, scores_inv, 0.01)
    ef5  = enrichment_factor(labels, scores_inv, 0.05)
    n_act_screen = int(labels.sum())
    n_tot = len(labels)

    print(f"\n  AUC-ROC : {auc:.4f}")
    print(f"  BEDROC  : {bd:.4f}")
    print(f"  EF  1%  : {ef1:.2f}x  (top-{max(1,n_tot//100)} of {n_tot})")
    print(f"  EF  5%  : {ef5:.2f}x")

    # Figures
    auc_fig = make_enrichment_figures(labels, scores, FIGS_DIR)
    print(f"  Figures -> {FIGS_DIR}")

    elapsed = round(time.time() - t0, 1)

    result = {
        "target":             "ABL1",
        "pdb_id":             PDB_ID,
        "pdb_resolution_ang": PDB_RES_ANG,
        "n_actives_docked":   int(labels.sum()),
        "n_decoys_docked":    int((labels==0).sum()),
        "n_total_docked":     n_tot,
        "redocking_rmsds":    [round(r,3) for r in rmsds] if rmsds else [],
        "redocking_rmsd_mean":round(rmsd_mean,3) if rmsd_mean else None,
        "redocking_success_n":success,
        "redocking_success_rate": round(success/max(1,len(rmsds)),3) if rmsds else None,
        "auc_roc":            round(auc,4),
        "bedroc":             round(bd,4),
        "ef1_percent":        round(ef1,2),
        "ef5_percent":        round(ef5,2),
        "runtime_s":          elapsed,
    }

    out = RESULTS_DIR / "sbdd_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out}")
    print(f"Total time: {elapsed}s")

    # LaTeX table row
    rmsd_str = f"{rmsd_mean:.2f}" if rmsd_mean else "N/A"
    sr_str   = f"{success}/{len(rmsds)}" if rmsds else "N/A"
    print("\n-- Table 2 row (paste into manuscript.tex) --")
    print(f"    ABL1 & {PDB_ID} & {PDB_RES_ANG:.2f} & {rmsd_str} & {sr_str} & "
          f"{auc:.4f} & {ef1:.1f}$\\times$ & {ef5:.1f}$\\times$ \\\\")
    print("---")

    return result


if __name__ == "__main__":
    asyncio.run(main())
