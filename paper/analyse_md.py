"""
analyse_md.py
=============
Standalone MDTraj analysis of the completed ABL1 10 ns trajectory.
Run after run_md_simulation.py has finished.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO   = Path(__file__).resolve().parent.parent
MD_DIR = REPO / "paper" / "results" / "md"
FIGS   = REPO / "paper" / "results" / "figures" / "ABL1_md"
FIGS.mkdir(parents=True, exist_ok=True)

# MDTraj fails on long Windows paths — use short copies in C:\tmp
import shutil as _shutil
_tmp_dcd = r"C:\tmp\prod.dcd"
_tmp_pdb = r"C:\tmp\prod.pdb"
_shutil.copy(str(MD_DIR / "production.dcd"), _tmp_dcd)
_shutil.copy(str(MD_DIR / "production_start.pdb"), _tmp_pdb)
DCD_PATH = _tmp_dcd
PDB_PATH = _tmp_pdb
LOG_PATH = str(MD_DIR / "production.log")
PROD_NS  = 10

BINDING_SITE_RES = {
    248,249,250,251,252,253,254,255,256,257,258,259,
    271,272,273,274,275,276,277,278,279,280,281,
    316,317,318,319,320,321,322,323,324,325,326,
    350,351,352,353,354,355,
    380,381,382,383,384,385,
}

print("Loading trajectory ...")
traj = md.load(DCD_PATH, top=PDB_PATH)
print(f"  {traj.n_frames} frames, {traj.n_atoms} atoms")

prot_idx  = traj.topology.select("protein")
traj_prot = traj.atom_slice(prot_idx)
time_ns   = np.linspace(0, PROD_NS, traj_prot.n_frames)
ref       = traj_prot[0]

# Cα RMSD
print("Computing Ca RMSD ...")
ca_idx   = traj_prot.topology.select("name CA")
traj_ca  = traj_prot.atom_slice(ca_idx)
ref_ca   = ref.atom_slice(ref.topology.select("name CA"))
rmsd_ang = md.rmsd(traj_ca, ref_ca) * 10

# Per-residue RMSF
print("Computing RMSF ...")
rmsf_ang = md.rmsf(traj_ca, ref_ca) * 10
ca_res   = np.array([traj_prot.topology.atom(i).residue.resSeq for i in ca_idx])

# Rg
print("Computing Rg ...")
rg_ang = md.compute_rg(traj_prot) * 10

# Potential energy from log
pot_e, t_log = [], []
try:
    with open(LOG_PATH) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    t_log.append(float(parts[1]) / 1000)
                    pot_e.append(float(parts[2]))
                except ValueError:
                    pass
    print(f"  Energy log: {len(pot_e)} points")
except Exception as e:
    print(f"  Energy log unavailable: {e}")
pot_e = np.array(pot_e); t_log = np.array(t_log)

# SASA
print("Computing SASA ...")
sasa      = md.shrake_rupley(traj_prot, mode="residue")
sasa_ang2 = sasa.sum(axis=1) * 100  # nm² -> Å²

# Binding-site RMSD
print("Computing binding-site RMSD ...")
bs_sel  = " or ".join(f"(name CA and resSeq {r})" for r in sorted(BINDING_SITE_RES))
bs_idx  = traj_prot.topology.select(bs_sel)
if len(bs_idx) > 0:
    traj_bs     = traj_prot.atom_slice(bs_idx)
    ref_bs      = ref.atom_slice(ref.topology.select(bs_sel))
    bs_rmsd_ang = md.rmsd(traj_bs, ref_bs) * 10
    print(f"  Binding-site CA atoms selected: {len(bs_idx)}")
else:
    bs_rmsd_ang = rmsd_ang
    print("  WARNING: no binding-site residues matched; using full-protein RMSD")

# ── Figures ───────────────────────────────────────────────────────────────────
print("Generating figures ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.subplots_adjust(hspace=0.42, wspace=0.35)

# (a) Ca RMSD
ax = axes[0,0]
ax.plot(time_ns, rmsd_ang, lw=1.2, color="#2196F3")
ax.axhline(np.mean(rmsd_ang), ls="--", color="gray", lw=1,
           label=f"Mean {np.mean(rmsd_ang):.2f} Å")
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Cα RMSD (Å)")
ax.set_title("(a) Cα RMSD vs Time", fontweight="bold"); ax.legend(fontsize=8)

# (b) Per-residue RMSF
ax = axes[0,1]
ax.plot(ca_res, rmsf_ang, lw=1.0, color="#4CAF50")
for r in sorted(BINDING_SITE_RES):
    ax.axvline(r, color="red", alpha=0.05, lw=0.5)
ax.set_xlabel("Residue number"); ax.set_ylabel("Cα RMSF (Å)")
ax.set_title("(b) Per-residue RMSF", fontweight="bold")

# (c) Rg
ax = axes[0,2]
ax.plot(time_ns, rg_ang, lw=1.2, color="#9C27B0")
ax.axhline(np.mean(rg_ang), ls="--", color="gray", lw=1,
           label=f"Mean {np.mean(rg_ang):.2f} Å")
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Rg (Å)")
ax.set_title("(c) Radius of Gyration", fontweight="bold"); ax.legend(fontsize=8)

# (d) Potential energy
ax = axes[1,0]
if len(pot_e) > 0:
    ax.plot(t_log, pot_e, lw=1.0, color="#F44336", alpha=0.8)
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Potential Energy (kJ/mol)")
else:
    ax.text(0.5, 0.5, "Energy log not available\n(StateDataReporter file was empty)",
            ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=9)
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Potential Energy (kJ/mol)")
ax.set_title("(d) Potential Energy", fontweight="bold")

# (e) SASA
ax = axes[1,1]
ax.plot(time_ns, sasa_ang2, lw=1.2, color="#FF9800")
ax.axhline(np.mean(sasa_ang2), ls="--", color="gray", lw=1,
           label=f"Mean {np.mean(sasa_ang2):.0f} Å²")
ax.set_xlabel("Time (ns)"); ax.set_ylabel("SASA (Å²)")
ax.set_title("(e) SASA vs Time", fontweight="bold"); ax.legend(fontsize=8)

# (f) Binding-site RMSD
ax = axes[1,2]
ax.plot(time_ns, bs_rmsd_ang, lw=1.2, color="#009688")
ax.axhline(np.mean(bs_rmsd_ang), ls="--", color="gray", lw=1,
           label=f"Mean {np.mean(bs_rmsd_ang):.2f} Å")
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Binding-site Cα RMSD (Å)")
ax.set_title("(f) Binding-site RMSD", fontweight="bold"); ax.legend(fontsize=8)

fig.suptitle(
    "ABL1 Kinase (2HYY chain A) — 10 ns All-Atom MD, ff14SB/TIP3P, 300 K",
    fontsize=12, fontweight="bold"
)
for ext in ("png","pdf"):
    fig.savefig(FIGS / f"md_summary.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)

# Individual panels
panels = [
    ("md_rmsd",    time_ns, rmsd_ang,    "Time (ns)", "Cα RMSD (Å)",        "Cα RMSD vs Time",       "#2196F3"),
    ("md_rmsf",    ca_res,  rmsf_ang,    "Residue",   "Cα RMSF (Å)",        "Per-residue RMSF",      "#4CAF50"),
    ("md_rg",      time_ns, rg_ang,      "Time (ns)", "Rg (Å)",             "Radius of Gyration",    "#9C27B0"),
    ("md_sasa",    time_ns, sasa_ang2,   "Time (ns)", "SASA (Å²)",          "SASA vs Time",          "#FF9800"),
    ("md_bs_rmsd", time_ns, bs_rmsd_ang, "Time (ns)", "Binding-site RMSD (Å)","Binding-site RMSD",  "#009688"),
]
for fname, xd, yd, xl, yl, title, col in panels:
    f2, a2 = plt.subplots(figsize=(5,4))
    a2.plot(xd, yd, lw=1.5, color=col)
    a2.set_xlabel(xl); a2.set_ylabel(yl); a2.set_title(title, fontweight="bold")
    f2.tight_layout()
    f2.savefig(FIGS / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(f2)
if len(pot_e) > 0:
    f2, a2 = plt.subplots(figsize=(5,4))
    a2.plot(t_log, pot_e, lw=1.2, color="#F44336", alpha=0.8)
    a2.set_xlabel("Time (ns)"); a2.set_ylabel("Potential Energy (kJ/mol)")
    a2.set_title("Potential Energy", fontweight="bold")
    f2.tight_layout()
    f2.savefig(FIGS / "md_energy.png", dpi=150, bbox_inches="tight")
    plt.close(f2)

print(f"Figures saved to {FIGS}")

# ── Results JSON ───────────────────────────────────────────────────────────────
result = {
    "target"          : "ABL1",
    "pdb_id"          : "2HYY",
    "protocol"        : "protein-only ff14SB/TIP3P, 300 K/1 atm, GTX 1080 OpenCL",
    "simulation_ns"   : PROD_NS,
    "timestep_fs"     : 2,
    "n_frames"        : int(traj_prot.n_frames),
    "n_protein_atoms" : int(traj_prot.n_atoms),
    "rmsd_mean_ang"   : round(float(rmsd_ang.mean()), 3),
    "rmsd_max_ang"    : round(float(rmsd_ang.max()),  3),
    "rmsf_mean_ang"   : round(float(rmsf_ang.mean()), 3),
    "rmsf_max_ang"    : round(float(rmsf_ang.max()),  3),
    "rg_mean_ang"     : round(float(rg_ang.mean()),   3),
    "rg_std_ang"      : round(float(rg_ang.std()),    3),
    "sasa_mean_ang2"  : round(float(sasa_ang2.mean()), 1),
    "bs_rmsd_mean_ang": round(float(bs_rmsd_ang.mean()), 3),
    "bs_rmsd_max_ang" : round(float(bs_rmsd_ang.max()),  3),
    "n_bs_ca_atoms"   : int(len(bs_idx)),
}
out = REPO / "paper" / "results" / "md_results.json"
out.write_text(json.dumps(result, indent=2))
print(f"\nSaved: {out}")
print(json.dumps(result, indent=2))
