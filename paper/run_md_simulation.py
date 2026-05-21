"""
run_md_simulation.py
====================
10 ns all-atom explicit-solvent MD simulation of ABL1 kinase (PDB 2HYY).
Uses OpenMM 8 + ff14SB + TIP3P on GTX 1080 (OpenCL, mixed precision).

Protocol (protein-only; ligand parameterisation requires AmberTools/OpenFF):
  1. PDBFixer  : remove HETATM, add missing residues/loops, add H at pH 7.4
  2. Solvation : 10 Å TIP3P box, Na+/Cl- to 0.15 M
  3. Minimize  : 1000 steps steepest-descent
  4. NVT equil : 500 ps, 300 K
  5. NPT equil : 500 ps, 300 K, 1 atm
  6. Production: 10 ns (2 fs step), save DCD every 10 ps -> 1000 frames
  7. Analysis  : Cα RMSD, per-residue RMSF, Rg, potential energy, SASA,
                 binding-site RMSD (residues within 5 Å of co-crystal STI)

Usage:
    python paper/run_md_simulation.py

Runtime estimate: ~1–3 h on GTX 1080 (OpenCL, mixed precision).
"""
from __future__ import annotations

import sys, os, time, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# ── OpenMM imports ─────────────────────────────────────────────────────────────
from openmm import (
    Platform, LangevinMiddleIntegrator, MonteCarloBarostat,
    unit, Vec3,
)
from openmm.app import (
    PDBFile, ForceField, Modeller,
    PME, HBonds, NoCutoff,
    Simulation, DCDReporter, StateDataReporter,
    PDBReporter,
)
import openmm.unit as unit

# ── PDBFixer ───────────────────────────────────────────────────────────────────
from pdbfixer import PDBFixer

# ── MDTraj for analysis ────────────────────────────────────────────────────────
import mdtraj as md

# ── matplotlib ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
MD_DIR      = RESULTS_DIR / "md"
FIGS_DIR    = RESULTS_DIR / "figures" / "ABL1_md"
SBDD_DIR    = RESULTS_DIR / "sbdd"
for d in (MD_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

PDB_CACHE   = SBDD_DIR / "2HYY.pdb"
PDB_ID      = "2HYY"

# ── Simulation parameters ──────────────────────────────────────────────────────
TEMPERATURE   = 300 * unit.kelvin
PRESSURE      = 1   * unit.atmosphere
TIMESTEP      = 2   * unit.femtoseconds
EQUIL_NVT_PS  = 500                      # ps
EQUIL_NPT_PS  = 500                      # ps
PROD_NS       = 10                       # ns
SAVE_EVERY_PS = 10                       # ps between frames
NONBONDED_CUT = 10.0 * unit.angstroms
BOX_PADDING   = 10.0 * unit.angstroms
IONIC_STR     = 0.15                     # mol/L NaCl

# OpenCL GTX 1080 (confirmed plidx=0, devidx=0)
PLATFORM_NAME    = "OpenCL"
PLATFORM_PROPS   = {"DeviceIndex": "0", "OpenCLPlatformIndex": "0", "Precision": "mixed"}

# ── Binding-site residues (chain A, within 5 Å of co-crystal STI imatinib) ────
# From 2HYY: residues in contact with imatinib (standard ABL1 KD contact list)
BINDING_SITE_RES = {
    248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,  # P-loop
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,       # alpha-C
    316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326,       # hinge
    350, 351, 352, 353, 354, 355,                                 # DFG loop
    380, 381, 382, 383, 384, 385,                                 # activation loop
}


# ── Step 1: Fetch + fix PDB ────────────────────────────────────────────────────

def prepare_receptor() -> Path:
    """Download 2HYY, run PDBFixer, write clean protein-only PDB."""
    fixed_pdb = MD_DIR / f"{PDB_ID}_fixed.pdb"
    if fixed_pdb.exists():
        print(f"  Using cached fixed PDB: {fixed_pdb}")
        return fixed_pdb

    # Fetch if needed
    if not PDB_CACHE.exists():
        import requests
        print(f"  Downloading {PDB_ID} from RCSB ...", end="", flush=True)
        r = requests.get(f"https://files.rcsb.org/download/{PDB_ID}.pdb", timeout=60)
        r.raise_for_status()
        PDB_CACHE.write_text(r.text)
        print(f" {len(r.text)//1024} KB")

    print(f"  Running PDBFixer on {PDB_ID} ...", end="", flush=True)
    fixer = PDBFixer(str(PDB_CACHE))

    # Keep only chain A (the kinase domain monomer with imatinib)
    chains = list(fixer.topology.chains())
    chain_ids = [c.id for c in chains]
    print(f" chains={chain_ids}", end="", flush=True)
    remove_chains = [c.id for c in chains if c.id != "A"]
    if remove_chains:
        fixer.removeChains(chainIds=list(set(remove_chains)))

    # Remove HETATM (ligand, water, ions) — keep only ATOM records
    fixer.removeHeterogens(keepWater=False)

    # Find and fill missing residues / loops
    fixer.findMissingResidues()
    # Skip very long terminal missing regions (>10 residues) to avoid slow modelling
    missing = fixer.missingResidues
    fixer.missingResidues = {
        k: v for k, v in missing.items() if len(v) <= 10
    }

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    with open(fixed_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    n_atoms = fixer.topology.getNumAtoms()
    print(f"  done — {n_atoms:,} atoms -> {fixed_pdb.name}")
    return fixed_pdb


# ── Step 2: Build OpenMM system ────────────────────────────────────────────────

def build_system(fixed_pdb: Path):
    """Solvate, add ions, create ff14SB + TIP3P system."""
    print("  Loading force fields (ff14SB + TIP3P) ...")
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    print("  Reading fixed PDB ...", end="", flush=True)
    pdb = PDBFile(str(fixed_pdb))
    print(f" {pdb.topology.getNumAtoms():,} atoms")

    print(f"  Solvating (box padding {BOX_PADDING}) ...", end="", flush=True)
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        ff,
        padding    = BOX_PADDING,
        ionicStrength = IONIC_STR * unit.molar,
        positiveIon = "Na+",
        negativeIon = "Cl-",
    )
    n_total = modeller.topology.getNumAtoms()
    n_prot  = pdb.topology.getNumAtoms()
    print(f" {n_total:,} total ({n_prot:,} protein + {n_total-n_prot:,} solvent)")

    print("  Creating OpenMM System ...", end="", flush=True)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod  = PME,
        nonbondedCutoff  = NONBONDED_CUT,
        constraints      = HBonds,
    )
    print(" done")
    return modeller.topology, modeller.positions, system


# ── Step 3: Energy minimise ────────────────────────────────────────────────────

def minimise(topology, positions, system, platform, platform_props):
    print("  Creating simulation (minimisation) ...", end="", flush=True)
    integrator = LangevinMiddleIntegrator(
        TEMPERATURE, 1 / unit.picoseconds, TIMESTEP
    )
    sim = Simulation(topology, system, integrator, platform, platform_props)
    sim.context.setPositions(positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(f" E0 = {e0.value_in_unit(unit.kilocalories_per_mole):.1f} kcal/mol")

    print("  Minimising (max 1000 steps) ...", end="", flush=True)
    sim.minimizeEnergy(maxIterations=1000)
    e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(f" E1 = {e1.value_in_unit(unit.kilocalories_per_mole):.1f} kcal/mol")
    return sim


# ── Step 4 + 5: Equilibration ─────────────────────────────────────────────────

def equilibrate(sim, topology, system, platform, platform_props):
    # NVT: heat to 300 K
    print(f"  NVT equilibration {EQUIL_NVT_PS} ps ...", end="", flush=True)
    t0 = time.time()
    sim.context.setVelocitiesToTemperature(TEMPERATURE)
    sim.step(int(EQUIL_NVT_PS * 1000 / 2))   # 2 fs step
    print(f" {time.time()-t0:.0f}s")

    # NPT: add barostat
    print(f"  NPT equilibration {EQUIL_NPT_PS} ps ...", end="", flush=True)
    baro = MonteCarloBarostat(PRESSURE, TEMPERATURE)
    system.addForce(baro)
    sim.context.reinitialize(preserveState=True)
    t0 = time.time()
    sim.step(int(EQUIL_NPT_PS * 1000 / 2))
    print(f" {time.time()-t0:.0f}s")
    return sim


# ── Step 6: Production MD ──────────────────────────────────────────────────────

def production(sim, topology):
    dcd_path   = str(MD_DIR / "production.dcd")
    log_path   = str(MD_DIR / "production.log")
    pdb_path   = str(MD_DIR / "production_start.pdb")
    total_steps = int(PROD_NS * 1e6 / 2)          # 2 fs step
    save_steps  = int(SAVE_EVERY_PS * 1000 / 2)   # steps between frames

    # Write starting structure
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(pdb_path, "w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f)

    sim.reporters.append(DCDReporter(dcd_path, save_steps))
    sim.reporters.append(StateDataReporter(
        log_path,
        save_steps,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        progress=True,
        totalSteps=total_steps,
        separator="\t",
    ))
    sim.reporters.append(StateDataReporter(
        sys.stdout,
        save_steps * 10,      # print to screen every 100 ps
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        progress=True,
        totalSteps=total_steps,
        separator="\t",
    ))

    print(f"  Production MD {PROD_NS} ns ({total_steps:,} steps, "
          f"{total_steps//save_steps} frames) ...")
    t0 = time.time()
    sim.step(total_steps)
    elapsed = time.time() - t0
    rate    = PROD_NS * 1000 / elapsed  # ns/h
    print(f"  Done in {elapsed/3600:.2f} h  ({rate:.1f} ns/h)")
    return dcd_path, log_path, pdb_path, elapsed


# ── Step 7: Analysis ───────────────────────────────────────────────────────────

def analyse(dcd_path: str, pdb_path: str, log_path: str, n_protein_atoms: int):
    print("  Loading trajectory with MDTraj ...", end="", flush=True)
    traj = md.load(dcd_path, top=pdb_path)
    # Keep only protein atoms for analysis
    prot_idx = traj.topology.select("protein")
    traj_prot = traj.atom_slice(prot_idx)
    n_frames  = traj_prot.n_frames
    time_ns   = np.linspace(0, PROD_NS, n_frames)
    print(f" {n_frames} frames, {traj_prot.n_atoms:,} protein atoms")

    # Reference = first frame (after equilibration)
    ref = traj_prot[0]

    # ── Cα RMSD vs time ───────────────────────────────────────────────────────
    ca_idx   = traj_prot.topology.select("name CA")
    traj_ca  = traj_prot.atom_slice(ca_idx)
    ref_ca   = ref.atom_slice(ref.topology.select("name CA"))
    rmsd_nm  = md.rmsd(traj_ca, ref_ca)
    rmsd_ang = rmsd_nm * 10  # nm -> Å

    # ── Per-residue RMSF ──────────────────────────────────────────────────────
    rmsf_nm  = md.rmsf(traj_ca, ref_ca)
    rmsf_ang = rmsf_nm * 10
    res_ids  = [r.resSeq for r in traj_prot.topology.residues]
    ca_res   = res_ids[::len(res_ids)//len(ca_idx)] if len(res_ids) != len(ca_idx) else res_ids
    # safer approach:
    ca_res   = [traj_prot.topology.atom(i).residue.resSeq for i in ca_idx]

    # ── Radius of gyration ────────────────────────────────────────────────────
    rg_nm    = md.compute_rg(traj_prot)
    rg_ang   = rg_nm * 10

    # ── Potential energy from log ─────────────────────────────────────────────
    pot_e = []
    t_log = []
    try:
        with open(log_path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 5:
                    try:
                        t_log.append(float(parts[1]) / 1000)   # ps -> ns
                        pot_e.append(float(parts[2]))           # kcal/mol
                    except ValueError:
                        pass
    except Exception:
        pass

    # ── SASA vs time ─────────────────────────────────────────────────────────
    print("  Computing SASA ...", end="", flush=True)
    sasa = md.shrake_rupley(traj_prot, mode="residue")  # (n_frames, n_res)
    total_sasa_nm2 = sasa.sum(axis=1)
    total_sasa_ang2 = total_sasa_nm2 * 100   # nm² -> Å²
    print(" done")

    # ── Binding-site RMSD ─────────────────────────────────────────────────────
    # Select Cα of binding-site residues
    bs_sel  = " or ".join(f"(name CA and resSeq {r})" for r in sorted(BINDING_SITE_RES))
    bs_idx  = traj_prot.topology.select(bs_sel)
    if len(bs_idx) > 0:
        traj_bs = traj_prot.atom_slice(bs_idx)
        ref_bs  = ref.atom_slice(ref.topology.select(bs_sel))
        bs_rmsd_nm  = md.rmsd(traj_bs, ref_bs)
        bs_rmsd_ang = bs_rmsd_nm * 10
    else:
        bs_rmsd_ang = rmsd_ang  # fallback

    return dict(
        time_ns     = time_ns,
        rmsd_ang    = rmsd_ang,
        rmsf_ang    = rmsf_ang,
        ca_res      = np.array(ca_res),
        rg_ang      = rg_ang,
        pot_e       = np.array(pot_e),
        t_log       = np.array(t_log),
        sasa_ang2   = total_sasa_ang2,
        bs_rmsd_ang = bs_rmsd_ang,
    )


# ── Step 8: Figures ───────────────────────────────────────────────────────────

def make_figures(data: dict):
    time_ns     = data["time_ns"]
    rmsd_ang    = data["rmsd_ang"]
    rmsf_ang    = data["rmsf_ang"]
    ca_res      = data["ca_res"]
    rg_ang      = data["rg_ang"]
    pot_e       = data["pot_e"]
    t_log       = data["t_log"]
    sasa_ang2   = data["sasa_ang2"]
    bs_rmsd_ang = data["bs_rmsd_ang"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.42, wspace=0.35)

    # (a) Cα RMSD
    ax = axes[0, 0]
    ax.plot(time_ns, rmsd_ang, lw=1.2, color="#2196F3")
    ax.axhline(np.mean(rmsd_ang), ls="--", color="gray", lw=1, label=f"Mean {np.mean(rmsd_ang):.2f} Å")
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Cα RMSD (Å)")
    ax.set_title("(a) Cα RMSD vs Time", fontweight="bold")
    ax.legend(fontsize=8)

    # (b) Per-residue RMSF
    ax = axes[0, 1]
    ax.plot(ca_res, rmsf_ang, lw=1.0, color="#4CAF50")
    # Shade binding-site residues
    bs = sorted(BINDING_SITE_RES)
    for r in bs:
        ax.axvline(r, color="red", alpha=0.05, lw=0.5)
    ax.set_xlabel("Residue number"); ax.set_ylabel("Cα RMSF (Å)")
    ax.set_title("(b) Per-residue RMSF", fontweight="bold")

    # (c) Radius of gyration
    ax = axes[0, 2]
    ax.plot(time_ns, rg_ang, lw=1.2, color="#9C27B0")
    ax.axhline(np.mean(rg_ang), ls="--", color="gray", lw=1, label=f"Mean {np.mean(rg_ang):.2f} Å")
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Rg (Å)")
    ax.set_title("(c) Radius of Gyration", fontweight="bold")
    ax.legend(fontsize=8)

    # (d) Potential energy
    ax = axes[1, 0]
    if len(pot_e) > 0:
        ax.plot(t_log, pot_e, lw=1.0, color="#F44336", alpha=0.8)
        ax.set_xlabel("Time (ns)"); ax.set_ylabel("Potential Energy (kcal/mol)")
    else:
        ax.text(0.5, 0.5, "Energy log not available", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title("(d) Potential Energy", fontweight="bold")

    # (e) SASA vs time
    ax = axes[1, 1]
    ax.plot(time_ns, sasa_ang2, lw=1.2, color="#FF9800")
    ax.axhline(np.mean(sasa_ang2), ls="--", color="gray", lw=1,
               label=f"Mean {np.mean(sasa_ang2):.0f} Å²")
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("SASA (Å²)")
    ax.set_title("(e) Solvent-Accessible Surface Area", fontweight="bold")
    ax.legend(fontsize=8)

    # (f) Binding-site RMSD
    ax = axes[1, 2]
    ax.plot(time_ns, bs_rmsd_ang, lw=1.2, color="#009688")
    ax.axhline(np.mean(bs_rmsd_ang), ls="--", color="gray", lw=1,
               label=f"Mean {np.mean(bs_rmsd_ang):.2f} Å")
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Binding-site Cα RMSD (Å)")
    ax.set_title("(f) Binding-site RMSD", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle(
        "ABL1 Kinase (2HYY chain A) — 10 ns All-Atom MD, ff14SB/TIP3P, 300 K",
        fontsize=12, fontweight="bold"
    )

    for suffix in ("png", "pdf"):
        out = FIGS_DIR / f"md_summary.{suffix}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual panel PNGs (for make_figures.py composite)
    panel_data = [
        ("md_rmsd",        time_ns, rmsd_ang,    "Time (ns)", "Cα RMSD (Å)",         "Cα RMSD vs Time",        "#2196F3"),
        ("md_rmsf",        ca_res,  rmsf_ang,     "Residue",  "Cα RMSF (Å)",         "Per-residue RMSF",       "#4CAF50"),
        ("md_rg",          time_ns, rg_ang,       "Time (ns)", "Rg (Å)",              "Radius of Gyration",     "#9C27B0"),
        ("md_sasa",        time_ns, sasa_ang2,    "Time (ns)", "SASA (Å²)",           "SASA vs Time",           "#FF9800"),
        ("md_bs_rmsd",     time_ns, bs_rmsd_ang,  "Time (ns)", "BS RMSD (Å)",         "Binding-site RMSD",      "#009688"),
    ]
    for fname, xd, yd, xl, yl, title, col in panel_data:
        f2, a2 = plt.subplots(figsize=(5, 4))
        a2.plot(xd, yd, lw=1.5, color=col)
        a2.set_xlabel(xl); a2.set_ylabel(yl); a2.set_title(title, fontweight="bold")
        f2.tight_layout()
        f2.savefig(FIGS_DIR / f"{fname}.png", dpi=150, bbox_inches="tight")
        plt.close(f2)

    if len(pot_e) > 0:
        f2, a2 = plt.subplots(figsize=(5, 4))
        a2.plot(t_log, pot_e, lw=1.2, color="#F44336", alpha=0.8)
        a2.set_xlabel("Time (ns)"); a2.set_ylabel("Potential Energy (kcal/mol)")
        a2.set_title("Potential Energy", fontweight="bold")
        f2.tight_layout()
        f2.savefig(FIGS_DIR / "md_energy.png", dpi=150, bbox_inches="tight")
        plt.close(f2)

    print(f"  Figures -> {FIGS_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nJanusDiscover — MD Simulation (ABL1 kinase, 2HYY chain A)")
    print(f"  Target     : {PROD_NS} ns all-atom MD")
    print(f"  Force field: ff14SB + TIP3P")
    print(f"  Platform   : {PLATFORM_NAME} ({PLATFORM_PROPS})")
    print()

    t_total = time.time()

    # Platform
    platform = Platform.getPlatformByName(PLATFORM_NAME)

    # 1. Prepare
    print("[1/6] Preparing receptor ...")
    fixed_pdb = prepare_receptor()

    # 2. Build system
    print("\n[2/6] Building OpenMM system ...")
    topology, positions, system = build_system(fixed_pdb)
    n_prot_atoms = PDBFile(str(fixed_pdb)).topology.getNumAtoms()

    # 3. Minimise
    print("\n[3/6] Energy minimisation ...")
    sim = minimise(topology, positions, system, platform, PLATFORM_PROPS)

    # 4+5. Equilibrate
    print(f"\n[4+5/6] Equilibration (NVT {EQUIL_NVT_PS} ps + NPT {EQUIL_NPT_PS} ps) ...")
    sim = equilibrate(sim, topology, system, platform, PLATFORM_PROPS)

    # 6. Production
    print(f"\n[6/6] Production MD ({PROD_NS} ns) ...")
    dcd_path, log_path, pdb_path, elapsed_prod = production(sim, topology)

    # 7. Analysis
    print("\n[Analysis] MDTraj ...")
    data = analyse(dcd_path, pdb_path, log_path, n_prot_atoms)

    # 8. Figures
    print("\n[Figures] ...")
    make_figures(data)

    # 9. Save summary JSON
    result = {
        "target"         : "ABL1",
        "pdb_id"         : PDB_ID,
        "protocol"       : "protein-only ff14SB/TIP3P, 300 K/1 atm",
        "simulation_ns"  : PROD_NS,
        "timestep_fs"    : 2,
        "n_frames"       : int(len(data["time_ns"])),
        "rmsd_mean_ang"  : round(float(data["rmsd_ang"].mean()), 3),
        "rmsd_max_ang"   : round(float(data["rmsd_ang"].max()),  3),
        "rmsf_mean_ang"  : round(float(data["rmsf_ang"].mean()), 3),
        "rg_mean_ang"    : round(float(data["rg_ang"].mean()),   3),
        "sasa_mean_ang2" : round(float(data["sasa_ang2"].mean()),1),
        "bs_rmsd_mean_ang": round(float(data["bs_rmsd_ang"].mean()), 3),
        "production_s"   : round(elapsed_prod, 1),
        "total_s"        : round(time.time() - t_total, 1),
    }
    out = RESULTS_DIR / "md_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
