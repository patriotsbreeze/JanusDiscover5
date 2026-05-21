"""
vina_cli_wrapper.py
===================
Drop-in Python replacement for the ``vina`` wheel (which has no Windows
build for Python 3.12).  Exposes the same API surface used in sbdd.py:

    from vina import Vina          # original
    from vina_cli_wrapper import Vina  # this file

The wrapper writes temp files, calls the ``vina.exe`` CLI (must be on PATH
or in the same directory as this file / the repo root), and parses the
output PDBQT and stdout for energies.

Tested against AutoDock Vina v1.2.5.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

# ── locate vina.exe ───────────────────────────────────────────────────────────

def _find_vina() -> str:
    """Return absolute path to vina / vina.exe or raise RuntimeError."""
    # 1. Same directory as this file
    here = Path(__file__).resolve().parent
    for name in ("vina.exe", "vina"):
        candidate = here / name
        if candidate.exists():
            return str(candidate)

    # 2. PATH
    import shutil
    exe = shutil.which("vina") or shutil.which("vina.exe")
    if exe:
        return exe

    raise RuntimeError(
        "AutoDock Vina executable not found.  "
        "Place vina.exe next to this file or add it to PATH."
    )


_VINA_EXE = None  # resolved lazily


def _vina_exe() -> str:
    global _VINA_EXE
    if _VINA_EXE is None:
        _VINA_EXE = _find_vina()
    return _VINA_EXE


# ── Vina class ────────────────────────────────────────────────────────────────

class Vina:
    """CLI-backed drop-in for the vina Python wheel."""

    def __init__(self, sf_name: str = "vina", verbosity: int = 1):
        self._sf_name      = sf_name
        self._verbosity    = verbosity
        self._receptor     = None   # path string
        self._ligand_str   = None   # PDBQT string
        self._center       = None   # [x, y, z]
        self._box_size     = None   # [sx, sy, sz]
        self._output_pdbqt = None   # path to docked output
        self._stdout       = ""

    # ── setup ─────────────────────────────────────────────────────────────────

    def set_receptor(self, receptor_path: str) -> None:
        """receptor_path: path to a PDBQT file on disk."""
        self._receptor = str(receptor_path)

    def set_ligand_from_string(self, pdbqt_string: str) -> None:
        """pdbqt_string: PDBQT content as a Python string."""
        self._ligand_str = pdbqt_string

    def compute_vina_maps(
        self,
        center: list[float],
        box_size: list[float] = (25.0, 25.0, 25.0),
    ) -> None:
        """Store centre / box; actual map computation happens at dock()."""
        self._center   = list(center)
        self._box_size = list(box_size)

    # ── docking ───────────────────────────────────────────────────────────────

    def dock(self, exhaustiveness: int = 8, n_poses: int = 3) -> None:
        """Run Vina docking; results accessible via poses() and energies()."""
        if not self._receptor:
            raise RuntimeError("set_receptor() must be called before dock()")
        if self._ligand_str is None:
            raise RuntimeError("set_ligand_from_string() must be called before dock()")
        if self._center is None:
            raise RuntimeError("compute_vina_maps() must be called before dock()")

        with tempfile.TemporaryDirectory() as tmpdir:
            lig_in  = Path(tmpdir) / "ligand.pdbqt"
            lig_out = Path(tmpdir) / "out.pdbqt"

            lig_in.write_text(self._ligand_str)

            cx, cy, cz   = self._center
            sx, sy, sz   = self._box_size

            cmd = [
                _vina_exe(),
                "--receptor",     self._receptor,
                "--ligand",       str(lig_in),
                "--out",          str(lig_out),
                "--center_x",     str(cx),
                "--center_y",     str(cy),
                "--center_z",     str(cz),
                "--size_x",       str(sx),
                "--size_y",       str(sy),
                "--size_z",       str(sz),
                "--exhaustiveness", str(exhaustiveness),
                "--num_modes",    str(n_poses),
                "--scoring",      self._sf_name,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self._stdout = result.stdout + result.stderr

            if self._verbosity > 0:
                print(self._stdout)

            if lig_out.exists():
                self._output_pdbqt = lig_out.read_text()
            else:
                # Vina sometimes exits 0 even when docking fails (no poses)
                self._output_pdbqt = ""

    # ── result accessors ──────────────────────────────────────────────────────

    def poses(self, n_poses: int = 1) -> str:
        """Return the best n_poses from the docked output as a PDBQT string."""
        if self._output_pdbqt is None:
            raise RuntimeError("Call dock() first.")

        # Split multi-model PDBQT by MODEL records
        models = re.split(r"(?=^MODEL\s)", self._output_pdbqt, flags=re.MULTILINE)
        models = [m for m in models if m.strip()]
        if not models:
            return self._output_pdbqt  # return as-is if no MODEL markers

        return "\n".join(models[:n_poses])

    def energies(self, n_poses: int = 3) -> list[list[float]]:
        """Return [[score, rmsd_lb, rmsd_ub], ...] for up to n_poses poses.

        Parses the 'REMARK VINA RESULT' lines from the output PDBQT, which
        Vina writes as:
            REMARK VINA RESULT:    <score>      <rmsd_lb>    <rmsd_ub>
        """
        if self._output_pdbqt is None:
            return []

        energies: list[list[float]] = []
        for line in self._output_pdbqt.splitlines():
            if line.startswith("REMARK VINA RESULT"):
                parts = line.split()
                try:
                    score   = float(parts[3])
                    rmsd_lb = float(parts[4])
                    rmsd_ub = float(parts[5])
                    energies.append([score, rmsd_lb, rmsd_ub])
                except (IndexError, ValueError):
                    pass

        # Fall back: parse from stdout if output PDBQT had no REMARK lines
        if not energies:
            energies = self._parse_stdout_energies()

        return energies[:n_poses]

    def _parse_stdout_energies(self) -> list[list[float]]:
        """Parse energy table from Vina stdout (lines like '   1      -7.3 ...')."""
        energies: list[list[float]] = []
        in_table = False
        for line in self._stdout.splitlines():
            if re.match(r"\s*mode\s+\|\s+affinity", line, re.IGNORECASE):
                in_table = True
                continue
            if in_table:
                m = re.match(
                    r"\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line
                )
                if m:
                    energies.append(
                        [float(m.group(2)), float(m.group(3)), float(m.group(4))]
                    )
        return energies
