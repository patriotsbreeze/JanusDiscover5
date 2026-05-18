"""
Main discovery API routes.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from ...core import lit_review as lr_module
from ...core import lbdd as lbdd_module
from ...core import sbdd as sbdd_module
from ...core import md_simulation as md_module
from ...core import manuscript as ms_module
from ...core import session as session_module
from ...core.session import SessionStatus, update_session, session_to_dict
from ...models.schemas import (
    ApprovalRequest,
    DiscoveryMethod,
    LitReviewRequest,
    MDRequest,
    RunMode,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["discovery"])

FIGURES_BASE = Path(os.getenv("FIGURES_DIR", "static/figures"))
MANUSCRIPT_BASE = Path(os.getenv("MANUSCRIPT_DIR", "static/manuscripts"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _figures_dir(session_id: str) -> Path:
    d = FIGURES_BASE / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manuscript_dir(session_id: str) -> Path:
    d = MANUSCRIPT_BASE / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Literature Review ─────────────────────────────────────────────────────────

@router.post("/lit-review")
async def start_lit_review(req: LitReviewRequest, background_tasks: BackgroundTasks):
    session = session_module.create_session(req.protein_name, req.run_mode)
    update_session(session.session_id,
                   status=SessionStatus.lit_review_running,
                   progress_pct=5,
                   progress_msg="Starting literature review…",
                   figures_dir=str(_figures_dir(session.session_id)))
    background_tasks.add_task(_run_lit_review, session.session_id, req)
    return {"session_id": session.session_id, "status": "started"}


async def _run_lit_review(session_id: str, req: LitReviewRequest):
    try:
        update_session(session_id, progress_pct=20, progress_msg="Querying ChEMBL and PDB…")
        result = await lr_module.run_lit_review(req.protein_name, req.run_mode, session_id)
        update_session(
            session_id,
            lit_review=result.model_dump(),
            status=SessionStatus.awaiting_approval,
            progress_pct=100,
            progress_msg="Literature review complete — awaiting your approval.",
        )
    except Exception as e:
        logger.exception(e)
        update_session(session_id, status=SessionStatus.error, error_message=str(e), progress_pct=0)


# ── Session status ────────────────────────────────────────────────────────────

@router.get("/session/{session_id}")
async def get_session_status(session_id: str):
    s = session_module.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return session_to_dict(s)


# ── Approval & discovery ──────────────────────────────────────────────────────

@router.post("/approve")
async def approve_and_run(req: ApprovalRequest, background_tasks: BackgroundTasks):
    s = session_module.get_session(req.session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if not req.approved:
        update_session(req.session_id, status=SessionStatus.error,
                       error_message="User declined to proceed.")
        return {"status": "cancelled"}

    # Determine method
    method = req.chosen_method
    if method is None:
        lit = s.lit_review or {}
        method = DiscoveryMethod(lit.get("recommended_method", "hybrid"))

    update_session(req.session_id,
                   status=SessionStatus.discovery_running,
                   progress_pct=5,
                   progress_msg=f"Running {method.value.upper()} pipeline…")
    background_tasks.add_task(_run_discovery, req.session_id, method, req)
    return {"status": "running", "method": method}


async def _run_discovery(session_id: str, method: DiscoveryMethod, req: ApprovalRequest):
    try:
        s = session_module.get_session(session_id)
        fig_dir = _figures_dir(session_id)
        pdb_ids = (s.lit_review or {}).get("available_pdb_ids", [])

        update_session(session_id, progress_pct=20, progress_msg="Generating molecular fingerprints / preparing receptor…")

        if method == DiscoveryMethod.lbdd:
            result = await lbdd_module.run_lbdd(
                session_id, s.protein_name, req.dataset, req.run_mode, fig_dir, pdb_ids
            )
        elif method == DiscoveryMethod.sbdd:
            result = await sbdd_module.run_sbdd(
                session_id, s.protein_name, req.dataset, req.run_mode, fig_dir, pdb_ids
            )
        else:  # hybrid
            update_session(session_id, progress_pct=20, progress_msg="Running LBDD pass…")
            lbdd_res = await lbdd_module.run_lbdd(
                session_id, s.protein_name, req.dataset, req.run_mode, fig_dir, pdb_ids
            )
            update_session(session_id, progress_pct=55, progress_msg="Running SBDD pass…")
            sbdd_res = await sbdd_module.run_sbdd(
                session_id, s.protein_name, req.dataset, req.run_mode, fig_dir, pdb_ids
            )
            result = _merge_hybrid(lbdd_res, sbdd_res)

        update_session(session_id, progress_pct=85, progress_msg="Generating manuscript…")

        ms = ms_module.generate_manuscript(
            protein_name=s.protein_name,
            method=result["method_used"],
            dataset=req.dataset,
            val_metrics=result["validation_metrics"],
            md_results=None,
            figures_dir=fig_dir,
            output_dir=_manuscript_dir(session_id),
            session_id=session_id,
        )

        update_session(
            session_id,
            discovery_result=result,
            manuscript=ms,
            status=SessionStatus.awaiting_md_decision,
            progress_pct=100,
            progress_msg="Discovery complete. Do you want to run MD simulations?",
        )
    except Exception as e:
        logger.exception(e)
        update_session(session_id, status=SessionStatus.error, error_message=str(e))


def _merge_hybrid(lbdd: dict, sbdd: dict) -> dict:
    import numpy as np
    lhits = {h["compound_id"]: h for h in lbdd["top_hits"]}
    shits = {h["compound_id"]: h for h in sbdd["top_hits"]}
    merged_ids = list(set(list(lhits.keys())[:10] + list(shits.keys())[:10]))[:20]
    hits = []
    for cid in merged_ids:
        if cid in lhits:
            h = lhits[cid].copy()
        else:
            h = shits[cid].copy()
        lscore = lhits.get(cid, {}).get("score", 0.5)
        sscore = sbdd["top_hits"][0]["score"] if shits else 0.5
        h["score"] = float(0.4 * lscore + 0.6 * sscore)
        hits.append(h)

    lvm = lbdd["validation_metrics"]
    svm = sbdd["validation_metrics"]
    merged_metrics = {
        "auc_roc": round((lvm["auc_roc"] + svm["auc_roc"]) / 2, 4),
        "bedroc": round((lvm["bedroc"] + svm["bedroc"]) / 2, 4),
        "ef1_percent": round((lvm["ef1_percent"] + svm["ef1_percent"]) / 2, 2),
        "ef5_percent": round((lvm["ef5_percent"] + svm["ef5_percent"]) / 2, 2),
        "pose_rmsd_mean": svm.get("pose_rmsd_mean"),
        "pose_success_rate_2A": svm.get("pose_success_rate_2A"),
    }

    return {
        "method_used": "hybrid",
        "compounds_screened": max(lbdd["compounds_screened"], sbdd["compounds_screened"]),
        "top_hits": hits,
        "validation_metrics": merged_metrics,
        "figures": list(set(lbdd["figures"] + sbdd["figures"])),
        "run_time_seconds": lbdd["run_time_seconds"] + sbdd["run_time_seconds"],
        "mock": lbdd["mock"] and sbdd["mock"],
    }


# ── MD Simulation ─────────────────────────────────────────────────────────────

@router.post("/md")
async def run_md(req: MDRequest, background_tasks: BackgroundTasks):
    s = session_module.get_session(req.session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if not req.run_md:
        update_session(req.session_id,
                       status=SessionStatus.manuscript_done,
                       progress_msg="MD simulation skipped.")
        return {"status": "skipped"}

    duration_ns = 1.0  # default
    if req.duration:
        duration_ns = req.duration.value if req.duration.unit == "ns" else req.duration.value / 1000

    update_session(req.session_id,
                   status=SessionStatus.md_running,
                   progress_pct=5,
                   progress_msg=f"Starting {duration_ns} ns MD simulation…")
    background_tasks.add_task(_run_md_bg, req.session_id, duration_ns, req.run_mode)
    return {"status": "running"}


async def _run_md_bg(session_id: str, duration_ns: float, run_mode: str):
    try:
        s = session_module.get_session(session_id)
        fig_dir = _figures_dir(session_id)
        pdb_ids = (s.lit_review or {}).get("available_pdb_ids", [])
        top_hit_smiles = None
        if s.discovery_result and s.discovery_result.get("top_hits"):
            top_hit_smiles = s.discovery_result["top_hits"][0].get("smiles")

        update_session(session_id, progress_pct=20, progress_msg="Solvating system…")

        md_result = await md_module.run_md(
            session_id=session_id,
            duration_ns=duration_ns,
            run_mode=run_mode,
            figures_dir=fig_dir,
            top_hit_smiles=top_hit_smiles,
            pdb_id=pdb_ids[0] if pdb_ids else None,
        )

        update_session(session_id, progress_pct=90, progress_msg="Updating manuscript with MD results…")

        # Regenerate manuscript with MD data
        ms = ms_module.generate_manuscript(
            protein_name=s.protein_name,
            method=(s.discovery_result or {}).get("method_used", "hybrid"),
            dataset="",
            val_metrics=(s.discovery_result or {}).get("validation_metrics", {}),
            md_results=md_result,
            figures_dir=fig_dir,
            output_dir=_manuscript_dir(session_id),
            session_id=session_id,
        )

        update_session(
            session_id,
            md_result=md_result,
            manuscript=ms,
            status=SessionStatus.manuscript_done,
            progress_pct=100,
            progress_msg="MD simulation complete. Manuscript updated.",
        )
    except Exception as e:
        logger.exception(e)
        update_session(session_id, status=SessionStatus.error, error_message=str(e))


# ── Path traversal guard ──────────────────────────────────────────────────────

def _safe_path(base: Path, *parts: str) -> Path:
    """Resolve and verify path stays within base; raise 400 otherwise."""
    try:
        resolved = (base / Path(*parts)).resolve()
        base_resolved = base.resolve()
        resolved.relative_to(base_resolved)  # raises ValueError if outside base
        return resolved
    except (ValueError, Exception):
        raise HTTPException(400, "Invalid path")


# ── File download endpoints ───────────────────────────────────────────────────

@router.get("/figures/{session_id}/{filename}")
async def serve_figure(session_id: str, filename: str):
    path = _safe_path(FIGURES_BASE, session_id, filename)
    if not path.exists():
        raise HTTPException(404, "Figure not found")
    return FileResponse(str(path), media_type="image/png")


@router.get("/manuscript/{session_id}/tex")
async def download_tex(session_id: str):
    path = _safe_path(MANUSCRIPT_BASE, session_id, "manuscript.tex")
    if not path.exists():
        raise HTTPException(404, "Manuscript not found")
    return FileResponse(str(path), media_type="text/plain",
                        headers={"Content-Disposition": "attachment; filename=manuscript.tex"})


@router.get("/manuscript/{session_id}/pdf")
async def download_pdf(session_id: str):
    path = _safe_path(MANUSCRIPT_BASE, session_id, "manuscript.pdf")
    if not path.exists():
        raise HTTPException(404, "PDF not compiled (pdflatex not available)")
    return FileResponse(str(path), media_type="application/pdf",
                        headers={"Content-Disposition": "attachment; filename=manuscript.pdf"})


@router.get("/manuscript/{session_id}/bib")
async def download_bib(session_id: str):
    path = _safe_path(MANUSCRIPT_BASE, session_id, "refs.bib")
    if not path.exists():
        raise HTTPException(404, "Bibliography not found")
    return FileResponse(str(path), media_type="text/plain",
                        headers={"Content-Disposition": "attachment; filename=refs.bib"})
