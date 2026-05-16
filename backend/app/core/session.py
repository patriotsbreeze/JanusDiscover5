"""
In-memory session store for JanusDiscover.

Keeps track of lit review results, discovery results, MD results,
and figures per session so the frontend can poll for updates.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class SessionStatus(str, Enum):
    created = "created"
    lit_review_running = "lit_review_running"
    lit_review_done = "lit_review_done"
    awaiting_approval = "awaiting_approval"
    discovery_running = "discovery_running"
    discovery_done = "discovery_done"
    awaiting_md_decision = "awaiting_md_decision"
    md_running = "md_running"
    md_done = "md_done"
    manuscript_done = "manuscript_done"
    error = "error"


@dataclass
class Session:
    session_id: str
    protein_name: str
    run_mode: str
    status: SessionStatus = SessionStatus.created
    lit_review: Optional[dict] = None
    discovery_result: Optional[dict] = None
    md_result: Optional[dict] = None
    manuscript: Optional[dict] = None
    error_message: Optional[str] = None
    progress_pct: int = 0
    progress_msg: str = ""
    figures_dir: str = ""


_store: dict[str, Session] = {}


def create_session(protein_name: str, run_mode: str) -> Session:
    sid = str(uuid.uuid4())
    session = Session(session_id=sid, protein_name=protein_name, run_mode=run_mode)
    _store[sid] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    return _store.get(session_id)


def update_session(session_id: str, **kwargs: Any) -> None:
    s = _store.get(session_id)
    if s:
        for k, v in kwargs.items():
            setattr(s, k, v)


def session_to_dict(s: Session) -> dict:
    return {
        "session_id": s.session_id,
        "protein_name": s.protein_name,
        "run_mode": s.run_mode,
        "status": s.status,
        "progress_pct": s.progress_pct,
        "progress_msg": s.progress_msg,
        "lit_review": s.lit_review,
        "discovery_result": s.discovery_result,
        "md_result": s.md_result,
        "manuscript": s.manuscript,
        "error_message": s.error_message,
    }
