"""
Pydantic schemas for JanusDiscover API.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class RunMode(str, Enum):
    mock = "mock"
    real = "real"


class LLMProvider(str, Enum):
    anthropic = "anthropic"
    openai = "openai"
    google = "google"
    deepseek = "deepseek"
    ollama = "ollama"


class Dataset(str, Enum):
    fda_approved = "fda_approved"
    zinc_250k = "zinc_250k"
    chembl = "chembl"
    custom = "custom"


class DiscoveryMethod(str, Enum):
    lbdd = "lbdd"
    sbdd = "sbdd"
    hybrid = "hybrid"


class MDDuration(BaseModel):
    value: float = Field(..., gt=0, description="Duration value")
    unit: str = Field("ns", pattern="^(ps|ns)$")


# ── Request bodies ──────────────────────────────────────────────────────────────

class LitReviewRequest(BaseModel):
    protein_name: str = Field(..., min_length=2)
    run_mode: RunMode = RunMode.mock
    llm_provider: LLMProvider = LLMProvider.anthropic
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool
    chosen_method: Optional[DiscoveryMethod] = None
    dataset: Dataset = Dataset.zinc_250k
    run_mode: RunMode = RunMode.mock


class MDRequest(BaseModel):
    session_id: str
    run_md: bool
    duration: Optional[MDDuration] = None
    run_mode: RunMode = RunMode.mock


# ── Response bodies ─────────────────────────────────────────────────────────────

class TherapyEntry(BaseModel):
    name: str
    mechanism: str
    approval_status: str
    year: Optional[int] = None


class DiscoveryHistory(BaseModel):
    lbdd_done: bool
    sbdd_done: bool
    hybrid_done: bool
    key_papers: list[str]
    existing_scaffolds: list[str]


class LitReviewResult(BaseModel):
    session_id: str
    protein_name: str
    description: str
    existing_therapies: list[TherapyEntry]
    discovery_history: DiscoveryHistory
    recommended_method: DiscoveryMethod
    recommendation_rationale: str
    available_pdb_ids: list[str]
    known_actives_count: int


class CompoundResult(BaseModel):
    compound_id: str
    smiles: str
    score: float
    predicted_activity: Optional[float] = None
    docking_score: Optional[float] = None
    binding_affinity_kcal: Optional[float] = None


class ValidationMetrics(BaseModel):
    auc_roc: float
    bedroc: float
    ef1_percent: float
    ef5_percent: float
    enrichment_auc: Optional[float] = None
    pose_rmsd_mean: Optional[float] = None
    pose_success_rate_2A: Optional[float] = None


class DiscoveryResult(BaseModel):
    session_id: str
    method_used: DiscoveryMethod
    dataset_used: Dataset
    compounds_screened: int
    top_hits: list[CompoundResult]
    validation_metrics: ValidationMetrics
    figures: list[str]   # list of figure filenames/paths
    run_time_seconds: float
    mock: bool


class MDResult(BaseModel):
    session_id: str
    duration_ns: float
    frames_analyzed: int
    rmsd_mean_nm: float
    rmsd_std_nm: float
    rmsf_mean_nm: float
    radius_of_gyration_mean_nm: float
    potential_energy_mean_kj_mol: float
    figures: list[str]
    mock: bool


class ManuscriptResult(BaseModel):
    session_id: str
    tex_path: str
    pdf_path: Optional[str]
    bib_path: str
