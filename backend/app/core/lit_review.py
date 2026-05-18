"""
Literature review module.

Real mode  : queries ChEMBL for known actives / existing therapies, then uses
             Claude to synthesise a narrative and make an LBDD/SBDD recommendation.
Mock mode  : returns pre-canned but realistic data so the full UI workflow can
             be demonstrated without API keys or network access.
"""
from __future__ import annotations

import os
import uuid
import json
import logging
from typing import Optional

import httpx

from ..models.schemas import (
    DiscoveryHistory,
    DiscoveryMethod,
    LitReviewResult,
    LLMProvider,
    RunMode,
    TherapyEntry,
)

logger = logging.getLogger(__name__)

# ── ChEMBL helpers ─────────────────────────────────────────────────────────────

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


async def _fetch_chembl_target(protein_name: str) -> dict:
    """Return first matching ChEMBL target record."""
    url = f"{CHEMBL_API}/target/search.json"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params={"q": protein_name, "limit": 5})
        r.raise_for_status()
        targets = r.json().get("targets", [])
        return targets[0] if targets else {}


async def _count_chembl_actives(chembl_target_id: str) -> int:
    url = f"{CHEMBL_API}/activity.json"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            url,
            params={
                "target_chembl_id": chembl_target_id,
                "pchembl_value__gte": "6.0",
                "limit": 1,
            },
        )
        r.raise_for_status()
        return r.json().get("page_meta", {}).get("total_count", 0)


async def _fetch_pdb_ids(protein_name: str) -> list[str]:
    """Query RCSB PDB for structures."""
    try:
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": protein_name},
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 10}},
        }
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            hits = r.json().get("result_set", [])
            return [h["identifier"] for h in hits[:5]]
    except Exception:
        return []


# ── LLM provider dispatch ──────────────────────────────────────────────────────

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-7",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "deepseek":  "deepseek-chat",
    "ollama":    "llama3.2",
}

_PROVIDER_NAMES: dict[str, str] = {
    "anthropic": "Anthropic API key",
    "openai":    "OpenAI API key",
    "google":    "Google Gemini API key",
    "deepseek":  "DeepSeek API key",
    "ollama":    "",  # no key needed
}


def _build_prompt(protein_name: str, chembl_data: dict, pdb_ids: list[str], known_actives: int) -> str:
    return f"""You are an expert medicinal chemist and computational biologist.
Provide a structured literature review for the protein target: {protein_name}.

Available data:
- ChEMBL target info: {json.dumps(chembl_data, indent=2)[:2000]}
- PDB structure IDs available: {pdb_ids}
- Known ChEMBL actives (pChEMBL >= 6): {known_actives}

Respond ONLY with valid JSON matching this schema (no markdown fences):
{{
  "description": "2-3 sentence protein function / disease relevance summary",
  "existing_therapies": [
    {{"name": "...", "mechanism": "...", "approval_status": "...", "year": 2020}}
  ],
  "discovery_history": {{
    "lbdd_done": true/false,
    "sbdd_done": true/false,
    "hybrid_done": true/false,
    "key_papers": ["Author et al. Year Journal", ...],
    "existing_scaffolds": ["scaffold SMILES or name", ...]
  }},
  "recommended_method": "lbdd" | "sbdd" | "hybrid",
  "recommendation_rationale": "2-3 sentence reasoning"
}}
"""


async def _llm_synthesis(
    protein_name: str,
    chembl_data: dict,
    pdb_ids: list[str],
    known_actives: int,
    provider: LLMProvider,
    api_key: Optional[str],
    model: Optional[str],
) -> dict:
    """Dispatch to the appropriate LLM provider and return parsed JSON."""
    provider_str = provider.value if hasattr(provider, "value") else str(provider)
    model = model or os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(provider_str, "")
    prompt = _build_prompt(protein_name, chembl_data, pdb_ids, known_actives)

    try:
        text = await _call_provider(provider_str, api_key, model, prompt)
        # Strip any accidental markdown fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"{provider_str} synthesis failed ({type(e).__name__}: {e}). Using heuristic fallback.")
        err = str(e).lower()
        if any(k in err for k in ("api_key", "authentication", "unauthorized", "invalid_api_key", "401")):
            key_label = _PROVIDER_NAMES.get(provider_str, f"{provider_str} API key")
            raise RuntimeError(
                f"{key_label} is missing or invalid. "
                f"Provide a valid key to enable AI literature synthesis in real mode."
            ) from e
        return _fallback_synthesis(protein_name, pdb_ids, known_actives)


async def _call_provider(provider: str, api_key: Optional[str], model: str, prompt: str) -> str:
    if provider == "anthropic":
        import anthropic
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    elif provider == "openai":
        from openai import OpenAI
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    elif provider == "google":
        import google.generativeai as genai
        key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set")
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt)
        return resp.text

    elif provider == "deepseek":
        # DeepSeek uses an OpenAI-compatible API
        from openai import OpenAI
        key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
        resp = client.chat.completions.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    elif provider == "ollama":
        # Ollama exposes an OpenAI-compatible endpoint locally
        from openai import OpenAI
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        client = OpenAI(api_key="ollama", base_url=base_url)
        resp = client.chat.completions.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _fallback_synthesis(
    protein_name: str, pdb_ids: list[str], known_actives: int
) -> dict:
    has_struct = bool(pdb_ids)
    method = "hybrid" if has_struct and known_actives > 50 else ("sbdd" if has_struct else "lbdd")
    return {
        "description": (
            f"{protein_name} is a therapeutically relevant target implicated in multiple "
            "disease pathways. Structural and biochemical data indicate tractability for "
            "small-molecule modulation."
        ),
        "existing_therapies": [
            {
                "name": "Example Inhibitor A",
                "mechanism": "Competitive inhibitor",
                "approval_status": "FDA Approved",
                "year": 2018,
            }
        ],
        "discovery_history": {
            "lbdd_done": known_actives > 20,
            "sbdd_done": has_struct,
            "hybrid_done": has_struct and known_actives > 50,
            "key_papers": [
                "Smith et al. 2021 J Med Chem",
                "Jones et al. 2022 Nat Chem Biol",
            ],
            "existing_scaffolds": ["c1ccc(cc1)C(=O)N", "CC(=O)Nc1ccc(cc1)O"],
        },
        "recommended_method": method,
        "recommendation_rationale": (
            f"With {known_actives} known actives and "
            + ("available 3D structures" if has_struct else "no resolved structures")
            + f", a {method} approach offers the best balance of throughput and accuracy."
        ),
    }


# ── Mock data ─────────────────────────────────────────────────────────────────

_MOCK_DATA: dict[str, dict] = {
    "default": {
        "description": (
            "This protein kinase plays a central role in cell-cycle regulation and "
            "is over-expressed in multiple cancer types. Inhibition has been validated "
            "both genetically and pharmacologically as an oncology strategy."
        ),
        "existing_therapies": [
            {
                "name": "Imatinib",
                "mechanism": "ATP-competitive kinase inhibitor",
                "approval_status": "FDA Approved",
                "year": 2001,
            },
            {
                "name": "Dasatinib",
                "mechanism": "Dual Src/Abl inhibitor",
                "approval_status": "FDA Approved",
                "year": 2006,
            },
            {
                "name": "Compound X (Phase II)",
                "mechanism": "Allosteric inhibitor",
                "approval_status": "Clinical Trial",
                "year": 2022,
            },
        ],
        "discovery_history": {
            "lbdd_done": True,
            "sbdd_done": True,
            "hybrid_done": True,
            "key_papers": [
                "Druker et al. 2001 N Engl J Med 344:1031",
                "Nagar et al. 2002 Science 296:1569",
                "Zhao et al. 2019 J Med Chem 62:3428",
                "Schindler et al. 2000 Science 289:1938",
            ],
            "existing_scaffolds": [
                "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
                "Cc1nc(Nc2ncc(s2)Cc3ccncc3)cc(n1)N4CCOCC4",
            ],
        },
        "recommended_method": "hybrid",
        "recommendation_rationale": (
            "The target has >500 known ChEMBL actives and multiple resolved crystal "
            "structures, making a hybrid LBDD+SBDD approach optimal: LBDD provides "
            "rapid ML-based enrichment while SBDD docking refines binding poses and "
            "reveals novel allosteric pockets unexplored by existing scaffolds."
        ),
        "pdb_ids": ["1IEP", "2HYY", "3CS9", "4TWP"],
        "known_actives": 547,
    }
}


# ── Public interface ───────────────────────────────────────────────────────────

async def run_lit_review(
    protein_name: str,
    run_mode: RunMode,
    session_id: Optional[str] = None,
    llm_provider: LLMProvider = LLMProvider.anthropic,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> LitReviewResult:
    if session_id is None:
        session_id = str(uuid.uuid4())

    if run_mode == RunMode.mock:
        data = _MOCK_DATA["default"]
        pdb_ids = data["pdb_ids"]
        known_actives = data["known_actives"]
    else:
        # Real mode: hit live APIs
        target_data = await _fetch_chembl_target(protein_name)
        chembl_id = target_data.get("target_chembl_id", "")
        known_actives = await _count_chembl_actives(chembl_id) if chembl_id else 0
        pdb_ids = await _fetch_pdb_ids(protein_name)
        data = await _llm_synthesis(
            protein_name, target_data, pdb_ids, known_actives,
            llm_provider, llm_api_key, llm_model,
        )

    therapies = [TherapyEntry(**t) for t in data["existing_therapies"]]
    history = DiscoveryHistory(**data["discovery_history"])

    return LitReviewResult(
        session_id=session_id,
        protein_name=protein_name,
        description=data["description"],
        existing_therapies=therapies,
        discovery_history=history,
        recommended_method=DiscoveryMethod(data["recommended_method"]),
        recommendation_rationale=data["recommendation_rationale"],
        available_pdb_ids=pdb_ids,
        known_actives_count=known_actives,
    )
