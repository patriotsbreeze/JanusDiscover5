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


async def _fetch_existing_drugs(chembl_target_id: str) -> list[dict]:
    """
    Query ChEMBL mechanism table for drugs/compounds that act on this target,
    then enrich with molecule name + max_phase from the molecule endpoint.
    Returns list of dicts suitable for TherapyEntry.
    """
    if not chembl_target_id:
        return []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                f"{CHEMBL_API}/mechanism.json",
                params={"target_chembl_id": chembl_target_id, "limit": 20},
            )
            r.raise_for_status()
            mechs = r.json().get("mechanisms", [])

        if not mechs:
            return []

        # Deduplicate by molecule_chembl_id, keep highest-phase unique molecule
        seen: dict[str, dict] = {}
        for m in mechs:
            mid = m.get("molecule_chembl_id", "")
            if mid and mid not in seen:
                seen[mid] = m

        # Enrich with molecule details (name, max_phase, first_approval)
        therapies: list[dict] = []
        async with httpx.AsyncClient(timeout=20) as client:
            for mid, mech in list(seen.items())[:10]:
                try:
                    mr = await client.get(f"{CHEMBL_API}/molecule/{mid}.json")
                    mol = mr.json() if mr.status_code == 200 else {}
                except Exception:
                    mol = {}

                name = (
                    mol.get("pref_name")
                    or mech.get("molecule_name")
                    or mid
                )
                max_phase = mol.get("max_phase") or 0
                year = mol.get("first_approval")

                if max_phase >= 1:  # only include clinical/approved compounds
                    if max_phase == 4:
                        status = "FDA Approved"
                    elif max_phase == 3:
                        status = "Phase III"
                    elif max_phase == 2:
                        status = "Phase II"
                    else:
                        status = "Phase I"

                    therapies.append({
                        "name": name.title() if name == name.upper() else name,
                        "mechanism": mech.get("mechanism_of_action", "").capitalize() or "Inhibitor",
                        "approval_status": status,
                        "year": int(year) if year else None,
                    })

        therapies.sort(key=lambda x: -({"FDA Approved": 4, "Phase III": 3, "Phase II": 2, "Phase I": 1}.get(x["approval_status"], 0)))
        return therapies[:8]
    except Exception as exc:
        logger.warning(f"_fetch_existing_drugs failed: {exc}")
        return []


async def _fetch_key_papers(protein_name: str) -> list[str]:
    """
    Query Europe PMC for publications relevant to the target, sorted by citation count.
    Returns a list of formatted citation strings like 'Author et al. YYYY Journal'.
    """
    try:
        query = f'("{protein_name}" OR "{protein_name} inhibitor") AND (METHODS:"drug discovery" OR METHODS:"virtual screening" OR METHODS:"molecular docking" OR TITLE:"inhibitor")'
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": 8,
                    "sort": "CITED",
                    "resultType": "core",
                },
            )
            r.raise_for_status()
            results = r.json().get("resultList", {}).get("result", [])

        papers: list[str] = []
        for p in results:
            authors = p.get("authorString", "")
            year = p.get("pubYear", "")
            journal = p.get("journalTitle", "") or p.get("journalAbbreviation", "")
            volume = p.get("journalInfo", {}).get("volume", "") if isinstance(p.get("journalInfo"), dict) else ""
            # Format as "Last et al. YEAR Journal VOL" (trim author list)
            first_author = authors.split(",")[0].strip().split(" ")[-1] if authors else "Unknown"
            citation = f"{first_author} et al. {year} {journal}"
            if volume:
                citation += f" {volume}"
            if citation.strip():
                papers.append(citation.strip())

        return papers[:6]
    except Exception as exc:
        logger.warning(f"_fetch_key_papers failed: {exc}")
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


def _build_prompt(
    protein_name: str,
    chembl_data: dict,
    pdb_ids: list[str],
    known_actives: int,
    existing_drugs: list[dict],
    key_papers: list[str],
) -> str:
    drugs_json = json.dumps(existing_drugs, indent=2) if existing_drugs else "None found in ChEMBL"
    papers_text = "\n".join(f"  - {p}" for p in key_papers) if key_papers else "  None found"
    return f"""You are an expert medicinal chemist and computational biologist.
Provide a structured literature review for the protein target: {protein_name}.

Available data retrieved from public databases:
- ChEMBL target info: {json.dumps(chembl_data, indent=2)[:1500]}
- PDB structure IDs: {pdb_ids}
- Known ChEMBL actives (pChEMBL >= 6): {known_actives}
- Clinical/approved drugs from ChEMBL mechanism table:
{drugs_json}
- Key publications from Europe PMC (sorted by citations):
{papers_text}

Instructions:
1. Use the provided drugs list as the basis for "existing_therapies". Add or correct entries using your knowledge but keep factual names and years.
2. Use the provided papers list as a starting point for "key_papers". Add highly cited papers from your knowledge if relevant.
3. For existing_scaffolds, provide real SMILES of known active scaffolds if you know them.

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
    existing_drugs: list[dict],
    key_papers: list[str],
    provider: LLMProvider,
    api_key: Optional[str],
    model: Optional[str],
) -> dict:
    """Dispatch to the appropriate LLM provider and return parsed JSON."""
    provider_str = provider.value if hasattr(provider, "value") else str(provider)
    model = model or os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(provider_str, "")
    prompt = _build_prompt(protein_name, chembl_data, pdb_ids, known_actives, existing_drugs, key_papers)

    try:
        text = await _call_provider(provider_str, api_key, model, prompt)
        # Strip any accidental markdown fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"{provider_str} synthesis failed ({type(e).__name__}: {e}). Using data fallback.")
        err = str(e).lower()
        if any(k in err for k in ("api_key", "authentication", "unauthorized", "invalid_api_key", "401")):
            key_label = _PROVIDER_NAMES.get(provider_str, f"{provider_str} API key")
            raise RuntimeError(
                f"{key_label} is missing or invalid. "
                f"Provide a valid key to enable AI literature synthesis in real mode."
            ) from e
        return _fallback_synthesis(protein_name, pdb_ids, known_actives, existing_drugs, key_papers)


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
    protein_name: str,
    pdb_ids: list[str],
    known_actives: int,
    existing_drugs: list[dict] | None = None,
    key_papers: list[str] | None = None,
) -> dict:
    has_struct = bool(pdb_ids)
    method = "hybrid" if has_struct and known_actives > 50 else ("sbdd" if has_struct else "lbdd")

    therapies = existing_drugs if existing_drugs else []

    papers = key_papers if key_papers else []

    return {
        "description": (
            f"{protein_name} is a therapeutically relevant target implicated in multiple "
            "disease pathways. Structural and biochemical data indicate tractability for "
            "small-molecule modulation."
        ),
        "existing_therapies": therapies,
        "discovery_history": {
            "lbdd_done": known_actives > 20,
            "sbdd_done": has_struct,
            "hybrid_done": has_struct and known_actives > 50,
            "key_papers": papers,
            "existing_scaffolds": [],
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
        # Real mode: fetch from all live APIs concurrently
        import asyncio
        target_data = await _fetch_chembl_target(protein_name)
        chembl_id = target_data.get("target_chembl_id", "")

        async def _zero(): return 0

        (known_actives, pdb_ids, existing_drugs, key_papers) = await asyncio.gather(
            _count_chembl_actives(chembl_id) if chembl_id else _zero(),
            _fetch_pdb_ids(protein_name),
            _fetch_existing_drugs(chembl_id),
            _fetch_key_papers(protein_name),
        )

        data = await _llm_synthesis(
            protein_name, target_data, pdb_ids, known_actives,
            existing_drugs, key_papers,
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
