"""
Literature review module.

Real mode  : queries ChEMBL for known actives / existing therapies, then uses
             Claude to synthesise a narrative and make an LBDD/SBDD recommendation.
Mock mode  : returns pre-canned but realistic data so the full UI workflow can
             be demonstrated without API keys or network access.
"""
from __future__ import annotations

import os
import re
import uuid
import json
import logging
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from ..models.schemas import (
    DiscoveryHistory,
    DiscoveryMethod,
    LitReviewResult,
    LLMProvider,
    PaperEntry,
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


_PHASE_RANK = {"FDA Approved": 4, "EMA Approved": 4, "Phase III": 3, "Phase II": 2, "Phase I": 1}


def _phase_status(max_phase) -> Optional[str]:
    try:
        mp = int(max_phase)
    except (TypeError, ValueError):
        return None
    if mp == 4: return "FDA Approved"
    if mp == 3: return "Phase III"
    if mp == 2: return "Phase II"
    if mp == 1: return "Phase I"
    return None


def _nice_name(raw: str) -> str:
    """Title-case only if the string is ALL CAPS; leave mixed-case alone."""
    if raw and raw == raw.upper():
        return raw.title()
    return raw or ""


async def _enrich_molecules(mids_mechs: dict[str, dict]) -> list[dict]:
    """Look up molecule details for a set of ChEMBL IDs and return TherapyEntry dicts."""
    therapies: list[dict] = []
    async with httpx.AsyncClient(timeout=25) as client:
        for mid, mech in list(mids_mechs.items())[:15]:
            try:
                mr = await client.get(f"{CHEMBL_API}/molecule/{mid}.json")
                mol = mr.json() if mr.status_code == 200 else {}
            except Exception:
                mol = {}

            name = _nice_name(mol.get("pref_name") or mech.get("molecule_name") or mid)
            status = _phase_status(mol.get("max_phase") or mech.get("max_phase"))
            if status is None:
                continue  # skip pre-clinical / unannotated

            year = mol.get("first_approval")
            moa = (mech.get("mechanism_of_action") or "Active compound").capitalize()
            therapies.append({
                "name": name,
                "mechanism": moa,
                "approval_status": status,
                "year": int(year) if year else None,
            })

    therapies.sort(key=lambda x: -_PHASE_RANK.get(x["approval_status"], 0))
    return therapies[:10]


async def _fetch_existing_drugs(chembl_target_id: str) -> list[dict]:
    """
    Two-pass lookup for clinical compounds:
      1. ChEMBL mechanism table (explicitly annotated drug-target pairs)
      2. If sparse, also scan the activity table for any max_phase >= 1 molecule
         with pChEMBL >= 6, so targets like SARS-CoV-2 Mpro don't show empty.
    """
    if not chembl_target_id:
        return []
    try:
        import asyncio as _aio

        # Pass 1: mechanism table
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                f"{CHEMBL_API}/mechanism.json",
                params={"target_chembl_id": chembl_target_id, "limit": 25},
            )
            r.raise_for_status()
            mechs = r.json().get("mechanisms", [])

        seen: dict[str, dict] = {}
        for m in mechs:
            mid = m.get("molecule_chembl_id", "")
            if mid and mid not in seen:
                seen[mid] = m

        # Pass 2: activity table fallback when mechanism table is sparse
        if len(seen) < 3:
            async with httpx.AsyncClient(timeout=20) as client:
                r2 = await client.get(
                    f"{CHEMBL_API}/activity.json",
                    params={
                        "target_chembl_id": chembl_target_id,
                        "molecule_max_phase__gte": "1",
                        "pchembl_value__gte": "5.0",
                        "limit": 50,
                    },
                )
                activities = r2.json().get("activities", []) if r2.status_code == 200 else []

            for act in activities:
                mid = act.get("molecule_chembl_id", "")
                if mid and mid not in seen:
                    seen[mid] = {
                        "molecule_chembl_id": mid,
                        "molecule_name": act.get("molecule_pref_name") or "",
                        "mechanism_of_action": "",
                        "max_phase": act.get("molecule_max_phase"),
                    }

        if not seen:
            return []

        return await _enrich_molecules(seen)

    except Exception as exc:
        logger.warning(f"_fetch_existing_drugs failed: {exc}")
        return []


def _first_sentences(text: str, n: int = 2, max_chars: int = 320) -> str:
    """Truncate text to at most n sentences or max_chars, whichever comes first."""
    if not text:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    result = " ".join(parts[:n])
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0] + "…"
    return result


def _last_name(author_string: str) -> str:
    if not author_string:
        return "Unknown"
    first = author_string.split(",")[0].strip()
    return first.split(" ")[-1]


async def _fetch_pubmed_papers(protein_name: str) -> list[PaperEntry]:
    """
    Query PubMed via NCBI E-utilities:
    - esearch for IDs (relevance-sorted)
    - efetch XML for title + abstract + DOI in one batch call
    """
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        query = (
            f'("{protein_name}"[Title/Abstract]) AND '
            f'(inhibitor OR "drug discovery" OR "virtual screening" OR "molecular docking")'
        )
        async with httpx.AsyncClient(timeout=20) as client:
            sr = await client.get(f"{base}/esearch.fcgi", params={
                "db": "pubmed", "term": query,
                "sort": "relevance", "retmax": "8", "retmode": "json",
            })
            ids = sr.json().get("esearchresult", {}).get("idlist", []) if sr.status_code == 200 else []
            if not ids:
                return []

            # Batch fetch full records as XML for title + abstract + DOI
            fr = await client.get(f"{base}/efetch.fcgi", params={
                "db": "pubmed", "id": ",".join(ids),
                "rettype": "abstract", "retmode": "xml",
            })

        papers: list[PaperEntry] = []
        if fr.status_code == 200:
            try:
                root = ET.fromstring(fr.text)
            except ET.ParseError:
                return []

            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()) if title_el is not None else ""

                # Abstract may have multiple labeled sections
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join(
                    ("".join(p.itertext())) for p in abstract_parts
                ).strip()

                # Authors
                author_els = article.findall(".//Author")
                last_names = [
                    (a.findtext("LastName") or "")
                    for a in author_els if a.findtext("LastName")
                ]
                first_last = last_names[0] if last_names else "Unknown"

                # Journal + year + volume
                journal = article.findtext(".//ISOAbbreviation") or article.findtext(".//Title") or ""
                year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate", "")[:4]
                volume = article.findtext(".//Volume") or ""

                # DOI
                doi_el = article.find(".//ArticleId[@IdType='doi']")
                doi = doi_el.text.strip() if doi_el is not None else None
                url = f"https://doi.org/{doi}" if doi else (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "")

                citation = f"{first_last} et al. {year} {journal}"
                if volume:
                    citation += f" {volume}"

                papers.append(PaperEntry(
                    citation=citation.strip(),
                    title=title,
                    summary=_first_sentences(abstract),
                    url=url,
                ))

        return papers
    except Exception as exc:
        logger.warning(f"_fetch_pubmed_papers failed: {exc}")
        return []


async def _fetch_europepmc_papers(protein_name: str) -> list[PaperEntry]:
    """Query Europe PMC (sorted by citation count) — returns abstracts and DOIs."""
    try:
        query = (
            f'("{protein_name}" OR "{protein_name} inhibitor") AND '
            f'(METHODS:"drug discovery" OR METHODS:"virtual screening" OR '
            f'METHODS:"molecular docking" OR TITLE:"inhibitor")'
        )
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": query, "format": "json", "pageSize": "10",
                        "sort": "CITED", "resultType": "core"},
            )
            r.raise_for_status()
            results = r.json().get("resultList", {}).get("result", [])

        papers: list[PaperEntry] = []
        for p in results:
            authors = p.get("authorString", "")
            year = str(p.get("pubYear", ""))
            journal = p.get("journalTitle", "") or p.get("journalAbbreviation", "")
            volume = p.get("journalInfo", {}).get("volume", "") if isinstance(p.get("journalInfo"), dict) else ""
            title = p.get("title", "").rstrip(".")
            abstract = p.get("abstractText", "")
            doi = p.get("doi", "")
            pmid = p.get("pmid", "")
            url = f"https://doi.org/{doi}" if doi else (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "")

            citation = f"{_last_name(authors)} et al. {year} {journal}"
            if volume:
                citation += f" {volume}"

            papers.append(PaperEntry(
                citation=citation.strip(),
                title=title,
                summary=_first_sentences(abstract),
                url=url,
            ))

        return papers
    except Exception as exc:
        logger.warning(f"_fetch_europepmc_papers failed: {exc}")
        return []


async def _fetch_key_papers(protein_name: str) -> list[PaperEntry]:
    """
    Query PubMed AND Europe PMC concurrently, merge + deduplicate by author+year.
    Europe PMC results lead (better citation ranking); PubMed fills gaps.
    Returns up to 10 PaperEntry objects with title, summary, and URL.
    """
    import asyncio as _aio
    pubmed, epmc = await _aio.gather(
        _fetch_pubmed_papers(protein_name),
        _fetch_europepmc_papers(protein_name),
    )

    seen_keys: set[str] = set()
    merged: list[PaperEntry] = []
    for paper in epmc + pubmed:  # epmc first (has citation count ranking)
        key = " ".join(paper.citation.split()[:2]).lower()
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(paper)

    return merged[:10]


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
    key_papers: list[PaperEntry],
) -> str:
    drugs_json = json.dumps(existing_drugs, indent=2) if existing_drugs else "None found in ChEMBL"
    if key_papers:
        papers_text = "\n".join(
            f"  - {p.citation}" + (f" — {p.title}" if p.title else "")
            for p in key_papers
        )
    else:
        papers_text = "  None found"
    return f"""You are an expert medicinal chemist and computational biologist.
Provide a structured literature review for the protein target: {protein_name}.

Available data retrieved from public databases:
- ChEMBL target info: {json.dumps(chembl_data, indent=2)[:1500]}
- PDB structure IDs: {pdb_ids}
- Known ChEMBL actives (pChEMBL >= 6): {known_actives}
- Clinical/approved drugs from ChEMBL mechanism table:
{drugs_json}
- Key publications from PubMed + Europe PMC (merged, sorted by relevance/citations):
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
    key_papers: list[PaperEntry],
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
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(text)

        # LLM returns key_papers as plain strings. Merge with the rich database
        # entries (which carry title/summary/url) by matching citation prefixes.
        llm_citations: list[str] = data.get("discovery_history", {}).get("key_papers", [])
        db_map = {" ".join(p.citation.split()[:2]).lower(): p for p in key_papers}
        merged_papers: list[dict] = []
        seen: set[str] = set()
        for raw in llm_citations:
            key = " ".join(str(raw).split()[:2]).lower()
            if key in db_map and key not in seen:
                seen.add(key)
                merged_papers.append(db_map[key].model_dump())
            elif key not in seen:
                seen.add(key)
                merged_papers.append({"citation": raw, "title": "", "summary": "", "url": ""})
        # Append any database papers the LLM didn't mention
        for p in key_papers:
            k = " ".join(p.citation.split()[:2]).lower()
            if k not in seen:
                seen.add(k)
                merged_papers.append(p.model_dump())

        data["discovery_history"]["key_papers"] = merged_papers
        return data
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
    key_papers: list[PaperEntry] | None = None,
) -> dict:
    has_struct = bool(pdb_ids)
    method = "hybrid" if has_struct and known_actives > 50 else ("sbdd" if has_struct else "lbdd")
    return {
        "description": (
            f"{protein_name} is a therapeutically relevant target implicated in multiple "
            "disease pathways. Structural and biochemical data indicate tractability for "
            "small-molecule modulation."
        ),
        "existing_therapies": existing_drugs or [],
        "discovery_history": {
            "lbdd_done": known_actives > 20,
            "sbdd_done": has_struct,
            "hybrid_done": has_struct and known_actives > 50,
            "key_papers": [p.model_dump() for p in (key_papers or [])],
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
                {
                    "citation": "Druker et al. 2001 N Engl J Med 344",
                    "title": "Efficacy and Safety of a Specific Inhibitor of the BCR-ABL Tyrosine Kinase in Chronic Myeloid Leukemia",
                    "summary": "Landmark Phase I trial of imatinib in CML patients showing 98% haematologic response rate. Established BCR-ABL kinase inhibition as a viable therapeutic strategy.",
                    "url": "https://doi.org/10.1056/NEJM200104053441401",
                },
                {
                    "citation": "Nagar et al. 2002 Science 296",
                    "title": "Structural Basis for the Autoinhibition of c-Abl Tyrosine Kinase",
                    "summary": "Crystal structure of Abl kinase in complex with imatinib revealing the inactive DFG-out conformation. Provided the structural rationale for drug selectivity.",
                    "url": "https://doi.org/10.1126/science.1070150",
                },
                {
                    "citation": "Zhao et al. 2019 J Med Chem 62",
                    "title": "Discovery of Potent and Selective Covalent Inhibitors of SHP2",
                    "summary": "Application of structure-based and ligand-based virtual screening to identify allosteric inhibitor scaffolds against a challenging PTP target.",
                    "url": "https://doi.org/10.1021/acs.jmedchem.9b00323",
                },
                {
                    "citation": "Schindler et al. 2000 Science 289",
                    "title": "Structural Mechanism for STI-571 Inhibition of Abelson Tyrosine Kinase",
                    "summary": "X-ray structure of imatinib bound to the Abl kinase domain demonstrating type-II binding mode. Underpins all subsequent SBDD campaigns against ABL1.",
                    "url": "https://doi.org/10.1126/science.289.5486.1938",
                },
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
