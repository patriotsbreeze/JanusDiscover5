# JanusDiscover — AI Instructions for Replication and Extension

This document is written for another AI agent (Claude, GPT-4, Gemini, etc.) who needs to
understand, replicate, extend, or maintain the JanusDiscover project. Read this completely
before modifying any code.

---

## 1. What Is JanusDiscover?

JanusDiscover is a full-stack web application for computational drug discovery that:

1. Takes a **protein target name** as input from a researcher
2. Runs an **AI-powered literature review** (ChEMBL + RCSB PDB + Claude LLM)
3. Recommends and executes either **LBDD**, **SBDD**, or **hybrid** virtual screening
4. Produces **publication-quality validation figures** (ROC, enrichment, BEDROC, etc.)
5. Optionally runs **molecular dynamics simulations** (OpenMM 8)
6. Generates a **LaTeX manuscript** (JMLR/ICML level) with all methods and results
7. Operates in **mock mode** (no external deps) or **real mode** (live computation)

---

## 2. Repository Structure

```
JanusDiscover/
├── backend/
│   └── app/
│       ├── main.py              ← FastAPI app entry point
│       ├── api/routes/
│       │   └── discovery.py     ← All API routes
│       ├── core/
│       │   ├── lit_review.py    ← ChEMBL + RCSB + Claude LLM synthesis
│       │   ├── lbdd.py          ← ECFP4 fingerprints + Random Forest pipeline
│       │   ├── sbdd.py          ← AutoDock Vina docking pipeline
│       │   ├── md_simulation.py ← OpenMM 8 MD simulation pipeline
│       │   ├── manuscript.py    ← LaTeX manuscript generation
│       │   └── session.py       ← In-memory session store
│       └── models/
│           └── schemas.py       ← Pydantic request/response schemas
├── frontend/
│   └── src/
│       ├── App.tsx              ← Root component
│       ├── api.ts               ← Axios API client + TypeScript types
│       ├── hooks/usePolling.ts  ← SSE-style polling hook
│       ├── pages/
│       │   └── DiscoveryPage.tsx ← Main app page (orchestrates all steps)
│       └── components/
│           ├── LandingHero.tsx   ← Marketing landing page
│           ├── LitReviewPanel.tsx
│           ├── ApprovalPanel.tsx
│           ├── ResultsPanel.tsx
│           ├── MDPanel.tsx
│           └── ManuscriptPanel.tsx
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.backend
└── frontend/Dockerfile.frontend
```

---

## 3. User Workflow (Step by Step)

```
User → enter protein name + choose mock/real
     → POST /api/lit-review
     → Poll /api/session/{id} until status = "awaiting_approval"
     → Review LitReviewResult (therapies, history, PDB IDs, recommendation)
     → POST /api/approve {method, dataset, run_mode}
     → Poll until status = "awaiting_md_decision"
     → Review DiscoveryResult (metrics, top hits, figures)
     → POST /api/md {run_md, duration, run_mode}
     → Poll until status = "manuscript_done"
     → Download figures (GET /api/figures/{session_id}/{filename})
     → Download manuscript (GET /api/manuscript/{session_id}/tex|pdf|bib)
```

---

## 4. API Reference

### POST /api/lit-review
```json
{ "protein_name": "ABL1 kinase", "run_mode": "mock" }
```
Returns: `{ "session_id": "uuid", "status": "started" }`

### GET /api/session/{session_id}
Returns full `SessionState` including nested results as they complete.

Key status values:
- `lit_review_running` → polling
- `awaiting_approval` → lit review done, show results to user
- `discovery_running` → polling
- `awaiting_md_decision` → discovery done, show results to user
- `md_running` → polling
- `manuscript_done` → everything done
- `error` → check `error_message`

### POST /api/approve
```json
{
  "session_id": "uuid",
  "approved": true,
  "chosen_method": "hybrid",   // "lbdd" | "sbdd" | "hybrid"
  "dataset": "zinc_250k",       // "fda_approved" | "zinc_250k" | "chembl" | "custom"
  "run_mode": "mock"
}
```

### POST /api/md
```json
{
  "session_id": "uuid",
  "run_md": true,
  "duration": { "value": 10, "unit": "ns" },
  "run_mode": "mock"
}
```

### GET /api/figures/{session_id}/{filename}
Returns PNG image.

### GET /api/manuscript/{session_id}/tex|pdf|bib
Returns file download.

---

## 5. Science & Methodology (DO NOT CHANGE without updating citations)

### LBDD Pipeline
- **Fingerprints**: Morgan/ECFP4, radius=2, 2048 bits (RDKit)
  - Citation: Rogers & Hahn (2010) J Chem Inf Model 50:742
- **Model**: Random Forest, 500 trees, class-balanced, 5-fold stratified CV
  - Citation: Breiman (2001) Mach Learn 45:5
- **Validation metrics** (critical — these are the published standard):
  - **AUC-ROC**: global discrimination
  - **BEDROC** (α=20): Boltzmann-Enhanced Discrimination of ROC; emphasises early enrichment
    - Citation: Truchon & Bayly (2007) J Chem Inf Model 47:488
  - **EF1%, EF5%**: enrichment factor at top 1%/5% of ranked list
  - **DUD-E benchmark** for external validation: Mysinger et al. (2012) J Med Chem 55:6582

### SBDD Pipeline
- **Docking engine**: AutoDock Vina 1.2
  - Citation: Eberhardt et al. (2021) J Chem Inf Model 61:3891
  - Citation: Trott & Olson (2010) J Comput Chem 31:455
- **Receptor prep**: add hydrogens, remove waters, Gasteiger charges
- **Ligand prep**: ETKDG conformer generation + MMFF94 minimisation (RDKit)
- **Validation**: redocking RMSD vs co-crystal pose; success = RMSD < 2 Å
- **Same metrics as LBDD** (AUC-ROC, BEDROC, EF1%, EF5%)

### Hybrid Mode
- Consensus scoring: `score = 0.4 × LBDD_zscore + 0.6 × SBDD_zscore`
- Rationale: SBDD weighted higher when crystal structures are available

### MD Simulation
- **Engine**: OpenMM 8
  - Citation: Eastman et al. (2024) J Phys Chem B 128:109
- **Force field**: AMBER ff14SB (Maier et al. 2015) + TIP3P water (Jorgensen 1983)
- **Protocol**: minimise → NVT 100 ps (Langevin thermostat, 300K, γ=1/ps) → NPT 100 ps (Monte Carlo barostat, 1 atm) → production
- **Step size**: 2 fs, HBonds constrained
- **Validation figures** (all required for publication):
  1. RMSD vs time (protein backbone + ligand heavy atoms)
  2. RMSF per residue
  3. Radius of gyration vs time
  4. Potential energy vs time
  5. SASA (Solvent Accessible Surface Area) vs time
  6. RMSD distribution violin plot

---

## 6. Mock Mode Implementation

Mock mode generates **statistically realistic data** using NumPy random number generators
with fixed seeds (reproducible). Key mock data characteristics:
- AUC-ROC: typically 0.80–0.92 (realistic for a good VS campaign)
- BEDROC: 0.55–0.80
- EF1%: 8–25× (typical for high-quality LBDD/SBDD)
- RMSD: beta/exponential distribution, ~40–60% success rate (< 2 Å)
- MD RMSD: starts near 0, plateaus around 1–2 Å with thermal noise

All mock pipelines return the same JSON structure as real pipelines.
The `mock: true` flag in responses indicates simulated data.

---

## 7. Adding a New Discovery Method

1. Create `backend/app/core/new_method.py` with `async def run_new_method(...)` function
2. Return the same dict schema as `lbdd.py` / `sbdd.py`
3. Add `new_method` to `DiscoveryMethod` enum in `schemas.py`
4. Add routing in `discovery.py` `_run_discovery()` function
5. Add method card in `frontend/src/components/ApprovalPanel.tsx` METHODS array
6. Add color/label mappings in `ResultsPanel.tsx` and `DiscoveryPage.tsx`

---

## 8. Adding a New Dataset

1. Add to `Dataset` enum in `schemas.py`
2. Implement `_load_dataset()` in `lbdd.py` and `sbdd.py` to return SMILES list
3. Add UI card in `ApprovalPanel.tsx` DATASETS array

---

## 9. Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | For Claude LLM in real mode lit review | (empty = fallback) |
| `STATIC_DIR` | Root for figures/manuscripts | `static` |
| `FIGURES_DIR` | Where figures are saved | `static/figures` |
| `MANUSCRIPT_DIR` | Where manuscripts are saved | `static/manuscripts` |

---

## 10. Validation Standards (JMLR/ICML Level)

For a submission to be accepted at JMLR/ICML level, the following are required:
- [ ] AUC-ROC reported with 95% confidence interval (5-fold CV)
- [ ] BEDROC(α=20) as primary early-enrichment metric (not just AUC)
- [ ] EF at 1% and 5% reported
- [ ] Redocking RMSD success rate for SBDD
- [ ] External test set validation (DUD-E or similar benchmark)
- [ ] MD simulation convergence confirmed (plateau RMSD < 2–3 Å)
- [ ] All figures at 150 DPI minimum with axis labels and captions
- [ ] Bibliography contains only real, peer-reviewed references

---

## 11. Running Locally

### Quick start (mock mode, no external deps)
```bash
# Backend
cd /path/to/JanusDiscover
pip install -r requirements.txt
cd backend && uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### Real mode
```bash
# Install additional deps:
conda install -c conda-forge rdkit openmm
pip install vina chembl-webresource-client anthropic

# Set API key:
export ANTHROPIC_API_KEY=your_key_here

# Run same as above
```

### Docker
```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
docker-compose up --build
# → http://localhost:3000
```

---

## 12. Known Limitations and Future Work

1. **Real mode dataset loading**: Currently uses mock SMILES. Implement proper ZINC/ChEMBL
   download in `_load_dataset()` in `lbdd.py` and `sbdd.py`.
2. **AlphaFold integration**: Add AlphaFold2 structure prediction for targets without PDB entries.
3. **Deep learning scoring**: Replace RF with graph neural networks (e.g., AttentiveFP, DimeNet).
4. **Active learning**: Implement feedback loop with experimental results.
5. **GPU MD**: Automatic CUDA detection is implemented; ensure CUDA toolkit matches OpenMM build.
6. **Session persistence**: Currently in-memory; add Redis/SQLite for production.
7. **PDF compilation**: `pdflatex` must be installed on the server for PDF output.

---

## 13. Citation Policy

All scientific claims in the code and manuscript must cite real papers.
The BibTeX in `backend/app/core/manuscript.py` contains 14 real citations.
Do NOT invent paper titles, authors, or DOIs.

---

*Generated by JanusDiscover v1.0 — a unified LBDD+SBDD drug discovery platform.*
