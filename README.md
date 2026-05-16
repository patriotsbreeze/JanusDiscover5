# JanusDiscover

**Unified Ligand-Based and Structure-Based Drug Discovery Platform**

[![JMLR/ICML Level](https://img.shields.io/badge/Science-JMLR%2FICML%20Level-blue)](https://github.com/patriotsbreeze/JanusDiscover5)
[![Mock Mode](https://img.shields.io/badge/Mock%20Mode-Available-yellow)](https://github.com/patriotsbreeze/JanusDiscover5)
[![Real Mode](https://img.shields.io/badge/Real%20Mode-OpenMM%208%20%2B%20Vina%201.2-green)](https://github.com/patriotsbreeze/JanusDiscover5)

JanusDiscover is an end-to-end computational drug discovery web application that bridges
**ligand-based drug discovery (LBDD)** and **structure-based drug discovery (SBDD)** into
a single intuitive platform for researchers.

---

## Features

| Step | Description | Tools |
|------|-------------|-------|
| 📚 Literature Review | AI synthesis of existing therapies and prior computational work | Claude + ChEMBL + RCSB PDB |
| 🧬 LBDD Pipeline | ECFP4 fingerprints + Random Forest ensemble | RDKit, scikit-learn |
| 🔬 SBDD Pipeline | AutoDock Vina 1.2 molecular docking | AutoDock Vina, RDKit |
| 🔀 Hybrid Mode | Consensus LBDD+SBDD scoring | Both |
| ⚡ MD Simulations | All-atom dynamics, AMBER ff14SB, TIP3P | OpenMM 8 |
| 📊 Validation Figures | ROC, BEDROC, EF1%, enrichment curves, RMSD, RMSF | matplotlib, seaborn |
| 📄 Manuscript | Auto-generated LaTeX (JMLR/ICML level) | jinja2, pdflatex |

---

## Quick Start

### Mock Mode (no external dependencies)

```bash
# Backend
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
# → http://localhost:3000
```

### Real Mode

```bash
conda install -c conda-forge rdkit openmm
pip install vina anthropic chembl-webresource-client
export ANTHROPIC_API_KEY=your_key
# Then same as above
```

### Docker

```bash
cp .env.example .env  # add your ANTHROPIC_API_KEY
docker-compose up --build
# → http://localhost:3000
```

---

## Workflow

```
1. Enter protein target (e.g. "ABL1 kinase", "EGFR", "CDK2")
2. AI literature review → existing therapies, PDB structures, known actives
3. Approve method (LBDD / SBDD / Hybrid) and dataset (ZINC-250k, FDA-approved, ChEMBL)
4. Virtual screening runs → top-20 hits + validation metrics
5. Download publication figures (ROC curve, enrichment, score distributions)
6. Optionally run MD simulation (e.g. 10 ns) on top hit
7. Download updated figures (RMSD, RMSF, Rg, energy, SASA)
8. Download LaTeX manuscript (.tex + .bib) — JMLR/ICML level
```

---

## Validation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| AUC-ROC | Global discrimination | > 0.7 |
| BEDROC (α=20) | Early enrichment (Truchon & Bayly 2007) | > 0.5 |
| EF at 1% | Fold-enrichment at top 1% | > 5× |
| EF at 5% | Fold-enrichment at top 5% | > 3× |
| Redocking RMSD | Pose accuracy vs co-crystal | < 2 Å success |

---

## Science Stack

- **RDKit** — cheminformatics, Morgan fingerprints, conformer generation
- **scikit-learn** — Random Forest, cross-validation, ROC analysis
- **AutoDock Vina 1.2** — molecular docking (Eberhardt et al. 2021)
- **OpenMM 8** — molecular dynamics (Eastman et al. 2024)
- **AMBER ff14SB** — protein force field (Maier et al. 2015)
- **TIP3P** — water model (Jorgensen et al. 1983)
- **Claude (Anthropic)** — literature synthesis and method recommendation

---

## Modes

**MOCK mode** — fully deterministic workflow using pre-computed realistic data.
No API keys or external dependencies needed. Same UI, same figures, same JSON.

**REAL mode** — live computation:
- ChEMBL REST API for known actives
- RCSB PDB for crystal structures
- AutoDock Vina for docking
- OpenMM for MD simulation
- Claude API for literature review

---

## Manuscript

Every session generates a complete **LaTeX manuscript** including:
- Methods section with proper citations (14 real references)
- All validation figures embedded
- Metrics table
- MD results section (if simulation was run)

Compile with:
```bash
pdflatex manuscript.tex && bibtex manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex
```

---

## Project Structure

```
backend/app/
  main.py             FastAPI entry point
  api/routes/         REST endpoints
  core/               Science modules (lit_review, lbdd, sbdd, md_simulation, manuscript)
  models/schemas.py   Pydantic schemas
frontend/src/
  pages/              React pages
  components/         UI components
  hooks/              Custom hooks (polling)
  api.ts              Typed API client
```

---

## License

MIT License — see LICENSE file.

## Citation

If you use JanusDiscover in research, please cite:

```
JanusDiscover: A Unified LBDD+SBDD Drug Discovery Platform with Integrated MD Validation.
patriotsbreeze, 2025. https://github.com/patriotsbreeze/JanusDiscover5
```

---

*For AI agents extending this project: see [AI_INSTRUCTIONS.md](AI_INSTRUCTIONS.md)*
