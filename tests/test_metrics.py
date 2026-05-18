"""
Unit tests for BEDROC, enrichment factor, and SMILES validation.
These run with no external dependencies (numpy + scikit-learn only).
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.core.lbdd import bedroc_score, enrichment_factor, _validate_smiles


# ── BEDROC ────────────────────────────────────────────────────────────────────

class TestBEDROC:
    def test_perfect_classifier(self):
        """All actives ranked first → BEDROC close to 1."""
        n = 100
        y = np.array([1] * 10 + [0] * 90)
        scores = np.array([1.0] * 10 + [0.0] * 90)
        score = bedroc_score(y, scores)
        assert score > 0.9, f"Expected > 0.9, got {score}"

    def test_random_classifier(self):
        """Random scores → BEDROC near the theoretical random baseline (~0.115 for Ra=0.1, α=20).
        It is NOT 0.5 — BEDROC is not symmetric around 0.5.
        The key property is that a random classifier should score well below a perfect one.
        """
        rng = np.random.default_rng(0)
        n, n_act = 500, 50
        y = np.array([1] * n_act + [0] * (n - n_act))
        scores = rng.random(n)
        score = bedroc_score(y, scores)
        # Theoretical random BEDROC for Ra=0.1, alpha=20 ≈ 0.115
        # Allow ±0.15 statistical variation for n=500
        assert 0.0 < score < 0.4, f"Random BEDROC should be low (≈0.115 ± variance), got {score}"

    def test_worst_classifier(self):
        """All actives ranked last → BEDROC close to 0."""
        n = 100
        y = np.array([1] * 10 + [0] * 90)
        scores = np.array([0.0] * 10 + [1.0] * 90)
        score = bedroc_score(y, scores)
        assert score < 0.1, f"Expected < 0.1, got {score}"

    def test_returns_float_in_unit_interval(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 200)
        scores = rng.random(200)
        score = bedroc_score(y, scores)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── Enrichment Factor ─────────────────────────────────────────────────────────

class TestEnrichmentFactor:
    def test_perfect_at_1pct(self):
        """If all actives are at the top 1%, EF1% = 1/ra."""
        n, n_act = 1000, 10
        ra = n_act / n
        y = np.array([1] * n_act + [0] * (n - n_act))
        scores = np.array([1.0] * n_act + [0.0] * (n - n_act))
        ef = enrichment_factor(y, scores, 0.01)
        expected = 1.0 / ra  # = 100
        assert abs(ef - expected) < 1.0, f"Expected ≈{expected:.1f}, got {ef:.1f}"

    def test_random_ef_near_one(self):
        """Random ranking → EF averaged over multiple seeds should converge to ~1.
        Individual seeds can deviate (especially EF1% with only ~20 top compounds).
        """
        n, n_act = 2000, 200
        y = np.array([1] * n_act + [0] * (n - n_act))
        for frac in [0.05, 0.10]:  # skip 1% — too high-variance at small n
            ef_vals = []
            for seed in range(10):
                rng = np.random.default_rng(seed)
                scores = rng.random(n)
                ef_vals.append(enrichment_factor(y, scores, frac))
            mean_ef = np.mean(ef_vals)
            assert 0.6 < mean_ef < 1.6, \
                f"Mean EF at {frac} over 10 seeds should be ≈1, got {mean_ef:.3f}"

    def test_ef_decreases_at_larger_fractions(self):
        """For a good model, EF at 1% > EF at 10%."""
        n, n_act = 500, 25
        y = np.array([1] * n_act + [0] * (n - n_act))
        scores = np.concatenate([
            np.linspace(1.0, 0.6, n_act),
            np.linspace(0.4, 0.0, n - n_act),
        ])
        ef1 = enrichment_factor(y, scores, 0.01)
        ef10 = enrichment_factor(y, scores, 0.10)
        assert ef1 >= ef10, f"EF1% ({ef1:.2f}) should be ≥ EF10% ({ef10:.2f})"


# ── SMILES Validation ─────────────────────────────────────────────────────────

class TestSMILESValidation:
    def test_valid_smiles_pass(self):
        valid = [
            "CCO",
            "c1ccccc1",
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            "CN1CCC[C@H]1c2cccnc2",
        ]
        result = _validate_smiles(valid)
        assert len(result) == len(valid)

    def test_invalid_smiles_filtered(self):
        mixed = ["CCO", "INVALID###", "", "   ", "c1ccccc1"]
        result = _validate_smiles(mixed)
        assert "INVALID###" not in result
        assert "" not in result
        assert "CCO" in result
        assert "c1ccccc1" in result

    def test_empty_list(self):
        assert _validate_smiles([]) == []

    def test_all_invalid(self):
        assert _validate_smiles(["??", "XYZ", "123"]) == []

    def test_whitespace_stripped(self):
        result = _validate_smiles(["  CCO  ", "\tCCN\n"])
        assert len(result) == 2


# ── LaTeX Escaping ────────────────────────────────────────────────────────────

class TestLatexEscape:
    def test_special_chars_escaped(self):
        from backend.app.core.manuscript import _latex_escape
        assert _latex_escape("10% yield") == r"10\% yield"
        assert _latex_escape("A & B") == r"A \& B"
        assert _latex_escape("$money$") == r"\$money\$"
        assert _latex_escape("a_b") == r"a\_b"

    def test_normal_text_unchanged(self):
        from backend.app.core.manuscript import _latex_escape
        assert _latex_escape("ABL1 kinase") == "ABL1 kinase"
        assert _latex_escape("CDK2") == "CDK2"

    def test_empty_string(self):
        from backend.app.core.manuscript import _latex_escape
        assert _latex_escape("") == ""
