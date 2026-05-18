"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def reset_rate_store():
    """Clear the in-memory rate store before each test so tests don't bleed into each other."""
    from backend.app.main import _rate_store
    _rate_store.clear()
    yield
    _rate_store.clear()
