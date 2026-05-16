"""
JanusDiscover FastAPI application entry point.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes.discovery import router as discovery_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="JanusDiscover",
    description="Unified LBDD + SBDD Drug Discovery Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(discovery_router)

# Serve static figure/manuscript files
static_dir = Path(os.getenv("STATIC_DIR", "static"))
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "figures").mkdir(exist_ok=True)
(static_dir / "manuscripts").mkdir(exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "JanusDiscover"}
