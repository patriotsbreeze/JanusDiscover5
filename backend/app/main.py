"""
JanusDiscover FastAPI application entry point.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes.discovery import router as discovery_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Simple token-bucket rate limiter middleware ───────────────────────────────
# Limits POST /api/* to 20 req/min per IP (burst up to 20, refill 1/3s).

_RATE_WINDOW = 60          # seconds
_RATE_MAX = int(os.getenv("RATE_LIMIT_MAX", "20"))  # requests per window
_rate_store: dict[str, list[float]] = collections.defaultdict(list)


class RateLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            if request.method == "POST" and request.url.path.startswith("/api/"):
                ip = request.client.host if request.client else "unknown"
                now = time.monotonic()
                window_start = now - _RATE_WINDOW
                hits = _rate_store[ip]
                # Prune old entries
                _rate_store[ip] = [t for t in hits if t > window_start]
                if len(_rate_store[ip]) >= _RATE_MAX:
                    response = JSONResponse(
                        {"detail": "Rate limit exceeded. Max 20 requests/minute."},
                        status_code=429,
                    )
                    await response(scope, receive, send)
                    return
                _rate_store[ip].append(now)
        await self.app(scope, receive, send)


# ── Lifespan (session cleanup) ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_session_cleanup_loop())
    yield
    task.cancel()


async def _session_cleanup_loop():
    from .core.session import _store
    SESSION_TTL = 7200  # 2 hours
    while True:
        await asyncio.sleep(1800)
        now = time.time()
        stale = [sid for sid, s in list(_store.items())
                 if getattr(s, "_created_at", now) < now - SESSION_TTL]
        for sid in stale:
            _store.pop(sid, None)
        if stale:
            logger.info(f"Purged {len(stale)} stale sessions")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="JanusDiscover",
    description="Unified LBDD + SBDD Drug Discovery Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiting
app.add_middleware(RateLimitMiddleware)

# CORS — restrict to configured frontend origins (wildcard only in dev)
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
allowed_origins = [o.strip() for o in _allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(discovery_router)

# Static dirs
static_dir = Path(os.getenv("STATIC_DIR", "static"))
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "figures").mkdir(exist_ok=True)
(static_dir / "manuscripts").mkdir(exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "JanusDiscover"}
