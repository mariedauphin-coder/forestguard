"""FastAPI application factory for ForestGuard."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import detection_router, alerts_router
from .dependencies import get_model
from .schemas import HealthResponse
from .. import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ForestGuard API starting — loading model…")
    get_model()   # warm up the singleton
    logger.info("Model ready.")
    yield
    logger.info("ForestGuard API shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ForestGuard",
        description=(
            "Real-time deforestation detection via Sentinel-1 SAR + "
            "Sentinel-2 optical fusion using a Siamese MobileNetV2 + U-Net."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = str(
            round((time.perf_counter() - t0) * 1000, 1)
        )
        return response

    app.include_router(detection_router)
    app.include_router(alerts_router)

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health():
        from .dependencies import is_model_ready
        return HealthResponse(
            status="ok",
            model_loaded=is_model_ready(),
            version=__version__,
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check API logs."},
        )

    return app


# Entry point for `uvicorn forestguard.api.main:app`
app = create_app()
