"""
FastAPI application entry point.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from predictor import get_predictor
from router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model into memory on startup so first request is fast."""
    print("🚀 Loading churn model...")
    get_predictor()  # warm the singleton
    print("✅ Model ready.")
    yield
    print("👋 Shutting down.")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Predicts customer churn from RFM-based behavioral features.",
    lifespan=lifespan,
)

# CORS — permissive for portfolio demo. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["system"])
def root():
    """Root — points to API docs."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
