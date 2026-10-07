"""
FastAPI router for ML model endpoints (Milestone 2+).

Currently returns stubs. The actual Siamese U-Net will be
wired in during Milestone 2.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/models")
async def list_models():
    """List available change detection models."""
    return {
        "models": [
            {
                "name": "classical",
                "description": "Classical image differencing (no ML)",
                "available": True,
                "requires_gpu": False,
            },
            {
                "name": "siamese_unet",
                "description": "Siamese U-Net with shared encoder (Milestone 2)",
                "available": False,
                "requires_gpu": False,
            },
            {
                "name": "changeformer",
                "description": "Transformer-based change detection (Milestone 4)",
                "available": False,
                "requires_gpu": True,
            },
        ]
    }


class PredictRequest(BaseModel):
    before_path: str
    after_path: str
    model_name: str = "classical"
    device: str = "auto"


@router.post("/models/predict")
async def predict(req: PredictRequest):
    """Run a change detection model on a before/after pair."""
    if req.model_name != "classical":
        raise HTTPException(
            501,
            f"Model '{req.model_name}' not yet implemented. "
            "Use /api/analysis/compare for classical detection.",
        )

    return {
        "status": "error",
        "message": "Use POST /api/analysis/compare for classical change detection.",
    }
