"""
API router — defines all churn prediction endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from config import settings
from predictor import ChurnPredictor, get_predictor
from schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check. Used by Cloud Run."""
    return HealthResponse(version=settings.API_VERSION)


@router.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
def model_info(predictor: ChurnPredictor = Depends(get_predictor)) -> ModelInfoResponse:
    """Return model metadata: training date, input shape, feature names."""
    return ModelInfoResponse(**predictor.info())


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    features: CustomerFeatures,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Predict churn for a single customer."""
    try:
        result = predictor.predict_one(features.model_dump())
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(
    request: BatchPredictionRequest,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    """Predict churn for multiple customers in a single request."""
    try:
        feature_dicts = [c.model_dump() for c in request.customers]
        results = predictor.predict_batch(feature_dicts, threshold=request.threshold)
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            count=len(results),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
