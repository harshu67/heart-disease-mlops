"""FastAPI application for heart disease risk prediction."""
from pathlib import Path
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-disease-api")

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])

MODEL_PATH = Path("models/heart_pipeline.joblib")
app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")
model = None

class PatientRecord(BaseModel):
    age: float = Field(..., example=63)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=3)
    trestbps: float = Field(..., example=145)
    chol: float = Field(..., example=233)
    fbs: int = Field(..., example=1)
    restecg: int = Field(..., example=0)
    thalach: float = Field(..., example=150)
    exang: int = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=0)
    ca: float = Field(..., example=0)
    thal: float = Field(..., example=1)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_COUNT.labels(request.url.path, request.method, response.status_code).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(duration)
    logger.info("%s %s completed with %s in %.4fs", request.method, request.url.path, response.status_code, duration)
    return response

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("Model file not found. Train model before starting production API.")

@app.get("/")
def root():
    return {"message": "Heart Disease Risk Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(record: PatientRecord):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    input_df = pd.DataFrame([record.model_dump()])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    risk_label = "heart_disease_risk" if prediction == 1 else "no_heart_disease_risk"
    return {"prediction": prediction, "risk_label": risk_label, "confidence": round(probability, 4)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
