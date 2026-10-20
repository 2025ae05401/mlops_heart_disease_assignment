"""FastAPI serving layer; educational demonstration only."""
import logging, os, time
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from heartml.data import FEATURES

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger = logging.getLogger("heartml.api")
REQUEST_COUNT = Counter("heartml_api_requests_total","Total API requests",["method","endpoint","status"])
REQUEST_LATENCY = Histogram("heartml_api_request_duration_seconds","Request duration",["endpoint"])
MODEL_PATH = Path(os.getenv("MODEL_PATH","artifacts/model.joblib"))
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
app = FastAPI(title="Heart Disease Classification API", version="1.0.0",
              description="Educational classifier; not for medical decisions.")

class PatientFeatures(BaseModel):
    age: float = Field(ge=1,le=120)
    sex: int = Field(ge=0,le=1)
    cp: int = Field(ge=1,le=4)
    trestbps: float = Field(ge=0,le=300)
    chol: float = Field(ge=0,le=1000)
    fbs: int = Field(ge=0,le=1)
    restecg: int = Field(ge=0,le=2)
    thalach: float = Field(ge=0,le=300)
    exang: int = Field(ge=0,le=1)
    oldpeak: float = Field(ge=-10,le=20)
    slope: int = Field(ge=1,le=3)
    ca: float = Field(ge=0,le=3)
    thal: float = Field(ge=3,le=7)
    def as_frame(self):
        return pd.DataFrame([{name:getattr(self,name) for name in FEATURES}],columns=FEATURES)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start=time.perf_counter(); status=500
    try:
        response=await call_next(request); status=response.status_code; return response
    finally:
        duration=time.perf_counter()-start
        REQUEST_COUNT.labels(request.method,request.url.path,str(status)).inc()
        REQUEST_LATENCY.labels(request.url.path).observe(duration)
        logger.info("method=%s path=%s status=%s duration_seconds=%.4f",
                    request.method,request.url.path,status,duration)

@app.get("/health")
def health():
    return {"status":"ok" if model is not None else "model_not_loaded","model_loaded":model is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503,detail=f"Model artifact not found at {MODEL_PATH}")
    frame=patient.as_frame()
    prediction=int(model.predict(frame)[0])
    probability=float(model.predict_proba(frame)[0,1])
    return {"prediction":prediction,
            "prediction_label":"disease_present" if prediction else "disease_absent",
            "probability_disease_present":round(probability,6),
            "notice":"Educational demonstration only; not a clinical diagnosis."}
