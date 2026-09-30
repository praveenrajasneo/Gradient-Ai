from fastapi import FastAPI

from app.api.analyze import router
from app.api.pain_points import router as pain_points_router
from app.api.evidence import router as evidence_router

app = FastAPI(title="Gradient Nova AI")
app.include_router(router)
app.include_router(pain_points_router)
app.include_router(evidence_router)


@app.get("/")
def home():
    return {"message": "Gradient Nova AI backend running"}
