import httpx
from fastapi import APIRouter, HTTPException

from app.models.intelligence import Extraction, ExtractionRequest
from app.models.llm import LLMService

router = APIRouter(prefix="/api")


@router.post("/pain-points/extract", response_model=Extraction)
def extract(request: ExtractionRequest):
    try:
        return LLMService().extract([request.model_dump()])
    except httpx.HTTPError:
        raise HTTPException(503, "Ollama is unavailable or timed out; start it and pull qwen3:4b")
    except (ValueError, KeyError):
        raise HTTPException(502, "Model output failed structured/evidence validation")
