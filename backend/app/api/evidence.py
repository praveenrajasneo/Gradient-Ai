"""Statistics and evidence search, separate from the original input echo endpoint."""
from typing import Literal
from uuid import UUID
import subprocess

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.database.statistics import statistics
from app.embeddings.service import ModelExecutionError
from app.rag.pipeline import answer
from app.retrieval.pipeline import retrieve
import psycopg

router = APIRouter(prefix='/api', tags=['Company evidence'])


class Question(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    company: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=500)
    analysis_id: UUID | None = None


class SearchRequest(Question):
    mode: Literal['dense', 'hybrid'] = 'hybrid'
    top_k: int = Field(default=20, ge=1, le=20)
    rerank: bool = False


def guarded(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ModelExecutionError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (httpx.HTTPError, OSError, psycopg.Error, subprocess.TimeoutExpired) as exc:
        raise HTTPException(503, 'A local dependency is unavailable; check PostgreSQL, Qdrant, Ollama and model files') from exc
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get('/companies/{company}/statistics')
def company_statistics(company: str, analysis_id: UUID | None = None):
    return guarded(statistics, company, str(analysis_id) if analysis_id else None)


@router.post('/retrieve')
def search(request: SearchRequest):
    return guarded(retrieve, request.company, request.question,
                   str(request.analysis_id) if request.analysis_id else None,
                   request.mode, request.top_k, request.rerank)


@router.post('/ask')
def ask(request: Question):
    return guarded(answer, request.company, request.question,
                   str(request.analysis_id) if request.analysis_id else None)
