from fastapi import APIRouter

from app.models.company_analysis import CompanyAnalysisRequest

router = APIRouter(prefix="/api")


@router.post("/analyze", response_model=CompanyAnalysisRequest)
def analyze(request: CompanyAnalysisRequest) -> CompanyAnalysisRequest:
    """Echo validated input; collection is a separate command for now."""
    return request
