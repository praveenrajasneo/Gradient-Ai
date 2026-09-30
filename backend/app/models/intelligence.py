from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Aspect = Literal['workload', 'work_life_balance', 'management', 'career_growth',
                 'promotion', 'compensation', 'job_security', 'team_culture']
Sentiment = Literal['positive', 'negative', 'neutral', 'mixed']


class PainPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect: Aspect
    sentiment: Literal['negative'] = 'negative'
    pain_point: str = Field(min_length=1, max_length=160)
    evidence_quote: str = Field(min_length=1, max_length=2000)


class ModelExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pain_points: list[PainPoint] = Field(max_length=12)


class Extraction(ModelExtraction):
    rejections: list[dict] = Field(default_factory=list)


class ExtractionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    aspect: Aspect
