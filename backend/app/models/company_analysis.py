from pydantic import BaseModel, Field, field_validator


class CompanyAnalysisRequest(BaseModel):
    company: str = Field(min_length=1)
    role: str = Field(min_length=1)
    experience: int = Field(ge=0, strict=True)
    location: str | None = None

    @field_validator("company", "role")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value
