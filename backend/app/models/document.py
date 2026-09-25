"""Source-independent contract for all future collector adapters."""

from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    company: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    title: str = ""
    text: str
    url: HttpUrl
    author: str | None = None
    published_at: AwareDatetime | None = None
    collected_at: AwareDatetime
