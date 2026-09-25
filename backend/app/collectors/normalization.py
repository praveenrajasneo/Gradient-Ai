"""Map HN snapshots into the common schema without modifying raw data."""

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from app.models.document import Document


def normalize_hacker_news(snapshot: dict) -> list[Document]:
    if snapshot.get("source") != "hacker_news":
        raise ValueError("Expected a Hacker News snapshot")
    documents = []
    for item in snapshot["items"]:
        raw = item["raw"]
        url = f"https://news.ycombinator.com/item?id={item['id']}"
        published = raw.get("created_at")
        if not published and raw.get("created_at_i") is not None:
            published = datetime.fromtimestamp(raw["created_at_i"], tz=timezone.utc)
        if item["type"] == "comment":
            title = raw.get("story_title") or ""
            text = raw.get("comment_text") or ""
        elif item["type"] == "story":
            title = raw.get("title") or ""
            text = "\n".join(part for part in (title, raw.get("story_text")) if part)
        else:
            raise ValueError(f"Unsupported HN item type: {item['type']}")
        documents.append(Document(
            document_id=uuid5(NAMESPACE_URL, url), company=snapshot["company"],
            source="hacker_news", source_type="community", title=title, text=text,
            url=url, author=raw.get("author"), published_at=published,
            collected_at=snapshot["collected_at"],
        ))
    return documents
