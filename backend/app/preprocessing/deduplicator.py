import hashlib
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.models.document import Document


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path,
                       urlencode(sorted(query)), ""))


def deduplicate(documents: list[Document]) -> tuple[list[Document], list[dict]]:
    """Keep the first occurrence; retain duplicate provenance for inspection."""
    urls, texts = {}, {}
    kept, duplicates = [], []
    for document in documents:
        url = canonical_url(str(document.url))
        normalized = normalized_text(document.text)
        digest = text_hash(document.text)
        # Compare normalized text too, rather than relying on hash equality alone.
        key = (digest, normalized)
        duplicate_of = urls.get(url) or texts.get(key)
        if duplicate_of:
            duplicates.append({"document_id": str(document.document_id),
                               "duplicate_of": duplicate_of,
                               "reason": "same_url" if url in urls else "same_normalized_text",
                               "text_hash": digest})
        else:
            kept.append(document)
        representative = duplicate_of or str(document.document_id)
        urls[url] = representative
        texts[key] = representative
    return kept, duplicates
