import re
import unicodedata

from bs4 import BeautifulSoup

from app.models.document import Document


def clean_text(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    for element in soup.find_all(["p", "div", "br", "li", "pre", "blockquote"]):
        element.insert_before(" ")
        element.insert_after(" ")
    text = unicodedata.normalize("NFKC", soup.get_text())
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_documents(documents: list[Document]) -> tuple[list[Document], list[dict]]:
    kept, rejected = [], []
    for document in documents:
        text = clean_text(document.text)
        if not text:
            rejected.append({"document_id": str(document.document_id), "reason": "empty_text"})
            continue
        kept.append(document.model_copy(update={"title": clean_text(document.title), "text": text}))
    return kept, rejected
