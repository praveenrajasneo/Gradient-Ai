import json
from uuid import UUID, uuid4

import pytest

from app.classifiers.relevance import LABELS, RelevanceResult, WorkplaceRelevanceClassifier
from app.collectors.normalization import normalize_hacker_news
from app.models.document import Document
from app.preprocessing.cleaner import clean_documents, clean_text
from app.preprocessing.deduplicator import deduplicate, text_hash
from app.preprocessing.pipeline import run


def snapshot():
    return {"company": "Microsoft", "source": "hacker_news",
            "collected_at": "2026-09-25T12:00:00Z", "items": [
                {"id": "1", "type": "story", "raw": {
                    "title": "Microsoft announces Windows update", "created_at_i": 1700000000}},
                {"id": "2", "type": "comment", "raw": {
                    "story_title": "Layoffs", "comment_text": "<p>employees report mandatory office attendance</p>"}},
                {"id": "3", "type": "comment", "raw": {
                    "comment_text": "employees report mandatory office attendance"}},
                {"id": "4", "type": "comment", "raw": {
                    "comment_text": "<script>hidden</script>"}},
            ]}


def test_common_schema_stable_ids_and_missing_metadata():
    docs = normalize_hacker_news(snapshot())
    assert isinstance(docs[0].document_id, UUID)
    assert docs[0].document_id == normalize_hacker_news(snapshot())[0].document_id
    assert docs[0].published_at.tzinfo is not None
    assert docs[1].published_at is None
    assert docs[1].author is None
    assert docs[0].text == docs[0].title
    assert "Layoffs" not in docs[1].text
    assert str(docs[1].url).endswith("id=2")
    assert set(docs[0].model_dump()) == {
        "document_id", "company", "source", "source_type", "title", "text", "url",
        "author", "published_at", "collected_at",
    }
    with pytest.raises(ValueError):
        normalize_hacker_news({**snapshot(), "source": "other"})


def test_cleaning_example_blocks_entities_and_empty_removal():
    assert clean_text("<p>Great team!!!   But management is poor.</p>") == "Great team! But management is poor."
    assert clean_text("<p>A &amp; B</p><p>not <b>bad</b>.</p><script>ignore</script>") == "A & B not bad."
    docs, empty = clean_documents(normalize_hacker_news(snapshot()))
    assert len(docs) == 3
    assert len(empty) == 1
    assert "<p>" not in docs[1].text


def test_deduplication_by_url_or_normalized_text():
    original = normalize_hacker_news(snapshot())[0]
    same_url = original.model_copy(update={"document_id": uuid4(), "text": "Changed headline"})
    same_text = Document(**{**original.model_dump(), "document_id": uuid4(),
                           "url": "https://example.com/different",
                           "text": "MICROSOFT  announces Windows update"})
    distinct = Document(**{**original.model_dump(), "document_id": uuid4(),
                          "url": "https://news.ycombinator.com/item?id=99",
                          "text": "Another independent comment"})
    kept, duplicates = deduplicate([original, same_url, same_text, distinct])
    assert len(kept) == 2
    assert [item["reason"] for item in duplicates] == ["same_url", "same_normalized_text"]
    assert text_hash(" A  B ") == text_hash("a b")
    assert all(item["duplicate_of"] == str(original.document_id) for item in duplicates)


def test_relevance_label_order_threshold_and_long_documents():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(800))

        def decode(self, tokens, **kwargs):
            return str(tokens[0])

    class FakePipeline:
        tokenizer = Tokenizer()

        def __call__(self, chunks, **kwargs):
            assert chunks == ["0", "320", "640"]
            return [{"labels": LABELS[::-1], "scores": [1 - score, score]}
                    for score in (0.1, 0.7, 0.2)]

    classifier = WorkplaceRelevanceClassifier.__new__(WorkplaceRelevanceClassifier)
    classifier.threshold = 0.7
    classifier.pipeline = FakePipeline()
    assert classifier.classify("long text").model_dump() == {"relevant": True, "score": 0.7}
    assert classifier.classify(" ").relevant is False
    with pytest.raises(ValueError):
        WorkplaceRelevanceClassifier(2)


def test_pipeline_counts_audit_and_common_outputs(tmp_path):
    class Classifier:
        threshold = 0.7
        revision = "test-model"

        def classify(self, text):
            score = 0.9 if "employees" in text else 0.1
            return RelevanceResult(relevant=score >= self.threshold, score=score)

    source = tmp_path / "raw.json"
    source.write_text(json.dumps(snapshot()), encoding="utf-8")
    output = tmp_path / "processed"
    counts = run(source, output, Classifier())
    assert counts == {"raw": 4, "empty_removed": 1, "duplicates_removed": 1,
                      "cleaned_unique": 2, "workplace_relevant": 1, "irrelevant": 1}
    assert json.loads(source.read_text()) == snapshot()
    relevant = json.loads((output / "relevant.json").read_text())
    assert len(relevant) == 1
    Document.model_validate(relevant[0])
    audit = json.loads((output / "audit.json").read_text())
    assert len(audit["decisions"]) == 2
    assert len(audit["duplicates"]) == 1
    assert (output / "review.csv").is_file()
