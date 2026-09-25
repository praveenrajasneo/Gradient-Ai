"""Run from backend: python -m app.preprocessing.pipeline --input <snapshot>."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.classifiers.relevance import HYPOTHESIS, LABELS, MODEL_ID, WorkplaceRelevanceClassifier
from app.collectors.normalization import normalize_hacker_news
from app.preprocessing.cleaner import clean_documents
from app.preprocessing.deduplicator import deduplicate

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(input_path: Path, output_dir: Path, classifier) -> dict:
    snapshot = json.loads(input_path.read_text(encoding="utf-8"))
    normalized = normalize_hacker_news(snapshot)
    cleaned, empty = clean_documents(normalized)
    unique, duplicates = deduplicate(cleaned)
    print(f"Raw {len(normalized)} -> nonempty {len(cleaned)} -> unique {len(unique)}", flush=True)
    decisions, relevant = [], []
    for index, document in enumerate(unique, 1):
        result = classifier.classify(document.text)
        decisions.append({"document_id": str(document.document_id), **result.model_dump()})
        if result.relevant:
            relevant.append(document)
        if index % 25 == 0 or index == len(unique):
            print(f"Classified {index}/{len(unique)}; kept {len(relevant)}", flush=True)
    counts = {"raw": len(normalized), "empty_removed": len(empty),
              "duplicates_removed": len(duplicates), "cleaned_unique": len(unique),
              "workplace_relevant": len(relevant), "irrelevant": len(unique) - len(relevant)}
    metadata = {"company": snapshot["company"], "processed_at": datetime.now(timezone.utc).isoformat(),
                "input_file": str(input_path.resolve()), "model": MODEL_ID,
                "model_revision": classifier.revision, "threshold": classifier.threshold,
                "labels": LABELS, "hypothesis_template": HYPOTHESIS,
                "chunk_tokens": 384, "chunk_overlap": 64, "aggregation": "max_chunk_score",
                "counts": counts, "empty_documents": empty, "duplicates": duplicates,
                "decisions": decisions}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, documents in (("normalized", normalized), ("cleaned", unique), ("relevant", relevant)):
        write_json(output_dir / f"{name}.json", [doc.model_dump(mode="json") for doc in documents])
    write_json(output_dir / "audit.json", metadata)
    # Pandas creates a compact review table alongside the full common documents.
    review = pd.DataFrame([
        {"document_id": str(doc.document_id), "title": doc.title, "text": doc.text,
         "url": str(doc.url), **decision}
        for doc, decision in zip(unique, decisions)
    ], columns=["document_id", "title", "text", "url", "relevant", "score"])
    review.to_csv(output_dir / "review.csv", index=False)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/hacker_news_microsoft.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    # Validate source before downloading/loading model weights.
    normalize_hacker_news(json.loads(args.input.read_text(encoding="utf-8")))
    output = args.output_dir or PROJECT_ROOT / "data/processed" / args.input.stem
    print("Loading local relevance classifier...", flush=True)
    classifier = WorkplaceRelevanceClassifier(args.threshold)
    print(json.dumps(run(args.input, output, classifier), indent=2))
    print(f"Saved processed documents and audit to {output}")


if __name__ == "__main__":
    main()
