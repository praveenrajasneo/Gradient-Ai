"""Checkpoint 4: classify relevant documents, extract issues, and store in PostgreSQL."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

from app.classifiers.aspects import AspectClassifier, DESCRIPTIONS
from app.classifiers.relevance import MODEL_ID as ASPECT_MODEL, MODEL_REVISION as ASPECT_REVISION
from app.config import PROJECT_ROOT
from app.models.document import Document
from app.models.llm import LLMService
from app.preprocessing.pipeline import write_json
from app.sentiment.aspect_sentiment import AspectSentimentClassifier, aggregate, MODEL_ID, MODEL_REVISION


def classify(input_path: Path, output: Path, threshold: float):
    content = input_path.read_bytes()
    documents = [Document.model_validate(item) for item in json.loads(content)]
    report = {"status": "classifying", "input_hash": hashlib.sha256(content).hexdigest(),
              "config": {"pipeline_version": 1, "aspect_model": ASPECT_MODEL,
                         "aspect_revision": ASPECT_REVISION, "aspect_threshold": threshold,
                         "aspect_descriptions": DESCRIPTIONS,
                         "sentiment_model": MODEL_ID, "sentiment_revision": MODEL_REVISION},
              "documents": []}
    aspect_output = output.with_name(output.stem + '.aspects.json')
    cached = json.loads(aspect_output.read_text(encoding='utf-8')) if aspect_output.exists() else None
    if cached and cached['input_hash'] == report['input_hash'] and cached['config'] == report['config']:
        report = cached
        print('Reusing completed aspect stage for this input and configuration', flush=True)
    else:
        print(f"Classifying aspects in {len(documents)} relevant documents", flush=True)
        classifier = AspectClassifier(threshold)
        for index, document in enumerate(documents, 1):
            report["documents"].append({"document": document.model_dump(mode="json"),
                                         "observations": classifier.classify(document.text)})
            print(f"Aspects {index}/{len(documents)}", flush=True)
        del classifier
        gc.collect()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(aspect_output, report)
    classifier = AspectSentimentClassifier()
    for index, item in enumerate(report["documents"], 1):
        item["aspects"] = aggregate(classifier.classify(item.pop("observations")))
        print(f"Sentiment {index}/{len(documents)}", flush=True)
    del classifier
    gc.collect()
    report["status"] = "classified"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def extract(report: dict, output: Path, service):
    for index, item in enumerate(report["documents"], 1):
        if item.get("extracted"):
            continue
        evidence = [{"aspect": aspect["aspect"], "text": e["text"]}
                    for aspect in item["aspects"] for e in aspect["evidence"]
                    if e["sentiment"] == "negative"]
        points, rejections = [], []
        for start in range(0, len(evidence), 4):
            result = service.extract(evidence[start:start + 4])
            points.extend(result.pain_points)
            rejections.extend(result.rejections)
        for aspect in item["aspects"]:
            unique = {(p.pain_point, p.evidence_quote): p.model_dump() for p in points if p.aspect == aspect["aspect"]}
            aspect["pain_points"] = list(unique.values())
            aspect["pain_point"] = "; ".join(dict.fromkeys(p["pain_point"] for p in unique.values())) or None
        item["extracted"] = True
        item['extraction_rejections'] = rejections
        write_json(output, report)
        print(f"Pain points {index}/{len(report['documents'])}: {len(points)} accepted, {len(rejections)} rejected", flush=True)
    report["status"] = "extracted"
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/processed/hacker_news_microsoft/relevant.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["all", "classify", "extract", "store"], default="all")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    output = args.output or args.input.parent / "intelligence.json"
    if args.stage in {"all", "classify"}:
        report = classify(args.input, output, args.threshold)
    else:
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["input_hash"] != hashlib.sha256(args.input.read_bytes()).hexdigest():
            raise ValueError("Input changed; rerun classification before continuing")
    if args.stage in {"all", "extract"}:
        import httpx
        service = LLMService()
        tags = httpx.get(service.base_url + "/api/tags", timeout=10).json()["models"]
        digest = next(item["digest"] for item in tags if item["name"] == service.model)
        previous = report["config"].get("qwen_digest")
        if previous and previous != digest:
            raise ValueError("Qwen model changed; rerun classification to start a new run")
        report["config"].update({"qwen_model": service.model, "qwen_digest": digest,
                                  "prompt_version": 1, "temperature": 0,
                                  "validation_policy": "retry_then_single_span_then_quarantine"})
        report = extract(report, output, service)
    if args.stage in {"all", "store"}:
        from app.database.repository import store_report
        analysis_id, counts = store_report(report)
        print(json.dumps({"analysis_id": analysis_id, "negative_aspects": counts}, indent=2))
    print(f"Checkpoint 4 stage complete: {output}")


if __name__ == "__main__":
    main()
