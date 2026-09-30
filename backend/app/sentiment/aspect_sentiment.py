"""Sentiment conditioned on the aspect and its evidence, never whole-post sentiment."""

import re

from app.config import PROJECT_ROOT

MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"
MODEL_REVISION = "10c9dff335a44073e1352360c3a7bc54dc58eb01"
TARGETS = {
    "workload": ["deadlines", "deadline", "workload", "pressure"],
    "work_life_balance": ["working hours", "work-life balance", "weekends", "overtime"],
    "management": ["managers", "manager", "management", "leadership"],
    "career_growth": ["mentorship", "learning", "career growth", "training"],
    "promotion": ["promotion", "promotions", "advancement"],
    "compensation": ["salary", "pay", "compensation", "benefits"],
    "job_security": ["layoffs", "job security", "jobs", "layoff"],
    "team_culture": ["team", "colleagues", "coworkers", "culture"],
}


class AspectSentimentClassifier:
    def __init__(self):
        import torch
        from transformers import pipeline
        torch.set_num_threads(min(4, torch.get_num_threads()))
        local = PROJECT_ROOT / ".local/models/sentiment" / MODEL_REVISION
        model = str(local) if (local / 'model.safetensors').exists() else MODEL_ID
        self.pipeline = pipeline("text-classification", model=model, revision=MODEL_REVISION,
                                 device=-1, model_kwargs={"use_safetensors": True})

    def classify(self, observations: list[dict]) -> list[dict]:
        if not observations:
            return []
        pairs = []
        for observation in observations:
            text, aspect = observation["text"], observation["aspect"]
            target = next((term for term in TARGETS[aspect]
                           if re.search(r"\b" + re.escape(term) + r"\b", text, re.I)),
                          aspect.replace("_", " "))
            pairs.append({"text": text, "text_pair": target})
        predictions = self.pipeline(pairs, truncation=True, max_length=512, batch_size=8)
        results = []
        for observation, prediction in zip(observations, predictions):
            label = prediction["label"].lower()
            if label not in {"positive", "negative", "neutral"}:
                raise ValueError(f"Unexpected sentiment label: {label}")
            results.append({**observation, "sentiment": label, "sentiment_score": prediction["score"]})
        return results


def aggregate(observations: list[dict]) -> list[dict]:
    grouped = {}
    for item in observations:
        grouped.setdefault(item["aspect"], []).append(item)
    results = []
    for aspect, evidence in grouped.items():
        labels = {item["sentiment"] for item in evidence}
        sentiment = "mixed" if {"positive", "negative"} <= labels else (
            "negative" if "negative" in labels else "positive" if "positive" in labels else "neutral")
        results.append({"aspect": aspect, "sentiment": sentiment,
                        "confidence": min(max(e["aspect_score"] for e in evidence),
                                          min(e["sentiment_score"] for e in evidence)),
                        "evidence": evidence, "pain_points": [], "pain_point": None})
    return results
