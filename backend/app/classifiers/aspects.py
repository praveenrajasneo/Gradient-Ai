"""Multi-label NLI classification over evidence clauses, using the fixed taxonomy."""

import re

from app.classifiers.relevance import MODEL_ID, MODEL_REVISION

DESCRIPTIONS = {
    "workload": "workload and deadlines",
    "work_life_balance": "work-life balance",
    "management": "management",
    "career_growth": "career development",
    "promotion": "promotion",
    "compensation": "salary and benefits",
    "job_security": "job security and layoffs",
    "team_culture": "team",
}


def evidence_clauses(text: str) -> list[str]:
    """Use exact substrings; short overlapping spans avoid truncating long posts."""
    clauses = []
    for match in re.finditer(r"[^.!?;]+(?:[.!?;]+|$)", text):
        for part in re.split(r"\s+(?:but|although|however|and|yet)\s+", match.group(), flags=re.I):
            words = list(re.finditer(r"\S+", part))
            for start in range(0, len(words), 60):
                end = min(start + 70, len(words))
                clause = part[words[start].start():words[end - 1].end()].strip()
                if len(clause) >= 4 and clause not in clauses:
                    clauses.append(clause)
                if end == len(words):
                    break
    return clauses


class AspectClassifier:
    def __init__(self, threshold=0.65):
        if not 0 <= threshold <= 1:
            raise ValueError("Aspect threshold must be between 0 and 1")
        import torch
        from transformers import pipeline
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.threshold = threshold
        self.pipeline = pipeline("zero-shot-classification", model=MODEL_ID,
                                 revision=MODEL_REVISION, device=-1,
                                 model_kwargs={"use_safetensors": True})

    def classify(self, text: str) -> list[dict]:
        clauses = evidence_clauses(text)
        if not clauses:
            return []
        predictions = self.pipeline(clauses, candidate_labels=list(DESCRIPTIONS.values()),
                                    multi_label=True, hypothesis_template="This text is about {}.",
                                    batch_size=8)
        reverse = {value: key for key, value in DESCRIPTIONS.items()}
        return [{"aspect": reverse[label], "text": clause, "aspect_score": score}
                for clause, result in zip(clauses, predictions)
                for label, score in zip(result["labels"], result["scores"])
                if score >= self.threshold]
