"""Local zero-shot workplace relevance; no sentiment or aspect predictions."""

from pydantic import BaseModel, Field

MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
LABELS = [
    "workplace conditions and employee experiences",
    "technology products and business news unrelated to employees",
]
HYPOTHESIS = "This text is about {}."


class RelevanceResult(BaseModel):
    relevant: bool
    score: float = Field(ge=0, le=1)


class WorkplaceRelevanceClassifier:
    def __init__(self, threshold: float = 0.7):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        # Lazy imports keep the echo API and preprocessing usable without ML packages.
        import torch
        from transformers import pipeline

        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.pipeline = pipeline("zero-shot-classification", model=MODEL_ID,
                                 revision=MODEL_REVISION, device=-1,
                                 model_kwargs={"use_safetensors": True})
        self.revision = getattr(self.pipeline.model.config, "_commit_hash", None)

    def classify(self, text: str) -> RelevanceResult:
        if not text.strip():
            return RelevanceResult(relevant=False, score=0)
        tokenizer = self.pipeline.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        # Overlap preserves context; every token is examined rather than truncating
        # long comments. The maximum chunk score is an uncalibrated relevance score.
        chunks = [tokenizer.decode(tokens[start:start + 384], skip_special_tokens=True)
                  for start in range(0, len(tokens), 320)]
        results = self.pipeline(chunks, candidate_labels=LABELS,
                                hypothesis_template=HYPOTHESIS, multi_label=False, batch_size=8)
        score = max(result["scores"][result["labels"].index(LABELS[0])] for result in results)
        return RelevanceResult(relevant=score >= self.threshold, score=score)
