"""Opt-in real model smoke test; not a held-out accuracy benchmark."""

import gc
import os

import pytest


@pytest.mark.skipif(os.getenv('RUN_MODEL_TESTS') != '1', reason='Set RUN_MODEL_TESTS=1 after model downloads')
def test_real_multilabel_aspect_sentiment_and_qwen():
    from app.classifiers.aspects import AspectClassifier
    from app.sentiment.aspect_sentiment import AspectSentimentClassifier, aggregate
    from app.models.llm import LLMService
    text = 'The team is excellent, but promotion is slow and deadlines are unreasonable.'
    classifier = AspectClassifier()
    observations = classifier.classify(text)
    assert {item['aspect'] for item in observations} == {'team_culture', 'promotion', 'workload'}
    del classifier
    gc.collect()
    classifier = AspectSentimentClassifier()
    rows = aggregate(classifier.classify(observations))
    assert {row['aspect']: row['sentiment'] for row in rows} == {
        'team_culture': 'positive', 'promotion': 'negative', 'workload': 'negative',
    }
    del classifier
    gc.collect()
    evidence = 'The promotion process is not transparent and nobody knows what is required.'
    result = LLMService().extract([{'aspect': 'promotion', 'text': evidence}])
    assert result.pain_points
    assert all(point.aspect == 'promotion' and point.evidence_quote in evidence for point in result.pain_points)
