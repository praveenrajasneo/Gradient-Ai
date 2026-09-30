import json
import os
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.classifiers.aspects import AspectClassifier, DESCRIPTIONS, evidence_clauses
from app.main import app
from app.models.intelligence import Extraction
from app.models.llm import LLMService
from app.sentiment.aspect_sentiment import AspectSentimentClassifier, aggregate


def test_multilabel_evidence_is_verbatim():
    text = "The team is excellent, but promotion is slow and deadlines are unreasonable."
    clauses = evidence_clauses(text)
    assert len(clauses) == 3 and all(clause in text for clause in clauses)
    def predict(inputs, **kwargs):
        assert kwargs['multi_label'] is True
        assert set(kwargs['candidate_labels']) == set(DESCRIPTIONS.values())
        return [{'labels': [DESCRIPTIONS['promotion'], DESCRIPTIONS['workload']],
                 'scores': [0.9, 0.8]} for _ in inputs]
    classifier = AspectClassifier.__new__(AspectClassifier)
    classifier.pipeline, classifier.threshold = predict, 0.65
    results = classifier.classify(text)
    assert {result['aspect'] for result in results} == {'promotion', 'workload'}
    assert all(result['text'] in text for result in results)


def test_sentiment_conditions_on_each_aspect_and_preserves_mixed():
    observations = [{'aspect': 'team_culture', 'text': 'The team is excellent', 'aspect_score': 0.9},
                    {'aspect': 'promotion', 'text': 'promotion is slow', 'aspect_score': 0.8}]
    def predict(inputs, **kwargs):
        assert inputs == [{'text': 'The team is excellent', 'text_pair': 'team'},
                          {'text': 'promotion is slow', 'text_pair': 'promotion'}]
        return [{'label': 'Positive', 'score': 0.95}, {'label': 'Negative', 'score': 0.96}]
    classifier = AspectSentimentClassifier.__new__(AspectSentimentClassifier)
    classifier.pipeline = predict
    results = classifier.classify(observations)
    assert [r['sentiment'] for r in results] == ['positive', 'negative']
    results.append({**results[0], 'text': 'The team is toxic', 'sentiment': 'negative'})
    rows = aggregate(results)
    assert rows[0]['sentiment'] == 'mixed'
    assert len(rows[0]['evidence']) == 2


def response_transport(content):
    def respond(request):
        body = json.loads(request.content)
        assert body['think'] is False and body['format']['type'] == 'object'
        return httpx.Response(200, json={'message': {'content': json.dumps(content)}})
    return httpx.MockTransport(respond)


def test_qwen_validated_json_and_quote():
    result = {'pain_points': [{'aspect': 'promotion', 'sentiment': 'negative',
                              'pain_point': 'slow promotion cycle', 'evidence_quote': 'promotion is slow'}]}
    with httpx.Client(transport=response_transport(result)) as client:
        service = LLMService(client)
        assert service.extract([{'aspect': 'promotion', 'text': 'promotion is slow'}]).pain_points[0].pain_point == 'slow promotion cycle'
        rejected = service.extract([{'aspect': 'promotion', 'text': 'promotion is fast'}])
        assert rejected.pain_points == []
        assert rejected.rejections[0]['reason'] == 'quote_not_in_matching_evidence'
    assert LLMService().extract([]).pain_points == []


def test_qwen_repairs_invalid_quote_once():
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        point = {'aspect': 'promotion', 'sentiment': 'negative',
                 'pain_point': 'slow promotion cycle',
                 'evidence_quote': 'invented' if len(calls) == 1 else 'promotion is slow'}
        return httpx.Response(200, json={'message': {'content': json.dumps({'pain_points': [point]})}})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = LLMService(client).extract([{'aspect': 'promotion', 'text': 'promotion is slow'}])
    assert result.pain_points[0].evidence_quote == 'promotion is slow'
    assert len(calls) == 2
    assert len(calls[1]['messages']) == 4


def test_failed_batch_retries_individual_spans(monkeypatch):
    calls = []
    def batch(self, evidence):
        calls.append(len(evidence))
        if len(evidence) > 1:
            raise ValueError('mismatched quote')
        return Extraction(pain_points=[])
    monkeypatch.setattr(LLMService, '_extract_batch', batch)
    assert LLMService().extract([
        {'aspect': 'promotion', 'text': 'slow promotion'},
        {'aspect': 'workload', 'text': 'unreasonable deadlines'},
    ]).pain_points == []
    assert calls == [2, 1, 1]


@pytest.mark.parametrize('point', [
    {'aspect': 'invented', 'sentiment': 'negative', 'pain_point': 'issue', 'evidence_quote': 'text'},
    {'aspect': 'promotion', 'sentiment': 'positive', 'pain_point': 'issue', 'evidence_quote': 'text'},
])
def test_qwen_rejects_unknown_aspects_or_sentiments(point):
    with httpx.Client(transport=response_transport({'pain_points': [point]})) as client:
        with pytest.raises(ValueError):
            LLMService(client).extract([{'aspect': 'promotion', 'text': 'text'}])


def test_api_reports_ollama_failure(monkeypatch):
    def fail(*args):
        raise httpx.ConnectError('offline')
    monkeypatch.setattr(LLMService, 'extract', fail)
    with TestClient(app) as client:
        result = client.post('/api/pain-points/extract', json={'text': 'promotion is slow', 'aspect': 'promotion'})
        assert result.status_code == 503
        assert client.get('/').status_code == 200


def test_extraction_resume_only_skips_completed_documents(tmp_path):
    from app.pain_points.pipeline import extract
    calls = []
    class Service:
        def extract(self, evidence):
            calls.append(evidence)
            return Extraction(pain_points=[])
    report = {'status': 'classified', 'documents': [
        {'aspects': [{'aspect': 'promotion', 'evidence': [
            {'text': 'promotion is slow', 'sentiment': 'negative'}]}]},
        {'aspects': []},
    ]}
    output = tmp_path / 'report.json'
    extract(report, output, Service())
    extract(report, output, Service())
    assert len(calls) == 1
    assert json.loads(output.read_text())['status'] == 'extracted'


@pytest.mark.skipif(os.getenv('RUN_DB_TESTS') != '1', reason='Set RUN_DB_TESTS=1 for isolated PostgreSQL integration test')
def test_postgres_idempotence_and_rollback(monkeypatch):
    from app.database import repository
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from app.config import database_url
    from pathlib import Path
    schema = 'checkpoint4_test_' + uuid4().hex
    connection_string = database_url()
    with psycopg.connect(connection_string, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        try:
            url = make_conninfo(connection_string, options=f'-c search_path={schema}')
            with psycopg.connect(url) as connection:
                connection.execute(Path(repository.__file__).with_name('schema.sql').read_text())
            monkeypatch.setattr(repository, 'database_url', lambda: url)
            document = {'document_id': str(uuid4()), 'company': 'Test company', 'source': 'hacker_news',
                        'source_type': 'community', 'title': '', 'text': 'promotion is slow',
                        'url': 'https://news.ycombinator.com/item?id=1', 'author': None,
                        'published_at': None, 'collected_at': '2026-09-28T00:00:00Z'}
            report = {'status': 'extracted', 'input_hash': 'test', 'config': {}, 'documents': [
                {'document': document, 'aspects': [{'aspect': 'promotion', 'sentiment': 'negative',
                 'confidence': 0.9, 'pain_point': None, 'pain_points': [],
                 'evidence': [{'text': 'promotion is slow', 'sentiment': 'negative'}]}]}]}
            first = repository.store_report(report)
            assert first == repository.store_report(report)
            assert first[1] == [{'aspect': 'promotion', 'negative_documents': 1}]
            report['documents'][0]['aspects'][0]['evidence'][0]['text'] = 'invented quote'
            with pytest.raises(ValueError):
                repository.store_report(report)
            with psycopg.connect(url) as connection:
                assert connection.execute('SELECT COUNT(*) FROM aspects').fetchone()[0] == 1
                assert connection.execute('SELECT status FROM analyses').fetchone()[0] == 'completed'
        finally:
            admin.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
