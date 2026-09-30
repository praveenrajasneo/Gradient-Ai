"""Opt-in local service checks; no model downloads or real collection mutations."""
import json
import os
import uuid

import pytest

from app.retrieval import pipeline


@pytest.mark.skipif(os.getenv('RUN_QDRANT_TESTS') != '1', reason='Set RUN_QDRANT_TESTS=1 for local Qdrant checks')
def test_qdrant_persistence_and_company_isolation():
    name = 'test_checkpoint5_' + uuid.uuid4().hex
    vector = [1.0] + [0.0] * 1023
    try:
        pipeline.qdrant('PUT', '/collections/' + name, {'vectors': {'size': 1024, 'distance': 'Cosine'}})
        point_id = str(uuid.uuid4())
        for _ in range(2):
            pipeline.qdrant('PUT', f'/collections/{name}/points?wait=true', {'points': [
                {'id': point_id, 'vector': vector, 'payload': {'company': 'Test only'}}]})
        assert pipeline.qdrant('POST', f'/collections/{name}/points/count', {'exact': True})['count'] == 1
        result = pipeline.qdrant('POST', f'/collections/{name}/points/query',
                                 {'query': vector, 'limit': 20, 'with_payload': True})['points']
        assert result[0]['id'] == point_id and result[0]['score'] == pytest.approx(1)
    finally:
        pipeline.qdrant('DELETE', '/collections/' + name)


def test_index_completion_and_retrieval_of_partial_index(monkeypatch, tmp_path):
    doc = {'document_id': str(uuid.uuid4()), 'company': 'Test', 'text': 'promotion is slow', 'aspects': [],
           'source': 'hacker_news', 'source_type': 'community'}
    monkeypatch.setattr(pipeline, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'load_corpus', lambda *args: ('run', [doc]))
    calls = []
    def worker(request):
        calls.append(request)
        return {'chunks': [{**doc, 'start': 0, 'end': len(doc['text'])}], 'vectors': [[1.0] * 1024]}
    monkeypatch.setattr(pipeline, 'model_job', worker)
    counts = {'value': 0}
    def storage(method, path, body=None):
        if path == '/collections':
            return {'collections': []}
        if path.endswith('/points?wait=true'):
            counts['value'] = len(body['points'])
        if path.endswith('/points/count'):
            return {'count': counts['value']}
        return True
    monkeypatch.setattr(pipeline, 'qdrant', storage)
    assert pipeline.build_index('Test')['passages'] == 1
    assert pipeline.build_index('Test')['passages'] == 1
    assert len(calls) == 1  # Persisted embeddings survive an index retry.
    counts['value'] = 0
    with pytest.raises(RuntimeError, match='Incomplete'):
        pipeline.retrieve('Test', 'promotion')
    assert len(calls) == 1  # Fail before loading a model for a damaged index.
