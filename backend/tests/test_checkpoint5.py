import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from app.database.statistics import calculate
from app.embeddings.worker import chunk_document
from app.main import app
from app.rag.pipeline import GeneratedAnswer, generate, validate_citations
from app.retrieval.pipeline import fingerprint, reciprocal_rank_fusion, unique_documents


def test_statistics_distinct_documents_overlap_mixed_and_empty_aspect():
    documents = [
        {'document_id': 'a', 'source': 'hacker_news', 'published_at': None, 'aspects': [
            {'aspect': 'workload', 'sentiment': 'negative'},
            {'aspect': 'promotion', 'sentiment': 'positive'}]},
        {'document_id': 'b', 'source': 'hacker_news', 'published_at': '2024-02-01T12:00:00Z', 'aspects': [
            {'aspect': 'workload', 'sentiment': 'mixed'}]},
        {'document_id': 'c', 'source': 'hacker_news', 'published_at': None, 'aspects': []},
    ]
    result = calculate(documents)
    assert result['document_count'] == 3
    assert result['source_frequency'] == {'hacker_news': 3}
    assert result['date_distribution'] == {'2024-02': 1, 'unknown': 2}
    assert result['documents_by_sentiment'] == {'positive': 1, 'negative': 1, 'neutral': 0, 'mixed': 1}
    aspects = {item['aspect']: item for item in result['aspect_frequency']}
    assert aspects['workload']['negative_ratio_percent'] == 50.0
    assert aspects['management']['negative_ratio_percent'] is None
    assert calculate([])['document_count'] == 0


def test_chunking_keeps_offsets_quotes_overlap_and_document_tail():
    text = 'One two three four five six seven eight nine ten eleven twelve'
    import re
    def tokenizer(value, **kwargs):
        return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', value)]}
    result = chunk_document({'text': text, 'document_id': 'doc'}, tokenizer, size=5, overlap=2)
    assert all(item['text'] == text[item['start']:item['end']] for item in result)
    assert result[-1]['text'].endswith('twelve')
    assert result[1]['start'] < result[0]['end']
    assert chunk_document({'text': ''}, tokenizer) == []


def test_fusion_and_unique_evidence():
    order, scores = reciprocal_rank_fusion(['a', 'b', 'c'], ['c', 'b', 'd'])
    assert order == ['c', 'b', 'a', 'd']
    assert scores['b'] == 2 / 62
    assert reciprocal_rank_fusion(['a', 'a'])[1]['a'] == 1 / 61
    assert unique_documents([{'document_id': 'a', 'rank': 1}, {'document_id': 'a', 'rank': 2},
                             {'document_id': 'b', 'rank': 3}], 2) == [
        {'document_id': 'a', 'rank': 1}, {'document_id': 'b', 'rank': 3}]


def test_fingerprint_separates_company_runs_and_mutated_documents():
    assert fingerprint('run-a', [{'text': 'one'}]) != fingerprint('run-b', [{'text': 'one'}])
    assert fingerprint('run-a', [{'text': 'one'}]) != fingerprint('run-a', [{'text': 'two'}])


EVIDENCE = [{'evidence_id': 'E1', 'source_type': 'community', 'text': 'Promotion was slow in my team.'}]
VALID = {'insufficient_evidence': False, 'observations': [
    {'statement': 'A commenter reported slow promotion in their team.', 'evidence_kind': 'public_opinion',
     'citations': [{'evidence_id': 'E1', 'quote': 'Promotion was slow'}]}]}


@pytest.mark.parametrize('change', ['id', 'quote', 'official', 'empty'])
def test_bad_citations_fail_closed(change):
    data = deepcopy(VALID)
    if change == 'id':
        data['observations'][0]['citations'][0]['evidence_id'] = 'E999'
    elif change == 'quote':
        data['observations'][0]['citations'][0]['quote'] = 'An invented quote'
    elif change == 'official':
        data['observations'][0]['evidence_kind'] = 'official_evidence'
    else:
        data['observations'] = []
    with pytest.raises(ValueError):
        validate_citations(GeneratedAnswer.model_validate(data), EVIDENCE)


def test_generation_retries_then_returns_only_checked_response():
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        assert 'untrusted' in body['messages'][0]['content']
        data = deepcopy(VALID)
        if len(calls) == 1:
            data['observations'][0]['citations'][0]['evidence_id'] = 'E999'
        return httpx.Response(200, json={'message': {'content': json.dumps(data)}})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = generate('What promotion problems?', EVIDENCE, client)
    assert len(calls) == 2
    assert result.model_dump() == VALID
    assert generate('Any evidence?', []).insufficient_evidence


def test_repeated_citation_failure_is_not_returned_as_answer():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={'message': {'content': '{"insufficient_evidence": false, "observations": []}'}}))) as client:
        with pytest.raises(ValueError, match='after retry'):
            generate('Question', EVIDENCE, client)


def test_weak_retrieval_abstains_without_calling_ollama(monkeypatch):
    from app.rag import pipeline
    monkeypatch.setenv('RAG_MIN_RERANKER_SCORE', '-5')
    monkeypatch.setattr(pipeline, 'retrieve', lambda *args, **kwargs: {
        'analysis_id': 'run', 'candidate_count': 50, 'embedding_model': 'BGE-M3', 'reranker_model': 'BGE',
        'evidence': [{'reranker_score': -8.2, 'text': 'Unrelated salary information'}]})
    # generate([], ...) returns before any HTTP request; an HTTP call would fail this test.
    monkeypatch.setattr(httpx.Client, 'post', lambda *args, **kwargs: pytest.fail('Unexpected Ollama request'))
    result = pipeline.answer('Microsoft', 'What is the cafeteria menu?')
    assert result['insufficient_evidence'] is True
    assert result['observations'] == [] and result['sources'] == []
    assert result['retrieval']['evidence_passages'] == 0


def test_reranker_gate_keeps_supported_boundary_and_rejects_nonfinite():
    from app.rag.pipeline import relevant_evidence
    strong, weak, invalid = {'reranker_score': -4.65}, {'reranker_score': -8.2}, {'reranker_score': float('nan')}
    assert relevant_evidence([strong, weak, invalid], -5) == [strong]
    with pytest.raises(ValueError):
        relevant_evidence([strong], float('nan'))


def test_answerability_check_cannot_invent_evidence_ids():
    from app.rag.pipeline import assess
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        'message': {'content': json.dumps({'reason': 'test', 'evidence_ids': ['E999']})}}))) as client:
        with pytest.raises(ValueError, match='unknown reference'):
            assess('Question', EVIDENCE, client)


def test_unrelated_high_scoring_evidence_is_not_summarized(monkeypatch):
    from app.rag import pipeline
    monkeypatch.setattr(pipeline, 'retrieve', lambda *args, **kwargs: {
        'analysis_id': 'run', 'candidate_count': 50, 'embedding_model': 'BGE-M3', 'reranker_model': 'BGE',
        'evidence': [{'reranker_score': 1.5, 'text': 'Microsoft layoffs coming soon'}]})
    monkeypatch.setattr(pipeline, 'assess', lambda *args: pipeline.EvidenceAssessment(
        reason='A layoff headline does not describe a menu.', evidence_ids=[]))
    monkeypatch.setattr(pipeline, 'unload_ollama', lambda: None)
    monkeypatch.setattr(httpx.Client, 'post', lambda *args, **kwargs: pytest.fail('Unexpected answer generation'))
    result = pipeline.answer('Microsoft', 'What is the cafeteria menu?')
    assert result['insufficient_evidence'] and result['observations'] == []


def test_api_validation_no_index_and_unknown_company(monkeypatch):
    from app.api import evidence
    def no_index(*args):
        raise RuntimeError('Build the index first')
    def unknown(*args):
        raise LookupError('No completed analysis')
    monkeypatch.setattr(evidence, 'retrieve', no_index)
    monkeypatch.setattr(evidence, 'statistics', unknown)
    with TestClient(app) as client:
        assert client.post('/api/ask', json={'company': 'M', 'question': '   '}).status_code == 422
        assert client.post('/api/retrieve', json={'company': 'M', 'question': 'promotion', 'top_k': 21}).status_code == 422
        assert client.post('/api/retrieve', json={'company': 'M', 'question': 'promotion'}).status_code == 409
        assert client.get('/api/companies/Unknown/statistics').status_code == 404
