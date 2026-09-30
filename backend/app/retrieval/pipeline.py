"""Analysis-scoped dense + BM25 retrieval, followed by BGE reranking."""
import argparse
import hashlib
import json
import math
import os
import re
import uuid
from collections import defaultdict

import httpx

from app.config import PROJECT_ROOT
from app.database.corpus import load_corpus
from app.embeddings.download import MODELS
from app.embeddings.service import MODEL_LOCK, model_job

INDEX_VERSION = 'cls-fp16-256-overlap48-v1'


def fingerprint(analysis_id, documents):
    content = json.dumps([analysis_id, documents, MODELS['embedding'], INDEX_VERSION], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def index_path(key):
    return PROJECT_ROOT / '.local/indexes' / (key + '.json')


def query_vector(question):
    key = hashlib.sha256(json.dumps([MODELS['embedding'], INDEX_VERSION, question]).encode()).hexdigest()
    path = PROJECT_ROOT / '.local/query_vectors' / (key + '.json')
    if path.exists():
        vector = json.loads(path.read_text(encoding='utf-8'))
    else:
        vector = model_job({'action': 'encode', 'texts': [question]})['vectors'][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(vector), encoding='utf-8')
        temporary.replace(path)
    if len(vector) != 1024 or not all(math.isfinite(value) for value in vector):
        raise ValueError('Invalid query vector cache')
    return vector


def qdrant(method, path, body=None):
    with httpx.Client(base_url=os.getenv('QDRANT_URL', 'http://127.0.0.1:6333'), timeout=120) as client:
        response = client.request(method, path, json=body)
        response.raise_for_status()
        return response.json()['result']


def build_index(company, analysis_id=None):
    with MODEL_LOCK:
        run, documents = load_corpus(company, analysis_id)
        key = fingerprint(run, documents)
        path = index_path(key)
        collection = 'gn_' + key[:32]
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
        else:
            generated = model_job({'action': 'index', 'documents': documents})
            points = []
            for chunk, vector in zip(generated['chunks'], generated['vectors'], strict=True):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key + chunk['document_id'] + str(chunk['start'])))
                payload = {name: chunk.get(name) for name in ('document_id', 'company', 'source', 'source_type',
                           'role', 'title', 'text', 'url', 'author', 'published_at', 'collected_at', 'start', 'end')}
                payload.update(analysis_id=run, aspect=sorted({a['aspect'] for a in chunk['aspects']}),
                               date=chunk.get('published_at'), chunk_id=point_id)
                points.append({'id': point_id, 'vector': vector, 'payload': payload})
            saved = {'analysis_id': run, 'fingerprint': key, 'collection': collection,
                     'embedding_model': MODELS['embedding'], 'points': points}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')
            temporary.replace(path)
        names = {item['name'] for item in qdrant('GET', '/collections')['collections']}
        if collection not in names:
            qdrant('PUT', '/collections/' + collection, {'vectors': {'size': 1024, 'distance': 'Cosine'}})
        for offset in range(0, len(saved['points']), 32):
            qdrant('PUT', f'/collections/{collection}/points?wait=true', {'points': saved['points'][offset:offset + 32]})
        count = qdrant('POST', f'/collections/{collection}/points/count', {'exact': True})['count']
        if count != len(saved['points']):
            raise RuntimeError('Index count mismatch')
        path.with_suffix('.ready').write_text(str(count), encoding='utf-8')
        return {'analysis_id': run, 'documents': len(documents), 'passages': count, 'collection': collection}


def tokens(text):
    return re.findall(r'\w+', text.casefold())


def reciprocal_rank_fusion(*rankings):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] += 1 / (60 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item)), dict(scores)


def unique_documents(items, limit):
    seen, selected = set(), []
    for item in items:
        if item['document_id'] not in seen:
            selected.append(item)
            seen.add(item['document_id'])
            if len(selected) >= limit:
                break
    return selected


def retrieve(company, question, analysis_id=None, mode='hybrid', top_k=20, rerank=False):
    if not question.strip() or len(question) > 500:
        raise ValueError('Question must contain 1–500 characters')
    if mode not in ('dense', 'hybrid') or not 1 <= top_k <= 20:
        raise ValueError('Invalid retrieval options')
    with MODEL_LOCK:
        run, documents = load_corpus(company, analysis_id)
        key = fingerprint(run, documents)
        path = index_path(key)
        if not path.exists() or not path.with_suffix('.ready').exists():
            raise RuntimeError('Build the index for this analysis first')
        saved = json.loads(path.read_text(encoding='utf-8'))
        collection = saved['collection']
        expected = len(saved['points'])
        count = qdrant('POST', f'/collections/{collection}/points/count', {'exact': True})['count']
        if count != expected:
            raise RuntimeError('Incomplete Qdrant index; rebuild this analysis')
        vector = query_vector(question)
        # Retrieve all passages in this small MVP corpus, then collapse to distinct documents.
        dense = qdrant('POST', f'/collections/{collection}/points/query',
                       {'query': vector, 'limit': expected, 'with_payload': False})['points']
        lookup = {p['id']: p['payload'] for p in saved['points']}
        dense_ids = [str(p['id']) for p in dense]
        dense_scores = {str(p['id']): p['score'] for p in dense}
        if mode == 'hybrid':
            from rank_bm25 import BM25Okapi
            ids = list(lookup)
            bm25 = BM25Okapi([tokens(lookup[item]['text']) for item in ids])
            scores = bm25.get_scores(tokens(question))
            lexical = sorted([i for i in range(len(ids)) if scores[i] > 0], key=lambda i: (-scores[i], ids[i]))
            ranked, fusion = reciprocal_rank_fusion(dense_ids, [ids[i] for i in lexical])
        else:
            ranked, fusion = dense_ids, {}
        candidates = unique_documents([{**lookup[item], 'dense_score': dense_scores.get(item),
                                        'fusion_score': fusion.get(item)} for item in ranked], 50 if rerank else top_k)
        if rerank and candidates:
            scores = model_job({'action': 'rerank', 'question': question,
                                'texts': [item['text'] for item in candidates]})['scores']
            for item, score in zip(candidates, scores, strict=True):
                item['reranker_score'] = score
            candidates.sort(key=lambda item: (-item['reranker_score'], item['chunk_id']))
        return {'analysis_id': run, 'company': company, 'question': question, 'mode': mode,
                'embedding_model': MODELS['embedding'],
                'reranker_model': MODELS['reranker'] if rerank else None,
                'reranked': rerank, 'candidate_count': len(candidates),
                'evidence': candidates[:min(top_k, 10) if rerank else top_k]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['index', 'search'])
    parser.add_argument('--company', default='Microsoft')
    parser.add_argument('--analysis-id')
    parser.add_argument('--question', default='What promotion problems are reported at Microsoft?')
    parser.add_argument('--mode', choices=['dense', 'hybrid'], default='hybrid')
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = build_index(args.company, args.analysis_id) if args.action == 'index' else retrieve(
        args.company, args.question, args.analysis_id, args.mode, rerank=args.rerank)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(rendered, encoding='utf-8')
    else:
        print(rendered)
