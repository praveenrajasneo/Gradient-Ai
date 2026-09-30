"""Isolated model process: releases all model memory on exit."""
import json
import os
import sys
from pathlib import Path

from app.embeddings.download import model_path


def chunk_document(document, tokenizer, size=256, overlap=48):
    if not 0 <= overlap < size:
        raise ValueError('Chunk overlap must be smaller than chunk size')
    text = document['text']
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    chunks = []
    for start in range(0, len(offsets), size - overlap):
        selected = offsets[start:start + size]
        left, right = selected[0][0], selected[-1][1]
        excerpt = text[left:right]
        if excerpt.strip():
            chunks.append({**document, 'text': excerpt, 'start': left, 'end': right})
        if start + size >= len(offsets):
            break
    return chunks


def run(request):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    kind = 'reranker' if request['action'] == 'rerank' else 'embedding'
    path = model_path(kind)
    print(f'Loading local {kind} model', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    cls = AutoModelForSequenceClassification if kind == 'reranker' else AutoModel
    model = cls.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16,
                               use_safetensors=kind == 'reranker', weights_only=True).eval()
    print(f'{kind} model ready', flush=True)
    if kind == 'reranker':
        values = []
        for text in request['texts']:
            inputs = tokenizer(request['question'], text, return_tensors='pt',
                               truncation='only_second', max_length=512)
            with torch.inference_mode():
                score = model(**inputs).logits.float().reshape(-1)[0]
            if not torch.isfinite(score):
                raise ValueError('Non-finite reranker score')
            values.append(float(score))
        return {'scores': values}
    chunks = None
    texts = request.get('texts', [])
    if request['action'] == 'index':
        chunks = [chunk for doc in request['documents'] for chunk in chunk_document(doc, tokenizer)]
        texts = [chunk['text'] for chunk in chunks]
    vectors = []
    for index, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=384)
        with torch.inference_mode():
            vector = model(**inputs).last_hidden_state[:, 0].float()
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        if not torch.isfinite(vector).all():
            raise ValueError('Non-finite embedding')
        vectors.append(vector[0].tolist())
        if request['action'] == 'index' and (index + 1) % 10 == 0:
            print(f'Embedded {index + 1}/{len(texts)} passages', flush=True)
    return {'vectors': vectors, 'chunks': chunks}


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    result = run(request)
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
