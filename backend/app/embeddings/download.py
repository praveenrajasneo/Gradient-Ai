"""Download pinned BGE models, resuming partial files and verifying LFS SHA-256."""
import argparse
import hashlib
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.config import PROJECT_ROOT

MODELS = {
    'embedding': ('BAAI/bge-m3', '5617a9f61b028005a4858fdac845db406aefb181'),
    'reranker': ('BAAI/bge-reranker-base', '2cfc18c9415c912f9d8155881c133215df768a70'),
}


def model_path(kind):
    return PROJECT_ROOT / '.local/models' / kind / MODELS[kind][1]


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def download_weights(url, path, size, expected):
    """Bounded parallel ranges, assembled only after every range is complete."""
    partial = path.with_suffix(path.suffix + '.part')
    prefix = partial.stat().st_size if partial.exists() else 0
    if prefix > size:
        raise ValueError('Partial download larger than expected file')
    ranges = path.parent / (path.name + '.ranges')
    ranges.mkdir(exist_ok=True)
    step = 32 * 1024 * 1024
    spans = [(start, min(start + step, size) - 1) for start in range(prefix, size, step)]

    def fetch(span):
        start, end = span
        target = ranges / f'{start}-{end}'
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        working = target.with_suffix('.part')
        for attempt in range(5):
            try:
                with httpx.Client(timeout=120, follow_redirects=True) as client:
                    with client.stream('GET', url, headers={'Range': f'bytes={start}-{end}'}) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('content-range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Server did not return the requested byte range')
                        with working.open('wb') as output:
                            for chunk in response.iter_bytes(1024 * 1024):
                                output.write(chunk)
                if working.stat().st_size != end - start + 1:
                    raise ValueError('Incomplete byte range')
                working.replace(target)
                print(f'{path.name}: received bytes {start}-{end}', flush=True)
                return target
            except httpx.TransportError:
                if attempt == 4:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pieces = list(pool.map(fetch, spans))
    assembled = path.with_suffix(path.suffix + '.assembled')
    with assembled.open('wb') as output:
        if prefix:
            with partial.open('rb') as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
        for piece in pieces:
            with piece.open('rb') as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    if assembled.stat().st_size != size or sha256(assembled) != expected:
        raise ValueError(f'SHA-256 mismatch: {path.name}')
    assembled.replace(path)
    if partial.exists():
        partial.unlink()
    for piece in pieces:
        piece.unlink()


def download(kind):
    model, revision = MODELS[kind]
    destination = model_path(kind)
    destination.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        metadata = client.get(f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true')
        metadata.raise_for_status()
        siblings = metadata.json()['siblings']
        weights = 'model.safetensors' if any(i['rfilename'] == 'model.safetensors' for i in siblings) else 'pytorch_model.bin'
        for item in siblings:
            name = item['rfilename']
            if name not in {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                            'special_tokens_map.json', 'sentencepiece.bpe.model',
                            weights, 'modules.json', 'sentence_bert_config.json',
                            '1_Pooling/config.json'}:
                continue
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            expected = item.get('lfs', {}).get('sha256')
            if name == weights and not expected:
                raise ValueError('Missing model SHA-256')
            if path.exists() and (not expected or sha256(path) == expected):
                continue
            if name == weights:
                download_weights(f'https://huggingface.co/{model}/resolve/{revision}/{name}?download=true',
                                 path, item['lfs']['size'], expected)
                print(f'Verified {kind}/{name}', flush=True)
                continue
            partial = path.with_suffix(path.suffix + '.part')
            for attempt in range(3):
                offset = partial.stat().st_size if partial.exists() else 0
                try:
                    url = f'https://huggingface.co/{model}/resolve/{revision}/{name}?download=true'
                    with client.stream('GET', url, headers={'Range': f'bytes={offset}-'} if offset else {}) as response:
                        response.raise_for_status()
                        append = offset > 0 and response.status_code == 206
                        if not append:
                            offset = 0
                        progress = offset // (100 * 1024 * 1024)
                        with partial.open('ab' if append else 'wb') as output:
                            for chunk in response.iter_bytes(1024 * 1024):
                                output.write(chunk)
                                offset += len(chunk)
                                if offset // (100 * 1024 * 1024) > progress:
                                    progress = offset // (100 * 1024 * 1024)
                                    print(f'{kind}/{name}: {offset // (1024 * 1024)} MiB', flush=True)
                    break
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            if expected and sha256(partial) != expected:
                raise ValueError(f'SHA-256 mismatch: {name}')
            partial.replace(path)
            print(f'Verified {kind}/{name}', flush=True)
    if not (destination / weights).exists():
        raise RuntimeError('Model weights missing')
    print(f'{kind} ready: {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=MODELS)
    download(parser.parse_args().kind)
