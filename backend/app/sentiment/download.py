"""Download the pinned sentiment checkpoint with progress and SHA-256 verification."""

import hashlib

import httpx

from app.config import PROJECT_ROOT
from app.sentiment.aspect_sentiment import MODEL_ID, MODEL_REVISION


def main():
    destination = PROJECT_ROOT / ".local/models/sentiment" / MODEL_REVISION
    destination.mkdir(parents=True, exist_ok=True)
    files = ['config.json', 'added_tokens.json', 'special_tokens_map.json',
             'spm.model', 'tokenizer_config.json', 'model.safetensors']
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true")
        response.raise_for_status()
        metadata = {item['rfilename']: item for item in response.json()['siblings']}
        for name in files:
            target = destination / name
            expected = metadata[name].get('lfs', {}).get('sha256')
            if target.exists() and (not expected or hashlib.sha256(target.read_bytes()).hexdigest() == expected):
                continue
            temporary = target.with_suffix(target.suffix + '.part')
            digest = hashlib.sha256()
            downloaded = 0
            with client.stream('GET', f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}?download=true") as download:
                download.raise_for_status()
                with temporary.open('wb') as output:
                    for chunk in download.iter_bytes(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if downloaded % (50 * 1024 * 1024) == 0:
                            print(f"{name}: {downloaded // (1024 * 1024)} MiB", flush=True)
            if expected and digest.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch for {name}")
            temporary.replace(target)
            print(f"Verified {name}: {downloaded} bytes", flush=True)
    print(f"Sentiment checkpoint ready: {destination}")


if __name__ == '__main__':
    main()
