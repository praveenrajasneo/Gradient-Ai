"""Run memory-heavy models sequentially outside the API process."""
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import httpx

from app.config import PROJECT_ROOT

MODEL_LOCK = threading.RLock()


class ModelExecutionError(RuntimeError):
    """A local inference process failed, independently of index readiness."""


def unload_ollama():
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434') + '/api/generate',
                                   json={'model': os.getenv('OLLAMA_MODEL', 'qwen3:4b'), 'keep_alive': 0})
            response.raise_for_status()
    except httpx.ConnectError:
        pass  # Retrieval also works with Ollama stopped.


def model_job(request):
    with MODEL_LOCK:
        unload_ollama()
        scratch = PROJECT_ROOT / '.local/jobs'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            source, target = Path(directory) / 'input.json', Path(directory) / 'output.json'
            source.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            result = subprocess.run([sys.executable, '-m', 'app.embeddings.worker', str(source), str(target)],
                                    cwd=PROJECT_ROOT / 'backend', timeout=7200, check=False)
            if result.returncode:
                raise ModelExecutionError(f'BGE model worker failed (exit {result.returncode}); check server output and model downloads')
            return json.loads(target.read_text(encoding='utf-8'))
