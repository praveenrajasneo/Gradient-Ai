"""Use installed PostgreSQL client libraries on Windows when available."""

import os
from pathlib import Path

_dll_directory = None
if os.name == "nt":
    _binaries = Path(os.getenv("POSTGRES_BIN", r"C:\Program Files\PostgreSQL\18\bin"))
    if (_binaries / "libpq.dll").is_file():
        _dll_directory = os.add_dll_directory(str(_binaries))
        os.environ["PATH"] = str(_binaries) + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("PSYCOPG_IMPL", "python")
