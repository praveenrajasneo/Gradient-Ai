# Gradient Nova AI

An evidence-based workplace intelligence project. Currently at Checkpoint 1:
project setup and a minimal FastAPI backend.

Repository: https://github.com/praveenrajasneo/Gradient-Ai (branch `main`).
The local project directory is named `gradient-nova-ai`.

## Version 1 scope

- Input: company, role, experience in years.
- Source: Hacker News only.
- Aspects: Workload, Work-Life Balance, Management, Career Growth, Promotion,
  Compensation, Job Security, Team Culture.
- Output: pain points, aspect sentiment, evidence, source links, grounded summary.
- Preserve positive and contradictory evidence. Calculate counts in code and
  state when evidence is insufficient. Community opinions are not verified facts.

Bluesky, GDELT, SEBI, other sources, and advanced AI features are later work.
No collection or AI logic is implemented at this checkpoint.

## Run the backend on Windows

From the project directory in PowerShell:

```powershell
cd backend
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The explicit virtual environment executable works without activating scripts or
changing PowerShell execution policy. For an existing environment, run only the
last command. To activate it optionally, use `.\venv\Scripts\Activate.ps1`.
The system's default `python` currently selects Python 3.10; use `py -3.13`
when creating this project's environment.

- Backend: http://localhost:8000/
- Swagger UI: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json

Expected root response:

```json
{"message": "Gradient Nova AI backend running"}
```

Press Ctrl+C in the server terminal to stop it.

## Structure

`backend/app/` contains the entry point and reserved module folders. `frontend/`,
`data/`, `notebooks/`, and `docker/` are placeholders for later steps.
`.env` is local and ignored by Git; `.env.example` is safe to commit.

## Checkpoint 1 acceptance

- Root endpoint returns the expected message with HTTP 200.
- Swagger UI is available at `/docs`.
- OpenAPI schema identifies Gradient Nova AI and the GET `/` endpoint.
- Backend remains running for manual inspection.

Stop here before implementing additional endpoints or the Hacker News collector.
