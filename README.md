# Gradient Nova AI

An evidence-based workplace intelligence project. Checkpoint 2 adds a validated
input echo endpoint, the fixed workplace taxonomy, and a standalone HN collector.

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
Collection runs separately from the API. No classification or AI logic is implemented.

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

## Input endpoint

`POST /api/analyze` validates and echoes this request. It does not collect data.

```json
{
  "company": "Microsoft",
  "role": "Senior Software Engineer",
  "experience": 10,
  "location": "India"
}
```

Company and role must not be blank; experience must be a nonnegative JSON integer.
Location is optional and defaults to null. Try the endpoint in Swagger UI.
The eight fixed category identifiers are in `backend/app/classifiers/workplace_aspects.py`.

## Collect Hacker News candidates

From the project directory:

```powershell
cd backend
.\venv\Scripts\python.exe app\collectors\hacker_news.py
```

Or, after activating the virtual environment, from `backend/app/collectors`:

```powershell
python hacker_news.py
```

Default: Microsoft, up to 200 unique items. To customize:

```powershell
.\venv\Scripts\python.exe app\collectors\hacker_news.py --company "Microsoft" --limit 300
```

Output: `data/raw/hacker_news_microsoft.json`, resolved relative to the project,
independent of the terminal's current directory. A successful rerun replaces that
company's snapshot. Failed collection leaves the previous snapshot unchanged.
Raw files are ignored by Git.

The collector uses the [HN Algolia search API](https://hn.algolia.com/api),
searching stories and comments separately for the company name and the suffixes
employees, layoffs, management, and promotion. It samples results across these
queries, removes repeated HN IDs, and stops at the requested count, exhausted
results, or five pages per query/type. Each page contains up to 30 hits.
The limit is a maximum, not a guarantee of that many available documents.

The JSON includes queries, collection time, actual counts, stop reason, and items.
Each item retains its HN link, item type, observed matching queries, and untouched
Algolia hit under `raw`, including HTML text, author, date, and thread metadata
where available. Comments are search matches, not complete discussion threads.
External linked articles are not fetched. Candidate matches may be irrelevant;
they have not been filtered for workplace relevance, role, or experience.

## Tests

From `backend`:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
New-Item -ItemType Directory -Path .pytest-tmp -Force | Out-Null
$env:PYTEST_DEBUG_TEMPROOT = (Resolve-Path .pytest-tmp).Path
.\venv\Scripts\python.exe -m pytest -q
```

Tests use mocked search responses; the collector command above verifies live access.
The local temporary directory avoids an existing Windows system-temp permission issue.

Checkpoint 2 stops at real raw candidate documents; cleaning, sentiment, pain-point
extraction, and RAG remain future work.
