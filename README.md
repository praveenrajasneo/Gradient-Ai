# Gradient Nova AI

An evidence-based workplace intelligence project. Checkpoint 3 adds a common
document schema, preprocessing, duplicate detection, and local workplace relevance.

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
Collection and relevance classification run separately from the echo API.
Sentiment, aspect classification, pain-point extraction, and RAG are future work.

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

## Checkpoint 3: common documents and workplace relevance

Install the added dependencies from `backend`:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

This includes BeautifulSoup, Transformers, and PyTorch. The first classification
run downloads the open `cross-encoder/nli-MiniLM2-L6-H768` model (about 330 MB of
weights) into the Hugging Face cache. Inference runs locally on CPU; no paid API,
Ollama, or GPU is required. Later runs reuse cached weights.
The model revision is pinned in `classifiers/relevance.py` for repeatable runs.
After the initial download, set `$env:HF_HUB_OFFLINE = '1'` to run with no Hub access.

```powershell
.\venv\Scripts\python.exe app\collectors\hacker_news.py --company Microsoft --limit 300
.\venv\Scripts\python.exe -m app.preprocessing.pipeline --threshold 0.7
```

For another saved snapshot, provide `--input <path>` and optionally
`--output-dir <path>`. The default output folder derives from the input filename.
Successful reruns replace the processed outputs; raw input is never modified.

Every normalized document has exactly these fields:

```text
document_id, company, source, source_type, title, text, url,
author, published_at, collected_at
```

The Pydantic contract lives in `backend/app/models/document.py`. IDs are stable
UUIDs derived from the source item's URL. Timestamps include time zones; unknown
authors and publication dates are null rather than invented. HN link-only stories
use their headline as text; comments use their own body and retain the parent
headline only as metadata. Future sources must supply adapters to this contract.

Preprocessing removes HTML and script/style content, normalizes Unicode and
whitespace, collapses repeated exclamation/question marks, and removes empty text.
Duplicates are detected by canonical URL or SHA-256 of case-folded normalized
text, with text equality also checked. HN item IDs remain part of URLs so distinct
comments are not all collapsed into their shared thread. Duplicate provenance is
retained. This is exact deduplication, not semantic near-duplicate detection or a
guarantee that retained evidence is independent.

Outputs under `data/processed/hacker_news_microsoft/`:

- `normalized.json`: all documents in the source-independent format.
- `cleaned.json`: nonempty, cleaned, unique documents before classification.
- `relevant.json`: documents at or above the configured relevance threshold.
- `audit.json`: model revision, threshold, label wording, counts, removals, and
  every document's `{relevant, score}` decision.
- `review.csv`: Pandas-generated review table with text, links, and scores.

The NLI classifier compares workplace and non-workplace descriptions. Long text
is covered by overlapping 384-token chunks; the highest workplace score controls
retention. Scores are model outputs, **not calibrated probabilities or evidence
strength**. The default 0.7 cutoff is provisional and needs evaluation on manually
labelled data. No target retention count is forced. A workplace topic match also
does not prove that the text refers to the requested company or role.

All raw/processed outputs are ignored by Git. Below-threshold documents remain in
`cleaned.json` with decisions in `audit.json` for later review. The API still only
echoes input and does not load the model. Stop here before sentiment analysis.

Model reference: [MiniLM NLI model card](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768).
