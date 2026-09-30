# Checkpoint 4 verification

Completed on 2026-09-30 using the 63 Microsoft candidate documents retained at
Checkpoint 3. These are model predictions on public discussions, not verified
company facts or a representative employee survey.

## Stored results

- Database: `gradient_nova` on `127.0.0.1:5433`.
- Analysis: `9bc590f5-21aa-572c-ba23-6f12fe490d79`.
- Documents: 63.
- Document/aspect rows: 183 (62 negative, 38 positive, 77 neutral, 6 mixed).
- Accepted pain-point records: 88. These are evidence-linked records, not 88 unique themes.
- Rejected generated candidates: 1, excluded from accepted pain points and saved
  in the local JSON audit and `analyses.config.extraction_rejections`.

## Actual SQL result

| Aspect | Negative document/aspect rows |
| --- | ---: |
| management | 17 |
| workload | 12 |
| job_security | 10 |
| team_culture | 7 |
| compensation | 6 |
| promotion | 4 |
| work_life_balance | 4 |
| career_growth | 2 |

Counts come from PostgreSQL, scoped to the analysis above. Mixed rows are excluded
from this negative-only query and retain both sentiment directions in evidence.

## Validation

- 27 regression/database tests passed on the final code; the optional real-model
  test is skipped by default and passed separately during implementation.
- The live model smoke test identified team culture, promotion, and workload from
  the user's example, with positive, negative, and negative sentiment respectively.
- Live FastAPI to Ollama extraction returned structured promotion pain-point JSON.
- PostgreSQL integration tests verified repeat storage does not inflate counts and
  invalid evidence causes a transaction rollback.
- All five requested tables exist. Final SQL checks confirmed stored row counts,
  accepted pain-point totals, and the rejected-candidate audit.

The aspect classifier uses independent MiniLM NLI scores with a provisional 0.65
cutoff. The dedicated DeBERTa ABSA model receives each evidence span and aspect
target. Qwen3 4B performs only structured pain extraction. Pinned model revisions
and Qwen's digest are recorded in the saved report and analysis configuration.

One candidate incorrectly interpreted a statement about not being paid when not
working as unpaid work and changed the quote. Exact-quote validation excluded it.
Other predictions still need manual accuracy evaluation: quote membership alone
does not establish semantic entailment or company/role attribution.

## Run again

From `backend`:

```powershell
.\venv\Scripts\python.exe -m app.database.setup --local
.\venv\Scripts\python.exe -m app.database.query --analysis-id 9bc590f5-21aa-572c-ba23-6f12fe490d79
```

Full pipeline and stage/resume commands are in README. Local results are in
`data/processed/hacker_news_microsoft/intelligence.json` and are ignored by Git.
The existing PostgreSQL installation on port 5432 was not modified or reset;
the isolated development cluster's generated credentials remain only in `.env`.

No retrieval, RAG, or final-report generation has been added.
