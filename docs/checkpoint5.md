# Checkpoint 5 — statistics and normal RAG

Validated locally on 30 September 2026 using the completed Microsoft analysis
`9bc590f5-21aa-572c-ba23-6f12fe490d79`.

## Stored sample and deterministic statistics

- 63 cleaned, workplace-relevant Hacker News documents from Checkpoint 4.
- 73 overlapping passages embedded with BGE-M3 and persisted in Qdrant.
- Every vector has 1,024 dimensions and unit length within floating-point tolerance.
- 16 documents have a positive aspect and 28 have a negative aspect. These counts
  overlap; a document can discuss multiple aspects. Neutral: 39; mixed: 4.

| Aspect | Negative documents | Total matching documents | Negative ratio |
|---|---:|---:|---:|
| Workload | 12 | 23 | 52.2% |
| Work-life balance | 4 | 15 | 26.7% |
| Management | 17 | 35 | 48.6% |
| Career growth | 2 | 20 | 10.0% |
| Promotion | 4 | 22 | 18.2% |
| Compensation | 6 | 22 | 27.3% |
| Job security | 10 | 29 | 34.5% |
| Team culture | 7 | 17 | 41.2% |

These are counts of model labels in the collected sample, not measured workforce
conditions. The classifications contain false positives and need evaluation.
All sources are `hacker_news`; source frequency and publication-month distribution
are returned by the statistics endpoint. Dates in the sample span 2009–2026.

## Live search and answer check

The retrieval smoke test used **What promotion problems are reported at Microsoft?**

1. Dense search returned 20 distinct documents.
2. Dense + BM25 reciprocal rank fusion selected 50 distinct candidate documents.
3. BGE Reranker Base selected 10 evidence passages, one per document.
4. An initial Qwen3 4B response returned exact quotes and a stored source URL.

The answer used a [26 July 2018 Hacker News post](https://news.ycombinator.com/item?id=17618228)
by a self-described Microsoft engineer. The relevant statement was that their
manager could not adjust salary without a promotion. The response labeled this
as a community report and included the original passage, date, and document ID.
It did not cite the unrelated layoff passages returned by retrieval.

The separate hybrid reranking request took about 47 seconds. These are local
smoke-test timings, not performance benchmarks.

The final pipeline adds a weak-match cutoff and a separate structured Qwen
answerability check. Its supported-question API test used **What salary concerns
are reported by Microsoft employees?** It returned dated references to salary
fairness and salary sharing in [a 2021 post](https://news.ycombinator.com/item?id=29572550)
and [a 2020 post](https://news.ycombinator.com/item?id=24355801). All returned quotes
were checked against the stored sources. This request took about 71 seconds with
the question vector cached. Both sources contain only headlines, which limits
the detail the answer can support.

The final unsupported-question API test asked **How many paid parental leave
weeks does Microsoft guarantee in India in 2026?** The response explicitly
reported insufficient evidence, with no observations or cited sources. It took
about 198 seconds. The intermediate evidence selector still admitted unrelated
passages, so its judgments must not be treated as a verified relevance filter;
the final generator correctly declined this question. Both final API checks
passed, but they are smoke tests rather than a broad accuracy evaluation.

Retrieval quality is still limited: this relevant post ranked tenth after
reranking, and higher-ranked results included unrelated workplace topics. The
working pipeline does not establish that reranking improves accuracy on this
dataset; a labeled retrieval evaluation remains necessary.

## Regression and persistence checks

- 44 automated tests passed, including PostgreSQL and Qdrant integration tests.
- One optional Checkpoint 4 live-model test was skipped in this regression run.
- Reindexing reused the existing embeddings and kept 73 points for 63 documents.
- All 73 Qdrant points survived a container restart and subsequent Docker recovery.
- Statistics remained available for all 63 documents after restarting services.
- No secrets, model weights, or raw/processed source data are committed to Git.

## Models and local services

- BGE-M3: `5617a9f61b028005a4858fdac845db406aefb181`.
- BGE Reranker Base: `2cfc18c9415c912f9d8155881c133215df768a70`.
- Qwen: `qwen3:4b`, served by local Ollama.
- PostgreSQL: project cluster on localhost 5433.
- Qdrant: Docker image `qdrant/qdrant:v1.17.0`, localhost 6333, persistent volume.
- Model processes run sequentially with small batches to fit the 8 GB machine.

The implementation uses Hugging Face Transformers directly. An optional
Sentence Transformers dependency was incompatible with Windows Application
Control and was removed; security policies were not changed.

## API and local artifacts

Use [Swagger](http://localhost:8000/docs):

- `GET /api/companies/Microsoft/statistics`
- `POST /api/retrieve`
- `POST /api/ask`

The original `/api/analyze` input echo is preserved. The new endpoints select a
completed stored analysis; they do not collect fresh data during a request.

Local, Git-ignored outputs live in `data/processed/hacker_news_microsoft/`:

- `statistics.json`
- `index_report.json`
- `dense_demo.json`
- `retrieval_demo.json`
- `rag_demo.json`
- `rag_insufficient_demo.json`

Citation validation checks evidence IDs, exact quotes, and source type. It does
not prove that a model paraphrase follows from a quote, or that the original post
is true. Unknown source roles remain null; role and company attribution require
further evaluation. Historical posts are not evidence of current conditions.
