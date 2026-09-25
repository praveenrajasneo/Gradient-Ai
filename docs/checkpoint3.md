# Checkpoint 3 verification

Live run on 2026-09-25, company: Microsoft.

| Stage | Actual count |
| --- | ---: |
| Raw candidates | 300 |
| Empty documents removed | 0 |
| Duplicates removed | 6 |
| Unique cleaned documents | 294 |
| Classified workplace-relevant | 63 |
| Below threshold | 231 |

Raw collection contained 150 stories and 150 comments. The relevance threshold
was 0.70. These are classifier decisions, not manually verified labels.

Model: `cross-encoder/nli-MiniLM2-L6-H768`

Revision: `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`

The five development smoke checks below used the same relevance label descriptions
and model. Scores are rounded here; the pipeline audit stores full precision.

| Text | Workplace score | Decision |
| --- | ---: | --- |
| Microsoft announces Windows update | 0.1593 | Discard |
| Microsoft employees report mandatory office attendance | 0.9716 | Keep |
| The salary is good but managers constantly change priorities. | 0.9679 | Keep |
| Microsoft reports strong cloud revenue. | 0.0459 | Discard |
| Microsoft lays off thousands of employees. | 0.7125 | Keep |

These examples informed the label wording, so they are development smoke checks,
not a held-out accuracy evaluation. The threshold still needs validation on a
separate manually labelled dataset. Long documents use maximum chunk score,
which can increase false positives. Company attribution is not yet validated.

Validation: 17 automated tests passed; dependency compatibility checks passed;
the existing live API continued to expose the echo endpoint. Raw data and all
processed document files remain local and are excluded from Git.

See the README for collection, preprocessing, classification, and test commands.
