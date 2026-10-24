# Evaluation status

Evaluation fixtures are in `backend/tests/evaluation_cases.json` and are intended for human review. Each expected source list is empty pending independent verification of the supplied records and acquisition of authoritative legislation/guidance.

| Metric | Result |
| --- | --- |
| Recall@K | Not measured: no verified expected-source labels |
| Precision@K | Not measured: no verified expected-source labels |
| Mean Reciprocal Rank | Not measured: no verified expected-source labels |
| Citation accuracy | Not measured: no verified answer/source adjudications |
| Unsupported legal claim rate | Not measured: no generated legal answers against eligible sources |
| Uncertain question handling | Covered by a source-empty API test; automated test suite unrun because Python is unavailable |

No benchmark or production-readiness claim is made.
