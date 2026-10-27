---
title: Consumer Rights AI
emoji: ⚖️
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# Consumer Rights AI

An evidence-grounded Indian consumer law assistant with a FastAPI backend, ChromaDB, BGE embeddings, hybrid semantic and keyword retrieval, a configurable LLM, and a React/TypeScript interface.

## Current project state

- The workspace contained 32 CSVs (832 rows), not a `datasets.zip`. The originals in `datasets/` are preserved.
- The audit is in [`data/reports/dataset_audit.md`](data/reports/dataset_audit.md). It found 112 duplicate URL groups, 124 normalized-text duplicate groups, and 112 URLs associated with multiple provision files. Generic titles and topic mismatches require document-level review.
- Cleaning produced 539 unique normalized text records with source-file and row associations in `data/processed/candidate_documents.jsonl`. Supplied verification labels are retained as claims from the CSV; no candidate record is independently verified.
- One supplied case URL was spot-checked and appears unrelated to the assigned Consumer Protection Act section. See [`data/reports/manual_source_review.md`](data/reports/manual_source_review.md).
- One official primary source is currently approved for scoped use: the IRDAI (Protection of Policyholders' Interests, operations and allied matters of insurers) Regulations, 2024. Its locally downloaded Gazette PDF is indexed in Chroma. The IRDAI index listed it as non-archived on 24 October 2026. Its use still requires checking later issue-specific amendments and circulars. The exact checksum and review note are in [`data/processed/verification_ledger.json`](data/processed/verification_ledger.json).
- The DGCA passenger-refund CAR PDF is a scanned image and is not indexed because OCR is not installed. Three Consumer Affairs PDFs could not be downloaded from this environment. See [`data/official_sources.json`](data/official_sources.json) and `data/raw/official_sources/download_report.json` (local, ignored file).

The app will not produce a legal answer unless it retrieves an eligible independently reviewed source and an LLM provider is configured. The currently indexed IRDAI regulation grounds insurance-related information only within its scope; it does not support general consumer-law conclusions.

## Requirements

- Python 3.11 or newer and PowerShell
- Node.js 20 or newer and npm
- Internet access for package installation and the first BGE model download
- An OpenAI-compatible API key/model or a locally running Ollama model for generated answers

## Setup (PowerShell)

From the repository root:

```powershell
node scripts/audit_dataset.mjs datasets data/reports
node scripts/clean_dataset.mjs datasets data/processed
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

Edit `.env` to configure the provider. For OpenAI, set `LLM_PROVIDER=openai`, `OPENAI_MODEL`, and `OPENAI_API_KEY`. For Ollama, set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL`. No key is needed for auditing or cleaning. With `LLM_PROVIDER=none`, `/api/chat` returns a clear setup error rather than simulating an answer.

### Build the knowledge base

Candidate CSV records are indexed for review but excluded from answer context by default. Run the official-source downloader and importer to update the reviewed sources:

```powershell
python scripts\download_official_sources.py
python scripts\ingest_official_sources.py
```

The downloader accepts only configured government domains, saves checksums, and never marks documents verified. A source can enter the answer context only when its PDF checksum matches an explicit entry in `data/processed/verification_ledger.json`. Do not set `ALLOW_UNVERIFIED_CONTEXT=true` for normal use.

Indexing downloads `BAAI/bge-base-en-v1.5` from Hugging Face if it is not cached. For local-only use after download, set `HF_HUB_OFFLINE=1` in the shell before starting Python. Scanned PDFs need an OCR engine as well as the optional Python OCR packages; OCR output requires human review before approval.

Run the API in one terminal:

```powershell
uvicorn app.main:app --app-dir backend --reload --env-file .env --host 127.0.0.1 --port 8000
```

Run the web app in another:

```powershell
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` if the API is not at `http://localhost:8000`.

Run backend tests:

```powershell
python -m pytest backend\tests -q
```

## API

- `GET /health`
- `GET /api/topics`
- `POST /api/chat` — `{ "message": "...", "conversation_id": "optional", "history": [] }`
- `POST /api/complaint/analyze` — classify and extract explicit complaint facts
- `POST /api/complaint/draft` — create a reviewable draft using supplied facts
- `POST /api/retrieval/search` — search eligible local documents
- `GET /api/sources/{document_id}` — retrieve indexed source metadata

The API does not persist conversation history. Requests are size-bounded, CORS origins are configurable, and complaint text is not logged. Retrieved text and user-provided content are treated as untrusted input. The frontend displays source status and links citations.

## Data quality and limits

- CSV content and metadata may be duplicated, mislabeled, incomplete, or irrelevant to the assigned provision. A URL or supplied verification label is not proof.
- Audit overlap detection does not determine legal relevance. Most candidate records remain for manual source-by-source review.
- Only the IRDAI 2024 regulation has been reviewed and indexed so far. Other legal topics may return a knowledge-base limitation.
- The IRDAI source is a 2024 Gazette text. The official index listed it as non-archived when checked; check current circulars and amendments for the user's specific issue.
- The DGCA refund CAR is currently unsearchable locally because its downloaded PDF contains scanned pages and OCR is unavailable. Three other official PDFs timed out during download.
- Classification is a lightweight keyword heuristic, not a legal determination. The chat endpoint accepts conversation history from the client but does not retain server-side state.
- Retrieval uses BGE semantic search plus keyword scoring and reciprocal-rank fusion. A cross-encoder reranker is not enabled. No retrieval or answer-quality metrics are claimed because a reviewed evaluation set has not been created.
- Complaint letters are editable drafts from user-provided facts; the app does not submit them.
- The app is not a substitute for advice from a qualified lawyer.

## Vector store maintenance

The local Chroma database is stored in `vector_store/` and is ignored by Git. The rebuild helper deletes the configured vector store inside the project before re-indexing; back it up first if needed:

```powershell
python scripts\rebuild_vector_store.py
```

Never commit `.env`, raw or processed complaint data, datasets, or local vector databases.
