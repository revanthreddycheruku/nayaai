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

## Legal Frameworks Covered

The knowledge base indexes verified statutory provisions across key consumer sectors:
- **Aviation:** DGCA Civil Aviation Requirements (CAR) Section 3, Series M, Part II (Refunds), Part IV (Flight Delays, Cancellations, Denied Boarding), and Part VI & Carriage by Air Act, 1972 (Baggage Rules and Statutory Compensation up to ₹20,000 / 1,288 SDRs).
- **E-Commerce:** Consumer Protection (E-Commerce) Rules, 2020 (Rule 6 return refusal prohibition, Rule 4 marketplace duties) and Consumer Protection Act, 2019 (Sections 2(11) Deficiency in Service, 2(47) Unfair Trade Practice, Chapter VI Product Liability).
- **Insurance:** IRDAI (Protection of Policyholders' Interests) Regulations, 2024 (Mandatory 30-day claim settlement TAT, penal interest at Bank Rate + 2%, 30-day Free-Look Period) and Insurance Ombudsman Rules, 2017.

## Key Features

- **Exact Statutory Section Pinpointing:** Automatically identifies the grievance sector and sub-issue, citing only the precise governing statutory sections rather than generic sector rules.
- **Structured 9-Section Legal Briefing:**
  1. Governing Statutory Section
  2. Legal Test & Statutory Criteria
  3. Supporting Consumer Protection Act Provisions
  4. 3-Tier Escalation Path (Nodal Officer $\rightarrow$ Sector Regulator $\rightarrow$ e-Jagriti)
  5. Actionable Steps for Consumer
  6. Limitations & Timelines (Section 69 CPA 2-year limitation period)
  7. Evidentiary Checklist
  8. Executive Legal Summary
  9. Strategic Suggestions & Proactive Steps Beyond Immediate Escalation
- **Built-in Text-to-Speech (TTS):** Natural client-side vocalization via the Web Speech API with Play, Pause, Resume, and Stop controls, complete with automatic markdown cleaning and currency phonetic formatting.
- **Zero-Config Offline Legal Engine:** Built-in statutory synthesis engine generates comprehensive legal responses out-of-the-box without requiring paid third-party API keys (`LLM_PROVIDER=offline`).
- **Low-Memory Cloud Architecture:** Optimized for free hosting tiers (e.g., Render free tier 512MB RAM) using high-speed BM25 ranking and statutory indexing with < 65 MB RAM footprint.

## Free Cloud Deployment

### Deploy on Render (Zero Docker, 100% Free)
1. Fork or push this repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com/), click **New + $\rightarrow$ Web Service** and connect this repository.
3. Configure:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r backend/requirements.txt && cd frontend && npm install && npm run build && cd ..`
   - **Start Command:** `uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT`
   - **Plan:** `Free ($0/month)`
4. Click **Deploy Web Service**. Render serves the unified React frontend and FastAPI backend from a single URL.

## Disclaimer

This application provides AI-generated consumer legal information based on official Indian statutory frameworks. It does not constitute formal legal representation and is not a substitute for advice from a qualified advocate or consumer law practitioner.
