# Framework: private Bloom-aware academic assistance

Framework is a local academic-assistance application for university learning material. The recommended deployment path is **Next.js 15 → FastAPI → existing Python Framework modules**. The existing Streamlit application remains a research compatibility interface; it is not the production UI.

## Architecture

```text
Browser (Next.js 15) → FastAPI → FrameworkService
                              ├─ DocumentIngestor / OCR
                              ├─ FAISS + configured local retrieval encoder
                              ├─ PrivacyRetriever and PrivacyGuard
                              ├─ Qwen2.5-0.5B Bloom sequence classifier
                              ├─ uncertainty gate + central Bloom response policy
                              └─ Qwen2.5-1.5B GGUF local generator → output screening
```

The Bloom classifier only classifies/routs questions; it never generates responses. The Qwen2.5-1.5B GGUF generator produces answers and summaries locally. Runtime does not train models, run federated learning, merge adapters, or call cloud LLM APIs.

For a hosted deployment, use the Railway FastAPI/model service plus Vercel frontend configuration in [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md). The model artifacts are intentionally excluded from Git and must be placed on the Railway persistent volume.

## Local setup

Use Python 3.11 where possible.

```powershell
cd C:\Users\USER\Downloads\Framework\Framework
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env: set a prepared Bloom checkpoint and local GGUF path.
uvicorn backend.main:app --reload --port 8000
```

In another terminal:

```powershell
cd C:\Users\USER\Downloads\Framework\Framework\frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The default local access codes in `.env.example` are `student-local` and `teacher-local`; replace them and `SESSION_SECRET` before any shared deployment.

### Offline launcher

After the one-time local installation above, run `start_offline.cmd` (double-click) or:

```powershell
.\start_offline.ps1
```

It binds both local processes to `127.0.0.1`: the browser uses HTTP only to reach the Python application on the same computer. This is an internal loopback bridge required for a Next.js browser UI to use the existing Python models; it makes **no external API, model download, cloud inference, or Internet request**. A conventional “web app with no API at all” cannot call the repository’s Python modules from a browser. The launcher is therefore the supported downloadable/offline distribution form.

The launcher uses the stable Next.js production server by default. Use `./start_offline.ps1 -Development` only while modifying frontend code.

### Install from the browser (PWA)

When Framework is served through HTTPS (or from `http://127.0.0.1`), the **Install local app** button appears in supported browsers. Installing adds Framework to the desktop/start menu and caches the UI shell for offline opening. It does not cache API responses or protected learning material. For local AI features after installation, run `start_offline.cmd` so the on-device FastAPI/model service is available.

## Model configuration and readiness

Set `BLOOM_MODEL_DIR` to a **prepared** merged checkpoint containing `config.json`, or set `BLOOM_USE_QUANTIZED=true` for a valid deployed quantized checkpoint. Otherwise the profile selected by `BLOOM_MODEL_SIZE` (`0.5b` by default) is used. Set `GENERATOR_MODEL_PATH` to the local Qwen2.5-1.5B GGUF. `/models/status` reports readiness without returning machine paths or secrets. Models load lazily: the Bloom classifier on first classification and the GGUF generator on first QA/summary request.

`OFFLINE_MODE=true` prevents Hugging Face downloads during normal operation. Install the configured retrieval encoder and OCR dependencies before going offline. The API rejects missing/unready model assets rather than silently changing model or backend.

## Security model

FastAPI validates a signed local session for each request. Corpus scope is authorized server-side: students may only index/retrieve public content, while teachers may use protected exam corpora. Stores are namespaced per signed browser session; protected text is not sent in source metadata or errors. PrivacyGuard runs before student generation and output screening runs afterward. Protected teacher output is restricted to abstract moderation support rather than copied question text.

Teacher moderation includes six-level classification and probabilities, confidence/uncertainty, a classifier-aligned rationale, a local-GGUF higher-level rewrite, and an approve/needs-revision/reject review action. The rationale is generated from policy and classifier probabilities; it never exposes chain-of-thought.

Supported ingestion: PDF, PNG, JPG/JPEG, TIFF, BMP, WEBP, TXT, Markdown, and pasted text. The API validates type, size, and removes temporary upload files.

## Verification

```powershell
pytest backend/tests -q
python -m compileall backend bloom_policy.py
```

For a full local smoke test after installing model assets: start the API, call `/health`, authenticate, index a small public document, call `/bloom/classify`, `/qa`, `/summarize`; then confirm a student receives `403` when posting `scope=protected` while a teacher can index and query their protected session corpus.
