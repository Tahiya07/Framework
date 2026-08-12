# Railway deployment: Framework model service

Railway hosts the persistent FastAPI inference service. Vercel hosts the Next.js frontend. This separation is required because the generator and Bloom classifier are local CPU model assets, not browser or serverless-function assets.

## 1. Create the Railway service

1. Create a Railway project and select **Deploy from GitHub repo** for this repository.
2. Set the service root directory to `Framework` if GitHub is connected at the parent directory; otherwise use the repository root containing `Dockerfile`.
3. Railway detects `Dockerfile` and builds the backend container.
4. Add a **Volume** mounted at `/data`.
5. Generate a public domain for the service after the first successful deploy.

## 2. Required Railway variables

Set these in the Railway service Variables tab. Do not use the values from the local `.env` in production.

```text
OFFLINE_MODE=true
DATA_DIR=/data/web_sessions
GENERATOR_MODEL_PATH=/data/models/qwen.gguf
BLOOM_MODEL_DIR=/data/models/qwen_bloom_merged0.5B
BLOOM_MODEL_SIZE=0.5b
BLOOM_USE_QUANTIZED=false
RETRIEVAL_ENCODER=bge-small
MAX_UPLOAD_MB=20
BLOOM_GATE_THRESHOLD=0.40
SESSION_SECRET=<generate-a-long-random-secret>
STUDENT_ACCESS_CODE=<set-a-strong-student-code>
TEACHER_ACCESS_CODE=<set-a-strong-teacher-code>
CORS_ORIGINS=https://<your-vercel-domain>
RAILWAY_HEALTHCHECK_TIMEOUT_SEC=300
```

For a smaller Bloom deployment, upload the prepared quantized model and set:

```text
BLOOM_MODEL_DIR=/data/models/qwen_bloom_quantized0.5B
BLOOM_USE_QUANTIZED=true
```

## 3. Populate the Railway volume with models

The repository intentionally ignores model artifacts. After attaching the volume, upload these local assets into its model directory:

```text
/data/models/qwen.gguf
/data/models/qwen_bloom_merged0.5B/config.json
/data/models/qwen_bloom_merged0.5B/model.safetensors
/data/models/qwen_bloom_merged0.5B/tokenizer files
```

The model directory must contain the complete prepared checkpoint, not a LoRA adapter. After the first deployment is running and the Railway Volume is mounted at `/data`, use `railway service files upload` (or Railway's secure file interface) to upload the assets to `/data/models`. Do not commit model files or secrets to Git.

## 4. Deploy the frontend to Vercel

Create a Vercel project with root directory `Framework/frontend`. Set:

```text
NEXT_PUBLIC_API_URL=https://<your-railway-service-domain>
```

Redeploy Vercel after setting the variable. Then update Railway `CORS_ORIGINS` to the final Vercel domain, redeploy Railway, and test `/health`, `/models/status`, a public upload, and a teacher protected-corpus flow.

## Production notes

- Choose a Railway service size with at least 8 GB RAM for concurrent CPU inference; more memory reduces risk while loading the GGUF and Bloom model.
- The Railway volume persists documents/indexes and models across redeployments; the container filesystem does not.
- `OFFLINE_MODE=true` prevents runtime Hugging Face model downloads. The selected retrieval encoder must therefore be available locally in the image/cache before first offline use; provision it during image preparation if the target environment has no build-time network access.
- The built-in access-code mechanism is suitable only for a small trusted deployment. Use a real identity provider and per-user authorization before broad public release.
