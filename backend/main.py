from __future__ import annotations
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .auth import create_token, current_session, require_teacher
from .config import settings
from .schemas import BloomRequest, ChatRequest, IndexTextRequest, LoginRequest, ModerationReviewRequest, TextRequest
from .service import service

app = FastAPI(title="Framework Academic API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.origins(), allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
def run(fn):
    try: return fn()
    except PermissionError as e: raise HTTPException(403, str(e))
    except (ValueError, RuntimeError) as e: raise HTTPException(422, str(e))

@app.get("/health")
def health(): return {"status": "ok", "offline_mode": settings.offline_mode}
@app.get("/models/status")
def models_status(session: dict = Depends(current_session)): return service.status()
@app.post("/auth/session")
def login(body: LoginRequest):
    expected = settings.teacher_access_code if body.role == "teacher" else settings.student_access_code
    if body.access_code != expected: raise HTTPException(401, "Invalid access code.")
    return {"access_token": create_token(body.role), "role": body.role}
@app.post("/documents/index")
def index_text(body: IndexTextRequest, session: dict = Depends(current_session)): return run(lambda: service.index_text(session["sid"], session["role"], body.text, body.name, body.scope, body.content_type))
@app.post("/documents/upload")
async def upload(file: UploadFile = File(...), scope: str = Form("public"), content_type: str = Form("study_material"), session: dict = Depends(current_session)):
    payload = await file.read()
    return run(lambda: service.ingest_file(session["sid"], session["role"], payload, file.filename or "upload", scope, content_type))
@app.post("/bloom/classify")
def classify(body: BloomRequest, session: dict = Depends(current_session)): return run(lambda: service.classify(body.question))
@app.post("/teacher/exam/classify")
def classify_exam(body: BloomRequest, session: dict = Depends(require_teacher)): return run(lambda: service.classify(body.question))
@app.post("/teacher/exam/moderate")
def moderate_exam(body: BloomRequest, session: dict = Depends(require_teacher)): return run(lambda: service.moderate_exam_question(session["sid"], body.question))
@app.post("/teacher/exam/review")
def review_exam(body: ModerationReviewRequest, session: dict = Depends(require_teacher)): return service.record_moderation_review(session["sid"], body.question, body.decision, body.notes)
@app.post("/qa")
def qa(body: TextRequest, session: dict = Depends(current_session)): return run(lambda: service.answer(session["sid"], session["role"], body.question, body.scope, body.top_k))
@app.post("/chat")
def chat(body: ChatRequest, session: dict = Depends(current_session)): return run(lambda: service.student_chat(session["sid"], session["role"], body.question, body.scope, body.top_k, [turn.model_dump() for turn in body.history], summary=body.summary))
@app.post("/summarize")
def summarize(body: TextRequest, session: dict = Depends(current_session)): return run(lambda: service.answer(session["sid"], session["role"], body.question, body.scope, body.top_k, summary=True))
@app.post("/retrieval/search")
def search(body: TextRequest, session: dict = Depends(current_session)):
    return run(lambda: {"sources": service.answer(session["sid"], session["role"], body.question, body.scope, body.top_k)["sources"]})
