from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import streamlit as st

from predict_bloom import BLOOM_LABELS, BLOOM_LEVELS, QwenBloomPredictor, is_deploy_checkpoint
from bloom_model_profiles import DEFAULT_MODEL_SIZE, get_profile
from bloom_prompt import moderate_bloom_question, BloomModerationResult

from runtime_utils import (
    DEFAULT_N_CTX,
    DEFAULT_N_THREADS,
    _apply_retrieval_governor,
    exact_match,
    measure_model_file_mb,
    measure_rss_mb,
    measure_uss_mb,
    meteor_lite,
    rouge_l,
    token_f1,
)
from ingestion import DocumentChunk, DocumentIngestor, ocr_backend_status
from models import RAGGenerator
from privacy.privacy_guard import (
    STUDENT_REFUSAL,
    assess_student_query_against_protected_corpus,
    policy_instruction,
    screen_generation_output,
)
from role_access import (
    Role,
    check_retriever_binding,
    check_task,
    check_upload_target,
    normalize_role,
    resolve_search_scope,
    student_visible_chunks,
    teacher_visible_chunks,
)
from architecture_compliance import check_all as architecture_compliance_report
from retriever import PrivacyRetriever, RetrievalResult
from summarizer import CognitiveSummarizer
from uncertainty import UncertaintyEngine


os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

APP_TITLE = "Lightweight Multi-Modal Tiny LLM Demo"
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "txt", "md"]
DEFAULT_FAISS_POOL = 20
VECTOR_STORE_DIR = Path("data/vector_store")
BLOOM_DEPLOY_MODE_FP32 = "fp32_merged"
BLOOM_DEPLOY_MODE_FP16 = "fp16_deploy"


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)


def _bloom_deploy_settings() -> Dict[str, Any]:
    model_size = os.environ.get("BLOOM_MODEL_SIZE", DEFAULT_MODEL_SIZE)
    use_lightweight = os.environ.get("BLOOM_USE_QUANTIZED", "0").strip().lower() in ("1", "true", "yes")
    profile = get_profile(model_size)
    if use_lightweight:
        deploy_path = Path(profile.quantized_dir)
        mode = BLOOM_DEPLOY_MODE_FP16
    else:
        deploy_path = Path(profile.merged_dir)
        mode = BLOOM_DEPLOY_MODE_FP32
    return {
        "model_size": model_size,
        "profile": profile,
        "use_lightweight": use_lightweight,
        "deploy_path": deploy_path,
        "mode": mode,
    }


def _bloom_checkpoint_ready(settings: Dict[str, Any]) -> bool:
    path: Path = settings["deploy_path"]
    if settings["mode"] == BLOOM_DEPLOY_MODE_FP16:
        return is_deploy_checkpoint(path)
    return (path / "config.json").is_file()


def _predict_bloom(runtime: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Bloom level + distribution from trained Qwen LoRA (predict_bloom.py)."""
    predictor: QwenBloomPredictor | None = runtime.get("bloom_predictor")
    if predictor is None:
        raise RuntimeError(
            runtime.get("bloom_predictor_error")
            or "Bloom LoRA predictor is not loaded. Train with train_qwen_bloom.py."
        )
    out = predictor.predict(text)
    profile_name = getattr(getattr(predictor, "profile", None), "display_name", "Qwen2.5 Bloom")
    return {
        "level": out["prediction"],
        "rag_key": out["rag_key"],
        "distribution": out["distribution"],
        "confidence": float(out["confidence"]),
        "probabilities": out["probabilities"],
        "source": f"{profile_name} LoRA (train_qwen_bloom.py)",
    }


# ============================================================
# EVERYTHING BELOW THIS IS UNCHANGED (YOUR ORIGINAL APP)
# ============================================================

def _show_teacher_bloom_moderation(query: str, bloom_pred: Dict[str, Any]) -> BloomModerationResult | None:
    """Teacher architecture: LoRA Bloom level + GGUF reason + higher-order rewrite."""
    st.subheader("Teacher Bloom Moderation")
    cols = st.columns(3)
    cols[0].metric("Bloom Level (LoRA)", bloom_pred["level"])
    cols[1].metric("Confidence", f"{bloom_pred['confidence']:.3f}")
    cols[2].metric("Label source", "0.5B classifier")
    st.caption(bloom_pred.get("source", ""))

    with st.expander("Class probabilities (LoRA)", expanded=False):
        st.dataframe(
            [
                {"level": label, "probability": bloom_pred["probabilities"][label]}
                for label in BLOOM_LABELS
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.spinner("Generating pedagogical reason and higher-level rewrite (local 1.5B GGUF)..."):
        moderation = moderate_bloom_question(
            query,
            lora_level=bloom_pred["level"],
            lora_confidence=bloom_pred["confidence"],
        )

    if moderation.error:
        st.error(
            "Could not generate reason/rewrite. Ensure "
            "`models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` and llama-cli are available. "
            f"Detail: {moderation.error}"
        )
        return moderation

    info_cols = st.columns(2)
    info_cols[0].metric("Confirmed / adjusted level", moderation.bloom_level)
    info_cols[1].metric("Target for rewrite", moderation.target_higher_level)

    st.markdown("**Reason**")
    st.write(moderation.reason)

    st.markdown("**Higher-level rewrite**")
    st.info(moderation.higher_level_rewrite)

    if moderation.latency_s:
        st.caption(f"Moderation backend: {moderation.backend} · {moderation.latency_s:.1f}s")
    return moderation


def _init_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(APP_TITLE)
    st.caption(
        "Role-separated local stack: students use public RAG + PrivacyGuard; teachers use "
        "Qwen2.5-0.5B LoRA Bloom moderation (six levels + rewrite) and optional protected corpora. "
        "Student/teacher answers use the local Qwen2.5-1.5B GGUF generator."
    )


@st.cache_resource(show_spinner="Loading Qwen Bloom classifier...")
def _load_bloom_predictor(model_size: str, use_lightweight: bool) -> QwenBloomPredictor:
    profile = get_profile(model_size)
    if use_lightweight:
        deploy_path = Path(profile.quantized_dir)
        if not is_deploy_checkpoint(deploy_path):
            raise FileNotFoundError(
                f"Lightweight Bloom model not found at {deploy_path}. "
                f"Run: python quantize_bloom.py --model-size {model_size} --force"
            )
        return QwenBloomPredictor(
            model_dir=str(deploy_path),
            model_size=model_size,
            prefer_quantized=True,
            quantized=True,
        )

    merged_path = Path(profile.merged_dir)
    if not (merged_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Merged Bloom classifier not found at {merged_path}. "
            f"Run: python merge_model.py --model-size {model_size}"
        )
    return QwenBloomPredictor(
        model_dir=str(merged_path),
        model_size=model_size,
        prefer_merged=True,
        prefer_quantized=False,
        quantized=False,
    )


def _restore_vector_stores(runtime: Dict[str, Any]) -> Dict[str, int]:
    """Reload persisted FAISS corpora saved under data/vector_store/."""
    from ingestion import DocumentChunk

    restored: Dict[str, int] = {}
    for scope, retr_key, state_key in (
        ("public", "retriever", "public_chunks"),
        ("protected", "protected_retriever", "protected_chunks"),
    ):
        store_dir = VECTOR_STORE_DIR / scope
        if not (store_dir / "index.faiss").is_file():
            continue
        try:
            runtime[retr_key].load_vector_store(store_dir)
            chunks = [
                doc for doc in (runtime[retr_key]._docs or []) if isinstance(doc, DocumentChunk)
            ]
            if chunks:
                st.session_state[state_key] = chunks
                restored[scope] = len(chunks)
        except Exception as exc:
            st.session_state[f"{scope}_vector_store_error"] = str(exc)
    if restored:
        st.session_state["public_corpus_ready"] = bool(st.session_state.get("public_chunks"))
        st.session_state["protected_corpus_ready"] = bool(st.session_state.get("protected_chunks"))
        st.session_state["corpus_ready"] = bool(
            st.session_state.get("public_corpus_ready") or st.session_state.get("protected_corpus_ready")
        )
        all_chunks = list(st.session_state.get("public_chunks", [])) + list(
            st.session_state.get("protected_chunks", [])
        )
        st.session_state.active_chunks = all_chunks
        st.session_state.active_sources = sorted({c.source for c in all_chunks})
    return restored


def _runtime() -> Dict[str, Any]:
    if "demo_runtime" not in st.session_state:
        deploy = _bloom_deploy_settings()
        ingestor = DocumentIngestor(chunk_size=220, chunk_overlap=32)
        retriever = PrivacyRetriever(lambda_privacy=0.1)
        protected_retriever = PrivacyRetriever(lambda_privacy=0.5, model=retriever.model)

        bloom_predictor = None
        bloom_predictor_error = ""
        if _bloom_checkpoint_ready(deploy):
            try:
                bloom_predictor = _load_bloom_predictor(
                    deploy["model_size"],
                    deploy["use_lightweight"],
                )
            except Exception as exc:
                bloom_predictor_error = str(exc)
        else:
            bloom_predictor_error = (
                f"Bloom deploy checkpoint missing at {deploy['deploy_path']}. "
                f"Run: python merge_model.py --model-size {deploy['model_size']}"
            )

        generator = None
        generator_error = ""
        summarizer = None

        try:
            generator = RAGGenerator(
                retriever=retriever,
                n_ctx=DEFAULT_N_CTX,
                n_threads=DEFAULT_N_THREADS,
                max_tokens=200,
            )
            summarizer = CognitiveSummarizer(
                retriever=retriever,
                generator=generator,
                bloom_predictor=bloom_predictor,
                hierarchical=False,
                per_chunk_max_tokens=64,
            )
        except Exception as exc:
            generator_error = str(exc)

        uncertainty = UncertaintyEngine(K=len(BLOOM_LEVELS), n_bins=10)

        runtime_payload = {
            "ingestor": ingestor,
            "retriever": retriever,
            "protected_retriever": protected_retriever,
            "bloom_predictor": bloom_predictor,
            "bloom_predictor_error": bloom_predictor_error,
            "bloom_deploy": deploy,
            "generator": generator,
            "generator_error": generator_error,
            "summarizer": summarizer,
            "uncertainty": uncertainty,
        }
        st.session_state.demo_runtime = runtime_payload
        _restore_vector_stores(runtime_payload)

    return st.session_state.demo_runtime



def _ingest_uploaded_files(
    ingestor: DocumentIngestor,
    uploads: Sequence[Any],
    pasted_text: str,
) -> tuple[List[DocumentChunk], List[str]]:
    chunks: List[DocumentChunk] = []
    warnings: List[str] = []
    temp_paths: List[Path] = []
    try:
        for upload in uploads or []:
            suffix = Path(upload.name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getbuffer())
                tmp_path = Path(tmp.name)
            temp_paths.append(tmp_path)
            try:
                prev_len = len(chunks)
                chunks.extend(ingestor.process(tmp_path))
                for c in chunks[prev_len:]:
                    if c.source == str(tmp_path):
                        c.source = upload.name
            except Exception as exc:
                warnings.append(f"Skipped `{upload.name}`: {exc}")
        if pasted_text.strip():
            chunks.extend(
                ingestor.chunk_text(
                    pasted_text,
                    source="<pasted_text>",
                    modality="text",
                )
            )
        return chunks, warnings
    finally:
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _set_active_corpus(
    runtime: Dict[str, Any],
    chunks: Sequence[DocumentChunk],
    lambda_privacy: float,
    upload_scope: str,
    content_type: str,
) -> None:
    texts = [c.text for c in chunks if c.text.strip()]
    if not texts:
        raise ValueError("No usable text chunks were extracted from the uploaded inputs.")
    for chunk in chunks:
        chunk.access_level = "protected" if upload_scope == "protected" else "public"
        chunk.content_type = str(content_type)

    state_key = "protected_chunks" if upload_scope == "protected" else "public_chunks"
    retr_key = "protected_retriever" if upload_scope == "protected" else "retriever"
    existing: List[DocumentChunk] = list(st.session_state.get(state_key, []))
    existing.extend(chunks)
    retriever: PrivacyRetriever = runtime[retr_key]
    retriever.lambda_privacy = float(lambda_privacy)
    retriever.build_index([c for c in existing if c.text.strip()])
    retriever.save_vector_store(VECTOR_STORE_DIR / upload_scope)

    st.session_state[state_key] = existing
    st.session_state["public_corpus_ready"] = bool(st.session_state.get("public_chunks"))
    st.session_state["protected_corpus_ready"] = bool(st.session_state.get("protected_chunks"))
    st.session_state["corpus_ready"] = bool(
        st.session_state.get("public_corpus_ready") or st.session_state.get("protected_corpus_ready")
    )
    all_chunks = list(st.session_state.get("public_chunks", [])) + list(st.session_state.get("protected_chunks", []))
    st.session_state.active_chunks = all_chunks
    st.session_state.active_sources = sorted({c.source for c in all_chunks})


def _preview_chunk_table(chunks: Sequence[DocumentChunk]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in chunks[:18]:
        rows.append(
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "modality": c.modality,
                "page": c.page,
                "access": c.access_level,
                "content_type": c.content_type,
                "chars": len(c.text),
                "preview": c.text[:180] + ("..." if len(c.text) > 180 else ""),
            }
        )
    return rows


def _governed_chunks(
    retriever: PrivacyRetriever,
    query: str,
    top_k: int,
    governor_preset: str,
    *,
    student_mode: bool = False,
) -> List[RetrievalResult]:
    pool_n = max(DEFAULT_FAISS_POOL, int(top_k))
    pool = retriever.retrieve(
        query,
        top_k=pool_n,
        candidate_pool=pool_n,
        rank_by="relevance" if student_mode else "privacy",
    )
    chunks, _ = _apply_retrieval_governor(
        pool,
        governor_preset,
        query,
        final_k=top_k,
        retr=retriever,
    )
    return chunks


def _run_qa(
    runtime: Dict[str, Any],
    query: str,
    bloom_level: str,
    top_k: int,
    governor_preset: str,
    retriever: PrivacyRetriever,
    safety_instruction: str,
    max_tokens: int,
    *,
    student_mode: bool = False,
) -> Dict[str, Any]:
    generator: RAGGenerator = runtime["generator"]
    if generator is None:
        raise RuntimeError(f"Generation backend is unavailable: {runtime.get('generator_error') or 'unknown error'}")
    t0 = time.perf_counter()
    chunks = _governed_chunks(
        retriever,
        query,
        top_k=top_k,
        governor_preset=governor_preset,
        student_mode=student_mode,
    )
    output = generator.generate_from_chunks(
        query,
        chunks,
        bloom_level=bloom_level,
        max_tokens=max_tokens,
        safety_instruction=safety_instruction,
    )
    elapsed = time.perf_counter() - t0
    return {
        "text": output.answer,
        "chunks": output.chunks,
        "prompt": output.prompt,
        "latency_s": elapsed,
        "metadata": output.metadata,
    }


def _run_summary(
    runtime: Dict[str, Any],
    query: str,
    top_k: int,
    max_tokens: int,
    governor_preset: str,
    retriever: PrivacyRetriever,
    safety_instruction: str,
    *,
    student_mode: bool = False,
) -> Dict[str, Any]:
    summarizer: CognitiveSummarizer = runtime["summarizer"]
    if summarizer is None:
        raise RuntimeError(f"Summarization backend is unavailable: {runtime.get('generator_error') or 'unknown error'}")
    t0 = time.perf_counter()
    chunks = _governed_chunks(
        retriever,
        query,
        top_k=top_k,
        governor_preset=governor_preset,
        student_mode=student_mode,
    )
    output = summarizer.summarize(
        query=query,
        k=top_k,
        max_tokens=max_tokens,
        retrieved_chunks=chunks,
        safety_instruction=safety_instruction,
    )
    elapsed = time.perf_counter() - t0
    return {
        "text": output.summary,
        "chunks": output.chunks,
        "prompt": output.prompt,
        "latency_s": elapsed,
        "metadata": output.metadata,
    }


def _reference_metrics(prediction: str, reference: str) -> Dict[str, float]:
    return {
        "exact_match": float(exact_match(prediction, reference)),
        "token_f1": float(token_f1(prediction, reference)),
        "rouge_l": float(rouge_l(prediction, reference)),
        "meteor_lite": float(meteor_lite(prediction, reference)),
    }


def _show_retrieval_trace(chunks: Sequence[RetrievalResult], protected_mode: bool) -> None:
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "rank": chunk.rank,
                "doc_id": chunk.doc_id,
                "privacy_score": round(float(chunk.privacy_score), 4),
                "cosine": round(float(chunk.cosine), 4),
                "infonce_risk": round(float(chunk.infonce_risk), 4),
                "preview": (
                    "[protected snippet hidden]"
                    if protected_mode
                    else chunk.text[:220] + ("..." if len(chunk.text) > 220 else "")
                ),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    _init_page()

    try:
        runtime = _runtime()
    except Exception as exc:
        st.error(f"Failed to initialize the local demo stack: {exc}")
        st.stop()
    if runtime.get("bloom_predictor_error"):
        st.warning(f"Bloom classifier unavailable: {runtime['bloom_predictor_error']}")
    elif runtime.get("bloom_predictor") is not None:
        deploy = runtime.get("bloom_deploy", _bloom_deploy_settings())
        mode_label = "FP32 merged (recommended)" if deploy["mode"] == BLOOM_DEPLOY_MODE_FP32 else "FP16 deploy (smaller, slower on CPU)"
        st.caption(
            f"Bloom deploy: **{deploy['profile'].display_name}** · {mode_label} · "
            f"`{deploy['deploy_path']}` · {_dir_size_mb(deploy['deploy_path']):.0f} MB"
        )
    if runtime.get("generator_error"):
        st.error(
            "Qwen GGUF generator unavailable — student Q&A will not work until you place "
            "`models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` on disk. "
            f"Detail: {runtime['generator_error']}"
        )

    with st.sidebar:
        st.header("Demo Controls")
        lambda_privacy = st.slider("Privacy lambda (retrieval)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=8, value=4, step=1)
        max_tokens = st.slider("Max generation tokens", min_value=48, max_value=256, value=160, step=16)
        requester_role = st.radio("Requester role", options=["Student", "Teacher / Moderator"], index=0)
        role = normalize_role(requester_role)
        upload_options = (
            ["Public Learning Corpus", "Protected Exam Corpus"]
            if role == Role.TEACHER
            else ["Public Learning Corpus"]
        )
        upload_scope = st.radio("Upload target", options=upload_options, index=0)
        content_type = st.selectbox(
            "Uploaded content type",
            options=["study_material", "lecture_notes", "exam_paper", "moderation_material"],
            index=0,
        )
        task_options = (
            ["Exam Question Classification", "Question Answering", "Summarization"]
            if role == Role.TEACHER
            else ["Question Answering", "Summarization"]
        )
        mode = st.radio("Task", options=task_options, index=0)
        protected_mode = st.toggle("Protected exam mode", value=False)
        if role == Role.STUDENT:
            governor_preset = "qa"
        else:
            governor_preset = "strong" if protected_mode else "qa"
        search_scope = resolve_search_scope(role, upload_scope)
        st.caption(
            "Students: public corpus + Output Privacy Guard only. Teachers: 0.5B LoRA Bloom classifier "
            "(6 cognitive levels) + 1.5B GGUF reason/rewrite; protected index when upload target is protected."
        )
        deploy = runtime.get("bloom_deploy", _bloom_deploy_settings())
        with st.expander("Deployment status", expanded=False):
            st.markdown("**Bloom classifier (teacher labels)**")
            st.write(f"Model: {deploy['profile'].display_name}")
            st.write(f"Checkpoint: `{deploy['deploy_path']}`")
            st.write(f"Mode: {deploy['mode']}")
            st.write(f"On disk: {_dir_size_mb(deploy['deploy_path']):.1f} MB")
            st.write(f"Loaded: {'yes' if runtime.get('bloom_predictor') else 'no'}")
            if deploy["use_lightweight"]:
                st.caption("FP16 is smaller but slower on CPU. Unset BLOOM_USE_QUANTIZED for FP32 merged.")
            st.markdown("**Student / teacher text (RAG + moderation)**")
            st.write("Generator: `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`")
            st.write(f"Loaded: {'yes' if runtime.get('generator') else 'no'}")
            public_n = len(st.session_state.get("public_chunks", []))
            protected_n = len(st.session_state.get("protected_chunks", []))
            st.markdown("**Vector stores**")
            st.write(f"Public chunks restored: {public_n}")
            st.write(f"Protected chunks restored: {protected_n}")
            ocr = ocr_backend_status()
            st.markdown("**Image OCR**")
            if ocr.get("available"):
                st.write(f"Available: yes ({ocr.get('engine')})")
            else:
                st.write("Available: no — PDF/TXT/paste still work")
                st.caption(str(ocr.get("install_hint") or ocr.get("reason")))
        with st.expander("Architecture compliance (live)", expanded=False):
            report = architecture_compliance_report()
            st.write(f"Checks passed: {report['passed']}/{report['total']}")
            for row in report["checks"]:
                icon = "✅" if row["ok"] else "⚠️"
                st.markdown(f"{icon} **{row['pillar']}** — {row['check']}: {row['detail']}")
            for note in report.get("notes", []):
                st.caption(note)
        st.caption(
            "PDF and pasted text work without Tesseract. Images need Tesseract on PATH "
            "(or easyocr). Install: winget install UB-Mannheim.TesseractOCR"
        )

    uploads = st.file_uploader(
        "Upload academic sources",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, image, TXT, and Markdown files.",
    )
    pasted_text = st.text_area(
        "Or paste text directly",
        height=180,
        placeholder="Paste notes, lecture text, textbook excerpts, or any academic content here.",
    )

    col_build, col_status = st.columns([1, 2])
    with col_build:
        build_clicked = st.button("Build / Refresh Corpus", use_container_width=True)
    with col_status:
        if st.session_state.get("corpus_ready"):
            public_n = len(st.session_state.get("public_chunks", []))
            protected_n = len(st.session_state.get("protected_chunks", []))
            st.success(f"Corpora ready. Public chunks: {public_n}, Protected chunks: {protected_n}.")
        else:
            st.info("Build the corpus after uploading or pasting source material.")

    if build_clicked:
        try:
            upload_decision = check_upload_target(role, upload_scope)
            if not upload_decision.allowed:
                st.error(upload_decision.reason)
                st.stop()
            if not uploads and not pasted_text.strip():
                st.error("Upload at least one PDF/TXT file or paste text before building the corpus.")
                st.stop()
            chunks, ingest_warnings = _ingest_uploaded_files(
                runtime["ingestor"],
                uploads or [],
                pasted_text,
            )
            if not chunks:
                st.error(
                    "No chunks were indexed. "
                    + (ingest_warnings[0] if ingest_warnings else "Upload PDF/TXT or paste text.")
                )
                if len(ingest_warnings) > 1:
                    with st.expander("Ingestion details"):
                        for note in ingest_warnings:
                            st.write(note)
                st.stop()
            _set_active_corpus(
                runtime,
                chunks,
                lambda_privacy=lambda_privacy,
                upload_scope="protected" if upload_scope == "Protected Exam Corpus" else "public",
                content_type=content_type,
            )
            st.success(f"Indexed {len(chunks)} chunks into the {upload_scope.lower()}.")
            for note in ingest_warnings:
                st.warning(note)
        except Exception as exc:
            st.error(f"Corpus build failed: {exc}")

    if st.session_state.get("corpus_ready") and mode != "Exam Question Classification":
        public_chunks = list(st.session_state.get("public_chunks", []))
        protected_chunks = list(st.session_state.get("protected_chunks", []))
        if role == Role.STUDENT:
            visible_chunks = student_visible_chunks(public_chunks, protected_chunks)
        else:
            visible_chunks = teacher_visible_chunks(public_chunks, protected_chunks, search_scope)
        chunks: List[DocumentChunk] = visible_chunks
        modalities = Counter(c.modality for c in chunks)
        stat_cols = st.columns(4)
        stat_cols[0].metric("Visible Sources", len({c.source for c in chunks}))
        stat_cols[1].metric("Chunks", len(chunks))
        stat_cols[2].metric("Modalities", ", ".join(f"{k}:{v}" for k, v in sorted(modalities.items())))
        stat_cols[3].metric("Private RAM (USS)", f"{measure_uss_mb():.1f} MB")

        with st.expander("Corpus Preview", expanded=False):
            if protected_mode:
                st.info("Protected exam mode hides raw chunk previews.")
            else:
                st.dataframe(_preview_chunk_table(chunks), use_container_width=True, hide_index=True)

    if mode == "Question Answering":
        query_label = "Ask a question"
        query_placeholder = "Example: Explain the main idea and give one application."
    elif mode == "Summarization":
        query_label = "Request a summary"
        query_placeholder = "Example: Summarize this topic for an undergraduate learner."
    else:
        query_label = "Enter an exam question"
        query_placeholder = "Example: Compare TCP and UDP for reliability and latency trade-offs."

    query = st.text_area(query_label, height=120, placeholder=query_placeholder)
    reference = st.text_area(
        "Optional reference answer / summary for scoring",
        height=120,
        placeholder="Paste a gold answer or summary to compute EM/F1/ROUGE-L/METEOR-lite.",
    )

    if st.button("Run Inference", type="primary", use_container_width=True):
        task_decision = check_task(role, mode)
        if not task_decision.allowed:
            st.error(task_decision.reason)
            st.stop()
        public_chunks = list(st.session_state.get("public_chunks", []))
        protected_chunks = list(st.session_state.get("protected_chunks", []))
        role_key = "teacher" if role == Role.TEACHER else "student"
        if role == Role.STUDENT:
            visible_chunks = student_visible_chunks(public_chunks, protected_chunks)
        else:
            visible_chunks = teacher_visible_chunks(public_chunks, protected_chunks, search_scope)
        use_protected_retriever = role == Role.TEACHER and search_scope == "protected"
        bind_decision = check_retriever_binding(role, search_scope, use_protected_retriever)
        if not bind_decision.allowed:
            st.error(bind_decision.reason)
            st.stop()
        if mode != "Exam Question Classification" and not visible_chunks:
            st.error("Build the appropriate corpus first.")
            st.stop()
        if not query.strip():
            st.error("Enter a question or summary request first.")
            st.stop()

        runtime["retriever"].lambda_privacy = float(lambda_privacy)
        runtime["protected_retriever"].lambda_privacy = float(lambda_privacy)
        uncertainty: UncertaintyEngine = runtime["uncertainty"]

        try:
            query_policy = assess_student_query_against_protected_corpus(query, protected_chunks)
            if role_key == "student" and protected_chunks and not query_policy.allowed:
                st.error(STUDENT_REFUSAL)
                st.stop()

            bloom_pred = _predict_bloom(runtime, query)
            bloom_summary = uncertainty.aggregate_summary(bloom_p=bloom_pred["distribution"])
            bloom_gate = uncertainty.gate(bloom_pred["distribution"], threshold=0.35)
            bloom_uncertainty = bloom_summary.bloom_uncertainty
            bloom_level = (
                bloom_pred["rag_key"] if bloom_gate["accepted"] else "understand"
            )
            gate_instruction = ""
            if not bloom_gate["accepted"]:
                gate_instruction = (
                    "Bloom LoRA confidence is low; use a generalized academic "
                    "response and avoid over-specializing to a single Bloom level."
                )

            if mode == "Exam Question Classification":
                _show_teacher_bloom_moderation(query, bloom_pred)
                if not bloom_gate["accepted"]:
                    st.warning(
                        "Low-confidence LoRA prediction; review moderation output before approval."
                    )
                st.stop()

            teacher_moderation = None
            if role == Role.TEACHER and mode != "Exam Question Classification":
                teacher_moderation = _show_teacher_bloom_moderation(query, bloom_pred)
                if teacher_moderation and not bloom_gate["accepted"]:
                    st.warning(
                        "Low-confidence LoRA prediction; review moderation output before using the RAG answer."
                    )

            if mode == "Question Answering":
                result = _run_qa(
                    runtime,
                    query=query,
                    bloom_level=bloom_level,
                    top_k=top_k,
                    governor_preset=governor_preset,
                    retriever=runtime["protected_retriever"] if use_protected_retriever else runtime["retriever"],
                    safety_instruction="\n".join(
                        part for part in [policy_instruction(role_key, search_scope), gate_instruction] if part
                    ),
                    max_tokens=max_tokens,
                    student_mode=(role_key == "student"),
                )
            else:
                result = _run_summary(
                    runtime,
                    query=query,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    governor_preset=governor_preset,
                    retriever=runtime["protected_retriever"] if use_protected_retriever else runtime["retriever"],
                    safety_instruction="\n".join(
                        part for part in [policy_instruction(role_key, search_scope), gate_instruction] if part
                    ),
                    student_mode=(role_key == "student"),
                )

            output_policy = screen_generation_output(role_key, query, result["text"], protected_chunks)
            if not output_policy.allowed:
                result["text"] = STUDENT_REFUSAL
                result["metadata"]["privacy_block_reason"] = output_policy.reason
                result["metadata"]["privacy_risk_score"] = round(float(output_policy.risk_score), 4)
                result["chunks"] = []
                result["prompt"] = "[protected prompt hidden after policy block]"

            rss_mb = measure_rss_mb()
            uss_mb = measure_uss_mb()
            model_mb = measure_model_file_mb(runtime["generator"])
            metrics = _reference_metrics(result["text"], reference) if reference.strip() else None

            st.subheader("Model Output")
            st.write(result["text"])

            metric_cols = st.columns(5)
            metric_cols[0].metric("Bloom Level", bloom_pred["level"])
            metric_cols[1].metric("Top-1 Confidence", f"{bloom_summary.top1_confidence:.3f}")
            metric_cols[2].metric("Bloom Uncertainty", f"{bloom_uncertainty:.3f}")
            metric_cols[3].metric("Latency", f"{result['latency_s']:.2f} s")
            metric_cols[4].metric("Retrieved Chunks", len(result["chunks"]))
            st.caption(f"Bloom label source: {bloom_pred['source']}.")
            if not bloom_gate["accepted"]:
                st.warning(
                    "Bloom gate used generalized fallback "
                    f"'understand' because entropy confidence "
                    f"{bloom_gate['confidence']:.3f} is below threshold."
                )

            infra_cols = st.columns(4)
            infra_cols[0].metric("RSS", f"{rss_mb:.1f} MB")
            infra_cols[1].metric("USS", f"{uss_mb:.1f} MB")
            infra_cols[2].metric("Model mmap", f"{model_mb:.1f} MB")
            infra_cols[3].metric("Privacy lambda", f"{lambda_privacy:.2f}")

            if role_key == "student" and protected_chunks:
                st.caption("Protected-corpus policy is active for student-facing requests.")

            if metrics:
                st.subheader("Reference-Based Quality Metrics")
                ref_cols = st.columns(4)
                ref_cols[0].metric("EM", f"{metrics['exact_match']:.3f}")
                ref_cols[1].metric("Token F1", f"{metrics['token_f1']:.3f}")
                ref_cols[2].metric("ROUGE-L", f"{metrics['rouge_l']:.3f}")
                ref_cols[3].metric("METEOR-lite", f"{metrics['meteor_lite']:.3f}")

            with st.expander("Bloom probabilities (LoRA)", expanded=False):
                rows = [
                    {"level": label, "probability": round(float(prob), 4)}
                    for label, prob in bloom_pred["probabilities"].items()
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

            with st.expander("Retrieved Contexts", expanded=True):
                _show_retrieval_trace(result["chunks"], protected_mode=protected_mode)

            with st.expander("Prompt / Audit Trace", expanded=False):
                if protected_mode:
                    st.info("Protected exam mode hides raw prompt bodies to reduce reconstruction risk.")
                else:
                    st.code(result["prompt"], language="text")

            with st.expander("Run Metadata", expanded=False):
                st.json(result["metadata"])

        except Exception as exc:
            st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()
