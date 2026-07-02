from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from models import RAGGenerator
from retriever import PrivacyRetriever, RetrievalResult


DEFAULT_N_CTX = 2048
DEFAULT_N_THREADS = max(1, (os.cpu_count() or 4) // 2)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _norm_tokens(s: str) -> List[str]:
    if not s:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(str(s))]


def _norm_text(s: str) -> str:
    return " ".join(_norm_tokens(s))


def exact_match(pred: str, ref: str) -> int:
    return int(_norm_text(pred) == _norm_text(ref))


def token_f1(pred: str, ref: str) -> float:
    p, r = _norm_tokens(pred), _norm_tokens(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(r)
    return float(2 * prec * rec / (prec + rec))


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(cur[j - 1], prev[j])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str, beta: float = 1.2) -> float:
    p, r = _norm_tokens(pred), _norm_tokens(ref)
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return float(((1 + beta**2) * prec * rec) / (rec + beta**2 * prec))


def meteor_lite(pred: str, ref: str, alpha: float = 0.9) -> float:
    p, r = set(_norm_tokens(pred)), set(_norm_tokens(ref))
    if not p or not r:
        return 0.0
    inter = len(p & r)
    if inter == 0:
        return 0.0
    precision = inter / len(p)
    recall = inter / len(r)
    return float((precision * recall) / ((1 - alpha) * recall + alpha * precision + 1e-12))


GOVERNOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "off": {"max_chunk_chars": 20_000, "max_total_chars": 100_000, "diversify": False},
    "qa": {"max_chunk_chars": 700, "max_total_chars": 2800, "diversify": False},
    "mild": {"max_chunk_chars": 900, "max_total_chars": 3600, "diversify": False},
    "summary": {"max_chunk_chars": 1200, "max_total_chars": 7200, "diversify": False},
    "strong": {"max_chunk_chars": 420, "max_total_chars": 1680, "diversify": True},
}


def _truncate_chunk_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    for sep in (".\n", ".\n\n", ". ", "\n", " "):
        j = cut.rfind(sep)
        if j > max(40, max_chars // 3):
            return cut[: j + len(sep)].strip()
    return cut.strip()


def _apply_retrieval_governor(
    pool: List[RetrievalResult],
    preset: str,
    query: str,
    final_k: int,
    retr: Optional[PrivacyRetriever],
) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
    del query
    key = (preset or "off").lower()
    pr = GOVERNOR_PRESETS.get(key, GOVERNOR_PRESETS["off"])
    stats: Dict[str, Any] = {
        "preset": key,
        "pool_n": len(pool),
        "final_k": int(final_k),
        "diversify": bool(pr["diversify"]),
    }
    if not pool or final_k <= 0:
        return [], stats

    max_c = int(pr["max_chunk_chars"])
    max_tot = int(pr["max_total_chars"])
    trunc: List[RetrievalResult] = []
    for chunk in pool:
        nt = _truncate_chunk_text(chunk.text, max_c)
        trunc.append(
            RetrievalResult(
                rank=chunk.rank,
                doc_id=chunk.doc_id,
                text=nt,
                cosine=chunk.cosine,
                infonce_risk=chunk.infonce_risk,
                privacy_score=chunk.privacy_score,
                l2_distance=chunk.l2_distance,
            )
        )
    lens = [len(chunk.text) for chunk in trunc]
    stats["avg_snippet_chars"] = float(np.mean(lens)) if lens else 0.0

    if not pr["diversify"] or retr is None or len(trunc) <= 1:
        chosen_idx = list(range(min(final_k, len(trunc))))
    else:
        thresh = 0.80
        model = retr.model
        texts = [chunk.text for chunk in trunc]
        emb = model.encode(texts, normalize_embeddings=True)
        top = emb[0:1]
        used: set[int] = {0}
        order = [0]
        cos_to_top: List[Tuple[float, int]] = []
        for j in range(1, len(trunc)):
            cos_j = float((top @ emb[j : j + 1].T).squeeze())
            cos_to_top.append((cos_j, j))
        cos_to_top.sort(key=lambda item: (item[0], item[1]))
        for cos_j, j in cos_to_top:
            del cos_j
            if len(order) >= final_k:
                break
            if j not in used:
                used.add(j)
                order.append(j)
        chosen_idx = order[:final_k]

    out: List[RetrievalResult] = []
    total_chars = 0
    for new_rank, idx in enumerate(chosen_idx, start=1):
        chunk = trunc[idx]
        if total_chars + len(chunk.text) > max_tot and out:
            break
        out.append(
            RetrievalResult(
                rank=new_rank,
                doc_id=chunk.doc_id,
                text=chunk.text,
                cosine=chunk.cosine,
                infonce_risk=chunk.infonce_risk,
                privacy_score=chunk.privacy_score,
                l2_distance=chunk.l2_distance,
            )
        )
        total_chars += len(chunk.text)
    stats["context_char_total"] = int(total_chars)
    stats["out_n"] = len(out)
    return out, stats


def measure_rss_mb() -> float:
    try:
        import psutil  # type: ignore[import-not-found]

        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024 * 1024)
    except Exception:
        return float("nan")


def measure_uss_mb() -> float:
    try:
        import psutil  # type: ignore[import-not-found]

        proc = psutil.Process(os.getpid())
        info = proc.memory_full_info()
        uss = getattr(info, "uss", None)
        if uss is None:
            return float("nan")
        return float(uss) / (1024 * 1024)
    except Exception:
        return float("nan")


def measure_model_file_mb(rag: Optional[RAGGenerator]) -> float:
    if rag is None:
        return 0.0
    try:
        path = Path(getattr(rag, "model_path", "") or "")
        if path.is_file():
            return float(path.stat().st_size) / (1024 * 1024)
    except Exception:
        pass
    return 0.0
