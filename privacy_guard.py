from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from ingestion import DocumentChunk
from retriever import RetrievalResult


STUDENT_REFUSAL = (
    "I can't reveal or reconstruct protected exam content. "
    "I can help with Bloom classification guidance, study concepts, or similarly scoped practice questions instead."
)

QUERY_RISK_PATTERNS = [
    r"\bverbatim\b",
    r"\bexact (text|question|paper)\b",
    r"\breconstruct\b",
    r"\brecover\b",
    r"\breveal\b",
    r"\bshow (me )?(the )?(question paper|exam paper|uploaded exam)\b",
    r"\bwhat (are|were) the questions\b",
    r"\bfull (question paper|exam paper|paper)\b",
    r"\blist (all|the) questions\b",
    r"\bcopy\b",
    r"\bquote\b",
    r"\bmoderation paper\b",
    r"\bwithout (copying|quoting)\b.*\b(exam|paper|question)",
    r"\bin (your own words|plain english)\b.*\b(exam|paper|question)",
    r"\bparaphrase\b.*\b(exam|paper|question|protected|uploaded)",
    r"\btopics?\b.*\b(uploaded|protected|exam|question paper)",
    r"\bwhat does the (uploaded|protected) (exam|paper)\b.*\bcover\b",
    r"\bsummarize\b.*\b(uploaded|protected|previously uploaded)\b.*\b(document|documents|exam|paper|questions?)\b",
    r"\b(previously uploaded|uploaded)\b.*\b(document|documents|exam|paper|questions?)\b",
    r"\bgive\b.*\b(uploaded|protected|previously uploaded)\b.*\b(document|documents|exam|paper|questions?)\b",
    r"\bfirst\b.*\bthen\b.*\b(protected|exam|paper)",
    r"\bprevious answer\b.*\b(protected|exam|question)",
]

PROTECTED_ARTIFACT_PATTERNS = [
    r"\buploaded\b",
    r"\bprotected\b",
    r"\bexam\b",
    r"\bquestion paper\b",
    r"\buploaded document\b",
    r"\buploaded documents\b",
    r"\bpreviously uploaded\b",
    r"\bpaper\b",
    r"\bmoderation\b",
]

MIN_EXTRACTIVE_SPAN_TOKENS = 8
MIN_NGRAM_SIZE = 5
MAX_STUDENT_PROTECTED_OVERLAP = 0.55
MAX_SEMANTIC_CONCEPT_RATIO = 0.62

_CONCEPT_STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "can", "for", "from", "give",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "q1", "q2", "q3",
    "question", "the", "this", "to", "was", "what", "with", "you",
}

_SEMANTIC_ALIASES: Dict[str, Set[str]] = {
    "current": {"current", "amperage", "flow"},
    "voltage": {"voltage", "potential", "potentialdifference"},
    "resistor": {"resistor", "resistance", "ohm", "ohms"},
    "tcp": {"tcp", "transmissioncontrol"},
    "udp": {"udp", "datagram"},
    "reliability": {"reliability", "reliable", "delivery", "resend", "acknowledge"},
    "ordering": {"ordering", "ordered", "sequence", "sequencing"},
    "overhead": {"overhead", "cost", "extra"},
    "merge": {"merge", "mergesort", "divide", "conquer"},
    "complexity": {"complexity", "runtime", "time", "efficiency"},
}


@dataclass
class PrivacyDecision:
    allowed: bool
    reason: str
    risk_score: float


def _norm_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _concept_tokens(text: str) -> Set[str]:
    toks = {tok for tok in _norm_tokens(text) if len(tok) >= 3 and tok not in _CONCEPT_STOPWORDS}
    expanded = set(toks)
    for canonical, aliases in _SEMANTIC_ALIASES.items():
        if toks & aliases:
            expanded.add(canonical)
    return expanded


def _token_ngrams(tokens: Sequence[str], n: int) -> Set[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def _overlap_ratio(a: str, b: str) -> float:
    aa = Counter(_norm_tokens(a))
    bb = Counter(_norm_tokens(b))
    n = max(1, sum(aa.values()))
    return float(sum((aa & bb).values()) / n)


def _longest_common_token_run(a: str, b: str) -> int:
    """Return the longest contiguous token span shared by two texts."""
    aa = _norm_tokens(a)
    bb = _norm_tokens(b)
    if not aa or not bb:
        return 0
    prev = [0] * (len(bb) + 1)
    best = 0
    for tok_a in aa:
        cur = [0] * (len(bb) + 1)
        for j, tok_b in enumerate(bb, start=1):
            if tok_a == tok_b:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return int(best)


def _ngram_containment_ratio(candidate: str, protected: str, n: int = MIN_NGRAM_SIZE) -> float:
    """Fraction of candidate n-grams that appear in protected content."""
    cand = _token_ngrams(_norm_tokens(candidate), n)
    if not cand:
        return 0.0
    prot = _token_ngrams(_norm_tokens(protected), n)
    if not prot:
        return 0.0
    return float(len(cand & prot) / len(cand))


def protected_leakage_score(candidate: str, protected_text: str) -> Dict[str, float]:
    """Compute deterministic extractive-overlap leakage indicators.

    These are not formal privacy guarantees. They are local, auditable guards
    against reconstructing or copying protected exam-paper text.
    """
    cand_concepts = _concept_tokens(candidate)
    protected_concepts = _concept_tokens(protected_text)
    semantic_ratio = 0.0
    if cand_concepts and protected_concepts:
        semantic_ratio = len(cand_concepts & protected_concepts) / max(1, len(cand_concepts))
    return {
        "overlap_ratio": _overlap_ratio(candidate, protected_text),
        "longest_common_span": float(_longest_common_token_run(candidate, protected_text)),
        "ngram_containment": _ngram_containment_ratio(candidate, protected_text),
        "semantic_concept_ratio": float(semantic_ratio),
    }


def assess_query_privacy_risk(query: str) -> PrivacyDecision:
    q = (query or "").strip()
    if not q:
        return PrivacyDecision(False, "empty_query", 1.0)
    hits = 0
    for pat in QUERY_RISK_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            hits += 1
    risk = min(1.0, 0.2 * hits)
    if hits > 0:
        return PrivacyDecision(False, "reconstruction_intent_detected", risk)
    return PrivacyDecision(True, "ok", risk)


def assess_student_query_against_protected_corpus(
    query: str,
    protected_chunks: Sequence[DocumentChunk],
) -> PrivacyDecision:
    base = assess_query_privacy_risk(query)
    if not base.allowed:
        return base
    union = protected_text_union(protected_chunks)
    if not union:
        return PrivacyDecision(True, "no_protected_corpus", 0.0)
    overlap = _overlap_ratio(query, union)
    leakage = protected_leakage_score(query, union)
    mentions_artifact = any(
        re.search(pat, query or "", flags=re.IGNORECASE) for pat in PROTECTED_ARTIFACT_PATTERNS
    )
    if mentions_artifact and overlap >= 0.12:
        return PrivacyDecision(False, "artifact_reference_with_overlap", overlap)
    if leakage["longest_common_span"] >= MIN_EXTRACTIVE_SPAN_TOKENS:
        return PrivacyDecision(
            False,
            "protected_span_in_query",
            min(1.0, leakage["longest_common_span"] / MIN_EXTRACTIVE_SPAN_TOKENS),
        )
    if mentions_artifact and leakage["ngram_containment"] >= 0.35:
        return PrivacyDecision(False, "protected_ngram_query", leakage["ngram_containment"])
    if mentions_artifact and leakage["semantic_concept_ratio"] >= MAX_SEMANTIC_CONCEPT_RATIO:
        return PrivacyDecision(False, "protected_semantic_query", leakage["semantic_concept_ratio"])
    if overlap >= 0.85 and len(_norm_tokens(query)) >= 8:
        return PrivacyDecision(False, "query_overlap_high", overlap)
    return PrivacyDecision(True, "ok", overlap)


def partition_chunks(chunks: Sequence[DocumentChunk]) -> Dict[str, List[DocumentChunk]]:
    out = {"public": [], "protected": []}
    for chunk in chunks:
        level = "protected" if getattr(chunk, "access_level", "public") == "protected" else "public"
        out[level].append(chunk)
    return out


def allowed_chunks_for_role(
    requester_role: str,
    public_chunks: Sequence[DocumentChunk],
    protected_chunks: Sequence[DocumentChunk],
    access_scope: str,
) -> List[DocumentChunk]:
    role = requester_role.lower().strip()
    scope = access_scope.lower().strip()
    if role in {"teacher", "moderator", "admin"} and scope == "protected":
        return list(protected_chunks)
    return list(public_chunks)


def protected_text_union(chunks: Sequence[DocumentChunk], max_chars: int = 40_000) -> str:
    buf: List[str] = []
    total = 0
    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        piece = text[:remaining]
        buf.append(piece)
        total += len(piece)
    return "\n".join(buf)


def screen_generation_output(
    requester_role: str,
    query: str,
    answer: str,
    protected_chunks: Sequence[DocumentChunk],
) -> PrivacyDecision:
    role = requester_role.lower().strip()
    if role in {"teacher", "moderator", "admin"}:
        return PrivacyDecision(True, "teacher_access", 0.0)
    risk = assess_student_query_against_protected_corpus(query, protected_chunks)
    if not risk.allowed:
        return risk
    union = protected_text_union(protected_chunks)
    if not union:
        return PrivacyDecision(True, "no_protected_corpus", 0.0)
    leakage = protected_leakage_score(answer, union)
    overlap = leakage["overlap_ratio"]
    if leakage["longest_common_span"] >= MIN_EXTRACTIVE_SPAN_TOKENS:
        return PrivacyDecision(
            False,
            "protected_span_copied",
            min(1.0, leakage["longest_common_span"] / MIN_EXTRACTIVE_SPAN_TOKENS),
        )
    if leakage["ngram_containment"] >= 0.35 and len(_norm_tokens(answer)) >= MIN_NGRAM_SIZE:
        return PrivacyDecision(False, "protected_ngram_copied", leakage["ngram_containment"])
    if leakage["semantic_concept_ratio"] >= MAX_SEMANTIC_CONCEPT_RATIO and len(_concept_tokens(answer)) >= 4:
        return PrivacyDecision(False, "protected_semantic_leakage", leakage["semantic_concept_ratio"])
    if overlap >= MAX_STUDENT_PROTECTED_OVERLAP and len(_norm_tokens(answer)) >= 8:
        return PrivacyDecision(False, "protected_overlap_high", overlap)
    return PrivacyDecision(True, "ok", max(overlap, leakage["ngram_containment"]))


def policy_instruction(requester_role: str, access_scope: str) -> str:
    role = requester_role.lower().strip()
    scope = access_scope.lower().strip()
    if role in {"teacher", "moderator", "admin"} and scope == "protected":
        return (
            "This is a protected moderation workflow. Use the context only for classification, moderation support, "
            "or high-level analysis. Do not quote long spans verbatim."
        )
    return (
        "Never reveal, reconstruct, or quote protected exam content. If the request seeks exact exam wording, "
        "full question lists, or reconstruction, refuse and offer high-level study help instead. "
        "Use protected material only to produce abstract guidance, concepts, or new practice questions."
    )
