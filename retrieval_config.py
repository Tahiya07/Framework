"""CPU retrieval encoder profiles (shared by PrivacyRetriever)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEncoderProfile:
    key: str
    model_name: str
    embed_dim: int
    query_prefix: str
    passage_prefix: str
    max_length: int = 384
    encode_batch_size: int = 12
    display_name: str = ""

    @property
    def label(self) -> str:
        return self.display_name or self.model_name


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RETRIEVAL_ENCODER_PROFILES: dict[str, RetrievalEncoderProfile] = {
    "bge-small": RetrievalEncoderProfile(
        key="bge-small",
        model_name="BAAI/bge-small-en-v1.5",
        embed_dim=384,
        query_prefix=BGE_QUERY_PREFIX,
        passage_prefix="",
        max_length=384,
        encode_batch_size=12,
        display_name="BGE-small-en-v1.5 (recommended, CPU)",
    ),
    "minilm": RetrievalEncoderProfile(
        key="minilm",
        model_name="all-MiniLM-L6-v2",
        embed_dim=384,
        query_prefix="",
        passage_prefix="",
        max_length=256,
        encode_batch_size=16,
        display_name="all-MiniLM-L6-v2 (legacy)",
    ),
}

DEFAULT_RETRIEVAL_ENCODER_KEY = "bge-small"


def normalize_encoder_key(value: str | None) -> str:
    if not value:
        return DEFAULT_RETRIEVAL_ENCODER_KEY
    key = value.strip().lower().replace("_", "-")
    aliases = {
        "bge": "bge-small",
        "bge-small-en-v1.5": "bge-small",
        "baai/bge-small-en-v1.5": "bge-small",
        "mini-lm": "minilm",
        "all-minilm-l6-v2": "minilm",
        "sentence-transformers/all-minilm-l6-v2": "minilm",
    }
    return aliases.get(key, key)


def get_retrieval_encoder_profile(encoder: str | None = None) -> RetrievalEncoderProfile:
    """Resolve encoder profile from env ``RETRIEVAL_ENCODER`` or explicit key/name."""
    raw = encoder or os.environ.get("RETRIEVAL_ENCODER") or DEFAULT_RETRIEVAL_ENCODER_KEY
    key = normalize_encoder_key(raw)
    if key in RETRIEVAL_ENCODER_PROFILES:
        return RETRIEVAL_ENCODER_PROFILES[key]
    if "/" in raw:
        return RetrievalEncoderProfile(
            key=raw,
            model_name=raw.strip(),
            embed_dim=384,
            query_prefix=BGE_QUERY_PREFIX if "bge" in raw.lower() else "",
            passage_prefix="",
            max_length=384,
            encode_batch_size=12,
            display_name=raw.strip(),
        )
    raise ValueError(
        f"Unknown retrieval encoder {raw!r}. "
        f"Choose from: {', '.join(RETRIEVAL_ENCODER_PROFILES)} or a HuggingFace model id."
    )
