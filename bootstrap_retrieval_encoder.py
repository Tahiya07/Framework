#!/usr/bin/env python
"""One-time download of the CPU retrieval encoder (for offline Streamlit)."""

from __future__ import annotations

from retrieval_config import get_retrieval_encoder_profile


def main() -> None:
    from transformers import AutoModel, AutoTokenizer

    profile = get_retrieval_encoder_profile()
    print(f"Downloading retrieval encoder: {profile.model_name}")
    AutoTokenizer.from_pretrained(profile.model_name)
    AutoModel.from_pretrained(profile.model_name)
    print("Done. Restart Streamlit and rebuild your corpus (Build / Refresh Corpus).")


if __name__ == "__main__":
    main()
