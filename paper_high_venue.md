# A Lightweight Multi-Modal Tiny LLM Framework for Privacy-Constrained Academic Assistance in University Environments

This draft is evidence-aligned and intentionally avoids formal privacy or universal generalization claims.

## Core Claim
A local role-separated framework can support public student RAG and teacher-only protected exam moderation with bounded empirical leakage resistance.

## Current Evidence
- Figshare Bloom test accuracy: 0.769; macro-F1: 0.744.
- Student attack block rate: 1.000.
- Student benign allow rate: 0.867.
- Teacher moderation allow rate: 1.000.
- Privacy evaluation rows: 125; attack prompts: 104.
- QA/RAG questions: 10.
- PDF RAG smoke test: True; image RAG smoke test: True.
- OCR backend: pytesseract.

## Claims To Avoid
- Perfect privacy.
- Differential privacy, unless a formal DP mechanism is added and evaluated.
- Solved cross-domain Bloom generalization.
- Fully validated multimodal assistance beyond the measured OCR/image-ingestion path.
