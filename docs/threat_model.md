# Threat Model: Privacy-Preserving Educational RAG and Question Moderation

## Protected Assets

The system treats the following as protected assessment assets:

- Exact uploaded exam/question wording.
- Near-verbatim paraphrases of protected questions.
- Full lists of protected questions or sections.
- Extractive spans, clauses, keywords, or answerable fragments that allow reconstruction.
- Protected assessment context retrieved from teacher-only corpora.
- Teacher/client-local moderation data used to train privacy-risk models.

The system does not treat general study concepts as protected when they can be answered without reconstructing uploaded assessment wording.

## Roles

### Student

Students may query public learning material and receive high-level study support. Students must not retrieve, reconstruct, paraphrase, list, or infer protected uploaded question context.

### Teacher / Moderator

Teachers may access protected context for moderation tasks such as Bloom labeling, ambiguity review, difficulty review, and abstract revision guidance. Teacher-facing outputs must still avoid quoting or closely paraphrasing protected question wording.

### Aggregation Server

The server may receive only secure-aggregation-compatible clipped and noised model updates for the federated privacy-risk model. It must not receive raw teacher questions, raw protected exam text, per-teacher examples, per-teacher metrics, or client vocabularies.

### Adversarial Student

The adversary may attempt direct reconstruction, indirect leakage, paraphrase probing, partial-span extraction, semantic cloning, role spoofing, and jailbreak prompts.

## Security Boundaries

The system uses layered controls:

1. Role-separated public/protected corpora.
2. Student retrieval restricted to public chunks.
3. Teacher retrieval allowed for protected chunks only within moderation workflows.
4. Query-intent screening for reconstruction attempts.
5. Output screening for copied spans, protected n-grams, high overlap, and semantic leakage.
6. Federated privacy-risk model trained from client-local examples.
7. Clipped and noised client updates with privacy accounting metadata.
8. Fail-closed moderation output: teacher output is withheld if it copies protected wording.

## Out-of-Scope Threats

The current prototype does not fully defend against:

- Compromised local machine or filesystem.
- Malicious teacher intentionally copying content outside the system.
- Model inversion attacks against saved aggregate model beyond the reported DP/noise setting.
- OCR or ingestion bugs that mislabel protected content as public.
- Full formal secure aggregation protocol implementation.

## Publishable Claim Boundary

Acceptable claim:

> The system reduces protected-question leakage risk using role-separated retrieval, output leakage screening, and a federated DP-noised privacy-risk backstop under an explicit adversarial prompt taxonomy.

Claims to avoid:

- Privacy is guaranteed.
- Leakage is impossible.
- Federated learning alone ensures privacy.
- Formal differential privacy is proven beyond the reported accounting assumptions.

