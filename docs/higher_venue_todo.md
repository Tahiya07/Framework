# Higher-Venue Readiness Checklist

## Required Before Submission

- [ ] Run large privacy benchmark with at least 500 prompts.
- [ ] Compare against no-guard, role-only, rule-only, centralized learned guard, federated no-DP, federated DP, and full hybrid systems.
- [ ] Report formal threat model.
- [ ] Report DP parameters: clipping norm, noise multiplier, rounds, sampling rate, delta, epsilon estimate/accountant.
- [ ] Run non-IID federated splits by teacher/course/domain.
- [ ] Add at least 300 QA/RAG questions from real course or public QA data.
- [ ] Add at least 100 multimodal PDF/image cases.
- [ ] Run human teacher moderation evaluation.
- [ ] Fix pipeline reproducibility so one command regenerates all paper artifacts.
- [ ] Remove or archive stale scripts before artifact submission.

## Strong Paper Framing

The main contribution should be framed as a privacy-constrained educational RAG and moderation architecture, not as a new Bloom classifier.

## Weak Claims To Avoid

- Guaranteed privacy.
- Impossible leakage.
- Production-ready DP.
- Robust cross-domain Bloom transfer.
- Multi-SLM improvement without specialist training evidence.

