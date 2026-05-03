from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT = ROOT / "paper_draft.pdf"
MD_OUT = ROOT / "paper_high_venue.md"


def _load_json(name: str, default: Any = None) -> Any:
    path = RESULTS / name
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _style() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=16,
            leading=19,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "authors": ParagraphStyle(
            "Authors",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=12,
            leading=14,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=10.5,
            leading=12,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.2,
            leading=11.4,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8,
            leading=9.5,
            spaceAfter=3,
        ),
    }
    return styles


def _p(text: str, styles: Dict[str, ParagraphStyle], key: str = "body") -> Paragraph:
    return Paragraph(text, styles[key])


def _fig(path: Path, caption: str, styles: Dict[str, ParagraphStyle], width: float = 5.8) -> List[Any]:
    if not path.is_file():
        return []
    img = Image(str(path))
    img._restrictSize(width * inch, 3.2 * inch)
    return [Spacer(1, 4), img, _p(caption, styles, "small"), Spacer(1, 4)]


def _table(rows: List[List[Any]], widths: List[float], styles: Dict[str, ParagraphStyle]) -> Table:
    wrapped = []
    for row in rows:
        wrapped.append([
            cell if not isinstance(cell, str) else Paragraph(cell, styles["small"])
            for cell in row
        ])
    tbl = Table(wrapped, colWidths=[w * inch for w in widths], repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AAB4C3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def _extract_cross_rows(unified: List[Dict[str, Any]]) -> List[List[str]]:
    wanted = []
    for row in unified:
        if row.get("evidence_area") == "cognitive robustness":
            setting = row.get("setting", "")
            if "in-domain" in setting or "->" in setting:
                wanted.append(
                    [
                        row.get("protocol", ""),
                        setting,
                        row.get("model", ""),
                        _fmt(row.get("primary_value")),
                        _fmt(row.get("accuracy")),
                        _fmt(row.get("within_one_level_accuracy")),
                        _fmt(row.get("severe_error_rate")),
                    ]
                )
    return wanted


def build() -> None:
    styles = _style()
    figshare = _load_json("figshare_bloom_v1_evaluation.json")
    guard = _load_json("privacy_guard_eval.json")
    qa_eval = _load_json("qa_rag_eval.json")
    qwen_eval = _load_json("qwen_rag_eval.json")
    ocr_eval = _load_json("ocr_image_pipeline_eval.json")
    multimodal_rag = _load_json("multimodal_rag_eval.json")
    unified_payload = _load_json("unified_results_table.json", [])
    unified = unified_payload.get("rows", []) if isinstance(unified_payload, dict) else unified_payload
    refs = json.loads((ROOT / "refs.json").read_text(encoding="utf-8"))
    fig_test = figshare.get("test_metrics", {})
    audit = figshare.get("preprocessing_audit", {})

    story: List[Any] = []
    story.append(
        _p(
            "A Lightweight Multi-Modal Tiny LLM Framework for Privacy-Constrained "
            "Academic Assistance in University Environments",
            styles,
            "title",
        )
    )
    story.append(_p("Draft manuscript - evidence-aligned version", styles, "authors"))

    story.append(_p("Abstract", styles, "h1"))
    story.append(
        _p(
            "Universities increasingly need local academic assistants that can answer questions "
            "from course material while preventing protected assessment artifacts from leaking to "
            "students. We present a lightweight, CPU-oriented framework that separates public "
            "learning corpora from protected exam corpora, supports auditable PDF/text ingestion "
            "with an optional OCR-backed image path, "
            "performs Bloom-taxonomy classification for moderation, and applies role-aware query "
            "and output screening before student-facing generation. The paper does not claim "
            "formal differential privacy. Instead, it evaluates bounded empirical resistance under "
            "an explicit university threat model. On the Figshare Bloom exam dataset, the selected "
            "classifier reaches 0.769 test accuracy and 0.744 macro-F1 after duplicate/conflict "
            "auditing. Cross-dataset transfer with MoocRadar shows large exact-label degradation, "
            "indicating that Bloom classifiers remain domain-sensitive; ordinal metrics expose "
            "where severe cognitive-level jumps are reduced or amplified. An expanded role-aware "
            "privacy evaluation blocks all evaluated student reconstruction attacks, allows 0.857 "
            "of benign student study prompts, and allows teacher moderation prompts. A small "
            "offline QA/RAG evaluation reports answer F1, retrieval hit rate, and unsupported-answer "
            "risk; Qwen2.5-1.5B-Instruct-Q4_K_M.gguf is then evaluated over the same "
            "retrieved contexts. The image pipeline evaluation reports OCR readiness or measured OCR quality "
            "depending on the installed backend. These results "
            "support a conservative claim: local role separation plus leakage screening is a "
            "practical deployment pattern for privacy-constrained academic assistance, but not a "
            "proof of universal privacy or cross-domain Bloom generalization.",
            styles,
        )
    )

    story.append(_p("1. Introduction", styles, "h1"))
    story.append(
        _p(
            "RAG-based educational assistants can help students ask questions over lecture PDFs, "
            "images, and notes, while helping teachers moderate exam questions and label Bloom "
            "levels. In a university environment these two workflows must not share the same "
            "retrieval surface: a student should not be able to recover protected exam uploads by "
            "asking for previous documents, summaries, paraphrases, or topic lists. This creates a "
            "deployment gap between general privacy-preserving RAG research and education-specific "
            "Bloom-classification systems.",
            styles,
        )
    )
    story.append(
        _p(
            "The central research question is: how far can a lightweight local framework go in "
            "combining multimodal academic ingestion, Bloom-aware moderation, and role-aware "
            "privacy controls without overclaiming formal security? We answer with a system and "
            "evaluation protocol that makes failures visible: exact Bloom transfer degrades under "
            "domain shift, privacy is evaluated with attack prompts rather than asserted, and "
            "protected assessment material is handled as a teacher-only resource.",
            styles,
        )
    )
    story.append(_p("Contributions", styles, "h2"))
    for item in [
        "A deployment pattern for privacy-constrained educational RAG that separates public student retrieval from protected teacher moderation.",
        "An evaluation protocol spanning answer correctness, retrieval quality, unsupported-answer risk, role-aware privacy attacks, Bloom robustness, and image/OCR ingestion.",
        "A Bloom-classification analysis framed as evaluation insight: exact cross-domain labels are brittle, while ordinal error metrics reveal usable moderation risk structure.",
        "A conservative claim policy that distinguishes taxonomy-based empirical leakage resistance from formal privacy guarantees.",
    ]:
        story.append(_p(f"- {item}", styles))

    story.append(_p("2. Related Work and Gap", styles, "h1"))
    story.append(
        _p(
            "Prior RAG work establishes the utility of retrieval augmentation and also documents "
            "privacy risks in retrieval and generation. Recent privacy-preserving RAG work explores "
            "differential privacy, query protection, cloud RAG, tabular settings, and multimodal "
            "leakage. In parallel, educational NLP work has improved automated Bloom classification "
            "for exam questions. The gap addressed here is not a new cryptographic mechanism; it is "
            "the integrated university setting where public study material, protected exams, "
            "student access, teacher moderation, and Bloom labeling coexist in one lightweight local "
            "workflow.",
            styles,
        )
    )

    story.append(_p("3. System and Threat Model", styles, "h1"))
    story.extend(
        _fig(
            FIGURES / "system_architecture.png",
            "Figure 1. Role-separated local architecture. Public material is indexed for student RAG; protected exam material is routed through teacher-only moderation and student-facing leakage screening.",
            styles,
            width=6.2,
        )
    )
    story.append(
        _p(
            "The attacker is a student with black-box access to the assistant. The attacker may ask "
            "for uploaded documents, paraphrases, summaries, exact questions, partial spans, or "
            "semantically equivalent practice items. The teacher/moderator role is trusted to view "
            "protected exam material for moderation and Bloom labeling. The framework targets "
            "bounded reconstruction resistance and operational separation; it does not defend "
            "against compromised teacher accounts, database exfiltration, side channels, or formal "
            "membership inference.",
            styles,
        )
    )

    story.append(_p("4. Datasets and Protocol", styles, "h1"))
    dataset_rows = [
        ["Dataset", "Use", "Rows after audit", "Notes"],
        [
            "Figshare Bloom exam questions",
            "Primary in-domain Bloom classification",
            str(audit.get("final_rows", "NA")),
            "Exact duplicates removed; conflicting duplicate labels removed.",
        ],
        [
            "MoocRadar problem set",
            "External cross-domain Bloom transfer",
            "9324",
            "Used to test domain shift, not to claim universal generalization.",
        ],
        [
            "Synthetic protected exam micro-corpus",
            "Privacy-guard attack and utility probes",
            str(guard.get("n_rows", "NA")),
            "Small, controlled corpus for policy evaluation; not a real deployment study.",
        ],
        [
            "Offline academic QA set",
            "Answer correctness and retrieval grounding; extractive and Qwen-generated",
            str(qa_eval.get("n_questions", "NA")),
            "Small local QA benchmark; external HotpotQA/NQ-style evaluation remains future work.",
        ],
        [
            "Synthetic typed image set",
            "OCR/image ingestion path",
            str(ocr_eval.get("n_samples", "NA")),
            "Reports OCR quality when a backend is installed; otherwise reports backend unavailability.",
        ],
        [
            "Generated PDF/image RAG smoke set",
            "End-to-end file ingestion and extractive RAG",
            str(multimodal_rag.get("n_cases", "NA")),
            "Tests PDF RAG and OCR-backed image RAG as separate pipeline cases.",
        ],
    ]
    story.append(_table(dataset_rows, [1.45, 1.65, 0.95, 2.45], styles))
    story.append(
        _p(
            "Dataset usage is therefore aligned with claim scope: Figshare supports in-domain "
            "Bloom evidence, MoocRadar supports cross-domain stress testing, the protected "
            "micro-corpus supports guard-behavior analysis, the offline QA set tests grounding "
            "behavior, and the synthetic image set tests whether the OCR ingestion path is "
            "measurable. The present artifacts do not support claims about large-scale real "
            "university deployment or formal privacy.",
            styles,
        )
    )

    story.append(_p("5. Results", styles, "h1"))
    story.append(_p("5.1 In-domain Bloom classification", styles, "h2"))
    bloom_rows = [
        ["Metric", "Value"],
        ["Accuracy", _fmt(fig_test.get("accuracy"))],
        ["Macro-F1", _fmt(fig_test.get("macro_f1"))],
        ["Weighted-F1", _fmt(fig_test.get("weighted_f1"))],
        ["Mean ordinal error", _fmt(fig_test.get("mean_ordinal_error"))],
        ["Within-one-level accuracy", _fmt(fig_test.get("within_one_level_accuracy"))],
        ["Severe ordinal error", _fmt(fig_test.get("severe_error_rate"))],
    ]
    story.append(_table(bloom_rows, [2.7, 1.2], styles))

    story.append(_p("5.2 Domain shift", styles, "h2"))
    cross_rows = [["Protocol", "Setting", "Model", "Macro-F1", "Acc.", "Within-1", "Severe"]]
    cross_rows.extend(_extract_cross_rows(unified)[:8])
    story.append(_table(cross_rows, [1.2, 1.35, 1.05, 0.65, 0.55, 0.65, 0.55], styles))
    story.extend(
        _fig(
            FIGURES / "cross_domain_performance_table.png",
            "Figure 2. Cross-domain Bloom performance. Exact transfer degrades sharply, so the claim is cognitive robustness analysis rather than solved generalization.",
            styles,
            width=6.1,
        )
    )
    story.extend(
        _fig(
            FIGURES / "domain_shift_preserves_ordinal_structure.png",
            "Figure 3. Ordinal analysis under domain shift. Within-one and severe-error metrics expose structure hidden by exact accuracy.",
            styles,
            width=5.5,
        )
    )

    story.append(_p("5.3 Role-aware privacy guard", styles, "h2"))
    privacy_rows = [
        ["Metric", "Value", "Interpretation"],
        [
            "Student attack block rate",
            _fmt(guard.get("student_attack_block_rate")),
            "Measured on the defined prompt taxonomy only.",
        ],
        [
            "Student benign allow rate",
            _fmt(guard.get("student_benign_allow_rate")),
            "Ordinary study help is mostly preserved; blocked cases motivate interface wording.",
        ],
        [
            "Teacher moderation allow rate",
            _fmt(guard.get("teacher_moderation_allow_rate")),
            "Trusted moderation workflow remains available.",
        ],
        [
            "Attack prompts",
            str(guard.get("n_attack_prompts", "NA")),
            "Expanded synthetic adversarial set, still below a full external red-team study.",
        ],
        [
            "Total privacy rows",
            str(guard.get("n_rows", "NA")),
            "Includes attacks, benign student prompts, and teacher moderation prompts.",
        ],
    ]
    story.append(_table(privacy_rows, [1.65, 0.8, 3.5], styles))
    story.append(
        _p(
            "The privacy guard evidence is promising but bounded. A high venue version should "
            "emphasize that the expanded prompt evaluation is an attack taxonomy, not a proof. The "
            "student benign allow rate below 1.0 is also useful: it reveals the utility cost of "
            "strict protected-artifact screening and motivates future adaptive policies.",
            styles,
        )
    )

    story.append(_p("5.4 QA/RAG grounding", styles, "h2"))
    qa_rows = [["System", "Token-F1", "EM", "Hit@1", "Hit@3", "Unsupported"]]
    qwen_qa = qwen_eval.get("academic_qa", {}) if isinstance(qwen_eval, dict) else {}
    if qwen_qa:
        qa_rows.append(
            [
                "Qwen2.5-1.5B-Q4_K_M",
                _fmt(qwen_qa.get("token_f1", {}).get("mean")),
                _fmt(qwen_qa.get("exact_match", {}).get("mean")),
                _fmt(qwen_qa.get("retrieval_hit_at_1", {}).get("mean")),
                _fmt(qwen_qa.get("retrieval_hit_at_3", {}).get("mean")),
                _fmt(qwen_qa.get("hallucination_proxy_rate", {}).get("mean")),
            ]
        )
    systems = qa_eval.get("systems", {}) if isinstance(qa_eval, dict) else {}
    for system_name in ["Proposed", "VanillaRAG", "BM25", "NoRAG"]:
        item = systems.get(system_name, {})
        if not item:
            continue
        qa_rows.append(
            [
                system_name,
                _fmt(item.get("token_f1", {}).get("mean")),
                _fmt(item.get("exact_match", {}).get("mean")),
                _fmt(item.get("retrieval_hit_at_1", {}).get("mean")),
                _fmt(item.get("retrieval_hit_at_3", {}).get("mean")),
                _fmt(item.get("hallucination_proxy_rate", {}).get("mean")),
            ]
        )
    if len(qa_rows) == 1:
        qa_rows.append(["Not run", "NA", "NA", "NA", "NA", "NA"])
    story.append(_table(qa_rows, [1.2, 0.75, 0.55, 0.55, 0.55, 0.85], styles))
    story.append(
        _p(
            "This QA/RAG evaluation is deliberately small and offline. It measures answer "
            "correctness, support-document retrieval, and an unsupported-answer proxy. The primary "
            "generator row uses Qwen2.5-1.5B-Instruct-Q4_K_M.gguf through llama.cpp with greedy "
            "decoding; the extractive rows remain as diagnostic controls. It should be replaced "
            "or supplemented with HotpotQA, Natural "
            "Questions, DocVQA, or instructor-authored course QA before making broad utility claims.",
            styles,
        )
    )

    story.append(_p("5.5 OCR and image pipeline", styles, "h2"))
    ocr_status = ocr_eval.get("backend_status", {}) if isinstance(ocr_eval, dict) else {}
    ocr_metrics = ocr_eval.get("metrics", {}) if isinstance(ocr_eval, dict) else {}
    ocr_rows = [
        ["Metric", "Value"],
        ["Qwen model", str(qwen_eval.get("model_id", "NA") if isinstance(qwen_eval, dict) else "NA")],
        ["Qwen model file MB", _fmt(qwen_eval.get("model_file_mb") if isinstance(qwen_eval, dict) else None)],
        ["Qwen PDF RAG", str(qwen_eval.get("multimodal_rag", {}).get("pdf_rag_ok", "NA") if isinstance(qwen_eval, dict) else "NA")],
        ["Qwen image RAG", str(qwen_eval.get("multimodal_rag", {}).get("image_rag_ok", "NA") if isinstance(qwen_eval, dict) else "NA")],
        ["PDF RAG smoke test", str(multimodal_rag.get("pdf_rag_ok", "NA") if isinstance(multimodal_rag, dict) else "NA")],
        ["Image RAG smoke test", str(multimodal_rag.get("image_rag_ok", "NA") if isinstance(multimodal_rag, dict) else "NA")],
        ["RAG answer accuracy on OK cases", _fmt(multimodal_rag.get("answer_accuracy_on_ok_cases") if isinstance(multimodal_rag, dict) else None)],
        ["Backend", str(ocr_status.get("engine", "NA"))],
        ["Backend available", str(ocr_status.get("available", "NA"))],
        ["Samples evaluated", str(ocr_eval.get("n_samples", "NA") if isinstance(ocr_eval, dict) else "NA")],
        ["Mean OCR token-F1", _fmt(ocr_metrics.get("mean_token_f1"))],
        ["Mean word error rate", _fmt(ocr_metrics.get("mean_word_error_rate"))],
        ["Access-level preservation", _fmt(ocr_metrics.get("access_level_preservation"))],
        ["Modality preservation", _fmt(ocr_metrics.get("modality_preservation"))],
    ]
    story.append(_table(ocr_rows, [2.15, 1.35], styles))
    story.append(
        _p(
            "The multimodal claim is therefore scoped to an implemented and measurable image "
            "ingestion path plus a PDF RAG path, both evaluated with Qwen over retrieved context. "
            "PDF RAG is tested with native text extraction. "
            "Image RAG is OCR-backed: if no OCR backend is installed, the result is reported as "
            "missing backend evidence rather than image-understanding performance. Scanned "
            "handwriting, complex layouts, and DocVQA-style reasoning remain outside the current evidence.",
            styles,
        )
    )

    story.append(_p("6. Discussion", styles, "h1"))
    story.append(
        _p(
            "The results do not show a universally robust Bloom classifier or a formally private "
            "RAG system. They do show a publishable systems contribution if framed around the "
            "university deployment gap: local resource-aware operation, explicit role separation, "
            "protected assessment handling, and evaluation that reports failures. The strongest "
            "finding is that exact cognitive-label transfer is brittle, while ordinal metrics and "
            "guarded deployment policies provide a safer basis for moderation support.",
            styles,
        )
    )

    story.append(_p("7. Limitations and Required Next Evidence", styles, "h1"))
    for item in [
        "Privacy is not differential privacy; add DP/query-privacy baselines only if the implementation actually includes them.",
        "The privacy attack set is synthetic and should be expanded with external red-team prompts before submission.",
        "The multimodal image/PDF path needs OCR-quality reporting if the paper foregrounds multimodality.",
        "The QA/RAG benchmark is small and local; replace or supplement it with HotpotQA, Natural Questions, DocVQA, or instructor-authored course QA before broad claims.",
        "A teacher/user study or expert moderation agreement study would strengthen claims for higher venues.",
    ]:
        story.append(_p(f"- {item}", styles))

    story.append(_p("8. Conclusion", styles, "h1"))
    story.append(
        _p(
            "A lightweight university assistant can be made more publishable by narrowing the "
            "claim: not a general privacy solution, but a reproducible local framework that "
            "separates public learning RAG from protected exam moderation and evaluates the "
            "result under domain-shift and leakage probes. This framing fills a real gap between "
            "general privacy-preserving RAG and educational Bloom-classification work.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(_p("References", styles, "h1"))
    for ref in refs:
        title = ref.get("title", "")
        authors = ref.get("authors", "")
        year = ref.get("year", "")
        url = ref.get("url", "")
        story.append(_p(f"{authors}. ({year}). <i>{title}</i>. {url}", styles, "small"))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=LETTER,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    doc.build(story)

    md_lines = [
        "# A Lightweight Multi-Modal Tiny LLM Framework for Privacy-Constrained Academic Assistance in University Environments",
        "",
        "This draft is evidence-aligned and intentionally avoids formal privacy or universal generalization claims.",
        "",
        "## Core Claim",
        "A local role-separated framework can support public student RAG and teacher-only protected exam moderation with bounded empirical leakage resistance.",
        "",
        "## Current Evidence",
        f"- Figshare Bloom test accuracy: {_fmt(fig_test.get('accuracy'))}; macro-F1: {_fmt(fig_test.get('macro_f1'))}.",
        f"- Student attack block rate: {_fmt(guard.get('student_attack_block_rate'))}.",
        f"- Student benign allow rate: {_fmt(guard.get('student_benign_allow_rate'))}.",
        f"- Teacher moderation allow rate: {_fmt(guard.get('teacher_moderation_allow_rate'))}.",
        f"- Privacy evaluation rows: {guard.get('n_rows', 'NA')}; attack prompts: {guard.get('n_attack_prompts', 'NA')}.",
        f"- QA/RAG questions: {qa_eval.get('n_questions', 'NA') if isinstance(qa_eval, dict) else 'NA'}.",
        f"- Qwen GGUF QA token-F1: {_fmt(qwen_eval.get('academic_qa', {}).get('token_f1', {}).get('mean')) if isinstance(qwen_eval, dict) else 'NA'}; EM: {_fmt(qwen_eval.get('academic_qa', {}).get('exact_match', {}).get('mean')) if isinstance(qwen_eval, dict) else 'NA'}.",
        f"- Qwen model: {qwen_eval.get('model_id', 'NA') if isinstance(qwen_eval, dict) else 'NA'} ({_fmt(qwen_eval.get('model_file_mb')) if isinstance(qwen_eval, dict) else 'NA'} MB).",
        f"- PDF RAG smoke test: {multimodal_rag.get('pdf_rag_ok', 'NA') if isinstance(multimodal_rag, dict) else 'NA'}; image RAG smoke test: {multimodal_rag.get('image_rag_ok', 'NA') if isinstance(multimodal_rag, dict) else 'NA'}.",
        f"- OCR backend: {ocr_eval.get('backend_status', {}).get('engine', 'NA') if isinstance(ocr_eval, dict) else 'NA'}.",
        "",
        "## Claims To Avoid",
        "- Perfect privacy.",
        "- Differential privacy, unless a formal DP mechanism is added and evaluated.",
        "- Solved cross-domain Bloom generalization.",
        "- Fully validated multimodal assistance beyond the measured OCR/image-ingestion path.",
    ]
    MD_OUT.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
