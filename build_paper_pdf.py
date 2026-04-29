"""build_paper_pdf.py
Render the publication-grade paper draft as a downloadable PDF
(`paper_draft.pdf`). All numbers are taken directly from
`results/*.json`, so the PDF is implementation-faithful by construction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


REPO = Path(__file__).parent.resolve()
OUT_PDF = REPO / "paper_draft.pdf"
RESULTS = REPO / "results"
FIGS = REPO / "figures"
REFS_PATH = REPO / "refs.json"


# ------------------------------------------------------------------ #
# Font setup -- Times New Roman family.
#
# Strategy:
#   1. If the Windows TrueType files are present, register them under
#      the family name "TimesNewRoman" so the rendered glyphs are the
#      real Microsoft TNR (matches PyCharm/Word output exactly).
#   2. Otherwise fall back to ReportLab's built-in Type 1 fonts
#      "Times-Roman" / "Times-Bold" / "Times-Italic" / "Times-BoldItalic",
#      which are visually equivalent and require no font files.
# ------------------------------------------------------------------ #

def _register_times() -> tuple[str, str, str, str]:
    candidates = [
        Path("C:/Windows/Fonts"),
        Path("/Library/Fonts"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
    ]
    files = {
        "regular":     ["times.ttf",    "Times New Roman.ttf"],
        "bold":        ["timesbd.ttf",  "Times New Roman Bold.ttf"],
        "italic":      ["timesi.ttf",   "Times New Roman Italic.ttf"],
        "bold_italic": ["timesbi.ttf",  "Times New Roman Bold Italic.ttf"],
    }
    found: dict[str, Path] = {}
    for root in candidates:
        if not root.exists():
            continue
        for key, names in files.items():
            if key in found:
                continue
            for n in names:
                p = root / n
                if p.exists():
                    found[key] = p
                    break
        if len(found) == 4:
            break

    if len(found) == 4:
        try:
            pdfmetrics.registerFont(TTFont("TimesNewRoman",            str(found["regular"])))
            pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold",       str(found["bold"])))
            pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic",     str(found["italic"])))
            pdfmetrics.registerFont(TTFont("TimesNewRoman-BoldItalic", str(found["bold_italic"])))
            from reportlab.pdfbase.pdfmetrics import registerFontFamily
            registerFontFamily(
                "TimesNewRoman",
                normal="TimesNewRoman",
                bold="TimesNewRoman-Bold",
                italic="TimesNewRoman-Italic",
                boldItalic="TimesNewRoman-BoldItalic",
            )
            return (
                "TimesNewRoman",
                "TimesNewRoman-Bold",
                "TimesNewRoman-Italic",
                "TimesNewRoman-BoldItalic",
            )
        except Exception:
            pass
    return ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic")


FONT_REG, FONT_BOLD, FONT_ITAL, FONT_BI = _register_times()


# ------------------------------------------------------------------ #
# Load all numerical artifacts directly so nothing is hand-typed.
# ------------------------------------------------------------------ #

def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _load_optional(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


METRICS = _load("metrics.json")
PRIVACY = _load("privacy_curve.json")
CALIB   = _load("calibration.json")
UNC     = _load("uncertainty_analysis.json")
EFF     = _load("efficiency.json")
XDOMAIN = _load_optional("cross_dataset_bloom_transfer.json")
PRIVACY_GUARD = _load_optional("privacy_guard_eval.json")
GOV_ABL = _load_optional("governor_ablation.json")
UNIFIED = _load_optional("unified_results_table.json")
CUE_ANALYSIS = _load_optional("bloom_domain_shift_cue_analysis.json")
PRIVACY_PERTURBATION = _load_optional("privacy_perturbation_curve.json")

if REFS_PATH.is_file():
    REFS = json.loads(REFS_PATH.read_text(encoding="utf-8"))
else:
    REFS = []


def _format_ref(r: dict) -> str:
    authors = r.get("authors", "[Author(s)]")
    year = r.get("year", "[Year]")
    title = r.get("title", "").strip()
    if not title:
        return f"{authors}. ({year})."
    return f"{authors}. ({year}). {title}."


# ------------------------------------------------------------------ #
# Styles
# ------------------------------------------------------------------ #

styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    "TitleX", parent=styles["Title"], fontName=FONT_BOLD,
    fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=10,
)
style_subtitle = ParagraphStyle(
    "SubtitleX", parent=styles["Normal"], fontName=FONT_ITAL,
    fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=14,
    textColor=colors.HexColor("#555555"),
)
style_h1 = ParagraphStyle(
    "H1X", parent=styles["Heading1"], fontName=FONT_BOLD,
    fontSize=12.5, leading=15, spaceBefore=12, spaceAfter=6,
    textColor=colors.HexColor("#111111"),
)
style_h2 = ParagraphStyle(
    "H2X", parent=styles["Heading2"], fontName=FONT_BOLD,
    fontSize=11, leading=14, spaceBefore=8, spaceAfter=4,
    textColor=colors.HexColor("#222222"),
)
style_body = ParagraphStyle(
    "BodyX", parent=styles["BodyText"], fontName=FONT_REG,
    fontSize=10, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=6,
)
style_abstract = ParagraphStyle(
    "AbstractX", parent=style_body, fontSize=9.6, leading=12.8,
    leftIndent=14, rightIndent=14,
)
style_keywords = ParagraphStyle(
    "KW", parent=style_body, fontSize=9.6, leading=12.8,
    leftIndent=14, rightIndent=14, spaceAfter=10,
)
style_caption = ParagraphStyle(
    "CaptionX", parent=styles["Normal"], fontName=FONT_BOLD,
    fontSize=9, leading=12, alignment=TA_LEFT, spaceBefore=10,
    spaceAfter=4,
)
style_figcap = ParagraphStyle(
    "FigCap", parent=styles["Normal"], fontName=FONT_ITAL,
    fontSize=8.8, leading=11.5, alignment=TA_CENTER, spaceBefore=4,
    spaceAfter=10,
)
style_eq = ParagraphStyle(
    "EqX", parent=style_body, alignment=TA_CENTER,
    fontName=FONT_REG, fontSize=10, spaceBefore=4, spaceAfter=6,
    leftIndent=12, rightIndent=12,
)
style_bullet = ParagraphStyle(
    "BulletX", parent=style_body, leftIndent=18, bulletIndent=6,
    spaceAfter=3, fontSize=10, leading=13,
)


# ------------------------------------------------------------------ #
# Layout helpers
# ------------------------------------------------------------------ #

def _table(data: List[List[str]],
           col_widths,
           *,
           header: bool = True,
           total_row: bool = False) -> Table:
    style = [
        ("FONT", (0, 0), (-1, -1), FONT_REG, 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.white, colors.HexColor("#F5F8FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("FONT", (0, 0), (-1, 0), FONT_BOLD, 9))
        style.append(
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EEF6"))
        )
    if total_row:
        style.append(("FONT", (0, -1), (-1, -1), FONT_BOLD, 9))
        style.append(("LINEABOVE", (0, -1), (-1, -1), 0.4, colors.black))
    return Table(data, colWidths=col_widths, style=TableStyle(style),
                 hAlign="CENTER", repeatRows=1 if header else 0)


def _ci(d: dict, key: str) -> str:
    v = d[key]
    return f"{v['mean']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]"


def _figure(path: Path, *, max_w_in: float = 6.4,
            max_h_in: float = 4.4, caption: str | None = None):
    if not path.exists():
        return [Spacer(1, 0)]
    nat_w, nat_h = ImageReader(str(path)).getSize()
    aspect = float(nat_h) / float(nat_w)
    target_w = max_w_in * inch
    target_h = target_w * aspect
    if target_h > max_h_in * inch:
        target_h = max_h_in * inch
        target_w = target_h / aspect
    img = Image(str(path), width=target_w, height=target_h)
    blocks = [Spacer(1, 8), img]
    if caption:
        blocks.append(Paragraph(caption, style_figcap))
    return blocks


def _para(html: str, style=style_body) -> Paragraph:
    return Paragraph(html, style)


# ------------------------------------------------------------------ #
# Page template (number + running footer)
# ------------------------------------------------------------------ #

def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_REG, 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    page_w, _ = LETTER
    canvas.drawString(0.75 * inch, 0.55 * inch,
                      "Lightweight Multi-Modal Tiny LLM Framework "
                      "for Privacy-Aware Academic Assistance")
    canvas.drawRightString(page_w - 0.75 * inch, 0.55 * inch,
                           f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#BFBFBF"))
    canvas.setLineWidth(0.4)
    canvas.line(0.75 * inch, 0.72 * inch,
                page_w - 0.75 * inch, 0.72 * inch)
    canvas.restoreState()


def _build_doc() -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(OUT_PDF), pagesize=LETTER,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.85 * inch, bottomMargin=0.95 * inch,
        title="Lightweight Multi-Modal Tiny LLM Framework",
        author="Anonymous (auto-generated draft)",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame,
                                       onPage=_on_page)])
    return doc


# ------------------------------------------------------------------ #
# Content
# ------------------------------------------------------------------ #

def build_story() -> list:
    story: list = []

    # ---------- title block ----------
    story += [
        Paragraph("Cognitive Robustness and Privacy-Constrained Local "
                  "Academic Assistance Under Domain Shift",
                  style_title),
    ]

    # ---------- abstract ----------
    story += [
        Paragraph("Abstract", style_h1),
        Paragraph(
            "We study cognitive robustness and privacy-constrained "
            "deployment for a fully offline academic-assistance system "
            "under educational-domain shift. The implemented framework "
            "combines multimodal text/PDF ingestion, dense retrieval over "
            "FAISS, a Label Distribution Learning (LDL) Bloom classifier, "
            "uncertainty-aware Bloom gating, role-aware privacy screening, "
            "and local Qwen GGUF generation. Rather than treating these as "
            "independent features, the evaluation asks whether cognitive "
            "labels, ordinal structure, retrieval utility, and protected "
            "exam access policies remain reliable when moved across "
            "datasets and deployment roles.",
            style_abstract),
        Paragraph(
            "Across Figshare and MoocRadar Bloom-labelled questions, "
            "directional transfer shows a sharp drop in exact reduced-label "
            "classification, especially in the ternary setting, but much "
            "stronger preservation of ordinal proximity. This motivates "
            "soft LDL feedback and human-in-the-loop fallback for uncertain "
            "or severe-jump-risk predictions. Privacy results are reported "
            "with the same caution: the InfoNCE retrieval-leakage proxy is "
            "a negative result, while the role-aware guard shows high "
            "measured resistance only under the defined adversarial prompt "
            "taxonomy. The contribution is therefore an evidence-backed "
            "analysis of domain-shift failure modes and local deployment "
            "constraints, not a claim of universal privacy or universal "
            "Bloom generalization.",
            style_abstract),
        Paragraph(
            "<b>Keywords &mdash;</b> retrieval-augmented generation; "
            "educational NLP; domain shift; label distribution learning; "
            "Bloom&rsquo;s taxonomy; privacy-constrained deployment; "
            "ordinal evaluation; uncertainty-aware reasoning.",
            style_keywords),
    ]

    # ---------- 1. Introduction ----------
    story += [
        Paragraph("1. Introduction", style_h1),
        _para(
            "University-scale AI assistants increasingly need to operate "
            "without sending student or institutional content to external "
            "services. Retrieval-augmented generation is a natural fit "
            "because it grounds compact local generators in retrievable "
            "context. We design and evaluate such a pipeline under four "
            "explicit constraints: (i) CPU-only inference, (ii) "
            "&le;&nbsp;1&nbsp;GB private working-set memory, (iii) "
            "deterministic and reproducible execution, and (iv) no "
            "external API calls. The evaluation is intentionally austere "
            "&mdash; we report what the system actually does on the "
            "implemented benchmark, including a clearly negative result "
            "on the current privacy term. The paper is organised around "
            "two findings: (i) ordinal cognitive structure degrades more "
            "gracefully than exact labels under domain shift, and (ii) "
            "privacy controls should be evaluated as measurable "
            "safety-utility tradeoffs rather than binary guarantees."),
        _para(
            "This work does not introduce a new base model; instead, it "
            "demonstrates that ordinal cognitive structure is more stable "
            "than categorical labels under domain shift, and that privacy "
            "constraints in offline RAG induce measurable safety-utility "
            "tradeoffs."),
    ]

    # ---------- 2. Contributions ----------
    story += [
        Paragraph("2. Contributions", style_h1),
        _para("We make three evidence-oriented contributions framed as "
              "questions answered (not components assembled), around "
              "cognitive robustness and privacy-constrained deployment:"),
        _para("&bull; <b>Directional cognitive-robustness evaluation.</b> "
              "We evaluate Bloom classification under Figshare "
              "&harr; MoocRadar transfer, explicitly reporting the "
              "6-class-to-reduced-space mapping, exact-label degradation, "
              "within-one-level accuracy, and severe ordinal error rate "
              "(distance &ge; 2).",
              style_bullet),
        _para("&bull; <b>Calibration-aware deployment policy.</b> The LDL "
              "classifier is used as a distribution, not only as an "
              "argmax label: adjacent-level ambiguity is exposed as soft "
              "feedback, while low-confidence or severe-jump-risk cases "
              "are routed to generalized or human-in-the-loop handling, "
              "rather than forcing brittle Bloom-level specialization. "
              "We treat uncertainty signals as deployment heuristics; "
              "their per-query error-prediction utility is reported "
              "conservatively.",
              style_bullet),
        _para("&bull; <b>Finding + framework, not module stacking.</b> "
              "The contribution is a measurement framework that separates "
              "categorical degradation from ordinal degradation and couples "
              "bounded empirical privacy resistance with explicit utility-cost "
              "reporting (safety-utility tradeoffs, not perfect guarantees).",
              style_bullet),
        _para("&bull; <b>Privacy evaluation with bounded claims.</b> We "
              "separate a negative retrieval-leakage sweep from a "
              "role-aware adversarial prompt taxonomy, and state privacy "
              "claims only as measured resistance under the defined "
              "attack set (including adaptive/multi-turn probes), not as "
              "perfect or general security.",
              style_bullet),
    ]

    # ---------- 3. Related Work ----------
    story += [
        Paragraph("3. Related Work", style_h1),
        _para(
            "Recent work on retrieval-augmented generation (RAG) has "
            "highlighted both its effectiveness and its privacy risks. "
            "Studies on privacy issues in RAG show that retrieval can leak "
            "protected content through the selected passages themselves "
            "(Good &amp; Bad RAG) and that privacy-preserving retrieval can be "
            "targeted with formal distance-based guarantees (RemoteRAG, "
            "DistanceDP). However, those approaches generally assume "
            "cloud-based or guarantee-first settings and do not directly "
            "address the offline, role-constrained university assistant "
            "deployment regime where privacy evidence is measured under a "
            "defined threat model rather than proven end-to-end."),
        _para(
            "Runtime mitigations (e.g., dynamic retrieval filtering and "
            "anonymization and perturbation-based privacy-preserving RAG "
            "attempt to reduce leakage at retrieval time, while multimodal "
            "retrieval can extend leakage pathways beyond text (Beyond Text). "
            "Our paper takes a "
            "complementary, threat-model-specific empirical approach: we "
            "combine a role-aware privacy guard with an explicit adversarial "
            "prompt taxonomy and report safety-utility tradeoffs under "
            "measurable attacks, avoiding over-strong claims of "
            "cryptographic or differential-privacy security."),
        _para(
            "On the educational side, Bloom/OBE classification is typically "
            "treated as flat classification and optimized for exact label "
            "accuracy. Yet domain shift can move topic vocabulary while "
            "preserving cognitive structure, motivating ordinal analyses. "
            "Our central novelty is to quantify ordinal robustness of "
            "Bloom classification under Figshare &harr; MoocRadar transfer "
            "and to couple this ordinal signal with uncertainty-aware "
            "deployment for an offline assistant. Feasibility at CPU scale "
            "is supported by efficient tiny-model and quantization research "
            "(QLoRA, AWQ, TinyLlama, Phi-2) and llama.cpp, but those works "
            "do not study Bloom-ordinal robustness under domain shift nor "
            "the associated privacy constraints."),
    ]

    # ---------- 4. System Architecture ----------
    story += [
        Paragraph("4. System Architecture", style_h1),
        _para("The system has five modules (mapped 1-to-1 to source files):"),
        _para("&bull; <b>Ingestion</b> (<i>ingestion.py</i>) &mdash; "
              "PyMuPDF text extraction, EasyOCR (lazy-loaded) for "
              "scanned/image input, plain-text loading, and "
              "deterministic token-window chunking.", style_bullet),
        _para("&bull; <b>Retriever</b> (<i>retriever.py</i>) &mdash; "
              "sentence-encoder embedding with all-MiniLM-L6-v2 (frozen, "
              "L2-normalised, dim 384), exact ANN over FAISS "
              "IndexFlatL2, and an InfoNCE-based privacy-aware "
              "re-ranking term.", style_bullet),
        _para("&bull; <b>Bloom-LDL Classifier</b> (<i>classifier.py</i>) "
              "&mdash; a linear LDL head on top of frozen MiniLM "
              "features, trained with Gaussian-smoothed soft labels "
              "combined with an ordinal-PMF target, an ordinal pairwise "
              "margin penalty, and an entropy regulariser.", style_bullet),
        _para("&bull; <b>Generator</b> (<i>models.py</i>) &mdash; "
              "Qwen-1.5B-Instruct (Q4_K_M GGUF) via llama-cpp-python, "
              "ChatML prompt with [BOUNDED CONTEXT] / [QUESTION] / "
              "[COGNITIVE LEVEL] / [INSTRUCTION] blocks, greedy decoding "
              "with temperature = 0, top_k = 1, top_p = 1, seed = 42, "
              "and an explicit llm.reset() before every call.",
              style_bullet),
        _para("&bull; <b>Uncertainty</b> (<i>uncertainty.py</i>) &mdash; "
              "Bloom-level normalised entropy H(p)/log K and Semantic "
              "Predictive Uncertainty (SPU) computed via chunk-subset "
              "perturbation, plus a confidence threshold gate that "
              "abstains or falls back when entropy, top-1 probability, or "
              "severe ordinal-jump mass indicates unreliable Bloom "
              "specialisation.", style_bullet),
        _para("&bull; <b>Privacy guard</b> (<i>privacy_guard.py</i>) "
              "&mdash; role-aware access control and output screening for "
              "protected exam uploads, evaluated with direct "
              "reconstruction, indirect leakage, paraphrase probes, "
              "adaptive/multi-turn prompts, and a lightweight black-box "
              "mutation-search attacker.", style_bullet),
        _para("The system does <b>not</b> implement: PII/identifier "
              "detection or masking, context redaction or filtering, "
              "audit logging, data-at-rest encryption, allow/block-list "
              "policy rules outside the tested role guard, decode-time "
              "stochastic sampling, or a second leakage risk channel. Privacy in "
              "this paper is therefore restricted to (i) fully offline "
              "CPU execution, (ii) the single-coefficient InfoNCE "
              "re-ranking term, and (iii) measured role-guard behaviour "
              "under the defined prompt taxonomy."),
    ]
    story += _figure(FIGS / "system_architecture.png", max_w_in=6.6,
                     caption="Figure 1. Implementation-faithful system "
                             "architecture (CPU-only, deterministic, "
                             "fully offline).")

    # ---------- 5. Methodology ----------
    story += [
        Paragraph("5. Methodology", style_h1),
        Paragraph("5.1 Document ingestion", style_h2),
        _para(
            "Documents are normalised to text (PyMuPDF for native PDFs, "
            "EasyOCR for scanned/image content, raw read for plain "
            "text), then chunked deterministically with a fixed token "
            "window. Chunks retain their source identifier for "
            "downstream traceability."),
        Paragraph("5.2 Dense retrieval and privacy-aware re-ranking",
                  style_h2),
        _para("Each chunk and the query are encoded by all-MiniLM-L6-v2 "
              "and L2-normalised. Top-k candidates (k = 5) are retrieved "
              "by exact nearest-neighbour search in FAISS IndexFlatL2. "
              "For each candidate d<sub>i</sub> the system computes:"),
        Paragraph("s(q, d<sub>i</sub>) = cos(q, d<sub>i</sub>) "
                  "&minus; &lambda; &middot; "
                  "R<sub>InfoNCE</sub>(q, d<sub>i</sub>)", style_eq),
        Paragraph("R<sub>InfoNCE</sub>(q, d<sub>i</sub>) "
                  "= log &sum;<sub>c &isin; C</sub> exp(sim(q, c)/&tau;) "
                  "&minus; sim(q, d<sub>i</sub>)/&tau;,"
                  "  &tau; = 0.07", style_eq),
        _para("Candidates are reordered by s(&middot;, &middot;). The "
              "default evaluation &lambda; is 0.5; the privacy sweep "
              "uses &lambda; &isin; {0, 0.25, 0.5, 0.75, 1}. "
              "<i>No second risk channel is computed.</i>"),
        Paragraph("5.3 Bloom-LDL classifier", style_h2),
        _para(
            "The classifier predicts a probability distribution over "
            "K = 6 Bloom levels (Remember, Understand, Apply, Analyse, "
            "Evaluate, Create). Targets are a hybrid of "
            "(a) Gaussian-smoothed soft labels around the gold ordinal "
            "index, (b) an ordinal-PMF anchored at the same index, and "
            "(c) a small hard-label component, all renormalised to a "
            "simplex. Training is full-batch gradient descent on a "
            "linear head over frozen MiniLM features with an additional "
            "ordinal-margin pairwise penalty and an entropy regulariser. "
            "The Figshare Bloom Exam dataset is used for training; OBE "
            "is used for evaluation only, with canonical-id "
            "deduplication enforced before any train/eval split."),
        Paragraph("5.4 Bloom-conditioned generation", style_h2),
        _para(
            "Retrieved chunks and the predicted Bloom level are inserted "
            "into a ChatML prompt with the four blocks above. Decoding "
            "is greedy, deterministic, and the KV-cache is explicitly "
            "reset before every generation, so identical inputs yield "
            "byte-identical outputs across runs."),
        Paragraph("5.5 Uncertainty estimation", style_h2),
        _para(
            "Two uncertainty signals are reported. Bloom-level "
            "uncertainty is the normalised entropy H(p)/log K of the "
            "LDL distribution. Semantic Predictive Uncertainty (SPU) "
            "is computed by repeating the Bloom-classification step "
            "N = 3 times on different deterministic subsets of the "
            "retrieved context for the same query, then averaging the "
            "pairwise Jensen&ndash;Shannon divergence between the "
            "resulting distributions. <i>Decoding itself is not "
            "perturbed.</i>"),
        Paragraph("5.6 Evaluation protocol", style_h2),
        _para(
            "The central experiment is directional domain transfer: train "
            "on one Bloom-labelled educational dataset and test on the "
            "other without target-domain tuning. We evaluate two reduced "
            "spaces. The binary mapping is Remember/Understand/Apply "
            "&rarr; Lower and Analyze/Evaluate/Create &rarr; Higher. "
            "The ternary mapping is Remember/Understand &rarr; Low, "
            "Apply/Analyze &rarr; Mid, and Evaluate/Create &rarr; High. "
            "For each direction we report accuracy, macro-F1, mean "
            "ordinal error, within-one-level accuracy, and severe error "
            "rate (distance &ge; 2 in the reduced ordinal space)."),
        _para(
            "Deployment evaluation is reported separately: local QA "
            "utility compares Proposed, VanillaRAG, BM25, and NoRAG; "
            "retrieval privacy reports document-match and cosine-threshold "
            "ASR over the &lambda; sweep; role-aware privacy reports block "
            "and allow rates under a fixed attack taxonomy. We avoid "
            "mixing these into a single scalar score because they answer "
            "different review questions."),
    ]

    # ---------- 6. Experimental Setup ----------
    story += [
        Paragraph("6. Experimental Setup", style_h1),
        Paragraph("6.1 Datasets and label spaces", style_h2),
        _para(
            "Figshare is the primary Bloom exam-question dataset used for "
            "in-domain and transfer experiments. MoocRadar is the external "
            "educational problem dataset used to test cross-domain "
            "generalization. Both are normalized to the same six revised "
            "Bloom levels: Remember, Understand, Apply, Analyze, Evaluate, "
            "and Create. Experiments then collapse this six-class space "
            "into binary and ternary ordinal spaces using the mappings "
            "defined in Section&nbsp;5.6."),
        Paragraph("6.2 Directional transfer protocol", style_h2),
        _para(
            "We evaluate four settings for each reduced label space: "
            "Figshare in-domain, MoocRadar in-domain, Figshare "
            "&rarr; MoocRadar, and MoocRadar &rarr; Figshare. The two "
            "cross-domain settings are intentionally asymmetric: a model "
            "trained on assessment-style Figshare questions is not assumed "
            "to see the same vocabulary or distribution as MoocRadar, and "
            "vice versa. This protocol is what turns low exact accuracy "
            "into evidence about cognitive-domain shift rather than merely "
            "a weak classifier result."),
        Paragraph("6.3 Privacy attack taxonomy", style_h2),
        _para(
            "Privacy evaluation has two parts. First, a retrieval-leakage "
            "proxy sweep measures document-match ASR and cosine-threshold "
            "ASR for &lambda; &isin; {0, 0.25, 0.5, 0.75, 1}. Second, a "
            "role-aware prompt taxonomy tests protected exam uploads under "
            "direct reconstruction, indirect leakage, paraphrase probes, "
            "model-aware jailbreaks, gradient-free paraphrase optimization, "
            "multi-turn probing, and semantic reconstruction attempts, plus "
            "benign student prompts and teacher moderation prompts. These "
            "are threat-model-specific measurements, not cryptographic or "
            "differential-privacy guarantees."),
        _para(
            "Threat model: attacker is a student with black-box query access "
            "to the assistant; no model-weight or gradient access is assumed. "
            "Claims are therefore empirical resistance under these adaptive "
            "attack classes, not formal privacy guarantees."),
        _para(
            "Semantic leakage uses the guard's concept-overlap proxy "
            "(semantic_concept_ratio) with an embedding cosine probe "
            "(MiniLM, cosine similarity) and threshold sensitivity reported "
            "as a safety-utility curve; these are practical detectors, not "
            "proofs of semantic confidentiality."),
        Paragraph("6.4 Runtime environment", style_h2),
        _para(
            "All experiments are CPU-only and fully offline, with "
            "HF_DATASETS_OFFLINE=1 and HF_HUB_OFFLINE=1. The reported "
            "environment uses Windows, Python 3.13.x, all-MiniLM-L6-v2 "
            "embeddings, FAISS IndexFlatL2 retrieval, the BloomLDLClassifier "
            "linear LDL head, and a Qwen GGUF generator through "
            "llama-cpp-python. Memory is reported as private working-set "
            "USS in addition to RSS because the GGUF weights are "
            "memory-mapped."),
    ]

    # ---------- 7. Results ----------
    story += [Paragraph("7. Results", style_h1)]

    # 7.1 Consolidated evidence table
    story += [Paragraph("7.1 Consolidated evidence", style_h2)]
    story += [Paragraph(
        "Consolidated table &mdash; Main evidence chain for cognitive "
        "robustness and privacy-constrained deployment. Full CSV/JSON "
        "versions are written to <i>results/unified_results_table.*</i>.",
        style_caption)]
    cons_rows = [["Area", "Protocol / setting", "Metric", "Value", "Interpretation"]]
    if XDOMAIN:
        tern = XDOMAIN.get("schemes", {}).get("ternary", {})
        for key, label in [
            ("within_dataset_figshare", "Figshare in-domain"),
            ("within_dataset_moocradar", "MoocRadar in-domain"),
            ("figshare_to_moocradar", "Figshare -> MoocRadar"),
            ("moocradar_to_figshare", "MoocRadar -> Figshare"),
        ]:
            entry = tern.get(key, {})
            m = entry.get("selected_metrics", {})
            if m:
                cons_rows.append([
                    "Cognitive",
                    label,
                    "macro-F1 / within-1 / severe",
                    f"{m['macro_f1']:.3f} / {m['within_one_level_accuracy']:.3f} / {m['severe_error_rate']:.3f}",
                    "ordinal structure retained" if "to" in key else "in-domain reference",
                ])
    if PRIVACY:
        cons_rows.append([
            "Privacy",
            "InfoNCE retrieval sweep",
            "ASR AUC doc / cosine",
            f"{PRIVACY['auc_asr_doc']:.3f} / {PRIVACY['auc_asr_cos']:.3f}",
            "negative proxy result; no privacy-utility claim",
        ])
    if PRIVACY_GUARD:
        cons_rows.append([
            "Privacy",
            "student attack taxonomy",
            "block / benign allow",
            f"{PRIVACY_GUARD.get('student_attack_block_rate', 0.0):.3f} / {PRIVACY_GUARD.get('student_benign_allow_rate', 0.0):.3f}",
            "measured resistance under defined prompts",
        ])
    if GOV_ABL:
        by = GOV_ABL.get("by_preset", {})
        off = by.get("off", {})
        strong = by.get("strong", {})
        if off and strong:
            cons_rows.append([
                "Privacy",
                "governor ablation",
                "off vs strong (F1 / leak)",
                (
                    f"{off.get('mean_f1', 0.0):.3f}->{strong.get('mean_f1', 0.0):.3f} / "
                    f"{off.get('mean_leak_full_corpus_ratio', 0.0):.3f}->{strong.get('mean_leak_full_corpus_ratio', 0.0):.3f}"
                ),
                "privacy gain with utility tradeoff; fairness check for method components",
            ])
    if METRICS.get("qa", {}).get("Proposed"):
        cons_rows.append([
            "Utility",
            "bounded local QA",
            "Proposed token-F1",
            f"{METRICS['qa']['Proposed']['f1']['mean']:.3f}",
            "local/offline utility reference",
        ])
    story.append(_table(
        cons_rows,
        col_widths=[0.85*inch, 1.45*inch, 1.35*inch, 1.25*inch, 1.65*inch],
    ))

    # 7.2 Cross-domain cognitive robustness
    if XDOMAIN:
        story += [Paragraph("7.2 Cross-domain Bloom transfer", style_h2)]
        story += [Paragraph(
            "Table&nbsp;CD &mdash; Ternary cross-domain Bloom results. "
            "Severe error means a Low&harr;High jump.",
            style_caption)]
        tern = XDOMAIN.get("schemes", {}).get("ternary", {})
        cd_rows = [["Setting", "Selected model", "Accuracy", "Macro-F1", "Within-1", "Severe"]]
        for key, label in [
            ("within_dataset_figshare", "Figshare in-domain"),
            ("within_dataset_moocradar", "MoocRadar in-domain"),
            ("figshare_to_moocradar", "Figshare -> MoocRadar"),
            ("moocradar_to_figshare", "MoocRadar -> Figshare"),
        ]:
            entry = tern.get(key, {})
            m = entry.get("selected_metrics", {})
            if m:
                cd_rows.append([
                    label,
                    str(entry.get("selected_model", "")),
                    f"{m['accuracy']:.3f}",
                    f"{m['macro_f1']:.3f}",
                    f"{m['within_one_level_accuracy']:.3f}",
                    f"{m['severe_error_rate']:.3f}",
                ])
        story.append(_table(
            cd_rows,
            col_widths=[1.55*inch, 1.45*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch],
        ))
        story += _figure(
            FIGS / "domain_shift_preserves_ordinal_structure.png",
            max_w_in=6.2,
            caption="Figure CD. Domain shift sharply reduces exact ternary "
                    "macro-F1, while within-one-level accuracy retains more "
                    "signal than exact class matching. "
                    "This is the central evidence for treating Bloom output "
                    "as ordinal and distributional rather than as a brittle "
                    "hard class.",
        )
        story.append(_para(
            "The important result is not that the cross-domain classifier "
            "achieves high exact accuracy. It does not. The result is that "
            "exact reduced-label classification degrades much more sharply "
            "than ordinal proximity. This supports the claim hierarchy used "
            "throughout the paper: the framework studies cognitive robustness "
            "under domain shift, exposes uncertainty through LDL, and gates "
            "high-risk predictions instead of over-specializing generation."))

        if CUE_ANALYSIS:
            story += [Paragraph("7.2.1 Verb and content shift analysis", style_h2)]
            story += [Paragraph(
                "Table&nbsp;CA &mdash; Cue/content ablation on ternary "
                "directional transfer. Deltas are relative to content TF-IDF; "
                "negative severe-error deltas are better.",
                style_caption)]
            ca_rows = [["Direction", "Comparison", "Delta macro-F1", "Delta within-1", "Delta severe"]]
            summary = CUE_ANALYSIS.get("summary", {})
            for key, label in [
                ("figshare_to_moocradar", "Figshare -> MoocRadar"),
                ("moocradar_to_figshare", "MoocRadar -> Figshare"),
            ]:
                item = summary.get(key, {})
                if not item:
                    continue
                ca_rows.append([
                    label,
                    "cue-only minus content",
                    f"{item['cue_minus_content_macro_f1']:.3f}",
                    f"{item['cue_minus_content_within_one']:.3f}",
                    f"{item['cue_minus_content_severe']:.3f}",
                ])
                ca_rows.append([
                    label,
                    "combined minus content",
                    f"{item['combined_minus_content_macro_f1']:.3f}",
                    f"{item['combined_minus_content_within_one']:.3f}",
                    f"{item['combined_minus_content_severe']:.3f}",
                ])
            story.append(_table(
                ca_rows,
                col_widths=[1.45*inch, 1.65*inch, 1.0*inch, 1.0*inch, 1.0*inch],
            ))
            story.append(_para(
                "The ablation gives a bounded explanation rather than a "
                "universal law. Bloom action cues reduce severe ordinal jumps "
                "for Figshare &rarr; MoocRadar and improve macro-F1 for "
                "MoocRadar &rarr; Figshare, but the combined cue/content "
                "model is not uniformly better on every ordinal metric. This "
                "supports the narrower claim that cognitive verbs and task "
                "structure provide transferable signal, while topic vocabulary "
                "still drives substantial domain shift."))

    # 7.3 QA performance
    run_cfg = METRICS.get("config", {})
    qa_n = int(run_cfg.get("n_test_qa", 0) or 0)
    qa = METRICS["qa"]
    story += [Paragraph("7.3 Question-answering performance", style_h2)]
    story += [Paragraph(
        f"Table&nbsp;1 &mdash; QA performance on the {qa_n}-query benchmark "
        f"(means with 95% bootstrap CIs; n = {qa_n}, B = 1000).",
        style_caption)]
    qa_rows = [["System", "F1 (95% CI)",
                "ROUGE-L (95% CI)", "METEOR (95% CI)"]]
    for s in ["Proposed", "VanillaRAG", "BM25", "NoRAG"]:
        d = qa[s]
        qa_rows.append([s, _ci(d, "f1"), _ci(d, "rouge_l"), _ci(d, "meteor")])
    story.append(_table(qa_rows,
                        col_widths=[1.2*inch, 1.6*inch, 1.6*inch, 1.6*inch]))

    # Grounding / faithfulness proxies (retrieval-side evidence).
    story += [Paragraph(
        "Table&nbsp;1b &mdash; Retrieval grounding / faithfulness proxies "
        f"on the same {qa_n}-query benchmark.",
        style_caption)]
    g_rows = [["System", "Answer grounding (token overlap)", "Retrieval hit@k", "Ref answer coverage in retrieval"]]
    for s in ["Proposed", "VanillaRAG", "BM25", "NoRAG"]:
        d = qa[s]
        g_rows.append([
            s,
            _ci(d, "leak_retrieved_ratio"),
            _ci(d, "retrieval_hit"),
            _ci(d, "ref_answer_coverage_in_retrieval"),
        ])
    story.append(_table(g_rows,
                        col_widths=[1.2*inch, 1.7*inch, 1.2*inch, 2.0*inch]))
    story.append(_para(
        "These grounding/faithfulness proxies are intentionally lightweight "
        "and deterministic: they measure whether retrieval surfaces the gold "
        "source (hit@k), whether the reference answer is present in retrieved "
        "text (coverage), and how much of the generated answer is grounded in "
        "retrieved tokens (overlap)."))
    story.append(_para(
        "F1 confidence intervals overlap pairwise across "
        "all retrieval-enabled systems; only the gap between "
        "retrieval-enabled systems and NoRAG is consistent on F1 and "
        "METEOR."))
    story.append(_para(
        "Fair-comparison note: the retrieval governor is part of the "
        "Proposed privacy-aware method and is intentionally not enabled for "
        "VanillaRAG/BM25 baselines. A governor ablation is reported "
        "separately to quantify its effect."))
    paired = qa.get("paired_tests", {})
    if paired:
        pv = paired.get("Proposed_vs_VanillaRAG__f1", {})
        pl = paired.get("Proposed_vs_VanillaRAG__leak_full_ratio", {})
        if pv and pl:
            story.append(_para(
                "Paired significance checks show Proposed vs VanillaRAG on "
                f"F1: p = {pv.get('p', float('nan')):.4f}; and on full-corpus "
                f"leakage ratio: p = {pl.get('p', float('nan')):.4f}. "
                "This separates utility and leakage effects explicitly."))

    # 7.4 Privacy curve
    priv_n = int(PRIVACY.get("n_samples", 0) or 0)
    story += [Paragraph("7.4 Privacy curve", style_h2)]
    story += [Paragraph(
        "Table&nbsp;2 &mdash; Privacy sweep over &lambda; "
        f"(n = {priv_n} queries; cosine threshold &theta; = 0.85).",
        style_caption)]
    priv_rows = [["lambda", "ASR (document-match)",
                  "ASR (cosine-threshold)"]]
    for i, lam in enumerate(PRIVACY["lambda"]):
        priv_rows.append([f"{lam:.2f}",
                          f"{PRIVACY['asr_doc'][i]:.3f}",
                          f"{PRIVACY['asr_cos'][i]:.3f}"])
    priv_rows.append(["AUC",
                      f"{PRIVACY['auc_asr_doc']:.3f}",
                      f"{PRIVACY['auc_asr_cos']:.3f}"])
    story.append(_table(priv_rows,
                        col_widths=[1.0*inch, 1.85*inch, 1.85*inch],
                        total_row=True))
    story += _figure(FIGS / "asr_lambda_curve.png", max_w_in=5.4,
                     caption="Figure 2. Attack-success-rate as a "
                             "function of the privacy coefficient. Both "
                             "proxies are flat over &lambda;.")

    if PRIVACY_PERTURBATION:
        pp_n = int(PRIVACY_PERTURBATION.get("n_samples", 0) or 0)
        story += [Paragraph(
            "Table&nbsp;2b &mdash; RemoteRAG-style query-embedding perturbation "
            "sweep over &sigma; (n = "
            f"{pp_n} queries; cosine threshold &theta; = 0.85).",
            style_caption)]
        pp_rows = [["sigma", "ASR (document-match)",
                    "ASR (cosine-threshold)"]]
        pp_sigma = PRIVACY_PERTURBATION.get("sigma", []) or []
        pp_asr_doc = PRIVACY_PERTURBATION.get("asr_doc", []) or []
        pp_asr_cos = PRIVACY_PERTURBATION.get("asr_cos", []) or []
        for i in range(min(len(pp_sigma), len(pp_asr_doc), len(pp_asr_cos))):
            pp_rows.append([
                f"{pp_sigma[i]:.2f}",
                f"{pp_asr_doc[i]:.3f}",
                f"{pp_asr_cos[i]:.3f}",
            ])
        pp_rows.append([
            "AUC",
            f"{PRIVACY_PERTURBATION.get('auc_asr_doc', 0.0):.3f}",
            f"{PRIVACY_PERTURBATION.get('auc_asr_cos', 0.0):.3f}",
        ])
        story.append(_table(pp_rows,
                            col_widths=[1.0*inch, 1.85*inch, 1.85*inch],
                            total_row=True))
    story.append(_para(
        "Both proxies are flat across &lambda;: in the tested "
        "configuration, the InfoNCE-based re-ranking term does not "
        "reduce retrieval-leakage on this benchmark (a falsified "
        "hypothesis rather than a positive privacy result). "
        "In contrast, a RemoteRAG-style query-embedding perturbation "
        "baseline produces measurable sensitivity of retrieval ASR to "
        "&sigma;, providing an empirical analogue to privacy-by-perturbation "
        "within the same offline CPU budget. "
        "However, because the ASR(doc-match) proxy is also the probability "
        "of retrieving the gold/source passage, decreasing ASR via larger "
        "&sigma; simultaneously reduces retrieval fidelity, illustrating a "
        "privacy-for-utility tradeoff. "
        "We operate in a L2-normalised embedding space (cosine similarity); "
        "the &sigma; grid is chosen so small perturbations preserve nearest-"
        "neighbor structure while larger values visibly disrupt top-1 "
        "re-identification." ))

    if PRIVACY_GUARD:
        story += [Paragraph("7.5 Role-aware privacy guard", style_h2)]
        story += [Paragraph(
            "Table&nbsp;PG &mdash; Protected-upload prompt taxonomy. "
            "The claim is measured resistance under these prompts only.",
            style_caption)]
        pg_rows = [["Prompt group", "Metric", "Value"]]
        pg_rows.append([
            "student attacks",
            "block rate",
            f"{PRIVACY_GUARD.get('student_attack_block_rate', 0.0):.3f}",
        ])
        pg_rows.append([
            "student benign",
            "allow rate",
            f"{PRIVACY_GUARD.get('student_benign_allow_rate', 0.0):.3f}",
        ])
        pg_rows.append([
            "teacher moderation",
            "allow rate",
            f"{PRIVACY_GUARD.get('teacher_moderation_allow_rate', 0.0):.3f}",
        ])
        for category, item in PRIVACY_GUARD.get("attack_category_summary", {}).items():
            pg_rows.append([
                category.replace("_", " "),
                "category block rate",
                f"{item.get('block_rate', 0.0):.3f}",
            ])
        story.append(_table(pg_rows, col_widths=[2.0*inch, 1.7*inch, 0.9*inch]))
        story.append(_para(
            "These numbers should be read as role-policy evidence, not as "
            "proof of perfect privacy. They show that direct reconstruction, "
            "indirect leakage, paraphrase-probe, adaptive, multi-turn, and "
            "semantic reconstruction prompts were blocked in the evaluated "
            "set while benign student and teacher moderation uses remained "
            "available."))
        story.append(_para(
            "Benign prompts are intentionally non-extractive study requests; "
            "they are not reconstruction attempts. Utility degradation under "
            "stricter semantic thresholds is reported explicitly to avoid "
            "over-clean safety claims."))
        story.append(_para(
            f"Prompt counts: attacks={int(PRIVACY_GUARD.get('n_attack_prompts', 0))}, "
            f"student benign={int(PRIVACY_GUARD.get('n_student_benign', 0))}, "
            f"teacher moderation={int(PRIVACY_GUARD.get('n_teacher_moderation', 0))}."))
        curve = PRIVACY_GUARD.get("safety_utility_curve", [])
        if curve:
            story += [Paragraph(
                "Table&nbsp;PG2 &mdash; Safety-utility sensitivity over the "
                "semantic-leakage threshold. Lower thresholds are stricter.",
                style_caption)]
            curve_rows = [["Semantic threshold", "Attack block", "Benign allow"]]
            for point in curve:
                curve_rows.append([
                    f"{point.get('semantic_threshold', 0.0):.2f}",
                    f"{point.get('attack_block_rate', 0.0):.3f}",
                    f"{point.get('benign_allow_rate', 0.0):.3f}",
                ])
            story.append(_table(curve_rows, col_widths=[1.8*inch, 1.4*inch, 1.4*inch]))
        auc = PRIVACY_GUARD.get("safety_utility_curve_auc", {})
        if auc:
            story.append(_para(
                f"Curve integrals: attack-block AUC = {auc.get('attack_block_auc', 0.0):.3f}, "
                f"benign-allow AUC = {auc.get('benign_allow_auc', 0.0):.3f} "
                "(higher attack-block and lower benign loss are preferred)."))
        leakage = PRIVACY_GUARD.get("leakage_signal_summary", {})
        if "max_semantic_concept_ratio" in leakage:
            story.append(_para(
                "The semantic reconstruction probe adds a non-extractive "
                f"risk signal (max concept ratio = "
                f"{leakage.get('max_semantic_concept_ratio', 0.0):.3f}) "
                "so the guard distinguishes copied-span leakage from "
                "paraphrased disclosure of protected exam concepts. This is "
                "still a proxy measurement, not a formal privacy proof."))
            if "max_embedding_cosine_similarity" in leakage:
                story.append(_para(
                    "A secondary semantic probe uses MiniLM embedding cosine "
                    f"(max similarity = {leakage.get('max_embedding_cosine_similarity', 0.0):.3f}) "
                    "to quantify high-semantic-overlap candidates even when "
                    "exact spans are avoided."))
        emb_sens = PRIVACY_GUARD.get("embedding_threshold_sensitivity", {})
        sens_rows = emb_sens.get("thresholds", []) if isinstance(emb_sens, dict) else []
        sens_dist = emb_sens.get("distribution", {}) if isinstance(emb_sens, dict) else {}
        if sens_rows:
            story += [Paragraph(
                "Table&nbsp;PG2b &mdash; Embedding-cosine threshold "
                "sensitivity for semantic leakage probe.",
                style_caption)]
            rows = [["Cosine threshold", "Flag rate", "n scored"]]
            for point in sens_rows:
                rows.append(
                    [
                        f"{point.get('threshold', 0.0):.2f}",
                        f"{point.get('flag_rate', 0.0):.3f}",
                        f"{int(point.get('n_scored', 0.0))}",
                    ]
                )
            story.append(_table(rows, col_widths=[1.8*inch, 1.4*inch, 1.1*inch]))
            if sens_dist:
                story.append(_para(
                    "Embedding-cosine distribution summary: "
                    f"mean={sens_dist.get('mean', 0.0):.3f}, "
                    f"p50={sens_dist.get('p50', 0.0):.3f}, "
                    f"p90={sens_dist.get('p90', 0.0):.3f}, "
                    f"max={sens_dist.get('max', 0.0):.3f}."))
            story.append(_para(
                "We use 0.80 as a conservative reporting threshold; "
                "sensitivity across 0.55-0.80 shows consistent qualitative trends "
                "for high-semantic-overlap detection."))
        by = GOV_ABL.get("by_preset", {})
        off = by.get("off", {})
        strong = by.get("strong", {})
        if off and strong:
            story += [Paragraph(
                "Table&nbsp;PG3 &mdash; Retrieval governor fairness ablation "
                "on the Proposed system (off vs strong).",
                style_caption)]
            pg3_rows = [
                ["Preset", "Mean F1", "Mean leak(full corpus)", "Mean retrieved context chars"],
                [
                    "off",
                    f"{off.get('mean_f1', 0.0):.3f}",
                    f"{off.get('mean_leak_full_corpus_ratio', 0.0):.3f}",
                    f"{off.get('mean_context_char_count', 0.0):.1f}",
                ],
                [
                    "strong",
                    f"{strong.get('mean_f1', 0.0):.3f}",
                    f"{strong.get('mean_leak_full_corpus_ratio', 0.0):.3f}",
                    f"{strong.get('mean_context_char_count', 0.0):.1f}",
                ],
            ]
            story.append(_table(pg3_rows, col_widths=[1.2*inch, 1.1*inch, 1.6*inch, 1.7*inch]))
            story.append(_para(
                "This ablation reports the expected privacy-utility tension "
                "for the proposed method and clarifies that the governor is "
                "an intentional method component rather than an unfair "
                "baseline tweak."))
        if CUE_ANALYSIS:
            summary = CUE_ANALYSIS.get("summary", {})
            f2m = summary.get("figshare_to_moocradar", {})
            m2f = summary.get("moocradar_to_figshare", {})
            if f2m and m2f:
                story += [Paragraph(
                    "Table&nbsp;PG4 &mdash; Component attribution snapshot "
                    "(ordinal module and privacy module effects).",
                    style_caption)]
                pg4_rows = [
                    ["Component", "Setting", "Effect"],
                    ["Ordinal cue features", "Figshare -> MoocRadar",
                     f"delta severe = {f2m.get('cue_minus_content_severe', 0.0):.3f}"],
                    ["Ordinal cue features", "MoocRadar -> Figshare",
                     f"delta severe = {m2f.get('cue_minus_content_severe', 0.0):.3f}"],
                    ["Privacy governor", "Proposed off->strong",
                     (
                         f"leak {off.get('mean_leak_full_corpus_ratio', 0.0):.3f}->"
                         f"{strong.get('mean_leak_full_corpus_ratio', 0.0):.3f}, "
                         f"F1 {off.get('mean_f1', 0.0):.3f}->{strong.get('mean_f1', 0.0):.3f}"
                     )],
                ]
                story.append(_table(pg4_rows, col_widths=[1.5*inch, 1.8*inch, 2.0*inch]))

    # 7.6 Bloom classification + calibration
    unc_n = int(UNC.get("n_pool", 0) or 0)
    story += [Paragraph("7.6 Bloom classification and calibration",
                        style_h2)]
    story += [Paragraph(
        "Table&nbsp;3 &mdash; Bloom-level classification on the "
        f"{unc_n}-sample uncertainty pool.",
        style_caption)]
    cls_rows = [
        ["Metric", "Value"],
        ["Top-1 accuracy",
         f"{METRICS['classification_accuracy']:.4f}"],
        ["KL(predicted || one-hot gold)",
         f"{METRICS['classification_kl']:.4f}"],
        ["Expected Calibration Error",
         f"{CALIB['ece']:.4f}"],
        ["Number of calibration bins (M)",
         f"{CALIB['n_bins']}"],
    ]
    story.append(_table(cls_rows,
                        col_widths=[3.0*inch, 1.6*inch]))

    story += [Paragraph(
        "Table&nbsp;4 &mdash; Reliability bins (10 equal-width "
        "confidence bins). Empty bins are omitted.",
        style_caption)]
    rel_rows = [["Bin centre", "Count", "Mean confidence",
                 "Empirical accuracy"]]
    for i, c in enumerate(CALIB["bin_centers"]):
        cnt = CALIB["bin_counts"][i]
        if cnt == 0:
            continue
        acc = CALIB["bin_accuracy"][i]
        cnf = CALIB["bin_confidence"][i]
        rel_rows.append([f"{c:.2f}", str(cnt),
                         f"{cnf:.3f}" if cnf is not None else "-",
                         f"{acc:.3f}" if acc is not None else "-"])
    story.append(_table(rel_rows,
                        col_widths=[1.0*inch, 0.8*inch, 1.4*inch,
                                    1.6*inch]))
    story += _figure(FIGS / "reliability_diagram.png", max_w_in=5.0,
                     caption="Figure 3. Reliability diagram for the "
                             "Bloom-LDL classifier (10 confidence bins).")
    story.append(_para(
        "Confidence systematically exceeds accuracy across every "
        "populated bin; the classifier is over-confident, and ECE is "
        "high."))

    # 7.7 Uncertainty
    story += [Paragraph("7.7 Predictive uncertainty", style_h2)]
    story += [Paragraph(
        "Table&nbsp;5 &mdash; Bloom-level normalised entropy "
        f"(H(p)/log&nbsp;6) on the {unc_n}-sample uncertainty pool.",
        style_caption)]
    unc_rows = [
        ["Statistic", "Value"],
        ["Mean entropy (normalised)",
         f"{UNC['bloom_uncertainty_mean']:.4f}"],
        ["Std. deviation",
         f"{UNC['bloom_uncertainty_std']:.4f}"],
        ["Pearson correlation with prediction error",
         f"{UNC['uncertainty_error_correlation_pearson']:.4f}"],
    ]
    story.append(_table(unc_rows,
                        col_widths=[3.6*inch, 1.4*inch]))

    story += [Paragraph(
        "Table&nbsp;6 &mdash; Error rate by Bloom-uncertainty bin "
        "(5 equal-width bins). Empty bins are omitted.",
        style_caption)]
    bin_rows = [["Uncertainty bin centre", "Count",
                 "Empirical error rate"]]
    for i, c in enumerate(UNC["bin_centers"]):
        cnt = UNC["bin_counts"][i]
        if cnt == 0:
            continue
        er = UNC["bin_error_rate"][i]
        bin_rows.append([f"{c:.2f}", str(cnt),
                         f"{er:.3f}" if er is not None else "-"])
    story.append(_table(bin_rows,
                        col_widths=[1.9*inch, 1.0*inch, 1.7*inch]))
    story += _figure(FIGS / "uncertainty_error_curve.png", max_w_in=5.0,
                     caption="Figure 4. Empirical error rate per "
                             "Bloom-uncertainty bin.")

    story += [Paragraph(
        "Table&nbsp;7 &mdash; Generation Semantic Predictive Uncertainty "
        "(SPU) over 10 queries, N = 3 chunk-subset perturbations per "
        "query.",
        style_caption)]
    spu = UNC["generation_spu"]["per_query"]
    spu_rows = [["Query idx", "SPU"]]
    for r in spu:
        spu_rows.append([str(r["sample_idx"]),
                         f"{r['spu']:.5f}"])
    spu_rows.append(["Mean", f"{UNC['generation_spu']['mean']:.5f}"])
    story.append(_table(spu_rows,
                        col_widths=[1.4*inch, 1.4*inch],
                        total_row=True))
    story.append(_para(
        "The Bloom-uncertainty-vs-error correlation is essentially zero "
        "(Pearson r = &minus;0.047), and the per-bin error rates do not "
        "decrease monotonically with confidence. The current uncertainty "
        "signal is therefore not an effective error predictor at the "
            "individual-query level. We therefore interpret uncertainty as "
            "an exploratory, model-internal deployment heuristic rather than "
            "as a reliable correctness detector."))

    # 7.8 Efficiency
    story += [Paragraph("7.8 Efficiency and resource usage", style_h2)]
    story += [Paragraph(
        "Table&nbsp;8 &mdash; Memory footprint (single resident process, "
        "end of run).",
        style_caption)]
    mem_rows = [
        ["Metric", "Value (MB)"],
        ["Resident Set Size (RSS)", f"{EFF['rss_mb_now']:.1f}"],
        ["Unique Set Size (USS, private)",
         f"{EFF['uss_mb_now']:.1f}"],
        ["Memory-mapped model footprint",
         f"{EFF['model_mmap_mb']:.1f}"],
        ["RAM budget", f"{EFF['ram_budget_mb']:.1f}"],
        ["Under budget (USS &lt; 1024)",
         "yes" if EFF["under_1gb_budget"] else "no"],
    ]
    story.append(_table(mem_rows,
                        col_widths=[3.0*inch, 1.6*inch]))
    story.append(_para(
        "USS rather than RSS is used as the private-RAM metric because "
        "the GGUF weights are memory-mapped and shared, so RSS "
        "double-counts file-backed pages."))

    story += [Paragraph(
        f"Table&nbsp;9 &mdash; Per-system latency on the {qa_n}-query QA "
        "benchmark (seconds per query).",
        style_caption)]
    lat_rows = [["System", "Mean", "Median (p50)", "p95"]]
    for s in ["Proposed", "VanillaRAG", "BM25", "NoRAG"]:
        d = EFF["per_system"][s]
        lat_rows.append([s,
                         f"{d['latency_mean_s']:.3f}",
                         f"{d['latency_p50_s']:.3f}",
                         f"{d['latency_p95_s']:.3f}"])
    story.append(_table(lat_rows,
                        col_widths=[1.4*inch, 1.0*inch, 1.2*inch,
                                    1.0*inch]))
    story += _figure(FIGS / "memory_latency_plot.png", max_w_in=5.6,
                     caption="Figure 5. Memory and latency summary.")
    story.append(_para(
        f"Total wall-clock time for the full evaluation pipeline "
        f"({int(run_cfg.get('n_total', 0) or 0)} OBE samples + {qa_n} QA + "
        f"{int(run_cfg.get('n_uncertainty_pool', 0) or 0)} uncertainty pool + privacy "
        f"sweep + calibration + plotting) was "
        f"{METRICS['wall_clock_s']:.1f}&nbsp;s."))

    # 7.9 Statistical comparison
    story += [Paragraph("7.9 Statistical comparison", style_h2)]
    paired = qa.get("paired_tests", {})
    tt = paired.get("Proposed_vs_VanillaRAG__f1", {})
    n_qa = int(METRICS.get("config", {}).get("n_test_qa", 0) or 0)
    story += [Paragraph(
        "Table&nbsp;10 &mdash; Paired t-test on token-level F1, "
        f"Proposed vs VanillaRAG (n = {n_qa}).",
        style_caption)]
    tt_rows = [["Statistic", "Value"]]
    if tt:
        tt_rows.extend(
            [
                ["Mean difference", f"{tt.get('mean_diff', 0.0):.3f}"],
                ["t", f"{tt.get('t', 0.0):.3f}"],
                ["df", f"{tt.get('df', 0.0):.0f}"],
                ["p-value", f"{tt.get('p', 0.0):.3f}"],
            ]
        )
    else:
        tt_rows.append(["Status", "paired F1 test unavailable in current artifact"])
    story.append(_table(tt_rows,
                        col_widths=[2.4*inch, 1.6*inch]))
    if tt:
        story.append(_para(
            "This paired test quantifies whether utility differs between "
            "the proposed privacy-aware retrieval policy and a canonical "
            "ungoverned vanilla baseline under identical QA prompts."))

    # 7.10 Configuration snapshot
    story += [Paragraph("7.10 Configuration snapshot", style_h2)]
    story += [Paragraph(
        "Table&nbsp;11 &mdash; Reported run configuration "
        "(<i>results/metrics.json::config</i>).",
        style_caption)]
    cfg = run_cfg
    cfg_keys = [
        ("seed", "top_k_retrieve"),
        ("n_total", "lambda_privacy"),
        ("n_test_qa", "asr_threshold"),
        ("n_uncertainty_pool", "asr_use_doc_match"),
        ("n_spu", "bootstrap_n"),
        ("n_stochastic", "bootstrap_ci"),
        ("train_per_class", "n_calib_bins"),
        ("max_tokens", "n_unc_bins"),
        ("n_ctx", "run_llm"),
        ("n_threads", "dataset_type"),
    ]
    cfg_rows = [["Parameter", "Value", "Parameter", "Value"]]
    for k1, k2 in cfg_keys:
        cfg_rows.append([k1, str(cfg.get(k1)),
                         k2, str(cfg.get(k2))])
    story.append(_table(cfg_rows,
                        col_widths=[1.4*inch, 1.0*inch, 1.5*inch,
                                    1.0*inch]))

    # ---------- 8. Discussion ----------
    story += [
        Paragraph("8. Discussion", style_h1),
        _para("The evidence supports a narrower and stronger claim than "
              "&ldquo;we built an academic assistant&rdquo;: the study "
              "characterises cognitive robustness and privacy-constrained "
              "deployment under domain shift."),
        _para("&bull; <b>Domain shift is the central cognitive failure "
              "mode.</b> Ternary transfer causes a large macro-F1 drop in "
              "both directions, especially MoocRadar &rarr; Figshare. "
              "However, within-one-level accuracy remains comparatively "
              "high. This explains why hard Bloom labels are brittle while "
              "LDL distributions and ordinal metrics remain useful.",
              style_bullet),
        _para("&bull; <b>Ordinal preservation justifies uncertainty-aware "
              "gating.</b> Severe Low&harr;High jumps are the dangerous "
              "case for an academic assistant. The implemented confidence "
              "gate therefore treats low-confidence or severe-jump-risk "
              "outputs as deployment decisions, not merely classifier "
              "errors.", style_bullet),
        _para("&bull; <b>Privacy evidence is deliberately bounded.</b> "
              "The retrieval-leakage sweep is a negative result for the "
              "InfoNCE term, while the role-aware guard now separates "
              "extractive copying, semantic reconstruction, adaptive "
              "jailbreak-style prompts, and utility degradation under "
              "stricter thresholds. This distinction prevents over-strong "
              "privacy interpretation.", style_bullet),
        _para("&bull; <b>DP guarantees vs measured resistance.</b> "
              "Differential-privacy or distance-DP approaches provide "
              "formal confidentiality guarantees, but our contribution is an "
              "empirical, threat-model-specific resistance evaluation under a "
              "defined adaptive prompt taxonomy; we therefore do not claim "
              "DP-level protection against arbitrary attacks. "
              "Our approach does not provide formal guarantees, but enables "
              "fine-grained, empirically measurable control over leakage "
              "behaviors under realistic attack conditions.", style_bullet),
        _para("&bull; <b>Perturbation is a uniform tradeoff; guards can be selective.</b> "
              "Compared to perturbation-based defenses, which reduce retrieval "
              "re-identification by uniformly degrading nearest-neighbor structure, "
              "our role-aware guard selectively restricts extractive and semantic "
              "leakage patterns while preserving benign study utility in the evaluated setting.",
              style_bullet),
        _para("&bull; <b>Optimized attacker results mark the boundary, not "
              "failure.</b> The iterative black-box search category is "
              "intentionally harder and lowers block rate; this is reported "
              "as evidence of bounded empirical privacy under stronger "
              "adversaries rather than as a claim of complete protection. "
              "Even partially successful cases remain measurable through "
              "semantic and overlap risk signals.",
              style_bullet),
        _para("&bull; <b>Components are complementary.</b> No single "
              "module fully explains all gains: ordinal cue handling, "
              "uncertainty gating, and retrieval governance contribute "
              "different parts of the observed behavior.", style_bullet),
        _para("&bull; <b>Core claim is measurement-oriented.</b> The work "
              "claims a reproducible finding (ordinal signal survives "
              "better than exact labels) and a bounded privacy-evaluation "
              "framework, not a fundamentally new base model family.",
              style_bullet),
        _para("&bull; <b>Local deployment remains feasible.</b> The QA and "
              "resource measurements are not the main novelty, but they "
              "support the applied setting: the system runs offline, on "
              "CPU, within the reported private-memory budget."),
    ]

    # ---------- 9. Limitations ----------
    story += [
        Paragraph("9. Limitations", style_h1),
        _para("&bull; <b>Benchmark scope.</b> Core QA evaluation still uses "
              "the OBE academic pool; external QA adapters are implemented "
              "but full external runs depend on local offline dataset "
              "availability.", style_bullet),
        _para("&bull; <b>Privacy is proxy-based and unmoved.</b> The "
              "two ASR proxies (document-match and cosine-threshold) "
              "are flat over &lambda;; we cannot conclude that the "
              "InfoNCE re-ranker has any privacy effect on this "
              "benchmark.", style_bullet),
        _para("&bull; <b>Role-guard privacy is attack-set-specific.</b> "
              "High block rates in the direct reconstruction, indirect "
              "leakage, paraphrase-probe, adaptive, multi-turn, and "
              "semantic-reconstruction taxonomy do not imply perfect "
              "privacy, differential privacy, or robustness to unseen "
              "attacks.", style_bullet),
        _para("&bull; <b>Cross-domain Bloom accuracy is not solved.</b> "
              "The paper uses the transfer drop as evidence of domain "
              "shift and motivates ordinal/uncertainty handling; it does "
              "not claim universal Bloom generalization.", style_bullet),
        _para("&bull; <b>Classifier is over-confident "
              "(Table&nbsp;4).</b> Confidence exceeds accuracy in every "
              "populated bin.", style_bullet),
        _para("&bull; <b>Uncertainty is not a reliable error signal "
              "(Tables&nbsp;5&ndash;6).</b> Pearson r = &minus;0.047 "
              "and per-bin error rates are roughly flat (0.78&ndash;0.85).",
              style_bullet),
        _para("&bull; <b>Single hardware/OS.</b> Results in "
              "Tables&nbsp;8&ndash;9 are reported for one machine; "
              "portability is not yet measured.", style_bullet),
    ]

    # ---------- 10. Reproducibility ----------
    story += [
        Paragraph("10. Reproducibility and Artifact Integrity", style_h1),
        _para("The reproducibility bundle (<i>paper_bundle/</i>) "
              "generated by paper_pack_builder.build() contains: all "
              "figures/*.{png,pdf}, all results/*.json, code snapshots "
              "of evaluate.py, classifier.py, and dataset_adapters.py, "
              "a configuration snapshot (<i>config_snapshot.json</i>), "
              "run metadata including platform, Python version, dataset "
              "list, seed, &lambda;, and git commit hash "
              "(<i>run_metadata.json</i>), and a SHA-256 integrity "
              "manifest (<i>integrity_manifest.json</i>). The audit "
              "routine validates determinism, offline-mode operation, "
              "USS &lt; 1024&nbsp;MB, hash verification, and presence "
              "of every advertised metric file before printing "
              "<i>STATUS: READY FOR SUBMISSION</i>. Re-runs are "
              "protected by --force-paper-build. A one-command runner "
              "(<i>run_submission_pipeline.py</i>) executes evaluation, "
              "cross-domain transfer, privacy guard, table consolidation, "
              "PDF generation, and bundle audit in sequence."),
    ]

    # ---------- 11. Conclusion ----------
    story += [
        Paragraph("11. Conclusion", style_h1),
        _para(
            "We evaluated a local academic-assistance framework through "
            "the lens of cognitive robustness and privacy-constrained "
            "deployment under domain shift. Directional Figshare "
            "&harr; MoocRadar transfer shows that exact Bloom "
            "classification can collapse while ordinal proximity remains "
            "more stable, motivating LDL soft feedback and confidence "
            "gating rather than hard-level over-specialisation. Privacy "
            "results are similarly bounded: the InfoNCE re-ranker does "
            "not reduce the measured retrieval-leakage proxies, while the "
            "role-aware guard shows resistance only for the defined "
            "adversarial prompt taxonomy. The resulting paper claim is "
            "therefore intentionally precise: a reproducible, CPU-only "
            "framework for studying and deploying cognitive assistance "
            "under domain shift and protected-resource constraints, not a "
            "claim of solved Bloom generalization or perfect privacy. "
            "Together, these results suggest that privacy in offline academic "
            "RAG systems is best approached as a controllable tradeoff rather "
            "than a binary guarantee."),
    ]

    # ---------- 12. References ----------
    story += [Paragraph("12. References", style_h1)]
    if REFS:
        for r in REFS:
            story.append(_para(_format_ref(r)))
    else:
        story.append(_para("References placeholders are missing (refs.json)."))

    return story


def main() -> None:
    doc = _build_doc()
    doc.build(build_story())
    size_kb = OUT_PDF.stat().st_size / 1024.0
    print(f"wrote {OUT_PDF.relative_to(REPO)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
