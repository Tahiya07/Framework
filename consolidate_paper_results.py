#!/usr/bin/env python
"""Merge canonical evaluation JSON files into a publication-ready results table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


RESULTS_DIR = Path("results")
OUT_JSON = RESULTS_DIR / "unified_results_table.json"
OUT_CSV = RESULTS_DIR / "unified_results_table.csv"
OUT_MD = RESULTS_DIR / "unified_results_table.md"


def _row(
    *,
    evidence_area: str,
    protocol: str,
    setting: str,
    model: str,
    primary_metric: str,
    primary_value: Optional[float],
    accuracy: Optional[float] = None,
    within_one_level: Optional[float] = None,
    severe_error: Optional[float] = None,
    interpretation: str = "",
) -> Dict[str, Any]:
    return {
        "evidence_area": evidence_area,
        "protocol": protocol,
        "setting": setting,
        "model": model,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "accuracy": accuracy,
        "within_one_level": within_one_level,
        "severe_error": severe_error,
        "interpretation": interpretation,
    }


def _load(path: Path) -> Optional[dict]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _baseline_comparison_rows(data: dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    split_notes = {
        "SVM": data.get("sources", {}).get("holdout_split", "15% hold-out"),
        "zero_shot_gguf": data.get("sources", {}).get("holdout_split", "15% hold-out"),
        "lora_trained": data.get("sources", {}).get("lora_split", "official test split"),
    }
    for key, label in (
        ("SVM", "TF-IDF + LinearSVC"),
        ("zero_shot_gguf", "Qwen2.5 zero-shot (GGUF)"),
        ("lora_trained", "Qwen2.5 LoRA (trained)"),
    ):
        metrics = data.get(key)
        if not metrics:
            continue
        n_test = data.get("n_test_holdout") if key != "lora_trained" else data.get("n_test_lora")
        rows.append(
            _row(
                evidence_area="cognitive robustness",
                protocol="Bloom baseline comparison",
                setting=f"{split_notes.get(key, 'Figshare')} (n={n_test or '?'})",
                model=label,
                primary_metric="macro_f1",
                primary_value=metrics.get("macro_f1"),
                accuracy=metrics.get("accuracy"),
                within_one_level=metrics.get("within_one_level_accuracy"),
                severe_error=metrics.get("severe_error_rate"),
                interpretation="bloom_evaluation.py on shared 15% hold-out split",
            )
        )
    ref = data.get("trained_model_reference") or {}
    if ref and not any(r.get("model") == "Qwen2.5 LoRA (full test eval)" for r in rows):
        rows.append(
            _row(
                evidence_area="cognitive robustness",
                protocol="Bloom classification (full test)",
                setting="Figshare official test split",
                model="Qwen2.5 LoRA (full test eval)",
                primary_metric="macro_f1",
                primary_value=ref.get("macro_f1"),
                accuracy=ref.get("accuracy"),
                within_one_level=ref.get("within_one_level_accuracy"),
                severe_error=ref.get("severe_error_rate"),
                interpretation="evaluation_results/metrics.json from train_qwen_bloom.py",
            )
        )
    return rows


def _bloom_rows(data: dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lora = data.get("qwen_lora", {})
    rows.append(
        _row(
            evidence_area="cognitive robustness",
            protocol="Bloom classification",
            setting="Figshare held-out test",
            model="Qwen2.5-1.5B LoRA",
            primary_metric="macro_f1",
            primary_value=lora.get("macro_f1"),
            accuracy=lora.get("accuracy"),
            within_one_level=lora.get("within_one_level_accuracy"),
            severe_error=lora.get("severe_error_rate"),
            interpretation="merged Qwen Bloom classifier (train_qwen_bloom.py + merge_model.py)",
        )
    )
    svm = data.get("tfidf_svm_baseline")
    if svm:
        rows.append(
            _row(
                evidence_area="cognitive robustness",
                protocol="Bloom classification baseline",
                setting="Figshare held-out test",
                model="TF-IDF + LinearSVC",
                primary_metric="macro_f1",
                primary_value=svm.get("macro_f1"),
                accuracy=svm.get("accuracy"),
                within_one_level=svm.get("within_one_level_accuracy"),
                severe_error=svm.get("severe_error_rate"),
                interpretation="classical lexical baseline",
            )
        )
    if "lora_svm_agreement" in data:
        rows.append(
            _row(
                evidence_area="cognitive robustness",
                protocol="LoRA vs SVM agreement",
                setting="Figshare held-out test",
                model="Qwen LoRA vs TF-IDF SVM",
                primary_metric="agreement",
                primary_value=data.get("lora_svm_agreement"),
                interpretation="label agreement between neural and classical baselines",
            )
        )
    return rows


def _rag_rows(data: dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    qa = data.get("academic_qa", {})
    if qa:
        rows.append(
            _row(
                evidence_area="student learning (RAG)",
                protocol="academic QA smoke benchmark",
                setting="FAISS + Qwen GGUF",
                model="academic_qa SLM",
                primary_metric="token_f1_mean",
                primary_value=(qa.get("token_f1") or {}).get("mean"),
                accuracy=(qa.get("exact_match") or {}).get("mean"),
                interpretation="small grounded QA set (evaluate_qwen_rag.py)",
            )
        )
        rows.append(
            _row(
                evidence_area="student learning (RAG)",
                protocol="retrieval hit@3",
                setting="FAISS + MiniLM",
                model="PrivacyRetriever",
                primary_metric="hit_at_3_mean",
                primary_value=(qa.get("retrieval_hit_at_3") or {}).get("mean"),
                interpretation="retrieval supports answer context",
            )
        )
    mm = data.get("multimodal_rag", {})
    if mm:
        rows.append(
            _row(
                evidence_area="multimodal ingestion",
                protocol="PDF + image RAG smoke",
                setting="PyMuPDF + OCR path",
                model="MultiModalAcademicRAG",
                primary_metric="answer_accuracy_mean",
                primary_value=(mm.get("answer_accuracy") or {}).get("mean"),
                interpretation="extractive multimodal smoke test",
            )
        )
    return rows


def _multimodal_rows(data: dict) -> List[Dict[str, Any]]:
    if not data:
        return []
    return [
        _row(
            evidence_area="multimodal ingestion",
            protocol="multimodal_rag_smoke_v1",
            setting="pdf + image",
            model="MultiModalAcademicRAG",
            primary_metric="answer_accuracy_on_ok_cases",
            primary_value=data.get("answer_accuracy_on_ok_cases"),
            interpretation="; ".join(data.get("limitations", [])[:2]),
        )
    ]


def _ocr_rows(data: dict) -> List[Dict[str, Any]]:
    if not data:
        return []
    backend = data.get("ocr_backend", {})
    return [
        _row(
            evidence_area="multimodal ingestion",
            protocol="OCR pipeline readiness",
            setting="Tesseract / fallback",
            model=backend.get("engine", "unknown"),
            primary_metric="available",
            primary_value=float(bool(backend.get("available"))),
            interpretation=backend.get("reason", ""),
        )
    ]


def _privacy_rows(data: dict) -> List[Dict[str, Any]]:
    if not data:
        return []
    rows = [
        _row(
            evidence_area="privacy-constrained deployment",
            protocol="student attack block rate",
            setting="adversarial prompt taxonomy",
            model="PrivacyGuard",
            primary_metric="block_rate",
            primary_value=data.get("student_attack_block_rate"),
            interpretation="measured under defined attack prompts",
        ),
        _row(
            evidence_area="privacy-constrained deployment",
            protocol="student benign allow rate",
            setting="benign study prompts",
            model="PrivacyGuard",
            primary_metric="allow_rate",
            primary_value=data.get("student_benign_allow_rate"),
            interpretation="utility under non-adversarial student queries",
        ),
    ]
    for category, summary in (data.get("attack_category_summary") or {}).items():
        rows.append(
            _row(
                evidence_area="privacy-constrained deployment",
                protocol="attack taxonomy",
                setting=category,
                model="PrivacyGuard",
                primary_metric="category_block_rate",
                primary_value=(summary or {}).get("block_rate"),
                interpretation="per-category adversarial result",
            )
        )
    return rows


def _privacy_benchmark_rows(data: dict) -> List[Dict[str, Any]]:
    if not data:
        return []
    rows: List[Dict[str, Any]] = []
    for name, block in (data.get("baselines") or {}).items():
        rows.append(
            _row(
                evidence_area="privacy-constrained deployment",
                protocol="privacy baseline ablation",
                setting="student attacks",
                model=str(name),
                primary_metric="attack_block_rate",
                primary_value=block.get("attack_block_rate"),
                interpretation="compare guard variants on shared attack suite",
            )
        )
    return rows


def _federated_rows(data: dict) -> List[Dict[str, Any]]:
    if not data:
        return []
    metrics = data.get("metrics") or {}
    return [
        _row(
            evidence_area="federated privacy layer",
            protocol="federated privacy-risk model",
            setting="aggregate-only updates",
            model="FederatedPrivacyGuard",
            primary_metric="attack_block_rate",
            primary_value=metrics.get("attack_block_rate"),
            interpretation="no raw teacher items sent to server; parameters only",
        ),
        _row(
            evidence_area="federated privacy layer",
            protocol="federated privacy-risk model",
            setting="benign student prompts",
            model="FederatedPrivacyGuard",
            primary_metric="benign_allow_rate",
            primary_value=metrics.get("benign_allow_rate"),
            interpretation="utility after federated guard training",
        ),
    ]


def _federated_lora_rows(data: dict) -> List[Dict[str, Any]]:
    if not data:
        return []
    return [
        _row(
            evidence_area="federated LLM layer",
            protocol="teacher Bloom LoRA FedAvg",
            setting=f"{data.get('num_clients', '?')} simulated clients",
            model="Qwen2.5-1.5B LoRA",
            primary_metric="rounds",
            primary_value=float(data.get("rounds", 0)),
            interpretation="encrypted adapter bundles aggregated server-side; raw questions stay on clients",
        )
    ]


def build_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fl_sim = _load(RESULTS_DIR / "federated_lora_simulation.json")
    rows.extend(_federated_lora_rows(fl_sim or {}))
    baseline = _load(RESULTS_DIR / "bloom_baseline_comparison.json")
    if baseline:
        rows.extend(_baseline_comparison_rows(baseline))
    bloom = _load(RESULTS_DIR / "bloom_lora_eval.json")
    if bloom:
        rows.extend(_bloom_rows(bloom))
    rag = _load(RESULTS_DIR / "qwen_rag_eval.json")
    if rag:
        rows.extend(_rag_rows(rag))
    mm = _load(RESULTS_DIR / "multimodal_rag_eval.json")
    rows.extend(_multimodal_rows(mm or {}))
    ocr = _load(RESULTS_DIR / "ocr_image_pipeline_eval.json")
    rows.extend(_ocr_rows(ocr or {}))
    priv = _load(RESULTS_DIR / "privacy_guard_eval.json")
    rows.extend(_privacy_rows(priv or {}))
    bench = _load(RESULTS_DIR / "privacy_benchmark_baselines.json")
    rows.extend(_privacy_benchmark_rows(bench or {}))
    fed = _load(RESULTS_DIR / "federated_privacy_guard.json")
    rows.extend(_federated_rows(fed or {}))
    return rows


def _to_markdown(df: pd.DataFrame) -> str:
    cols = [
        "evidence_area",
        "protocol",
        "setting",
        "model",
        "primary_metric",
        "primary_value",
        "accuracy",
        "within_one_level",
        "severe_error",
        "interpretation",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = ["# Unified Results Table", "", header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.3f}" if val == val else "")
            else:
                cells.append(str(val) if val is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = build_table()
    if not rows:
        raise SystemExit("No evaluation artifacts found under results/. Run run_evaluation_pipeline.py first.")
    payload = {"rows": rows, "n_rows": len(rows)}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(_to_markdown(df), encoding="utf-8")
    print(f"Wrote {len(rows)} rows -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
