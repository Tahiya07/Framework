from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")


def _load(name: str) -> Dict[str, Any]:
    path = RESULTS_DIR / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _cross_domain_rows() -> List[Dict[str, Any]]:
    data = _load("cross_dataset_bloom_transfer.json")
    if not data:
        return []
    rows: List[Dict[str, Any]] = []
    for scheme in ["binary", "ternary"]:
        block = data.get("schemes", {}).get(scheme, {})
        for setting, label in [
            ("within_dataset_figshare", "Figshare in-domain"),
            ("within_dataset_moocradar", "MoocRadar in-domain"),
            ("figshare_to_moocradar", "Figshare -> MoocRadar"),
            ("moocradar_to_figshare", "MoocRadar -> Figshare"),
        ]:
            entry = block.get(setting, {})
            metrics = entry.get("selected_metrics", {})
            if not metrics:
                continue
            rows.append(
                {
                    "evidence_area": "cognitive robustness",
                    "protocol": f"{scheme} Bloom transfer",
                    "setting": label,
                    "model": entry.get("selected_model", ""),
                    "primary_metric": "macro_f1",
                    "primary_value": metrics.get("macro_f1"),
                    "accuracy": metrics.get("accuracy"),
                    "within_one_level_accuracy": metrics.get("within_one_level_accuracy"),
                    "severe_error_rate": metrics.get("severe_error_rate"),
                    "interpretation": (
                        "cross-domain class-level degradation"
                        if "->" in label
                        else "in-domain reference point"
                    ),
                }
            )
    return rows


def _privacy_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    curve = _load("privacy_curve.json")
    if curve:
        rows.append(
            {
                "evidence_area": "privacy-constrained deployment",
                "protocol": "retrieval leakage sweep",
                "setting": "InfoNCE lambda sweep",
                "model": "PrivacyRetriever",
                "primary_metric": "document_match_asr_auc",
                "primary_value": curve.get("auc_asr_doc"),
                "accuracy": "",
                "within_one_level_accuracy": "",
                "severe_error_rate": "",
                "interpretation": "proxy leakage did not improve with lambda; report as negative result",
            }
        )
        rows.append(
            {
                "evidence_area": "privacy-constrained deployment",
                "protocol": "retrieval leakage sweep",
                "setting": "InfoNCE lambda sweep",
                "model": "PrivacyRetriever",
                "primary_metric": "cosine_threshold_asr_auc",
                "primary_value": curve.get("auc_asr_cos"),
                "accuracy": "",
                "within_one_level_accuracy": "",
                "severe_error_rate": "",
                "interpretation": "proxy leakage did not improve with lambda; avoid over-claiming privacy",
            }
        )

    guard = _load("privacy_guard_eval.json")
    if guard:
        rows.append(
            {
                "evidence_area": "privacy-constrained deployment",
                "protocol": "role-aware adversarial prompt taxonomy",
                "setting": "student reconstruction attacks",
                "model": "PrivacyGuard",
                "primary_metric": "block_rate",
                "primary_value": guard.get("student_attack_block_rate"),
                "accuracy": "",
                "within_one_level_accuracy": "",
                "severe_error_rate": "",
                "interpretation": "measured resistance under defined attack prompts, not proof of perfect privacy",
            }
        )
        rows.append(
            {
                "evidence_area": "privacy-constrained deployment",
                "protocol": "role-aware benign-use check",
                "setting": "student benign prompts",
                "model": "PrivacyGuard",
                "primary_metric": "allow_rate",
                "primary_value": guard.get("student_benign_allow_rate"),
                "accuracy": "",
                "within_one_level_accuracy": "",
                "severe_error_rate": "",
                "interpretation": "guard preserves benign study assistance in the evaluated set",
            }
        )
        for category, item in guard.get("attack_category_summary", {}).items():
            rows.append(
                {
                    "evidence_area": "privacy-constrained deployment",
                    "protocol": "attack taxonomy",
                    "setting": category,
                    "model": "PrivacyGuard",
                    "primary_metric": "category_block_rate",
                    "primary_value": item.get("block_rate"),
                    "accuracy": "",
                    "within_one_level_accuracy": "",
                    "severe_error_rate": "",
                    "interpretation": "category-level adversarial prompt result",
                }
            )
        leakage_summary = guard.get("leakage_signal_summary", {})
        if "max_semantic_concept_ratio" in leakage_summary:
            rows.append(
                {
                    "evidence_area": "privacy-constrained deployment",
                    "protocol": "semantic leakage probe",
                    "setting": "protected concept overlap",
                    "model": "PrivacyGuard",
                    "primary_metric": "max_semantic_concept_ratio",
                    "primary_value": leakage_summary.get("max_semantic_concept_ratio"),
                    "accuracy": "",
                    "within_one_level_accuracy": "",
                    "severe_error_rate": "",
                    "interpretation": "semantic-risk proxy for paraphrased leakage without long copied spans",
                }
            )
        for point in guard.get("safety_utility_curve", []):
            rows.append(
                {
                    "evidence_area": "privacy-constrained deployment",
                    "protocol": "safety-utility curve",
                    "setting": f"semantic_threshold={point.get('semantic_threshold')}",
                    "model": "PrivacyGuard",
                    "primary_metric": "attack_block_rate / benign_allow_rate",
                    "primary_value": (
                        f"{point.get('attack_block_rate', '')} / "
                        f"{point.get('benign_allow_rate', '')}"
                    ),
                    "accuracy": point.get("benign_allow_rate"),
                    "within_one_level_accuracy": "",
                    "severe_error_rate": "",
                    "interpretation": "stricter semantic thresholds increase safety pressure and may reduce utility",
                }
            )
        auc = guard.get("safety_utility_curve_auc", {})
        if auc:
            rows.append(
                {
                    "evidence_area": "privacy-constrained deployment",
                    "protocol": "safety-utility curve",
                    "setting": "integral summary",
                    "model": "PrivacyGuard",
                    "primary_metric": "attack_block_auc / benign_allow_auc",
                    "primary_value": (
                        f"{auc.get('attack_block_auc', '')} / "
                        f"{auc.get('benign_allow_auc', '')}"
                    ),
                    "accuracy": "",
                    "within_one_level_accuracy": "",
                    "severe_error_rate": "",
                    "interpretation": "integrates strictness-vs-utility response across semantic thresholds",
                }
            )
    return rows


def _cue_analysis_rows() -> List[Dict[str, Any]]:
    data = _load("bloom_domain_shift_cue_analysis.json")
    if not data:
        return []
    rows: List[Dict[str, Any]] = []
    summary = data.get("summary", {})
    for setting, label in [
        ("figshare_to_moocradar", "Figshare -> MoocRadar"),
        ("moocradar_to_figshare", "MoocRadar -> Figshare"),
    ]:
        item = summary.get(setting, {})
        if not item:
            continue
        rows.append(
            {
                "evidence_area": "domain-shift explanation",
                "protocol": "cue vs content ablation",
                "setting": label,
                "model": "cue_only - content_tfidf",
                "primary_metric": "delta_severe_error",
                "primary_value": item.get("cue_minus_content_severe"),
                "accuracy": "",
                "within_one_level_accuracy": item.get("cue_minus_content_within_one"),
                "severe_error_rate": item.get("cue_minus_content_severe"),
                "interpretation": "negative severe-error delta means Bloom cue features reduce severe ordinal jumps",
            }
        )
        rows.append(
            {
                "evidence_area": "domain-shift explanation",
                "protocol": "cue plus content ablation",
                "setting": label,
                "model": "combined - content_tfidf",
                "primary_metric": "delta_macro_f1",
                "primary_value": item.get("combined_minus_content_macro_f1"),
                "accuracy": "",
                "within_one_level_accuracy": item.get("combined_minus_content_within_one"),
                "severe_error_rate": item.get("combined_minus_content_severe"),
                "interpretation": "tests whether cognitive cues add stable signal beyond topic vocabulary",
            }
        )
    return rows


def _qa_resource_rows() -> List[Dict[str, Any]]:
    metrics = _load("metrics.json")
    efficiency = _load("efficiency.json")
    rows: List[Dict[str, Any]] = []
    qa = metrics.get("qa", {})
    for system in ["Proposed", "VanillaRAG", "BM25", "NoRAG"]:
        item = qa.get(system, {})
        if not item:
            continue
        rows.append(
            {
                "evidence_area": "deployment utility",
                "protocol": "bounded local QA",
                "setting": system,
                "model": system,
                "primary_metric": "token_f1",
                "primary_value": item.get("f1", {}).get("mean"),
                "accuracy": "",
                "within_one_level_accuracy": "",
                "severe_error_rate": "",
                "interpretation": "utility reference under local/offline generation",
            }
        )
    if efficiency:
        rows.append(
            {
                "evidence_area": "deployment utility",
                "protocol": "resource constraint",
                "setting": "private working-set memory",
                "model": "full framework",
                "primary_metric": "uss_mb",
                "primary_value": efficiency.get("uss_mb_now"),
                "accuracy": "",
                "within_one_level_accuracy": "",
                "severe_error_rate": "",
                "interpretation": "private RAM footprint for CPU-only deployment",
            }
        )
    return rows


def _write_markdown(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Evidence area",
        "Protocol",
        "Setting",
        "Model",
        "Primary metric",
        "Primary value",
        "Accuracy",
        "Within-one-level",
        "Severe error",
        "Interpretation",
    ]
    lines = ["# Unified Results Table", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("evidence_area")),
                    _fmt(row.get("protocol")),
                    _fmt(row.get("setting")),
                    _fmt(row.get("model")),
                    _fmt(row.get("primary_metric")),
                    _fmt(row.get("primary_value")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("within_one_level_accuracy")),
                    _fmt(row.get("severe_error_rate")),
                    _fmt(row.get("interpretation")),
                ]
            )
            + " |"
        )
    (RESULTS_DIR / "unified_results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _cross_domain_rows() + _cue_analysis_rows() + _privacy_rows() + _qa_resource_rows()

    out_json = RESULTS_DIR / "unified_results_table.json"
    out_csv = RESULTS_DIR / "unified_results_table.csv"
    fig_csv = FIG_DIR / "unified_results_table.csv"
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    fieldnames = [
        "evidence_area",
        "protocol",
        "setting",
        "model",
        "primary_metric",
        "primary_value",
        "accuracy",
        "within_one_level_accuracy",
        "severe_error_rate",
        "interpretation",
    ]
    for path in [out_csv, fig_csv]:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    _write_markdown(rows)
    print(f"unified-results-written rows={len(rows)}")


if __name__ == "__main__":
    main()
