from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("results/bloom_lora_eval.json")
PRIVACY = Path("results/privacy_guard_eval.json")
UNIFIED = Path("results/unified_results_table.csv")
FIG_DIR = Path("figures")


def _plot_bloom_confusion() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    cm = np.array(data.get("confusion_matrix") or [], dtype=int)
    if cm.size == 0:
        # fallback: regenerate from rows if only metrics saved
        rows_path = Path("results/bloom_lora_eval_rows.csv")
        if not rows_path.is_file():
            return
        return
    labels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Qwen LoRA Bloom Confusion Matrix (Figshare Test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bloom_lora_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_unified_table() -> None:
    if not UNIFIED.is_file():
        return
    df = pd.read_csv(UNIFIED)
    disp = df[
        ["evidence_area", "protocol", "setting", "model", "primary_metric", "primary_value"]
    ].copy()
    disp["primary_value"] = disp["primary_value"].map(
        lambda x: f"{float(x):.3f}" if pd.notna(x) and str(x) != "" else ""
    )
    fig, ax = plt.subplots(figsize=(16, max(3, 0.35 * len(disp) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    ax.set_title("Unified Evaluation Summary", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "unified_results_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_privacy_bars() -> None:
    if not PRIVACY.is_file():
        return
    data = json.loads(PRIVACY.read_text(encoding="utf-8"))
    labels = ["Attack block", "Benign allow", "Teacher allow"]
    values = [
        float(data.get("student_attack_block_rate", 0)),
        float(data.get("student_benign_allow_rate", 0)),
        float(data.get("teacher_moderation_allow_rate", 0)),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#d62728", "#2ca02c", "#1f77b4"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("PrivacyGuard Role-Aware Outcomes")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "privacy_guard_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS.is_file():
        _plot_bloom_confusion()
    _plot_unified_table()
    _plot_privacy_bars()
    print("figures-generated")


if __name__ == "__main__":
    main()
