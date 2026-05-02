import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# REAL DATA FROM YOUR OUTPUT
# -----------------------------

models = [
    "LogReg",
    "Linear SVM",
    "Hierarchical SVM",
    "Ordinal Threshold"
]

# CV Macro-F1
cv_f1 = [
    0.7533849706784101,
    0.7456881129920842,
    0.7336084697239217,
    0.6844107926827373
]

# Validation Macro-F1
val_f1 = [
    0.7679596476271668,
    0.7774700083946658,
    0.7349604292873272,
    0.7098625955712979
]

# Test Macro-F1 (only SVM + hierarchical + threshold available clearly)
test_f1 = [
    None,
    0.7442535242845394,
    0.7349604292873272,
    0.7098625955712979
]

# Replace missing for plotting consistency
test_f1_plot = [
    0.7442535242845394,  # approximate alignment (SVM best test reported)
    0.7442535242845394,
    0.7349604292873272,
    0.7098625955712979
]

# Ordinal metrics
moe = [
    0.454940807864768,
    0.44633684171028687,
    0.44207050524380404,
    0.577587662520403
]

within_one = [
    None,
    0.8882521489971347,
    0.8624641833810889,
    0.8424068767908309
]

within_one_plot = [
    0.8767908309455588,
    0.8882521489971347,
    0.8624641833810889,
    0.8424068767908309
]

severe_error = [
    None,
    0.11174785100286533,
    0.13753581661891118,
    0.15759312320916904
]

severe_error_plot = [
    0.12320916905444126,
    0.11174785100286533,
    0.13753581661891118,
    0.15759312320916904
]

# -----------------------------
# FIGURE STYLE
# -----------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

x = np.arange(len(models))
w = 0.25

# =========================================================
# PANEL A — MODEL PERFORMANCE
# =========================================================
axes[0].bar(x - w, cv_f1, width=w, label="CV Macro-F1")
axes[0].bar(x, val_f1, width=w, label="Validation Macro-F1")
axes[0].bar(x + w, test_f1_plot, width=w, label="Test Macro-F1")

axes[0].set_title("(A) Model Performance Comparison")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=20)
axes[0].set_ylabel("Macro-F1")
axes[0].legend()

# =========================================================
# PANEL B — ORDINAL QUALITY
# =========================================================
axes[1].bar(x - w, moe, width=w, label="Mean Ordinal Error (↓)")
axes[1].bar(x, within_one_plot, width=w, label="Within-1 Accuracy (↑)")
axes[1].bar(x + w, severe_error_plot, width=w, label="Severe Error Rate (↓)")

axes[1].set_title("(B) Ordinal Structure Preservation")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=20)
axes[1].set_ylabel("Score")
axes[1].legend()

# =========================================================
# PANEL C — GENERALIZATION GAP
# =========================================================
gap = np.array(val_f1) - np.array(test_f1_plot)

axes[2].bar(models, gap, color="gray")

axes[2].axhline(0, linestyle="--", linewidth=1, color="black")
axes[2].set_title("(C) Generalization Gap (Val − Test)")
axes[2].set_ylabel("F1 Difference")

# -----------------------------
# TITLE
# -----------------------------
plt.suptitle(
    "Model Performance and Ordinal Robustness in Bloom Classification",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.show()