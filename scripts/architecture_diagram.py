"""Render the architecture diagram for the hybrid TCN-BiGRU-Attention + Semantic + XGBoost pipeline."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path


COLORS = {
    "data":    "#E3F2FD",
    "prep":    "#BBDEFB",
    "encoder": "#FFE0B2",
    "tcn":     "#FFCC80",
    "rnn":     "#FFB74D",
    "attn":    "#FFA726",
    "head":    "#FFCDD2",
    "sem":     "#C8E6C9",
    "concat":  "#D1C4E9",
    "xgb":     "#F8BBD0",
    "calib":   "#F0F4C3",
    "thr":     "#B2DFDB",
    "out":     "#CFD8DC",
}

EDGE = "#37474F"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def box(ax, x, y, w, h, text, face, fontsize=9, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2, edgecolor=EDGE, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight)
    return (x + w / 2, y, x + w / 2, y + h)  # (cx, ybottom, cx, ytop)


def arrow(ax, x1, y1, x2, y2, label=None, style="-|>", color=EDGE, lw=1.4):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, label,
                fontsize=8, color="#263238",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.85))


fig, ax = plt.subplots(figsize=(15, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.set_aspect("equal")
ax.axis("off")

ax.text(7, 21.3,
        "Hybrid TCN–BiGRU–Attention + Semantic + XGBoost (CMAPSS)",
        ha="center", fontsize=14, weight="bold")

# 1. Raw data
_, _, dx, dy = box(ax, 4, 19.6, 6, 0.9,
                   "CMAPSS raw data (FD001–FD004): train_FD0xx · test_FD0xx · RUL_FD0xx",
                   COLORS["data"], fontsize=10, weight="bold")

# 2. Preprocessing
_, py_b, _, py_t = box(ax, 2.5, 17.5, 9, 1.6,
    "Preprocessing\n"
    "add_train_rul / add_test_rul   •   split_train_val_by_engine\n"
    "StandardScaler (fit on train)   •   make_sliding_windows  →  X ∈ ℝ^(N × T × F)",
    COLORS["prep"], fontsize=9)
arrow(ax, 7, 19.6, 7, py_t)

# Split point
arrow(ax, 7, py_b, 7, 16.6)
ax.text(7, 16.75, "X (sliding windows)",
        ha="center", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", edgecolor="#90A4AE"))

# Branch lines
arrow(ax, 7, 16.4, 3.4, 15.6)   # to encoder
arrow(ax, 7, 16.4, 10.6, 15.6)  # to semantic

# 3a. Deep encoder container
enc_x, enc_y, enc_w, enc_h = 0.6, 5.6, 6.4, 10.0
patch = FancyBboxPatch(
    (enc_x, enc_y), enc_w, enc_h,
    boxstyle="round,pad=0.05,rounding_size=0.18",
    linewidth=1.6, edgecolor=EDGE, facecolor=COLORS["encoder"], alpha=0.45,
)
ax.add_patch(patch)
ax.text(enc_x + enc_w / 2, enc_y + enc_h - 0.35,
        "TCNBiGRUAttention  (deep encoder)",
        ha="center", fontsize=10.5, weight="bold")

# TCN
box(ax, 0.85, 13.2, 5.9, 1.9,
    "TCN  —  Sequential[ CausalConvBlock × len(channels) ]\n"
    "Conv1D(weight-norm, causal pad, dilation = 2^i)\n"
    "→ ReLU → Dropout → Conv1D → ReLU → Dropout\n"
    "+ residual (1×1 Conv if Δch)  → ReLU",
    COLORS["tcn"], fontsize=8.2)

# BiGRU
box(ax, 1.0, 11.4, 5.6, 1.3,
    "BiGRU\n"
    "hidden = gru_hidden, layers = gru_layers, bidirectional\n"
    "out: (B, T, 2·gru_hidden)",
    COLORS["rnn"], fontsize=8.5)

# Attention pool
box(ax, 1.0, 9.8, 5.6, 1.1,
    "AttentionPool\n"
    "α_t = softmax(W·h_t)   ;   pooled = Σ α_t · h_t",
    COLORS["attn"], fontsize=8.5)

# Projection
box(ax, 0.85, 8.15, 5.9, 1.2,
    "Projection\n"
    "Linear → LayerNorm → GELU → Dropout(0.20)\n"
    "→  z ∈ ℝ^(B × embed_dim)",
    COLORS["attn"], fontsize=8.2)

# Two heads
box(ax, 1.0, 6.5, 2.6, 1.2,
    "cls_head\nLinear(embed,1) → logit\n(FocalLoss / BCE pos_weight)",
    COLORS["head"], fontsize=8)
box(ax, 4.0, 6.5, 2.6, 1.2,
    "recon_head\nLinear→ReLU→Linear(F)\n(auto-encoder reg.)",
    COLORS["head"], fontsize=8)

# Internal arrows
arrow(ax, 3.8, 13.3, 3.8, 12.7)   # TCN -> BiGRU
arrow(ax, 3.8, 11.4, 3.8, 10.9)   # BiGRU -> Attn
arrow(ax, 3.8, 9.8,  3.8, 9.3)    # Attn -> Proj
arrow(ax, 3.8, 8.2,  2.3, 7.7)    # Proj -> cls
arrow(ax, 3.8, 8.2,  5.3, 7.7)    # Proj -> recon

# 3b. Semantic features
box(ax, 8.4, 11.3, 5.2, 4.3,
    "Semantic features\nextract_semantic_features(X)\n\n"
    "Per feature, 12 statistics over the window:\n"
    "mean · std · min · max · range · drift\n"
    "slope · |Δ|mean · Δstd · energy\n"
    "pos_frac · neg_frac\n\n"
    "→  ℝ^(N × F·12)",
    COLORS["sem"], fontsize=9)

# Embedding output going down/right to concat
# embedding leaves projection
arrow(ax, 6.6, 8.75, 8.0, 5.0, label="z (embedding)")
# semantic leaves
arrow(ax, 11.0, 11.3, 8.6, 5.0, label="semantic vec")

# 4. Concatenation
box(ax, 5.4, 4.2, 5.2, 0.9,
    "Concatenate  (build_feature_matrix)",
    COLORS["concat"], fontsize=10, weight="bold")
arrow(ax, 8, 4.2, 8, 3.7)

# 5. XGBoost ensemble
box(ax, 3.6, 2.6, 8.8, 1.1,
    "XGBoost Ensemble  (train_xgboost_ensemble)\n"
    "1) GridSearchCV (F-beta) → best_params   2) K boosters / different seeds   3) ensemble_predict = mean prob",
    COLORS["xgb"], fontsize=9)
arrow(ax, 8, 2.6, 8, 2.1)

# 6. Calibration
box(ax, 4.6, 1.4, 6.8, 0.7,
    "Optional isotonic calibration  (fit_isotonic_calibrator)",
    COLORS["calib"], fontsize=9)
arrow(ax, 8, 1.4, 8, 0.95)

# 7. Threshold / decision
box(ax, 1.0, 0.05, 12.0, 1.15,
    "Threshold / decision\n"
    "benchmark → choose_window_threshold (balanced)\n"
    "production → choose_engine_setting (early-alarm + moving-avg + persistence)",
    COLORS["thr"], fontsize=8.8)

# Output annotation – placed below the diagram
ax.text(7, -0.55, "Outputs:  P · R · F1 · F-beta · AUC      |      save_artifacts(encoder.pt, xgb_*.ubj, scaler)",
        ha="center", va="center", fontsize=8.8, style="italic",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=COLORS["out"], edgecolor=EDGE))
ax.set_ylim(-1.2, 22)

# Legend
legend_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["prep"],
           markeredgecolor=EDGE, markersize=12, label="Data / Preprocessing"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["tcn"],
           markeredgecolor=EDGE, markersize=12, label="Deep encoder (TCN/BiGRU/Attn)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["sem"],
           markeredgecolor=EDGE, markersize=12, label="Semantic statistics"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["xgb"],
           markeredgecolor=EDGE, markersize=12, label="XGBoost ensemble"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["thr"],
           markeredgecolor=EDGE, markersize=12, label="Calibration / Decision"),
]
ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, -0.085), ncol=5, frameon=False, fontsize=8.5)

out_path = PROJECT_ROOT / "assets" / "images" / "architecture_diagram.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print("Saved:", out_path)
