from __future__ import annotations

import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "article_assets"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]


DATASET_META = {
    "FD001": {"conditions": 1, "fault_modes": 1, "fault_type": "HPC degradation"},
    "FD002": {"conditions": 6, "fault_modes": 1, "fault_type": "HPC degradation"},
    "FD003": {"conditions": 1, "fault_modes": 2, "fault_type": "HPC and fan degradation"},
    "FD004": {"conditions": 6, "fault_modes": 2, "fault_type": "HPC and fan degradation"},
}


# Reported notebook output: tcn_bigru_xgboost_cmapss.ipynb
PROPOSED = {
    "FD001": {
        "f1": 0.8889,
        "auc": 0.9968,
        "pr_auc": 0.9902,
        "precision": 1.0000,
        "recall": 0.8000,
        "specificity": 1.0000,
        "support_normal": 75,
        "support_anomaly": 25,
        "tp": 20,
        "fp": 0,
        "fn": 5,
        "tn": 75,
    },
    "FD002": {
        "f1": 0.9302,
        "auc": 0.9980,
        "pr_auc": 0.9938,
        "precision": 0.8696,
        "recall": 1.0000,
        "specificity": 0.9548,
        "support_normal": 199,
        "support_anomaly": 60,
        "tp": 60,
        "fp": 9,
        "fn": 0,
        "tn": 190,
    },
    "FD003": {
        "f1": 0.9231,
        "auc": 0.9988,
        "pr_auc": 0.9951,
        "precision": 0.9474,
        "recall": 0.9000,
        "specificity": 0.9875,
        "support_normal": 80,
        "support_anomaly": 20,
        "tp": 18,
        "fp": 1,
        "fn": 2,
        "tn": 79,
    },
    "FD004": {
        "f1": 0.8932,
        "auc": 0.9826,
        "pr_auc": 0.9469,
        "precision": 0.9020,
        "recall": 0.8846,
        "specificity": 0.9745,
        "support_normal": 196,
        "support_anomaly": 52,
        "tp": 46,
        "fp": 5,
        "fn": 6,
        "tn": 191,
    },
}


# Reported notebook outputs from comparison notebooks.
COMPARISON_F1 = {
    "TCN-GRU + XGBoost": {
        "FD001": 0.8095,
        "FD002": 0.8485,
        "FD003": 0.9500,
        "FD004": 0.7556,
    },
    "PatchTST + XGBoost": {
        "FD001": 0.9200,
        "FD002": 0.8421,
        "FD003": 0.9268,
        "FD004": 0.8269,
    },
    "ABGRU": {
        "FD001": 0.8511,
        "FD002": 0.9587,
        "FD003": 0.9756,
        "FD004": 0.8421,
    },
    "TCN-BiGRU + XGBoost": {
        "FD001": 0.8889,
        "FD002": 0.9302,
        "FD003": 0.9231,
        "FD004": 0.8932,
    },
}


COMPARISON_AUC = {
    "TCN-GRU + XGBoost": {
        "FD001": 0.9883,
        "FD002": 0.9674,
        "FD003": 0.9931,
        "FD004": 0.9653,
    },
    "PatchTST + XGBoost": {
        "FD001": 0.9915,
        "FD002": 0.9809,
        "FD003": 0.9956,
        "FD004": 0.9709,
    },
    "TCN-BiGRU + XGBoost": {
        "FD001": 0.9968,
        "FD002": 0.9980,
        "FD003": 0.9988,
        "FD004": 0.9826,
    },
}


ABGRU_REGRESSION = {
    "FD001": {"rmse": 13.54, "mae": 10.32, "r2": 0.8938, "score": 279.85},
    "FD002": {"rmse": 31.69, "mae": 21.36, "r2": 0.6527, "score": 25719.09},
    "FD003": {"rmse": 13.36, "mae": 9.73, "r2": 0.8958, "score": 242.97},
    "FD004": {"rmse": 39.66, "mae": 26.05, "r2": 0.4709, "score": 1126946.50},
}


ARCHITECTURE_COMPARISON = [
    {
        "component": "Topology",
        "tcn_gru": "Parallel TCN and GRU branches",
        "tcn_bigru": "Sequential TCN -> BiGRU -> attention",
    },
    {
        "component": "TCN normalization",
        "tcn_gru": "BatchNorm1d",
        "tcn_bigru": "WeightNorm",
    },
    {
        "component": "TCN channels",
        "tcn_gru": "[64, 128, 128]",
        "tcn_bigru": "[64, 128, 128, 64]",
    },
    {
        "component": "Recurrent unit",
        "tcn_gru": "Bidirectional GRU, hidden 64",
        "tcn_bigru": "Bidirectional GRU, hidden 128",
    },
    {
        "component": "Window length",
        "tcn_gru": "30 cycles",
        "tcn_bigru": "40 cycles",
    },
    {
        "component": "Deep embedding",
        "tcn_gru": "128 dimensions",
        "tcn_bigru": "128 dimensions",
    },
    {
        "component": "XGBoost input",
        "tcn_gru": "Embedding + handcrafted statistics",
        "tcn_bigru": "Embedding only",
    },
    {
        "component": "Early stopping",
        "tcn_gru": "Validation loss",
        "tcn_bigru": "Validation F1",
    },
]


HYPERPARAMETERS = [
    ("Sequence length", "40 cycles"),
    ("Batch size", "256"),
    ("Epochs", "60"),
    ("Patience", "12"),
    ("Learning rate", "1e-3"),
    ("Weight decay", "1e-4"),
    ("Focal gamma", "2.0"),
    ("TCN channels", "[64, 128, 128, 64]"),
    ("GRU hidden size", "128 bidirectional"),
    ("GRU layers", "2"),
    ("GRU dropout", "0.30"),
    ("Embedding size", "128"),
    ("XGBoost rounds", "800"),
    ("XGBoost learning rate", "0.03"),
    ("XGBoost max depth", "6"),
    ("XGBoost subsample", "0.8"),
    ("XGBoost colsample_bytree", "0.8"),
    ("XGBoost min_child_weight", "5"),
    ("XGBoost early stopping", "40 rounds"),
]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def escape_tex(value: object) -> str:
    text = fmt(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    text = text.replace("->", r"$\rightarrow$")
    return text


def write_table(name: str, headers: list[str], rows: list[list[object]], caption: str, label: str) -> None:
    csv_path = TABLE_DIR / f"{name}.csv"
    md_path = TABLE_DIR / f"{name}.md"
    tex_path = TABLE_DIR / f"{name}.tex"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        md.append("| " + " | ".join(fmt(x) for x in row) + " |")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    colspec = "l" + "c" * (len(headers) - 1)
    tex = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{escape_tex(caption)}}}",
        rf"\label{{{escape_tex(label)}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(escape_tex(h) for h in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        tex.append(" & ".join(escape_tex(x) for x in row) + r" \\")
    tex.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    tex_path.write_text("\n".join(tex), encoding="utf-8")


def read_txt_stats(fd: str) -> dict[str, object]:
    data_dir = ROOT / "data"
    row = {"Dataset": fd, **DATASET_META[fd]}
    for split in ["train", "test"]:
        path = data_dir / f"{split}_{fd}.txt"
        engines: dict[int, int] = {}
        rows = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                uid = int(float(parts[0]))
                engines[uid] = engines.get(uid, 0) + 1
                rows += 1
        counts = list(engines.values())
        row[f"{split}_engines"] = len(engines)
        row[f"{split}_samples"] = rows
        row[f"{split}_min_cycles"] = min(counts)
        row[f"{split}_max_cycles"] = max(counts)
        row[f"{split}_avg_cycles"] = round(sum(counts) / len(counts), 1)
    return row


def collect_dataset_stats() -> list[dict[str, object]]:
    return [read_txt_stats(fd) for fd in DATASETS]


def svg_header(w: int, h: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}",
        ".title{font-size:22px;font-weight:700}",
        ".axis{font-size:12px}",
        ".label{font-size:13px}",
        ".small{font-size:11px}",
        ".box{fill:#f8fafc;stroke:#1f2933;stroke-width:1.4}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def save_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + ["</svg>", ""]), encoding="utf-8")


def grouped_bar_svg(
    path: Path,
    title: str,
    categories: list[str],
    series: dict[str, list[float]],
    y_max: float = 1.0,
    y_label: str = "Score",
) -> None:
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"]
    w, h = 960, 560
    left, right, top, bottom = 78, 36, 64, 96
    plot_w, plot_h = w - left - right, h - top - bottom
    lines = svg_header(w, h)
    lines.append(f'<text x="{w/2}" y="34" text-anchor="middle" class="title">{html.escape(title)}</text>')
    lines.append(f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" class="axis">{html.escape(y_label)}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>')
    for i in range(6):
        val = y_max * i / 5
        y = top + plot_h - (val / y_max) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{val:.1f}</text>')

    n_cat = len(categories)
    n_ser = len(series)
    group_w = plot_w / n_cat
    bar_w = min(32, group_w * 0.72 / n_ser)
    for ci, cat in enumerate(categories):
        cx = left + group_w * ci + group_w / 2
        group_start = cx - (bar_w * n_ser) / 2
        for si, (name, values) in enumerate(series.items()):
            val = values[ci]
            bar_h = (val / y_max) * plot_h
            x = group_start + si * bar_w
            y = top + plot_h - bar_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{bar_h:.1f}" fill="{colors[si % len(colors)]}"/>')
            lines.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 4:.1f}" text-anchor="middle" class="small">{val:.3f}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="label">{html.escape(cat)}</text>')

    legend_x = left
    legend_y = h - 42
    offset = 0
    for si, name in enumerate(series.keys()):
        x = legend_x + offset
        lines.append(f'<rect x="{x}" y="{legend_y}" width="13" height="13" fill="{colors[si % len(colors)]}"/>')
        lines.append(f'<text x="{x + 18}" y="{legend_y + 11}" class="small">{html.escape(name)}</text>')
        offset += 18 + len(name) * 7
    save_svg(path, lines)


def architecture_svg(path: Path) -> None:
    w, h = 1060, 420
    lines = svg_header(w, h)
    lines.append(f'<text x="{w/2}" y="34" text-anchor="middle" class="title">TCN-BiGRU + XGBoost Pipeline</text>')
    boxes = [
        (40, 120, 130, 74, "Sensor window", "40 cycles x F"),
        (210, 120, 150, 74, "Causal TCN", "dilated convs"),
        (400, 120, 150, 74, "BiGRU", "2 layers, h=128"),
        (590, 120, 140, 74, "Attention", "temporal pooling"),
        (770, 120, 120, 74, "Embedding", "128-D"),
        (930, 120, 90, 74, "XGBoost", "failure prob."),
    ]
    for x, y, bw, bh, title, sub in boxes:
        lines.append(f'<rect x="{x}" y="{y}" rx="6" ry="6" width="{bw}" height="{bh}" class="box"/>')
        lines.append(f'<text x="{x + bw/2}" y="{y + 31}" text-anchor="middle" class="label" font-weight="700">{html.escape(title)}</text>')
        lines.append(f'<text x="{x + bw/2}" y="{y + 52}" text-anchor="middle" class="small">{html.escape(sub)}</text>')
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        y1 = boxes[i][1] + boxes[i][3] / 2
        x2 = boxes[i + 1][0]
        lines.append(f'<line x1="{x1 + 8}" y1="{y1}" x2="{x2 - 12}" y2="{y1}" stroke="#111827" stroke-width="1.8" marker-end="url(#arrow)"/>')
    lines.insert(
        10,
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#111827"/></marker></defs>',
    )
    lines.append('<text x="40" y="260" class="label" font-weight="700">Training objective</text>')
    lines.append('<text x="40" y="286" class="label">Focal-loss deep classifier is trained first; the learned embedding is then used by XGBoost.</text>')
    lines.append('<text x="40" y="326" class="label" font-weight="700">Decision rule</text>')
    lines.append('<text x="40" y="352" class="label">A validation-optimized probability threshold converts XGBoost scores into normal vs near-failure labels.</text>')
    save_svg(path, lines)


def detailed_architecture_svg(path: Path) -> None:
    w, h = 1380, 820
    lines = svg_header(w, h)
    lines.insert(
        10,
        '<defs><marker id="arrow-detail" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#111827"/></marker></defs>',
    )
    lines.append(f'<text x="{w/2}" y="36" text-anchor="middle" class="title">Detailed TCN-BiGRU + XGBoost Architecture</text>')
    lines.append('<text x="690" y="62" text-anchor="middle" class="small">Based on tcn_bigru_xgboost_cmapss.ipynb</text>')

    def box(x, y, bw, bh, title, sub="", fill="#f8fafc", stroke="#1f2933"):
        lines.append(f'<rect x="{x}" y="{y}" rx="7" ry="7" width="{bw}" height="{bh}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        lines.append(f'<text x="{x + bw/2}" y="{y + 28}" text-anchor="middle" class="label" font-weight="700">{html.escape(title)}</text>')
        if sub:
            for i, part in enumerate(sub.split("\\n")):
                lines.append(f'<text x="{x + bw/2}" y="{y + 50 + i*17}" text-anchor="middle" class="small">{html.escape(part)}</text>')

    def arrow(x1, y1, x2, y2):
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111827" stroke-width="1.8" marker-end="url(#arrow-detail)"/>')

    # Data and preprocessing
    box(36, 115, 158, 92, "Raw CMAPSS", "train/test/RUL files\nFD001-FD004", "#eef2ff")
    box(236, 115, 190, 92, "RUL + Labels", "RUL=max_cycle-cycle\nnear failure: RUL&lt;30", "#eef2ff")
    box(468, 115, 200, 92, "Preprocessing", "valid sensor/op features\nstandardized values", "#eef2ff")
    box(710, 115, 186, 92, "Sliding Windows", "seq_len=40, stride=1\nlabel from final cycle", "#eef2ff")
    arrow(194, 161, 236, 161)
    arrow(426, 161, 468, 161)
    arrow(668, 161, 710, 161)

    # Deep encoder panel
    lines.append('<rect x="36" y="255" width="1010" height="330" rx="10" ry="10" fill="#ffffff" stroke="#64748b" stroke-dasharray="6 4"/>')
    lines.append('<text x="54" y="282" class="label" font-weight="700">Deep temporal encoder trained with focal loss</text>')

    box(64, 326, 132, 92, "Input tensor", "B x 40 x F", "#f0fdf4")
    box(235, 326, 155, 92, "Permute", "B x F x 40\nfor Conv1D", "#f0fdf4")
    box(430, 294, 172, 150, "Causal TCN", "4 residual blocks\nkernel=3, dropout=0.2\nWeightNorm convs\ndilations=1,2,4,8", "#ecfeff")
    box(642, 294, 156, 150, "BiGRU", "2 layers\nhidden=128\nbidirectional\noutput dim=256", "#fff7ed")
    box(838, 326, 150, 92, "Attention Pool", "Linear(256,1)\nsoftmax over time", "#fef2f2")
    arrow(196, 372, 235, 372)
    arrow(390, 372, 430, 372)
    arrow(602, 372, 642, 372)
    arrow(798, 372, 838, 372)

    # TCN internals
    tcn_x, tcn_y = 442, 468
    for i, (ch_in, ch_out, dil) in enumerate([("F", "64", 1), ("64", "128", 2), ("128", "128", 4), ("128", "64", 8)]):
        x = tcn_x + i * 138
        box(x, tcn_y, 112, 72, f"TCN block {i+1}", f"{ch_in}->{ch_out}\nd={dil}", "#ecfeff")
        if i < 3:
            arrow(x + 112, tcn_y + 36, x + 138, tcn_y + 36)
    lines.append('<text x="704" y="560" text-anchor="middle" class="small">Each block: WeightNorm Conv1D -> ReLU -> Dropout -> WeightNorm Conv1D -> ReLU -> Dropout + residual</text>')

    # Projection and XGBoost
    box(1080, 296, 190, 96, "Projection", "Linear(256,128)\nLayerNorm + GELU\nDropout=0.2", "#f5f3ff")
    box(1080, 436, 190, 86, "Encoder Head", "Linear(128,1)\nused during training", "#f5f3ff")
    arrow(988, 372, 1080, 344)
    arrow(1175, 392, 1175, 436)
    lines.append('<text x="1168" y="566" text-anchor="middle" class="small">Best encoder selected by validation F2</text>')

    box(64, 640, 180, 92, "Embedding Extraction", "model.embed(x)\n128-D vectors", "#f8fafc")
    box(292, 640, 196, 92, "GridSearchCV", "3-fold search\nXGBoost params", "#f8fafc")
    box(536, 640, 190, 92, "XGBoost Train", "binary:logistic\nhist tree method\nscale_pos_weight", "#f8fafc")
    box(774, 640, 198, 92, "Threshold Search", "validation probability\nF-beta, beta=2\nmin_precision=0.35", "#f8fafc")
    box(1020, 640, 210, 92, "Final Decision", "normal / near failure\nmetrics on test-last", "#f8fafc")
    arrow(244, 686, 292, 686)
    arrow(488, 686, 536, 686)
    arrow(726, 686, 774, 686)
    arrow(972, 686, 1020, 686)
    arrow(1175, 522, 154, 640)

    # Config note
    box(940, 110, 330, 108, "Key Configuration", "epochs=60, batch_size=256, lr=1e-3\nweight_decay=1e-4, patience=12\nfocal_gamma=2.5, pos_weight_boost=2.0\nXGBoost rounds=800, early stop=40", "#fffbeb", "#92400e")
    save_svg(path, lines)


def dataset_distribution_svg(path: Path, stats: list[dict[str, object]]) -> None:
    series = {
        "Train engines": [float(s["train_engines"]) for s in stats],
        "Test engines": [float(s["test_engines"]) for s in stats],
    }
    grouped_bar_svg(path, "CMAPSS Engine Counts by Dataset", DATASETS, series, y_max=280, y_label="Engines")


def confusion_svg(path: Path) -> None:
    w, h = 980, 620
    lines = svg_header(w, h)
    lines.append(f'<text x="{w/2}" y="34" text-anchor="middle" class="title">TCN-BiGRU + XGBoost Confusion Matrices</text>')
    cell = 64
    gap_x, gap_y = 245, 250
    start_x, start_y = 90, 92
    max_count = max(max(m["tn"], m["fp"], m["fn"], m["tp"]) for m in PROPOSED.values())
    labels = [("TN", "Normal->Normal"), ("FP", "Normal->Near failure"), ("FN", "Near failure->Normal"), ("TP", "Near failure->Near failure")]
    for idx, fd in enumerate(DATASETS):
        x0 = start_x + (idx % 2) * gap_x
        y0 = start_y + (idx // 2) * gap_y
        m = PROPOSED[fd]
        values = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        lines.append(f'<text x="{x0 + cell}" y="{y0 - 22}" text-anchor="middle" class="label" font-weight="700">{fd}</text>')
        lines.append(f'<text x="{x0 + cell}" y="{y0 + cell*2 + 34}" text-anchor="middle" class="small">Predicted class</text>')
        lines.append(f'<text x="{x0 - 34}" y="{y0 + cell}" transform="rotate(-90 {x0 - 34} {y0 + cell})" class="small">True class</text>')
        for r in range(2):
            for c in range(2):
                val = values[r][c]
                intensity = 245 - int(145 * math.sqrt(val / max_count))
                fill = f"rgb({intensity},{intensity + 8},{255})"
                x = x0 + c * cell
                y = y0 + r * cell
                lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#1f2933"/>')
                lines.append(f'<text x="{x + cell/2}" y="{y + 27}" text-anchor="middle" class="small">{labels[r*2+c][0]}</text>')
                lines.append(f'<text x="{x + cell/2}" y="{y + 49}" text-anchor="middle" class="label" font-weight="700">{val}</text>')
        lines.append(f'<text x="{x0 + cell/2}" y="{y0 - 5}" text-anchor="middle" class="small">Normal</text>')
        lines.append(f'<text x="{x0 + 1.5*cell}" y="{y0 - 5}" text-anchor="middle" class="small">Near fail</text>')
        lines.append(f'<text x="{x0 - 8}" y="{y0 + cell/2}" text-anchor="end" class="small">Normal</text>')
        lines.append(f'<text x="{x0 - 8}" y="{y0 + 1.5*cell}" text-anchor="end" class="small">Near fail</text>')
    lines.append('<text x="610" y="108" class="label" font-weight="700">Labels</text>')
    for i, (short, long) in enumerate(labels):
        lines.append(f'<text x="610" y="{136 + i*26}" class="small">{short}: {html.escape(long)}</text>')
    save_svg(path, lines)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def write_report(stats: list[dict[str, object]]) -> None:
    proposed_mean_f1 = mean([PROPOSED[fd]["f1"] for fd in DATASETS])
    proposed_mean_auc = mean([PROPOSED[fd]["auc"] for fd in DATASETS])
    proposed_mean_ap = mean([PROPOSED[fd]["pr_auc"] for fd in DATASETS])
    best_fd = max(DATASETS, key=lambda fd: PROPOSED[fd]["f1"])
    hardest_fd = min(DATASETS, key=lambda fd: PROPOSED[fd]["f1"])
    comparison_means = {
        name: mean([scores[fd] for fd in DATASETS]) for name, scores in COMPARISON_F1.items()
    }
    report = f"""# IEEE Article Result Report: TCN-BiGRU + XGBoost

## Scope

This package summarizes the CMAPSS near-failure detection results for the TCN-BiGRU + XGBoost method. The binary target is near failure when RUL < 30 cycles and normal otherwise. The reported model first learns a temporal representation using causal TCN blocks, bidirectional GRU layers, and attention pooling; the 128-dimensional embedding is then passed to an XGBoost classifier.

## Dataset

The experiments use all four NASA CMAPSS subsets: FD001, FD002, FD003, and FD004. FD001 and FD003 contain one operating condition, while FD002 and FD004 contain six operating conditions. FD003 and FD004 are more complex because they contain two fault modes instead of one. Dataset counts are generated directly from the local files in `data/`.

Use `tables/dataset_characteristics.tex` for the IEEE dataset table and `figures/fig_dataset_engine_counts.svg` for an engine-count plot.

## Main Results

Across all four subsets, TCN-BiGRU + XGBoost obtained mean F1 = {proposed_mean_f1:.4f}, mean ROC-AUC = {proposed_mean_auc:.4f}, and mean PR-AUC = {proposed_mean_ap:.4f}. The highest F1 was observed on {best_fd} ({PROPOSED[best_fd]['f1']:.4f}), while the lowest F1 was observed on {hardest_fd} ({PROPOSED[hardest_fd]['f1']:.4f}). FD004 remained challenging in ranking quality and average precision because it combines multiple operating conditions with two fault modes.

The model produced high ROC-AUC on every subset: 0.9968 on FD001, 0.9980 on FD002, 0.9988 on FD003, and 0.9826 on FD004. This indicates strong ranking quality even on the harder multi-condition subsets. Precision and recall show the expected safety trade-off: FD001 reached perfect precision with lower recall, whereas FD002 reached perfect recall with moderate false positives.

## Comparison

The comparison table uses test-last F1 values reported in the existing notebooks. Mean F1 values are:

- TCN-GRU + XGBoost: {comparison_means['TCN-GRU + XGBoost']:.4f}
- PatchTST + XGBoost: {comparison_means['PatchTST + XGBoost']:.4f}
- ABGRU: {comparison_means['ABGRU']:.4f}
- TCN-BiGRU + XGBoost: {comparison_means['TCN-BiGRU + XGBoost']:.4f}

The proposed TCN-BiGRU + XGBoost has the strongest average F1 among the compared methods, although ABGRU is very close and is strongest on FD002 and FD003. This should be written carefully in the paper: the proposed method improves average cross-dataset F1, not every individual subset.

## Suggested IEEE Result Paragraph

Table III reports the near-failure classification results of the TCN-BiGRU + XGBoost model on all CMAPSS subsets. The method achieved an average F1-score of {proposed_mean_f1:.4f} and an average ROC-AUC of {proposed_mean_auc:.4f}. The best subset-level F1-score was obtained on {best_fd}, while FD004 remained the most challenging subset due to the combination of six operating conditions and two fault modes. Compared with TCN-GRU + XGBoost and PatchTST + XGBoost baselines, the proposed sequential TCN-BiGRU representation achieved the best mean F1-score across the four datasets. These results suggest that extracting local temporal degradation patterns with TCN layers before bidirectional recurrent modeling provides a robust representation for failure detection.

## Suggested Figure Captions

Fig. 1. Overall architecture of the TCN-BiGRU + XGBoost near-failure detection pipeline.

Fig. 2. Cross-dataset precision, recall, F1-score, ROC-AUC, PR-AUC, and specificity of the proposed method.

Fig. 3. Test-last F1-score comparison across CMAPSS subsets for TCN-GRU + XGBoost, PatchTST + XGBoost, ABGRU, and TCN-BiGRU + XGBoost.

Fig. 4. Confusion matrices of the proposed method on the final test sequence of each engine.

## Notes for Submission

The numbers in this report come from executed notebook outputs in this repository. Before final IEEE submission, re-run the final notebook or script once with fixed seeds and save the console output, because GPU libraries and XGBoost versions can slightly change results.
"""
    (OUT / "ieee_results_report.md").write_text(report, encoding="utf-8")


def write_index() -> None:
    figures = sorted(FIG_DIR.glob("*.svg"))
    tables = sorted(TABLE_DIR.glob("*.md"))
    fig_html = "\n".join(
        f'<section><h2>{html.escape(fig.stem)}</h2><img src="figures/{html.escape(fig.name)}" alt="{html.escape(fig.stem)}"></section>'
        for fig in figures
    )
    table_html = "\n".join(
        f'<li><a href="tables/{html.escape(table.name)}">{html.escape(table.name)}</a></li>'
        for table in tables
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IEEE Article Assets</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 32px; color: #17202a; }}
    h1 {{ margin-bottom: 4px; }}
    section {{ margin: 32px 0; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <h1>IEEE Article Assets</h1>
  <p>Generated figures and links to Markdown/CSV/LaTeX tables.</p>
  <h2>Tables</h2>
  <ul>
    {table_html}
  </ul>
  {fig_html}
</body>
</html>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    stats = collect_dataset_stats()

    dataset_rows = [
        [
            s["Dataset"],
            s["train_engines"],
            s["test_engines"],
            s["train_samples"],
            s["test_samples"],
            s["conditions"],
            s["fault_modes"],
            s["fault_type"],
        ]
        for s in stats
    ]
    write_table(
        "dataset_characteristics",
        ["Dataset", "Train engines", "Test engines", "Train samples", "Test samples", "Conditions", "Fault modes", "Fault type"],
        dataset_rows,
        "NASA CMAPSS dataset characteristics.",
        "tab:dataset_characteristics",
    )

    proposed_rows = [
        [
            fd,
            PROPOSED[fd]["precision"],
            PROPOSED[fd]["recall"],
            PROPOSED[fd]["f1"],
            PROPOSED[fd]["auc"],
            PROPOSED[fd]["pr_auc"],
            PROPOSED[fd]["specificity"],
            f'{PROPOSED[fd]["tp"]}/{PROPOSED[fd]["fp"]}/{PROPOSED[fd]["fn"]}',
        ]
        for fd in DATASETS
    ]
    proposed_rows.append(
        [
            "Mean",
            mean([PROPOSED[fd]["precision"] for fd in DATASETS]),
            mean([PROPOSED[fd]["recall"] for fd in DATASETS]),
            mean([PROPOSED[fd]["f1"] for fd in DATASETS]),
            mean([PROPOSED[fd]["auc"] for fd in DATASETS]),
            mean([PROPOSED[fd]["pr_auc"] for fd in DATASETS]),
            mean([PROPOSED[fd]["specificity"] for fd in DATASETS]),
            "-",
        ]
    )
    write_table(
        "proposed_tcn_bigru_xgboost_results",
        ["Dataset", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Specificity", "TP/FP/FN"],
        proposed_rows,
        "Near-failure detection results of TCN-BiGRU + XGBoost.",
        "tab:proposed_results",
    )

    comparison_rows = []
    for method, scores in COMPARISON_F1.items():
        comparison_rows.append([method] + [scores[fd] for fd in DATASETS] + [mean([scores[fd] for fd in DATASETS])])
    write_table(
        "method_comparison_f1",
        ["Method", "FD001", "FD002", "FD003", "FD004", "Mean"],
        comparison_rows,
        "Test-last F1-score comparison across CMAPSS subsets.",
        "tab:method_comparison_f1",
    )

    auc_rows = []
    for method, scores in COMPARISON_AUC.items():
        auc_rows.append([method] + [scores[fd] for fd in DATASETS] + [mean([scores[fd] for fd in DATASETS])])
    write_table(
        "method_comparison_auc",
        ["Method", "FD001", "FD002", "FD003", "FD004", "Mean"],
        auc_rows,
        "Test-last ROC-AUC comparison across CMAPSS subsets.",
        "tab:method_comparison_auc",
    )

    abgru_rows = [
        [fd, ABGRU_REGRESSION[fd]["rmse"], ABGRU_REGRESSION[fd]["mae"], ABGRU_REGRESSION[fd]["r2"], ABGRU_REGRESSION[fd]["score"]]
        for fd in DATASETS
    ]
    write_table(
        "abgru_regression_results",
        ["Dataset", "RMSE", "MAE", "R2", "PHM score"],
        abgru_rows,
        "ABGRU RUL regression results reported in the existing notebook.",
        "tab:abgru_regression",
    )

    arch_rows = [[x["component"], x["tcn_gru"], x["tcn_bigru"]] for x in ARCHITECTURE_COMPARISON]
    write_table(
        "architecture_comparison",
        ["Component", "TCN-GRU + XGBoost", "TCN-BiGRU + XGBoost"],
        arch_rows,
        "Architectural differences between the prior TCN-GRU baseline and TCN-BiGRU method.",
        "tab:architecture_comparison",
    )

    write_table(
        "tcn_bigru_hyperparameters",
        ["Hyperparameter", "Value"],
        [[k, v] for k, v in HYPERPARAMETERS],
        "Main hyperparameters of TCN-BiGRU + XGBoost.",
        "tab:hyperparameters",
    )

    architecture_svg(FIG_DIR / "fig_tcn_bigru_xgboost_architecture.svg")
    detailed_architecture_svg(FIG_DIR / "fig_tcn_bigru_xgboost_architecture_detailed.svg")
    dataset_distribution_svg(FIG_DIR / "fig_dataset_engine_counts.svg", stats)
    grouped_bar_svg(
        FIG_DIR / "fig_proposed_cross_dataset_metrics.svg",
        "TCN-BiGRU + XGBoost Cross-Dataset Metrics",
        DATASETS,
        {
            "Precision": [PROPOSED[fd]["precision"] for fd in DATASETS],
            "Recall": [PROPOSED[fd]["recall"] for fd in DATASETS],
            "F1": [PROPOSED[fd]["f1"] for fd in DATASETS],
            "ROC-AUC": [PROPOSED[fd]["auc"] for fd in DATASETS],
            "PR-AUC": [PROPOSED[fd]["pr_auc"] for fd in DATASETS],
            "Specificity": [PROPOSED[fd]["specificity"] for fd in DATASETS],
        },
        y_max=1.05,
        y_label="Score",
    )
    grouped_bar_svg(
        FIG_DIR / "fig_method_comparison_f1.svg",
        "Test-Last F1 Comparison",
        DATASETS,
        {name: [scores[fd] for fd in DATASETS] for name, scores in COMPARISON_F1.items()},
        y_max=1.05,
        y_label="F1-score",
    )
    grouped_bar_svg(
        FIG_DIR / "fig_method_comparison_auc.svg",
        "Test-Last ROC-AUC Comparison",
        DATASETS,
        {name: [scores[fd] for fd in DATASETS] for name, scores in COMPARISON_AUC.items()},
        y_max=1.05,
        y_label="ROC-AUC",
    )
    confusion_svg(FIG_DIR / "fig_proposed_confusion_matrices.svg")
    write_report(stats)
    write_index()

    all_tex = []
    for tex_path in sorted(TABLE_DIR.glob("*.tex")):
        all_tex.append(f"% {tex_path.name}")
        all_tex.append(tex_path.read_text(encoding="utf-8"))
    (OUT / "ieee_tables_all.tex").write_text("\n".join(all_tex), encoding="utf-8")

    print(f"Wrote article assets to {OUT}")


if __name__ == "__main__":
    main()
