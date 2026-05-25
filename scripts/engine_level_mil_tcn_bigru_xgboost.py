import copy
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMNS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


DEFAULT_CFG = {
    "seed": 42,
    "data_dir": "data",
    "dataset": "FD004",
    "rul_threshold": 30,
    "seq_len": 40,
    "window_stride": 1,
    "val_engine_frac": 0.20,
    "snapshot_stride": 10,
    "bag_window_stride": 2,
    "max_windows_per_bag": 60,
    "recent_windows_only": True,
    "use_semantic_features": True,
    "tcn_channels": [64, 128, 128, 64],
    "tcn_kernel": 3,
    "tcn_dropout": 0.20,
    "gru_hidden": 128,
    "gru_layers": 2,
    "gru_dropout": 0.30,
    "embed_dim": 128,
    "epochs": 60,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 12,
    "min_delta": 1e-4,
    "focal_gamma": 2.5,
    "recon_weight": 0.30,
    "pos_weight_boost": 2.0,
    "recall_beta": 2.0,
    "use_amp": True,
    "min_precision": 0.70,
    "f1_tolerance": 0.005,
    "threshold_grid": (0.05, 0.95, 0.0025),
    "grid_cv_folds": 3,
    "grid_n_jobs": -1,
    "xgb_n_jobs": 1,
    "xgb_estimators": 500,
    "xgb_param_grid": {
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.10],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 2.0],
    },
}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_dir(data_dir):
    data_path = Path(data_dir)
    candidates = [
        data_path,
        Path.cwd() / data_path,
        PROJECT_ROOT / data_path,
        Path("/content/drive/MyDrive/data"),
        Path("/content/data"),
    ]
    for path in candidates:
        if (path / "train_FD001.txt").exists():
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError("CMAPSS data directory not found. Checked: " + checked)


def load_raw(fd, data_dir):
    data_dir = resolve_data_dir(data_dir)
    train = pd.read_csv(data_dir / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(data_dir / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(data_dir / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["RUL"])
    return train, test, rul


def add_train_rul(df):
    max_cycle = df.groupby("unit_number")["time_in_cycles"].max()
    out = df.join(max_cycle.rename("max_cycle"), on="unit_number")
    out["RUL"] = out["max_cycle"] - out["time_in_cycles"]
    return out.drop(columns=["max_cycle"])


def add_test_rul(test_df, rul_df):
    out = test_df.copy()
    max_cycle = out.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycle.columns = ["unit_number", "max_cycle"]
    rul = rul_df.copy()
    rul["unit_number"] = np.arange(1, len(rul) + 1)
    max_cycle = max_cycle.merge(rul, on="unit_number", how="left")
    max_cycle["final_cycle"] = max_cycle["max_cycle"] + max_cycle["RUL"]
    out = out.merge(max_cycle[["unit_number", "final_cycle"]], on="unit_number", how="left")
    out["RUL"] = out["final_cycle"] - out["time_in_cycles"]
    return out.drop(columns=["final_cycle"])


def get_feature_columns(df):
    drop = {"unit_number", "time_in_cycles", "RUL", "label", *DROP_SENSORS}
    return [col for col in df.columns if col not in drop]


def split_train_val_by_engine(train_df, val_frac, seed):
    units = np.array(sorted(train_df["unit_number"].unique()))
    rng = np.random.RandomState(seed)
    val_size = max(1, int(round(len(units) * val_frac)))
    val_units = set(rng.choice(units, size=val_size, replace=False).tolist())
    val_mask = train_df["unit_number"].isin(val_units)
    return train_df.loc[~val_mask].copy(), train_df.loc[val_mask].copy()


def make_sliding_windows(df, feat_cols, seq_len, stride, threshold):
    windows, labels = [], []
    for _, group in df.groupby("unit_number"):
        group = group.sort_values("time_in_cycles")
        arr = group[feat_cols].values.astype(np.float32)
        ruls = group["RUL"].values.astype(np.float32)
        if len(arr) < seq_len:
            continue
        for start in range(0, len(arr) - seq_len + 1, stride):
            end = start + seq_len
            windows.append(arr[start:end])
            labels.append(int(ruls[end - 1] <= threshold))
    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def make_engine_arrays(df, feat_cols):
    arrays = {}
    for unit, group in df.groupby("unit_number"):
        group = group.sort_values("time_in_cycles")
        arrays[int(unit)] = {
            "x": group[feat_cols].values.astype(np.float32),
            "rul": group["RUL"].values.astype(np.float32),
            "cycle": group["time_in_cycles"].values.astype(np.float32),
        }
    return arrays


def make_snapshot_records(df, seq_len, threshold, snapshot_stride, one_per_engine=False):
    records = []
    for unit, group in df.groupby("unit_number"):
        group = group.sort_values("time_in_cycles")
        n = len(group)
        if n < seq_len:
            continue
        ruls = group["RUL"].values.astype(np.float32)
        cycles = group["time_in_cycles"].values.astype(np.float32)
        if one_per_engine:
            ends = [n]
        else:
            ends = list(range(seq_len, n + 1, snapshot_stride))
            if ends[-1] != n:
                ends.append(n)
        for end in ends:
            records.append(
                {
                    "unit_number": int(unit),
                    "end": int(end),
                    "cycle": float(cycles[end - 1]),
                    "RUL": float(ruls[end - 1]),
                    "label": int(ruls[end - 1] <= threshold),
                }
            )
    return pd.DataFrame(records)


def snapshot_windows(engine_arrays, unit, end, cfg):
    arr = engine_arrays[int(unit)]["x"]
    seq_len = cfg["seq_len"]
    if end < seq_len:
        raise ValueError("Snapshot end is shorter than seq_len.")
    starts = np.arange(0, end - seq_len + 1, cfg["bag_window_stride"])
    if len(starts) == 0:
        starts = np.asarray([end - seq_len])
    max_windows = cfg.get("max_windows_per_bag")
    if max_windows and len(starts) > max_windows:
        if cfg.get("recent_windows_only", True):
            starts = starts[-max_windows:]
        else:
            idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
            starts = starts[idx]
    return np.stack([arr[start : start + seq_len] for start in starts]).astype(np.float32)


def extract_semantic_features(windows):
    n, t_len, n_feat = windows.shape
    t = np.arange(t_len, dtype=np.float32)
    t_mean = t.mean()
    denom = max(float(np.sum((t - t_mean) ** 2)), 1e-9)
    x_mean = windows.mean(axis=1)
    x_std = windows.std(axis=1)
    x_min = windows.min(axis=1)
    x_max = windows.max(axis=1)
    x_range = x_max - x_min
    x_drift = windows[:, -1, :] - windows[:, 0, :]
    slope = ((t[None, :, None] - t_mean) * (windows - x_mean[:, None, :])).sum(axis=1) / denom
    diff = np.diff(windows, axis=1)
    diff_abs_mean = np.abs(diff).mean(axis=1)
    diff_std = diff.std(axis=1)
    energy = (windows**2).sum(axis=1) / t_len
    pos_frac = (diff > 0).mean(axis=1)
    neg_frac = (diff < 0).mean(axis=1)
    feats = np.stack(
        [
            x_mean,
            x_std,
            x_min,
            x_max,
            x_range,
            x_drift,
            slope,
            diff_abs_mean,
            diff_std,
            energy,
            pos_frac,
            neg_frac,
        ],
        axis=2,
    )
    return feats.reshape(n, n_feat * 12).astype(np.float32)


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm_conv(in_ch, out_ch, kernel_size, pad, dilation)
        self.conv2 = weight_norm_conv(out_ch, out_ch, kernel_size, pad, dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        t_len = x.size(2)
        out = self.activation(self.conv1(x)[:, :, :t_len])
        out = self.dropout(out)
        out = self.activation(self.conv2(out)[:, :, :t_len])
        out = self.dropout(out)
        res = x if self.residual is None else self.residual(x)
        return self.activation(out + res)


def weight_norm_conv(in_ch, out_ch, kernel_size, padding, dilation):
    conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
    if hasattr(nn.utils, "parametrizations") and hasattr(nn.utils.parametrizations, "weight_norm"):
        return nn.utils.parametrizations.weight_norm(conv)
    return nn.utils.weight_norm(conv)


class TCN(nn.Module):
    def __init__(self, in_ch, channels, kernel_size, dropout):
        super().__init__()
        layers = []
        cur = in_ch
        for idx, out_ch in enumerate(channels):
            layers.append(CausalConvBlock(cur, out_ch, kernel_size, 2**idx, dropout))
            cur = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.fc(x), dim=1)
        return (weights * x).sum(dim=1)


class TCNBiGRUAttention(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        self.tcn = TCN(n_features, cfg["tcn_channels"], cfg["tcn_kernel"], cfg["tcn_dropout"])
        tcn_out = cfg["tcn_channels"][-1]
        self.bigru = nn.GRU(
            input_size=tcn_out,
            hidden_size=cfg["gru_hidden"],
            num_layers=cfg["gru_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=cfg["gru_dropout"] if cfg["gru_layers"] > 1 else 0.0,
        )
        gru_out = cfg["gru_hidden"] * 2
        self.attn = AttentionPool(gru_out)
        self.proj = nn.Sequential(
            nn.Linear(gru_out, cfg["embed_dim"]),
            nn.LayerNorm(cfg["embed_dim"]),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.cls_head = nn.Linear(cfg["embed_dim"], 1)
        self.recon_head = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def embed(self, x):
        seq, _ = self.bigru(self.tcn(x))
        return self.proj(self.attn(seq))

    def forward(self, x):
        z = self.embed(x)
        return self.cls_head(z).squeeze(-1), self.recon_head(z), z


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        return ((1 - pt) ** self.gamma * bce).mean()


def make_loader(x, y=None, batch_size=256, shuffle=False):
    xt = torch.from_numpy(x).float()
    if y is None:
        ds = TensorDataset(xt)
    else:
        ds = TensorDataset(xt, torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def train_encoder(x_train, y_train, x_val, y_val, cfg, device):
    model = TCNBiGRUAttention(x_train.shape[2], cfg).to(device)
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    pos_weight = torch.tensor([neg / pos * cfg["pos_weight_boost"]], dtype=torch.float32, device=device)
    cls_loss = FocalLoss(gamma=cfg["focal_gamma"], pos_weight=pos_weight)
    recon_loss = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-5)
    train_loader = make_loader(x_train, y_train, cfg["batch_size"], shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg["batch_size"], shuffle=False)
    use_amp = bool(cfg["use_amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    wait = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, recon, _ = model(xb)
                loss = cls_loss(logits, yb) + cfg["recon_weight"] * recon_loss(recon, xb[:, -1, :])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, _, _ = model(xb.to(device))
                probs.extend(torch.sigmoid(logits).cpu().numpy())
                trues.extend(yb.numpy())
        score = fbeta_score(trues, (np.asarray(probs) >= 0.5).astype(int), beta=cfg["recall_beta"], zero_division=0)
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | val_F{cfg['recall_beta']:.0f}={score:.4f}")
        if score > best_score + cfg["min_delta"]:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print("Early stopping at epoch", epoch)
                break
    model.load_state_dict(best_state)
    return model, float(best_score)


@torch.no_grad()
def extract_embeddings(model, windows, cfg, device):
    model.eval()
    parts = []
    for (xb,) in make_loader(windows, None, batch_size=512, shuffle=False):
        parts.append(model.embed(xb.to(device)).cpu().numpy())
    return np.vstack(parts).astype(np.float32)


def window_feature_matrix(model, windows, cfg, device):
    emb = extract_embeddings(model, windows, cfg, device)
    if cfg["use_semantic_features"]:
        sem = extract_semantic_features(windows)
        return np.concatenate([emb, sem], axis=1).astype(np.float32)
    return emb.astype(np.float32)


def aggregate_bag_features(window_features):
    n = len(window_features)
    weights = np.linspace(1.0, 2.0, n, dtype=np.float32)
    weights = weights / weights.sum()
    weighted_mean = (window_features * weights[:, None]).sum(axis=0)
    slope = (window_features[-1] - window_features[0]) / max(n - 1, 1)
    return np.concatenate(
        [
            window_features[-1],
            window_features.mean(axis=0),
            window_features.std(axis=0),
            window_features.max(axis=0),
            window_features.min(axis=0),
            weighted_mean,
            slope,
            np.asarray([n], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_mil_features(model, engine_arrays, records, cfg, device, chunk_records=128):
    all_features = []
    labels = records["label"].values.astype(np.int64)
    groups = records["unit_number"].values.astype(np.int64)
    for start in range(0, len(records), chunk_records):
        chunk = records.iloc[start : start + chunk_records]
        windows_list, lengths = [], []
        for row in chunk.itertuples(index=False):
            win = snapshot_windows(engine_arrays, row.unit_number, row.end, cfg)
            windows_list.append(win)
            lengths.append(len(win))
        flat_windows = np.concatenate(windows_list, axis=0)
        flat_features = window_feature_matrix(model, flat_windows, cfg, device)
        pos = 0
        for length in lengths:
            bag = flat_features[pos : pos + length]
            all_features.append(aggregate_bag_features(bag))
            pos += length
        print(f"MIL features: {min(start + chunk_records, len(records))}/{len(records)}", end="\r")
    print()
    return np.vstack(all_features).astype(np.float32), labels, groups


def choose_threshold(y_true, y_prob, cfg):
    rows = []
    start, stop, step = cfg["threshold_grid"]
    for th in np.arange(start, stop, step):
        pred = (y_prob >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        rows.append((float(th), float(p), float(r), float(f1)))
    valid = [row for row in rows if row[1] >= cfg["min_precision"]]
    if not valid:
        valid = rows
    best_f1 = max(row[3] for row in valid)
    near = [row for row in valid if row[3] >= best_f1 - cfg["f1_tolerance"]]
    return sorted(near, key=lambda row: (-row[3], -row[1], -row[2], -row[0]))[0]


def metric_pack(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        "Specificity": float(tn / max(tn + fp, 1)),
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Avg Prec": float(average_precision_score(y_true, y_prob)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def train_grouped_xgboost(x_train, y_train, groups, cfg):
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    scale_pos = neg / pos * cfg["pos_weight_boost"]
    base = XGBClassifier(
        objective="binary:logistic",
        n_estimators=cfg["xgb_estimators"],
        scale_pos_weight=scale_pos,
        tree_method="hist",
        eval_metric="logloss",
        random_state=cfg["seed"],
        verbosity=0,
        n_jobs=cfg["xgb_n_jobs"],
    )
    n_groups = len(np.unique(groups))
    if n_groups >= cfg["grid_cv_folds"]:
        cv = GroupKFold(n_splits=cfg["grid_cv_folds"])
        fit_kwargs = {"groups": groups}
    else:
        cv = StratifiedKFold(n_splits=cfg["grid_cv_folds"], shuffle=True, random_state=cfg["seed"])
        fit_kwargs = {}
    scorer = make_scorer(fbeta_score, beta=cfg["recall_beta"], zero_division=0)
    grid = GridSearchCV(
        base,
        cfg["xgb_param_grid"],
        scoring=scorer,
        cv=cv,
        n_jobs=cfg["grid_n_jobs"],
        verbose=1,
        refit=True,
    )
    grid.fit(x_train, y_train, **fit_kwargs)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_), float(scale_pos)


def run_engine_mil_pipeline(fd="FD004", cfg=None):
    cfg = copy.deepcopy(DEFAULT_CFG if cfg is None else cfg)
    cfg["dataset"] = fd
    seed_everything(cfg["seed"])
    device = get_device()
    print("Device:", device)
    print("Dataset:", fd)
    print("Mode: engine-level multi-instance aggregation")

    train_raw, test_raw, rul_raw = load_raw(fd, cfg["data_dir"])
    train_raw = add_train_rul(train_raw)
    test_raw = add_test_rul(test_raw, rul_raw)
    train_df, val_df = split_train_val_by_engine(train_raw, cfg["val_engine_frac"], cfg["seed"])
    feature_cols = get_feature_columns(train_df)

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_raw[feature_cols] = scaler.transform(test_raw[feature_cols])

    x_win_train, y_win_train = make_sliding_windows(
        train_df, feature_cols, cfg["seq_len"], cfg["window_stride"], cfg["rul_threshold"]
    )
    x_win_val, y_win_val = make_sliding_windows(
        val_df, feature_cols, cfg["seq_len"], cfg["window_stride"], cfg["rul_threshold"]
    )
    print("Encoder windows:", x_win_train.shape, x_win_val.shape)
    encoder, encoder_val = train_encoder(x_win_train, y_win_train, x_win_val, y_win_val, cfg, device)
    print("Best encoder validation F-beta:", round(encoder_val, 4))

    train_arrays = make_engine_arrays(train_df, feature_cols)
    val_arrays = make_engine_arrays(val_df, feature_cols)
    test_arrays = make_engine_arrays(test_raw, feature_cols)

    train_records = make_snapshot_records(
        train_df,
        cfg["seq_len"],
        cfg["rul_threshold"],
        cfg["snapshot_stride"],
        one_per_engine=False,
    )
    val_records = make_snapshot_records(
        val_df,
        cfg["seq_len"],
        cfg["rul_threshold"],
        cfg["snapshot_stride"],
        one_per_engine=False,
    )
    test_records = make_snapshot_records(
        test_raw,
        cfg["seq_len"],
        cfg["rul_threshold"],
        cfg["snapshot_stride"],
        one_per_engine=True,
    )
    print("MIL snapshots:")
    print("  train:", train_records.shape, "pos ratio:", round(float(train_records["label"].mean()), 4))
    print("  val  :", val_records.shape, "pos ratio:", round(float(val_records["label"].mean()), 4))
    print("  test :", test_records.shape, "pos ratio:", round(float(test_records["label"].mean()), 4))

    x_mil_train, y_mil_train, groups_train = build_mil_features(encoder, train_arrays, train_records, cfg, device)
    x_mil_val, y_mil_val, _ = build_mil_features(encoder, val_arrays, val_records, cfg, device)
    x_mil_test, y_mil_test, _ = build_mil_features(encoder, test_arrays, test_records, cfg, device)
    print("MIL feature shapes:", x_mil_train.shape, x_mil_val.shape, x_mil_test.shape)

    xgb_model, best_params, best_cv, scale_pos = train_grouped_xgboost(
        x_mil_train, y_mil_train, groups_train, cfg
    )
    print("Best grouped XGBoost CV F-beta:", round(best_cv, 4))
    print("Best params:", best_params)

    val_prob = xgb_model.predict_proba(x_mil_val)[:, 1]
    threshold, val_p, val_r, val_f1 = choose_threshold(y_mil_val, val_prob, cfg)
    test_prob = xgb_model.predict_proba(x_mil_test)[:, 1]
    metrics = metric_pack(y_mil_test, test_prob, threshold)
    metrics.update(
        {
            "Dataset": fd,
            "Threshold": float(threshold),
            "Val Precision": float(val_p),
            "Val Recall": float(val_r),
            "Val F1": float(val_f1),
            "Encoder Val Fbeta": float(encoder_val),
            "XGB CV Fbeta": float(best_cv),
            "Scale Pos Weight": float(scale_pos),
            "Best Params": best_params,
            "Train Snapshots": int(len(train_records)),
            "Val Snapshots": int(len(val_records)),
            "Test Engines": int(len(test_records)),
        }
    )
    print(
        "Final engine-level MIL test: "
        f"P={metrics['Precision']:.4f} R={metrics['Recall']:.4f} "
        f"F1={metrics['F1']:.4f} AUC={metrics['AUC']:.4f}"
    )
    return metrics, {
        "encoder": encoder,
        "xgb_model": xgb_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
    }


if __name__ == "__main__":
    metrics, _ = run_engine_mil_pipeline(DEFAULT_CFG["dataset"], DEFAULT_CFG)
    print(pd.DataFrame([metrics]).drop(columns=["Best Params"]).to_string(index=False))
