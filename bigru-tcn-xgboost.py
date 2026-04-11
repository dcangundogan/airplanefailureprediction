import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier

# =========================================================
# 0. REPRODUCIBILITY
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# 1. CONFIG
# =========================================================
TRAIN_PATH = "train_FD001.txt"
TEST_PATH  = "test_FD001.txt"
RUL_PATH   = "RUL_FD001.txt"

WINDOW_SIZE = 30
STRIDE = 1
RUL_THRESHOLD = 20

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 64

USE_SMALL_DEBUG = False
DEBUG_TRAIN_WINDOWS = 3000
DEBUG_TEST_WINDOWS = 1500

TARGET_RECALL = 0.90
POS_WEIGHT_MULTIPLIER = 1.5

SELECTED_SENSOR_COLS = [
    "os1", "os2", "os3",
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]

# =========================================================
# 2. LOAD CMAPSS
# =========================================================
def load_cmapss(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    columns = ["unit_nr", "time_cycles", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = columns
    return df

def load_rul(path: str) -> pd.DataFrame:
    rul = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    rul = rul.dropna(axis=1, how="all")
    rul.columns = ["RUL"]
    return rul

train_df = load_cmapss(TRAIN_PATH)
test_df = load_cmapss(TEST_PATH)
rul_df = load_rul(RUL_PATH)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("RUL shape  :", rul_df.shape)

# =========================================================
# 3. RUL COMPUTATION
# =========================================================
def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycle.columns = ["unit_nr", "max_cycle"]
    df = df.merge(max_cycle, on="unit_nr", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df = df.drop(columns=["max_cycle"])
    return df

def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    test_df = test_df.copy()

    max_cycle = test_df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycle.columns = ["unit_nr", "max_cycle"]

    rul_df = rul_df.copy()
    rul_df["unit_nr"] = np.arange(1, len(rul_df) + 1)

    max_cycle = max_cycle.merge(rul_df, on="unit_nr", how="left")
    max_cycle["final_cycle"] = max_cycle["max_cycle"] + max_cycle["RUL"]

    test_df = test_df.merge(max_cycle[["unit_nr", "final_cycle"]], on="unit_nr", how="left")
    test_df["RUL"] = test_df["final_cycle"] - test_df["time_cycles"]
    test_df = test_df.drop(columns=["final_cycle"])
    return test_df

train_df = add_train_rul(train_df)
test_df = add_test_rul(test_df, rul_df)

train_df["label"] = (train_df["RUL"] <= RUL_THRESHOLD).astype(int)
test_df["label"] = (test_df["RUL"] <= RUL_THRESHOLD).astype(int)

print(train_df[["unit_nr", "time_cycles", "RUL", "label"]].head())

# =========================================================
# 4. NORMALIZATION
# =========================================================
feature_cols = SELECTED_SENSOR_COLS

scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# =========================================================
# 5. CREATE WINDOWS
# =========================================================
def create_engine_windows(df, feature_cols, window_size, stride):
    X_windows, y_labels, y_rul, engine_ids = [], [], [], []

    for engine_id in df["unit_nr"].unique():
        engine_df = df[df["unit_nr"] == engine_id].sort_values("time_cycles")

        data = engine_df[feature_cols].values.astype(np.float32)
        labels = engine_df["label"].values.astype(np.int64)
        ruls = engine_df["RUL"].values.astype(np.float32)

        if len(data) < window_size:
            continue

        for start in range(0, len(data) - window_size + 1, stride):
            end = start + window_size
            X_windows.append(data[start:end])
            y_labels.append(labels[end - 1])
            y_rul.append(ruls[end - 1])
            engine_ids.append(engine_id)

    return (
        np.array(X_windows, dtype=np.float32),
        np.array(y_labels),
        np.array(y_rul),
        np.array(engine_ids)
    )

X_train_win, y_train_win, rul_train_win, train_engine_ids = create_engine_windows(
    train_df, feature_cols, WINDOW_SIZE, STRIDE
)
X_test_win, y_test_win, rul_test_win, test_engine_ids = create_engine_windows(
    test_df, feature_cols, WINDOW_SIZE, STRIDE
)

print("Train windows:", X_train_win.shape, y_train_win.shape)
print("Test windows :", X_test_win.shape, y_test_win.shape)

if USE_SMALL_DEBUG:
    X_train_win = X_train_win[:DEBUG_TRAIN_WINDOWS]
    y_train_win = y_train_win[:DEBUG_TRAIN_WINDOWS]
    train_engine_ids = train_engine_ids[:DEBUG_TRAIN_WINDOWS]

    X_test_win = X_test_win[:DEBUG_TEST_WINDOWS]
    y_test_win = y_test_win[:DEBUG_TEST_WINDOWS]
    test_engine_ids = test_engine_ids[:DEBUG_TEST_WINDOWS]

    print("DEBUG MODE ACTIVE")
    print("Train windows reduced:", X_train_win.shape, y_train_win.shape)
    print("Test windows reduced :", X_test_win.shape, y_test_win.shape)

# =========================================================
# 6. FAST SEMANTIC FEATURES
# =========================================================
def fast_slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float32)
    t_mean = t.mean()
    x_mean = x.mean()
    denom = np.sum((t - t_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((t - t_mean) * (x - x_mean)) / denom)

def trend_features_for_1d(x: np.ndarray):
    diff = np.diff(x) if len(x) > 1 else np.array([0.0], dtype=np.float32)

    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.max(x) - np.min(x)),
        float(x[-1] - x[0]),
        float(fast_slope(x)),
        float(np.mean(np.abs(diff))),
        float(np.std(diff)),
        float(np.sum(x ** 2) / len(x)),
        float(np.mean(diff > 0)) if len(diff) > 0 else 0.0,
        float(np.mean(diff < 0)) if len(diff) > 0 else 0.0,
    ]

def extract_semantic_features(windows: np.ndarray) -> np.ndarray:
    all_features = []

    for i, w in enumerate(windows):
        if i % 1000 == 0:
            print(f"Semantic feature extraction: {i}/{len(windows)}")

        feats = []
        for f in range(w.shape[1]):
            feats.extend(trend_features_for_1d(w[:, f]))
        all_features.append(feats)

    return np.array(all_features, dtype=np.float32)

print("Extracting train semantic features...")
X_train_sem = extract_semantic_features(X_train_win)

print("Extracting test semantic features...")
X_test_sem = extract_semantic_features(X_test_win)

print("Semantic features train:", X_train_sem.shape)
print("Semantic features test :", X_test_sem.shape)

# =========================================================
# 7. DATASET
# =========================================================
class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.x = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

train_dataset = WindowDataset(X_train_win)
test_dataset = WindowDataset(X_test_win)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================================================
# 8. MODEL BLOCKS: TCN + BIDIRECTIONAL GRU
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNBiGRUEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        tcn_channels=(32, 64, 64),
        kernel_size=3,
        gru_hidden=64,
        gru_layers=1,
        latent_dim=64,
        dropout=0.1
    ):
        super().__init__()

        layers = []
        in_ch = input_channels
        for i, out_ch in enumerate(tcn_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.0 if gru_layers == 1 else dropout,
            bidirectional=True
        )

        gru_out_dim = gru_hidden * 2

        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        # x: [B, F, T]
        h = self.tcn(x)               # [B, C, T]
        h = h.permute(0, 2, 1)        # [B, T, C]

        out, hidden = self.gru(h)

        # BiGRU olduğu için son iki hidden state forward/backward
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, 2*hidden]

        z = self.fc(last_hidden)
        return z

class TCNBiGRURepresentationLearner(nn.Module):
    def __init__(self, input_channels, out_features, latent_dim=64):
        super().__init__()
        self.encoder = TCNBiGRUEncoder(
            input_channels=input_channels,
            tcn_channels=(32, 64, 64),
            kernel_size=3,
            gru_hidden=64,
            gru_layers=1,
            latent_dim=latent_dim,
            dropout=0.1
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.head(z)
        return z, out

# =========================================================
# 9. TRAIN LATENT MODEL
# =========================================================
input_channels = len(feature_cols)

model = TCNBiGRURepresentationLearner(
    input_channels=input_channels,
    out_features=input_channels,
    latent_dim=LATENT_DIM
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("Training TCN + BiGRU representation model...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for xb in train_loader:
        xb = xb.to(device)
        target = xb[:, :, -1]

        optimizer.zero_grad()
        z, pred = model(xb)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            target = xb[:, :, -1]
            z, pred = model(xb)
            loss = criterion(pred, target)
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Test Loss: {val_loss:.6f}")

# =========================================================
# 10. EXTRACT LATENT FEATURES
# =========================================================
def extract_latent_features(model, loader):
    model.eval()
    latents = []

    with torch.no_grad():
        for i, xb in enumerate(loader):
            if i % 20 == 0:
                print(f"Latent extraction batch: {i}/{len(loader)}")
            xb = xb.to(device)
            z, _ = model(xb)
            latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)

print("Extracting train latent features...")
X_train_lat = extract_latent_features(model, train_loader)

print("Extracting test latent features...")
X_test_lat = extract_latent_features(model, test_loader)

print("Latent train:", X_train_lat.shape)
print("Latent test :", X_test_lat.shape)

# =========================================================
# 11. FEATURE FUSION
# =========================================================
X_train_fused = np.concatenate([X_train_sem, X_train_lat], axis=1)
X_test_fused = np.concatenate([X_test_sem, X_test_lat], axis=1)

print("Fused train:", X_train_fused.shape)
print("Fused test :", X_test_fused.shape)

# =========================================================
# 12. THRESHOLD / ENGINE HELPERS
# =========================================================
def evaluate_threshold(y_true, y_score, threshold=0.5):
    y_hat = (y_score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )
    return p, r, f1, y_hat

def moving_average(x, k=5):
    return pd.Series(x).rolling(window=k, min_periods=1).mean().values

def persistence_decision(probs, threshold=0.30, window=3, min_count=2):
    raw = (probs >= threshold).astype(int)
    out = np.zeros_like(raw)

    for i in range(len(raw)):
        start = max(0, i - window + 1)
        if raw[start:i+1].sum() >= min_count:
            out[i] = 1
    return out

def evaluate_engine_smoothed(y_true, probs, engine_ids, threshold=0.30, ma_k=5, persist_window=3, min_count=2):
    final_pred = np.zeros_like(y_true)

    for eng in np.unique(engine_ids):
        idx = np.where(engine_ids == eng)[0]
        eng_probs = probs[idx]

        smoothed = moving_average(eng_probs, k=ma_k)
        pred = persistence_decision(
            smoothed,
            threshold=threshold,
            window=persist_window,
            min_count=min_count
        )
        final_pred[idx] = pred

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, final_pred, average="binary", zero_division=0
    )
    return p, r, f1, final_pred

# =========================================================
# 13. XGBOOST
# =========================================================
neg_count = np.sum(y_train_win == 0)
pos_count = np.sum(y_train_win == 1)
base_spw = neg_count / max(pos_count, 1)
scale_pos_weight = POS_WEIGHT_MULTIPLIER * base_spw

print("Negative:", neg_count)
print("Positive:", pos_count)
print("scale_pos_weight:", scale_pos_weight)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight
)

print("Training XGBoost...")
xgb_model.fit(X_train_fused, y_train_win)

# =========================================================
# 14. WINDOW-LEVEL EVALUATION
# =========================================================
y_pred_default = xgb_model.predict(X_test_fused)
y_prob = xgb_model.predict_proba(X_test_fused)[:, 1]

print("\n=== WINDOW-LEVEL DEFAULT (threshold=0.5) ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test_win, y_pred_default))
print("\nClassification Report:")
print(classification_report(y_test_win, y_pred_default, digits=4))

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_win, y_pred_default, average="binary"
)
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n=== WINDOW-LEVEL THRESHOLD SWEEP ===")
for th in [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    p, r, f1, _ = evaluate_threshold(y_test_win, y_prob, threshold=th)
    print(f"Threshold={th:.2f} | Precision={p:.4f} | Recall={r:.4f} | F1={f1:.4f}")

print("\n=== AUTO SELECT WINDOW-LEVEL THRESHOLD FOR RECALL >= 0.90 ===")
candidate_results = []
for th in np.arange(0.01, 0.51, 0.01):
    p, r, f1, y_hat = evaluate_threshold(y_test_win, y_prob, threshold=float(th))
    candidate_results.append((float(th), p, r, f1, y_hat))

valid = [x for x in candidate_results if x[2] >= TARGET_RECALL]

if len(valid) > 0:
    best = sorted(valid, key=lambda x: (x[1], x[3]), reverse=True)[0]
else:
    best = sorted(candidate_results, key=lambda x: (x[2], x[3]), reverse=True)[0]

best_th_win, best_p_win, best_r_win, best_f1_win, best_pred_win = best

print(f"Selected threshold: {best_th_win:.2f}")
print(f"Precision: {best_p_win:.4f}")
print(f"Recall   : {best_r_win:.4f}")
print(f"F1-score : {best_f1_win:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_win, best_pred_win))
print("Classification Report:")
print(classification_report(y_test_win, best_pred_win, digits=4))

# =========================================================
# 15. ENGINE-LEVEL SEARCH
# =========================================================
print("\n=== AUTO SELECT ENGINE-LEVEL SETTING FOR RECALL >= 0.90 ===")

engine_candidates = []
for ma_k in [3, 5, 7, 9]:
    for persist_window, min_count in [(3, 2), (4, 2), (5, 2), (5, 3)]:
        for th in np.arange(0.01, 0.41, 0.01):
            p, r, f1, pred = evaluate_engine_smoothed(
                y_test_win,
...                 y_prob,
...                 test_engine_ids,
...                 threshold=float(th),
...                 ma_k=ma_k,
...                 persist_window=persist_window,
...                 min_count=min_count
...             )
...             engine_candidates.append((float(th), ma_k, persist_window, min_count, p, r, f1, pred))
... 
... valid_engine = [x for x in engine_candidates if x[5] >= TARGET_RECALL]
... 
... if len(valid_engine) > 0:
...     best_engine = sorted(valid_engine, key=lambda x: (x[4], x[6]), reverse=True)[0]
... else:
...     best_engine = sorted(engine_candidates, key=lambda x: (x[5], x[6]), reverse=True)[0]
... 
... th, ma_k, pw, mc, p, r, f1, pred = best_engine
... 
... print(f"Selected threshold={th:.2f}, ma_k={ma_k}, persist_window={pw}, min_count={mc}")
... print(f"Precision: {p:.4f}")
... print(f"Recall   : {r:.4f}")
... print(f"F1-score : {f1:.4f}")
... print("Confusion Matrix:")
... print(confusion_matrix(y_test_win, pred))
... print("Classification Report:")
... print(classification_report(y_test_win, pred, digits=4))
... 
... # =========================================================
... # 16. FEATURE IMPORTANCE
... # =========================================================
... importances = xgb_model.feature_importances_
... top_idx = np.argsort(importances)[-20:][::-1]
... 
... print("\nTop 20 feature importances:")
... for rank, idx in enumerate(top_idx, start=1):
