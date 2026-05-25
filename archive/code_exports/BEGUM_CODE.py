Python 3.13.0 (v3.13.0:60403a5409f, Oct  7 2024, 00:37:40) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import os
... import copy
... import random
... import warnings
... warnings.filterwarnings("ignore")
... 
... import numpy as np
... import pandas as pd
... 
... import torch
... import torch.nn as nn
... from torch.utils.data import Dataset, DataLoader
... 
... from sklearn.model_selection import train_test_split
... from sklearn.preprocessing import StandardScaler
... from sklearn.metrics import (
...     classification_report,
...     confusion_matrix,
...     precision_recall_fscore_support
... )
... from xgboost import XGBClassifier
... 
... # =========================================================
... # 0. REPRODUCIBILITY
... # =========================================================
... SEED = 42
... random.seed(SEED)
... np.random.seed(SEED)
... torch.manual_seed(SEED)
... torch.cuda.manual_seed_all(SEED)
... 
... device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
... print("Device:", device)
... 
... # =========================================================
... # 1. CONFIG
... # =========================================================
TRAIN_PATH = "train_FD002.txt"
TEST_PATH  = "test_FD002.txt"
RUL_PATH   = "RUL_FD002.txt"

WINDOW_SIZE = 30
STRIDE = 1
RUL_THRESHOLD = 20

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 64

USE_SMALL_DEBUG = False
DEBUG_TRAIN_ENGINES = 20
DEBUG_VAL_ENGINES = 5
DEBUG_TEST_ENGINES = 20

VAL_RATIO = 0.20

# Final business target
TARGET_RECALL_FINAL = 0.90

# Validation threshold selection target
TARGET_RECALL_FOR_SELECTION = 0.97
MIN_PRECISION_FLOOR = 0.70

# If F1 values are very close, prefer lower threshold
F1_TOLERANCE = 0.005

# ReduceLROnPlateau
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MIN_DELTA = 1e-4

SELECTED_SENSOR_COLS = [
    "os1", "os2", "os3",
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]

# Adaptive XGBoost search space
MAX_DEPTH_CANDIDATES = [4, 5]
LEARNING_RATE_CANDIDATES = [0.03, 0.05]
N_ESTIMATORS_CANDIDATES = [300, 500, 700]
POS_WEIGHT_MULTIPLIER_CANDIDATES = [1.5, 2.0, 2.5, 3.0]

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
# 4. ENGINE-BASED SPLIT
# =========================================================
all_train_engines = np.array(sorted(train_df["unit_nr"].unique()))
all_test_engines = np.array(sorted(test_df["unit_nr"].unique()))

train_engines, val_engines = train_test_split(
    all_train_engines,
    test_size=VAL_RATIO,
    random_state=SEED
)

if USE_SMALL_DEBUG:
    train_engines = train_engines[:DEBUG_TRAIN_ENGINES]
    val_engines = val_engines[:DEBUG_VAL_ENGINES]
    all_test_engines = all_test_engines[:DEBUG_TEST_ENGINES]
    test_df = test_df[test_df["unit_nr"].isin(all_test_engines)].copy()

train_df_split = train_df[train_df["unit_nr"].isin(train_engines)].copy()
val_df_split = train_df[train_df["unit_nr"].isin(val_engines)].copy()

print("Num train engines:", len(train_engines))
print("Num val engines  :", len(val_engines))
print("Num test engines :", len(all_test_engines))
print("Train split rows:", train_df_split.shape)
print("Val split rows  :", val_df_split.shape)
print("Test rows       :", test_df.shape)

# =========================================================
# 5. NORMALIZATION
# =========================================================
feature_cols = SELECTED_SENSOR_COLS

scaler = StandardScaler()
train_df_split[feature_cols] = scaler.fit_transform(train_df_split[feature_cols])
val_df_split[feature_cols] = scaler.transform(val_df_split[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# =========================================================
# 6. CREATE WINDOWS
# =========================================================
def create_engine_windows(df, feature_cols, window_size, stride):
    X_windows, y_labels, y_rul, engine_ids = [], [], [], []

    for engine_id in sorted(df["unit_nr"].unique()):
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
        np.array(y_labels, dtype=np.int64),
        np.array(y_rul, dtype=np.float32),
        np.array(engine_ids, dtype=np.int64)
    )

X_train_win, y_train_win, rul_train_win, train_engine_ids = create_engine_windows(
    train_df_split, feature_cols, WINDOW_SIZE, STRIDE
)
X_val_win, y_val_win, rul_val_win, val_engine_ids = create_engine_windows(
    val_df_split, feature_cols, WINDOW_SIZE, STRIDE
)
X_test_win, y_test_win, rul_test_win, test_engine_ids = create_engine_windows(
    test_df, feature_cols, WINDOW_SIZE, STRIDE
)

print("Train windows:", X_train_win.shape, y_train_win.shape)
print("Val windows  :", X_val_win.shape, y_val_win.shape)
print("Test windows :", X_test_win.shape, y_test_win.shape)

# =========================================================
# 7. FAST SEMANTIC FEATURES
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

print("Extracting val semantic features...")
X_val_sem = extract_semantic_features(X_val_win)

print("Extracting test semantic features...")
X_test_sem = extract_semantic_features(X_test_win)

print("Semantic features train:", X_train_sem.shape)
print("Semantic features val  :", X_val_sem.shape)
print("Semantic features test :", X_test_sem.shape)

# =========================================================
# 8. DATASET
# =========================================================
class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.x = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

train_dataset = WindowDataset(X_train_win)
val_dataset = WindowDataset(X_val_win)
test_dataset = WindowDataset(X_test_win)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================================================
# 9. MODEL BLOCKS
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
        h = self.tcn(x)
        h = h.permute(0, 2, 1)
        _, hidden = self.gru(h)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
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
# 10. TRAIN LATENT MODEL
# =========================================================
input_channels = len(feature_cols)

model = TCNBiGRURepresentationLearner(
    input_channels=input_channels,
    out_features=input_channels,
    latent_dim=LATENT_DIM
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=MIN_LR
)

print("Training TCN + BiGRU representation model...")

best_val_loss = float("inf")
best_model_state = copy.deepcopy(model.state_dict())
early_stop_counter = 0
prev_lr = optimizer.param_groups[0]["lr"]

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
        for xb in val_loader:
            xb = xb.to(device)
            target = xb[:, :, -1]
            z, pred = model(xb)
            loss = criterion(pred, target)
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"LR: {current_lr:.7f}"
    )

    if current_lr < prev_lr:
        print(f"  -> ReduceLROnPlateau lowered LR: {prev_lr:.7f} -> {current_lr:.7f}")
    prev_lr = current_lr

    if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_counter = 0
        print(f"  -> New best model saved. Best val loss: {best_val_loss:.6f}")
    else:
        early_stop_counter += 1
        print(f"  -> No improvement. Counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")

    if early_stop_counter >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered.")
        break

model.load_state_dict(best_model_state)
print(f"Best model restored. Best val loss: {best_val_loss:.6f}")

# =========================================================
# 11. EXTRACT LATENT FEATURES
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

print("Extracting val latent features...")
X_val_lat = extract_latent_features(model, val_loader)

print("Extracting test latent features...")
X_test_lat = extract_latent_features(model, test_loader)

print("Latent train:", X_train_lat.shape)
print("Latent val  :", X_val_lat.shape)
print("Latent test :", X_test_lat.shape)

# =========================================================
# 12. FEATURE FUSION
# =========================================================
X_train_fused = np.concatenate([X_train_sem, X_train_lat], axis=1)
X_val_fused = np.concatenate([X_val_sem, X_val_lat], axis=1)
X_test_fused = np.concatenate([X_test_sem, X_test_lat], axis=1)

print("Fused train:", X_train_fused.shape)
print("Fused val  :", X_val_fused.shape)
print("Fused test :", X_test_fused.shape)

# =========================================================
# 13. HELPERS
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

def choose_best_threshold_by_validation(
    y_true_val,
    y_prob_val,
    validation_target=0.93,
    min_precision=0.80,
    f1_tolerance=0.005
):
    candidate_results = []
    thresholds = np.arange(0.10, 0.80, 0.0025)

    for th in thresholds:
        p, r, f1, y_hat = evaluate_threshold(y_true_val, y_prob_val, threshold=float(th))
        candidate_results.append((float(th), p, r, f1, y_hat))

    valid = [
        x for x in candidate_results
        if x[2] >= validation_target and x[1] >= min_precision
    ]

    if len(valid) == 0:
        precision_ok = [x for x in candidate_results if x[1] >= min_precision]
        if len(precision_ok) > 0:
            best = sorted(precision_ok, key=lambda x: (x[2], x[3], x[1]), reverse=True)[0]
            return best, candidate_results
        best = sorted(candidate_results, key=lambda x: (x[2], x[3], x[1]), reverse=True)[0]
        return best, candidate_results

    best_f1 = max(x[3] for x in valid)
    near_best = [x for x in valid if x[3] >= best_f1 - f1_tolerance]

    # near-best F1 candidates içinde daha düşük threshold seç
    best = sorted(
    near_best,
    key=lambda x: (-x[2], -x[3], -x[1], x[0])
)[0]
    return best, candidate_results

def choose_best_engine_setting_by_validation(
    y_true_val,
    y_prob_val,
    engine_ids_val,
    validation_target=0.93,
    min_precision=0.80,
    f1_tolerance=0.005
):
    engine_candidates = []

    for ma_k in [3, 5, 7]:
        for persist_window, min_count in [(2, 1), (3, 1), (3, 2), (4, 2)]:
            for th in np.arange(0.10, 0.80, 0.01):
                p, r, f1, pred = evaluate_engine_smoothed(
                    y_true_val,
                    y_prob_val,
                    engine_ids_val,
                    threshold=float(th),
                    ma_k=ma_k,
                    persist_window=persist_window,
                    min_count=min_count
                )
                engine_candidates.append((float(th), ma_k, persist_window, min_count, p, r, f1, pred))

    valid = [
        x for x in engine_candidates
        if x[5] >= validation_target and x[4] >= min_precision
    ]

    if len(valid) == 0:
        precision_ok = [x for x in engine_candidates if x[4] >= min_precision]
        if len(precision_ok) > 0:
            best_engine = sorted(precision_ok, key=lambda x: (x[5], x[6], x[4]), reverse=True)[0]
            return best_engine
        best_engine = sorted(engine_candidates, key=lambda x: (x[5], x[6], x[4]), reverse=True)[0]
        return best_engine

    best_f1 = max(x[6] for x in valid)
    near_best = [x for x in valid if x[6] >= best_f1 - f1_tolerance]

    # near-best F1 candidates içinde daha düşük threshold seç
    best_engine = sorted(near_best, key=lambda x: (x[0], -x[4], -x[5]))[0]
    return best_engine

# =========================================================
# 14. ADAPTIVE XGBOOST SEARCH
# =========================================================
neg_count = np.sum(y_train_win == 0)
pos_count = np.sum(y_train_win == 1)
base_spw = neg_count / max(pos_count, 1)

print("Negative:", neg_count)
print("Positive:", pos_count)
print("Base scale_pos_weight:", base_spw)

search_results = []
best_model = None
best_cfg = None

total_trials = (
    len(MAX_DEPTH_CANDIDATES)
    * len(LEARNING_RATE_CANDIDATES)
    * len(N_ESTIMATORS_CANDIDATES)
    * len(POS_WEIGHT_MULTIPLIER_CANDIDATES)
)

trial_no = 0

for max_depth in MAX_DEPTH_CANDIDATES:
    for learning_rate in LEARNING_RATE_CANDIDATES:
        for n_estimators in N_ESTIMATORS_CANDIDATES:
            for pwm in POS_WEIGHT_MULTIPLIER_CANDIDATES:
                trial_no += 1
                scale_pos_weight = base_spw * pwm

                print(
                    f"\n[Trial {trial_no}/{total_trials}] "
                    f"max_depth={max_depth}, lr={learning_rate}, "
                    f"n_estimators={n_estimators}, pwm={pwm}, "
                    f"scale_pos_weight={scale_pos_weight:.4f}"
                )

                xgb_model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight
                )

                xgb_model.fit(X_train_fused, y_train_win)

                y_val_prob = xgb_model.predict_proba(X_val_fused)[:, 1]

                best_val_threshold_pack, _ = choose_best_threshold_by_validation(
                    y_true_val=y_val_win,
                    y_prob_val=y_val_prob,
                    validation_target=TARGET_RECALL_FOR_SELECTION,
                    min_precision=MIN_PRECISION_FLOOR,
                    f1_tolerance=F1_TOLERANCE
                )

                best_th, best_p, best_r, best_f1, _ = best_val_threshold_pack

                print(
                    f"  -> best_th={best_th:.4f}, "
                    f"val_precision={best_p:.4f}, "
                    f"val_recall={best_r:.4f}, "
                    f"val_f1={best_f1:.4f}"
                )

                result = {
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "pos_weight_multiplier": pwm,
                    "scale_pos_weight": scale_pos_weight,
                    "threshold": best_th,
                    "val_precision": best_p,
                    "val_recall": best_r,
                    "val_f1": best_f1,
                    "model": xgb_model
                }
                search_results.append(result)

valid_results = [
    r for r in search_results
    if r["val_recall"] >= TARGET_RECALL_FOR_SELECTION and r["val_precision"] >= MIN_PRECISION_FLOOR
]

if len(valid_results) > 0:
    best_f1 = max(r["val_f1"] for r in valid_results)
    near_best = [r for r in valid_results if r["val_f1"] >= best_f1 - F1_TOLERANCE]

    # near-best configler içinde daha düşük threshold'u tercih et
    best_cfg = sorted(
        near_best,
        key=lambda x: (x["threshold"], -x["val_precision"], -x["val_recall"])
    )[0]
else:
    precision_ok_results = [r for r in search_results if r["val_precision"] >= MIN_PRECISION_FLOOR]
    if len(precision_ok_results) > 0:
        best_cfg = sorted(
            precision_ok_results,
            key=lambda x: (x["val_recall"], x["val_f1"], x["val_precision"]),
            reverse=True
        )[0]
    else:
        best_cfg = sorted(
            search_results,
            key=lambda x: (x["val_recall"], x["val_f1"], x["val_precision"]),
            reverse=True
        )[0]

best_model = best_cfg["model"]

print("\n" + "=" * 70)
print("BEST ADAPTIVE CONFIG SELECTED ON VALIDATION")
print("=" * 70)
print(f"max_depth            : {best_cfg['max_depth']}")
print(f"learning_rate        : {best_cfg['learning_rate']}")
print(f"n_estimators         : {best_cfg['n_estimators']}")
print(f"pos_weight_multiplier: {best_cfg['pos_weight_multiplier']}")
print(f"scale_pos_weight     : {best_cfg['scale_pos_weight']:.4f}")
print(f"threshold            : {best_cfg['threshold']:.4f}")
print(f"val_precision        : {best_cfg['val_precision']:.4f}")
print(f"val_recall           : {best_cfg['val_recall']:.4f}")
print(f"val_f1               : {best_cfg['val_f1']:.4f}")

# =========================================================
# 15. VALIDATION-SELECTED ENGINE SETTING
# =========================================================
y_val_prob_best = best_model.predict_proba(X_val_fused)[:, 1]

best_engine_val = choose_best_engine_setting_by_validation(
    y_true_val=y_val_win,
    y_prob_val=y_val_prob_best,
    engine_ids_val=val_engine_ids,
    validation_target=TARGET_RECALL_FOR_SELECTION,
    min_precision=MIN_PRECISION_FLOOR,
    f1_tolerance=F1_TOLERANCE
)

eng_th, eng_ma_k, eng_pw, eng_mc, eng_p_val, eng_r_val, eng_f1_val, eng_pred_val = best_engine_val

print("\n=== SELECTED ENGINE SETTING ON VALIDATION ===")
print(f"Selected threshold={eng_th:.2f}, ma_k={eng_ma_k}, persist_window={eng_pw}, min_count={eng_mc}")
print(f"Validation Precision: {eng_p_val:.4f}")
print(f"Validation Recall   : {eng_r_val:.4f}")
print(f"Validation F1-score : {eng_f1_val:.4f}")

# =========================================================
# 16. FINAL TEST EVALUATION
# =========================================================
y_test_prob = best_model.predict_proba(X_test_fused)[:, 1]
best_th_win = best_cfg["threshold"]

y_test_pred_default = (y_test_prob >= 0.5).astype(int)

print("\n=== TEST WINDOW-LEVEL DEFAULT (threshold=0.5) ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test_win, y_test_pred_default))
print("\nClassification Report:")
print(classification_report(y_test_win, y_test_pred_default, digits=4))

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_win, y_test_pred_default, average="binary", zero_division=0
)
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

best_p_test, best_r_test, best_f1_test, best_pred_test = evaluate_threshold(
    y_test_win,
    y_test_prob,
    threshold=best_th_win
)

print("\n=== TEST WINDOW-LEVEL USING ADAPTIVE VALIDATION-SELECTED THRESHOLD ===")
print(f"Threshold: {best_th_win:.4f}")
print(f"Precision: {best_p_test:.4f}")
print(f"Recall   : {best_r_test:.4f}")
print(f"F1-score : {best_f1_test:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_win, best_pred_test))
print("Classification Report:")
print(classification_report(y_test_win, best_pred_test, digits=4))

eng_p_test, eng_r_test, eng_f1_test, eng_pred_test = evaluate_engine_smoothed(
    y_test_win,
    y_test_prob,
    test_engine_ids,
    threshold=eng_th,
    ma_k=eng_ma_k,
    persist_window=eng_pw,
    min_count=eng_mc
)

print("\n=== TEST ENGINE-LEVEL USING VALIDATION-SELECTED SETTING ===")
print(f"Threshold={eng_th:.2f}, ma_k={eng_ma_k}, persist_window={eng_pw}, min_count={eng_mc}")
print(f"Precision: {eng_p_test:.4f}")
print(f"Recall   : {eng_r_test:.4f}")
print(f"F1-score : {eng_f1_test:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_win, eng_pred_test))
print("Classification Report:")
print(classification_report(y_test_win, eng_pred_test, digits=4))

# =========================================================
# 17. TEST THRESHOLD SWEEP
# Analysis only - do not use for selection
# =========================================================
print("\n=== TEST THRESHOLD SWEEP (ANALYSIS ONLY) ===")
for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    p, r, f1, _ = evaluate_threshold(y_test_win, y_test_prob, threshold=th)
    print(f"Threshold={th:.2f} | Precision={p:.4f} | Recall={r:.4f} | F1={f1:.4f}")

# =========================================================
# 18. FEATURE IMPORTANCE
# =========================================================
importances = best_model.feature_importances_
top_idx = np.argsort(importances)[-20:][::-1]

print("\nTop 20 feature importances:")
for rank, idx in enumerate(top_idx, start=1):
    print(f"{rank:02d}. Feature {idx} -> {importances[idx]:.6f}")

# =========================================================
# 19. SEARCH RESULTS TABLE
# =========================================================
results_df = pd.DataFrame([
    {
        "max_depth": r["max_depth"],
        "learning_rate": r["learning_rate"],
        "n_estimators": r["n_estimators"],
        "pos_weight_multiplier": r["pos_weight_multiplier"],
        "scale_pos_weight": r["scale_pos_weight"],
        "threshold": r["threshold"],
        "val_precision": r["val_precision"],
        "val_recall": r["val_recall"],
        "val_f1": r["val_f1"],
    }
    for r in search_results
])

print("\nTop 15 adaptive search results:")
valid_df = results_df[
    (results_df["val_recall"] >= TARGET_RECALL_FOR_SELECTION) &
    (results_df["val_precision"] >= MIN_PRECISION_FLOOR)
].copy()

if len(valid_df) > 0:
    best_f1 = valid_df["val_f1"].max()
    valid_df["near_best_f1"] = valid_df["val_f1"] >= (best_f1 - F1_TOLERANCE)
    valid_df = valid_df.sort_values(
        ["near_best_f1", "threshold", "val_precision", "val_recall"],
        ascending=[False, True, False, False]
    )
    print(valid_df.head(15).to_string(index=False))
else:
    results_df = results_df.sort_values(
        ["val_recall", "val_f1", "val_precision"],
        ascending=False
    )
