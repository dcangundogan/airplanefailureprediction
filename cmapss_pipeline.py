# Auto-extracted CMAPSS hybrid pipeline (TCN-BiGRU-Attention + Semantic XGBoost).
# Source: notebooks/hybrid_tcn_bigru_attention_semantic_xgboost_cmapss.ipynb
# Edit the notebook as the source of truth; regenerate this file if it changes.


# ===== notebook cell 2 =====
import copy
import json
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError('xgboost is required. Set INSTALL_MISSING_PACKAGES=True in the previous cell, run it, then rerun imports.') from exc

warnings.filterwarnings('ignore')


CFG = {
    'seed': 42,
    'deterministic': True,
    'profile': 'benchmark',  # benchmark or production
    'data_dir': 'data',
    'dataset': 'FD002',

    'rul_threshold': 30,
    'seq_len': 40,
    'stride': 1,
    'val_engine_frac': 0.20,
    'test_mode': 'last',  # benchmark: last, production: sliding

    'tcn_channels': [64, 128, 128, 64],
    'tcn_kernel': 3,
    'tcn_dropout': 0.20,
    'gru_hidden': 128,
    'gru_layers': 2,
    'gru_dropout': 0.30,
    'embed_dim': 128,

    'epochs': 60,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 12,
    'min_delta': 1e-4,
    'focal_gamma': 2.5,
    'recon_weight': 0.30,
    'pos_weight_boost': 2.0,
    'recall_beta': 2.0,
    'use_amp': True,

    'use_semantic_features': True,
    'use_engine_smoothing': True,
    'threshold_policy': 'balanced',  # balanced or early_alarm
    'target_recall_for_selection': None,
    'min_precision': 0.70,
    'f1_tolerance': 0.005,
    'threshold_grid': (0.05, 0.95, 0.0025),
    'engine_threshold_grid': (0.10, 0.80, 0.01),
    'ma_k_grid': [3, 5, 7],
    'persist_grid': [(2, 1), (3, 1), (3, 2), (4, 2)],

    # --- Hybrid F1 boost: ensemble + test-like val + calibration ---
    'ensemble_size': 5,                  # XGBoost boosters trained with different seeds on shared encoder
    'use_testlike_val_threshold': True,  # build val set that mirrors the CMAPSS test "last-window-per-engine" format
    'testlike_samples_per_engine': 12,   # truncations per val engine (more = more stable threshold)
    'calibrate_probabilities': True,     # isotonic-calibrate booster probabilities using sliding val
    'min_recall_for_selection': 0.85,    # benchmark threshold floor: prefer thresholds keeping val recall above this

    'grid_cv_folds': 3,
    'grid_n_jobs': -1,
    'xgb_n_jobs': 1,
    'grid_xgb_estimators': 400,
    'xgb_rounds': 800,
    'xgb_es_rounds': 40,
    'xgb_param_grid': {
        'max_depth': [3, 4, 6],
        'learning_rate': [0.03, 0.05, 0.10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0.0, 0.1],
        'reg_lambda': [1.0, 2.0],
    },
    'xgb_alpha': 0.1,
    'xgb_lambda': 1.0,

    'save_artifacts': False,
    'artifact_dir': 'artifacts/hybrid_tcn_bigru_xgb',
}


def seed_everything(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


seed_everything(CFG['seed'], CFG.get('deterministic', True))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)


# ===== notebook cell 3 =====
COLUMNS = (
    ['unit_number', 'time_in_cycles']
    + [f'op_{i}' for i in range(1, 4)]
    + [f's{i}' for i in range(1, 22)]
)

DROP_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']


def resolve_data_dir(data_dir):
    candidates = [
        Path(data_dir),
        Path.cwd() / data_dir,
        Path('/content/drive/MyDrive/data'),
        Path('/content/data'),
    ]
    for path in candidates:
        if (path / 'train_FD001.txt').exists():
            return path
    checked = ', '.join(str(p) for p in candidates)
    raise FileNotFoundError('CMAPSS data directory not found. Checked: ' + checked)


def load_raw(fd, data_dir):
    data_dir = resolve_data_dir(data_dir)
    train = pd.read_csv(data_dir / f'train_{fd}.txt', sep=r'\s+', header=None, names=COLUMNS)
    test = pd.read_csv(data_dir / f'test_{fd}.txt', sep=r'\s+', header=None, names=COLUMNS)
    rul = pd.read_csv(data_dir / f'RUL_{fd}.txt', sep=r'\s+', header=None, names=['RUL'])
    return train, test, rul


def add_train_rul(df):
    max_cycle = df.groupby('unit_number')['time_in_cycles'].max()
    out = df.join(max_cycle.rename('max_cycle'), on='unit_number')
    out['RUL'] = out['max_cycle'] - out['time_in_cycles']
    return out.drop(columns=['max_cycle'])


def add_test_rul(test_df, rul_df):
    out = test_df.copy()
    max_cycle = out.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycle.columns = ['unit_number', 'max_cycle']
    rul = rul_df.copy()
    rul['unit_number'] = np.arange(1, len(rul) + 1)
    max_cycle = max_cycle.merge(rul, on='unit_number', how='left')
    max_cycle['final_cycle'] = max_cycle['max_cycle'] + max_cycle['RUL']
    out = out.merge(max_cycle[['unit_number', 'final_cycle']], on='unit_number', how='left')
    out['RUL'] = out['final_cycle'] - out['time_in_cycles']
    return out.drop(columns=['final_cycle'])


def get_feature_columns(df):
    drop = {'unit_number', 'time_in_cycles', 'RUL', 'label', *DROP_SENSORS}
    return [col for col in df.columns if col not in drop]


def split_train_val_by_engine(train_df, val_frac, seed):
    units = np.array(sorted(train_df['unit_number'].unique()))
    rng = np.random.RandomState(seed)
    val_size = max(1, int(round(len(units) * val_frac)))
    val_units = set(rng.choice(units, size=val_size, replace=False).tolist())
    val_mask = train_df['unit_number'].isin(val_units)
    return train_df.loc[~val_mask].copy(), train_df.loc[val_mask].copy()


def make_sliding_windows(df, feat_cols, seq_len, stride, threshold):
    X_list, y_list, rul_list, eng_list = [], [], [], []
    for unit, group in df.groupby('unit_number'):
        group = group.sort_values('time_in_cycles')
        arr = group[feat_cols].values.astype(np.float32)
        ruls = group['RUL'].values.astype(np.float32)
        labels = (ruls <= threshold).astype(np.int64)
        if len(arr) < seq_len:
            continue
        for start in range(0, len(arr) - seq_len + 1, stride):
            end = start + seq_len
            X_list.append(arr[start:end])
            y_list.append(labels[end - 1])
            rul_list.append(ruls[end - 1])
            eng_list.append(int(unit))
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(rul_list, dtype=np.float32),
        np.asarray(eng_list, dtype=np.int64),
    )


def make_test_windows_last(test_df, rul_df, feat_cols, seq_len, threshold):
    X_list, y_list, rul_list, eng_list = [], [], [], []
    for idx, (unit, group) in enumerate(test_df.groupby('unit_number')):
        group = group.sort_values('time_in_cycles')
        arr = group[feat_cols].values.astype(np.float32)
        if len(arr) >= seq_len:
            window = arr[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(arr), arr.shape[1]), dtype=np.float32)
            window = np.vstack([pad, arr])
        true_rul = float(rul_df.iloc[idx]['RUL'])
        X_list.append(window)
        y_list.append(int(true_rul <= threshold))
        rul_list.append(true_rul)
        eng_list.append(int(unit))
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(rul_list, dtype=np.float32),
        np.asarray(eng_list, dtype=np.int64),
    )


def make_testlike_val_windows(val_df, feat_cols, seq_len, threshold, samples_per_engine, seed):
    """Simulate CMAPSS test construction on val engines.

    For each val engine we sample `samples_per_engine` truncation cycles uniformly
    over the engine's life, and emit ONE window ending at each truncation cycle
    (mirroring how the real test set keeps only the last window per engine).
    The resulting val set has the same window-per-engine semantics as the test
    set, so a threshold selected on it transfers cleanly.
    """
    rng = np.random.RandomState(seed)
    X_list, y_list, rul_list, eng_list = [], [], [], []
    for unit, group in val_df.groupby('unit_number'):
        group = group.sort_values('time_in_cycles')
        arr = group[feat_cols].values.astype(np.float32)
        ruls = group['RUL'].values.astype(np.float32)
        n = len(arr)
        if n < seq_len:
            continue
        possible_ends = np.arange(seq_len, n + 1)
        n_samples = min(samples_per_engine, len(possible_ends))
        chosen = rng.choice(len(possible_ends), size=n_samples, replace=False)
        for i in chosen:
            end = int(possible_ends[i])
            window = arr[end - seq_len:end]
            X_list.append(window)
            y_list.append(int(ruls[end - 1] <= threshold))
            rul_list.append(float(ruls[end - 1]))
            eng_list.append(int(unit))
    if not X_list:
        return (
            np.zeros((0, seq_len, len(feat_cols)), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(rul_list, dtype=np.float32),
        np.asarray(eng_list, dtype=np.int64),
    )


def extract_semantic_features(X):
    n, t_len, n_feat = X.shape
    t = np.arange(t_len, dtype=np.float32)
    t_mean = t.mean()
    denom = max(float(np.sum((t - t_mean) ** 2)), 1e-9)

    x_mean = X.mean(axis=1)
    x_std = X.std(axis=1)
    x_min = X.min(axis=1)
    x_max = X.max(axis=1)
    x_range = x_max - x_min
    x_drift = X[:, -1, :] - X[:, 0, :]
    slope = ((t[None, :, None] - t_mean) * (X - x_mean[:, None, :])).sum(axis=1) / denom

    diff = np.diff(X, axis=1)
    diff_abs_mean = np.abs(diff).mean(axis=1)
    diff_std = diff.std(axis=1)
    energy = (X ** 2).sum(axis=1) / t_len
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


# ===== notebook cell 4 =====
def make_weight_norm_conv(in_ch, out_ch, kernel_size, padding, dilation):
    conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
    if hasattr(nn.utils, 'parametrizations'):
        return nn.utils.parametrizations.weight_norm(conv)
    return nn.utils.weight_norm(conv)


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = make_weight_norm_conv(in_ch, out_ch, kernel_size, pad, dilation)
        self.conv2 = make_weight_norm_conv(out_ch, out_ch, kernel_size, pad, dilation)
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


class TCN(nn.Module):
    def __init__(self, in_ch, channels, kernel_size, dropout):
        super().__init__()
        layers = []
        cur = in_ch
        for idx, out_ch in enumerate(channels):
            layers.append(CausalConvBlock(cur, out_ch, kernel_size, 2 ** idx, dropout))
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
        self.tcn = TCN(n_features, cfg['tcn_channels'], cfg['tcn_kernel'], cfg['tcn_dropout'])
        tcn_out = cfg['tcn_channels'][-1]
        self.bigru = nn.GRU(
            input_size=tcn_out,
            hidden_size=cfg['gru_hidden'],
            num_layers=cfg['gru_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=cfg['gru_dropout'] if cfg['gru_layers'] > 1 else 0.0,
        )
        gru_out = cfg['gru_hidden'] * 2
        self.attn = AttentionPool(gru_out)
        self.proj = nn.Sequential(
            nn.Linear(gru_out, cfg['embed_dim']),
            nn.LayerNorm(cfg['embed_dim']),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.cls_head = nn.Linear(cfg['embed_dim'], 1)
        self.recon_head = nn.Sequential(
            nn.Linear(cfg['embed_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def embed(self, x):
        tcn_out = self.tcn(x)
        seq_out, _ = self.bigru(tcn_out)
        pooled = self.attn(seq_out)
        return self.proj(pooled)

    def forward(self, x):
        z = self.embed(x)
        logit = self.cls_head(z).squeeze(-1)
        recon = self.recon_head(z)
        return logit, recon, z


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
            reduction='none',
        )
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        return ((1 - pt) ** self.gamma * bce).mean()


def make_loader(X, y=None, batch_size=256, shuffle=False):
    x_tensor = torch.from_numpy(X).float()
    if y is None:
        ds = TensorDataset(x_tensor)
    else:
        ds = TensorDataset(x_tensor, torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def train_encoder(X_train, y_train, X_val, y_val, cfg):
    model = TCNBiGRUAttention(X_train.shape[2], cfg).to(DEVICE)
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    pos_weight = torch.tensor([neg / pos * cfg['pos_weight_boost']], dtype=torch.float32, device=DEVICE)
    cls_loss = FocalLoss(gamma=cfg['focal_gamma'], pos_weight=pos_weight)
    recon_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-5)

    train_loader = make_loader(X_train, y_train, cfg['batch_size'], shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg['batch_size'], shuffle=False)
    use_amp = bool(cfg['use_amp'] and DEVICE.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    wait = 0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, recon, _ = model(xb)
                loss = cls_loss(logits, yb) + cfg['recon_weight'] * recon_loss(recon, xb[:, -1, :])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(xb)

        scheduler.step()
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, _, _ = model(xb.to(DEVICE))
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(yb.numpy())
        val_probs = np.asarray(val_probs)
        val_true = np.asarray(val_true)
        val_pred = (val_probs >= 0.5).astype(int)
        val_fb = fbeta_score(val_true, val_pred, beta=cfg['recall_beta'], zero_division=0)
        val_auc = roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else np.nan
        avg_loss = total_loss / max(len(X_train), 1)
        history.append({'epoch': epoch, 'train_loss': avg_loss, 'val_fbeta': val_fb, 'val_auc': val_auc})

        if epoch == 1 or epoch % 5 == 0:
            print(f'Epoch {epoch:03d}: loss={avg_loss:.4f} val_F{cfg["recall_beta"]:.0f}={val_fb:.4f} val_auc={val_auc:.4f}')

        if val_fb > best_score + cfg['min_delta']:
            best_score = val_fb
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg['patience']:
                print('Early stopping at epoch', epoch)
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), float(best_score)


@torch.no_grad()
def extract_embeddings(model, X, batch_size=512):
    model.eval()
    parts = []
    for (xb,) in make_loader(X, None, batch_size=batch_size, shuffle=False):
        parts.append(model.embed(xb.to(DEVICE)).cpu().numpy())
    return np.vstack(parts).astype(np.float32)


# ===== notebook cell 5 =====
def evaluate_threshold(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
    return float(p), float(r), float(f1), pred


def moving_average(values, k):
    return pd.Series(values).rolling(window=k, min_periods=1).mean().values


def persistence_decision(probs, threshold, window, min_count):
    raw = (probs >= threshold).astype(int)
    out = np.zeros_like(raw)
    for idx in range(len(raw)):
        start = max(0, idx - window + 1)
        out[idx] = int(raw[start:idx + 1].sum() >= min_count)
    return out


def evaluate_engine_smoothed(y_true, y_prob, engine_ids, threshold, ma_k, persist_w, min_count):
    pred = np.zeros_like(y_true)
    for engine in np.unique(engine_ids):
        idx = np.where(engine_ids == engine)[0]
        smooth = moving_average(y_prob[idx], ma_k)
        pred[idx] = persistence_decision(smooth, threshold, persist_w, min_count)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
    return float(p), float(r), float(f1), pred


def threshold_values(grid_tuple):
    start, stop, step = grid_tuple
    return np.arange(start, stop, step)


def recall_floor_enabled(cfg):
    return cfg.get('target_recall_for_selection') is not None


def choose_from_near_best(candidates, f1_idx, recall_idx, precision_idx, threshold_idx, cfg):
    best_f1 = max(c[f1_idx] for c in candidates)
    near = [c for c in candidates if c[f1_idx] >= best_f1 - cfg['f1_tolerance']]
    if cfg.get('threshold_policy') == 'early_alarm':
        return sorted(near, key=lambda c: (c[threshold_idx], -c[recall_idx], -c[f1_idx], -c[precision_idx]))[0]
    return sorted(near, key=lambda c: (-c[f1_idx], -c[precision_idx], -c[recall_idx], -c[threshold_idx]))[0]


def choose_window_threshold(y_true, y_prob, cfg):
    candidates = []
    for th in threshold_values(cfg['threshold_grid']):
        p, r, f1, _ = evaluate_threshold(y_true, y_prob, float(th))
        candidates.append((float(th), p, r, f1))

    if recall_floor_enabled(cfg):
        valid = [c for c in candidates if c[1] >= cfg['min_precision'] and c[2] >= cfg['target_recall_for_selection']]
    else:
        valid = []
    if not valid:
        valid = [c for c in candidates if c[1] >= cfg['min_precision']]
    if not valid:
        valid = candidates

    # Hybrid F1 boost: when in balanced/benchmark mode, also enforce a soft recall floor
    # so the threshold does not drift to pure-precision regimes that miss positives.
    min_r = cfg.get('min_recall_for_selection')
    if (
        cfg.get('threshold_policy') != 'early_alarm'
        and not recall_floor_enabled(cfg)
        and min_r is not None
    ):
        recall_filtered = [c for c in valid if c[2] >= min_r]
        if recall_filtered:
            valid = recall_filtered

    return choose_from_near_best(valid, f1_idx=3, recall_idx=2, precision_idx=1, threshold_idx=0, cfg=cfg)


def choose_engine_setting(y_true, y_prob, engine_ids, cfg):
    candidates = []
    for ma_k in cfg['ma_k_grid']:
        for persist_w, min_count in cfg['persist_grid']:
            for th in threshold_values(cfg['engine_threshold_grid']):
                p, r, f1, _ = evaluate_engine_smoothed(
                    y_true,
                    y_prob,
                    engine_ids,
                    float(th),
                    ma_k,
                    persist_w,
                    min_count,
                )
                candidates.append((float(th), ma_k, persist_w, min_count, p, r, f1))

    if recall_floor_enabled(cfg):
        valid = [c for c in candidates if c[4] >= cfg['min_precision'] and c[5] >= cfg['target_recall_for_selection']]
    else:
        valid = []
    if not valid:
        valid = [c for c in candidates if c[4] >= cfg['min_precision']]
    if not valid:
        valid = candidates

    return choose_from_near_best(valid, f1_idx=6, recall_idx=5, precision_idx=4, threshold_idx=0, cfg=cfg)


def metric_pack(y_true, y_pred, y_prob=None):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'specificity': float(tn / max(tn + fp, 1)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        out['pr_auc'] = float(average_precision_score(y_true, y_prob))
    else:
        out['roc_auc'] = np.nan
        out['pr_auc'] = np.nan
    return out


def _xgb_grid_search(feat_train, y_train, scale_pos, cfg):
    """One-shot grid search for XGBoost hyperparameters (shared across ensemble members)."""
    scorer = make_scorer(fbeta_score, beta=cfg['recall_beta'], zero_division=0)
    base = XGBClassifier(
        objective='binary:logistic',
        n_estimators=cfg['grid_xgb_estimators'],
        scale_pos_weight=scale_pos,
        tree_method='hist',
        eval_metric='logloss',
        random_state=cfg['seed'],
        verbosity=0,
        n_jobs=cfg['xgb_n_jobs'],
    )
    cv = StratifiedKFold(n_splits=cfg['grid_cv_folds'], shuffle=True, random_state=cfg['seed'])
    grid = GridSearchCV(
        estimator=base,
        param_grid=cfg['xgb_param_grid'],
        scoring=scorer,
        cv=cv,
        n_jobs=cfg['grid_n_jobs'],
        verbose=1,
        refit=True,
    )
    grid.fit(feat_train, y_train)
    return grid.best_params_, float(grid.best_score_)


def _train_single_booster(feat_train, y_train, feat_val, y_val, best_params, scale_pos, cfg, member_seed):
    dtrain = xgb.DMatrix(feat_train, label=y_train)
    dval = xgb.DMatrix(feat_val, label=y_val)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'eta': best_params.get('learning_rate', 0.03),
        'max_depth': best_params.get('max_depth', 6),
        'min_child_weight': best_params.get('min_child_weight', 3),
        'subsample': best_params.get('subsample', 0.8),
        'colsample_bytree': best_params.get('colsample_bytree', 0.8),
        'colsample_bylevel': 0.8,
        'reg_alpha': best_params.get('reg_alpha', cfg['xgb_alpha']),
        'reg_lambda': best_params.get('reg_lambda', cfg['xgb_lambda']),
        'scale_pos_weight': scale_pos,
        'tree_method': 'hist',
        'seed': int(member_seed),
    }
    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg['xgb_rounds'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=cfg['xgb_es_rounds'],
        evals_result=evals_result,
        verbose_eval=False,
    )
    return booster, evals_result


def train_xgboost_ensemble(feat_train, y_train, feat_val, y_val, cfg):
    """Train an XGBoost ensemble: one grid search, then K boosters with different seeds."""
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    scale_pos = neg / pos * cfg['pos_weight_boost']

    best_params, best_cv_score = _xgb_grid_search(feat_train, y_train, scale_pos, cfg)

    boosters = []
    last_evals = {}
    ensemble_size = max(1, int(cfg.get('ensemble_size', 1)))
    base_seed = int(cfg['seed'])
    for k in range(ensemble_size):
        member_seed = base_seed + 1009 * k
        booster, evals = _train_single_booster(
            feat_train, y_train, feat_val, y_val, best_params, scale_pos, cfg, member_seed
        )
        boosters.append(booster)
        last_evals = evals
        print(f'  ensemble member {k + 1}/{ensemble_size} trained (seed={member_seed}, best_iter={booster.best_iteration})')
    return boosters, best_params, best_cv_score, scale_pos, last_evals


def ensemble_predict(boosters, X):
    dmat = xgb.DMatrix(X)
    probs = np.zeros(X.shape[0], dtype=np.float64)
    for booster in boosters:
        probs += booster.predict(dmat)
    return (probs / len(boosters)).astype(np.float32)


def fit_isotonic_calibrator(y_val, val_probs):
    """Fit isotonic regression to map raw probs -> calibrated probs."""
    if len(np.unique(y_val)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(val_probs.astype(np.float64), y_val.astype(np.float64))
    return iso


def apply_calibrator(calibrator, probs):
    if calibrator is None:
        return probs
    return calibrator.predict(probs.astype(np.float64)).astype(np.float32)


def build_feature_matrix(model, X, cfg):
    emb = extract_embeddings(model, X)
    if cfg['use_semantic_features']:
        sem = extract_semantic_features(X)
        return np.concatenate([emb, sem], axis=1).astype(np.float32)
    return emb.astype(np.float32)


def save_artifacts(fd, cfg, model, boosters, scaler, feature_cols, metadata):
    out_dir = Path(cfg['artifact_dir']) / fd
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg, 'feature_cols': feature_cols}, out_dir / 'encoder.pt')
    for k, booster in enumerate(boosters):
        booster.save_model(str(out_dir / f'xgb_{k:02d}.ubj'))
    joblib.dump(scaler, out_dir / 'scaler.joblib')
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved artifacts to', out_dir)


def make_production_cfg(base_cfg=None):
    cfg = copy.deepcopy(CFG if base_cfg is None else base_cfg)
    cfg.update({
        'profile': 'production',
        'test_mode': 'sliding',
        'threshold_policy': 'early_alarm',
        'target_recall_for_selection': 0.97,
        'f1_tolerance': 0.02,
        'use_engine_smoothing': True,
    })
    return cfg


def make_benchmark_cfg(base_cfg=None):
    cfg = copy.deepcopy(CFG if base_cfg is None else base_cfg)
    cfg.update({
        'profile': 'benchmark',
        'test_mode': 'last',
        'threshold_policy': 'balanced',
        'target_recall_for_selection': None,
        'f1_tolerance': 0.005,
    })
    return cfg


def run_pipeline(fd, cfg, return_artifacts=False):
    print('\n' + '=' * 72)
    print('Dataset:', fd)
    print('Profile:', cfg.get('profile', 'custom'), '| test_mode:', cfg['test_mode'], '| threshold_policy:', cfg.get('threshold_policy'))
    print('Ensemble size:', cfg.get('ensemble_size', 1), '| testlike_val:', cfg.get('use_testlike_val_threshold'), '| calibrate:', cfg.get('calibrate_probabilities'))
    print('=' * 72)
    seed_everything(cfg['seed'], cfg.get('deterministic', True))

    train_raw, test_raw, rul_raw = load_raw(fd, cfg['data_dir'])
    train_raw = add_train_rul(train_raw)
    test_raw = add_test_rul(test_raw, rul_raw)
    train_df, val_df = split_train_val_by_engine(train_raw, cfg['val_engine_frac'], cfg['seed'])
    feature_cols = get_feature_columns(train_df)

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_raw[feature_cols] = scaler.transform(test_raw[feature_cols])

    seq_len = cfg['seq_len']
    stride = cfg['stride']
    threshold = cfg['rul_threshold']
    X_train, y_train, rul_train, eng_train = make_sliding_windows(train_df, feature_cols, seq_len, stride, threshold)
    X_val, y_val, rul_val, eng_val = make_sliding_windows(val_df, feature_cols, seq_len, stride, threshold)
    if cfg['test_mode'] == 'last':
        X_test, y_test, rul_test, eng_test = make_test_windows_last(test_raw, rul_raw, feature_cols, seq_len, threshold)
    else:
        X_test, y_test, rul_test, eng_test = make_sliding_windows(test_raw, feature_cols, seq_len, stride, threshold)

    use_testlike = bool(cfg.get('use_testlike_val_threshold')) and cfg['test_mode'] == 'last'
    if use_testlike:
        X_val_tl, y_val_tl, rul_val_tl, eng_val_tl = make_testlike_val_windows(
            val_df, feature_cols, seq_len, threshold,
            samples_per_engine=int(cfg.get('testlike_samples_per_engine', 8)),
            seed=int(cfg['seed']),
        )
    else:
        X_val_tl = y_val_tl = rul_val_tl = eng_val_tl = None

    print('Features:', len(feature_cols), feature_cols)
    print('Train windows:', X_train.shape, 'positive ratio:', round(float(y_train.mean()), 4))
    print('Val windows  :', X_val.shape, 'positive ratio:', round(float(y_val.mean()), 4))
    if use_testlike:
        print('Testlike val :', X_val_tl.shape, 'positive ratio:', round(float(y_val_tl.mean()), 4))
    print('Test windows :', X_test.shape, 'positive ratio:', round(float(y_test.mean()), 4))

    model, history, best_encoder_score = train_encoder(X_train, y_train, X_val, y_val, cfg)
    print('Best encoder validation F-beta:', round(best_encoder_score, 4))

    feat_train = build_feature_matrix(model, X_train, cfg)
    feat_val = build_feature_matrix(model, X_val, cfg)
    feat_test = build_feature_matrix(model, X_test, cfg)
    feat_val_tl = build_feature_matrix(model, X_val_tl, cfg) if use_testlike else None
    print('Feature matrix:', feat_train.shape, feat_val.shape, feat_test.shape)

    boosters, best_params, best_cv_score, scale_pos, evals_result = train_xgboost_ensemble(
        feat_train, y_train, feat_val, y_val, cfg,
    )
    print('Best XGBoost CV F-beta:', round(best_cv_score, 4))
    print('Best XGBoost params  :', best_params)
    print('Ensemble members     :', len(boosters))

    val_probs_raw = ensemble_predict(boosters, feat_val)
    test_probs_raw = ensemble_predict(boosters, feat_test)

    calibrator = None
    if cfg.get('calibrate_probabilities', False):
        calibrator = fit_isotonic_calibrator(y_val, val_probs_raw)
    val_probs = apply_calibrator(calibrator, val_probs_raw)
    test_probs = apply_calibrator(calibrator, test_probs_raw)

    if use_testlike:
        val_tl_probs_raw = ensemble_predict(boosters, feat_val_tl)
        val_tl_probs = apply_calibrator(calibrator, val_tl_probs_raw)
        win_th, win_p_val, win_r_val, win_f1_val = choose_window_threshold(y_val_tl, val_tl_probs, cfg)
    else:
        win_th, win_p_val, win_r_val, win_f1_val = choose_window_threshold(y_val, val_probs, cfg)

    win_p_test, win_r_test, win_f1_test, win_pred_test = evaluate_threshold(y_test, test_probs, win_th)
    win_metrics = metric_pack(y_test, win_pred_test, test_probs)
    win_metrics.update({'threshold': float(win_th), 'val_precision': win_p_val, 'val_recall': win_r_val, 'val_f1': win_f1_val})

    engine_metrics = None
    engine_setting = None
    if cfg['use_engine_smoothing'] and cfg['test_mode'] == 'sliding':
        engine_setting = choose_engine_setting(y_val, val_probs, eng_val, cfg)
        e_th, e_ma, e_pw, e_mc, e_p_val, e_r_val, e_f1_val = engine_setting
        e_p_test, e_r_test, e_f1_test, e_pred_test = evaluate_engine_smoothed(
            y_test,
            test_probs,
            eng_test,
            e_th,
            e_ma,
            e_pw,
            e_mc,
        )
        engine_metrics = metric_pack(y_test, e_pred_test, test_probs)
        engine_metrics.update({
            'threshold': float(e_th),
            'ma_k': int(e_ma),
            'persist_window': int(e_pw),
            'min_count': int(e_mc),
            'val_precision': e_p_val,
            'val_recall': e_r_val,
            'val_f1': e_f1_val,
        })

    if engine_metrics is not None and engine_metrics['val_f1'] >= win_metrics['val_f1']:
        final_mode = 'engine'
        final_metrics = engine_metrics
    else:
        final_mode = 'window'
        final_metrics = win_metrics

    row = {
        'dataset': fd,
        'decision_mode': final_mode,
        'precision': final_metrics['precision'],
        'recall': final_metrics['recall'],
        'f1': final_metrics['f1'],
        'specificity': final_metrics['specificity'],
        'roc_auc': final_metrics['roc_auc'],
        'pr_auc': final_metrics['pr_auc'],
        'tp': final_metrics['tp'],
        'fp': final_metrics['fp'],
        'fn': final_metrics['fn'],
        'tn': final_metrics['tn'],
        'best_encoder_val_fbeta': best_encoder_score,
        'best_xgb_cv_fbeta': best_cv_score,
        'scale_pos_weight': scale_pos,
        'seed': cfg['seed'],
        'deterministic': bool(cfg.get('deterministic', True)),
        'profile': cfg.get('profile', 'custom'),
        'test_mode': cfg['test_mode'],
        'threshold_policy': cfg.get('threshold_policy'),
        'rul_threshold': cfg['rul_threshold'],
        'seq_len': cfg['seq_len'],
        'best_xgb_params': best_params,
        'window_metrics': win_metrics,
        'engine_metrics': engine_metrics,
        'ensemble_size': len(boosters),
        'calibrated': calibrator is not None,
        'testlike_val_used': bool(use_testlike),
    }

    print('\nValidation-selected decision mode:', final_mode)
    print('Decision rule : selected by validation F1, not by test F1')
    print('Window VAL   : P={:.4f} R={:.4f} F1={:.4f}'.format(win_metrics['val_precision'], win_metrics['val_recall'], win_metrics['val_f1']))
    print('Window test  : P={:.4f} R={:.4f} F1={:.4f} thr={:.4f}'.format(win_metrics['precision'], win_metrics['recall'], win_metrics['f1'], win_metrics['threshold']))
    if engine_metrics is not None:
        print('Engine VAL   : P={:.4f} R={:.4f} F1={:.4f}'.format(engine_metrics['val_precision'], engine_metrics['val_recall'], engine_metrics['val_f1']))
        print('Engine test  : P={:.4f} R={:.4f} F1={:.4f} thr={:.2f} ma={} persist=({},{})'.format(
            engine_metrics['precision'],
            engine_metrics['recall'],
            engine_metrics['f1'],
            engine_metrics['threshold'],
            engine_metrics['ma_k'],
            engine_metrics['persist_window'],
            engine_metrics['min_count'],
        ))
    print('Final test   : P={:.4f} R={:.4f} F1={:.4f} AUC={:.4f}'.format(row['precision'], row['recall'], row['f1'], row['roc_auc']))

    if cfg['save_artifacts']:
        save_artifacts(fd, cfg, model, boosters, scaler, feature_cols, row)

    if return_artifacts:
        artifacts = {
            'model': model,
            'boosters': boosters,
            'calibrator': calibrator,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'history': history,
            'evals_result': evals_result,
            'test_probs': test_probs,
            'y_test': y_test,
            'eng_test': eng_test,
            'rul_test': rul_test,
        }
        return row, artifacts
    return row
