"""Run the CMAPSS hybrid pipeline locally (no Colab) and generate paper figures.

Examples
--------
  python run_local.py                  # FAST single-dataset run (FD001) + all single-dataset figures
  python run_local.py --dataset FD002  # fast run on a different subset
  python run_local.py --full           # paper-quality config (slow on CPU: ~1h+)
  python run_local.py --batch          # also run FD001-FD004 and make the benchmark bar chart
  python run_local.py --ablation       # also run the ablation study (very slow on CPU)

Figures are written to ./figures as both .pdf (use in LaTeX) and .png.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

import cmapss_pipeline as P  # extracted notebook definitions

import matplotlib
matplotlib.use('Agg')  # no display needed; we save to files
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve

FIG_DIR = Path('figures')
C = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00',
    'purple': '#CC79A7', 'sky': '#56B4E9', 'yellow': '#F0E442', 'gray': '#999999',
}


def setup_style():
    mpl.rcParams.update({
        'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.size': 11, 'font.family': 'serif', 'axes.titlesize': 12, 'axes.labelsize': 11,
        'axes.grid': True, 'grid.alpha': 0.30, 'grid.linestyle': '--',
        'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
    })
    FIG_DIR.mkdir(exist_ok=True)


def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'{name}.{ext}')
    plt.close(fig)
    print('  saved ->', (FIG_DIR / f'{name}.pdf').as_posix())


# ----------------------------------------------------------------------------
# Config builders
# ----------------------------------------------------------------------------
def fast_overrides(cfg, dataset, epochs):
    cfg = copy.deepcopy(cfg)
    cfg.update({
        'dataset': dataset,
        'epochs': epochs, 'patience': 6, 'stride': 2,
        'ensemble_size': 1,
        'use_testlike_val_threshold': True,
        'calibrate_probabilities': True,
        'grid_cv_folds': 2, 'grid_xgb_estimators': 200,
        'xgb_rounds': 300, 'xgb_es_rounds': 30,
        'xgb_param_grid': {
            'max_depth': [4], 'learning_rate': [0.05], 'min_child_weight': [3],
            'subsample': [0.8], 'colsample_bytree': [0.8], 'reg_alpha': [0.0], 'reg_lambda': [1.0],
        },
    })
    return cfg


def threshold_sweep_table(y_true, y_prob):
    rows = []
    for th in np.round(np.arange(0.05, 0.95, 0.05), 2):
        p, r, f1, pred = P.evaluate_threshold(y_true, y_prob, float(th))
        rows.append({'threshold': float(th), 'precision': p, 'recall': r, 'f1': f1})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Single-dataset figures
# ----------------------------------------------------------------------------
def figs_single(result, art, cfg):
    y = np.asarray(art['y_test']); p = np.asarray(art['test_probs'])
    rul = np.asarray(art['rul_test'])
    wm = result['window_metrics']
    sel_thr = wm['threshold']

    # Encoder training curves
    hist = art['history']
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    l1, = ax1.plot(hist['epoch'], hist['train_loss'], color=C['red'], label='Train loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training loss', color=C['red'])
    ax1.tick_params(axis='y', labelcolor=C['red'])
    ax2 = ax1.twinx(); ax2.grid(False)
    l2, = ax2.plot(hist['epoch'], hist['val_fbeta'], color=C['blue'], label=r'Val F$_\beta$')
    l3, = ax2.plot(hist['epoch'], hist['val_auc'], color=C['green'], ls='--', label='Val ROC-AUC')
    ax2.set_ylabel('Validation metric')
    ax1.legend(handles=[l1, l2, l3], loc='center right')
    ax1.set_title('Encoder training dynamics')
    save_fig(fig, 'fig_encoder_training')

    # XGBoost learning curves
    ev = art.get('evals_result') or {}
    if ev:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for split, color in [('train', C['blue']), ('val', C['orange'])]:
            if split in ev and 'logloss' in ev[split]:
                axes[0].plot(ev[split]['logloss'], color=color, label=split)
            if split in ev and 'auc' in ev[split]:
                axes[1].plot(ev[split]['auc'], color=color, label=split)
        axes[0].set_title('XGBoost log-loss'); axes[0].set_xlabel('Boosting round'); axes[0].set_ylabel('Log-loss'); axes[0].legend()
        axes[1].set_title('XGBoost ROC-AUC'); axes[1].set_xlabel('Boosting round'); axes[1].set_ylabel('AUC'); axes[1].legend()
        fig.tight_layout(); save_fig(fig, 'fig_xgb_learning')

    # ROC + PR
    fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
    base = float(y.mean())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].plot(fpr, tpr, color=C['blue'], lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], color=C['gray'], ls='--', lw=1)
    axes[0].set_xlabel('False positive rate'); axes[0].set_ylabel('True positive rate'); axes[0].set_title('ROC curve'); axes[0].legend(loc='lower right')
    axes[1].plot(rec, prec, color=C['green'], lw=2, label=f'AP = {ap:.3f}')
    axes[1].axhline(base, color=C['gray'], ls='--', lw=1, label=f'No-skill = {base:.2f}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('Precision-Recall curve'); axes[1].legend(loc='lower left')
    fig.tight_layout(); save_fig(fig, 'fig_roc_pr')

    # Confusion matrix
    cm = np.array([[wm['tn'], wm['fp']], [wm['fn'], wm['tp']]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Healthy', 'Failure'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Healthy', 'Failure'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    th = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:d}', ha='center', va='center', color='white' if cm[i, j] > th else 'black', fontsize=15)
    ax.set_title(f'Confusion matrix (thr = {sel_thr:.3f})'); ax.grid(False)
    save_fig(fig, 'fig_confusion_matrix')

    # Threshold sweep
    ts = threshold_sweep_table(y, p)
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(ts['threshold'], ts['precision'], color=C['blue'], label='Precision')
    ax.plot(ts['threshold'], ts['recall'], color=C['orange'], label='Recall')
    ax.plot(ts['threshold'], ts['f1'], color=C['green'], lw=2.3, label='F1')
    ax.axvline(sel_thr, color=C['red'], ls='--', label=f'Selected = {sel_thr:.3f}')
    ax.set_xlabel('Decision threshold'); ax.set_ylabel('Score'); ax.set_title('Threshold sensitivity'); ax.legend()
    save_fig(fig, 'fig_threshold_sweep')

    # Probability distribution
    bins = np.linspace(0, 1, 31)
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.hist(p[y == 0], bins=bins, color=C['blue'], alpha=0.6, density=True, label='Healthy')
    ax.hist(p[y == 1], bins=bins, color=C['red'], alpha=0.6, density=True, label='Failure')
    ax.axvline(sel_thr, color='black', ls='--', label='Threshold')
    ax.set_xlabel('Predicted failure probability'); ax.set_ylabel('Density'); ax.set_title('Class-conditional probability distributions'); ax.legend()
    save_fig(fig, 'fig_prob_distribution')

    # Calibration
    try:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], color=C['gray'], ls='--', label='Perfectly calibrated')
        ax.plot(mean_pred, frac_pos, marker='o', color=C['purple'], label='Model')
        ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed frequency'); ax.set_title('Reliability diagram'); ax.legend(loc='upper left')
        save_fig(fig, 'fig_calibration')
    except Exception as e:
        print('  skipped calibration:', e)

    # RUL vs probability
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.scatter(rul[y == 0], p[y == 0], s=12, alpha=0.5, color=C['blue'], label='Healthy')
    ax.scatter(rul[y == 1], p[y == 1], s=12, alpha=0.5, color=C['red'], label='Failure')
    ax.axvline(cfg['rul_threshold'], color='black', ls='--', label=f"RUL threshold = {cfg['rul_threshold']}")
    ax.set_xlabel('True RUL (cycles)'); ax.set_ylabel('Predicted failure probability'); ax.set_title('Predicted probability vs. remaining useful life'); ax.legend()
    save_fig(fig, 'fig_rul_vs_prob')

    # Feature importance
    booster = art['boosters'][0]
    fcols = art['feature_cols']
    STAT_NAMES = ['mean', 'std', 'min', 'max', 'range', 'drift', 'slope', 'dabs', 'dstd', 'energy', 'pos_frac', 'neg_frac']
    names = [f'emb_{i:03d}' for i in range(cfg['embed_dim'])]
    if cfg['use_semantic_features']:
        names += [f'{s}:{st}' for s in fcols for st in STAT_NAMES]
    score = booster.get_score(importance_type='gain')
    imp = np.zeros(len(names))
    for k, v in score.items():
        idx = int(k[1:])
        if idx < len(names):
            imp[idx] = v
    order = np.argsort(imp)[::-1][:20][::-1]
    colors = [C['green'] if names[i].startswith('emb_') else C['orange'] for i in order]
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.barh(np.arange(len(order)), imp[order], color=colors)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel('Importance (gain)'); ax.set_title('Top-20 XGBoost feature importances')
    ax.legend(handles=[Patch(color=C['green'], label='Deep embedding'), Patch(color=C['orange'], label='Semantic feature')], loc='lower right')
    ax.grid(axis='y')
    save_fig(fig, 'fig_feature_importance')


def fig_data_overview(cfg):
    tr, _, _ = P.load_raw(cfg['dataset'], cfg['data_dir'])
    tr = P.add_train_rul(tr)
    fcols = P.get_feature_columns(tr)
    eng_id = int(tr['unit_number'].unique()[0])
    g = tr[tr['unit_number'] == eng_id].sort_values('time_in_cycles')
    show = [c for c in ['s2', 's3', 's4', 's7', 's11', 's12', 's15', 's17', 's20', 's21'] if c in fcols][:6]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c in show:
        v = g[c].values.astype(float)
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        ax.plot(g['time_in_cycles'], v, lw=1.4, label=c)
    ax.set_xlabel('Cycle'); ax.set_ylabel('Normalized sensor value')
    ax.set_title(f'Sensor degradation trajectories (engine {eng_id}, {cfg["dataset"]})')
    ax.legend(ncol=3, fontsize=9); save_fig(fig, 'fig_sensor_degradation')

    ruls = tr['RUL'].values.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(ruls, bins=50, color=C['blue'], alpha=0.85)
    axes[0].axvline(cfg['rul_threshold'], color=C['red'], ls='--', label=f"threshold = {cfg['rul_threshold']}")
    axes[0].set_xlabel('RUL (cycles)'); axes[0].set_ylabel('Count'); axes[0].set_title('RUL distribution'); axes[0].legend()
    lab = (ruls <= cfg['rul_threshold']).astype(int)
    vals = [int((lab == 0).sum()), int((lab == 1).sum())]
    axes[1].bar(['Healthy', 'Failure'], vals, color=[C['blue'], C['red']])
    for i, v in enumerate(vals):
        axes[1].text(i, v, f'{v}\n({v / len(lab) * 100:.1f}%)', ha='center', va='bottom')
    axes[1].set_ylabel('Samples'); axes[1].set_title(f'Class balance ({cfg["dataset"]})'); axes[1].grid(False)
    fig.tight_layout(); save_fig(fig, 'fig_data_overview')


def fig_benchmark(batch_results):
    br = batch_results.set_index('dataset')
    metrics = ['precision', 'recall', 'f1', 'roc_auc']; labels = ['Precision', 'Recall', 'F1', 'ROC-AUC']
    colors = [C['blue'], C['orange'], C['green'], C['purple']]
    x = np.arange(len(br)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    for i, (mt, lab) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1.5) * w, br[mt], width=w, color=colors[i], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(br.index); ax.set_ylim(0, 1.08); ax.set_ylabel('Score')
    ax.set_title('Performance across CMAPSS subsets')
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.06))
    save_fig(fig, 'fig_benchmark_datasets')


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='FD001', choices=['FD001', 'FD002', 'FD003', 'FD004'])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--full', action='store_true', help='use the heavy paper config (slow on CPU)')
    ap.add_argument('--batch', action='store_true', help='also run all four subsets for the benchmark figure')
    args = ap.parse_args()

    setup_style()

    base = P.make_benchmark_cfg(P.CFG)
    if args.full:
        cfg = copy.deepcopy(base); cfg['dataset'] = args.dataset
        print('Mode: FULL (paper config) — this is slow on CPU.')
    else:
        cfg = fast_overrides(base, args.dataset, args.epochs)
        print(f'Mode: FAST — dataset={args.dataset}, epochs={args.epochs}, ensemble=1, tiny grid.')

    print('\n>>> Running single-dataset pipeline...')
    result, art = P.run_pipeline(cfg['dataset'], cfg, return_artifacts=True)

    print('\n>>> Generating figures...')
    fig_data_overview(cfg)
    figs_single(result, art, cfg)

    summary = {k: result[k] for k in ['dataset', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'tp', 'fp', 'fn', 'tn']}
    print('\nTest metrics:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})

    if args.batch:
        print('\n>>> Running FD001-FD004 benchmark...')
        rows = []
        for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
            rows.append(P.run_pipeline(fd, cfg, return_artifacts=False))
        batch_results = pd.DataFrame(rows)
        fig_benchmark(batch_results)
        batch_results.to_csv(FIG_DIR / 'benchmark_results.csv', index=False)
        print('\n% LaTeX benchmark table:')
        print(batch_results[['dataset', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']].to_latex(index=False, float_format='%.3f'))

    print('\nDone. Figures in', FIG_DIR.resolve())


if __name__ == '__main__':
    main()
