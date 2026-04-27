# Semantic Trend XGBoost Article Result Report

## Scope

This report summarizes the article-ready CMAPSS near-failure detection experiment
for the semantic trend embedding + XGBoost pipeline. The task is supervised binary
classification: a cycle is labelled near-failure when RUL <= 30
cycles.

## Method

The pipeline computes 30-cycle rolling statistics for each selected
sensor. For FD002 and FD004, operating-condition normalization is fit only on the
engine-level training split and then applied to validation and test data. Semantic
trend sentences are built from the top 8 absolute-slope
sensors in each window and embedded with `sentence-transformers/all-MiniLM-L6-v2`. XGBoost is evaluated with
three feature sets: numeric window statistics, semantic embeddings, and the hybrid
concatenation of both.

## Main Result

The strongest mean last-cycle F1-score is obtained by `numeric` with mean
F1 = 0.9388. The full ablation table is available at
`tables/semantic_trend_ablation_results.tex`, and the compact comparison table is
available at `tables/semantic_trend_last_f1_pivot.tex`.

## Article Claim

The defensible claim is that language-model sentence embeddings can be used as a
textual trend representation for CMAPSS near-failure detection, and their value
must be reported through the numeric-only, semantic-only, and hybrid ablation.
Avoid calling the method a generative LLM system; the encoder is a sentence
embedding model.

## Reproducibility

- Python: 3.9.13
- Platform: Windows-10-10.0.26200-SP0
- Random seed: 42
- Runtime: 0.0 minutes
- Command: `semantic_trend_xgboost_article_ready.py --datasets FD001 --methods numeric --n-estimators 80 --out-dir article_assets\semantic_trend_xgboost_smoke`
