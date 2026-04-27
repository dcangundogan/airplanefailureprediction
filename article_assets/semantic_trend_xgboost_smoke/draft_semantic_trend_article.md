# Semantic Trend Embedding and XGBoost for Turbofan Near-Failure Detection

## Abstract

This study proposes a semantic trend embedding assisted XGBoost framework for
near-failure detection in aircraft turbofan engines using the NASA C-MAPSS
dataset. A near-failure state is defined as RUL <= 30 cycles.
For each engine cycle, the method extracts rolling degradation statistics from a
30-cycle sensor window and converts the strongest sensor trends
into compact textual descriptions. These descriptions are encoded using the
`sentence-transformers/all-MiniLM-L6-v2` sentence embedding model and evaluated alone and in combination with
numeric window statistics. Experiments on FD001-FD004 show that the best feature
set achieves a mean last-cycle F1-score of 0.9388. Ablation results
confirm whether semantic trend embeddings add value beyond numeric window
statistics.

## Contributions

1. A leakage-controlled CMAPSS near-failure detection protocol with engine-level
   train-validation splitting and train-only preprocessing.
2. A semantic trend representation that converts rolling sensor slopes and
   variability into sentence embeddings.
3. A cross-dataset ablation comparing numeric-only, semantic-only, and hybrid
   XGBoost classifiers on FD001-FD004.

## Experimental Setup

Use `tables/semantic_trend_hyperparameters.tex` for the implementation settings.
Use `tables/semantic_trend_ablation_results.tex` and
`tables/semantic_trend_last_f1_pivot.tex` for the primary results. Use
`figures/fig_semantic_trend_ablation_f1.svg` for the F1 comparison.

## Wording Note

Do not describe the method as unsupervised anomaly detection. The correct task
name is supervised near-failure classification. Do not claim a generative LLM is
used; the method uses a pretrained sentence embedding model.
