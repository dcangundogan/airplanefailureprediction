# IEEE Article Result Report: TCN-BiGRU + XGBoost

## Scope

This package summarizes the CMAPSS near-failure detection results for the TCN-BiGRU + XGBoost method. The binary target is near failure when RUL < 30 cycles and normal otherwise. The reported model first learns a temporal representation using causal TCN blocks, bidirectional GRU layers, and attention pooling; the 128-dimensional embedding is then passed to an XGBoost classifier.

## Dataset

The experiments use all four NASA CMAPSS subsets: FD001, FD002, FD003, and FD004. FD001 and FD003 contain one operating condition, while FD002 and FD004 contain six operating conditions. FD003 and FD004 are more complex because they contain two fault modes instead of one. Dataset counts are generated directly from the local files in `data/`.

Use `tables/dataset_characteristics.tex` for the IEEE dataset table and `figures/fig_dataset_engine_counts.svg` for an engine-count plot.

## Main Results

Across all four subsets, TCN-BiGRU + XGBoost obtained mean F1 = 0.9089, mean ROC-AUC = 0.9941, and mean PR-AUC = 0.9815. The highest F1 was observed on FD002 (0.9302), while the lowest F1 was observed on FD001 (0.8889). FD004 remained challenging in ranking quality and average precision because it combines multiple operating conditions with two fault modes.

The model produced high ROC-AUC on every subset: 0.9968 on FD001, 0.9980 on FD002, 0.9988 on FD003, and 0.9826 on FD004. This indicates strong ranking quality even on the harder multi-condition subsets. Precision and recall show the expected safety trade-off: FD001 reached perfect precision with lower recall, whereas FD002 reached perfect recall with moderate false positives.

## Comparison

The comparison table uses test-last F1 values reported in the existing notebooks. Mean F1 values are:

- TCN-GRU + XGBoost: 0.8409
- PatchTST + XGBoost: 0.8789
- ABGRU: 0.9069
- TCN-BiGRU + XGBoost: 0.9089

The proposed TCN-BiGRU + XGBoost has the strongest average F1 among the compared methods, although ABGRU is very close and is strongest on FD002 and FD003. This should be written carefully in the paper: the proposed method improves average cross-dataset F1, not every individual subset.

## Suggested IEEE Result Paragraph

Table III reports the near-failure classification results of the TCN-BiGRU + XGBoost model on all CMAPSS subsets. The method achieved an average F1-score of 0.9089 and an average ROC-AUC of 0.9941. The best subset-level F1-score was obtained on FD002, while FD004 remained the most challenging subset due to the combination of six operating conditions and two fault modes. Compared with TCN-GRU + XGBoost and PatchTST + XGBoost baselines, the proposed sequential TCN-BiGRU representation achieved the best mean F1-score across the four datasets. These results suggest that extracting local temporal degradation patterns with TCN layers before bidirectional recurrent modeling provides a robust representation for failure detection.

## Suggested Figure Captions

Fig. 1. Overall architecture of the TCN-BiGRU + XGBoost near-failure detection pipeline.

Fig. 2. Cross-dataset precision, recall, F1-score, ROC-AUC, PR-AUC, and specificity of the proposed method.

Fig. 3. Test-last F1-score comparison across CMAPSS subsets for TCN-GRU + XGBoost, PatchTST + XGBoost, ABGRU, and TCN-BiGRU + XGBoost.

Fig. 4. Confusion matrices of the proposed method on the final test sequence of each engine.

## Notes for Submission

The numbers in this report come from executed notebook outputs in this repository. Before final IEEE submission, re-run the final notebook or script once with fixed seeds and save the console output, because GPU libraries and XGBoost versions can slightly change results.
