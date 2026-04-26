# TCN-BiGRU and XGBoost-Based Near-Failure Detection for Aircraft Turbofan Engines

**Authors:** [FILL: Author 1], [FILL: Author 2], [FILL: Author 3]  
**Affiliation:** [FILL: University / Department / City / Country]  
**Conference:** [FILL: IEEE conference name]

## Abstract

Predictive maintenance of aircraft engines requires accurate identification of near-failure operating states from multivariate sensor sequences. This study proposes a hybrid TCN-BiGRU + XGBoost approach for near-failure detection using the NASA C-MAPSS turbofan engine degradation dataset. The proposed model first extracts local temporal degradation patterns using causal temporal convolutional network (TCN) blocks, then models bidirectional long-range temporal dependencies using a BiGRU layer and attention pooling. The learned embedding is finally classified by XGBoost to distinguish normal and near-failure engine states. Experiments are conducted on all four C-MAPSS subsets, FD001-FD004. The proposed method achieves an average F1-score of 0.9089, average ROC-AUC of 0.9941, and average PR-AUC of 0.9815 across the four datasets. These results indicate that combining convolutional temporal feature extraction, recurrent sequence modeling, and gradient-boosted classification provides a robust framework for aircraft engine failure detection.

**Keywords:** predictive maintenance, turbofan engine, C-MAPSS, TCN, BiGRU, XGBoost, anomaly detection, remaining useful life

## I. Introduction

Aircraft engine reliability is a critical requirement for operational safety, maintenance planning, and cost reduction. Modern turbofan engines are monitored using multivariate sensor streams that reflect changes in pressure, temperature, speed, and other operating conditions. As degradation progresses, these sensor signals contain temporal patterns that can be used to identify near-failure behavior before complete engine failure occurs.

Traditional maintenance strategies rely on scheduled inspections or rule-based thresholds. However, fixed thresholds may not capture complex nonlinear degradation patterns, especially when operating conditions and fault modes vary across engines. Data-driven predictive maintenance methods address this limitation by learning degradation signatures directly from historical sensor data.

The NASA C-MAPSS dataset is widely used for evaluating prognostics and health management algorithms. It contains run-to-failure simulations for turbofan engines under different operating conditions and fault modes. Most studies formulate the task as remaining useful life (RUL) regression, but near-failure detection is also important because maintenance decisions are often made as binary or multi-class health-state decisions.

This paper proposes a hybrid TCN-BiGRU + XGBoost architecture for near-failure detection. In the proposed approach, TCN layers capture local temporal degradation patterns, BiGRU layers model forward and backward temporal dependencies, attention pooling produces a compact engine health embedding, and XGBoost performs the final classification. The binary target is defined as near failure when RUL < 30 cycles and normal otherwise.

The main contributions of this study are:

1. [FILL: Contribution 1. Example: A sequential TCN-BiGRU representation learning architecture for CMAPSS near-failure detection.]
2. [FILL: Contribution 2. Example: A cross-dataset evaluation on FD001-FD004 instead of only FD001.]
3. [FILL: Contribution 3. Example: A comparison with TCN-GRU + XGBoost, PatchTST + XGBoost, and ABGRU baselines.]

## II. Related Work

Deep learning has been widely applied to turbofan engine prognostics using the C-MAPSS dataset. Recurrent neural networks such as LSTM and GRU models are commonly used because engine degradation is sequential in nature. Attention-based recurrent models further improve interpretability and temporal weighting by allowing the model to focus on informative time steps.

Temporal convolutional networks have also been used for time-series modeling because dilated causal convolutions can capture local and medium-range temporal dependencies efficiently. Compared with recurrent models, TCNs are easier to parallelize and often provide stable training behavior.

Transformer-based methods and PatchTST-style architectures have recently been applied to time-series classification and RUL prediction. These models can capture long-range dependencies through self-attention, but they may require careful tuning and more computational resources.

Gradient boosting methods such as XGBoost remain strong classifiers for structured representations. In hybrid deep-learning pipelines, neural networks can be used as feature extractors, while XGBoost performs final classification on learned embeddings.

[FILL: Add 5-8 IEEE-formatted citations here. Suggested citation topics: CMAPSS dataset, TCN, GRU/BiGRU, XGBoost, PatchTST/time-series transformer, aircraft predictive maintenance.]

## III. Dataset and Problem Definition

### A. C-MAPSS Dataset

The experiments use the NASA C-MAPSS turbofan engine degradation dataset. It contains four subsets: FD001, FD002, FD003, and FD004. These subsets differ in the number of operating conditions and fault modes. FD001 contains one operating condition and one fault mode, while FD004 is the most complex subset because it contains six operating conditions and two fault modes.

Use this table in the paper:

`article_assets/tables/dataset_characteristics.tex`

### B. Label Definition

For each training engine, RUL is computed as:

```text
RUL = max_cycle(engine) - current_cycle
```

The binary label is defined as:

```text
Near-failure = 1, if RUL < 30
Normal       = 0, otherwise
```

[FILL: Explain why threshold 30 was selected. Example: It follows common CMAPSS health-state definitions and represents a practical maintenance warning region.]

## IV. Proposed Method

The proposed architecture is shown in:

`article_assets/figures/fig_tcn_bigru_xgboost_architecture.svg`

For a more detailed version of the same notebook architecture, use:

`article_assets/figures/fig_tcn_bigru_xgboost_architecture_detailed.svg`

### A. Sliding Window Generation

Each engine trajectory is converted into fixed-length temporal windows. A window contains sensor and operating-condition measurements over consecutive cycles. The model uses a sequence length of 40 cycles. Each window is assigned the label of its final time step.

[FILL: Add exact preprocessing details from your final notebook: selected sensors, normalization method, stride, train/validation split.]

### B. TCN Feature Extraction

The first stage uses causal TCN blocks with dilated convolutions. These layers learn local degradation patterns while preserving temporal order. The TCN channel configuration is `[64, 128, 128, 64]`, with residual connections and weight normalization.

### C. BiGRU Temporal Modeling

The TCN output is passed to a bidirectional GRU. The BiGRU models temporal dependencies in both forward and backward directions over the extracted TCN features. This is useful because the complete degradation window is available during offline classification.

### D. Attention Pooling

Attention pooling converts the sequence of BiGRU hidden states into a fixed-length embedding by assigning higher weights to more informative time steps. The final embedding dimension is 128.

### E. XGBoost Classification

The learned embedding is used as input to an XGBoost binary classifier. A validation-optimized probability threshold is used to convert classifier probabilities into normal or near-failure predictions.

Use this table for the hyperparameters:

`article_assets/tables/tcn_bigru_hyperparameters.tex`

## V. Experimental Setup

All experiments are conducted on FD001-FD004. The evaluation uses the final available test sequence for each engine, which corresponds to a realistic decision point where the latest engine state is classified as normal or near failure.

The following metrics are reported:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Specificity

The F1-score is emphasized because the near-failure class is imbalanced and both missed failures and false alarms are important.

[FILL: Add software/hardware details. Example: Python version, PyTorch version, CUDA/GPU, XGBoost version, random seed.]

## VI. Results and Discussion

### A. Proposed Method Results

The proposed TCN-BiGRU + XGBoost model achieves strong results across all CMAPSS subsets. The mean F1-score is 0.9089, the mean ROC-AUC is 0.9941, and the mean PR-AUC is 0.9815.

Use this table:

`article_assets/tables/proposed_tcn_bigru_xgboost_results.tex`

Use this figure:

`article_assets/figures/fig_proposed_cross_dataset_metrics.svg`

The highest F1-score is obtained on FD002, while FD004 remains challenging because it combines multiple operating conditions with two fault modes. The high ROC-AUC values across all subsets show that the model ranks near-failure samples well even when the classification threshold changes.

### B. Confusion Matrix Analysis

Use this figure:

`article_assets/figures/fig_proposed_confusion_matrices.svg`

The confusion matrices show the trade-off between false alarms and missed near-failure engines. FD001 achieves perfect precision, meaning no normal engine is incorrectly classified as near failure. FD002 achieves perfect recall, meaning all near-failure engines are detected.

### C. Comparison with Other Methods

Use this table:

`article_assets/tables/method_comparison_f1.tex`

Use this figure:

`article_assets/figures/fig_method_comparison_f1.svg`

The proposed method obtains the best average F1-score among the compared methods. However, ABGRU performs slightly better on FD002 and FD003. Therefore, the main claim should be written as average cross-dataset improvement, not superiority on every individual dataset.

### D. Architecture Discussion

Use this table:

`article_assets/tables/architecture_comparison.tex`

Compared with the prior TCN-GRU + XGBoost baseline, the proposed model uses a sequential architecture in which the TCN first extracts local temporal features and the BiGRU then models temporal dependencies over those features. This design provides a hierarchical representation of degradation behavior.

## VII. Conclusion

This study presented a hybrid TCN-BiGRU + XGBoost method for near-failure detection in aircraft turbofan engines. The model combines causal temporal convolution, bidirectional recurrent modeling, attention pooling, and gradient-boosted classification. Experiments on all four C-MAPSS subsets demonstrate strong classification performance, with an average F1-score of 0.9089 and average ROC-AUC of 0.9941. The results suggest that sequential TCN-BiGRU feature learning provides a robust representation for engine health-state classification.

Future work will investigate [FILL: examples: multi-class health-state classification, RUL regression, explainability with attention or SHAP, real-time deployment, uncertainty estimation, testing on additional datasets].

## References

[1] A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," in *Proc. Int. Conf. Prognostics and Health Management*, 2008.

[2] [FILL: TCN reference]

[3] [FILL: GRU/BiGRU reference]

[4] [FILL: XGBoost reference]

[5] [FILL: PatchTST or transformer time-series reference]

[6] [FILL: Recent aircraft predictive maintenance paper]
