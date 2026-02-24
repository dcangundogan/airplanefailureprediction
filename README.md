# Airplane Failure Prediction — CMAPSS Turbofan Engine

Machine learning experiments for predicting aircraft engine failures. Various deep learning and classical ML approaches are benchmarked using NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

---

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**

| Subset | Train Engines | Test Engines | Op. Conditions | Fault Modes |
|--------|--------------|-------------|----------------|-------------|
| FD001  | 100          | 100         | 1              | 1           |
| FD002  | 260          | 259         | 6              | 1           |
| FD003  | 100          | 100         | 1              | 2           |
| FD004  | 249          | 248         | 6              | 2           |

Each subset contains 21 sensor measurements and 3 operational settings. The goal is to predict the Remaining Useful Life (RUL) of an engine or detect an impending failure.

Data is located in the `data/` folder (`train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`).

---

## Models & Experiments

### Exploratory Data Analysis
| File | Description |
|------|-------------|
| `edacmapss.ipynb` | Sensor analysis, correlation heatmaps, and RUL label generation on FD001 |

### Classical Machine Learning
| File | Model | Task | Dataset | Results |
|------|-------|------|---------|---------|
| `RFCforCMAPSS.ipynb` | Random Forest Classifier | Binary classification (failure/normal) | FD001–FD004 | Accuracy: 91–94%, F1: 80–85%, ROC-AUC: 0.97–0.98 |

### LSTM & GRU
| File | Model | Task | Dataset | Results |
|------|-------|------|---------|---------|
| `Can_LSTM.ipynb` | Bidirectional LSTM | RUL prediction (regression) | FD001 | Hyperparameter tuning via GridSearchCV |
| `abgru_all_datasets_f1.ipynb` | Attention-Based GRU | RUL prediction + multi-class (Critical/Warning/Normal) | FD001–FD004 | FD001: RMSE=13.54, R²=0.89 · FD003: RMSE=13.36, R²=0.90 |

### Autoencoder-Based
| File | Model | Task | Dataset | Description |
|------|-------|------|---------|-------------|
| `dae.ipynb` | Deep Autoencoder (DAE) | Anomaly detection (unsupervised) | FD001 | Learns normal engine behaviour; reconstruction error flags anomalies |
| `vq_flow_cmapss.ipynb` | VQ-VAE + Normalizing Flow | Anomaly detection (generative) | CMAPSS | Density estimation over complex data distributions |

### TCN-Based
| File | Model | Task | Dataset | Description |
|------|-------|------|---------|-------------|
| `TCN AE Binary Classification CMAPSS.ipynb` | TCN Autoencoder | Binary classification | CMAPSS | Dilated convolutions + autoencoder + classification head |

### Transformer-Based
| File | Model | Task | Dataset | Description |
|------|-------|------|---------|-------------|
| `Transformer Classifier CMAPSS.ipynb` | Transformer Encoder | Binary classification | CMAPSS | Multi-head attention with positional encoding |
| `hybrid_tcn_transformer_fd001.ipynb` | Hybrid TCN-Transformer | Binary classification | FD001 | TCN for feature extraction + Transformer for sequence modelling |
| `Hybrid Transformer CMAPSS.ipynb` | Hybrid Transformer AE + Classifier | Anomaly detection | CMAPSS | Unsupervised AE branch + supervised classification head (Target: Recall ≥ 0.90, F1 ≥ 0.80) |
| `ttsad_cmapss.ipynb` | TTSAD (TCN-Transformer-SVDD) | Binary anomaly detection | FD001–FD004 | Dilated TCN + Transformer + SVDD threshold |

### Production Model
| File | Model | Description |
|------|-------|-------------|
| `turbofan_final_model.py` | Deep Autoencoder + MLP Classifier | Standalone deployment-ready script. Accuracy: 93%, F1: 88%, ROC-AUC: 99% |

---

## Project Structure

```
airplanefailureprediction/
│
├── data/
│   ├── train_FD001.txt / .csv          # Training data (FD001–FD004)
│   ├── test_FD001.txt / .csv           # Test data
│   ├── RUL_FD001.txt                   # Ground-truth RUL values
│   └── Damage Propagation Modeling.pdf # Reference paper
│
├── edacmapss.ipynb                     # Exploratory data analysis
├── RFCforCMAPSS.ipynb                  # Random Forest classifier
├── Can_LSTM.ipynb                      # Bidirectional LSTM (RUL prediction)
├── abgru_all_datasets_f1.ipynb         # Attention-Based GRU (all subsets)
├── dae.ipynb                           # Deep Autoencoder (anomaly detection)
├── vq_flow_cmapss.ipynb                # VQ-VAE + Normalizing Flow
├── TCN AE Binary Classification CMAPSS.ipynb
├── Transformer Classifier CMAPSS.ipynb
├── hybrid_tcn_transformer_fd001.ipynb
├── Hybrid Transformer CMAPSS.ipynb
├── ttsad_cmapss.ipynb                  # TCN-Transformer-SVDD
│
├── turbofan_final_model.py             # Deployment-ready final model
└── README.md
```

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision
```

Notebooks were developed in Google Colab (GPU-enabled). A CUDA-capable GPU is recommended for local execution; CPU works as well but is slower.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<username>/airplanefailureprediction.git
cd airplanefailureprediction

# 2. Start with EDA
jupyter notebook edacmapss.ipynb

# 3. Run the final model directly
python turbofan_final_model.py
```

---

## Task Definitions

Two primary tasks are explored:

- **RUL Prediction (Regression):** Estimate how many cycles remain before engine failure.
- **Anomaly / Failure Detection (Classification):** Determine whether an engine is in a danger zone, using binary or multi-class labels:
  - `Critical` — RUL < 30
  - `Warning` — 30 ≤ RUL < 60
  - `Normal` — RUL ≥ 60

---

## References

- A. Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," ICAS 2008.
- Luo et al., "TTSAD: TCN-Transformer-SVDD Model for Anomaly Detection," *Computers & Security*, 2024.
- Jeong et al., "AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection," ICLR 2023.
- [NASA CMAPSS Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
