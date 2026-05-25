# Airplane Failure Prediction

Machine learning experiments for aircraft turbofan failure prediction using NASA C-MAPSS data.

## Project Layout

```text
airplanefailureprediction/
  data/                 C-MAPSS train/test/RUL files
  notebooks/            Jupyter and Colab experiments
  scripts/              Runnable Python scripts
  article_assets/       Article tables, figures, drafts, and generated assets
  assets/images/        Root-level diagrams and result images
  docs/                 Notes, comparisons, and change logs
  presentations/        Slide decks
  archive/code_exports/ Python console exports kept for reference
  outputs/              Local script outputs
```

## Data

The dataset lives in `data/`:

- `train_FD001.txt` through `train_FD004.txt`
- `test_FD001.txt` through `test_FD004.txt`
- `RUL_FD001.txt` through `RUL_FD004.txt`
- CSV copies for experiments that expect headered CSV files

## Notebooks

Most experiments are under `notebooks/`, including:

- `notebooks/edacmapss.ipynb`
- `notebooks/RFCforCMAPSS.ipynb`
- `notebooks/Can_LSTM.ipynb`
- `notebooks/abgru_all_datasets_f1.ipynb`
- `notebooks/dae.ipynb`
- `notebooks/vq_flow_cmapss.ipynb`
- `notebooks/TCN AE Binary Classification CMAPSS.ipynb`
- `notebooks/Transformer Classifier CMAPSS.ipynb`
- `notebooks/Hybrid Transformer CMAPSS.ipynb`
- `notebooks/ttsad_cmapss.ipynb`
- `notebooks/tcn_bigru_xgboost_cmapss.ipynb`
- `notebooks/semantic_trend_xgboost_article_ready.ipynb`

Several notebooks were originally developed in Google Colab and may still point to `/content/drive/MyDrive/data`. For local runs, use the project `data/` folder or update the notebook path cell.

## Scripts

Runnable scripts are under `scripts/`:

```bash
python scripts/turbofan_final_model.py
python scripts/patchtst_xgboost_cmapss.py --base data
python scripts/engine_level_mil_tcn_bigru_xgboost.py
python scripts/architecture_diagram.py
```

The local scripts resolve data from the project `data/` folder by default. You can override the final model paths with:

```powershell
$env:CMAPSS_DATA_DIR = "C:\path\to\data"
$env:CMAPSS_OUTPUT_DIR = "C:\path\to\outputs"
```

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision xgboost
```

CUDA is recommended for deep learning experiments, but the scripts can run on CPU with longer runtimes.

## References

- A. Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," ICAS 2008.
- NASA C-MAPSS dataset: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
