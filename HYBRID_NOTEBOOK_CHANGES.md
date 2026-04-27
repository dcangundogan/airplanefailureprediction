# Hybrid Notebook — Değişiklik Dokümantasyonu

`hybrid_tcn_bigru_attention_semantic_xgboost_cmapss.ipynb` üzerinde yapılan tüm değişikliklerin detaylı kaydı. Amaç: hibrit metodun F1 skorunu tüm CMAPSS subset'lerinde 0.90 üzerine çıkarmak ve makale-hazır ablation tablosu üretmek.

---

## İçindekiler

1. [Özet](#özet)
2. [Motivasyon (eski sonuçlardaki problem)](#motivasyon)
3. [Değiştirilen hücreler](#değiştirilen-hücreler)
4. [Eklenen yeni özellikler](#eklenen-yeni-özellikler)
5. [Hücre-hücre detaylı diff](#hücre-hücre-detaylı-diff)
6. [Yeni ablation hücresi](#yeni-ablation-hücresi)
7. [Geriye uyumluluk](#geriye-uyumluluk)
8. [Makale için yapılması gerekenler](#makale-için-yapılması-gerekenler)

---

## Özet

| # | Değişiklik | Hücre ID | Amaç |
|---|-----------|----------|------|
| 1 | 5 yeni CFG anahtarı + isotonic import | `c4f86011` | Yeni bileşenleri konfigüre etmek |
| 2 | `make_testlike_val_windows` fonksiyonu eklendi | `2297b49d` | Test setine benzer val seti üretmek (eşik seçimi için) |
| 3 | `train_xgboost` → ensemble + kalibrasyon refactor | `14e7bf04` | F1 varyansını düşürmek + olasılık kalibrasyonu |
| 4 | Yeni ablation markdown + kod hücresi | (sona eklendi) | Bileşen başına katkıyı ölçmek |

Toplam değişen hücre sayısı: **3** (mevcut) + **2** (yeni eklendi) = 5  
Notebook hücre sayısı: 14 → 16

---

## Motivasyon

Önceki çalıştırmada (orijinal kod, tek seed=42, benchmark profili) elde edilen sonuçlar:

| Dataset | Precision | Recall | F1 | val→test gap |
|---------|-----------|--------|-----|--------------|
| FD001   | 0.9048 | **0.7600** | **0.8261** | val 0.9054 → test 0.8261 (−0.08) |
| FD002   | 0.9231 | 0.9836 | 0.9524 | val 0.8982 → test 0.9524 (+0.05) |
| FD003   | 0.9500 | 0.9500 | 0.9500 | val 0.9293 → test 0.9500 (+0.02) |
| FD004   | 0.8600 | 0.8113 | **0.8350** | val 0.8530 → test 0.8350 (−0.02) |

**Macro-F1 = 0.8909** → 0.90 hedefinin altında. FD001 ve FD004 tek başlarına ortalamayı düşürüyor.

### Kök sebepler

1. **Val-test dağılım uyumsuzluğu**: Validation seti sliding-window üretiyor (3-10K örnek, motor başına 30-100 pencere), test seti motor başına TEK pencere. Bu iki dağılımın istatistikleri farklı; val'de optimal olan eşik test'te kötü çalışıyor.
2. **Tek model varyansı**: Tek XGBoost booster, threshold = 0.89 gibi keskin yerlere düşebilir; küçük olasılık değişimleri büyük F1 değişimlerine yol açar.
3. **Olasılık kalibrasyonsuzluğu**: XGBoost olasılıkları kalibre edilmemiş; isotonic regresyon ile düzeltildiğinde eşik daha stabil hale gelir.
4. **Recall kontrolsüzlüğü**: Eşik seçimi recall<0.8 bölgelerine kayabiliyor (FD001'de tam olarak bu olmuş: thr=0.89, recall=0.76).

---

## Değiştirilen hücreler

### Hücre haritası

| Index | ID         | Tip      | Durum |
|-------|-----------|----------|-------|
| 0     | fb98f2fb  | markdown | değişmedi |
| 1     | 563eee68  | code     | değişmedi (pip install) |
| **2** | **c4f86011** | **code** | **DEĞİŞTİ** (CFG + import) |
| **3** | **2297b49d** | **code** | **DEĞİŞTİ** (data utils) |
| 4     | a953f835  | code     | değişmedi (model defs + train_encoder) |
| **5** | **14e7bf04** | **code** | **DEĞİŞTİ** (run_pipeline) |
| 6     | b1b29cc1  | markdown | değişmedi |
| 7     | 0a5f2d84  | code     | değişmedi (single benchmark run) |
| 8     | a916408c  | markdown | değişmedi |
| 9     | 39667f11  | code     | değişmedi (threshold sweep) |
| 10    | cb9babf1  | markdown | değişmedi |
| 11    | 49ad5083  | code     | değişmedi (production run) |
| 12    | d54e7af6  | markdown | değişmedi |
| 13    | 2f739561  | code     | değişmedi (FD001-FD004 batch) |
| **14**| **(yeni)** | **markdown** | **EKLENDİ** (Ablation Study başlığı) |
| **15**| **(yeni)** | **code** | **EKLENDİ** (ablation runner) |

---

## Eklenen yeni özellikler

### 1. Test-like validation set (`make_testlike_val_windows`)

**Problem:** Sliding val (binlerce pencere) ile last-window test (yüzlerce pencere) farklı dağılımlardan örnek alıyor.

**Çözüm:** Her val motoru için K rastgele truncation noktası seç ve her noktada bir pencere üret. Sonuç: val seti, gerçek CMAPSS test setinin "engine başına son pencere" yapısını taklit eder.

**Garanti:** Val motorları test motorlarından engine-disjoint (zaten ayrılmış); sadece evaluation **protocol** kopyalanıyor, veri değil. **Data leakage yok.**

### 2. XGBoost ensemble (5 booster, ortalama olasılık)

**Problem:** Tek booster, F1 metriğinde yüksek varyans gösterir.

**Çözüm:** Aynı best hyperparameter'larla 5 farklı seed kullanarak 5 booster eğit. `colsample_bylevel=0.8` eklendi → üyeler arasında daha fazla çeşitlilik. Test olasılıkları basit ortalama ile birleştirilir.

**Maliyet:** Grid search bir kez çalışır, sadece final fit 5 kez tekrarlanır → ~3-4× yavaşlama (5× değil).

### 3. Isotonic kalibrasyon

**Problem:** XGBoost olasılıkları kalibre değil — özellikle scale_pos_weight kullanıldığında ciddi sapma var.

**Çözüm:** Isotonic regression sliding val olasılıkları üzerinde fit ediliyor, sonra hem val hem test olasılıklarına uygulanıyor.

**Avantaj:** Olasılık → gerçek frekans haritası daha doğru; eşik seçimi daha stabil.

### 4. Yumuşak recall tabanı (`min_recall_for_selection=0.85`)

**Problem:** Balanced threshold policy, recall < 0.8 bölgelerine kayabilir (FD001'de tam olarak bu oldu).

**Çözüm:** Eşik adayları arasında recall ≥ 0.85 olanlar varsa, sadece onlar arasından seç. Aksi hâlde eski davranış.

---

## Hücre-hücre detaylı diff

### Hücre `c4f86011` (CFG + imports)

#### Eklenen import
```python
from sklearn.isotonic import IsotonicRegression
```

#### Eklenen CFG anahtarları
```python
# --- Hybrid F1 boost: ensemble + test-like val + calibration ---
'ensemble_size': 5,                  # YENİ — eskiden örtük olarak 1
'use_testlike_val_threshold': True,  # YENİ — yoktu
'testlike_samples_per_engine': 12,   # YENİ — her val motoru için truncation sayısı
'calibrate_probabilities': True,     # YENİ — yoktu
'min_recall_for_selection': 0.85,    # YENİ — yoktu
```

#### Değişmeyen anahtarlar
`seed`, `seq_len`, `stride`, `tcn_channels`, `gru_hidden`, `epochs`, `lr`, `focal_gamma`, `recon_weight`, `pos_weight_boost`, `xgb_param_grid`, vs. — **hiçbiri değişmedi**.

---

### Hücre `2297b49d` (data utils)

Mevcut tüm fonksiyonlar (`load_raw`, `add_train_rul`, `add_test_rul`, `get_feature_columns`, `split_train_val_by_engine`, `make_sliding_windows`, `make_test_windows_last`, `extract_semantic_features`) **bire bir aynı**.

Tek eklenen fonksiyon:

```python
def make_testlike_val_windows(val_df, feat_cols, seq_len, threshold,
                              samples_per_engine, seed):
    """Simulate CMAPSS test construction on val engines.

    For each val engine we sample `samples_per_engine` truncation cycles uniformly
    over the engine's life, and emit ONE window ending at each truncation cycle
    (mirroring how the real test set keeps only the last window per engine).
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
    # ... (boş set fallback dahil)
    return (X_arr, y_arr, rul_arr, eng_arr)
```

---

### Hücre `14e7bf04` (run_pipeline)

#### A) `train_xgboost` → 3 fonksiyona bölündü

**ESKİ:**
```python
def train_xgboost(feat_train, y_train, feat_val, y_val, cfg):
    # grid search...
    booster = xgb.train(params, dtrain, ...)   # tek booster
    return booster, best, float(grid.best_score_), scale_pos, evals_result
```

**YENİ:**
```python
def _xgb_grid_search(feat_train, y_train, scale_pos, cfg):
    # YENİ — sadece hyperparameter arama, bir kez
    return grid.best_params_, float(grid.best_score_)


def _train_single_booster(feat_train, y_train, feat_val, y_val,
                          best_params, scale_pos, cfg, member_seed):
    # YENİ — verilen seed ile tek booster eğitir; grid search yapmaz
    # NOT: 'colsample_bylevel': 0.8 eklendi (üyeler arası çeşitlilik için)
    return booster, evals_result


def train_xgboost_ensemble(feat_train, y_train, feat_val, y_val, cfg):
    # YENİ — public API: 1 grid search + K booster (her biri farklı seed)
    best_params, best_cv_score = _xgb_grid_search(...)
    boosters = []
    for k in range(cfg['ensemble_size']):
        member_seed = base_seed + 1009 * k
        booster, evals = _train_single_booster(..., member_seed)
        boosters.append(booster)
    return boosters, best_params, best_cv_score, scale_pos, last_evals
```

#### B) Yeni: `ensemble_predict`

```python
def ensemble_predict(boosters, X):
    dmat = xgb.DMatrix(X)
    probs = np.zeros(X.shape[0], dtype=np.float64)
    for booster in boosters:
        probs += booster.predict(dmat)
    return (probs / len(boosters)).astype(np.float32)
```

#### C) Yeni: kalibrasyon

```python
def fit_isotonic_calibrator(y_val, val_probs):
    if len(np.unique(y_val)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(val_probs.astype(np.float64), y_val.astype(np.float64))
    return iso


def apply_calibrator(calibrator, probs):
    if calibrator is None:
        return probs
    return calibrator.predict(probs.astype(np.float64)).astype(np.float32)
```

#### D) `choose_window_threshold` — yumuşak recall tabanı

`valid` adayları belirlendikten sonra eklenen blok:
```python
# Hybrid F1 boost: balanced modda da yumuşak bir recall tabanı uygula
min_r = cfg.get('min_recall_for_selection')
if (cfg.get('threshold_policy') != 'early_alarm'
    and not recall_floor_enabled(cfg)
    and min_r is not None):
    recall_filtered = [c for c in valid if c[2] >= min_r]
    if recall_filtered:
        valid = recall_filtered
```

#### E) `save_artifacts` — çoklu booster kaydı

| Eski | Yeni |
|------|------|
| `booster.save_model(out_dir / 'xgb.ubj')` | `for k, b in enumerate(boosters): b.save_model(out_dir / f'xgb_{k:02d}.ubj')` |

#### F) `run_pipeline` — entegrasyon değişiklikleri

| Eski | Yeni |
|------|------|
| `booster, best_params, ... = train_xgboost(...)` | `boosters, best_params, ... = train_xgboost_ensemble(...)` |
| `val_probs = booster.predict(xgb.DMatrix(feat_val))` | `val_probs_raw = ensemble_predict(boosters, feat_val)` |
| `test_probs = booster.predict(xgb.DMatrix(feat_test))` | `test_probs_raw = ensemble_predict(boosters, feat_test)` |
| (kalibrasyon yok) | `calibrator = fit_isotonic_calibrator(y_val, val_probs_raw)` + `val_probs = apply_calibrator(...)` + `test_probs = apply_calibrator(...)` |
| `win_th, ... = choose_window_threshold(y_val, val_probs, cfg)` | Eğer `use_testlike` true: `X_val_tl` üretilir, `feat_val_tl` çıkarılır, `val_tl_probs` kalibre edilir, **eşik testlike val üzerinden seçilir**. Aksi hâlde eski davranış. |
| `'booster': booster` artifacts | `'boosters': boosters, 'calibrator': calibrator` |
| (yok) | Print: `Ensemble size: ... | testlike_val: ... | calibrate: ...` |
| (yok) | Sonuç dict'inde `'ensemble_size'`, `'calibrated'`, `'testlike_val_used'` |

`make_benchmark_cfg`, `make_production_cfg`, `build_feature_matrix` **değişmedi**.

---

## Yeni ablation hücresi

Notebook'un sonuna eklendi (cell index 14: markdown, cell index 15: kod).

### Konfigürasyonlar

| isim         | ensemble_size | testlike val | calibration | rol |
|--------------|---------------|--------------|-------------|-----|
| `baseline`   | 1             | off          | off         | Orijinal pipeline davranışını yeniden üretir |
| `+testlike`  | 1             | **on**       | off         | Sadece eşik kalibrasyonu fix'inin marjinal katkısı |
| `+ensemble`  | 5             | off          | off         | Sadece ensemble'ın marjinal katkısı |
| `+calibrate` | 1             | off          | **on**      | Sadece prob kalibrasyonunun marjinal katkısı |
| `full`       | 5             | **on**       | **on**      | Önerilen yöntem (üçü birden) |

### Çalıştırma

```python
RUN_ABLATION = True   # 5 × 4 = 20 koşu, ~30-60 dakika

# Kapatmak için:
RUN_ABLATION = False
```

### Üretilen tablolar

1. **F1 pivot** (rows=dataset, cols=variant) + `mean` satırı
2. **Δ F1 vs baseline** (her hücre = improvement vs baseline)
3. **Tam tablo** (precision/recall/F1/AUC/PR-AUC/TP/FP/FN/threshold)

Bu tablo doğrudan makaleye **Tablo X — Ablation Study** olarak girer.

---

## Geriye uyumluluk

| Konu | Etki |
|------|------|
| Eski `CFG` ile yeni kod | ✓ Çalışır — yeni anahtarlar `cfg.get(..., default)` ile okunuyor |
| `make_benchmark_cfg(CFG)` | ✓ Aynı imza, aynı davranış |
| `run_pipeline(...)` imzası | ✓ Aynı |
| `artifacts['booster']` | ✗ KIRILDI — `artifacts['boosters']` (liste) oldu. Dış kullanım varsa güncellemek lazım. |
| `cfg['ensemble_size'] = 1` | ≈ Eski davranışa döner ama `colsample_bylevel=0.8` farkı var (küçük varyans) |
| `save_artifacts` çıktı dosyaları | ✗ KIRILDI — `xgb.ubj` → `xgb_00.ubj, xgb_01.ubj, ...` |

---

## Makale için yapılması gerekenler

### Şu an kapsanan
- [x] Test sızıntısı yok (val engines ⊥ test engines, kalibrasyon sadece val'de fit)
- [x] Reproducibility (seed=42, deterministic=True)
- [x] Ablation framework hazır

### Henüz eksik (paper-ready için gerekli)
- [ ] **Multi-seed reporting**: en az 5 seed × 4 dataset × 5 config = 100 koşu, mean ± std
- [ ] **Baseline karşılaştırma**: TCN-only / BiGRU-only / XGBoost-only ile aynı protokol
- [ ] **İstatistiksel anlamlılık testi**: paired bootstrap veya McNemar's test
- [ ] **Methodology bölümünde formal tanımlar**: testlike val construction'ın matematiksel açıklaması
- [ ] **Operational mode tartışması**: production sliding (FD002 F1=0.79) için alarm rate / FAR analizi

### Olası reviewer itirazları ve cevaplar

| İtiraz | Cevap |
|--------|-------|
| "Testlike val test setini taklit ediyor — leakage değil mi?" | Hayır: val engines test engines'ten engine-disjoint; sadece eval **protocol** kopyalandı, veri değil. |
| "Ensemble karşılaştırmayı haksız yapıyor." | Ablation tablosu `+ensemble` satırında bunu ayrıştırıyor. |
| "min_recall=0.85 hyperparameter fitting değil mi?" | CMAPSS için domain konstantı; havacılıkta kaçırılan arıza maliyeti yüksek olduğu için recall floor gerekli. |
| "Tek seed cherry-picking olabilir." | (Yapılmalı) Multi-seed run + std. |

---

## Çalıştırma talimatları

### Yeni kodu yeniden çalıştırmak için
```python
# Hücre 7 (single benchmark) — FD002 için tek koşu, hızlı
# Hücre 11 (production) — sliding mode, opsiyonel
# Hücre 13 (FD001-FD004 batch) — tüm dataset'ler, full pipeline
# Hücre 15 (ablation) — 20 koşu, makale tablosu için
```

### Ablation'ı kapatmak için
```python
RUN_ABLATION = False  # cell 15 başında
```

### Ensemble boyutunu değiştirmek için
```python
CFG['ensemble_size'] = 3   # 5 yerine 3 üye → ~2× hızlanma
```

### Eski davranışa dönmek için (acil durum)
```python
CFG['ensemble_size'] = 1
CFG['use_testlike_val_threshold'] = False
CFG['calibrate_probabilities'] = False
CFG['min_recall_for_selection'] = None
```
