# Model Karşılaştırması: TCN-GRU vs TCN-BiGRU + XGBoost

**Dosyalar:**
- `../notebooks/tcn_gru_xgboost_cmapss.ipynb` → **Model A** (co-author versiyonu)
- `../notebooks/tcn_bigru_xgboost_cmapss.ipynb` → **Model B** (yeni versiyon)

---

## 1. Mimari Topoloji

| | Model A (TCN-GRU) | Model B (TCN-BiGRU) |
|---|---|---|
| **Akış** | Paralel dal | Seri akış |
| **TCN** | Ham giriş → TCN Branch | Ham giriş → TCN |
| **GRU** | Ham giriş → GRU Branch | **TCN çıktısı** → BiGRU |
| **Birleştirme** | TCN çıktısı + GRU çıktısı → **concat** | TCN → BiGRU → Attention (tek yol) |

Model A'da TCN ve GRU ham veriyi **bağımsız** olarak işler, sonuçları birleştirir.  
Model B'de TCN önce yerel özellikleri çıkarır, BiGRU bu özellikler üzerinde çalışır (hiyerarşik).

```
Model A:  X ─┬─► TCN ──────┐
              │              ├─► concat ─► proj ─► head
              └─► BiGRU ────┘

Model B:  X ─► TCN ─► BiGRU ─► Attention ─► proj ─► head
```

---

## 2. TCN Detayları

| | Model A | Model B |
|---|---|---|
| **Blok adı** | `CausalBlock` | `CausalConvBlock` |
| **Normalizasyon** | `BatchNorm1d` | `WeightNorm` (ağırlık normaliz.) |
| **Kanallar** | `[64, 128, 128]` (3 blok) | `[64, 128, 128, 64]` (4 blok) |
| **Kernel** | 3 | 3 |
| **Dilation** | 2^i | 2^i |
| **Dropout** | 0.1 | 0.2 |
| **Residual** | Var | Var |

> **Not:** BatchNorm batch boyutuna duyarlıdır. WeightNorm sekans uzunluğundan bağımsız çalışır — zaman serilerinde genellikle daha kararlı.

---

## 3. GRU Detayları

| | Model A | Model B |
|---|---|---|
| **Yön** | Bidirectional (her ne kadar "GRU" dense) | Bidirectional (BiGRU) |
| **Giriş** | Ham özellikler (n_feat) | TCN çıktısı (64 kanal) |
| **Hidden** | 64 → çıkış **128** (2×) | 128 → çıkış **256** (2×) |
| **Katman** | 2 | 2 |
| **Dropout** | 0.1 | 0.3 |

---

## 4. Attention

| | Model A | Model B |
|---|---|---|
| **Sınıf** | `TemporalAttention` | `AttentionPool` |
| **Mekanizma** | `Linear(dim, 1)` + softmax | `Linear(dim, 1)` + softmax |
| **Fark** | Yok — aynı mekanizma | Yok — aynı mekanizma |

---

## 5. XGBoost Giriş Özellikleri

| | Model A | Model B |
|---|---|---|
| **Derin embedding** | Var (embed_dim=128) | Var (embed_dim=128) |
| **El yapımı özellikler** | **Var** — her sensör için mean, std, min, max, trend (son−ilk), peak-to-peak → F×6 boyut | **Yok** |
| **XGBoost giriş boyutu** | 128 + (n_feat × 6) | 128 |

Model A'daki istatistiksel özellikler:
```python
mean, std, min, max,
trend = X[:,-1,:] - X[:,0,:],   # son adım - ilk adım
peak_to_peak = max - min
```

---

## 6. Hiperparametreler

| | Model A | Model B |
|---|---|---|
| **seq_len** | 30 | 40 |
| **batch_size** | 512 | 256 |
| **epochs** | 50 | 60 |
| **patience** | 10 | 12 |
| **Early stop metriği** | `val_loss` (düşük = iyi) | `val_F1` (yüksek = iyi) |
| **lr** | 1e-3 | 1e-3 |
| **weight_decay** | 1e-4 | 1e-4 |
| **focal_gamma** | 2.0 | 2.0 |

---

## 7. XGBoost Hiperparametreleri

| | Model A | Model B |
|---|---|---|
| **rounds** | 600 | 800 |
| **learning rate** | 0.05 | 0.03 |
| **max_depth** | 6 | 6 |
| **subsample** | 0.8 | 0.8 |
| **colsample_bytree** | 0.8 | 0.8 |
| **min_child_weight** | 5 | 5 |
| **early_stopping_rounds** | 30 | 40 |

---

## 8. Eğitim Akışı Farkları

| | Model A | Model B |
|---|---|---|
| **Model çıktısı** | `(embedding, logit)` tuple | Sadece `logit` (embed ayrı metod) |
| **Embedding çıkarım** | `model(x)[0]` | `model.embed(x)` |
| **XGBoost eğitimi için** | Train+Val embedding + handcrafted | Train+Val embedding |
| **Eşik optimizasyonu** | `np.arange(0.05, 0.95, 0.01)` | `np.arange(0.1, 0.9, 0.01)` |
| **Seed** | Belirtilmemiş | `seed_everything(42)` ile sabitlendi |

---

## 9. Özet: Temel Tasarım Farkı

**Model A:** TCN ve GRU bağımsız çalışır → çoklu bakış açısı → birleştirme. Ek olarak istatistiksel özellikler XGBoost'a elle beslenir.

**Model B:** TCN çıktısı BiGRU'ya girer → hiyerarşik temsil. Daha büyük gizli katman (128 vs 64), daha uzun pencere (40 vs 30), daha güçlü regularizasyon (dropout 0.3 vs 0.1).

---

## 10. Makale İçin Önerilen Tablo

| Bileşen | Model A | Model B |
|---|---|---|
| Mimari | Parallel TCN + BiGRU | Sequential TCN → BiGRU |
| TCN norm. | BatchNorm | WeightNorm |
| TCN derinlik | 3 blok | 4 blok |
| GRU hidden | 64 (çıkış 128) | 128 (çıkış 256) |
| GRU dropout | 0.10 | 0.30 |
| Pencere uzunluğu | 30 | 40 |
| XGBoost ek özellik | mean/std/min/max/trend/ptp | — |
| Early stop metriği | Val Loss | Val F1 |
| XGBoost rounds | 600 | 800 |
| XGBoost lr | 0.05 | 0.03 |
