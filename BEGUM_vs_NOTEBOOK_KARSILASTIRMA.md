# BEGUM_CODE.py vs tcn_bigru_xgboost_cmapss.ipynb — Karşılaştırma

İki dosya da aynı fikre dayanıyor (TCN + BiGRU + XGBoost) ama tasarım felsefeleri çok farklı. Aşağıda farklar, etkileri ve iyileştirme önerileri özetlenmiştir.

---

## 1. Veri ve Etiketleme

| Konu | BEGUM_CODE.py | tcn_bigru_xgboost_cmapss.ipynb |
|---|---|---|
| Dataset | **FD002** (sabit) | **FD001** default, döngüde 4'ünü de çalıştırıyor |
| RUL threshold | **20** (daha agresif "near-failure") | **30** |
| Window size | **30** | **40** |
| Test pencereleri | Engine başına **sliding** (çok pencere) | Engine başına **sadece son 1 pencere** |
| Val split | %20 engine | %15 engine |

**Etkisi:** Notebook FD001'de `100 test örneği` üzerinden çalışıyor — bu çok düşük örneklem; tek bir yanlış sınıflandırma %1 fark yaratıyor (Recall=0.76 = sadece 19/25 doğru). BEGUM yaklaşımı (sliding test) hem **istatistiksel olarak daha güvenilir** hem de **engine-level smoothing'e izin veriyor** (yumuşatma + persistence ile alarm istikrarı). Notebook'un tek pencere yaklaşımı CMAPSS'in klasik sunum tarzı ama makale/üretim için BEGUM'un yaklaşımı daha sağlam.

---

## 2. Model ve Eğitim Hedefi (en kritik fark)

| Konu | BEGUM | Notebook |
|---|---|---|
| Encoder eğitimi | **MSE reconstruction** (self-supervised, son timestep'i tahmin) | **Focal Loss** (supervised, label tahmini) |
| Architecture | TCN(32,64,64) + BiGRU(64, 1L), **last hidden** | TCN(64,128,128,64) + BiGRU(128, 2L) + **Attention Pool** + WeightNorm |
| Embedding | latent **64** | embed **128** |
| XGBoost'a giren özellik | **latent(64) + semantic(204)** = 268 | sadece embedding **128** |

**Etkisi:**
- BEGUM'un encoder'ı **etiket görmüyor** → daha kötü diskriminatif öğrenir, ama overfit riski daha düşük. Buna karşılık 12×17=204 elle çıkarılmış semantik feature ekleyerek bu açığı **XGBoost düzeyinde kapatıyor**. Sonuç: latent kalitesi düşse bile XGBoost güçlü kalıyor.
- Notebook'un encoder'ı **doğrudan label öğreniyor** + Attention Pool + daha geniş kapasite → embedding daha bilgilendirici, ama küçük datasette overfit eğilimi yüksek (training F2 0.99'a çıkarken val 0.94'te kalıyor, log'lardan görülüyor).
- Notebook'ta `train + val embedding'leri birleştirilip` GridSearchCV'ye veriliyor — bu **veri sızıntısı (leakage)**: aynı val seti hem early-stop, hem grid search, hem threshold seçimi için kullanılıyor.

---

## 3. XGBoost Arama ve Eşik Seçimi

| Konu | BEGUM | Notebook |
|---|---|---|
| Grid boyutu | 4×2×3×4 = **96** trial | **2187** trial × 3-fold = 6561 fit |
| Skor | Custom: target recall + min precision + F1-tolerance | **F2** (recall-weighted) |
| `scale_pos_weight` | base × {1.5, 2, 2.5, 3} (search) | base × 2 (sabit) |
| Threshold seçimi | Val'da target recall **0.97**, min prec **0.70**, eşit F1'lerde **daha düşük threshold** | F2 max, min prec **0.35** |
| Engine-level smoothing | **MA + persistence** (search) | Yok |

**Etkisi:**
- Notebook'un grid'i 67× daha büyük ama tek skor kriteri (F2). BEGUM'un grid'i küçük ama **business-aware** (recall floor + precision floor + threshold tie-breaking).
- BEGUM'un `min_precision=0.70` notebook'un `0.35`'inden çok daha katı → daha az "boş alarm".
- BEGUM threshold'u "**en düşük yeterli threshold**" mantığıyla seçiyor (yakın F1'lerde) → operasyonel olarak daha güvenli (tetik daha erken).

---

## 4. Olası Hatalar / Risk Noktaları

### BEGUM_CODE.py
1. Encoder MSE ile sadece **son timestep'in feature vektörünü** tahmin etmeye eğitiliyor (`target = xb[:, :, -1]`) — bu RUL için zayıf bir self-supervised hedef; arızaya yakın temporal patternleri öğrenmeyi garanti etmez.
2. Semantic feature çıkarımı **çift döngülü Python** (yavaş): büyük datasette ciddi süre alır.
3. Eğitim sırasında AMP / mixed precision yok.
4. `best_engine_val` seçiminde sıralama anahtarı `(x[0], -x[4], -x[5])` — burada `x[0]=threshold` ascending sıralandığı için "lowest threshold" tercih ediliyor ama bunun "near best F1" filtresi içinde **F1'i bütünüyle göz ardı ettiği** için bazen anlamsız ayarlar seçebilir.

### Notebook
1. **Veri sızıntısı:** val seti hem TCN early-stop, hem GridSearchCV (train+val birleşmiş), hem threshold seçimi için kullanılıyor → val metrikleri optimistik, gerçek genelleme test'te düşüyor (val F2 0.99 vs test F1 0.86).
2. Test'te tek pencere → sonuçlar **gürültülü** ve istatistiksel anlamı zayıf.
3. GridSearchCV'de `early_stopping` yok, `n_estimators=300` sabit → arama hem yavaş hem sub-optimal.
4. `min_precision=0.35` — yarısından az precision'ı kabul ediyor, bu false alarm tolerans için çok düşük.

---

## 5. İyileştirme Önerileri

### Hızlı kazanımlar (BEGUM tarafında)
- Encoder'ı **supervised hibrit** hale getir: `loss = MSE_reconstruction + α · BCE(label)`. MSE self-supervised yararını korurken latent diskriminatifliği artar.
- `extract_semantic_features`'i **NumPy vektörel** yaz (engine bazında batch). 5–10× hızlanır.
- AMP (`torch.cuda.amp.autocast`) ekle.
- Attention pooling ekle (notebook'tan kopyala) — last hidden yerine ağırlıklı pool genelde +1-2 puan getirir.

### Hızlı kazanımlar (Notebook tarafında)
- **Leakage'i kır**: GridSearchCV'yi sadece `emb_tr` ile yap, `emb_val` sadece threshold seçimi için kalsın. Ya da nested CV kullan.
- Test için BEGUM'un sliding + engine-level smoothing yaklaşımını uygula → daha güvenilir metrikler.
- `min_precision`'ı 0.7+'a çıkar.
- GridSearchCV içine `early_stopping_rounds` ile eval set ver.

### Birleştirilmiş "best of both" mimarisi
1. Notebook'un mimarisi (TCN+BiGRU+Attention, weight_norm) +
2. **Hibrit kayıp** (focal + reconstruction) +
3. BEGUM'un **semantic + latent fusion** girdisi +
4. BEGUM'un **engine-level smoothing + business-aware threshold** seçimi +
5. **Engine-disjoint nested CV** (XGBoost val ≠ TCN val ≠ test).

Bu beşi birden uygulanırsa makale için hem teorik hem pratik açıdan çok daha sağlam bir baseline elde edilir.
