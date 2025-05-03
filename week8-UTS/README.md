# Proyek Analisis Model Machine Learning

Repositori ini berisi implementasi dan evaluasi berbagai model machine learning, termasuk model regresi, model klasifikasi umum, dan model klasifikasi gambar ikan menggunakan deep learning.

## 📋 Daftar Isi
- [Ringkasan](#ringkasan)
- [Hasil Evaluasi Model Regresi](#hasil-evaluasi-model-regresi)
- [Hasil Evaluasi Model Klasifikasi](#hasil-evaluasi-model-klasifikasi)
- [Proyek Klasifikasi Gambar Ikan](#proyek-klasifikasi-gambar-ikan)
- [Kesimpulan](#kesimpulan)
- [Catatan Tambahan](#catatan-tambahan)

## 🔍 Ringkasan

Proyek ini berisi implementasi dan evaluasi berbagai model machine learning untuk tugas regresi dan klasifikasi, dengan fokus khusus pada klasifikasi gambar ikan menggunakan pendekatan deep learning. Evaluasi menyeluruh disediakan untuk semua model dengan metrik-metrik relevan.

## 📊 Hasil Evaluasi Model Regresi

### Mean Squared Error (MSE)

Mean Squared Error (MSE) dihitung dengan rumus:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

di mana:
* $y_i$ adalah nilai aktual
* $\hat{y}_i$ adalah nilai prediksi
* $n$ adalah jumlah data

Untuk model ini, MSE adalah: **MSE = 75.4008**

### Root Mean Squared Error (RMSE)

Root Mean Squared Error (RMSE) adalah akar kuadrat dari MSE:

$$RMSE = \sqrt{MSE}$$

Untuk model ini, RMSE adalah: **RMSE = 8.6834**

### R-squared (R²)

R-squared mengukur proporsi varians pada variabel dependen yang dapat diprediksi dari variabel independen. R² dihitung dengan rumus:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

di mana:
* $\bar{y}$ adalah rata-rata nilai aktual

Untuk model ini, R² adalah: **R² = 0.3647**

### Evaluasi Model Regresi

Model regresi menunjukkan:
- MSE dan RMSE menunjukkan bahwa prediksi model agak meleset, dengan RMSE sebesar 8.6834
- Model mungkin perlu penyesuaian atau tuning lebih lanjut
- Nilai R² sebesar 0.3647 mengindikasikan bahwa hanya sekitar 36% dari varians pada variabel dependen dapat dijelaskan oleh model
- Ada ruang untuk perbaikan lebih lanjut pada model regresi ini

## 📈 Hasil Evaluasi Model Klasifikasi

### Akurasi

Akurasi adalah proporsi prediksi yang benar:

$$\text{Akurasi} = \frac{TP + TN}{TP + TN + FP + FN}$$

di mana:
* TP: True Positives
* TN: True Negatives
* FP: False Positives
* FN: False Negatives

Untuk model ini, akurasi adalah: **Akurasi = 0.7354**

### Presisi

Presisi adalah proporsi prediksi positif yang benar dari semua prediksi positif yang dibuat:

$$\text{Presisi} = \frac{TP}{TP + FP}$$

Untuk model ini, presisi adalah: **Presisi = 0.7239**

### Recall

Recall (juga dikenal sebagai Sensitivitas atau True Positive Rate) adalah proporsi prediksi positif yang benar dari semua kasus positif yang sebenarnya:

$$\text{Recall} = \frac{TP}{TP + FN}$$

Untuk model ini, recall adalah: **Recall = 0.7182**

### F1-Score

F1-Score adalah rata-rata harmonis antara Presisi dan Recall:

$$F1\_Score = \frac{2 \cdot \text{Presisi} \cdot \text{Recall}}{\text{Presisi} + \text{Recall}}$$

Untuk model ini, F1-Score adalah: **F1_Score = 0.7210**

### AUC-ROC

Area Under the Receiver Operating Characteristic Curve (AUC-ROC) mengukur seberapa baik model membedakan antara kelas-kelas. Nilai 1 menunjukkan klasifikasi yang sempurna, sementara 0.5 menunjukkan bahwa model tidak lebih baik dari tebakan acak.

Untuk model ini, AUC-ROC adalah: **AUC-ROC = 0.8102**

### Evaluasi Model Klasifikasi

Model klasifikasi menunjukkan:
- Kinerja yang cukup baik, dengan akurasi 0.7354
- Presisi dan Recall yang seimbang di sekitar 0.72
- F1-Score sebesar 0.7210 menunjukkan keseimbangan yang baik antara Presisi dan Recall
- AUC-ROC sebesar 0.8102 mengindikasikan bahwa model cukup baik dalam membedakan antara kelas-kelas

## 🐟 Proyek Klasifikasi Gambar Ikan

### Dataset

Dataset terdiri dari gambar ikan yang dikategorikan ke dalam 31 kelas berbeda. Data dibagi menjadi tiga set:
- **Train**: Digunakan untuk melatih model
- **Validation**: Digunakan untuk mengevaluasi model selama pelatihan
- **Test**: Digunakan untuk mengevaluasi performa akhir model

### Metodologi

#### 1. Pengumpulan dan Pembersihan Data
- Dataset gambar ikan dimuat dari Google Drive
- Struktur dataset terdiri dari direktori `train`, `val`, dan `test`
- Validasi data dilakukan untuk memastikan tidak ada nilai yang hilang

#### 2. Rekayasa Fitur
- Label kategori ikan diubah menjadi kode numerik menggunakan `LabelEncoder`
- Pemetaan kategori dan label disimpan untuk penggunaan selanjutnya

#### 3. Pra-pemrosesan dan Augmentasi Data
- Teknik augmentasi diterapkan pada data pelatihan:
  - Rotasi
  - Pergeseran lebar/tinggi
  - Geser
  - Zoom
  - Flip horizontal
- Data validasi dan uji hanya dinormalisasi tanpa augmentasi

#### 4. Konstruksi Model

##### Model CNN Kustom
- Beberapa lapisan konvolusional
- Max-pooling
- Dropout untuk regularisasi
- Lapisan padat untuk klasifikasi

##### Transfer Learning dengan MobileNetV2
- Model dasar MobileNetV2 pre-trained pada ImageNet
- Base model dibekukan selama pelatihan
- Lapisan atas disesuaikan dengan:
  - Global average pooling
  - Dropout
  - Lapisan padat untuk klasifikasi 31 kategori ikan

#### 5. Pelatihan Model
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Callbacks**:
  - ModelCheckpoint: Menyimpan model terbaik
  - EarlyStopping: Mencegah overfitting
  - ReduceLROnPlateau: Menyesuaikan learning rate

### Hasil Evaluasi Model Klasifikasi Gambar Ikan

#### Metrik Performa

| Metrik         | Nilai   | Deskripsi                                                      |
|----------------|---------|----------------------------------------------------------------|
| Akurasi        | 0.8409  | 84.09% gambar diklasifikasikan dengan benar                    |
| Presisi        | 0.8479  | 84.79% dari prediksi positif benar-benar akurat                |
| Recall         | 0.8409  | 84.09% dari semua kasus positif berhasil terdeteksi            |
| F1 Score       | 0.8382  | 83.82% keseimbangan antara presisi dan recall                  |
| ROC AUC Score  | 0.9934  | 99.34% kemampuan model membedakan antara kelas                 |

#### Performa per Kelas

Beberapa kelas menunjukkan performa yang berbeda:
- **Kinerja Tinggi**: `Green Spotted Puffer` dan `Scat Fish` memiliki presisi dan recall tinggi
- **Kinerja Rendah**: `Mudfish` (presisi 0.50, recall 0.59) dan `Bangus` (presisi 0.61, recall 0.65) memerlukan peningkatan

#### Rata-rata Metrik

- **Weighted Average**:
  - Presisi: 0.85
  - Recall: 0.84
  - F1-score: 0.84

- **Macro Average**:
  - Presisi: 0.85
  - Recall: 0.81
  - F1-score: 0.82

## 🎯 Kesimpulan

1. **Model Regresi**: 
   - MSE sebesar 75.4008 dan RMSE sebesar 8.6834 menunjukkan adanya error prediksi yang cukup signifikan
   - R² sebesar 0.3647 menunjukkan bahwa model hanya dapat menjelaskan sekitar 36% variasi dalam data
   - Model ini memerlukan penyesuaian lebih lanjut untuk meningkatkan performa

2. **Model Klasifikasi Umum**:
   - Menunjukkan performa yang cukup baik dengan akurasi 73.54%
   - Keseimbangan yang baik antara presisi (72.39%) dan recall (71.82%)
   - AUC-ROC sebesar 0.8102 menunjukkan kemampuan diskriminatif yang baik

3. **Model Klasifikasi Gambar Ikan**:
   - Performa terbaik di antara model-model yang dievaluasi dengan akurasi 84.09%
   - ROC AUC yang sangat tinggi (0.9934) menunjukkan kemampuan diskriminatif yang sangat baik
   - Beberapa kelas ikan masih memerlukan perbaikan lebih lanjut
   - Pendekatan transfer learning dengan MobileNetV2 terbukti efektif untuk tugas klasifikasi gambar

## 📝 Catatan Tambahan

Analisis Tugas ada di notebook masing masing
---

© 2025 Proyek Analisis Model Machine Learning
