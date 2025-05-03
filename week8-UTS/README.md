# Proyek Klasifikasi Gambar Ikan Menggunakan Deep Learning

Repositori ini berisi implementasi model deep learning untuk klasifikasi gambar ikan ke dalam 31 kategori berbeda menggunakan Convolutional Neural Networks (CNN).

## 📋 Daftar Isi
- [Ringkasan](#ringkasan)
- [Struktur Proyek](#struktur-proyek)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Kesimpulan](#kesimpulan)

## 🔍 Ringkasan

Proyek ini mengembangkan sistem klasifikasi gambar ikan menggunakan pendekatan deep learning dengan CNN. Kami mengimplementasikan dua model: CNN kustom dan arsitektur transfer learning berbasis MobileNetV2. Model dilatih untuk mengklasifikasikan 31 jenis ikan berbeda dengan akurasi tinggi.

## 🗂️ Struktur Proyek

```
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── custom_cnn.h5
│   └── mobilenetv2_transfer.h5
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
└── README.md
```

## 📊 Dataset

Dataset terdiri dari gambar ikan yang dikategorikan ke dalam 31 kelas berbeda. Data dibagi menjadi tiga set:
- **Train**: Digunakan untuk melatih model
- **Validation**: Digunakan untuk mengevaluasi model selama pelatihan
- **Test**: Digunakan untuk mengevaluasi performa akhir model

## 🧠 Metodologi

### 1. Pengumpulan dan Pembersihan Data
- Dataset gambar ikan dimuat dari Google Drive
- Struktur dataset terdiri dari direktori `train`, `val`, dan `test`
- Validasi data dilakukan untuk memastikan tidak ada nilai yang hilang

### 2. Rekayasa Fitur
- Label kategori ikan diubah menjadi kode numerik menggunakan `LabelEncoder`
- Pemetaan kategori dan label disimpan untuk penggunaan selanjutnya

### 3. Pra-pemrosesan dan Augmentasi Data
- Teknik augmentasi diterapkan pada data pelatihan:
  - Rotasi
  - Pergeseran lebar/tinggi
  - Geser
  - Zoom
  - Flip horizontal
- Data validasi dan uji hanya dinormalisasi tanpa augmentasi

### 4. Konstruksi Model

#### Model CNN Kustom
- Beberapa lapisan konvolusional
- Max-pooling
- Dropout untuk regularisasi
- Lapisan padat untuk klasifikasi

#### Transfer Learning dengan MobileNetV2
- Model dasar MobileNetV2 pre-trained pada ImageNet
- Base model dibekukan selama pelatihan
- Lapisan atas disesuaikan dengan:
  - Global average pooling
  - Dropout
  - Lapisan padat untuk klasifikasi 31 kategori ikan

### 5. Pelatihan Model
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Callbacks**:
  - ModelCheckpoint: Menyimpan model terbaik
  - EarlyStopping: Mencegah overfitting
  - ReduceLROnPlateau: Menyesuaikan learning rate

## 📈 Hasil Evaluasi

### Metrik Performa

| Metrik         | Nilai   | Deskripsi                                                      |
|----------------|---------|----------------------------------------------------------------|
| Akurasi        | 0.8409  | 84.09% gambar diklasifikasikan dengan benar                    |
| Presisi        | 0.8479  | 84.79% dari prediksi positif benar-benar akurat                |
| Recall         | 0.8409  | 84.09% dari semua kasus positif berhasil terdeteksi            |
| F1 Score       | 0.8382  | 83.82% keseimbangan antara presisi dan recall                  |
| ROC AUC Score  | 0.9934  | 99.34% kemampuan model membedakan antara kelas                 |

### Performa per Kelas

Beberapa kelas menunjukkan performa yang berbeda:
- **Kinerja Tinggi**: `Green Spotted Puffer` dan `Scat Fish` memiliki presisi dan recall tinggi
- **Kinerja Rendah**: `Mudfish` (presisi 0.50, recall 0.59) dan `Bangus` (presisi 0.61, recall 0.65) memerlukan peningkatan

### Rata-rata Metrik

- **Weighted Average**:
  - Presisi: 0.85
  - Recall: 0.84
  - F1-score: 0.84

- **Macro Average**:
  - Presisi: 0.85
  - Recall: 0.81
  - F1-score: 0.82

## 🎯 Kesimpulan

Model menunjukkan performa yang sangat baik secara keseluruhan dengan akurasi 84.09% dan ROC AUC 0.9934. Namun, beberapa kategori ikan masih memerlukan perbaikan lebih lanjut. Pendekatan transfer learning dengan MobileNetV2 terbukti efektif untuk tugas klasifikasi gambar ikan ini.

Untuk pengembangan lebih lanjut, fokus dapat diberikan pada peningkatan performa untuk kelas dengan presisi dan recall rendah melalui:
1. Menambah data untuk kelas yang kurang terwakilkan
2. Strategi augmentasi yang lebih spesifik
3. Teknik ensemble dengan beberapa model

---

© 2025 Proyek Klasifikasi Ikan
