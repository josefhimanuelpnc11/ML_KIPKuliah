# KIP Kuliah Classification System

🎓 **Sistem Klasifikasi Penerima KIP Kuliah berbasis Analisis Faktor Sosial-Ekonomi**

## 📋 Deskripsi Proyek

Sistem ini menggunakan teknik machine learning untuk menganalisis dan memprediksi kelayakan penerima bantuan KIP Kuliah berdasarkan faktor-faktor sosial ekonomi. Sistem ini mengimplementasikan tiga algoritma klasifikasi utama: Random Forest, XGBoost, dan Support Vector Machine (SVM).

## 🎯 Tujuan Utama

- Menganalisis faktor-faktor sosial ekonomi yang mempengaruhi kelayakan KIP Kuliah
- Membangun model prediksi yang akurat untuk klasifikasi penerima bantuan
- Memberikan insights untuk pengambilan keputusan dalam distribusi bantuan pendidikan
- Mengidentifikasi pola dan karakteristik mahasiswa yang membutuhkan bantuan

## 📊 Dataset

Sistem ini menganalisis data dari **Politeknik Negeri Cilacap** periode **2022-2024** yang mencakup:

### Data Pendaftar KIP Kuliah:
- **2022**: 3 file CSV (SBMPN, Seleksi Mandiri PTN, SNMPN Politeknik)
- **2023**: 3 file CSV (SNBT, SNBP, Seleksi Mandiri PTN)
- **2024**: 1 file CSV (SNBP dan SNBT)

### Data Penerima KIP Kuliah:
- Angkatan 2022, 2023, dan 2024

### Variabel Utama yang Dianalisis:

**🏠 Kondisi Sosial Ekonomi:**
- Penghasilan orang tua (Ayah & Ibu)
- Pekerjaan orang tua
- Status kepemilikan rumah
- Luas tanah dan bangunan
- Akses utilitas (listrik, air, MCK)

**👨‍👩‍👧‍👦 Struktur Keluarga:**
- Jumlah tanggungan keluarga
- Status hidup orang tua
- Jumlah orang tua yang bekerja

**📍 Faktor Geografis:**
- Lokasi asal sekolah
- Jarak ke pusat kota
- Tingkat perkembangan provinsi

**🎫 Status Bantuan Sosial:**
- Status DTKS (Data Terpadu Kesejahteraan Sosial)
- Status P3KE dan tingkat desil
- Kepemilikan KIP/KKS

## 🏗️ Arsitektur Sistem

```
KIP_Kuliah_Classification/
├── main.py                 # Entry point utama
├── src/
│   ├── data_exploration.py # Modul eksplorasi data
│   ├── preprocessing.py    # Pipeline preprocessing
│   └── models.py          # Training dan evaluasi model
├── reports/               # Laporan hasil analisis
├── models/               # Model tersimpan
├── visualizations/       # Grafik dan visualisasi
├── requirements.txt      # Dependencies
└── README.md            # Dokumentasi
```

## ⚙️ Pipeline Sistem

### 1. **Data Exploration** 📊
- Penemuan dan validasi file CSV
- Analisis struktur data dan variabel
- Identifikasi missing values
- Pembuatan target variable (status penerima)

### 2. **Data Preprocessing** 🔧
- **Feature Engineering**: Transformasi variabel sosial-ekonomi menjadi fitur numerik
- **Encoding**: Label encoding untuk variabel kategorikal
- **Missing Value Handling**: Strategi imputation yang sesuai domain
- **Scaling**: Standardisasi fitur numerik
- **Feature Selection**: Seleksi fitur terbaik menggunakan statistical tests

### 3. **Model Training & Evaluation** 🤖

#### Algoritma yang Diimplementasikan:
1. **Random Forest Classifier**
   - Ensemble learning dengan multiple decision trees
   - Robust terhadap overfitting
   - Memberikan feature importance

2. **XGBoost Classifier**
   - Gradient boosting yang optimized
   - Performa tinggi untuk classification tasks
   - Built-in regularization

3. **Support Vector Machine (SVM)**
   - Efektif untuk high-dimensional data
   - Kernel trick untuk non-linear classification
   - Good generalization ability

#### Evaluasi Model:
- **Cross-validation** dengan 5-fold stratified
- **Hyperparameter tuning** menggunakan Grid Search
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrix** analysis
- **Feature Importance** ranking

## 🚀 Cara Menggunakan

### 1. Persiapan Environment

```bash
# Clone repository
git clone <repository-url>
cd ML_KIPKuliah

# Install dependencies
pip install -r requirements.txt
```

### 2. Persiapan Data

Pastikan struktur folder data sesuai:
```
Bahan Laporan KIP Kuliah 2022 s.d 2024/
├── CSV_Pendaftar/
│   ├── 2022/
│   ├── 2023/
│   └── 2024/
└── CSV_Penerima/
    ├── penerima KIP Kuliah angkatan 2022.csv
    ├── penerima KIP Kuliah angkatan 2023.csv
    └── penerima KIP Kuliah angkatan 2024.csv
```

### 3. Menjalankan Sistem

```bash
# Jalankan pipeline lengkap
python main.py
```

### 4. Menggunakan Modul Terpisah

```python
# Data exploration saja
from src.data_exploration import KIPDataExplorer
explorer = KIPDataExplorer(base_path)
explorer.run_exploration()

# Preprocessing saja
from src.preprocessing import KIPDataPreprocessor
preprocessor = KIPDataPreprocessor()
processed_data = preprocessor.preprocess_pipeline(df)

# Training model saja
from src.models import KIPClassificationModels
models = KIPClassificationModels()
models.train_and_evaluate(X_train, y_train, X_test, y_test)
```

## 📈 Output dan Hasil

### 1. **Reports** (folder `reports/`)
- `data_exploration_YYYYMMDD_HHMMSS.json`: Hasil eksplorasi data
- `preprocessing_YYYYMMDD_HHMMSS.json`: Summary preprocessing
- `final_classification_report_YYYYMMDD_HHMMSS.json`: Laporan lengkap

### 2. **Models** (folder `models/`)
- Model tersimpan dalam format `.pkl`
- Metadata dan parameter terbaik
- Feature importance rankings

### 3. **Visualizations** (folder `visualizations/`)
- Perbandingan performa model
- Confusion matrices
- Feature importance plots
- Distribution plots

### 4. **Key Metrics yang Dilaporkan**
- **Akurasi Klasifikasi**: % prediksi yang benar
- **Precision**: % prediksi positif yang benar
- **Recall**: % kasus positif yang terdeteksi
- **F1-Score**: Harmonic mean precision dan recall
- **Cross-validation scores**: Validasi robustness model

## 🔍 Feature Engineering

### Fitur Ekonomi:
- `total_family_income`: Total pendapatan keluarga
- `income_per_capita`: Pendapatan per anggota keluarga
- `working_parents_count`: Jumlah orang tua yang bekerja

### Fitur Perumahan:
- `housing_ownership_score`: Skor kepemilikan rumah
- `total_utility_score`: Skor akses utilitas
- `housing_space_ratio`: Rasio luas bangunan terhadap tanah

### Fitur Bantuan Sosial:
- `total_assistance_programs`: Total program bantuan yang diterima
- `has_dtks`, `has_p3ke`, `has_kip`, `has_kks`: Status program bantuan
- `p3ke_desil`: Level desil P3KE

### Fitur Geografis:
- `is_urban`, `is_rural`: Klasifikasi area tempat tinggal
- `from_developed_province`: Asal dari provinsi maju

## 📊 Interpretasi Hasil

### Model Performance:
- **F1-Score > 0.85**: Model sangat baik
- **F1-Score 0.70-0.85**: Model baik
- **F1-Score < 0.70**: Model perlu improvement

### Feature Importance:
Top features yang biasanya paling berpengaruh:
1. Status DTKS/P3KE
2. Total pendapatan keluarga
3. Jumlah program bantuan sosial
4. Kondisi kepemilikan rumah
5. Jumlah tanggungan keluarga

## 🛠️ Konfigurasi Lanjutan

Dalam file `main.py`, Anda dapat mengubah:

```python
config = {
    'test_size': 0.2,                    # Proporsi data test
    'random_state': 42,                  # Seed untuk reproducibility
    'cv_folds': 5,                       # Jumlah fold cross-validation
    'feature_selection': True,           # Aktifkan feature selection
    'hyperparameter_tuning': True       # Aktifkan hyperparameter tuning
}
```

## 🔧 Troubleshooting

### Error Common dan Solusi:

1. **FileNotFoundError**: 
   - Pastikan path data benar
   - Check struktur folder CSV

2. **Memory Error**:
   - Kurangi parameter `n_estimators` di model
   - Gunakan `feature_selection=True`

3. **Import Error**:
   - Install semua dependencies: `pip install -r requirements.txt`
   - Check Python version (minimal 3.8)

## 📝 Kontribusi dan Development

### Untuk Developer:

1. **Menambah Model Baru**:
   ```python
   # Di src/models.py
   def add_new_model(self):
       self.models['new_model'] = NewModelClassifier()
   ```

2. **Menambah Feature Engineering**:
   ```python
   # Di src/preprocessing.py
   def _engineer_new_features(self, df):
       # Implementasi feature baru
       return df
   ```

3. **Testing**:
   ```bash
   # Test individual modules
   python src/data_exploration.py
   python src/preprocessing.py
   python src/models.py
   ```

## 📚 Dependencies

- **pandas**: Data manipulation dan analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting framework
- **matplotlib & seaborn**: Data visualization
- **jupyter**: Interactive development

## 👥 Tim Pengembang

**Data Mining Professional Team**
- Focus: Socio-economic classification systems
- Expertise: Educational data analysis
- Methodology: Domain-driven feature engineering

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan analisis pendidikan dan penelitian.

## 📞 Support

Untuk pertanyaan atau issues:
1. Check dokumentasi di README ini
2. Review error messages dan logs
3. Validate data structure dan format

---

**🎯 "Mengoptimalkan distribusi bantuan pendidikan melalui analisis data yang akurat dan fair"**
