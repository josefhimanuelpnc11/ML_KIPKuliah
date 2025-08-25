# Analisis KIP Kuliah: Clustering dan Classification Pipeline

## 📋 Deskripsi Project

Project ini menyediakan analisis komprehensif untuk data KIP Kuliah menggunakan teknik machine learning yang menggabungkan **clustering** (unsupervised) dan **classification** (supervised) untuk mengidentifikasi pola dan tipologi pendaftar KIP Kuliah.

## 🎯 Tujuan Analisis

1. **Clustering Analysis**: Mengidentifikasi tipologi/pola pendaftar KIP Kuliah berdasarkan karakteristik mereka
2. **Classification Modeling**: Membangun model prediksi untuk mengklasifikasikan pendaftar ke dalam cluster tertentu
3. **Model Comparison**: Membandingkan performa berbagai algoritma machine learning
4. **Feature Analysis**: Menganalisis faktor-faktor yang paling berpengaruh dalam pengelompokan

## 🏗️ Struktur Project (Modular Python)

```
ML_KIPKuliah/
├── main.py                           # Script utama untuk menjalankan analisis
├── run_analysis.py                   # Runner script dengan CLI interface
├── src/                              # Source code modules
│   ├── config.py                     # Konfigurasi central untuk semua komponen
│   ├── data_loader.py                # Module untuk loading data CSV
│   ├── data_preprocessor.py          # Module untuk preprocessing data
│   ├── clustering_analyzer.py        # Module untuk analisis clustering
│   ├── classification_trainer.py     # Module untuk training classification
│   └── results_exporter.py           # Module untuk export hasil
├── logs/                             # Log files untuk monitoring
├── results/                          # Hasil analisis (Excel, CSV, reports, plots)
├── models/                           # Trained models (untuk future use)
├── requirements.txt                  # Dependencies Python
└── README.md                         # Dokumentasi ini
```

## 🚀 Workflow Analisis

### **Pipeline Terintegrasi:**
```
Data Loading → Preprocessing → Clustering → Classification → Evaluation → Export
```

### **1. Data Loading (`data_loader.py`)**
- Load data pendaftar KIP Kuliah dari tahun 2022-2024
- Kombinasi data dari berbagai jalur seleksi
- Data validation dan quality checks

### **2. Data Preprocessing (`data_preprocessor.py`)**
- Handle missing values dengan strategi yang tepat
- Identifikasi kolom numerik vs kategorikal
- Encoding untuk data kategorikal
- Scaling untuk data numerik
- Persiapan data mixed untuk K-Prototypes

### **3. Clustering Analysis (`clustering_analyzer.py`)**
- **Algoritma**: K-Prototypes (menangani data campuran numerik + kategorikal)
- **Optimal Clusters**: Elbow method dengan visualisasi
- **Cluster Profiling**: Analisis karakteristik setiap cluster
- **Quality Metrics**: Silhouette score dan internal validation

### **4. Classification Training (`classification_trainer.py`)**
- **Target**: Cluster labels dari hasil clustering
- **Algoritma**: Random Forest, XGBoost, SVM
- **Hyperparameter Tuning**: Grid Search dengan Cross Validation
- **Feature Importance**: Analisis variabel paling berpengaruh

### **5. Results Export (`results_exporter.py`)**
- **Excel**: Multiple sheets dengan semua hasil
- **CSV**: File terpisah untuk setiap dataset
- **Summary Report**: Laporan ringkasan komprehensif
- **Visualizations**: Plot dan grafik analisis

### 4. Classification Modeling (Supervised)
- **Target**: Cluster labels dari hasil clustering
- **Algoritma yang dibandingkan**:
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
- **Hyperparameter Tuning**: Grid Search dengan Cross Validation

### 5. Model Evaluation & Comparison
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualisasi**: Confusion matrices, performance comparison
- **Feature Importance**: Analisis variabel yang paling berpengaruh

### 6. Export Results
- **Excel**: Multiple sheets dengan semua hasil
- **CSV**: File terpisah untuk setiap dataset
- **Summary Report**: Laporan ringkasan dalam format text

## 📦 Instalasi dan Setup

### **1. Clone Repository**
```bash
git clone https://github.com/josefhimanuelpnc11/ML_KIPKuliah.git
cd ML_KIPKuliah
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Setup Data Structure**
Pastikan struktur data sesuai:
```
ML_KIPKuliah/
├── Bahan Laporan KIP Kuliah 2022 s.d 2024/
│   ├── CSV_Pendaftar/
│   │   ├── 2022/
│   │   ├── 2023/
│   │   └── 2024/
│   └── CSV_Penerima/
```

## 🔧 Cara Menjalankan Analisis

### **Method 1: Simple Runner (Recommended)**
```bash
python run_analysis.py
```

### **Method 2: Direct Main Script**
```bash
python main.py
```

### **Method 3: With Custom Parameters**
```bash
python run_analysis.py --data-path "path/to/data" --max-clusters 8 --verbose
```

## 📊 Output dan Hasil

### 1. Clustering Results
- **Cluster Profiles**: Karakteristik setiap cluster
- **Cluster Distribution**: Jumlah dan persentase pendaftar per cluster
- **Visualizations**: Grafik distribusi dan profil cluster

### 2. Classification Results
- **Model Performance**: Comparison table semua model
- **Best Model**: Model dengan performa terbaik
- **Feature Importance**: Ranking variabel paling berpengaruh

### 3. Export Files
- `results/kip_analysis_YYYYMMDD_HHMMSS.xlsx`: Excel dengan multiple sheets
- `results/csv_results_YYYYMMDD_HHMMSS/`: Folder berisi CSV files
- `results/analysis_summary_YYYYMMDD_HHMMSS.txt`: Summary report

## 🔍 Key Features

### Data Handling
- ✅ **Mixed Data Types**: Menangani data numerik dan kategorikal secara bersamaan
- ✅ **Missing Values**: Handling otomatis dengan strategi yang tepat
- ✅ **Data Validation**: Quality checks dan validation preprocessing

### Algorithms
- ✅ **K-Prototypes**: Clustering untuk data campuran
- ✅ **Multiple Classifiers**: Perbandingan 3 algoritma classification
- ✅ **Hyperparameter Tuning**: Grid search otomatis

### Visualizations
- ✅ **Cluster Analysis**: Distribusi dan profil cluster
- ✅ **Model Comparison**: Performance metrics visualization
- ✅ **Feature Importance**: Ranking dan visualization
- ✅ **Confusion Matrices**: Detailed model evaluation

### Export & Reporting
- ✅ **Excel Export**: Professional format dengan multiple sheets
- ✅ **CSV Export**: Individual files untuk analisis lanjutan
- ✅ **Summary Report**: Executive summary dalam text format

## 🎨 Dependencies

```
pandas>=1.5.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
matplotlib>=3.5.0      # Basic plotting
seaborn>=0.11.0        # Statistical visualization
scikit-learn>=1.1.0    # Machine learning
kmodes>=0.12.2         # K-Prototypes algorithm
xgboost>=1.7.0         # XGBoost classifier
openpyxl>=3.0.0        # Excel export
jupyter>=1.0.0         # Notebook environment
ipykernel>=6.0.0       # Jupyter kernel
```

## 🔬 Metodologi

### Clustering Approach
1. **Data Preprocessing**: Clean dan prepare data mixed types
2. **Algorithm Selection**: K-Prototypes untuk menangani numerical + categorical
3. **Optimal K**: Elbow method untuk menentukan jumlah cluster optimal
4. **Validation**: Silhouette analysis dan cluster interpretation

### Classification Approach
1. **Feature Engineering**: Encoding dan scaling yang appropriate
2. **Model Selection**: Multiple algorithms dengan karakteristik berbeda
3. **Hyperparameter Tuning**: Grid search dengan cross-validation
4. **Evaluation**: Multiple metrics untuk assessment komprehensif

## 📈 Expected Outcomes

### Research Insights
- **Tipologi Pendaftar**: Identifikasi profil pendaftar KIP Kuliah
- **Pattern Recognition**: Pola karakteristik yang membedakan kelompok
- **Predictive Model**: Model untuk klasifikasi pendaftar baru

### Business Value
- **Segmentasi**: Understanding pendaftar berdasarkan karakteristik
- **Targeting**: Strategi yang lebih tepat untuk setiap segment
- **Decision Support**: Data-driven insights untuk kebijakan

## ⚠️ Catatan Penting

1. **Data Privacy**: Pastikan data sensitif sudah di-anonymize
2. **Interpretability**: Hasil clustering bersifat descriptive, bukan prescriptive
3. **Validation**: Selalu validasi hasil dengan domain expert
4. **Limitations**: Model terbatas pada data yang tersedia (2022-2024)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

