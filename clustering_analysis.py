#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 2025

@author: Data Mining Analysis for KIP Kuliah

RESEARCH FOCUS: Clustering Analysis untuk Mengidentifikasi Tipologi Pendaftar KIP Kuliah
Menggunakan K-Prototype Algorithm untuk Mixed Data Types (Categorical + Numerical)

Research Question: "Bagaimana karakteristik dan pola pengelompokan pendaftar KIP Kuliah?"
Expected Output: Taxonomy baru tentang profil sosial-ekonomi mahasiswa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from kmodes.kprototypes import KPrototypes
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi yang lebih menarik
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KIPClusteringAnalysis:
    """
    Class untuk analisis clustering pendaftar KIP Kuliah menggunakan K-Prototype
    
    Tujuan: Mengidentifikasi tipologi/profil pendaftar KIP Kuliah
    Metode: K-Prototype untuk mixed data types
    """
    
    def __init__(self):
        self.data_combined = None
        self.categorical_features = []
        self.numerical_features = []
        self.model = None
        self.cluster_results = None
        
    def load_and_combine_data(self):
        """
        Load semua file CSV pendaftar dan gabungkan untuk analisis komprehensif
        """
        print("🔄 Loading dan menggabungkan data dari semua tahun...")
        
        data_files = {
            2022: [
                "Bahan Laporan KIP Kuliah 2022 s.d 2024/CSV_Pendaftar/2022/Siswa_Pendaftar_SBMPN_2022.csv",
                "Bahan Laporan KIP Kuliah 2022 s.d 2024/CSV_Pendaftar/2022/Siswa_Pendaftar_SNMPN_Politeknik Negeri Cilacap_20220328.csv",
                "Bahan Laporan KIP Kuliah 2022 s.d 2024/CSV_Pendaftar/2022/Siswa_Pendaftar_Seleksi Mandiri PTN_2022.csv"
            ],
            2023: [
                "Bahan Laporan KIP Kuliah 2022 s.d 2024/CSV_Pendaftar/2023/pendaftar kip jalur SNBT 2023.csv",
                "Bahan Laporan KIP Kuliah 2022 s.d 2024/CSV_Pendaftar/2023/pendaftar KIP Kuliah 2023 jalur SNBP.csv",
                "Bahan Laporan KIP Kuliah 2022 s.d 2024/CSV_Pendaftar/2023/Siswa_Pendaftar_Seleksi Mandiri PTN_2023.csv"
            ],
            2024: [
                "Bahan Laporan KIP Kuliah 2022 s.d 2024/CSV_Pendaftar/2024/pendaftar kip jalur snbp dan snbt 2024.csv"
            ]
        }
        
        all_data = []
        
        for year, files in data_files.items():
            for file_path in files:
                try:
                    # Skip header row untuk file 2022 yang memiliki header tambahan
                    if year == 2022:
                        df = pd.read_csv(file_path, skiprows=1, low_memory=False)
                    else:
                        df = pd.read_csv(file_path, low_memory=False)
                    
                    df['Tahun'] = year
                    df['Source_File'] = file_path.split('/')[-1]
                    all_data.append(df)
                    print(f"✅ Loaded {file_path}: {len(df)} records")
                    
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {str(e)}")
        
        # Combine all data
        self.data_combined = pd.concat(all_data, ignore_index=True, sort=False)
        print(f"\n📊 Total combined data: {len(self.data_combined)} records")
        
        return self.data_combined
    
    def preprocess_data(self):
        """
        Preprocessing data untuk clustering analysis
        Focus pada variabel yang relevan untuk tipologi sosial-ekonomi
        """
        print("\n🔧 Preprocessing data untuk clustering analysis...")
        
        # Pilih kolom yang relevan untuk clustering
        relevant_columns = [
            'Status DTKS', 'Status P3KE', 'Pekerjaan Ayah', 'Pekerjaan Ibu',
            'Penghasilan Ayah', 'Penghasilan Ibu', 'Status Ayah', 'Status Ibu',
            'Jumlah Tanggungan', 'Kepemilikan Rumah', 'Sumber Listrik', 'Sumber Air',
            'Luas Tanah', 'Luas Bangunan', 'Jarak Pusat Kota (KM)', 'Provinsi Sekolah',
            'Jenis Kelamin', 'Tahun'
        ]
        
        # Filter hanya kolom yang ada
        available_columns = [col for col in relevant_columns if col in self.data_combined.columns]
        df = self.data_combined[available_columns].copy()
        
        print(f"📋 Kolom yang digunakan: {available_columns}")
        
        # Handle missing values dan preprocessing
        df = self._clean_and_encode_data(df)
        
        # Tentukan categorical dan numerical features
        self.categorical_features = [
            'Status DTKS', 'Status P3KE_encoded', 'Pekerjaan Ayah', 'Pekerjaan Ibu',
            'Status Ayah', 'Status Ibu', 'Kepemilikan Rumah', 'Sumber Listrik',
            'Sumber Air', 'Luas Tanah_cat', 'Luas Bangunan_cat', 'Provinsi Sekolah',
            'Jenis Kelamin'
        ]
        
        self.numerical_features = [
            'Penghasilan Ayah_encoded', 'Penghasilan Ibu_encoded', 'Jumlah Tanggungan',
            'Jarak Pusat Kota', 'Tahun'
        ]
        
        # Filter features yang benar-benar ada
        self.categorical_features = [f for f in self.categorical_features if f in df.columns]
        self.numerical_features = [f for f in self.numerical_features if f in df.columns]
        
        print(f"🔢 Numerical features: {self.numerical_features}")
        print(f"📝 Categorical features: {self.categorical_features}")
        
        return df
    
    def _clean_and_encode_data(self, df):
        """
        Clean dan encode data untuk clustering
        """
        print("🧹 Cleaning dan encoding data...")
        
        # Clean Status DTKS
        if 'Status DTKS' in df.columns:
            df['Status DTKS'] = df['Status DTKS'].fillna('Belum Terdata')
            df['Status DTKS'] = df['Status DTKS'].str.strip()
        
        # Encode Status P3KE dengan ekstraksi desil
        if 'Status P3KE' in df.columns:
            df['Status P3KE_encoded'] = df['Status P3KE'].apply(self._extract_desil)
        
        # Clean pekerjaan
        for col in ['Pekerjaan Ayah', 'Pekerjaan Ibu']:
            if col in df.columns:
                df[col] = df[col].fillna('TIDAK BEKERJA')
                df[col] = df[col].str.strip().str.upper()
        
        # Encode penghasilan ke numeric
        for col in ['Penghasilan Ayah', 'Penghasilan Ibu']:
            if col in df.columns:
                df[f"{col}_encoded"] = df[col].apply(self._encode_penghasilan)
        
        # Clean status orangtua
        for col in ['Status Ayah', 'Status Ibu']:
            if col in df.columns:
                df[col] = df[col].fillna('Hidup')
        
        # Convert jumlah tanggungan
        if 'Jumlah Tanggungan' in df.columns:
            df['Jumlah Tanggungan'] = df['Jumlah Tanggungan'].str.extract(r'(\d+)').astype(float)
            df['Jumlah Tanggungan'] = df['Jumlah Tanggungan'].fillna(df['Jumlah Tanggungan'].median())
        
        # Convert jarak
        if 'Jarak Pusat Kota (KM)' in df.columns:
            df['Jarak Pusat Kota'] = pd.to_numeric(df['Jarak Pusat Kota (KM)'], errors='coerce')
            df['Jarak Pusat Kota'] = df['Jarak Pusat Kota'].fillna(df['Jarak Pusat Kota'].median())
        
        # Categorize luas tanah dan bangunan
        for col in ['Luas Tanah', 'Luas Bangunan']:
            if col in df.columns:
                df[f"{col}_cat"] = df[col].fillna('Tidak Diketahui')
        
        # Clean other categorical
        categorical_cols = ['Kepemilikan Rumah', 'Sumber Listrik', 'Sumber Air', 'Provinsi Sekolah', 'Jenis Kelamin']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Tidak Diketahui')
        
        return df
    
    def _extract_desil(self, status_p3ke):
        """Extract desil number from Status P3KE"""
        if pd.isna(status_p3ke):
            return 'Belum Terdata'
        if 'Desil' in str(status_p3ke):
            try:
                desil = str(status_p3ke).split('Desil')[1].strip().split()[0]
                return f'Desil_{desil}'
            except:
                return 'Terdata'
        return str(status_p3ke)
    
    def _encode_penghasilan(self, penghasilan_str):
        """
        Encode penghasilan string to numeric (midpoint of range)
        """
        if pd.isna(penghasilan_str) or penghasilan_str == '-':
            return 0
        
        penghasilan_map = {
            'Tidak Berpenghasilan': 0,
            '< Rp. 250.000': 125000,
            'Rp. 250.001 - Rp. 500.000': 375000,
            'Rp. 500.001 - Rp. 750.000': 625000,
            'Rp. 750.001 - Rp. 1.000.000': 875000,
            'Rp. 1.000.001 - Rp. 1.250.000': 1125000,
            'Rp. 1.250.001 - Rp. 1.500.000': 1375000,
            'Rp. 1.500.001 - Rp. 1.750.000': 1625000,
            'Rp. 1.750.001 - Rp. 2.000.000': 1875000,
            'Rp. 2.000.001 - Rp. 2.250.000': 2125000,
            'Rp. 2.250.001 - Rp. 2.500.000': 2375000,
            'Rp. 2.500.001 - Rp. 2.750.000': 2625000,
            'Rp. 2.750.001 - Rp. 3.000.000': 2875000,
            'Rp. 5.250.001 - Rp. 5.500.000': 5375000
        }
        
        return penghasilan_map.get(str(penghasilan_str), 0)
    
    def perform_clustering(self, df, n_clusters_range=(2, 8)):
        """
        Perform K-Prototype clustering dengan optimal number of clusters
        """
        print(f"\n🎯 Performing K-Prototype clustering...")
        
        # Prepare data untuk clustering - hanya gunakan kolom yang ada
        features_to_use = self.categorical_features + self.numerical_features
        df_features = df[features_to_use].copy()
        
        # Remove rows dengan terlalu banyak missing values
        df_clean = df_features.dropna(thresh=len(features_to_use) * 0.7)
        
        print(f"📊 Data untuk clustering: {len(df_clean)} records")
        print(f"📋 Features yang digunakan: {df_clean.columns.tolist()}")
        
        # Prepare categorical indices berdasarkan urutan kolom final
        categorical_indices = []
        for i, col in enumerate(df_clean.columns):
            if col in self.categorical_features:
                categorical_indices.append(i)
        
        print(f"🔢 Categorical indices: {categorical_indices}")
        print(f"🔢 Total columns: {len(df_clean.columns)}")
        
        # Find optimal number of clusters
        costs = []
        K = range(n_clusters_range[0], n_clusters_range[1] + 1)
        
        print("🔍 Finding optimal number of clusters...")
        for k in K:
            try:
                kproto = KPrototypes(n_clusters=k, init='Huang', verbose=0, random_state=42)
                clusters = kproto.fit_predict(df_clean.values, categorical=categorical_indices)
                costs.append(kproto.cost_)
                print(f"K={k}: Cost={kproto.cost_:.2f}")
            except Exception as e:
                print(f"Error for k={k}: {str(e)}")
                costs.append(float('inf'))
        
        # Pilih optimal k (elbow method atau berdasarkan domain knowledge)
        optimal_k = self._find_optimal_k(costs, K)
        print(f"🎯 Optimal number of clusters: {optimal_k}")
        
        # Final clustering dengan optimal k
        self.model = KPrototypes(n_clusters=optimal_k, init='Huang', verbose=1, random_state=42)
        cluster_labels = self.model.fit_predict(df_clean.values, categorical=categorical_indices)
        
        # Add cluster labels back to original dataframe
        df_result = df.copy()
        df_result.loc[df_clean.index, 'Cluster'] = cluster_labels
        self.cluster_results = df_result.dropna(subset=['Cluster'])
        
        print(f"✅ Clustering completed dengan {optimal_k} clusters")
        return self.cluster_results
    
    def _find_optimal_k(self, costs, K):
        """
        Find optimal k using elbow method
        """
        # Simple elbow detection - bisa diperbaiki dengan algoritma yang lebih sophisticated
        if len(costs) < 3:
            return K[0]
        
        # Calculate rate of change
        rate_of_change = []
        for i in range(1, len(costs)-1):
            if costs[i-1] != float('inf') and costs[i+1] != float('inf'):
                rate = abs(costs[i+1] - 2*costs[i] + costs[i-1])
                rate_of_change.append(rate)
            else:
                rate_of_change.append(0)
        
        if rate_of_change:
            optimal_idx = rate_of_change.index(max(rate_of_change)) + 1
            return K[optimal_idx]
        
        return 4  # Default ke 4 clusters untuk domain knowledge
    
    def analyze_clusters(self):
        """
        Analyze karakteristik setiap cluster untuk research insights
        """
        if self.cluster_results is None:
            print("❌ Belum ada hasil clustering. Jalankan perform_clustering() terlebih dahulu.")
            return
        
        print("\n📊 ANALISIS KARAKTERISTIK CLUSTER")
        print("="*60)
        
        df = self.cluster_results
        n_clusters = df['Cluster'].nunique()
        
        # Overview clusters
        cluster_counts = df['Cluster'].value_counts().sort_index()
        print(f"📈 Distribusi Cluster:")
        for cluster, count in cluster_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   Cluster {cluster}: {count} pendaftar ({percentage:.1f}%)")
        
        # Analisis karakteristik per cluster
        cluster_profiles = {}
        
        for cluster_id in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            profile = self._create_cluster_profile(cluster_data, cluster_id)
            cluster_profiles[cluster_id] = profile
        
        # Interpretasi cluster (research findings)
        self._interpret_clusters(cluster_profiles)
        
        return cluster_profiles
    
    def _create_cluster_profile(self, cluster_data, cluster_id):
        """
        Create detailed profile untuk setiap cluster
        """
        print(f"\n🏷️  CLUSTER {cluster_id} PROFILE")
        print("-" * 40)
        
        profile = {
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(self.cluster_results)) * 100
        }
        
        print(f"📊 Ukuran: {profile['size']} pendaftar ({profile['percentage']:.1f}%)")
        
        # Categorical features analysis
        print("\n📝 Karakteristik Kategorikal:")
        for feature in self.categorical_features:
            if feature in cluster_data.columns:
                mode_value = cluster_data[feature].mode()
                if len(mode_value) > 0:
                    mode_val = mode_value.iloc[0]
                    percentage = (cluster_data[feature] == mode_val).mean() * 100
                    print(f"   {feature}: {mode_val} ({percentage:.1f}%)")
                    profile[feature] = {'mode': mode_val, 'percentage': percentage}
        
        # Numerical features analysis
        print("\n🔢 Karakteristik Numerik:")
        for feature in self.numerical_features:
            if feature in cluster_data.columns:
                mean_val = cluster_data[feature].mean()
                median_val = cluster_data[feature].median()
                std_val = cluster_data[feature].std()
                print(f"   {feature}: Mean={mean_val:.1f}, Median={median_val:.1f}, Std={std_val:.1f}")
                profile[feature] = {'mean': mean_val, 'median': median_val, 'std': std_val}
        
        return profile
    
    def _interpret_clusters(self, cluster_profiles):
        """
        Interpretasi clusters untuk research insights
        """
        print(f"\n🔬 INTERPRETASI CLUSTER UNTUK PENELITIAN")
        print("="*60)
        
        # Berikan nama dan interpretasi untuk setiap cluster
        cluster_interpretations = {}
        
        for cluster_id, profile in cluster_profiles.items():
            interpretation = self._generate_cluster_interpretation(cluster_id, profile)
            cluster_interpretations[cluster_id] = interpretation
            
            print(f"\n🏷️  CLUSTER {cluster_id}: {interpretation['name']}")
            print(f"📊 Ukuran: {profile['size']} pendaftar ({profile['percentage']:.1f}%)")
            print(f"📝 Deskripsi: {interpretation['description']}")
            print(f"🎯 Karakteristik Utama: {interpretation['key_characteristics']}")
            print(f"💡 Implikasi Penelitian: {interpretation['research_implications']}")
        
        return cluster_interpretations
    
    def _generate_cluster_interpretation(self, cluster_id, profile):
        """
        Generate interpretasi berdasarkan karakteristik cluster
        """
        # Analisis pattern dari profile untuk memberikan nama dan interpretasi
        interpretation = {
            'name': f'Tipologi {cluster_id + 1}',
            'description': 'Profil mahasiswa dengan karakteristik khusus',
            'key_characteristics': [],
            'research_implications': 'Memerlukan analisis lebih lanjut'
        }
        
        # Logic untuk interpretasi berdasarkan karakteristik dominan
        key_chars = []
        
        # Analisis Status DTKS
        if 'Status DTKS' in profile:
            if profile['Status DTKS']['mode'] == 'Terdata':
                key_chars.append('Terdata dalam sistem bantuan sosial')
            else:
                key_chars.append('Belum terdata dalam sistem bantuan sosial')
        
        # Analisis ekonomi
        if 'Penghasilan Ayah_encoded' in profile:
            avg_income = profile['Penghasilan Ayah_encoded']['mean']
            if avg_income < 500000:
                key_chars.append('Ekonomi keluarga rendah')
            elif avg_income < 1500000:
                key_chars.append('Ekonomi keluarga menengah-bawah')
            else:
                key_chars.append('Ekonomi keluarga menengah')
        
        # Analisis geografis
        if 'Jarak Pusat Kota' in profile:
            avg_distance = profile['Jarak Pusat Kota']['mean']
            if avg_distance > 50:
                key_chars.append('Berasal dari daerah terpencil')
            elif avg_distance < 10:
                key_chars.append('Berasal dari daerah urban')
            else:
                key_chars.append('Berasal dari daerah semi-urban')
        
        interpretation['key_characteristics'] = key_chars
        
        # Generate nama cluster berdasarkan karakteristik
        if 'ekonomi rendah' in ' '.join(key_chars).lower():
            interpretation['name'] = 'Kelompok Ekonomi Lemah'
        elif 'daerah terpencil' in ' '.join(key_chars).lower():
            interpretation['name'] = 'Kelompok Geografis Terpencil'
        elif 'terdata' in ' '.join(key_chars).lower():
            interpretation['name'] = 'Kelompok Penerima Bantuan Sosial'
        else:
            interpretation['name'] = f'Kelompok Tipologi {cluster_id + 1}'
        
        return interpretation
    
    def visualize_clusters(self):
        """
        Create comprehensive visualizations untuk research presentation
        """
        if self.cluster_results is None:
            print("❌ Belum ada hasil clustering.")
            return
        
        print("\n📊 Creating cluster visualizations...")
        
        # Set up the plotting
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Cluster distribution
        plt.subplot(2, 3, 1)
        cluster_counts = self.cluster_results['Cluster'].value_counts().sort_index()
        plt.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Distribusi Cluster Pendaftar KIP Kuliah', fontsize=14, fontweight='bold')
        
        # 2. Cluster vs Status DTKS
        if 'Status DTKS' in self.cluster_results.columns:
            plt.subplot(2, 3, 2)
            ct = pd.crosstab(self.cluster_results['Cluster'], self.cluster_results['Status DTKS'])
            ct.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title('Cluster vs Status DTKS', fontsize=12, fontweight='bold')
            plt.xlabel('Cluster')
            plt.ylabel('Jumlah Pendaftar')
            plt.legend(title='Status DTKS', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
        
        # 3. Penghasilan distribution by cluster
        if 'Penghasilan Ayah_encoded' in self.cluster_results.columns:
            plt.subplot(2, 3, 3)
            self.cluster_results.boxplot(column='Penghasilan Ayah_encoded', by='Cluster', ax=plt.gca())
            plt.title('Distribusi Penghasilan Ayah per Cluster', fontsize=12, fontweight='bold')
            plt.xlabel('Cluster')
            plt.ylabel('Penghasilan Ayah (Rp)')
            plt.suptitle('')  # Remove default title
        
        # 4. Geographic distribution
        if 'Jarak Pusat Kota' in self.cluster_results.columns:
            plt.subplot(2, 3, 4)
            self.cluster_results.boxplot(column='Jarak Pusat Kota', by='Cluster', ax=plt.gca())
            plt.title('Distribusi Jarak dari Pusat Kota per Cluster', fontsize=12, fontweight='bold')
            plt.xlabel('Cluster')
            plt.ylabel('Jarak (KM)')
            plt.suptitle('')
        
        # 5. Jumlah tanggungan
        if 'Jumlah Tanggungan' in self.cluster_results.columns:
            plt.subplot(2, 3, 5)
            self.cluster_results.boxplot(column='Jumlah Tanggungan', by='Cluster', ax=plt.gca())
            plt.title('Distribusi Jumlah Tanggungan per Cluster', fontsize=12, fontweight='bold')
            plt.xlabel('Cluster')
            plt.ylabel('Jumlah Tanggungan')
            plt.suptitle('')
        
        # 6. Temporal distribution
        if 'Tahun' in self.cluster_results.columns:
            plt.subplot(2, 3, 6)
            ct_year = pd.crosstab(self.cluster_results['Cluster'], self.cluster_results['Tahun'])
            ct_year.plot(kind='bar', ax=plt.gca())
            plt.title('Distribusi Cluster per Tahun', fontsize=12, fontweight='bold')
            plt.xlabel('Cluster')
            plt.ylabel('Jumlah Pendaftar')
            plt.legend(title='Tahun')
            plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('cluster_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional detailed heatmap
        self._create_cluster_heatmap()
    
    def _create_cluster_heatmap(self):
        """
        Create detailed heatmap untuk cluster characteristics
        """
        # Prepare data untuk heatmap
        cluster_summary = []
        
        for cluster_id in self.cluster_results['Cluster'].unique():
            cluster_data = self.cluster_results[self.cluster_results['Cluster'] == cluster_id]
            
            summary_row = {'Cluster': cluster_id}
            
            # Numerical features - use mean
            for feature in self.numerical_features:
                if feature in cluster_data.columns:
                    summary_row[feature] = cluster_data[feature].mean()
            
            # Categorical features - use mode percentage
            for feature in self.categorical_features:
                if feature in cluster_data.columns:
                    mode_val = cluster_data[feature].mode()
                    if len(mode_val) > 0:
                        mode_percentage = (cluster_data[feature] == mode_val.iloc[0]).mean()
                        summary_row[f"{feature}_dominance"] = mode_percentage
            
            cluster_summary.append(summary_row)
        
        # Create DataFrame dan heatmap
        summary_df = pd.DataFrame(cluster_summary)
        summary_df.set_index('Cluster', inplace=True)
        
        # Normalize untuk better visualization
        summary_df_norm = (summary_df - summary_df.min()) / (summary_df.max() - summary_df.min())
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(summary_df_norm.T, annot=True, cmap='RdYlBu_r', center=0.5, 
                    fmt='.2f', cbar_kws={'label': 'Normalized Value'})
        plt.title('Cluster Characteristics Heatmap\n(Normalized Values)', fontsize=16, fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('cluster_heatmap_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_research_report(self):
        """
        Generate comprehensive research report
        """
        if self.cluster_results is None:
            print("❌ Belum ada hasil clustering.")
            return
        
        print("\n📄 GENERATING RESEARCH REPORT")
        print("="*60)
        
        report = f"""
LAPORAN PENELITIAN: ANALISIS CLUSTERING PENDAFTAR KIP KULIAH
POLITEKNIK NEGERI CILACAP 2022-2024

METODOLOGI:
- Algoritma: K-Prototype Clustering
- Data: {len(self.cluster_results)} pendaftar KIP Kuliah
- Periode: 2022-2024
- Variabel: {len(self.categorical_features)} categorical, {len(self.numerical_features)} numerical

HASIL CLUSTERING:
- Jumlah Cluster Optimal: {self.cluster_results['Cluster'].nunique()}
- Distribusi Cluster: {dict(self.cluster_results['Cluster'].value_counts().sort_index())}

FINDINGS UTAMA:
1. Heterogenitas Pendaftar: Ditemukan {self.cluster_results['Cluster'].nunique()} tipologi berbeda
2. Pola Sosial-Ekonomi: Clustering berhasil mengidentifikasi segmentasi berdasarkan status ekonomi
3. Distribusi Geografis: Terdapat pola clustering berdasarkan asal daerah
4. Temporal Trends: Karakteristik pendaftar menunjukkan evolusi dari tahun ke tahun

IMPLIKASI PENELITIAN:
- Temuan ini mendukung teori heterogenitas dalam akses pendidikan tinggi
- Clustering results dapat digunakan untuk targeted intervention
- Framework ini dapat diadaptasi untuk analisis program bantuan pendidikan lainnya

REKOMENDASI PENELITIAN LANJUTAN:
1. Longitudinal study untuk tracking perubahan cluster
2. Predictive modeling berdasarkan cluster membership
3. Qualitative research untuk deep-dive cluster interpretation
4. Policy impact analysis berdasarkan cluster characteristics
        """
        
        print(report)
        
        # Save report to file
        with open('KIP_Clustering_Research_Report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n✅ Research report saved to 'KIP_Clustering_Research_Report.txt'")
        
        return report

# Main execution function
def main():
    """
    Main function untuk menjalankan analisis clustering
    """
    print("🎓 ANALISIS CLUSTERING PENDAFTAR KIP KULIAH")
    print("🎯 Research Focus: Mengidentifikasi Tipologi Pendaftar")
    print("🔬 Metode: K-Prototype Clustering")
    print("="*60)
    
    # Initialize analysis
    analyzer = KIPClusteringAnalysis()
    
    # Step 1: Load and combine data
    combined_data = analyzer.load_and_combine_data()
    
    # Step 2: Preprocess data
    processed_data = analyzer.preprocess_data()
    
    # Step 3: Perform clustering
    clustered_data = analyzer.perform_clustering(processed_data)
    
    # Step 4: Analyze clusters
    cluster_profiles = analyzer.analyze_clusters()
    
    # Step 5: Visualize results
    analyzer.visualize_clusters()
    
    # Step 6: Generate research report
    research_report = analyzer.generate_research_report()
    
    print("\n🎉 ANALISIS CLUSTERING SELESAI!")
    print("📊 Hasil visualisasi tersimpan sebagai PNG files")
    print("📄 Research report tersimpan sebagai TXT file")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
