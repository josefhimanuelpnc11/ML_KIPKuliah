"""
KIP KULIAH INEQUALITY ANALYSIS
Publication-Ready Visualizations for Academic Journal

Author: Data Analysis Team
Date: August 2025
Purpose: Analyze socio-economic, geographic, and educational disparities in KIP Kuliah access
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif'
})

class KIPKuliahInequalityAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pendaftar_data = None
        self.penerima_data = None
        self.combined_data = None
        
    def load_and_prepare_data(self):
        """Load all CSV files and prepare combined dataset"""
        print("📂 Loading KIP Kuliah datasets...")
        
        # Load pendaftar files
        pendaftar_files = glob.glob(f'{self.data_path}/CSV_Pendaftar/**/*.csv', recursive=True)
        penerima_files = glob.glob(f'{self.data_path}/CSV_Penerima/*.csv', recursive=True)
        
        pendaftar_dfs = []
        penerima_dfs = []
        
        # Load pendaftar data
        for file in pendaftar_files:
            try:
                df = pd.read_csv(file, skiprows=1, low_memory=False)
                df['file_source'] = os.path.basename(file)
                df['year'] = self._extract_year(file)
                pendaftar_dfs.append(df)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        
        # Load penerima data
        for file in penerima_files:
            try:
                df = pd.read_csv(file, skiprows=1, low_memory=False)
                df['file_source'] = os.path.basename(file)
                df['year'] = self._extract_year(file)
                penerima_dfs.append(df)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        
        self.pendaftar_data = pd.concat(pendaftar_dfs, ignore_index=True)
        self.penerima_data = pd.concat(penerima_dfs, ignore_index=True)
        
        # Create combined dataset with acceptance status
        self._create_combined_dataset()
        
        print(f"✅ Data loaded: {len(self.pendaftar_data)} pendaftar, {len(self.penerima_data)} penerima")
        
    def _extract_year(self, filename):
        """Extract year from filename"""
        if '2022' in filename:
            return 2022
        elif '2023' in filename:
            return 2023
        elif '2024' in filename:
            return 2024
        return None
    
    def _create_combined_dataset(self):
        """Create combined dataset with acceptance indicator"""
        # Add acceptance status
        self.pendaftar_data['accepted'] = 0
        self.penerima_data['accepted'] = 1
        
        # Standardize common columns
        common_cols = ['No. Pendaftaran', 'Nama Siswa', 'NIK', 'NISN', 'Jenis Kelamin', 
                      'Tanggal Lahir', 'Asal Sekolah', 'Kab/Kota Sekolah', 'Provinsi Sekolah',
                      'Pekerjaan Ayah', 'Penghasilan Ayah', 'Pekerjaan Ibu', 'Penghasilan Ibu',
                      'Kepemilikan Rumah', 'Sumber Listrik', 'Luas Tanah', 'Luas Bangunan',
                      'year', 'accepted']
        
        # Add Status P3KE for penerima (not in pendaftar)
        if 'Status P3KE' in self.penerima_data.columns:
            common_cols.append('Status P3KE')
            self.pendaftar_data['Status P3KE'] = 'Unknown'
        
        # Filter common columns that exist in both datasets
        pendaftar_common = [col for col in common_cols if col in self.pendaftar_data.columns]
        penerima_common = [col for col in common_cols if col in self.penerima_data.columns]
        
        # Use intersection of columns
        final_cols = list(set(pendaftar_common) & set(penerima_common))
        
        self.combined_data = pd.concat([
            self.pendaftar_data[final_cols],
            self.penerima_data[final_cols]
        ], ignore_index=True)
        
        # Remove duplicates based on key identifiers
        self.combined_data = self.combined_data.drop_duplicates(subset=['No. Pendaftaran'], keep='last')
        
        print(f"✅ Combined dataset created: {len(self.combined_data)} total records")
        
    def clean_and_engineer_features(self):
        """Clean data and engineer features for analysis"""
        print("🔧 Engineering features for inequality analysis...")
        
        df = self.combined_data.copy()
        
        # Clean P3KE status
        if 'Status P3KE' in df.columns:
            df['p3ke_desil'] = df['Status P3KE'].str.extract(r'Desil (\d+)').astype(float)
            df['has_p3ke'] = df['Status P3KE'].str.contains('Terdata', na=False)
        
        # Engineer age from birth date
        if 'Tanggal Lahir' in df.columns:
            df['age'] = pd.to_datetime(df['Tanggal Lahir'], errors='coerce')
            df['age'] = 2024 - df['age'].dt.year
        
        # Clean income data
        income_cols = ['Penghasilan Ayah', 'Penghasilan Ibu']
        for col in income_cols:
            if col in df.columns:
                df[f'{col}_clean'] = self._categorize_income(df[col])
        
        # School type categorization
        if 'Asal Sekolah' in df.columns:
            df['school_type'] = df['Asal Sekolah'].apply(self._categorize_school_type)
        
        # Geographic features
        if 'Kab/Kota Sekolah' in df.columns:
            df['is_urban'] = df['Kab/Kota Sekolah'].str.contains('Kota|KOTA', na=False)
        
        # Housing quality index
        housing_features = ['Kepemilikan Rumah', 'Sumber Listrik', 'Luas Tanah', 'Luas Bangunan']
        df['housing_quality_score'] = self._calculate_housing_score(df, housing_features)
        
        self.combined_data = df
        print("✅ Feature engineering completed")
        
    def _categorize_income(self, income_series):
        """Categorize income into ordinal levels"""
        income_mapping = {
            'Tidak Berpenghasilan': 0,
            '< Rp. 250.000': 1,
            'Rp. 250.001 - Rp. 500.000': 2,
            'Rp. 500.001 - Rp. 750.000': 3,
            'Rp. 750.001 - Rp. 1.000.000': 4,
            'Rp. 1.000.001 - Rp. 1.250.000': 5,
            'Rp. 1.250.001 - Rp. 1.500.000': 6,
            'Rp. 1.500.001 - Rp. 1.750.000': 7,
            'Rp. 1.750.001 - Rp. 2.000.000': 8,
            '> Rp. 2.000.000': 9,
            '-': np.nan,  # Handle missing data
            '': np.nan,   # Handle empty strings
        }
        # Apply mapping and handle unmapped values as NaN
        result = income_series.map(income_mapping)
        # Fill remaining unmapped values with -1 (unknown category)
        result = result.fillna(-1)
        return result
    
    def _categorize_school_type(self, school_name):
        """Categorize school types"""
        if pd.isna(school_name):
            return 'Unknown'
        school_name = str(school_name).upper()
        if 'SMA' in school_name or 'SMAN' in school_name:
            return 'SMA Negeri'
        elif 'SMK' in school_name:
            return 'SMK'
        elif 'MA' in school_name:
            return 'Madrasah'
        elif 'SMAS' in school_name:
            return 'SMA Swasta'
        else:
            return 'Other'
    
    def _calculate_housing_score(self, df, features):
        """Calculate composite housing quality score"""
        score = 0
        
        # Ownership
        if 'Kepemilikan Rumah' in df.columns:
            ownership_score = df['Kepemilikan Rumah'].map({
                'Sendiri': 3, 'Sewa Tahunan': 2, 'Sewa Bulanan': 1, 
                'Menumpang': 0, 'Tidak Memiliki': 0
            }).fillna(0)
            score += ownership_score
        
        # Electricity
        if 'Sumber Listrik' in df.columns:
            electricity_score = df['Sumber Listrik'].map({
                'PLN': 2, 'PLN dan Genset': 1, 'Menumpang tetangga': 0, 
                'Tidak Ada': 0
            }).fillna(0)
            score += electricity_score
        
        return score
    
    def plot_socioeconomic_disparity(self, save_path=None):
        """Figure 1: Socio-Economic Disparity Analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Disparitas Sosial-Ekonomi dalam Akses KIP Kuliah', fontsize=16, fontweight='bold')
        
        df = self.combined_data.copy()
        
        # 1. P3KE Desil Analysis
        if 'p3ke_desil' in df.columns:
            desil_analysis = df.groupby('p3ke_desil')['accepted'].agg(['count', 'sum', 'mean']).reset_index()
            desil_analysis['acceptance_rate'] = desil_analysis['mean']
            desil_analysis['ci_lower'], desil_analysis['ci_upper'] = self._calculate_ci(
                desil_analysis['sum'], desil_analysis['count']
            )
            
            axes[0,0].bar(desil_analysis['p3ke_desil'], desil_analysis['acceptance_rate'], 
                         alpha=0.7, color='steelblue')
            axes[0,0].errorbar(desil_analysis['p3ke_desil'], desil_analysis['acceptance_rate'],
                              yerr=[desil_analysis['acceptance_rate'] - desil_analysis['ci_lower'],
                                    desil_analysis['ci_upper'] - desil_analysis['acceptance_rate']],
                              fmt='none', color='black', capsize=3)
            axes[0,0].set_title('Tingkat Penerimaan berdasarkan Desil P3KE')
            axes[0,0].set_xlabel('Desil P3KE (1=Termiskin, 10=Terkaya)')
            axes[0,0].set_ylabel('Tingkat Penerimaan')
        
        # 2. Family Income Analysis
        if 'Penghasilan Ayah_clean' in df.columns:
            income_analysis = df.groupby('Penghasilan Ayah_clean')['accepted'].agg(['count', 'mean']).reset_index()
            income_analysis = income_analysis[income_analysis['count'] >= 10]  # Filter small groups
            
            # Filter out non-numeric income values
            income_analysis = income_analysis[income_analysis['Penghasilan Ayah_clean'].apply(
                lambda x: str(x).replace('.', '').replace('-', '').isdigit() if pd.notna(x) else False
            )]
            
            if len(income_analysis) > 0:
                axes[0,1].bar(range(len(income_analysis)), income_analysis['mean'], 
                             alpha=0.7, color='lightcoral')
                axes[0,1].set_title('Tingkat Penerimaan berdasarkan Penghasilan Ayah')
                axes[0,1].set_xlabel('Level Penghasilan (0=Tidak Ada, 9=Tertinggi)')
                axes[0,1].set_ylabel('Tingkat Penerimaan')
                axes[0,1].set_xticks(range(len(income_analysis)))
                # Safe conversion to labels
                income_labels = [str(int(x)) if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() 
                               else str(x) for x in income_analysis['Penghasilan Ayah_clean']]
                axes[0,1].set_xticklabels(income_labels, rotation=45)
            else:
                axes[0,1].text(0.5, 0.5, 'Data Penghasilan Tidak Mencukupi', ha='center', va='center', transform=axes[0,1].transAxes)
        
        # 3. Housing Quality vs Acceptance
        if 'housing_quality_score' in df.columns:
            housing_analysis = df.groupby('housing_quality_score')['accepted'].agg(['count', 'mean']).reset_index()
            
            axes[1,0].scatter(housing_analysis['housing_quality_score'], housing_analysis['mean'],
                             s=housing_analysis['count']*5, alpha=0.6, color='green')
            axes[1,0].set_title('Kualitas Hunian vs Tingkat Penerimaan')
            axes[1,0].set_xlabel('Skor Kualitas Hunian')
            axes[1,0].set_ylabel('Tingkat Penerimaan')
        
        # 4. Multiple Disadvantage Analysis
        df['disadvantage_count'] = 0
        if 'p3ke_desil' in df.columns:
            df['disadvantage_count'] += (df['p3ke_desil'] <= 3).astype(int)
        if 'Penghasilan Ayah_clean' in df.columns:
            df['disadvantage_count'] += (df['Penghasilan Ayah_clean'] <= 2).astype(int)
        if 'housing_quality_score' in df.columns:
            df['disadvantage_count'] += (df['housing_quality_score'] <= 2).astype(int)
            
        disadvantage_analysis = df.groupby('disadvantage_count')['accepted'].agg(['count', 'mean']).reset_index()
        
        axes[1,1].bar(disadvantage_analysis['disadvantage_count'], disadvantage_analysis['mean'],
                     alpha=0.7, color='orange')
        axes[1,1].set_title('Dampak Kemiskinan Multidimensi')
        axes[1,1].set_xlabel('Jumlah Faktor Kemiskinan')
        axes[1,1].set_ylabel('Tingkat Penerimaan')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_educational_disparity(self, save_path=None):
        """Figure 2: Educational Access Inequality"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Disparitas Pendidikan dalam Akses KIP Kuliah', fontsize=16, fontweight='bold')
        
        df = self.combined_data.copy()
        
        # 1. School Type Analysis
        if 'school_type' in df.columns:
            school_analysis = df.groupby('school_type')['accepted'].agg(['count', 'mean']).reset_index()
            school_analysis = school_analysis[school_analysis['count'] >= 10]
            
            axes[0,0].barh(school_analysis['school_type'], school_analysis['mean'], alpha=0.7)
            axes[0,0].set_title('Tingkat Penerimaan berdasarkan Jenis Sekolah')
            axes[0,0].set_xlabel('Tingkat Penerimaan')
        
        # 2. Urban vs Rural
        if 'is_urban' in df.columns:
            urban_analysis = df.groupby('is_urban')['accepted'].agg(['count', 'mean']).reset_index()
            urban_labels = ['Pedesaan', 'Perkotaan']
            
            axes[0,1].pie(urban_analysis['mean'], labels=urban_labels, autopct='%1.1f%%', 
                         colors=['lightblue', 'salmon'])
            axes[0,1].set_title('Tingkat Penerimaan: Perkotaan vs Pedesaan')
        
        # 3. Provincial Distribution
        if 'Provinsi Sekolah' in df.columns:
            prov_analysis = df.groupby('Provinsi Sekolah')['accepted'].agg(['count', 'mean']).reset_index()
            prov_analysis = prov_analysis[prov_analysis['count'] >= 5].sort_values('mean')
            
            axes[1,0].barh(range(len(prov_analysis)), prov_analysis['mean'], alpha=0.7, color='purple')
            axes[1,0].set_title('Tingkat Penerimaan berdasarkan Provinsi')
            axes[1,0].set_xlabel('Tingkat Penerimaan')
            axes[1,0].set_yticks(range(len(prov_analysis)))
            axes[1,0].set_yticklabels(prov_analysis['Provinsi Sekolah'])
        
        # 4. Age Distribution
        if 'age' in df.columns:
            age_bins = pd.cut(df['age'], bins=5, labels=['Sangat Muda', 'Muda', 'Rata-rata', 'Tua', 'Sangat Tua'])
            age_analysis = df.groupby(age_bins)['accepted'].agg(['count', 'mean']).reset_index()
            
            axes[1,1].bar(range(len(age_analysis)), age_analysis['mean'], alpha=0.7, color='gold')
            axes[1,1].set_title('Tingkat Penerimaan berdasarkan Kelompok Usia')
            axes[1,1].set_xlabel('Kelompok Usia')
            axes[1,1].set_ylabel('Tingkat Penerimaan')
            axes[1,1].set_xticks(range(len(age_analysis)))
            axes[1,1].set_xticklabels(age_analysis['age'], rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_trends(self, save_path=None):
        """Figure 3: Temporal Inequality Trends (2022-2024)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tren Temporal Kesetaraan Akses KIP Kuliah (2022-2024)', fontsize=16, fontweight='bold')
        
        df = self.combined_data.copy()
        
        # 1. Overall acceptance trend
        yearly_trend = df.groupby('year')['accepted'].agg(['count', 'sum', 'mean']).reset_index()
        
        axes[0,0].plot(yearly_trend['year'], yearly_trend['mean'], marker='o', linewidth=2, markersize=8)
        axes[0,0].set_title('Tren Tingkat Penerimaan Secara Keseluruhan')
        axes[0,0].set_xlabel('Tahun')
        axes[0,0].set_ylabel('Tingkat Penerimaan')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Gender equity trend
        if 'Jenis Kelamin' in df.columns:
            gender_trend = df.groupby(['year', 'Jenis Kelamin'])['accepted'].mean().unstack()
            
            for gender in gender_trend.columns:
                axes[0,1].plot(gender_trend.index, gender_trend[gender], 
                              marker='o', label=f'{gender}', linewidth=2)
            axes[0,1].set_title('Tren Kesetaraan Gender')
            axes[0,1].set_xlabel('Tahun')
            axes[0,1].set_ylabel('Tingkat Penerimaan')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. P3KE Desil inequality trend
        if 'p3ke_desil' in df.columns:
            # Calculate inequality index (Gini-like)
            inequality_by_year = []
            for year in [2022, 2023, 2024]:
                year_data = df[df['year'] == year]
                if len(year_data) > 0:
                    desil_rates = year_data.groupby('p3ke_desil')['accepted'].mean()
                    inequality_index = desil_rates.std() / desil_rates.mean()  # Coefficient of variation
                    inequality_by_year.append({'year': year, 'inequality': inequality_index})
            
            if inequality_by_year:
                ineq_df = pd.DataFrame(inequality_by_year)
                axes[1,0].bar(ineq_df['year'], ineq_df['inequality'], alpha=0.7, color='red')
                axes[1,0].set_title('Indeks Ketimpangan berdasarkan P3KE')
                axes[1,0].set_xlabel('Tahun')
                axes[1,0].set_ylabel('Indeks Ketimpangan (CV)')
        
        # 4. Regional disparity trend
        if 'is_urban' in df.columns:
            urban_rural_trend = df.groupby(['year', 'is_urban'])['accepted'].mean().unstack()
            urban_rural_trend['gap'] = urban_rural_trend[True] - urban_rural_trend[False]
            
            axes[1,1].bar(urban_rural_trend.index, urban_rural_trend['gap'], alpha=0.7, color='teal')
            axes[1,1].set_title('Kesenjangan Penerimaan Perkotaan-Pedesaan')
            axes[1,1].set_xlabel('Tahun')
            axes[1,1].set_ylabel('Perkotaan - Pedesaan (Tingkat Penerimaan)')
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_intersectional_analysis(self, save_path=None):
        """Figure 4: Intersectional Inequality Analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analisis Interseksional Akses KIP Kuliah', fontsize=16, fontweight='bold')
        
        df = self.combined_data.copy()
        
        # 1. Gender x Economic Status
        if 'Jenis Kelamin' in df.columns and 'p3ke_desil' in df.columns:
            # Create economic status categories
            df['economic_status'] = pd.cut(df['p3ke_desil'], bins=[0, 3, 7, 10], 
                                         labels=['Miskin (1-3)', 'Menengah (4-7)', 'Kaya (8-10)'])
            
            crosstab = pd.crosstab(df['economic_status'], df['Jenis Kelamin'], 
                                  df['accepted'], aggfunc='mean')
            
            sns.heatmap(crosstab, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0,0])
            axes[0,0].set_title('Gender × Status Ekonomi: Tingkat Penerimaan')
        
        # 2. School Type x Urban/Rural
        if 'school_type' in df.columns and 'is_urban' in df.columns:
            crosstab2 = pd.crosstab(df['school_type'], df['is_urban'], 
                                   df['accepted'], aggfunc='mean')
            
            sns.heatmap(crosstab2, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0,1])
            axes[0,1].set_title('Jenis Sekolah × Lokasi Geografis')
        
        # 3. Economic Status x Geographic
        if 'economic_status' in df.columns and 'is_urban' in df.columns:
            eco_geo = df.groupby(['economic_status', 'is_urban'])['accepted'].agg(['count', 'mean']).reset_index()
            eco_geo['location'] = eco_geo['is_urban'].map({True: 'Perkotaan', False: 'Pedesaan'})
            
            pivot_data = eco_geo.pivot(index='economic_status', columns='location', values='mean')
            
            x = np.arange(len(pivot_data.index))
            width = 0.35
            
            axes[1,0].bar(x - width/2, pivot_data['Pedesaan'], width, label='Pedesaan', alpha=0.7)
            axes[1,0].bar(x + width/2, pivot_data['Perkotaan'], width, label='Perkotaan', alpha=0.7)
            axes[1,0].set_title('Status Ekonomi × Interaksi Geografis')
            axes[1,0].set_xlabel('Status Ekonomi')
            axes[1,0].set_ylabel('Tingkat Penerimaan')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(pivot_data.index)
            axes[1,0].legend()
        
        # 4. Compound Disadvantage Score
        df['compound_disadvantage'] = 0
        
        # Economic disadvantage
        if 'p3ke_desil' in df.columns:
            df['compound_disadvantage'] += (df['p3ke_desil'] <= 3).astype(int) * 2
        
        # Geographic disadvantage
        if 'is_urban' in df.columns:
            df['compound_disadvantage'] += (~df['is_urban']).astype(int)
        
        # Gender disadvantage (if applicable)
        if 'Jenis Kelamin' in df.columns:
            # Assuming female disadvantage based on context
            df['compound_disadvantage'] += (df['Jenis Kelamin'] == 'P').astype(int)
        
        compound_analysis = df.groupby('compound_disadvantage')['accepted'].agg(['count', 'mean']).reset_index()
        compound_analysis = compound_analysis[compound_analysis['count'] >= 5]
        
        axes[1,1].bar(compound_analysis['compound_disadvantage'], compound_analysis['mean'],
                     alpha=0.7, color='maroon')
        axes[1,1].set_title('Dampak Kemiskinan Multidimensi')
        axes[1,1].set_xlabel('Skor Kemiskinan Multidimensi')
        axes[1,1].set_ylabel('Tingkat Penerimaan')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_chart_descriptions(self, output_dir):
        """Generate comprehensive descriptions for all charts"""
        
        # First, let's analyze the actual data ranges
        df = self.combined_data.copy()
        
        descriptions = []
        
        # Header
        descriptions.append("=" * 80)
        descriptions.append("📊 DESKRIPSI LENGKAP GRAFIK ANALISIS KETIMPANGAN KIP KULIAH")
        descriptions.append("=" * 80)
        descriptions.append("")
        descriptions.append("File ini berisi penjelasan detail setiap grafik, parameter yang digunakan,")
        descriptions.append("rentang nilai, dan interpretasi untuk memudahkan pembacaan visualisasi.")
        descriptions.append("")
        
        # Chart 1: Socio-Economic Disparity
        descriptions.append("🎯 GAMBAR 1: DISPARITAS SOSIAL-EKONOMI")
        descriptions.append("-" * 50)
        descriptions.append("")
        
        # P3KE Desil Analysis
        descriptions.append("📈 1.1 Tingkat Penerimaan berdasarkan Desil P3KE")
        descriptions.append("   Deskripsi: Menunjukkan tingkat penerimaan KIP Kuliah berdasarkan")
        descriptions.append("              status kemiskinan keluarga dalam sistem P3KE")
        descriptions.append("")
        descriptions.append("   Parameter P3KE Desil:")
        descriptions.append("   • Desil 1-3  : Keluarga termiskin (prioritas tinggi)")
        descriptions.append("   • Desil 4-6  : Keluarga menengah bawah") 
        descriptions.append("   • Desil 7-10 : Keluarga menengah ke atas")
        descriptions.append("")
        if 'p3ke_desil' in df.columns:
            desil_counts = df['p3ke_desil'].value_counts().sort_index()
            descriptions.append("   Data tersedia:")
            for desil, count in desil_counts.items():
                if pd.notna(desil):
                    descriptions.append(f"   • Desil {int(desil)}: {count} mahasiswa")
        descriptions.append("")
        
        # Income Analysis
        descriptions.append("📈 1.2 Tingkat Penerimaan berdasarkan Penghasilan Ayah")
        descriptions.append("   Deskripsi: Tingkat penerimaan berdasarkan kategori penghasilan ayah")
        descriptions.append("")
        descriptions.append("   Parameter Penghasilan Ayah (Kode → Rentang):")
        descriptions.append("   • -1 : Data tidak tersedia/tidak diketahui")
        descriptions.append("   •  0 : Tidak berpenghasilan")
        descriptions.append("   •  1 : < Rp 250.000")
        descriptions.append("   •  2 : Rp 250.001 - Rp 500.000")
        descriptions.append("   •  3 : Rp 500.001 - Rp 750.000")
        descriptions.append("   •  4 : Rp 750.001 - Rp 1.000.000")
        descriptions.append("   •  5 : Rp 1.000.001 - Rp 1.250.000")
        descriptions.append("   •  6 : Rp 1.250.001 - Rp 1.500.000")
        descriptions.append("   •  7 : Rp 1.500.001 - Rp 1.750.000")
        descriptions.append("   •  8 : Rp 1.750.001 - Rp 2.000.000")
        descriptions.append("   •  9 : > Rp 2.000.000")
        descriptions.append("")
        if 'Penghasilan Ayah_clean' in df.columns:
            income_counts = df['Penghasilan Ayah_clean'].value_counts().sort_index()
            descriptions.append("   Data tersedia:")
            for income, count in income_counts.items():
                if pd.notna(income):
                    descriptions.append(f"   • Kode {int(income)}: {count} mahasiswa")
        descriptions.append("")
        
        # Housing Quality
        descriptions.append("📈 1.3 Kualitas Hunian vs Tingkat Penerimaan")
        descriptions.append("   Deskripsi: Skor kualitas hunian berdasarkan gabungan kepemilikan")
        descriptions.append("              rumah dan akses listrik")
        descriptions.append("")
        descriptions.append("   Parameter Skor Kualitas Hunian (0-5):")
        descriptions.append("   • Kepemilikan Rumah:")
        descriptions.append("     - Milik sendiri: +3 poin")
        descriptions.append("     - Sewa tahunan: +2 poin")
        descriptions.append("     - Sewa bulanan: +1 poin")
        descriptions.append("     - Menumpang/tidak punya: +0 poin")
        descriptions.append("   • Sumber Listrik:")
        descriptions.append("     - PLN: +2 poin")
        descriptions.append("     - PLN + Genset: +1 poin")
        descriptions.append("     - Menumpang/tidak ada: +0 poin")
        descriptions.append("")
        
        # Multiple Disadvantage
        descriptions.append("📈 1.4 Dampak Kemiskinan Multidimensi")
        descriptions.append("   Deskripsi: Menghitung dampak akumulatif berbagai faktor kemiskinan")
        descriptions.append("")
        descriptions.append("   Parameter Faktor Kemiskinan:")
        descriptions.append("   • Faktor Ekonomi: P3KE Desil 1-3 (+2 poin)")
        descriptions.append("   • Faktor Geografis: Tinggal di pedesaan (+1 poin)")
        descriptions.append("   • Faktor Gender: Perempuan (+1 poin)")
        descriptions.append("   • Skor 0-4: Semakin tinggi = semakin disadvantaged")
        descriptions.append("")
        
        # Chart 2: Educational Disparity
        descriptions.append("🎯 GAMBAR 2: DISPARITAS PENDIDIKAN")
        descriptions.append("-" * 50)
        descriptions.append("")
        
        descriptions.append("📈 2.1 Tingkat Penerimaan berdasarkan Jenis Sekolah")
        descriptions.append("   Deskripsi: Perbandingan tingkat penerimaan berdasarkan asal sekolah")
        descriptions.append("")
        descriptions.append("   Kategori Jenis Sekolah:")
        descriptions.append("   • SMA Negeri: Sekolah menengah atas negeri")
        descriptions.append("   • SMA Swasta: Sekolah menengah atas swasta") 
        descriptions.append("   • SMK: Sekolah menengah kejuruan")
        descriptions.append("   • Madrasah: Madrasah Aliyah")
        descriptions.append("   • Other: Sekolah lainnya/tidak terkategorisasi")
        descriptions.append("")
        
        descriptions.append("📈 2.2 Perkotaan vs Pedesaan")
        descriptions.append("   Deskripsi: Perbandingan akses berdasarkan lokasi geografis")
        descriptions.append("")
        descriptions.append("   Parameter Geografis:")
        descriptions.append("   • Perkotaan: Kab/Kota mengandung kata 'Kota'")
        descriptions.append("   • Pedesaan: Kab/Kota lainnya (umumnya 'Kabupaten')")
        descriptions.append("")
        
        descriptions.append("📈 2.3 Tingkat Penerimaan berdasarkan Provinsi")
        descriptions.append("   Deskripsi: Distribusi penerimaan per provinsi asal sekolah")
        descriptions.append("   Note: Hanya menampilkan provinsi dengan minimal 5 pendaftar")
        descriptions.append("")
        
        descriptions.append("📈 2.4 Tingkat Penerimaan berdasarkan Kelompok Usia")
        descriptions.append("   Deskripsi: Distribusi penerimaan berdasarkan usia mahasiswa")
        descriptions.append("")
        if 'age' in df.columns:
            age_min = df['age'].min()
            age_max = df['age'].max()
            age_mean = df['age'].mean()
            
            # Calculate age statistics
            descriptions.append("   Parameter Kelompok Usia:")
            descriptions.append("   • Sangat Muda: ≤ 18 tahun")
            descriptions.append("   • Muda: 19-20 tahun")
            descriptions.append("   • Rata-rata: 21-22 tahun")
            descriptions.append("   • Tua: 23-25 tahun")
            descriptions.append("   • Sangat Tua: ≥ 26 tahun")
            descriptions.append("")
            descriptions.append(f"   Statistik Usia:")
            descriptions.append(f"   • Rentang: {age_min:.0f} - {age_max:.0f} tahun")
            descriptions.append(f"   • Rata-rata: {age_mean:.1f} tahun")
        descriptions.append("")
        
        # Chart 3: Temporal Trends
        descriptions.append("🎯 GAMBAR 3: TREN TEMPORAL (2022-2024)")
        descriptions.append("-" * 50)
        descriptions.append("")
        
        descriptions.append("📈 3.1 Tren Tingkat Penerimaan Secara Keseluruhan")
        descriptions.append("   Deskripsi: Perubahan tingkat penerimaan KIP Kuliah dari 2022-2024")
        descriptions.append("")
        yearly_stats = df.groupby('year')['accepted'].agg(['count', 'sum', 'mean'])
        descriptions.append("   Data per Tahun:")
        for year, stats in yearly_stats.iterrows():
            if pd.notna(year):
                descriptions.append(f"   • {int(year)}: {stats['sum']}/{stats['count']} = {stats['mean']:.1%}")
        descriptions.append("")
        
        descriptions.append("📈 3.2 Tren Kesetaraan Gender")
        descriptions.append("   Deskripsi: Perbandingan tingkat penerimaan laki-laki vs perempuan")
        descriptions.append("              dari tahun ke tahun")
        descriptions.append("")
        
        descriptions.append("📈 3.3 Indeks Ketimpangan berdasarkan P3KE")
        descriptions.append("   Deskripsi: Mengukur tingkat ketimpangan antar desil P3KE")
        descriptions.append("   Metode: Coefficient of Variation (CV) = std/mean")
        descriptions.append("   • CV tinggi = ketimpangan tinggi")
        descriptions.append("   • CV rendah = distribusi lebih merata")
        descriptions.append("")
        
        descriptions.append("📈 3.4 Kesenjangan Penerimaan Perkotaan-Pedesaan")
        descriptions.append("   Deskripsi: Selisih tingkat penerimaan antara perkotaan dan pedesaan")
        descriptions.append("   • Nilai positif = perkotaan lebih tinggi")
        descriptions.append("   • Nilai negatif = pedesaan lebih tinggi")
        descriptions.append("")
        
        # Chart 4: Intersectional Analysis
        descriptions.append("🎯 GAMBAR 4: ANALISIS INTERSEKSIONAL")
        descriptions.append("-" * 50)
        descriptions.append("")
        
        descriptions.append("📈 4.1 Gender × Status Ekonomi")
        descriptions.append("   Deskripsi: Heatmap tingkat penerimaan berdasarkan kombinasi")
        descriptions.append("              gender dan status ekonomi")
        descriptions.append("")
        descriptions.append("   Parameter Status Ekonomi:")
        descriptions.append("   • Miskin (1-3): P3KE Desil 1-3")
        descriptions.append("   • Menengah (4-7): P3KE Desil 4-7")
        descriptions.append("   • Kaya (8-10): P3KE Desil 8-10")
        descriptions.append("")
        
        descriptions.append("📈 4.2 Jenis Sekolah × Lokasi Geografis")
        descriptions.append("   Deskripsi: Interaksi antara jenis sekolah dan lokasi geografis")
        descriptions.append("   Format: Heatmap dengan intensitas warna menunjukkan")
        descriptions.append("           tingkat penerimaan")
        descriptions.append("")
        
        descriptions.append("📈 4.3 Status Ekonomi × Interaksi Geografis")
        descriptions.append("   Deskripsi: Perbandingan dampak status ekonomi di perkotaan")
        descriptions.append("              vs pedesaan")
        descriptions.append("   Format: Bar chart berpasangan (perkotaan vs pedesaan)")
        descriptions.append("           untuk setiap kategori ekonomi")
        descriptions.append("")
        
        descriptions.append("📈 4.4 Dampak Kemiskinan Multidimensi")
        descriptions.append("   Deskripsi: Mengukur dampak akumulatif multiple disadvantage")
        descriptions.append("   Metode: Penjumlahan skor dari berbagai faktor kemiskinan")
        descriptions.append("   • Skor 0: Tidak ada faktor kemiskinan")
        descriptions.append("   • Skor 4: Semua faktor kemiskinan ada")
        descriptions.append("")
        
        # Chart 5: Multivariate Regression Analysis
        descriptions.append("🎯 GAMBAR 5: ANALISIS REGRESI LOGISTIK MULTIVARIAT")
        descriptions.append("-" * 50)
        descriptions.append("")
        
        descriptions.append("📈 5.1 Pentingnya Variabel (Magnitude Koefisien)")
        descriptions.append("   Deskripsi: Menunjukkan seberapa kuat pengaruh masing-masing variabel")
        descriptions.append("              terhadap probabilitas diterima KIP Kuliah")
        descriptions.append("")
        descriptions.append("   Interpretasi:")
        descriptions.append("   • Sumbu X: Magnitude (nilai absolut) koefisien regresi logistik")
        descriptions.append("   • Sumbu Y: Variabel-variabel yang dianalisis")
        descriptions.append("   • Semakin panjang bar = semakin kuat pengaruhnya")
        descriptions.append("   • Urutan dari atas: variabel paling berpengaruh")
        descriptions.append("")
        
        descriptions.append("📈 5.2 Arah Pengaruh Variabel")
        descriptions.append("   Deskripsi: Menunjukkan apakah variabel meningkatkan atau")
        descriptions.append("              menurunkan peluang diterima")
        descriptions.append("")
        descriptions.append("   Parameter Warna:")
        descriptions.append("   • Hijau (positif): Meningkatkan peluang diterima")
        descriptions.append("   • Merah (negatif): Menurunkan peluang diterima")
        descriptions.append("   • Garis putus-putus hitam: Titik netral (koefisien = 0)")
        descriptions.append("")
        
        descriptions.append("📈 5.3 Kalibrasi Model: Prediksi vs Aktual")
        descriptions.append("   Deskripsi: Mengukur seberapa akurat prediksi probabilitas model")
        descriptions.append("")
        descriptions.append("   Interpretasi:")
        descriptions.append("   • Sumbu X: Probabilitas prediksi model (0-1)")
        descriptions.append("   • Sumbu Y: Tingkat penerimaan aktual")
        descriptions.append("   • Garis merah putus-putus: Prediksi sempurna")
        descriptions.append("   • Titik mendekati garis = model terkalibrasi baik")
        descriptions.append("")
        
        descriptions.append("📈 5.4 Odds Ratio dengan Signifikansi Statistik")
        descriptions.append("   Deskripsi: Mengukur seberapa besar variabel mengubah odds")
        descriptions.append("              diterima, dengan tingkat signifikansi statistik")
        descriptions.append("")
        descriptions.append("   Parameter Odds Ratio:")
        descriptions.append("   • OR > 1 (hijau): Meningkatkan odds diterima")
        descriptions.append("   • OR < 1 (merah): Menurunkan odds diterima")
        descriptions.append("   • OR = 1: Tidak ada pengaruh")
        descriptions.append("")
        descriptions.append("   Tingkat Signifikansi:")
        descriptions.append("   • *** : p < 0.001 (sangat signifikan)")
        descriptions.append("   • ** : p < 0.01 (signifikan)")
        descriptions.append("   • * : p < 0.05 (cukup signifikan)")
        descriptions.append("   • (kosong): p ≥ 0.05 (tidak signifikan)")
        descriptions.append("")
        descriptions.append("   Contoh Interpretasi:")
        descriptions.append("   • OR = 1.5***: Meningkatkan odds 50%, sangat signifikan")
        descriptions.append("   • OR = 0.7**: Menurunkan odds 30%, signifikan")
        descriptions.append("")
        
        descriptions.append("🔍 VARIABEL YANG DIANALISIS DALAM REGRESI:")
        descriptions.append("")
        descriptions.append("   Ekonomi:")
        descriptions.append("   • P3KE Desil: Tingkat kemiskinan keluarga (1-10)")
        descriptions.append("   • Penghasilan Ayah: Kategori penghasilan ayah (0-9)")
        descriptions.append("   • Kualitas Hunian: Skor gabungan rumah + listrik (0-5)")
        descriptions.append("")
        descriptions.append("   Geografis:")
        descriptions.append("   • Perkotaan: 1=Perkotaan, 0=Pedesaan")
        descriptions.append("")
        descriptions.append("   Demografis:")
        descriptions.append("   • Perempuan: 1=Perempuan, 0=Laki-laki")
        descriptions.append("   • Usia: Usia mahasiswa saat mendaftar")
        descriptions.append("")
        descriptions.append("   Pendidikan:")
        descriptions.append("   • Jenis Sekolah: Encoded kategorical (SMA/SMK/MA/dll)")
        descriptions.append("")
        
        # Data Quality Notes
        descriptions.append("⚠️  CATATAN KUALITAS DATA")
        descriptions.append("-" * 50)
        descriptions.append("")
        descriptions.append("1. Data Missing:")
        descriptions.append("   • Kode -1 pada penghasilan = data tidak tersedia/invalid")
        descriptions.append("   • P3KE Desil kosong untuk sebagian data pendaftar")
        descriptions.append("   • Beberapa kategori dengan sampel kecil (<10) tidak ditampilkan")
        descriptions.append("")
        descriptions.append("2. Metodologi:")
        descriptions.append("   • Confidence intervals menggunakan distribusi normal")
        descriptions.append("   • Filter minimum 5-10 sampel untuk analisis statistik")
        descriptions.append("   • Missing values dihandle sesuai konteks masing-masing variabel")
        descriptions.append("")
        descriptions.append("3. Interpretasi:")
        descriptions.append("   • Semua tingkat penerimaan dalam rentang 0-1 (0% - 100%)")
        descriptions.append("   • Error bars menunjukkan 95% confidence interval")
        descriptions.append("   • Ukuran sampel mempengaruhi reliability analisis")
        descriptions.append("")
        
        # Statistical Summary
        descriptions.append("📊 RINGKASAN STATISTIK DATASET")
        descriptions.append("-" * 50)
        descriptions.append("")
        descriptions.append(f"• Total Mahasiswa: {len(df):,}")
        descriptions.append(f"• Diterima KIP: {df['accepted'].sum():,}")
        descriptions.append(f"• Tingkat Penerimaan Keseluruhan: {df['accepted'].mean():.1%}")
        descriptions.append(f"• Periode Data: 2022-2024")
        descriptions.append("")
        
        if 'Jenis Kelamin' in df.columns:
            gender_dist = df['Jenis Kelamin'].value_counts()
            descriptions.append("• Distribusi Gender:")
            for gender, count in gender_dist.items():
                gender_label = "Laki-laki" if gender == "L" else "Perempuan"
                descriptions.append(f"  - {gender_label}: {count:,} ({count/len(df):.1%})")
        descriptions.append("")
        
        if 'year' in df.columns:
            year_dist = df['year'].value_counts().sort_index()
            descriptions.append("• Distribusi per Tahun:")
            for year, count in year_dist.items():
                if pd.notna(year):
                    descriptions.append(f"  - {int(year)}: {count:,} mahasiswa")
        descriptions.append("")
        
        descriptions.append("=" * 80)
        descriptions.append("📅 Dokumen dibuat: " + pd.Timestamp.now().strftime("%d %B %Y, %H:%M WIB"))
        descriptions.append("🔗 Sumber: Analisis KIP Kuliah Politeknik Negeri Cilacap")
        descriptions.append("=" * 80)
        
        # Save to file
        desc_file = f"{output_dir}/deskripsi_grafik_lengkap.txt"
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(descriptions))
        
        print(f"📝 Deskripsi lengkap disimpan: {desc_file}")
        
        return desc_file
    
    def perform_multivariate_analysis(self, save_path=None):
        """Figure 5: Multivariate Logistic Regression Analysis"""
        print("🔍 Melakukan analisis regresi logistik multivariat...")
        
        df = self.combined_data.copy()
        
        # Prepare features for regression
        features_for_regression = []
        feature_names = []
        
        # Economic features
        if 'p3ke_desil' in df.columns:
            df['p3ke_desil_filled'] = df['p3ke_desil'].fillna(df['p3ke_desil'].median())
            features_for_regression.append(df['p3ke_desil_filled'])
            feature_names.append('P3KE Desil')
        
        if 'Penghasilan Ayah_clean' in df.columns:
            df['income_ayah_filled'] = df['Penghasilan Ayah_clean'].fillna(0)
            features_for_regression.append(df['income_ayah_filled'])
            feature_names.append('Penghasilan Ayah')
        
        # Housing quality
        if 'housing_quality_score' in df.columns:
            features_for_regression.append(df['housing_quality_score'])
            feature_names.append('Kualitas Hunian')
        
        # Geographic (binary)
        if 'is_urban' in df.columns:
            features_for_regression.append(df['is_urban'].astype(int))
            feature_names.append('Perkotaan (1=Ya)')
        
        # Gender (binary)
        if 'Jenis Kelamin' in df.columns:
            df['is_female'] = (df['Jenis Kelamin'] == 'P').astype(int)
            features_for_regression.append(df['is_female'])
            feature_names.append('Perempuan (1=Ya)')
        
        # Age
        if 'age' in df.columns:
            df['age_filled'] = df['age'].fillna(df['age'].median())
            features_for_regression.append(df['age_filled'])
            feature_names.append('Usia')
        
        # School type (encode categorically)
        if 'school_type' in df.columns:
            le_school = LabelEncoder()
            df['school_type_encoded'] = le_school.fit_transform(df['school_type'].fillna('Unknown'))
            features_for_regression.append(df['school_type_encoded'])
            feature_names.append('Jenis Sekolah')
        
        # Create feature matrix
        if len(features_for_regression) > 0:
            X = np.column_stack(features_for_regression)
            y = df['accepted'].values
            
            # Remove rows with any NaN values
            valid_rows = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X_clean = X[valid_rows]
            y_clean = y[valid_rows]
            
            print(f"   Dataset untuk regresi: {len(X_clean)} observasi, {len(feature_names)} variabel")
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Analisis Regresi Logistik Multivariat KIP Kuliah', fontsize=16, fontweight='bold')
            
            # 1. Logistic Regression with sklearn
            if len(X_clean) > 50:  # Minimum sample size
                X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, 
                                                                    test_size=0.3, random_state=42)
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Fit logistic regression
                log_reg = LogisticRegression(random_state=42, max_iter=1000)
                log_reg.fit(X_train_scaled, y_train)
                
                # Get coefficients
                coefficients = log_reg.coef_[0]
                
                # Plot feature importance (coefficients)
                coef_abs = np.abs(coefficients)
                sorted_idx = np.argsort(coef_abs)[::-1]
                
                axes[0,0].barh(range(len(feature_names)), coef_abs[sorted_idx], alpha=0.7, color='steelblue')
                axes[0,0].set_title('Pentingnya Variabel (|Koefisien Logistic Regression|)')
                axes[0,0].set_xlabel('Magnitude Koefisien')
                axes[0,0].set_yticks(range(len(feature_names)))
                axes[0,0].set_yticklabels([feature_names[i] for i in sorted_idx])
                
                # 2. Coefficient direction plot
                colors = ['red' if coef < 0 else 'green' for coef in coefficients[sorted_idx]]
                axes[0,1].barh(range(len(feature_names)), coefficients[sorted_idx], 
                              alpha=0.7, color=colors)
                axes[0,1].set_title('Arah Pengaruh Variabel')
                axes[0,1].set_xlabel('Koefisien (Hijau=Positif, Merah=Negatif)')
                axes[0,1].set_yticks(range(len(feature_names)))
                axes[0,1].set_yticklabels([feature_names[i] for i in sorted_idx])
                axes[0,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
                
                # 3. Predicted probabilities vs actual
                y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
                
                # Create probability bins
                prob_bins = np.linspace(0, 1, 11)
                bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
                actual_rates = []
                
                for i in range(len(prob_bins)-1):
                    mask = (y_pred_proba >= prob_bins[i]) & (y_pred_proba < prob_bins[i+1])
                    if mask.sum() > 0:
                        actual_rates.append(y_test[mask].mean())
                    else:
                        actual_rates.append(np.nan)
                
                # Remove NaN values for plotting
                valid_bins = ~np.isnan(actual_rates)
                bin_centers_clean = bin_centers[valid_bins]
                actual_rates_clean = np.array(actual_rates)[valid_bins]
                
                axes[1,0].scatter(bin_centers_clean, actual_rates_clean, alpha=0.7, s=50)
                axes[1,0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Prediction')
                axes[1,0].set_title('Kalibrasi Model: Prediksi vs Aktual')
                axes[1,0].set_xlabel('Probabilitas Prediksi')
                axes[1,0].set_ylabel('Tingkat Penerimaan Aktual')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
                
                # 4. Statistical significance with statsmodels
                try:
                    # Add constant for statsmodels
                    X_with_const = sm.add_constant(X_clean)
                    
                    # Fit statsmodels logistic regression
                    logit_model = sm.Logit(y_clean, X_with_const)
                    result = logit_model.fit(disp=0)
                    
                    # Get p-values and odds ratios
                    p_values = result.pvalues[1:]  # Exclude constant
                    odds_ratios = np.exp(result.params[1:])  # Exclude constant
                    
                    # Significance levels
                    significance = []
                    for p in p_values:
                        if p < 0.001:
                            significance.append('***')
                        elif p < 0.01:
                            significance.append('**')
                        elif p < 0.05:
                            significance.append('*')
                        else:
                            significance.append('')
                    
                    # Plot odds ratios
                    sorted_or_idx = np.argsort(np.abs(odds_ratios - 1))[::-1]
                    
                    colors_or = ['red' if or_val < 1 else 'green' for or_val in odds_ratios[sorted_or_idx]]
                    bars = axes[1,1].barh(range(len(feature_names)), odds_ratios[sorted_or_idx], 
                                         alpha=0.7, color=colors_or)
                    
                    # Add significance indicators
                    for i, (bar, sig) in enumerate(zip(bars, [significance[idx] for idx in sorted_or_idx])):
                        width = bar.get_width()
                        axes[1,1].text(width + 0.05 if width > 1 else width - 0.05, bar.get_y() + bar.get_height()/2, 
                                      sig, ha='left' if width > 1 else 'right', va='center', fontweight='bold')
                    
                    axes[1,1].set_title('Odds Ratio dengan Signifikansi Statistik')
                    axes[1,1].set_xlabel('Odds Ratio (>1=Meningkatkan, <1=Menurunkan)')
                    axes[1,1].set_yticks(range(len(feature_names)))
                    axes[1,1].set_yticklabels([feature_names[i] for i in sorted_or_idx])
                    axes[1,1].axvline(x=1, color='black', linestyle='--', alpha=0.5)
                    axes[1,1].text(0.02, 0.98, '***p<0.001, **p<0.01, *p<0.05', 
                                  transform=axes[1,1].transAxes, fontsize=8, va='top')
                    
                    # Store results for report
                    self.regression_results = {
                        'feature_names': feature_names,
                        'coefficients': coefficients,
                        'p_values': p_values,
                        'odds_ratios': odds_ratios,
                        'significance': significance,
                        'model_summary': result.summary(),
                        'accuracy': log_reg.score(X_test_scaled, y_test)
                    }
                    
                    print(f"   ✅ Model accuracy: {log_reg.score(X_test_scaled, y_test):.3f}")
                    
                except Exception as e:
                    print(f"   ⚠️ Error in statsmodels analysis: {e}")
                    axes[1,1].text(0.5, 0.5, 'Error dalam analisis statistik', 
                                  ha='center', va='center', transform=axes[1,1].transAxes)
            
            else:
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'Data tidak mencukupi untuk analisis', 
                           ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            print("   ❌ Tidak ada fitur yang cukup untuk analisis regresi")
    
    def generate_regression_report(self):
        """Generate detailed regression analysis report"""
        if hasattr(self, 'regression_results'):
            print("\n" + "="*80)
            print("📊 LAPORAN ANALISIS REGRESI LOGISTIK MULTIVARIAT")
            print("="*80)
            
            results = self.regression_results
            
            print(f"\n🎯 PERFORMA MODEL:")
            print(f"   Akurasi Prediksi: {results['accuracy']:.1%}")
            print(f"   Jumlah Variabel: {len(results['feature_names'])}")
            
            print(f"\n📊 VARIABEL PALING SIGNIFIKAN:")
            # Sort by p-value
            sorted_indices = np.argsort(results['p_values'])
            
            for i, idx in enumerate(sorted_indices[:5]):  # Top 5 most significant
                feat_name = results['feature_names'][idx]
                coef = results['coefficients'][idx]
                p_val = results['p_values'][idx]
                odds_ratio = results['odds_ratios'][idx]
                sig = results['significance'][idx]
                
                direction = "meningkatkan" if coef > 0 else "menurunkan"
                
                print(f"   {i+1}. {feat_name}:")
                print(f"      - Koefisien: {coef:.3f} ({direction} peluang)")
                print(f"      - Odds Ratio: {odds_ratio:.3f}")
                print(f"      - P-value: {p_val:.4f} {sig}")
                print(f"      - Interpretasi: Setiap peningkatan 1 unit {feat_name.lower()}")
                
                if odds_ratio > 1:
                    print(f"        meningkatkan odds diterima sebesar {(odds_ratio-1)*100:.1f}%")
                else:
                    print(f"        menurunkan odds diterima sebesar {(1-odds_ratio)*100:.1f}%")
                print()
            
            print(f"\n🔍 INSIGHT KEBIJAKAN:")
            
            # Economic factors
            econ_vars = ['P3KE Desil', 'Penghasilan Ayah', 'Kualitas Hunian']
            econ_indices = [i for i, name in enumerate(results['feature_names']) if any(e in name for e in econ_vars)]
            
            if econ_indices:
                print("   💰 Faktor Ekonomi:")
                for idx in econ_indices:
                    if results['p_values'][idx] < 0.05:
                        direction = "positif" if results['coefficients'][idx] > 0 else "negatif"
                        print(f"   - {results['feature_names'][idx]}: pengaruh {direction} signifikan")
            
            # Geographic factors
            geo_vars = ['Perkotaan', 'Urban']
            geo_indices = [i for i, name in enumerate(results['feature_names']) if any(g in name for g in geo_vars)]
            
            if geo_indices:
                print("   🗺️ Faktor Geografis:")
                for idx in geo_indices:
                    if results['p_values'][idx] < 0.05:
                        direction = "positif" if results['coefficients'][idx] > 0 else "negatif"
                        print(f"   - {results['feature_names'][idx]}: pengaruh {direction} signifikan")
            
            # Demographic factors
            demo_vars = ['Perempuan', 'Gender', 'Usia']
            demo_indices = [i for i, name in enumerate(results['feature_names']) if any(d in name for d in demo_vars)]
            
            if demo_indices:
                print("   👥 Faktor Demografis:")
                for idx in demo_indices:
                    if results['p_values'][idx] < 0.05:
                        direction = "positif" if results['coefficients'][idx] > 0 else "negatif"
                        print(f"   - {results['feature_names'][idx]}: pengaruh {direction} signifikan")
            
            print("\n" + "="*80)
            
        else:
            print("❌ Analisis regresi belum dilakukan atau gagal")

    def _calculate_ci(self, successes, trials, confidence=0.95):
        """Calculate confidence intervals for proportions"""
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / trials
        se = np.sqrt(p * (1 - p) / trials)
        ci_lower = p - z * se
        ci_upper = p + z * se
        return np.maximum(ci_lower, 0), np.minimum(ci_upper, 1)
    
    def generate_inequality_report(self):
        """Generate comprehensive inequality analysis report"""
        print("\n" + "="*80)
        print("📊 LAPORAN ANALISIS KETIMPANGAN KIP KULIAH")
        print("="*80)
        
        df = self.combined_data.copy()
        
        # Overall statistics
        total_applicants = len(df)
        total_accepted = df['accepted'].sum()
        overall_rate = df['accepted'].mean()
        
        print(f"\n📈 STATISTIK KESELURUHAN:")
        print(f"   Total Pendaftar: {total_applicants:,}")
        print(f"   Total Diterima: {total_accepted:,}")
        print(f"   Tingkat Penerimaan Keseluruhan: {overall_rate:.1%}")
        
        # P3KE Analysis
        if 'p3ke_desil' in df.columns:
            desil_stats = df.groupby('p3ke_desil')['accepted'].agg(['count', 'mean'])
            poorest_rate = desil_stats.loc[desil_stats.index <= 3, 'mean'].mean()
            richest_rate = desil_stats.loc[desil_stats.index >= 8, 'mean'].mean()
            
            print(f"\n💰 KETIMPANGAN EKONOMI:")
            print(f"   Termiskin (Desil 1-3): {poorest_rate:.1%}")
            print(f"   Terkaya (Desil 8-10): {richest_rate:.1%}")
            print(f"   Rasio Ketimpangan: {richest_rate/poorest_rate:.2f}x")
        
        # Geographic Analysis
        if 'is_urban' in df.columns:
            urban_rate = df[df['is_urban'] == True]['accepted'].mean()
            rural_rate = df[df['is_urban'] == False]['accepted'].mean()
            
            print(f"\n🗺️ KETIMPANGAN GEOGRAFIS:")
            print(f"   Perkotaan: {urban_rate:.1%}")
            print(f"   Pedesaan: {rural_rate:.1%}")
            print(f"   Keuntungan Perkotaan: {(urban_rate - rural_rate)*100:.1f} poin persentase")
        
        # Gender Analysis
        if 'Jenis Kelamin' in df.columns:
            gender_stats = df.groupby('Jenis Kelamin')['accepted'].mean()
            
            print(f"\n👥 ANALISIS GENDER:")
            for gender, rate in gender_stats.items():
                gender_label = "Laki-laki" if gender == "L" else "Perempuan"
                print(f"   {gender_label}: {rate:.1%}")
        
        print("\n" + "="*80)
        print("📝 REKOMENDASI INTERVENSI KEBIJAKAN:")
        print("   1. Outreach terfokus untuk daerah pedesaan")
        print("   2. Dukungan khusus untuk keluarga desil rendah")
        print("   3. Intervensi spesifik berdasarkan jenis sekolah")
        print("   4. Monitoring temporal tren ketimpangan")
        print("="*80)

def main():
    """Main execution function"""
    # Initialize analysis
    data_path = "c:/laragon/www/ML_KIPKuliah/Bahan Laporan KIP Kuliah 2022 s.d 2024"
    analysis = KIPKuliahInequalityAnalysis(data_path)
    
    # Load and prepare data
    analysis.load_and_prepare_data()
    analysis.clean_and_engineer_features()
    
    # Generate all visualizations
    print("\n🎨 Membuat visualisasi siap publikasi...")
    
    # Create output directory
    output_dir = "c:/laragon/www/ML_KIPKuliah/visualizations/inequality_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Socio-Economic Disparity
    analysis.plot_socioeconomic_disparity(
        save_path=f"{output_dir}/gambar1_disparitas_sosioekonomi.png"
    )
    
    # Figure 2: Educational Disparity
    analysis.plot_educational_disparity(
        save_path=f"{output_dir}/gambar2_disparitas_pendidikan.png"
    )
    
    # Figure 3: Temporal Trends
    analysis.plot_temporal_trends(
        save_path=f"{output_dir}/gambar3_tren_temporal.png"
    )
    
    # Figure 4: Intersectional Analysis
    analysis.plot_intersectional_analysis(
        save_path=f"{output_dir}/gambar4_analisis_interseksional.png"
    )
    
    # Figure 5: Multivariate Regression Analysis
    print("\n🔍 Melakukan analisis regresi logistik multivariat...")
    analysis.perform_multivariate_analysis(
        save_path=f"{output_dir}/gambar5_analisis_multivariat.png"
    )
    
    # Generate comprehensive report
    analysis.generate_inequality_report()
    
    # Generate regression report if available
    analysis.generate_regression_report()
    
    # Generate detailed chart descriptions
    analysis.generate_chart_descriptions(output_dir)
    
    print(f"\n✅ Analisis selesai! Visualisasi disimpan di: {output_dir}")
    print("📄 Siap untuk publikasi akademik!")
    print("📝 Deskripsi lengkap tersedia di file: deskripsi_grafik_lengkap.txt")

if __name__ == "__main__":
    main()
