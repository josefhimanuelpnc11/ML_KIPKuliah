"""
Data Exploration Module for KIP Kuliah Classification
Author: Data Mining Professional
Date: 2025

This module provides comprehensive data exploration and analysis
for KIP Kuliah applicant and recipient datasets.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class KIPDataExplorer:
    """
    Comprehensive data explorer for KIP Kuliah datasets.
    
    This class handles data loading, cleaning, and preliminary analysis
    of both applicant and recipient datasets across multiple years.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize the data explorer.
        
        Args:
            base_path (str): Base path to the data directory
        """
        self.base_path = base_path
        self.pendaftar_files = []
        self.penerima_files = []
        self.raw_data = {}
        self.processed_data = None
        
    def discover_files(self) -> Dict[str, List[str]]:
        """
        Discover all CSV files in the dataset directory.
        
        Returns:
            Dict containing lists of discovered files
        """
        print("🔍 Discovering dataset files...")
        
        # Find all pendaftar files
        pendaftar_pattern = os.path.join(self.base_path, "CSV_Pendaftar", "**", "*.csv")
        self.pendaftar_files = glob.glob(pendaftar_pattern, recursive=True)
        
        # Find all penerima files  
        penerima_pattern = os.path.join(self.base_path, "CSV_Penerima", "*.csv")
        self.penerima_files = glob.glob(penerima_pattern, recursive=True)
        
        print(f"✅ Found {len(self.pendaftar_files)} pendaftar files")
        print(f"✅ Found {len(self.penerima_files)} penerima files")
        
        return {
            'pendaftar': self.pendaftar_files,
            'penerima': self.penerima_files
        }
    
    def load_single_file(self, filepath: str) -> pd.DataFrame:
        """
        Load and clean a single CSV file.
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    # Read without header first to inspect structure
                    df_inspect = pd.read_csv(filepath, encoding=encoding, header=None, nrows=3)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"❌ Could not decode {filepath} with any encoding")
                return pd.DataFrame()
            
            # Determine correct header row
            header_row = 0
            if len(df_inspect) > 1:
                # Check if first row is a title (contains "DATA SISWA" or "DATA MAHASISWA")
                first_row_str = str(df_inspect.iloc[0, 0]) if not pd.isna(df_inspect.iloc[0, 0]) else ""
                if 'DATA SISWA' in first_row_str or 'DATA MAHASISWA' in first_row_str:
                    header_row = 1  # Use second row as header
            
            # Read the file properly with correct header
            df = pd.read_csv(filepath, encoding=encoding, header=header_row, low_memory=False)
            
            # Skip empty dataframes
            if df.empty:
                return pd.DataFrame()
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Handle duplicate column names
            if df.columns.duplicated().any():
                print(f"⚠️ Found duplicate columns in {os.path.basename(filepath)}")
                # Create unique column names
                new_columns = []
                for col in df.columns:
                    counter = 1
                    new_col = col
                    while new_col in new_columns:
                        new_col = f"{col}_{counter}"
                        counter += 1
                    new_columns.append(new_col)
                df.columns = new_columns
            
            # Map unnamed columns to their actual content
            df = self._map_unnamed_columns(df)
            
            # Add metadata
            filename = os.path.basename(filepath)
            year = None
            if '2022' in filepath:
                year = 2022
            elif '2023' in filepath:
                year = 2023
            elif '2024' in filepath:
                year = 2024
                
            df['source_file'] = filename
            df['year'] = year
            df['is_recipient'] = 'penerima' in filepath.lower()
            
            print(f"✅ Loaded {filepath}: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {str(e)}")
            return pd.DataFrame()
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all discovered files into memory.
        
        Returns:
            Dict containing loaded dataframes
        """
        print("\n📂 Loading all dataset files...")
        
        # Load pendaftar files
        pendaftar_dfs = []
        for file in self.pendaftar_files:
            df = self.load_single_file(file)
            if not df.empty:
                pendaftar_dfs.append(df)
        
        # Load penerima files
        penerima_dfs = []
        for file in self.penerima_files:
            df = self.load_single_file(file)
            if not df.empty:
                penerima_dfs.append(df)
        
        # Combine dataframes
        self.raw_data = {
            'pendaftar_combined': pd.concat(pendaftar_dfs, ignore_index=True) if pendaftar_dfs else pd.DataFrame(),
            'penerima_combined': pd.concat(penerima_dfs, ignore_index=True) if penerima_dfs else pd.DataFrame(),
            'pendaftar_individual': pendaftar_dfs,
            'penerima_individual': penerima_dfs
        }
        
        print(f"✅ Total pendaftar records: {len(self.raw_data['pendaftar_combined'])}")
        print(f"✅ Total penerima records: {len(self.raw_data['penerima_combined'])}")
        
        return self.raw_data
    
    def analyze_columns(self) -> Dict[str, Any]:
        """
        Analyze column structure across all datasets.
        
        Returns:
            Dict containing column analysis results
        """
        print("\n🔍 Analyzing column structure...")
        
        if not self.raw_data:
            print("❌ No data loaded. Run load_all_data() first.")
            return {}
        
        pendaftar_df = self.raw_data['pendaftar_combined']
        penerima_df = self.raw_data['penerima_combined']
        
        analysis = {
            'pendaftar_columns': list(pendaftar_df.columns),
            'penerima_columns': list(penerima_df.columns),
            'common_columns': list(set(pendaftar_df.columns) & set(penerima_df.columns)),
            'pendaftar_unique': list(set(pendaftar_df.columns) - set(penerima_df.columns)),
            'penerima_unique': list(set(penerima_df.columns) - set(pendaftar_df.columns))
        }
        
        print(f"📊 Pendaftar columns: {len(analysis['pendaftar_columns'])}")
        print(f"📊 Penerima columns: {len(analysis['penerima_columns'])}")
        print(f"📊 Common columns: {len(analysis['common_columns'])}")
        print(f"📊 Pendaftar unique: {len(analysis['pendaftar_unique'])}")
        print(f"📊 Penerima unique: {len(analysis['penerima_unique'])}")
        
        return analysis
    
    def identify_socioeconomic_features(self) -> Dict[str, List[str]]:
        """
        Identify socio-economic features relevant for classification.
        
        Returns:
            Dict containing categorized features
        """
        print("\n💰 Identifying socio-economic features...")
        
        # Define feature categories based on domain knowledge
        socioeconomic_features = {
            'economic_status': [
                'Penghasilan Ayah', 'Penghasilan Ibu', 'Pekerjaan Ayah', 'Pekerjaan Ibu',
                'Status DTKS', 'Status P3KE', 'No. KIP', 'No. KKS'
            ],
            'family_structure': [
                'Jumlah Tanggungan', 'Status Ayah', 'Status Ibu'
            ],
            'housing_conditions': [
                'Kepemilikan Rumah', 'Tahun Perolehan', 'Luas Tanah', 'Luas Bangunan',
                'Sumber Listrik', 'Sumber Air', 'MCK'
            ],
            'geographic': [
                'Kab/Kota Sekolah', 'Provinsi Sekolah', 'Jarak Pusat Kota (KM)'
            ],
            'demographic': [
                'Jenis Kelamin', 'Tempat Lahir', 'Tanggal Lahir'
            ],
            'education': [
                'Asal Sekolah', 'Prestasi Siswa', 'Prestasi'
            ]
        }
        
        # Verify which features actually exist in the data
        if self.raw_data and 'pendaftar_combined' in self.raw_data:
            available_columns = set(self.raw_data['pendaftar_combined'].columns)
            
            verified_features = {}
            for category, features in socioeconomic_features.items():
                verified_features[category] = [f for f in features if f in available_columns]
                print(f"📋 {category}: {len(verified_features[category])}/{len(features)} features available")
        
        return verified_features if self.raw_data else socioeconomic_features
    
    def analyze_missing_values(self) -> Dict[str, pd.Series]:
        """
        Analyze missing values across all features.
        
        Returns:
            Dict containing missing value analysis
        """
        print("\n🕳️ Analyzing missing values...")
        
        if not self.raw_data:
            print("❌ No data loaded.")
            return {}
        
        analysis = {}
        
        for dataset_name, df in [('pendaftar', self.raw_data['pendaftar_combined']), 
                                ('penerima', self.raw_data['penerima_combined'])]:
            if not df.empty:
                missing_counts = df.isnull().sum()
                missing_percentages = (missing_counts / len(df)) * 100
                
                analysis[f'{dataset_name}_missing_counts'] = missing_counts
                analysis[f'{dataset_name}_missing_percentages'] = missing_percentages
                
                print(f"📊 {dataset_name.title()} dataset:")
                print(f"   - Total records: {len(df)}")
                print(f"   - Columns with missing values: {(missing_counts > 0).sum()}")
                print(f"   - Highest missing percentage: {missing_percentages.max():.1f}%")
        
        return analysis
    
    def create_target_variable(self) -> pd.DataFrame:
        """
        Create unified dataset with target variable (recipient status).
        
        Returns:
            pd.DataFrame: Combined dataset with target variable
        """
        print("\n🎯 Creating target variable...")
        
        if not self.raw_data:
            print("❌ No data loaded.")
            return pd.DataFrame()
        
        pendaftar_df = self.raw_data['pendaftar_combined'].copy()
        penerima_df = self.raw_data['penerima_combined'].copy()
        
        # Reset index to ensure clean concatenation
        pendaftar_df = pendaftar_df.reset_index(drop=True)
        penerima_df = penerima_df.reset_index(drop=True)
        
        # Add target variable
        pendaftar_df['is_kip_recipient'] = 0
        penerima_df['is_kip_recipient'] = 1
        
        # Get common columns for merging - ensure both dataframes have the same columns
        common_cols = list(set(pendaftar_df.columns) & set(penerima_df.columns))
        
        # Prepare dataframes with same column structure
        df_pendaftar_aligned = pendaftar_df[common_cols].copy()
        df_penerima_aligned = penerima_df[common_cols].copy()
        
        # Ensure no duplicate columns in the dataframes
        df_pendaftar_aligned.columns = pd.Index(df_pendaftar_aligned.columns).drop_duplicates()
        df_penerima_aligned.columns = pd.Index(df_penerima_aligned.columns).drop_duplicates()
        
        # Reset index again before concatenation
        df_pendaftar_aligned = df_pendaftar_aligned.reset_index(drop=True)
        df_penerima_aligned = df_penerima_aligned.reset_index(drop=True)
        
        try:
            # Combine datasets
            combined_df = pd.concat([
                df_pendaftar_aligned,
                df_penerima_aligned
            ], ignore_index=True, sort=False)
            
            # Remove duplicates based on registration number if available
            if 'No. Pendaftaran' in combined_df.columns:
                before_dedup = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=['No. Pendaftaran'], keep='last')
                print(f"🔍 Removed {before_dedup - len(combined_df)} duplicate registrations")
            
            print(f"✅ Created combined dataset: {len(combined_df)} records")
            print(f"   - Recipients: {combined_df['is_kip_recipient'].sum()}")
            print(f"   - Non-recipients: {(combined_df['is_kip_recipient'] == 0).sum()}")
            print(f"   - Recipient rate: {(combined_df['is_kip_recipient'].mean() * 100):.1f}%")
            
            self.processed_data = combined_df
            return combined_df
            
        except Exception as e:
            print(f"❌ Error in create_target_variable: {str(e)}")
            print(f"   Pendaftar shape: {df_pendaftar_aligned.shape}")
            print(f"   Penerima shape: {df_penerima_aligned.shape}")
            print(f"   Common columns: {len(common_cols)}")
            return pd.DataFrame()
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Returns:
            Dict containing complete analysis summary
        """
        print("\n📋 Generating comprehensive summary report...")
        
        summary = {
            'dataset_info': {
                'total_files': len(self.pendaftar_files) + len(self.penerima_files),
                'pendaftar_files': len(self.pendaftar_files),
                'penerima_files': len(self.penerima_files),
                'years_covered': [2022, 2023, 2024] if self.raw_data else []
            }
        }
        
        if self.raw_data:
            summary.update({
                'data_shape': {
                    'pendaftar_records': len(self.raw_data['pendaftar_combined']),
                    'penerima_records': len(self.raw_data['penerima_combined']),
                    'total_records': len(self.processed_data) if self.processed_data is not None else 0
                },
                'features': self.identify_socioeconomic_features(),
                'missing_values': self.analyze_missing_values(),
                'columns': self.analyze_columns()
            })
            
            if self.processed_data is not None:
                summary['target_distribution'] = {
                    'recipients': int(self.processed_data['is_kip_recipient'].sum()),
                    'non_recipients': int((self.processed_data['is_kip_recipient'] == 0).sum()),
                    'recipient_rate': float(self.processed_data['is_kip_recipient'].mean())
                }
        
        print("✅ Summary report generated successfully!")
        return summary
    
    def _map_unnamed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map unnamed columns to meaningful names based on content analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with mapped column names
        """
        # Create mapping based on typical CSV structure from samples
        column_mapping = {}
        
        # Analyze first few rows to identify content patterns
        for col in df.columns:
            if col.startswith('Unnamed:'):
                # Sample non-null values to identify content
                sample_values = df[col].dropna().astype(str).head(10).tolist()
                
                if sample_values:
                    # Try to identify column based on content patterns
                    mapped_name = self._identify_column_by_content(sample_values, col)
                    if mapped_name:
                        column_mapping[col] = mapped_name
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        print(f"📋 Mapped {len(column_mapping)} unnamed columns to meaningful names")
        return df
    
    def _identify_column_by_content(self, sample_values: list, original_col: str) -> str:
        """
        Identify column name based on content patterns.
        
        Args:
            sample_values (list): Sample values from the column
            original_col (str): Original column name
            
        Returns:
            str: Identified column name or None
        """
        sample_str = ' '.join(sample_values).lower()
        
        # Define patterns for identification
        patterns = {
            'No. Pendaftaran': ['1122', '1123', '1124'],
            'Nama Siswa': [r'[A-Z][a-z]+ [A-Z]'],
            'NIK': [r'\d{16}'],
            'NISN': [r'\d{10}'],
            'Status DTKS': ['terdata', 'belum terdata'],
            'Status P3KE': ['desil', 'terdata'],
            'Jenis Kelamin': ['P', 'L'],
            'Penghasilan Ayah': ['rp.', 'tidak berpenghasilan'],
            'Penghasilan Ibu': ['rp.', 'tidak berpenghasilan'],
            'Pekerjaan Ayah': ['peg.', 'wirausaha', 'petani', 'tidak bekerja'],
            'Pekerjaan Ibu': ['peg.', 'wirausaha', 'petani', 'tidak bekerja'],
            'Status Ayah': ['hidup', 'wafat'],
            'Status Ibu': ['hidup', 'wafat'],
            'Jumlah Tanggungan': ['orang'],
            'Kepemilikan Rumah': ['sendiri', 'menumpang', 'sewa'],
            'Sumber Listrik': ['pln', 'non pln'],
            'Sumber Air': ['pdam', 'sumur'],
            'Luas Tanah': ['m2'],
            'Luas Bangunan': ['m2'],
            'MCK': ['kepemilikan', 'bersama'],
            'Provinsi Sekolah': ['jawa', 'sumatera', 'jakarta'],
            'Kab/Kota Sekolah': ['cilacap', 'banyumas', 'jakarta'],
            'Jarak Pusat Kota (KM)': [r'\d+$'],
        }
        
        # Check each pattern
        for col_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern.lower() in sample_str:
                    return col_name
        
        return None

def main():
    """
    Main function to run data exploration.
    """
    print("🚀 Starting KIP Kuliah Data Exploration...")
    
    # Initialize explorer
    base_path = r"c:\laragon\www\ML_KIPKuliah\Bahan Laporan KIP Kuliah 2022 s.d 2024"
    explorer = KIPDataExplorer(base_path)
    
    # Run exploration pipeline
    files = explorer.discover_files()
    data = explorer.load_all_data()
    
    if data:
        columns = explorer.analyze_columns()
        features = explorer.identify_socioeconomic_features()
        missing = explorer.analyze_missing_values()
        combined = explorer.create_target_variable()
        summary = explorer.generate_summary_report()
        
        print("\n" + "="*50)
        print("📊 EXPLORATION COMPLETE!")
        print("="*50)
        
        return explorer, summary
    else:
        print("❌ No data could be loaded. Please check file paths.")
        return None, None

if __name__ == "__main__":
    explorer, summary = main()
