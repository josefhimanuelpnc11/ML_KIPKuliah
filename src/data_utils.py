"""
Data Utilities for KIP Kuliah Analysis
Provides helper functions for data loading, preprocessing, and validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class KIPDataLoader:
    """
    Utility class for loading and combining KIP Kuliah data
    """
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        
    def load_all_data(self):
        """
        Load and combine all KIP Kuliah data from different years
        """
        pendaftar_files = {
            2022: [
                'CSV_Pendaftar/2022/Siswa_Pendaftar_SBMPN_2022.csv',
                'CSV_Pendaftar/2022/Siswa_Pendaftar_Seleksi Mandiri PTN_2022.csv',
                'CSV_Pendaftar/2022/Siswa_Pendaftar_SNMPN_Politeknik Negeri Cilacap_20220328.csv'
            ],
            2023: [
                'CSV_Pendaftar/2023/pendaftar kip jalur SNBT 2023.csv',
                'CSV_Pendaftar/2023/pendaftar KIP Kuliah 2023 jalur SNBP.csv',
                'CSV_Pendaftar/2023/Siswa_Pendaftar_Seleksi Mandiri PTN_2023.csv'
            ],
            2024: [
                'CSV_Pendaftar/2024/pendaftar kip jalur snbp dan snbt 2024.csv'
            ]
        }
        
        penerima_files = {
            2022: 'CSV_Penerima/penerima KIP Kuliah angkatan 2022.csv',
            2023: 'CSV_Penerima/penerima KIP Kuliah angkatan 2023.csv',
            2024: 'CSV_Penerima/penerima KIP Kuliah angkatan 2024.csv'
        }
        
        all_pendaftar = []
        all_penerima = []
        
        # Load pendaftar data
        for year, files in pendaftar_files.items():
            for file in files:
                try:
                    df = pd.read_csv(self.base_path / file)
                    df['tahun'] = year
                    df['jalur'] = file.split('/')[-1].replace('.csv', '')
                    all_pendaftar.append(df)
                    print(f"Loaded {file}: {len(df)} records")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        # Load penerima data
        for year, file in penerima_files.items():
            try:
                df = pd.read_csv(self.base_path / file)
                df['tahun'] = year
                df['status'] = 'diterima'
                all_penerima.append(df)
                print(f"Loaded {file}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        pendaftar_df = pd.concat(all_pendaftar, ignore_index=True) if all_pendaftar else pd.DataFrame()
        penerima_df = pd.concat(all_penerima, ignore_index=True) if all_penerima else pd.DataFrame()
        
        return pendaftar_df, penerima_df


class DataPreprocessor:
    """
    Utility class for data preprocessing
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def identify_column_types(self, df):
        """
        Identify numerical and categorical columns
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove ID-like columns
        exclude_cols = [col for col in categorical_cols if any(keyword in col.lower() 
                       for keyword in ['id', 'nama', 'name', 'no', 'nik', 'nisn'])]
        
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        return numerical_cols, categorical_cols, exclude_cols
    
    def handle_missing_values(self, df, numerical_cols, categorical_cols):
        """
        Handle missing values in dataset
        """
        df_clean = df.copy()
        
        # Numerical: fill with median
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Categorical: fill with mode or 'Unknown'
        for col in categorical_cols:
            if col in df_clean.columns:
                mode_val = df_clean[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
        
        return df_clean
    
    def prepare_for_clustering(self, df, numerical_cols, categorical_cols):
        """
        Prepare data for K-Prototypes clustering
        """
        # Select relevant columns
        available_num_cols = [col for col in numerical_cols if col in df.columns]
        available_cat_cols = [col for col in categorical_cols if col in df.columns]
        
        all_cols = available_num_cols + available_cat_cols
        df_final = df[all_cols].copy()
        
        # Scale numerical features
        if available_num_cols:
            scaler = StandardScaler()
            df_final[available_num_cols] = scaler.fit_transform(df_final[available_num_cols])
            self.scalers['clustering'] = scaler
        
        # Get categorical indices
        categorical_indices = list(range(len(available_num_cols), len(all_cols)))
        
        return df_final, categorical_indices, available_num_cols, available_cat_cols
    
    def prepare_for_classification(self, df, numerical_cols, categorical_cols, target_col='cluster'):
        """
        Prepare data for classification models
        """
        feature_cols = [col for col in numerical_cols + categorical_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
        
        # Scale features
        if numerical_cols:
            available_num_cols = [col for col in numerical_cols if col in X.columns]
            scaler = StandardScaler()
            X[available_num_cols] = scaler.fit_transform(X[available_num_cols])
            self.scalers['classification'] = scaler
        
        return X, y


class DataValidator:
    """
    Utility class for data validation and quality checks
    """
    
    @staticmethod
    def check_data_quality(df):
        """
        Perform comprehensive data quality checks
        """
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict()
        }
        
        # Missing value details
        missing_detail = df.isnull().sum()
        report['missing_by_column'] = missing_detail[missing_detail > 0].to_dict()
        
        # Memory usage
        report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
        
        return report
    
    @staticmethod
    def validate_preprocessing(original_df, processed_df):
        """
        Validate preprocessing results
        """
        validation = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'original_missing': original_df.isnull().sum().sum(),
            'processed_missing': processed_df.isnull().sum().sum(),
            'data_preserved': len(original_df) == len(processed_df)
        }
        
        return validation


def create_analysis_summary(df, cluster_col='cluster'):
    """
    Create a comprehensive analysis summary
    """
    summary = {
        'total_records': len(df),
        'n_clusters': df[cluster_col].nunique(),
        'cluster_distribution': df[cluster_col].value_counts().to_dict(),
        'cluster_percentages': (df[cluster_col].value_counts() / len(df) * 100).to_dict()
    }
    
    return summary
