"""
Data Preprocessing Module for KIP Kuliah Analysis
Handles data cleaning, feature engineering, and preparation for ML algorithms
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing tasks
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_info = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data for analysis
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        
        df_clean = df.copy()
        
        # Remove completely empty columns
        empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
        if empty_cols:
            df_clean = df_clean.drop(columns=empty_cols)
            logger.info(f"Removed empty columns: {empty_cols}")
        
        # Identify column types
        numerical_cols, categorical_cols, exclude_cols = self._identify_column_types(df_clean)
        
        # Store feature information
        self.feature_info = {
            'numerical': numerical_cols,
            'categorical': categorical_cols,
            'excluded': exclude_cols
        }
        
        logger.info(f"Column classification:")
        logger.info(f"  Numerical: {len(numerical_cols)} columns")
        logger.info(f"  Categorical: {len(categorical_cols)} columns") 
        logger.info(f"  Excluded: {len(exclude_cols)} columns")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean, numerical_cols, categorical_cols)
        
        # Remove duplicate rows
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_dups = initial_len - len(df_clean)
        if removed_dups > 0:
            logger.info(f"Removed {removed_dups} duplicate rows")
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        
        return df_clean
    
    def _identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """Identify numerical, categorical, and columns to exclude"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove system columns, IDs, and potential leakage features
        exclude_keywords = [
            'id', 'nama', 'name', 'no', 'nik', 'nisn', 'source_file', '#',
            'email', 'handphone', 'hp', 'telp', 'telepon', 'phone',
            'tanggal', 'tgl', 'date', 'waktu', 'jam', 'time',
            'dicalonkan', 'diterima', 'lulus', 'accepted', 'rejected'
        ]
        exclude_cols = []
        
        for col in categorical_cols:
            if any(keyword in col.lower() for keyword in exclude_keywords):
                exclude_cols.append(col)
        
        # Also check numerical columns for leakage
        for col in numerical_cols:
            if any(keyword in col.lower() for keyword in exclude_keywords):
                exclude_cols.append(col)
        
        # Also check exact column names that should be excluded
        exact_exclude_cols = ['#', 'No', 'no', 'Unnamed: 0', 'index', 'NISN', 'NIK']
        for col in df.columns:
            if col in exact_exclude_cols or col.startswith('Unnamed'):
                exclude_cols.append(col)
        
        # Remove excluded columns from both lists
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        return numerical_cols, categorical_cols, exclude_cols
    
    def _handle_missing_values(self, df: pd.DataFrame, numerical_cols: List[str], 
                              categorical_cols: List[str]) -> pd.DataFrame:
        """Handle missing values in dataset"""
        
        df_clean = df.copy()
        
        # Handle numerical missing values with median
        for col in numerical_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.debug(f"Filled {col} missing values with median: {median_val}")
        
        # Handle categorical missing values with mode or 'Unknown'
        for col in categorical_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_vals = df_clean[col].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
                logger.debug(f"Filled {col} missing values with: {fill_val}")
        
        # Final check for any remaining missing values
        remaining_missing = df_clean.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Still have {remaining_missing} missing values after cleaning")
        
        return df_clean
    
    def prepare_for_clustering(self, df: pd.DataFrame, exclude_cohort_features: bool = False) -> Tuple[np.ndarray, List[int]]:
        """
        Prepare data for K-Prototypes clustering
        
        Args:
            df: Cleaned dataframe
            exclude_cohort_features: If True, exclude year/path features for socio-economic focus
            
        Returns:
            Tuple of (data_array, categorical_indices)
        """
        logger.info("Preparing data for clustering...")
        
        numerical_cols = self.feature_info['numerical'].copy()
        categorical_cols = self.feature_info['categorical'].copy()
        
        # Remove cohort features if requested
        if exclude_cohort_features:
            cohort_features = ['tahun', 'jalur']
            logger.info(f"Excluding cohort features for socio-economic analysis: {cohort_features}")
            numerical_cols = [col for col in numerical_cols if col not in cohort_features]
            categorical_cols = [col for col in categorical_cols if col not in cohort_features]
            
            # Store this info for later use
            self.socioeconomic_features = {
                'numerical': numerical_cols,
                'categorical': categorical_cols
            }
        
        # Select only relevant columns
        available_num_cols = [col for col in numerical_cols if col in df.columns]
        available_cat_cols = [col for col in categorical_cols if col in df.columns]
        
        all_cols = available_num_cols + available_cat_cols
        df_cluster = df[all_cols].copy()
        
        # Scale numerical features
        if available_num_cols:
            scaler = StandardScaler()
            df_cluster[available_num_cols] = scaler.fit_transform(df_cluster[available_num_cols])
            scaler_key = 'clustering_socioeconomic' if exclude_cohort_features else 'clustering'
            self.scalers[scaler_key] = scaler
            logger.info(f"Scaled {len(available_num_cols)} numerical features")
        
        # Get categorical indices for K-Prototypes
        categorical_indices = list(range(len(available_num_cols), len(all_cols)))
        
        analysis_type = "socio-economic" if exclude_cohort_features else "full"
        logger.info(f"Clustering data prepared ({analysis_type}):")
        logger.info(f"  Shape: {df_cluster.shape}")
        logger.info(f"  Numerical features: {len(available_num_cols)}")
        logger.info(f"  Categorical features: {len(available_cat_cols)}")
        logger.info(f"  Categorical indices: {categorical_indices}")
        
        return df_cluster.values, categorical_indices
    
    def prepare_for_classification(self, df: pd.DataFrame, 
                                  target_col: str = 'cluster') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for classification models
        
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Preparing data for classification...")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        numerical_cols = self.feature_info['numerical']
        categorical_cols = self.feature_info['categorical']
        
        # Select feature columns
        feature_cols = [col for col in numerical_cols + categorical_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                logger.debug(f"Encoded categorical column: {col}")
        
        # Scale all features for classification
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        self.scalers['classification'] = scaler
        
        logger.info(f"Classification data prepared:")
        logger.info(f"  Features shape: {X_scaled.shape}")
        logger.info(f"  Target shape: {y.shape}")
        logger.info(f"  Target distribution: {y.value_counts().to_dict()}")
        
        return X_scaled, y
    
    def get_feature_info(self) -> Dict:
        """Get information about features"""
        return self.feature_info.copy()
    
    def validate_preprocessing(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict:
        """Validate preprocessing results"""
        
        validation = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'original_missing': original_df.isnull().sum().sum(),
            'processed_missing': processed_df.isnull().sum().sum(),
            'rows_preserved': len(original_df) == len(processed_df),
            'data_types_consistent': True
        }
        
        # Check for any unexpected data type changes
        common_cols = set(original_df.columns) & set(processed_df.columns)
        for col in common_cols:
            if original_df[col].dtype != processed_df[col].dtype:
                logger.info(f"Data type changed for {col}: {original_df[col].dtype} -> {processed_df[col].dtype}")
        
        logger.info("Preprocessing validation completed")
        return validation
