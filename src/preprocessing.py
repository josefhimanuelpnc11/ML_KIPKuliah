"""
Data Preprocessing Pipeline for KIP Kuliah Classification
Author: Data Mining Professional
Date: 2025

This module provides comprehensive data preprocessing including:
- Feature engineering for socio-economic variables
- Encoding categorical variables
- Handling missing values
- Data normalization and standardization
- Feature selection and validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import warnings
warnings.filterwarnings('ignore')

class KIPDataPreprocessor:
    """
    Comprehensive data preprocessor for KIP Kuliah classification.
    
    Handles all preprocessing steps including feature engineering,
    encoding, imputation, and scaling with domain-specific logic.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        self.numeric_features = []
        self.categorical_features = []
        self.engineered_features = []
        self.feature_importance = {}
        
    def identify_feature_types(self, df: pd.DataFrame) -> dict:
        """
        Identify and categorize features by type.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Categorized features
        """
        print("🔍 Identifying feature types...")
        
        # Define socio-economic features based on domain knowledge
        socioeconomic_mapping = {
            'income_features': ['Penghasilan Ayah', 'Penghasilan Ibu'],
            'occupation_features': ['Pekerjaan Ayah', 'Pekerjaan Ibu'],
            'assistance_features': ['Status DTKS', 'Status P3KE', 'No. KIP', 'No. KKS'],
            'family_features': ['Jumlah Tanggungan', 'Status Ayah', 'Status Ibu'],
            'housing_features': ['Kepemilikan Rumah', 'Tahun Perolehan', 'Luas Tanah', 
                               'Luas Bangunan', 'Sumber Listrik', 'Sumber Air', 'MCK'],
            'geographic_features': ['Kab/Kota Sekolah', 'Provinsi Sekolah', 'Jarak Pusat Kota (KM)'],
            'demographic_features': ['Jenis Kelamin', 'Tempat Lahir', 'Tanggal Lahir']
        }
        
        # Identify available features
        available_features = {
            category: [f for f in features if f in df.columns]
            for category, features in socioeconomic_mapping.items()
        }
        
        # Categorize by data type
        numeric_candidates = []
        categorical_candidates = []
        
        # Define columns that should definitely be categorical
        forced_categorical = [
            'Nama Siswa', 'Asal Sekolah', 'Kab/Kota Sekolah', 'Provinsi Sekolah', 
            'Tempat Lahir', 'Jenis Kelamin', 'Alamat Tinggal', 'Alamat Email',
            'Nama Ayah', 'Pekerjaan Ayah', 'Penghasilan Ayah', 'Status Ayah',
            'Nama Ibu', 'Pekerjaan Ibu', 'Penghasilan Ibu', 'Status Ibu',
            'Kepemilikan Rumah', 'Sumber Listrik', 'Luas Tanah', 'Luas Bangunan',
            'Sumber Air', 'MCK', 'Status DTKS', 'Status P3KE', 'No. KIP', 'No. KKS',
            'Pilihan PT', 'Pilihan Prodi 1', 'Pilihan Prodi 2', 'Prestasi Siswa', 'Prestasi',
            'Tanggal Lahir', 'Tanggal Daftar', 'Tanggal Ditetapkan', 'Tanggal Dicalonkan',
            'User Penetapan', 'User Pencalonan', 'Seleksi Penetapan', 'Perguruan Tinggi',
            'Program Studi', 'Akreditasi Prodi', 'Skema Bantuan Pembiayaan'
        ]
        
        # Define columns that should be numeric
        forced_numeric = [
            'Jumlah Tanggungan', 'Tahun Perolehan', 'Jarak Pusat Kota (KM)', 'year'
        ]
        
        for col in df.columns:
            # Skip ID fields, metadata, and non-predictive columns
            skip_columns = [
                'is_kip_recipient', 'source_file', 'year', '#', 
                'No. Pendaftaran', 'NIK', 'No. Kartu Keluarga', 'NIK Kepala Keluarga', 'NISN',
                'No. Handphone', 'KAP',  # Additional ID-like fields
                'Tanggal Daftar', 'Tanggal Ditetapkan', 'Tanggal Dicalonkan',  # Dates
                'User Penetapan', 'User Pencalonan',  # User fields
                'Seleksi Penetapan', 'SNMPN', 'SBMPN', 'Seleksi Mandiri PTN', 'SNBP', 'UTBK-SNBT', 'Seleksi Mandiri PTS'  # Selection types
            ]
            
            if col in skip_columns:
                continue
                
            # Force categorization based on domain knowledge
            if col in forced_categorical:
                categorical_candidates.append(col)
            elif col in forced_numeric:
                numeric_candidates.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                numeric_candidates.append(col)
            else:
                # For remaining columns, default to categorical to be safe
                categorical_candidates.append(col)
        
        self.numeric_features = numeric_candidates
        self.categorical_features = categorical_candidates
        
        print(f"✅ Identified {len(numeric_candidates)} numeric features")
        print(f"✅ Identified {len(categorical_candidates)} categorical features")
        
        return {
            'numeric': numeric_candidates,
            'categorical': categorical_candidates,
            'socioeconomic_mapping': available_features
        }
    
    def _is_numeric_string(self, series: pd.Series) -> bool:
        """
        Check if string series contains numeric values.
        
        Args:
            series (pd.Series): Series to check
            
        Returns:
            bool: True if numeric
        """
        try:
            # Skip columns with clear text indicators
            text_indicators = ['email', 'nama', 'alamat', 'sekolah', 'kota', 'provinsi', 'tempat', 'pendaftaran', 'penetapan', 'prodi', 'universitas', 'pt', 'perguruan']
            col_name_lower = series.name.lower() if series.name else ""
            
            if any(indicator in col_name_lower for indicator in text_indicators):
                return False
            
            # Try to convert sample to numeric - check for percentage that are actually numeric
            sample = series.dropna().head(50)
            if len(sample) == 0:
                return False
            
            # Try direct numeric conversion
            numeric_count = 0
            for val in sample:
                try:
                    float(str(val).replace(',', '').replace('Rp.', '').strip())
                    numeric_count += 1
                except:
                    pass
            
            # Consider numeric if more than 70% are convertible to numbers
            return (numeric_count / len(sample)) > 0.7
        except:
            return False
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer domain-specific features for socio-economic classification.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        print("⚙️ Engineering socio-economic features...")
        
        df_engineered = df.copy()
        
        # Reset engineered features list
        self.engineered_features = []
        
        # 1. Income-based features
        df_engineered = self._engineer_income_features(df_engineered)
        
        # 2. Family structure features
        df_engineered = self._engineer_family_features(df_engineered)
        
        # 3. Housing condition features
        df_engineered = self._engineer_housing_features(df_engineered)
        
        # 4. Geographic features
        df_engineered = self._engineer_geographic_features(df_engineered)
        
        # 5. Assistance program features
        df_engineered = self._engineer_assistance_features(df_engineered)
        
        # 6. Demographic features
        df_engineered = self._engineer_demographic_features(df_engineered)
        
        print(f"✅ Created {len(self.engineered_features)} engineered features")
        
        return df_engineered
    
    def _engineer_income_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer income-related features."""
        
        # More comprehensive income mapping for various formats found in data
        income_mapping = {
            'Tidak Berpenghasilan': 0,
            'tidak berpenghasilan': 0,
            '-': 0,
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
            '> Rp. 3.000.000': 3500000
        }
        
        # Initialize with default values
        df['penghasilan_ayah_numeric'] = 0
        df['penghasilan_ibu_numeric'] = 0
        df['total_family_income'] = 0
        df['income_category_encoded'] = 'Very Low'
        df['jumlah_tanggungan_numeric'] = 1
        df['income_per_capita'] = 0
        df['income_adequacy_ratio'] = 0
        df['working_parents_count'] = 0
        df['dependency_ratio'] = 1
        
        # Apply income mapping with proper preprocessing
        for parent in ['Ayah', 'Ibu']:
            income_col = f'Penghasilan {parent}'
            if income_col in df.columns:
                # Clean and standardize income data
                cleaned_income = df[income_col].fillna('Tidak Berpenghasilan').astype(str).str.strip()
                df[f'penghasilan_{parent.lower()}_numeric'] = cleaned_income.map(income_mapping).fillna(0)
        
        # Calculate total family income
        df['total_family_income'] = df['penghasilan_ayah_numeric'] + df['penghasilan_ibu_numeric']
        
        # Income categories for classification
        df['income_category'] = pd.cut(df['total_family_income'], 
                                     bins=[0, 500000, 1000000, 2000000, float('inf')],
                                     labels=['Very Low', 'Low', 'Medium', 'High'])
        df['income_category_encoded'] = df['income_category'].astype(str)
        
        # Enhanced family size and income per capita calculations
        if 'Jumlah Tanggungan' in df.columns:
            # Robust extraction of family size
            def extract_tanggungan_number(x):
                if pd.isna(x):
                    return 1
                if isinstance(x, (int, float)):
                    return max(1, int(x))
                # Extract number from various formats: "4 Orang", "3", "5 anak", etc.
                import re
                match = re.search(r'(\d+)', str(x))
                return int(match.group(1)) if match else 1
            
            df['jumlah_tanggungan_numeric'] = df['Jumlah Tanggungan'].apply(extract_tanggungan_number)
        
        # Income per capita (total income / total family members including parents)
        df['income_per_capita'] = df['total_family_income'] / (df['jumlah_tanggungan_numeric'] + 2)  # +2 for parents
        
        # Income adequacy ratio (relative to Indonesian poverty line ~500k per capita)
        df['income_adequacy_ratio'] = df['income_per_capita'] / 500000
        
        # Dependency ratio: dependents per working parent
        working_parents = (df['penghasilan_ayah_numeric'] > 0).astype(int) + (df['penghasilan_ibu_numeric'] > 0).astype(int)
        df['working_parents_count'] = working_parents
        df['dependency_ratio'] = df['jumlah_tanggungan_numeric'] / (working_parents + 0.1)  # +0.1 to avoid division by zero
        
        # Add all features to engineered features list
        income_features = [
            'penghasilan_ayah_numeric', 'penghasilan_ibu_numeric', 'total_family_income',
            'income_category_encoded', 'jumlah_tanggungan_numeric', 'income_per_capita',
            'income_adequacy_ratio', 'working_parents_count', 'dependency_ratio'
        ]
        self.engineered_features.extend(income_features)
        
        return df
    
    def _engineer_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer family structure features."""
        
        # Parents status
        if 'Status Ayah' in df.columns:
            df['ayah_hidup'] = (df['Status Ayah'] == 'Hidup').astype(int)
            self.engineered_features.append('ayah_hidup')
        
        if 'Status Ibu' in df.columns:
            df['ibu_hidup'] = (df['Status Ibu'] == 'Hidup').astype(int)
            self.engineered_features.append('ibu_hidup')
        
        # Both parents alive
        if 'ayah_hidup' in df.columns and 'ibu_hidup' in df.columns:
            df['both_parents_alive'] = df['ayah_hidup'] * df['ibu_hidup']
            self.engineered_features.append('both_parents_alive')
        
        # Working parents count
        working_jobs = ['Peg. Negeri Sipil', 'Peg. Swasta', 'Wirausaha', 'Petani']
        working_count = 0
        
        for parent in ['Ayah', 'Ibu']:
            job_col = f'Pekerjaan {parent}'
            if job_col in df.columns:
                df[f'{parent.lower()}_working'] = df[job_col].isin(working_jobs).astype(int)
                working_count += df[f'{parent.lower()}_working']
                self.engineered_features.append(f'{parent.lower()}_working')
        
        if working_count is not 0:
            df['working_parents_count'] = working_count
            self.engineered_features.append('working_parents_count')
        
        return df
    
    def _engineer_housing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer housing condition features."""
        
        # Enhanced housing ownership scoring
        ownership_mapping = {
            'Sendiri': 3,
            'sendiri': 3,
            'Orang Tua': 2,
            'orang tua': 2,
            'Menumpang': 1,
            'menumpang': 1,
            'Sewa': 1,
            'sewa': 1,
            'Tidak Memiliki': 0,
            'tidak memiliki': 0,
            '-': 0
        }
        
        if 'Kepemilikan Rumah' in df.columns:
            cleaned_ownership = df['Kepemilikan Rumah'].fillna('Tidak Memiliki').astype(str).str.strip()
            df['housing_ownership_score'] = cleaned_ownership.map(ownership_mapping).fillna(0)
            self.engineered_features.append('housing_ownership_score')
        
        # Enhanced utility access scoring
        utility_mappings = {
            'Sumber Listrik': {
                'PLN': 2, 'pln': 2, 'Non PLN': 1, 'non pln': 1, 
                'Generator': 1, 'generator': 1, 'Tidak Ada': 0, 'tidak ada': 0, '-': 0
            },
            'Sumber Air': {
                'PDAM': 2, 'pdam': 2, 'Sumur': 1, 'sumur': 1, 
                'Sumur Bor': 1, 'sumur bor': 1, 'Sungai': 0, 'sungai': 0,
                'Mata Air': 0, 'mata air': 0, 'Tidak Ada': 0, 'tidak ada': 0, '-': 0
            },
            'MCK': {
                'Kepemilikan Sendiri': 2, 'kepemilikan sendiri': 2, 'Sendiri': 2, 'sendiri': 2,
                'Bersama': 1, 'bersama': 1, 'Umum': 1, 'umum': 1,
                'Tidak Ada': 0, 'tidak ada': 0, '-': 0
            }
        }
        
        utility_scores = []
        for util_col, mapping in utility_mappings.items():
            if util_col in df.columns:
                score_col = f'{util_col.lower().replace(" ", "_").replace("/", "_")}_score'
                cleaned_util = df[util_col].fillna('Tidak Ada').astype(str).str.strip()
                df[score_col] = cleaned_util.map(mapping).fillna(0)
                utility_scores.append(score_col)
                self.engineered_features.append(score_col)
        
        # Composite utility score
        if len(utility_scores) >= 2:
            df['total_utility_score'] = df[utility_scores].sum(axis=1)
            df['avg_utility_score'] = df[utility_scores].mean(axis=1)
            self.engineered_features.extend(['total_utility_score', 'avg_utility_score'])
        
        # Enhanced housing space analysis
        area_mappings = {
            'Luas Tanah': {
                '< 25 M2': 12.5, '<25 M2': 12.5, '25-50 M2': 37.5, '25 - 50 M2': 37.5,
                '50-99 M2': 74.5, '50 - 99 M2': 74.5, '100 - 200 M2': 150, '100-200 M2': 150,
                '>200 M2': 250, '> 200 M2': 250, '200 M2 <': 250
            },
            'Luas Bangunan': {
                '< 25 M2': 12.5, '<25 M2': 12.5, '25-50 M2': 37.5, '25 - 50 M2': 37.5,
                '50-99 M2': 74.5, '50 - 99 M2': 74.5, '100 - 200 M2': 150, '100-200 M2': 150,
                '>200 M2': 250, '> 200 M2': 250, '200 M2 <': 250
            }
        }
        
        for area_col, mapping in area_mappings.items():
            if area_col in df.columns:
                numeric_col = f'{area_col.lower().replace(" ", "_")}_numeric'
                cleaned_area = df[area_col].fillna('< 25 M2').astype(str).str.strip()
                df[numeric_col] = cleaned_area.map(mapping).fillna(12.5)  # Default to smallest category
                self.engineered_features.append(numeric_col)
        
        # Housing adequacy indicators
        if 'luas_bangunan_numeric' in df.columns and 'jumlah_tanggungan_numeric' in df.columns:
            # Living space per person (building area / family members)
            total_family = df['jumlah_tanggungan_numeric'] + 2  # +2 for parents
            df['living_space_per_person'] = df['luas_bangunan_numeric'] / total_family
            self.engineered_features.append('living_space_per_person')
            
            # Overcrowding indicator (less than 10 m2 per person is considered overcrowded)
            df['is_overcrowded'] = (df['living_space_per_person'] < 10).astype(int)
            self.engineered_features.append('is_overcrowded')
        
        # Overall housing quality score
        housing_score_cols = [col for col in df.columns if col.endswith('_score') and any(x in col for x in ['housing', 'utility', 'sumber', 'mck'])]
        if len(housing_score_cols) >= 2:
            df['housing_quality_score'] = df[housing_score_cols].sum(axis=1)
            self.engineered_features.append('housing_quality_score')
        
        return df
    
    def _engineer_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer geographic features."""
        
        # Urban/rural classification based on distance
        if 'Jarak Pusat Kota (KM)' in df.columns:
            df['jarak_numeric'] = pd.to_numeric(df['Jarak Pusat Kota (KM)'], errors='coerce').fillna(0)
            df['is_urban'] = (df['jarak_numeric'] <= 10).astype(int)
            df['is_rural'] = (df['jarak_numeric'] > 50).astype(int)
            self.engineered_features.extend(['jarak_numeric', 'is_urban', 'is_rural'])
        
        # Province development level (simplified categorization)
        developed_provinces = ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Banten']
        if 'Provinsi Sekolah' in df.columns:
            df['from_developed_province'] = df['Provinsi Sekolah'].isin(developed_provinces).astype(int)
            self.engineered_features.append('from_developed_province')
        
        return df
    
    def _engineer_assistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer social assistance features."""
        
        # DTKS status
        if 'Status DTKS' in df.columns:
            df['has_dtks'] = (df['Status DTKS'] == 'Terdata').astype(int)
            self.engineered_features.append('has_dtks')
        
        # P3KE status and desil
        if 'Status P3KE' in df.columns:
            df['has_p3ke'] = df['Status P3KE'].str.contains('Terdata', na=False).astype(int)
            
            # Extract desil number
            desil_extract = df['Status P3KE'].str.extract(r'Desil (\d+)')
            df['p3ke_desil'] = pd.to_numeric(desil_extract[0], errors='coerce').fillna(10)
            
            self.engineered_features.extend(['has_p3ke', 'p3ke_desil'])
        
        # KIP/KKS status
        for program in ['No. KIP', 'No. KKS']:
            if program in df.columns:
                has_program = f'has_{program.split(". ")[1].lower()}'
                df[has_program] = (~df[program].isin(['-', '', np.nan])).astype(int)
                self.engineered_features.append(has_program)
        
        # Total assistance programs
        assistance_cols = [col for col in df.columns if col.startswith('has_') and col in self.engineered_features]
        if assistance_cols:
            df['total_assistance_programs'] = df[assistance_cols].sum(axis=1)
            self.engineered_features.append('total_assistance_programs')
        
        return df
    
    def _engineer_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer demographic features."""
        
        # Gender encoding
        if 'Jenis Kelamin' in df.columns:
            df['is_female'] = (df['Jenis Kelamin'] == 'P').astype(int)
            self.engineered_features.append('is_female')
        
        # Age calculation
        if 'Tanggal Lahir' in df.columns:
            try:
                df['birth_date'] = pd.to_datetime(df['Tanggal Lahir'], format='%d/%m/%Y', errors='coerce')
                current_year = 2024  # Assuming current analysis year
                df['age'] = current_year - df['birth_date'].dt.year
                df['age'] = df['age'].fillna(df['age'].median())
                self.engineered_features.append('age')
            except:
                pass
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mixed') -> pd.DataFrame:
        """
        Handle missing values with domain-appropriate strategies.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Imputation strategy ('mixed', 'simple', 'knn')
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        print(f"🔧 Handling missing values with {strategy} strategy...")
        
        df_imputed = df.copy()
        
        if strategy == 'mixed':
            # Use domain-specific imputation
            
            # Categorical features - mode imputation
            for col in self.categorical_features:
                if col in df_imputed.columns and df_imputed[col].isnull().any():
                    mode_value = df_imputed[col].mode()
                    if len(mode_value) > 0:
                        df_imputed[col] = df_imputed[col].fillna(mode_value[0])
                    else:
                        df_imputed[col] = df_imputed[col].fillna('Unknown')
            
            # Numeric features - median for skewed, mean for normal
            for col in self.numeric_features + self.engineered_features:
                if col in df_imputed.columns and df_imputed[col].isnull().any():
                    # Use median for most socio-economic variables (typically skewed)
                    fill_value = df_imputed[col].median()
                    df_imputed[col] = df_imputed[col].fillna(fill_value)
        
        elif strategy == 'simple':
            # Simple imputation
            cat_imputer = SimpleImputer(strategy='most_frequent')
            num_imputer = SimpleImputer(strategy='median')
            
            # Apply to categorical
            cat_cols = [col for col in self.categorical_features if col in df_imputed.columns]
            if cat_cols:
                df_imputed[cat_cols] = cat_imputer.fit_transform(df_imputed[cat_cols])
            
            # Apply to numeric
            num_cols = [col for col in self.numeric_features + self.engineered_features if col in df_imputed.columns]
            if num_cols:
                df_imputed[num_cols] = num_imputer.fit_transform(df_imputed[num_cols])
        
        elif strategy == 'knn':
            # KNN imputation for numeric features only
            num_cols = [col for col in self.numeric_features + self.engineered_features if col in df_imputed.columns]
            if num_cols:
                knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
                df_imputed[num_cols] = knn_imputer.fit_transform(df_imputed[num_cols])
            
            # Simple mode for categorical
            cat_cols = [col for col in self.categorical_features if col in df_imputed.columns]
            for col in cat_cols:
                if df_imputed[col].isnull().any():
                    mode_value = df_imputed[col].mode()
                    if len(mode_value) > 0:
                        df_imputed[col] = df_imputed[col].fillna(mode_value[0])
        
        print(f"✅ Missing values handled successfully")
        return df_imputed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        print("🔤 Encoding categorical features...")
        
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            if col in df_encoded.columns and col not in ['source_file']:
                # Initialize label encoder for this column
                le = LabelEncoder()
                
                # Handle any remaining missing values
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                
                # Check if column already has '_encoded' suffix to avoid double encoding
                if col.endswith('_encoded'):
                    # Already encoded feature, just ensure it's numeric
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    encoded_col = col
                else:
                    # Fit and transform with _encoded suffix
                    encoded_col = f'{col}_encoded'
                    df_encoded[encoded_col] = le.fit_transform(df_encoded[col].astype(str))
                
                # Store encoder for potential inverse transform
                self.label_encoders[col] = le
        
        print(f"✅ Encoded {len(self.label_encoders)} categorical features")
        return df_encoded
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      method: str = 'standard') -> tuple:
        """
        Scale numeric features.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            method (str): Scaling method ('standard', 'minmax')
            
        Returns:
            tuple: Scaled training and test sets
        """
        print(f"📏 Scaling features using {method} method...")
        
        # Select numeric columns for scaling
        numeric_cols = []
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64'] and col != 'is_kip_recipient':
                numeric_cols.append(col)
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Fit and transform training data
        X_train_scaled = X_train.copy()
        if numeric_cols:
            X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = X_test.copy()
            if numeric_cols:
                X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        print(f"✅ Scaled {len(numeric_cols)} numeric features")
        
        return X_train_scaled, X_test_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'chi2', k: int = 20) -> pd.DataFrame:
        """
        Select most relevant features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Selection method ('chi2', 'f_classif')
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Selected features
        """
        print(f"🎯 Selecting top {k} features using {method}...")
        
        # Ensure all features are numeric and non-negative for chi2
        X_numeric = X.select_dtypes(include=[np.number])
        
        if method == 'chi2':
            # Chi2 requires non-negative values
            X_numeric = X_numeric.clip(lower=0)
            selector = SelectKBest(score_func=chi2, k=min(k, X_numeric.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, X_numeric.shape[1]))
        else:
            raise ValueError("Method must be 'chi2' or 'f_classif'")
        
        X_selected = selector.fit_transform(X_numeric, y)
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        
        # Store feature importance scores
        self.feature_importance = dict(zip(selected_features, selector.scores_[selector.get_support()]))
        self.feature_selector = selector
        
        print(f"✅ Selected {len(selected_features)} features")
        print("Top 5 features:", list(self.feature_importance.keys())[:5])
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str = 'is_kip_recipient',
                           test_size: float = 0.2, feature_selection: bool = True) -> dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            test_size (float): Test set proportion
            feature_selection (bool): Whether to perform feature selection
            
        Returns:
            dict: Preprocessed data splits and metadata
        """
        print("🚀 Starting complete preprocessing pipeline...")
        
        # 1. Identify feature types
        feature_types = self.identify_feature_types(df)
        
        # 2. Engineer features
        df_engineered = self.engineer_features(df)
        
        # 3. Handle missing values
        df_clean = self.handle_missing_values(df_engineered, strategy='mixed')
        
        # 4. Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean)
        
        # 5. Prepare features and target
        # CRITICAL: Exclude target variable and ID columns from features
        exclude_columns = [
            target_col, 'is_recipient', 'is_kip_recipient',  # Target variables
            'source_file', 'year', '#',  # Metadata
            'No. Pendaftaran', 'NIK', 'No. Kartu Keluarga', 'NIK Kepala Keluarga', 'NISN',  # IDs
            'Nama Siswa', 'Alamat Tinggal', 'Alamat Email', 'No. Handphone',  # Personal info
            'Tanggal Daftar', 'Tanggal Ditetapkan', 'Tanggal Dicalonkan',  # Dates
            'User Penetapan', 'User Pencalonan'  # User fields
        ]
        
        # Select only relevant features for modeling
        feature_cols = []
        for col in df_encoded.columns:
            # Skip excluded columns
            if any(excl in col for excl in exclude_columns):
                continue
                
            # Include encoded categorical features and engineered numeric features
            if (col.endswith('_encoded') or 
                col in self.engineered_features or 
                (col in self.numeric_features and df_encoded[col].dtype in ['int64', 'float64'])):
                feature_cols.append(col)
        
        X = df_encoded[feature_cols]
        y = df_encoded[target_col]
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Target distribution: {y.value_counts().to_dict()}")
        
        # 6. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # 7. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, method='standard')
        
        # 8. Feature selection (optional)
        if feature_selection and X_train_scaled.shape[1] > 20:
            k_features = min(20, X_train_scaled.shape[1])
            X_train_selected = self.select_features(X_train_scaled, y_train, method='f_classif', k=k_features)
            
            # Ensure test set has the same columns as training set
            selected_features = list(X_train_selected.columns)
            
            # Check if all selected features exist in test set
            missing_in_test = [f for f in selected_features if f not in X_test_scaled.columns]
            if missing_in_test:
                print(f"⚠️ Warning: Features missing in test set: {missing_in_test}")
                # Use only common features
                common_features = [f for f in selected_features if f in X_test_scaled.columns]
                X_train_selected = X_train_selected[common_features]
                selected_features = common_features
            
            # Apply feature selection to test set
            X_test_selected = X_test_scaled[selected_features]
            
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # 9. Prepare result
        result = {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X_train_selected.columns),
            'preprocessing_info': {
                'n_features_original': len(feature_cols),
                'n_features_selected': X_train_selected.shape[1],
                'n_engineered_features': len(self.engineered_features),
                'n_encoded_features': len(self.label_encoders),
                'train_samples': X_train_selected.shape[0],
                'test_samples': X_test_selected.shape[0],
                'class_distribution': y.value_counts().to_dict()
            }
        }
        
        print("✅ Preprocessing pipeline completed successfully!")
        print(f"📊 Final feature count: {X_train_selected.shape[1]}")
        print(f"📊 Training samples: {X_train_selected.shape[0]}")
        print(f"📊 Test samples: {X_test_selected.shape[0]}")
        
        return result

def main():
    """
    Main function for testing preprocessing pipeline.
    """
    print("🧪 Testing preprocessing pipeline...")
    
    # This would typically be called from main.py with actual data
    print("ℹ️ Preprocessing module ready for use!")

if __name__ == "__main__":
    main()
