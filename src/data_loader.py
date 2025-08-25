"""
Data Loading Module for KIP Kuliah Analysis
Handles loading and basic validation of CSV files
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List

logger = logging.getLogger(__name__)


class KIPDataLoader:
    """
    Handles loading of KIP Kuliah data from multiple CSV files
    """
    
    def __init__(self, base_path: Path):
        # Convert to absolute path and handle relative paths properly
        current_dir = Path(__file__).parent  # This is the src directory
        project_root = current_dir.parent    # This is the project root
        
        if not Path(base_path).is_absolute():
            # Always resolve relative paths from project root
            self.base_path = (project_root / base_path).resolve()
        else:
            self.base_path = Path(base_path).resolve()
        
        logger.info(f"Data base path set to: {self.base_path}")
        self.pendaftar_files = self._define_pendaftar_files()
        self.penerima_files = self._define_penerima_files()
    
    def _define_pendaftar_files(self) -> dict:
        """Define pendaftar file mappings by year"""
        return {
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
    
    def _define_penerima_files(self) -> dict:
        """Define penerima file mappings by year"""
        return {
            2022: 'CSV_Penerima/penerima KIP Kuliah angkatan 2022.csv',
            2023: 'CSV_Penerima/penerima KIP Kuliah angkatan 2023.csv',
            2024: 'CSV_Penerima/penerima KIP Kuliah angkatan 2024.csv'
        }
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all pendaftar and penerima data
        
        Returns:
            Tuple of (pendaftar_df, penerima_df)
        """
        logger.info("Loading all KIP Kuliah data...")
        
        pendaftar_df = self._load_pendaftar_data()
        penerima_df = self._load_penerima_data()
        
        # Validate loaded data
        self._validate_data(pendaftar_df, penerima_df)
        
        return pendaftar_df, penerima_df
    
    def _load_pendaftar_data(self) -> pd.DataFrame:
        """Load and combine all pendaftar data"""
        all_pendaftar = []
        
        for year, files in self.pendaftar_files.items():
            for file_path in files:
                full_path = self.base_path / file_path
                
                if not full_path.exists():
                    logger.warning(f"File not found: {full_path}")
                    continue
                
                try:
                    df = pd.read_csv(full_path)
                    df['tahun'] = year
                    df['jalur'] = Path(file_path).stem
                    df['source_file'] = str(file_path)
                    
                    all_pendaftar.append(df)
                    logger.info(f"Loaded {file_path}: {len(df)} records")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        if not all_pendaftar:
            logger.warning("No pendaftar data loaded")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_pendaftar, ignore_index=True)
        logger.info(f"Combined pendaftar data: {len(combined_df)} total records")
        
        return combined_df
    
    def _load_penerima_data(self) -> pd.DataFrame:
        """Load and combine all penerima data"""
        all_penerima = []
        
        for year, file_path in self.penerima_files.items():
            full_path = self.base_path / file_path
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                continue
            
            try:
                df = pd.read_csv(full_path)
                df['tahun'] = year
                df['status'] = 'diterima'
                df['source_file'] = str(file_path)
                
                all_penerima.append(df)
                logger.info(f"Loaded {file_path}: {len(df)} records")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not all_penerima:
            logger.warning("No penerima data loaded")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_penerima, ignore_index=True)
        logger.info(f"Combined penerima data: {len(combined_df)} total records")
        
        return combined_df
    
    def _validate_data(self, pendaftar_df: pd.DataFrame, penerima_df: pd.DataFrame):
        """Validate loaded data quality"""
        
        # Check if pendaftar data exists
        if pendaftar_df.empty:
            raise ValueError("No pendaftar data loaded. Check file paths and data structure.")
        
        # Log data quality metrics with detailed missing values analysis
        logger.info("Data Quality Summary:")
        logger.info(f"  Pendaftar records: {len(pendaftar_df)}")
        
        if not pendaftar_df.empty:
            total_cells = pendaftar_df.shape[0] * pendaftar_df.shape[1]
            missing_cells = pendaftar_df.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100
            
            logger.info(f"  Pendaftar missing values: {missing_cells:,} ({missing_percentage:.1f}% of total cells)")
            logger.info(f"  Pendaftar columns: {len(pendaftar_df.columns)}")
            
            # Analyze missing values by column
            missing_by_column = pendaftar_df.isnull().sum().sort_values(ascending=False)
            top_missing = missing_by_column[missing_by_column > 0].head(10)
            
            if len(top_missing) > 0:
                logger.info("  Top 10 columns with missing values:")
                for col, missing_count in top_missing.items():
                    missing_pct = (missing_count / len(pendaftar_df)) * 100
                    logger.info(f"    {col}: {missing_count:,} ({missing_pct:.1f}%)")
        
        if not penerima_df.empty:
            total_cells_penerima = penerima_df.shape[0] * penerima_df.shape[1]
            missing_cells_penerima = penerima_df.isnull().sum().sum()
            missing_percentage_penerima = (missing_cells_penerima / total_cells_penerima) * 100
            
            logger.info(f"  Penerima records: {len(penerima_df)}")
            logger.info(f"  Penerima missing values: {missing_cells_penerima:,} ({missing_percentage_penerima:.1f}% of total cells)")
        
        # Check for completely empty columns
        empty_cols = pendaftar_df.columns[pendaftar_df.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Empty columns found: {empty_cols}")
        
        # Check for duplicate rows
        duplicates = pendaftar_df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows in pendaftar data")
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get comprehensive data summary"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'year_distribution': df['tahun'].value_counts().to_dict() if 'tahun' in df.columns else {}
        }
