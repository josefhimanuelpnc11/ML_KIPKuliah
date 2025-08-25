"""
Configuration settings for KIP Kuliah analysis
"""

from pathlib import Path


class Config:
    """
    Central configuration for all analysis components
    """
    
    def __init__(self):
        # Base paths - get project root (parent of src)
        self.project_root = Path(__file__).parent.parent
        self.data_path = self.project_root / "Bahan Laporan KIP Kuliah 2022 s.d 2024"
        self.results_path = self.project_root / "results"
        self.models_path = self.project_root / "models"
        self.logs_path = self.project_root / "logs"
        
        # Ensure directories exist
        self.results_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)
        
        # Data processing configuration
        self.data_config = {
            'cohort_features': ['tahun', 'jalur'],  # Features that indicate cohort/administrative info
            'socioeconomic_focus': True,  # Whether to run socio-economic focused analysis
            'run_dual_analysis': True,  # Run both full and socio-economic only analysis
            'minimum_cluster_size': 5  # Minimum records per cluster
        }
        
        # Clustering configuration
        self.clustering_config = {
            'algorithm': 'kprototypes',
            'max_k': 8,  # Reduced for better balance
            'min_k': 2,
            'random_state': 42,
            'init_method': 'Huang',
            'n_init': 5,
            # Quality thresholds
            'min_silhouette': 0.3,
            'max_imbalance_ratio': 0.7  # Max 70% in single cluster
        }
        
        # Classification configuration
        self.classification_config = {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 3,
            'scoring': 'accuracy',
            'models': {
                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        # Data processing configuration with dual analysis approach
        self.data_config = {
            'exclude_columns': ['id', 'nama', 'name', 'no', 'nik', 'nisn'],
            'missing_value_strategy': {
                'numerical': 'median',
                'categorical': 'mode'
            },
            'scaling_method': 'standard',
            'cohort_features': ['tahun', 'jalur'],  # Administrative/temporal features
            'socioeconomic_focus': True,  # Primary focus on socio-economic patterns
            'run_dual_analysis': True,  # Run both full and socio-economic focused analysis
        }
        
        # Export configuration
        self.export_config = {
            'excel_engine': 'openpyxl',
            'csv_encoding': 'utf-8',
            'include_index': True
        }
