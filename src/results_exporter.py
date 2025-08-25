"""
Results Export Module for KIP Kuliah Analysis
Handles exporting analysis results to various formats
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import openpyxl

logger = logging.getLogger(__name__)


class ResultsExporter:
    """
    Handles exporting analysis results to Excel, CSV, and summary reports
    """
    
    def __init__(self, results_path: Path, timestamp: str):
        self.results_path = Path(results_path)
        self.timestamp = timestamp
        
        # Ensure results directory exists
        self.results_path.mkdir(exist_ok=True)
        
    def export_to_excel(self, export_data: Dict) -> Path:
        """
        Export all analysis results to Excel with multiple sheets
        
        Args:
            export_data: Dictionary containing all datasets to export
            
        Returns:
            Path to the exported Excel file
        """
        logger.info("Exporting results to Excel...")
        
        excel_filename = f'kip_analysis_{self.timestamp}.xlsx'
        excel_path = self.results_path / excel_filename
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                
                # Handle dual analysis vs single analysis
                analysis_type = export_data.get('analysis_type', 'single')
                
                if analysis_type == 'dual':
                    # Dual analysis export
                    self._export_dual_analysis(writer, export_data)
                else:
                    # Single analysis export (backward compatibility)
                    self._export_single_analysis(writer, export_data)
                
                # Add metadata sheet
                self._add_metadata_sheet(writer, analysis_type)
                
            logger.info(f"Excel export completed: {excel_path}")
            return excel_path
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise
    
    def _export_dual_analysis(self, writer, export_data: Dict):
        """Export dual analysis results to Excel sheets with standardized format"""
        
        # Sheet 1: Dataset Overview (Original + Cleaned)
        self._create_dataset_overview_sheet(writer, export_data)
        
        # Sheet 2: Cluster Assignments & Distribution
        self._create_cluster_assignment_sheet(writer, export_data)
        
        # Sheet 3: Feature Importance (Top N with context)
        self._create_feature_importance_sheet(writer, export_data)
        
        # Sheet 4: Model Performance Comparison
        self._create_model_performance_sheet(writer, export_data)
        
        # Sheet 5: Analysis Comparison (Full vs Socio-Economic)
        self._create_analysis_comparison_sheet(writer, export_data)
        
        # Sheet 6: Cluster Profiles (Detailed)
        self._create_cluster_profiles_sheet(writer, export_data)
        
        # Sheet 7: Metadata & Technical Notes
        self._create_metadata_sheet(writer, export_data)
    
    def _create_dataset_overview_sheet(self, writer, export_data: Dict):
        """Create standardized dataset overview sheet"""
        
        # Get primary dataset for overview
        if 'data_socio_analysis' in export_data:
            primary_data = export_data['data_socio_analysis']
        elif 'data_full_analysis' in export_data:
            primary_data = export_data['data_full_analysis']
        else:
            return
        
        # Create overview summary
        overview_data = {
            'Metric': [
                'Total Records',
                'Total Features',
                'Missing Values',
                'Data Completeness (%)',
                'Years Covered',
                'Primary Analysis Type'
            ],
            'Value': [
                len(primary_data),
                len(primary_data.columns),
                primary_data.isnull().sum().sum(),
                f"{((1 - primary_data.isnull().sum().sum() / (len(primary_data) * len(primary_data.columns))) * 100):.1f}%",
                f"{primary_data['tahun'].min()} - {primary_data['tahun'].max()}" if 'tahun' in primary_data.columns else 'N/A',
                'Socio-Economic Focus' if 'cluster_socio' in primary_data.columns else 'Full Analysis'
            ]
        }
        
        overview_df = pd.DataFrame(overview_data)
        overview_df.to_excel(writer, sheet_name='Dataset_Overview', index=False)
        
        # Add year distribution if available
        if 'tahun' in primary_data.columns:
            year_dist = primary_data['tahun'].value_counts().sort_index()
            year_df = pd.DataFrame({
                'Year': year_dist.index,
                'Record_Count': year_dist.values,
                'Percentage': (year_dist.values / len(primary_data) * 100).round(1)
            })
            
            # Write to same sheet with offset
            year_df.to_excel(writer, sheet_name='Dataset_Overview', startrow=10, index=False)
        
        logger.info("Created Dataset Overview sheet")
    
    def _create_cluster_assignment_sheet(self, writer, export_data: Dict):
        """Create cluster assignment and distribution summary sheet"""
        
        cluster_data = []
        
        # Process both analyses if available
        if 'data_full_analysis' in export_data:
            full_data = export_data['data_full_analysis']
            if 'cluster_full' in full_data.columns:
                full_dist = full_data['cluster_full'].value_counts().sort_index()
                for cluster_id, count in full_dist.items():
                    cluster_data.append({
                        'Analysis_Type': 'Full Analysis',
                        'Cluster_ID': cluster_id,
                        'Record_Count': count,
                        'Percentage': round(count / len(full_data) * 100, 1),
                        'Description': self._get_cluster_description(full_data, 'cluster_full', cluster_id)
                    })
        
        if 'data_socio_analysis' in export_data:
            socio_data = export_data['data_socio_analysis']
            if 'cluster_socio' in socio_data.columns:
                socio_dist = socio_data['cluster_socio'].value_counts().sort_index()
                for cluster_id, count in socio_dist.items():
                    cluster_data.append({
                        'Analysis_Type': 'Socio-Economic',
                        'Cluster_ID': cluster_id,
                        'Record_Count': count,
                        'Percentage': round(count / len(socio_data) * 100, 1),
                        'Description': self._get_cluster_description(socio_data, 'cluster_socio', cluster_id)
                    })
        
        if cluster_data:
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df.to_excel(writer, sheet_name='Cluster_Distribution', index=False)
            logger.info("Created Cluster Distribution sheet")
    
    def _create_feature_importance_sheet(self, writer, export_data: Dict):
        """Create feature importance sheet with top N features and context"""
        
        if 'feature_importance' not in export_data:
            return
        
        importance_df = export_data['feature_importance']
        
        # Get top 20 features
        top_features = importance_df.head(20).copy()
        
        # Prepare display format
        if 'Display_Name' in top_features.columns:
            display_df = pd.DataFrame({
                'Rank': range(1, len(top_features) + 1),
                'Feature': top_features['Display_Name'],
                'Importance_Score': top_features['Average'].round(4),
                'Feature_Type': top_features.get('Feature_Type', 'Unknown'),
                'Original_Column': top_features.get('Original_Column', top_features.index),
                'Encoding_Method': top_features.get('Encoding_Method', 'None'),
                'Description': top_features.get('Description', 'No description available')
            })
        else:
            display_df = pd.DataFrame({
                'Rank': range(1, len(top_features) + 1),
                'Feature': top_features.index,
                'Importance_Score': top_features['Average'].round(4) if 'Average' in top_features.columns else top_features.mean(axis=1).round(4)
            })
        
        display_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        logger.info("Created Feature Importance sheet")
    
    def _create_model_performance_sheet(self, writer, export_data: Dict):
        """Create model performance comparison sheet"""
        
        if 'model_performance' not in export_data:
            return
        
        performance_df = export_data['model_performance'].copy()
        
        # Ensure proper formatting
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Best_CV_Score']
        for col in numeric_cols:
            if col in performance_df.columns:
                performance_df[col] = performance_df[col].round(4)
        
        # Add ranking
        if 'Accuracy' in performance_df.columns:
            performance_df['Rank'] = performance_df['Accuracy'].rank(ascending=False).astype(int)
            # Reorder columns to put Rank first
            cols = ['Rank'] + [col for col in performance_df.columns if col != 'Rank']
            performance_df = performance_df[cols]
        
        performance_df.to_excel(writer, sheet_name='Model_Performance', index=False)
        logger.info("Created Model Performance sheet")
    
    def _create_analysis_comparison_sheet(self, writer, export_data: Dict):
        """Create analysis comparison sheet for dual analysis"""
        
        if 'analysis_summary' not in export_data:
            return
        
        summary = export_data['analysis_summary']
        comparison_data = []
        
        for analysis_name, metrics in summary.items():
            quality = metrics.get('cluster_quality', {})
            comparison_data.append({
                'Analysis_Type': analysis_name.replace('_', ' ').title(),
                'Optimal_Clusters': metrics['optimal_k'],
                'Algorithm_Used': quality.get('algorithm_used', 'Unknown'),
                'Silhouette_Score': quality.get('silhouette_score', 'N/A'),
                'Davies_Bouldin_Score': quality.get('davies_bouldin_score', 'N/A'),
                'Cohesion_Score': quality.get('cohesion_score', 'N/A'),
                'Separation_Score': quality.get('separation_score', 'N/A'),
                'Quality_Interpretation': self._interpret_clustering_quality(quality)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Analysis_Comparison', index=False)
        logger.info("Created Analysis Comparison sheet")
    
    def _create_cluster_profiles_sheet(self, writer, export_data: Dict):
        """Create detailed cluster profiles sheet"""
        
        # Combine both cluster profiles if available
        profile_data = []
        
        if 'cluster_profiles_full' in export_data:
            full_profiles = export_data['cluster_profiles_full']
            full_profiles['Analysis_Type'] = 'Full Analysis'
            profile_data.append(full_profiles)
        
        if 'cluster_profiles_socio' in export_data:
            socio_profiles = export_data['cluster_profiles_socio']
            socio_profiles['Analysis_Type'] = 'Socio-Economic'
            profile_data.append(socio_profiles)
        
        if profile_data:
            combined_profiles = pd.concat(profile_data, ignore_index=True)
            combined_profiles.to_excel(writer, sheet_name='Cluster_Profiles', index=False)
            logger.info("Created Cluster Profiles sheet")
    
    def _create_metadata_sheet(self, writer, export_data: Dict):
        """Create metadata and technical notes sheet"""
        
        analysis_type = export_data.get('analysis_type', 'single')
        
        metadata_info = [
            ['Generated On', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Analysis Type', 'Dual Analysis (Full + Socio-Economic)' if analysis_type == 'dual' else 'Single Analysis'],
            ['', ''],
            ['CLUSTERING ALGORITHMS', ''],
            ['Full Analysis', 'K-Prototypes (mixed data types)'],
            ['Socio-Economic', 'K-modes (categorical only)'],
            ['', ''],
            ['CLASSIFICATION MODELS', ''],
            ['Random Forest', 'Ensemble method with hyperparameter tuning'],
            ['XGBoost', 'Gradient boosting with cross-validation'],
            ['SVM', 'Support Vector Machine with RBF/Linear kernel'],
            ['', ''],
            ['DATA PREPROCESSING', ''],
            ['Missing Values', 'Median (numerical), Mode (categorical)'],
            ['Scaling', 'StandardScaler for numerical features'],
            ['Categorical Encoding', 'One-hot encoding and Label encoding'],
            ['', ''],
            ['QUALITY METRICS', ''],
            ['Silhouette Score', '0.2-0.5: Fair, 0.5-0.7: Good, >0.7: Excellent'],
            ['Davies-Bouldin', 'Lower values indicate better clustering'],
            ['Cohesion Score', 'Higher values indicate more similar clusters'],
            ['Separation Score', 'Higher values indicate more distinct clusters'],
        ]
        
        metadata_df = pd.DataFrame(metadata_info, columns=['Parameter', 'Description'])
        metadata_df.to_excel(writer, sheet_name='Metadata_Notes', index=False)
        logger.info("Created Metadata & Notes sheet")
    
    def _get_cluster_description(self, data: pd.DataFrame, cluster_col: str, cluster_id: int) -> str:
        """Generate brief cluster description"""
        
        cluster_data = data[data[cluster_col] == cluster_id]
        
        # Get most common values for key categorical features
        description_parts = []
        
        # Check for year dominance
        if 'tahun' in cluster_data.columns:
            year_mode = cluster_data['tahun'].mode()
            if len(year_mode) > 0:
                year_pct = (cluster_data['tahun'] == year_mode.iloc[0]).mean() * 100
                if year_pct >= 50:
                    description_parts.append(f"Dominan {year_mode.iloc[0]} ({year_pct:.0f}%)")
        
        # Check for gender dominance
        if 'Jenis Kelamin' in cluster_data.columns:
            gender_mode = cluster_data['Jenis Kelamin'].mode()
            if len(gender_mode) > 0:
                gender_pct = (cluster_data['Jenis Kelamin'] == gender_mode.iloc[0]).mean() * 100
                if gender_pct >= 60:
                    description_parts.append(f"{gender_mode.iloc[0]} ({gender_pct:.0f}%)")
        
        return ', '.join(description_parts) if description_parts else f"Cluster {cluster_id}"
    
    def _interpret_clustering_quality(self, quality: Dict) -> str:
        """Interpret clustering quality metrics"""
        
        interpretations = []
        
        if 'silhouette_score' in quality and quality['silhouette_score'] != 'N/A':
            score = float(quality['silhouette_score'])
            if score > 0.7:
                interpretations.append("Excellent cluster separation")
            elif score > 0.5:
                interpretations.append("Good cluster quality")
            elif score > 0.2:
                interpretations.append("Fair cluster structure")
            else:
                interpretations.append("Weak cluster separation")
        
        if 'davies_bouldin_score' in quality and quality['davies_bouldin_score'] != 'N/A':
            score = float(quality['davies_bouldin_score'])
            if score < 1:
                interpretations.append("Well-separated clusters")
            elif score < 2:
                interpretations.append("Moderately separated")
            else:
                interpretations.append("Overlapping clusters")
        
        if 'cohesion_score' in quality and quality['cohesion_score'] != 'N/A':
            score = float(quality['cohesion_score'])
            if score > 0.7:
                interpretations.append("High intra-cluster similarity")
            elif score > 0.5:
                interpretations.append("Moderate cohesion")
            else:
                interpretations.append("Low internal consistency")
        
        return '; '.join(interpretations) if interpretations else "Quality metrics not available"
    
    def _export_single_analysis(self, writer, export_data: Dict):
        """Export single analysis results to Excel sheets (backward compatibility)"""
        
        # Export each dataset to a separate sheet
        for sheet_name, data in export_data.items():
            if sheet_name == 'analysis_type':
                continue  # Skip metadata
                
            if isinstance(data, pd.DataFrame):
                # Handle sheet name length limit (31 chars in Excel)
                safe_sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                
                data.to_excel(writer, sheet_name=safe_sheet_name, index=True)
                logger.info(f"Exported {sheet_name} to sheet '{safe_sheet_name}'")
                
            elif isinstance(data, dict):
                # Convert dict to DataFrame if possible
                try:
                    df = pd.DataFrame(data)
                    safe_sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                    df.to_excel(writer, sheet_name=safe_sheet_name, index=True)
                    logger.info(f"Exported {sheet_name} (dict) to sheet '{safe_sheet_name}'")
                except Exception as e:
                    logger.warning(f"Could not export {sheet_name} as DataFrame: {e}")
            
            else:
                logger.warning(f"Skipping {sheet_name} - unsupported data type: {type(data)}")
    
    def _export_standard_sheets(self, writer, export_data: Dict):
        """Export standard sheets common to both analysis types"""
        
        standard_sheets = ['model_performance', 'feature_importance', 'predictions']
        
        for sheet_name in standard_sheets:
            if sheet_name in export_data:
                data = export_data[sheet_name]
                if isinstance(data, pd.DataFrame):
                    data.to_excel(writer, sheet_name=sheet_name, index=True)
                    logger.info(f"Exported {sheet_name}")
                elif isinstance(data, dict):
                    try:
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=sheet_name, index=True)
                        logger.info(f"Exported {sheet_name} (dict)")
                    except Exception as e:
                        logger.warning(f"Could not export {sheet_name} as DataFrame: {e}")
    
    def _add_metadata_sheet(self, writer, analysis_type: str = 'single'):
        """Add metadata sheet to Excel workbook"""
        
        if analysis_type == 'dual':
            metadata = {
                'Export Information': [
                    f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    f'Analysis timestamp: {self.timestamp}',
                    'Analysis type: KIP Kuliah Dual Analysis (Full + Socio-Economic)',
                    '',
                    'Sheet Descriptions:',
                    '- data_full_analysis: Original data with all features clustered',
                    '- data_socio_analysis: Data with socio-economic focused clustering',
                    '- clusters_full: Statistical profiles for full analysis clusters',
                    '- clusters_socio: Statistical profiles for socio-economic clusters',
                    '- analysis_comparison: Comparison summary of both analyses',
                    '- model_performance: Classification model comparison metrics',
                    '- feature_importance: Feature importance rankings per model',
                    '- predictions: Model predictions on test data',
                    '',
                    'Note: Classification models trained on socio-economic focused data'
                ]
            }
        else:
            metadata = {
                'Export Information': [
                    f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    f'Analysis timestamp: {self.timestamp}',
                    'Analysis type: KIP Kuliah Clustering and Classification',
                    '',
                    'Sheet Descriptions:',
                    '- data_with_clusters: Original data with cluster assignments',
                    '- cluster_profiles: Statistical profiles for each cluster', 
                    '- model_performance: Classification model comparison metrics',
                    '- feature_importance: Feature importance rankings per model',
                    '- predictions: Model predictions on test data'
                ]
            }
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False, header=False)
    
    def export_to_csv(self, export_data: Dict) -> List[Path]:
        """
        Export each dataset to separate CSV files
        
        Args:
            export_data: Dictionary containing all datasets to export
            
        Returns:
            List of paths to exported CSV files
        """
        logger.info("Exporting results to CSV files...")
        
        csv_folder = self.results_path / f'csv_results_{self.timestamp}'
        csv_folder.mkdir(exist_ok=True)
        
        exported_files = []
        analysis_type = export_data.get('analysis_type', 'single')
        
        for name, data in export_data.items():
            # Skip metadata
            if name in ['analysis_type']:
                continue
                
            try:
                if isinstance(data, pd.DataFrame):
                    csv_path = csv_folder / f'{name}.csv'
                    data.to_csv(csv_path, index=True, encoding='utf-8')
                    exported_files.append(csv_path)
                    logger.info(f"Exported {name} to CSV: {csv_path}")
                    
                elif isinstance(data, dict):
                    # Handle nested dict structures (like analysis_summary)
                    if name == 'analysis_summary' and analysis_type == 'dual':
                        # Create comparison CSV for dual analysis
                        comparison_data = []
                        for analysis_name, metrics in data.items():
                            comparison_data.append({
                                'Analysis Type': analysis_name.replace('_', ' ').title(),
                                'Optimal Clusters': metrics['optimal_k'],
                                'Silhouette Score': metrics['cluster_quality'].get('silhouette_score', 'N/A'),
                                'Davies-Bouldin Score': metrics['cluster_quality'].get('davies_bouldin_score', 'N/A')
                            })
                        
                        df = pd.DataFrame(comparison_data)
                        csv_path = csv_folder / 'analysis_comparison.csv'
                        df.to_csv(csv_path, index=False, encoding='utf-8')
                        exported_files.append(csv_path)
                        logger.info(f"Exported analysis comparison to CSV: {csv_path}")
                    else:
                        # Convert regular dict to DataFrame
                        try:
                            df = pd.DataFrame(data)
                            csv_path = csv_folder / f'{name}.csv'
                            df.to_csv(csv_path, index=True, encoding='utf-8')
                            exported_files.append(csv_path)
                            logger.info(f"Exported {name} (dict) to CSV: {csv_path}")
                        except Exception as e:
                            logger.warning(f"Could not export {name} as CSV: {e}")
                
                else:
                    logger.warning(f"Skipping {name} - unsupported data type for CSV: {type(data)}")
                    
            except Exception as e:
                logger.error(f"Error exporting {name} to CSV: {e}")
        
        logger.info(f"CSV export completed: {len(exported_files)} files in {csv_folder}")
        return exported_files
    
    def create_summary_report(self, export_data: Dict, optimal_k: int, clustering_analyzer=None) -> Path:
        """
        Create a comprehensive text summary report
        
        Args:
            export_data: Dictionary containing analysis results
            optimal_k: Optimal number of clusters found
            clustering_analyzer: The clustering analyzer object
            
        Returns:
            Path to the summary report file
        """
        logger.info("Creating summary report...")
        
        report_path = self.results_path / f'analysis_summary_{self.timestamp}.txt'
        analysis_type = export_data.get('analysis_type', 'single')
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                
                # Header
                f.write("=" * 80 + "\n")
                if analysis_type == 'dual':
                    f.write("KIP KULIAH DUAL ANALYSIS SUMMARY REPORT\n")
                else:
                    f.write("KIP KULIAH ANALYSIS SUMMARY REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analysis ID: {self.timestamp}\n")
                f.write(f"Analysis Type: {analysis_type.title()}\n\n")
                
                # Dataset Overview
                f.write("DATASET OVERVIEW\n")
                f.write("-" * 40 + "\n")
                
                # Get primary dataset for overview
                primary_df = None
                if analysis_type == 'dual':
                    if 'data_socio_analysis' in export_data:
                        primary_df = export_data['data_socio_analysis']
                        f.write("Note: Overview based on socio-economic focused dataset\n")
                elif 'data_with_clusters' in export_data:
                    primary_df = export_data['data_with_clusters']
                
                if primary_df is not None:
                    f.write(f"Total Records: {len(primary_df):,}\n")
                    f.write(f"Total Features: {len(primary_df.columns):,}\n")
                    f.write(f"Missing Values: {primary_df.isnull().sum().sum():,}\n")
                    
                    if 'tahun' in primary_df.columns:
                        year_dist = primary_df['tahun'].value_counts().sort_index()
                        f.write(f"Data by Year:\n")
                        for year, count in year_dist.items():
                            f.write(f"  {year}: {count:,} records\n")
                
                f.write("\n")
                
                # Clustering Results
                if analysis_type == 'dual':
                    self._write_dual_clustering_results(f, export_data)
                else:
                    self._write_single_clustering_results(f, export_data, optimal_k, clustering_analyzer)
                
                # Model Performance
                self._write_model_performance(f, export_data)
                
                # Feature Importance
                self._write_feature_importance(f, export_data)
                
                # Recommendations
                self._write_recommendations(f, export_data, analysis_type)
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
                
            logger.info(f"Summary report created: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            raise
    
    def _write_dual_clustering_results(self, f, export_data: Dict):
        """Write clustering results for dual analysis"""
        
        f.write("DUAL CLUSTERING ANALYSIS RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write("Algorithm: K-Prototypes (mixed data types)\n\n")
        
        if 'analysis_summary' in export_data:
            summary = export_data['analysis_summary']
            
            # Full Analysis Results
            if 'full_analysis' in summary:
                full_metrics = summary['full_analysis']
                f.write("1. FULL ANALYSIS (All Features Including Administrative):\n")
                f.write(f"   Optimal Clusters: {full_metrics['optimal_k']}\n")
                
                quality = full_metrics.get('cluster_quality', {})
                if 'silhouette_score' in quality:
                    f.write(f"   Silhouette Score: {quality['silhouette_score']:.4f}\n")
                if 'davies_bouldin_score' in quality:
                    f.write(f"   Davies-Bouldin Index: {quality['davies_bouldin_score']:.4f}\n")
                
                if 'data_full_analysis' in export_data:
                    df_full = export_data['data_full_analysis']
                    cluster_dist = df_full['cluster_full'].value_counts().sort_index()
                    f.write(f"   Cluster Distribution:\n")
                    for cluster, count in cluster_dist.items():
                        pct = (count / len(df_full)) * 100
                        f.write(f"     Cluster {cluster}: {count:,} ({pct:.1f}%)\n")
                f.write("\n")
            
            # Socio-Economic Analysis Results
            if 'socio_analysis' in summary:
                socio_metrics = summary['socio_analysis']
                f.write("2. SOCIO-ECONOMIC FOCUS ANALYSIS (Excluding Administrative Features):\n")
                f.write(f"   Optimal Clusters: {socio_metrics['optimal_k']}\n")
                
                quality = socio_metrics.get('cluster_quality', {})
                if 'silhouette_score' in quality:
                    f.write(f"   Silhouette Score: {quality['silhouette_score']:.4f}\n")
                if 'davies_bouldin_score' in quality:
                    f.write(f"   Davies-Bouldin Index: {quality['davies_bouldin_score']:.4f}\n")
                
                if 'data_socio_analysis' in export_data:
                    df_socio = export_data['data_socio_analysis']
                    cluster_dist = df_socio['cluster_socio'].value_counts().sort_index()
                    f.write(f"   Cluster Distribution:\n")
                    for cluster, count in cluster_dist.items():
                        pct = (count / len(df_socio)) * 100
                        f.write(f"     Cluster {cluster}: {count:,} ({pct:.1f}%)\n")
                f.write("\n")
        
        f.write("Note: Classification models were trained on socio-economic focused data\n")
        f.write("      to predict based on inherent characteristics rather than administrative patterns.\n\n")
    
    def _write_single_clustering_results(self, f, export_data: Dict, optimal_k: int, clustering_analyzer):
        """Write clustering results for single analysis"""
        
        f.write("CLUSTERING ANALYSIS RESULTS\n")
        f.write("-" * 40 + "\n")
        
        # Add quality metrics if available
        algorithm_used = 'K-Prototypes'
        if hasattr(clustering_analyzer, 'quality_metrics'):
            metrics = clustering_analyzer.quality_metrics
            algorithm_used = metrics.get('algorithm_used', 'K-Prototypes')
            
            f.write(f"Algorithm: {algorithm_used}\n")
            f.write(f"Optimal Clusters: {optimal_k}\n")
            
            # Handle different metric types based on algorithm
            if algorithm_used == 'K-modes':
                if 'cohesion_score' in metrics:
                    f.write(f"Cluster Cohesion: {metrics['cohesion_score']:.4f}\n")
                if 'separation_score' in metrics:
                    f.write(f"Cluster Separation: {metrics['separation_score']:.4f}\n")
            else:
                # K-Prototypes or K-Means metrics
                silhouette = metrics.get('silhouette_score', 'N/A')
                davies_bouldin = metrics.get('davies_bouldin_score', 'N/A')
                
                if silhouette != 'N/A':
                    f.write(f"Silhouette Score: {silhouette:.4f}\n")
                else:
                    f.write(f"Silhouette Score: {silhouette}\n")
                    
                if davies_bouldin != 'N/A':
                    f.write(f"Davies-Bouldin Index: {davies_bouldin:.4f}\n")
                else:
                    f.write(f"Davies-Bouldin Index: {davies_bouldin}\n")
        else:
            f.write(f"Algorithm: {algorithm_used}\n")
            f.write(f"Optimal Clusters: {optimal_k}\n")
        
        if 'data_with_clusters' in export_data:
            df = export_data['data_with_clusters']
            if 'cluster' in df.columns:
                cluster_dist = df['cluster'].value_counts().sort_index()
                f.write(f"Cluster Distribution:\n")
                for cluster_id, count in cluster_dist.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  Cluster {cluster_id}: {count:,} records ({percentage:.1f}%)\n")
                
                # Add cluster interpretations if available
                if hasattr(clustering_analyzer, 'cluster_interpretations'):
                    f.write("\n")
                    f.write("CLUSTER INTERPRETATIONS\n")
                    f.write("-" * 40 + "\n")
                    for cluster_id in sorted(cluster_dist.index):
                        if cluster_id in clustering_analyzer.cluster_interpretations:
                            count = cluster_dist[cluster_id]
                            percentage = (count / len(df)) * 100
                            interpretation = clustering_analyzer.cluster_interpretations[cluster_id]
                            f.write(f"Cluster {cluster_id} ({count:,} records, {percentage:.1f}%):\n")
                            f.write(f"  {interpretation}\n\n")
        
        f.write("\n")
    
    def _write_model_performance(self, f, export_data: Dict):
        """Write model performance section"""
        
        f.write("CLASSIFICATION MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        
        if 'model_performance' in export_data:
            performance_df = export_data['model_performance']
            
            # Best model
            if not performance_df.empty:
                best_idx = performance_df['Accuracy'].idxmax()
                best_model = performance_df.loc[best_idx]
                
                f.write(f"Best Model: {best_model['Model']}\n")
                f.write(f"  Accuracy: {best_model['Accuracy']:.4f}\n")
                f.write(f"  Precision: {best_model['Precision']:.4f}\n")
                f.write(f"  Recall: {best_model['Recall']:.4f}\n")
                f.write(f"  F1-Score: {best_model['F1-Score']:.4f}\n\n")
                
                # All models comparison
                f.write("All Models Performance:\n")
                for _, row in performance_df.iterrows():
                    f.write(f"  {row['Model']}:\n")
                    f.write(f"    Accuracy: {row['Accuracy']:.4f}\n")
                    f.write(f"    F1-Score: {row['F1-Score']:.4f}\n")
                    f.write(f"    CV Score: {row['Best_CV_Score']:.4f}\n")
        
        f.write("\n")
    
    def _write_feature_importance(self, f, export_data: Dict):
        """Write feature importance section"""
        
        f.write("TOP INFLUENTIAL FEATURES\n")
        f.write("-" * 40 + "\n")
        
        if 'feature_importance' in export_data:
            importance_df = export_data['feature_importance']
            
            if not importance_df.empty:
                # Calculate average importance across all models
                avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
                
                f.write("Top 10 Features (Average Importance):\n")
                for i, (feature, importance) in enumerate(avg_importance.head(10).items(), 1):
                    f.write(f"  {i:2d}. {feature}: {importance:.4f}\n")
        
        f.write("\n")
    
    def _write_recommendations(self, f, export_data: Dict, analysis_type: str):
        """Write recommendations section"""
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if analysis_type == 'dual':
            f.write("Based on the dual analysis approach:\n\n")
            f.write("1. SOCIO-ECONOMIC INSIGHTS:\n")
            f.write("   - Focus on socio-economic cluster patterns for targeted interventions\n")
            f.write("   - Use socio-economic analysis for policy development\n")
            f.write("   - These clusters represent true socio-economic diversity\n\n")
            
            f.write("2. ADMINISTRATIVE PATTERNS:\n")
            f.write("   - Full analysis shows administrative/temporal clustering\n")
            f.write("   - Useful for operational planning and resource allocation\n")
            f.write("   - Shows how different cohorts/pathways behave\n\n")
            
            f.write("3. PREDICTIVE MODELING:\n")
            f.write("   - Models trained on socio-economic features are more generalizable\n")
            f.write("   - Better for predicting outcomes based on inherent characteristics\n")
            f.write("   - More robust across different time periods and pathways\n\n")
        else:
            f.write("Based on the analysis results:\n\n")
            f.write("1. Focus on the highest performing clusters for successful outcomes\n")
            f.write("2. Target intervention programs for underperforming clusters\n")
            f.write("3. Use predictive models for early identification of at-risk students\n")
            f.write("4. Monitor feature importance trends for program adjustments\n\n")
        
        # Files Generated
        f.write("FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write(f"Excel Report: kip_analysis_{self.timestamp}.xlsx\n")
        f.write(f"CSV Folder: csv_results_{self.timestamp}/\n")
        f.write(f"Summary Report: analysis_summary_{self.timestamp}.txt\n")
        f.write(f"Visualizations: *.png files in results folder\n\n")
        
        # Technical Notes
        f.write("TECHNICAL NOTES\n")
        f.write("-" * 40 + "\n")
        f.write("- Clustering: K-Prototypes algorithm for mixed data types\n")
        f.write("- Classification: Random Forest, XGBoost, SVM with hyperparameter tuning\n")
        f.write("- Validation: Cross-validation and train-test split\n")
        f.write("- Missing Values: Handled with median (numerical) and mode (categorical)\n")
        f.write("- Scaling: StandardScaler applied to all features\n\n")
    
    def export_visualizations_info(self) -> Path:
        """Create a file listing all generated visualizations"""
        
        viz_info_path = self.results_path / f'visualizations_info_{self.timestamp}.txt'
        
        visualizations = [
            'elbow_curve.png - Elbow method for optimal cluster selection',
            'cluster_distribution.png - Distribution of data points across clusters',
            'numerical_features_by_cluster.png - Histograms of numerical features by cluster',
            'feature_importance.png - Feature importance rankings for each model',
            'model_comparison.png - Performance metrics comparison across models',
            'model_comparison_grouped.png - Grouped comparison of all metrics',
            'confusion_matrices.png - Confusion matrices for all classification models'
        ]
        
        with open(viz_info_path, 'w', encoding='utf-8') as f:
            f.write("GENERATED VISUALIZATIONS\n")
            f.write("=" * 40 + "\n")
            f.write(f"Analysis ID: {self.timestamp}\n\n")
            
            for viz in visualizations:
                f.write(f"• {viz}\n")
        
        return viz_info_path
    
    def create_data_dictionary(self, df: pd.DataFrame) -> Path:
        """Create a data dictionary describing all columns"""
        
        dict_path = self.results_path / f'data_dictionary_{self.timestamp}.csv'
        
        data_dict = []
        
        for col in df.columns:
            col_info = {
                'Column_Name': col,
                'Data_Type': str(df[col].dtype),
                'Non_Null_Count': df[col].count(),
                'Null_Count': df[col].isnull().sum(),
                'Unique_Values': df[col].nunique(),
                'Sample_Values': str(df[col].dropna().head(3).tolist()[:3])
            }
            
            # Add statistics for numerical columns
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max()
                })
            
            data_dict.append(col_info)
        
        dict_df = pd.DataFrame(data_dict)
        dict_df.to_csv(dict_path, index=False)
        
        logger.info(f"Data dictionary created: {dict_path}")
        return dict_path
