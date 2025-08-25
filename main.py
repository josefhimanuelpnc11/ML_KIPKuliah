"""
Main Analysis Script for KIP Kuliah Data Mining
Comprehensive workflow: Data Loading → Preprocessing → Clustering → Classification → Evaluation → Export
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import KIPDataLoader
from data_preprocessor import DataPreprocessor
from clustering_analyzer import ClusteringAnalyzer
from classification_trainer import ClassificationTrainer
from results_exporter import ResultsExporter


# Dual Analysis Comparison Functions

def generate_detailed_comparison(full_results: Dict, socio_results: Dict, 
                               data_full: pd.DataFrame, data_socio: pd.DataFrame) -> Dict:
    """Generate comprehensive comparison between full and socio-economic analysis"""
    
    comparison = {}
    
    # 1. Administrative vs Inherent Pattern Analysis
    comparison['administrative_insights'] = analyze_administrative_patterns(
        full_results, data_full
    )
    
    comparison['socioeconomic_insights'] = analyze_socioeconomic_patterns(
        socio_results, data_socio
    )
    
    # 2. Clustering Algorithm Effectiveness
    comparison['clustering_effectiveness'] = compare_clustering_effectiveness(
        full_results, socio_results
    )
    
    # 3. Feature Importance Differences
    comparison['feature_differences'] = analyze_feature_importance_differences(
        full_results, socio_results
    )
    
    # 4. Key Differentiators
    comparison['key_differentiators'] = identify_key_differentiators(
        full_results, socio_results, data_full, data_socio
    )
    
    # 5. Pattern Overlap Analysis
    comparison['pattern_overlap'] = analyze_pattern_overlap(
        data_full, data_socio
    )
    
    # 6. Actionable Insights
    comparison['actionable_insights'] = generate_actionable_insights(
        full_results, socio_results, data_full, data_socio
    )
    
    # 7. Statistical Significance
    comparison['statistical_significance'] = assess_statistical_significance(
        full_results, socio_results
    )
    
    return comparison


def analyze_administrative_patterns(full_results: Dict, data_full: pd.DataFrame) -> str:
    """Analyze patterns related to administrative factors"""
    
    admin_insights = []
    
    # Year-based clustering patterns
    if 'tahun' in data_full.columns:
        year_cluster_analysis = data_full.groupby(['cluster_full', 'tahun']).size().unstack(fill_value=0)
        
        # Check for year-dominant clusters
        year_dominant_clusters = []
        for cluster_id in year_cluster_analysis.index:
            cluster_row = year_cluster_analysis.loc[cluster_id]
            max_year_pct = (cluster_row.max() / cluster_row.sum()) * 100
            if max_year_pct > 70:
                dominant_year = cluster_row.idxmax()
                year_dominant_clusters.append(f"Cluster {cluster_id} didominasi tahun {dominant_year} ({max_year_pct:.0f}%)")
        
        if year_dominant_clusters:
            admin_insights.append(f"Temporal clustering detected: {'; '.join(year_dominant_clusters[:2])}")
    
    # Pathway (jalur) concentration
    if 'jalur' in data_full.columns:
        jalur_cluster_analysis = data_full.groupby(['cluster_full', 'jalur']).size().unstack(fill_value=0)
        
        jalur_specific_clusters = []
        for cluster_id in jalur_cluster_analysis.index:
            cluster_row = jalur_cluster_analysis.loc[cluster_id]
            max_jalur_pct = (cluster_row.max() / cluster_row.sum()) * 100
            if max_jalur_pct > 60:
                dominant_jalur = cluster_row.idxmax()
                jalur_specific_clusters.append(f"Cluster {cluster_id} specialized in {dominant_jalur}")
        
        if jalur_specific_clusters:
            admin_insights.append(f"Pathway specialization: {'; '.join(jalur_specific_clusters[:2])}")
    
    return '; '.join(admin_insights) if admin_insights else "Administrative patterns are mixed across clusters"


def analyze_socioeconomic_patterns(socio_results: Dict, data_socio: pd.DataFrame) -> str:
    """Analyze inherent socio-economic patterns"""
    
    socio_insights = []
    
    # Income-based clustering
    if 'penghasilan_ortu' in data_socio.columns:
        income_cluster_analysis = data_socio.groupby(['cluster_socio', 'penghasilan_ortu']).size().unstack(fill_value=0)
        
        income_clusters = []
        for cluster_id in income_cluster_analysis.index:
            cluster_row = income_cluster_analysis.loc[cluster_id]
            max_income_pct = (cluster_row.max() / cluster_row.sum()) * 100
            if max_income_pct > 50:
                dominant_income = cluster_row.idxmax()
                income_clusters.append(f"Cluster {cluster_id} concentrated in {dominant_income} income bracket")
        
        if income_clusters:
            socio_insights.append(f"Economic stratification: {'; '.join(income_clusters)}")
    
    # Occupation-based patterns
    occupation_features = ['pekerjaan_ayah', 'pekerjaan_ibu']
    for occ_feature in occupation_features:
        if occ_feature in data_socio.columns:
            occ_cluster_analysis = data_socio.groupby(['cluster_socio', occ_feature]).size().unstack(fill_value=0)
            
            occ_clusters = []
            for cluster_id in occ_cluster_analysis.index:
                cluster_row = occ_cluster_analysis.loc[cluster_id]
                max_occ_pct = (cluster_row.max() / cluster_row.sum()) * 100
                if max_occ_pct > 40:
                    dominant_occ = cluster_row.idxmax()
                    parent_type = "paternal" if "ayah" in occ_feature else "maternal"
                    occ_clusters.append(f"Cluster {cluster_id} shows {parent_type} {dominant_occ} concentration")
            
            if occ_clusters:
                socio_insights.append(f"Occupational clustering: {'; '.join(occ_clusters[:1])}")
            break  # Avoid redundancy
    
    return '; '.join(socio_insights) if socio_insights else "Socio-economic patterns show balanced distribution"


def compare_clustering_effectiveness(full_results: Dict, socio_results: Dict) -> str:
    """Compare the effectiveness of full vs socio-economic clustering"""
    
    full_quality = full_results.get('cluster_quality', {})
    socio_quality = socio_results.get('cluster_quality', {})
    
    effectiveness_analysis = []
    
    # Compare silhouette scores
    full_silhouette = full_quality.get('silhouette_score', 0)
    socio_silhouette = socio_quality.get('silhouette_score', 0)
    
    if isinstance(full_silhouette, (int, float)) and isinstance(socio_silhouette, (int, float)):
        if socio_silhouette > full_silhouette + 0.1:
            effectiveness_analysis.append(f"Socio-economic clustering shows superior separation (Silhouette: {socio_silhouette:.3f} vs {full_silhouette:.3f})")
        elif full_silhouette > socio_silhouette + 0.1:
            effectiveness_analysis.append(f"Full analysis achieves better overall clustering (Silhouette: {full_silhouette:.3f} vs {socio_silhouette:.3f})")
        else:
            effectiveness_analysis.append(f"Both approaches show comparable clustering quality (Silhouette: Full={full_silhouette:.3f}, Socio={socio_silhouette:.3f})")
    
    # Compare number of clusters efficiency
    full_clusters = len(full_results.get('profiles', []))
    socio_clusters = len(socio_results.get('profiles', []))
    
    if full_clusters > socio_clusters:
        effectiveness_analysis.append(f"Full analysis identified more granular patterns ({full_clusters} vs {socio_clusters} clusters)")
    elif socio_clusters > full_clusters:
        effectiveness_analysis.append(f"Socio-economic analysis shows more complex inherent patterns ({socio_clusters} vs {full_clusters} clusters)")
    else:
        effectiveness_analysis.append(f"Both analyses converged on {full_clusters} distinct patterns")
    
    # Algorithm comparison
    full_algorithm = full_quality.get('algorithm_used', 'Unknown')
    socio_algorithm = socio_quality.get('algorithm_used', 'Unknown')
    
    if full_algorithm != socio_algorithm:
        effectiveness_analysis.append(f"Optimal algorithms differ: Full uses {full_algorithm}, Socio-economic uses {socio_algorithm}")
    
    return '; '.join(effectiveness_analysis)


def analyze_feature_importance_differences(full_results: Dict, socio_results: Dict) -> str:
    """Analyze differences in feature importance between analyses"""
    
    differences = []
    
    full_features = full_results.get('feature_count', 'Unknown')
    socio_features = socio_results.get('feature_count', 'Unknown')
    
    if isinstance(full_features, int) and isinstance(socio_features, int):
        feature_reduction = full_features - socio_features
        if feature_reduction > 0:
            reduction_pct = (feature_reduction / full_features) * 100
            differences.append(f"Socio-economic analysis uses {reduction_pct:.0f}% fewer features ({socio_features} vs {full_features})")
    
    differences.append("Full analysis includes administrative factors (tahun, jalur) while socio-economic focuses on inherent characteristics")
    differences.append("Socio-economic analysis likely emphasizes family background, income, and parental occupation patterns")
    
    return '; '.join(differences)


def identify_key_differentiators(full_results: Dict, socio_results: Dict, 
                               data_full: pd.DataFrame, data_socio: pd.DataFrame) -> str:
    """Identify what makes each analysis approach unique"""
    
    differentiators = []
    
    # Cluster size distribution comparison
    full_sizes = [profile.get('size', 0) for profile in full_results.get('profiles', [])]
    socio_sizes = [profile.get('size', 0) for profile in socio_results.get('profiles', [])]
    
    if full_sizes and socio_sizes:
        full_balance = np.std(full_sizes) / np.mean(full_sizes) if np.mean(full_sizes) > 0 else 0
        socio_balance = np.std(socio_sizes) / np.mean(socio_sizes) if np.mean(socio_sizes) > 0 else 0
        
        if full_balance > socio_balance + 0.1:
            differentiators.append("Full analysis shows more uneven cluster distribution (reveals administrative concentration)")
        elif socio_balance > full_balance + 0.1:
            differentiators.append("Socio-economic analysis shows more uneven distribution (reveals economic stratification)")
    
    # Administrative feature presence
    admin_features = ['tahun', 'jalur']
    admin_present = any(feature in data_full.columns for feature in admin_features)
    
    if admin_present:
        differentiators.append("Full analysis captures temporal and pathway-specific patterns")
    
    # Socio-economic feature focus
    socio_features = ['penghasilan_ortu', 'pekerjaan_ayah', 'pekerjaan_ibu']
    socio_present = any(feature in data_socio.columns for feature in socio_features)
    
    if socio_present:
        differentiators.append("Socio-economic analysis reveals family background stratification")
    
    return '; '.join(differentiators) if differentiators else "Both analyses show similar patterns"


def analyze_pattern_overlap(data_full: pd.DataFrame, data_socio: pd.DataFrame) -> str:
    """Analyze overlap between clustering patterns"""
    
    if 'cluster_full' not in data_full.columns or 'cluster_socio' not in data_socio.columns:
        return "Cannot assess pattern overlap - cluster assignments missing"
    
    # Create overlap analysis (assuming same row indices)
    if len(data_full) == len(data_socio):
        try:
            # Create contingency table
            overlap_table = pd.crosstab(data_full['cluster_full'], data_socio['cluster_socio'], margins=True)
            
            # Calculate overlap percentage
            total_samples = len(data_full)
            max_overlap = 0
            
            for full_cluster in overlap_table.index[:-1]:  # Exclude 'All' row
                for socio_cluster in overlap_table.columns[:-1]:  # Exclude 'All' column
                    overlap_count = overlap_table.loc[full_cluster, socio_cluster]
                    overlap_pct = (overlap_count / total_samples) * 100
                    max_overlap = max(max_overlap, overlap_pct)
            
            if max_overlap > 50:
                return f"High pattern overlap detected ({max_overlap:.0f}% maximum cluster correspondence)"
            elif max_overlap > 30:
                return f"Moderate pattern overlap ({max_overlap:.0f}% maximum cluster correspondence)"
            else:
                return f"Low pattern overlap ({max_overlap:.0f}% maximum cluster correspondence) - distinct clustering approaches"
                
        except Exception as e:
            return f"Pattern overlap analysis error: {str(e)}"
    
    return "Cannot assess pattern overlap - different data sizes"


def generate_actionable_insights(full_results: Dict, socio_results: Dict,
                               data_full: pd.DataFrame, data_socio: pd.DataFrame) -> str:
    """Generate actionable insights for stakeholders"""
    
    insights = []
    
    # Program targeting insights
    full_clusters = len(full_results.get('profiles', []))
    socio_clusters = len(socio_results.get('profiles', []))
    
    if full_clusters > socio_clusters:
        insights.append(f"Consider {full_clusters} distinct program tracks addressing both administrative and socio-economic factors")
    else:
        insights.append(f"Focus on {socio_clusters} core socio-economic segments for targeted intervention")
    
    # Administrative efficiency insights
    if 'tahun' in data_full.columns:
        year_variation = data_full['tahun'].nunique()
        if year_variation > 1:
            insights.append(f"Multi-year analysis reveals evolving patterns - consider year-specific strategies")
    
    # Socio-economic targeting insights
    if 'penghasilan_ortu' in data_socio.columns:
        income_diversity = data_socio['penghasilan_ortu'].nunique()
        if income_diversity > 3:
            insights.append(f"High income diversity ({income_diversity} categories) suggests need for income-specific support programs")
    
    # Geographic insights
    geo_features = ['Provinsi Asal', 'provinsi_asal']
    for geo_feature in geo_features:
        if geo_feature in data_full.columns:
            geo_diversity = data_full[geo_feature].nunique()
            if geo_diversity > 10:
                insights.append(f"Geographic diversity ({geo_diversity} regions) requires location-aware program delivery")
            break
    
    return '; '.join(insights[:3]) if insights else "Consider integrated approach combining both administrative and socio-economic factors"


def assess_statistical_significance(full_results: Dict, socio_results: Dict) -> str:
    """Assess statistical significance of clustering differences"""
    
    significance_notes = []
    
    # Sample size assessment
    full_profiles = full_results.get('profiles', [])
    socio_profiles = socio_results.get('profiles', [])
    
    if full_profiles and socio_profiles:
        full_total = sum(profile.get('size', 0) for profile in full_profiles)
        socio_total = sum(profile.get('size', 0) for profile in socio_profiles)
        
        if full_total >= 1000 and socio_total >= 1000:
            significance_notes.append("Large sample sizes provide high statistical power for both analyses")
        elif full_total >= 500 or socio_total >= 500:
            significance_notes.append("Moderate sample sizes provide adequate statistical power")
        else:
            significance_notes.append("Small sample sizes - interpret clustering patterns cautiously")
    
    # Cluster balance assessment
    if full_profiles:
        full_sizes = [profile.get('size', 0) for profile in full_profiles]
        if full_sizes:
            min_cluster_size = min(full_sizes)
            if min_cluster_size < 50:
                significance_notes.append("Some full analysis clusters are small (<50) - results may be less reliable")
    
    if socio_profiles:
        socio_sizes = [profile.get('size', 0) for profile in socio_profiles]
        if socio_sizes:
            min_cluster_size = min(socio_sizes)
            if min_cluster_size < 50:
                significance_notes.append("Some socio-economic clusters are small (<50) - interpret with caution")
    
    return '; '.join(significance_notes) if significance_notes else "Statistical significance assessment requires more detailed analysis"



def main():
    """Orchestrate the complete ML pipeline with dual analysis"""
    print("🚀 Starting Enhanced KIP Kuliah Analysis Pipeline")
    
    try:
        # 1. Data Loading
        base_path = Path(__file__).parent / "Bahan Laporan KIP Kuliah 2022 s.d 2024"
        loader = KIPDataLoader(base_path)
        pendaftar_data, penerima_data = loader.load_all_data()
        
        if pendaftar_data.empty:
            print("❌ No data loaded. Please check data files.")
            return
        
        print(f"📊 Total records loaded: {len(pendaftar_data)}")
        print(f"🗃️ Dataset structure: {pendaftar_data.shape}")
        
        # Use pendaftar data as the main dataset
        combined_data = pendaftar_data
        
        # 2. Full Analysis (Administrative + Socio-economic)
        print("\n" + "="*60)
        print("🔍 FULL ANALYSIS (Administrative + Socio-economic)")
        print("="*60)
        
        preprocessor_full = DataPreprocessor()
        data_full = preprocessor_full.clean_data(combined_data)
        
        # Clustering Analysis
        clustering_config = {'max_k': 10, 'init': 'Huang', 'n_init': 5, 'verbose': 1}
        clustering_analyzer = ClusteringAnalyzer(clustering_config)
        
        # Prepare data for clustering 
        data_for_clustering, categorical_indices = preprocessor_full.prepare_for_clustering(data_full)
        
        # Find optimal clusters and perform clustering
        optimal_k = clustering_analyzer.find_optimal_clusters(data_for_clustering, categorical_indices, max_k=8)
        cluster_assignments = clustering_analyzer.perform_clustering(data_for_clustering, categorical_indices, optimal_k)
        
        # Create cluster results
        cluster_results_full = {
            'assignments': cluster_assignments,
            'profiles': clustering_analyzer.get_cluster_profiles(data_full.assign(cluster=cluster_assignments), cluster_col='cluster'),
            'silhouette_score': clustering_analyzer.get_cluster_quality_metrics().get('silhouette_score', 'N/A'),
            'algorithm': 'K-Prototypes'
        }
        
        # Add cluster assignments to the data
        data_full['cluster_full'] = cluster_assignments
        
        # Classification Training
        classification_config = {'test_size': 0.2, 'random_state': 42}
        trainer_full = ClassificationTrainer(classification_config)
        # Prepare data for classification
        X_full, y_full = preprocessor_full.prepare_for_classification(data_full, target_col='cluster_full')
        
        # Train models
        trained_models_full = trainer_full.train_multiple_models(X_full, y_full)
        
        # Evaluate models
        results_df_full, predictions_full = trainer_full.evaluate_models(trained_models_full, X_full, y_full)
        
        # Print debug info
        print(f"Results DataFrame columns: {results_df_full.columns.tolist()}")
        print(f"Results DataFrame shape: {results_df_full.shape}")
        
        # Create classification results with safe column access
        best_accuracy = results_df_full.iloc[:, 1].max() if len(results_df_full.columns) > 1 else 0.0  # Assume accuracy is second column
        classification_results_full = {
            'accuracy': best_accuracy,
            'models_performance': results_df_full,
            'predictions': predictions_full
        }
        
        # 3. Socio-economic Analysis (Excluding administrative features)
        print("\n" + "="*60)
        print("🏠 SOCIO-ECONOMIC ANALYSIS (Family Background Focus)")
        print("="*60)
        
        # Remove administrative features for socio-economic analysis
        admin_features = ['tahun', 'jalur']
        socio_columns = [col for col in combined_data.columns 
                        if col not in admin_features]
        data_socio_raw = combined_data[socio_columns].copy()
        
        preprocessor_socio = DataPreprocessor()
        data_socio = preprocessor_socio.clean_data(data_socio_raw)
        
        # Clustering Analysis (fewer clusters for focused analysis)
        clustering_analyzer_socio = ClusteringAnalyzer(clustering_config)
        
        # Prepare data for clustering 
        data_for_clustering_socio, categorical_indices_socio = preprocessor_socio.prepare_for_clustering(data_socio)
        
        # Find optimal clusters and perform clustering (fewer clusters for socio-economic focus)
        optimal_k_socio = clustering_analyzer_socio.find_optimal_clusters(data_for_clustering_socio, categorical_indices_socio, max_k=5)
        cluster_assignments_socio = clustering_analyzer_socio.perform_clustering(data_for_clustering_socio, categorical_indices_socio, optimal_k_socio)
        
        # Create cluster results
        cluster_results_socio = {
            'assignments': cluster_assignments_socio,
            'profiles': clustering_analyzer_socio.get_cluster_profiles(data_socio.assign(cluster=cluster_assignments_socio), cluster_col='cluster'),
            'silhouette_score': clustering_analyzer_socio.get_cluster_quality_metrics().get('silhouette_score', 'N/A'),
            'algorithm': 'K-Prototypes'
        }
        
        # Add cluster assignments to the data
        data_socio['cluster_socio'] = cluster_assignments_socio
        
        # Classification Training
        trainer_socio = ClassificationTrainer(classification_config)
        # Prepare data for classification
        X_socio, y_socio = preprocessor_socio.prepare_for_classification(data_socio, target_col='cluster_socio')
        
        # Train models
        trained_models_socio = trainer_socio.train_multiple_models(X_socio, y_socio)
        
        # Evaluate models
        results_df_socio, predictions_socio = trainer_socio.evaluate_models(trained_models_socio, X_socio, y_socio)
        
        # Create classification results with safe column access
        best_accuracy_socio = results_df_socio.iloc[:, 1].max() if len(results_df_socio.columns) > 1 else 0.0
        classification_results_socio = {
            'accuracy': best_accuracy_socio,
            'models_performance': results_df_socio,
            'predictions': predictions_socio
        }
        
        # 4. Comparative Analysis
        print("\n" + "="*60)
        print("🔬 COMPARATIVE ANALYSIS")
        print("="*60)
        
        # Prepare results for comparison
        full_results = {
            'profiles': cluster_results_full['profiles'],
            'assignments': cluster_results_full['assignments'],
            'cluster_quality': {
                'silhouette_score': cluster_results_full.get('silhouette_score', 'N/A'),
                'algorithm_used': cluster_results_full.get('algorithm', 'Unknown')
            },
            'feature_count': len(data_full.columns) - 1,  # Exclude cluster column
            'classification_accuracy': classification_results_full.get('accuracy', 'N/A')
        }
        
        socio_results = {
            'profiles': cluster_results_socio['profiles'],
            'assignments': cluster_results_socio['assignments'],
            'cluster_quality': {
                'silhouette_score': cluster_results_socio.get('silhouette_score', 'N/A'),
                'algorithm_used': cluster_results_socio.get('algorithm', 'Unknown')
            },
            'feature_count': len(data_socio.columns) - 1,  # Exclude cluster column
            'classification_accuracy': classification_results_socio.get('accuracy', 'N/A')
        }
        
        # Generate detailed comparison
        detailed_comparison = generate_detailed_comparison(
            full_results, socio_results, data_full, data_socio
        )
        
        # Print comparison insights
        print("\n📋 Key Findings:")
        print(f"  • Administrative Patterns: {detailed_comparison['administrative_insights']}")
        print(f"  • Socio-economic Patterns: {detailed_comparison['socioeconomic_insights']}")
        print(f"  • Clustering Effectiveness: {detailed_comparison['clustering_effectiveness']}")
        print(f"  • Feature Impact: {detailed_comparison['feature_differences']}")
        print(f"  • Key Differentiators: {detailed_comparison['key_differentiators']}")
        print(f"  • Pattern Overlap: {detailed_comparison['pattern_overlap']}")
        print(f"  • Actionable Insights: {detailed_comparison['actionable_insights']}")
        print(f"  • Statistical Notes: {detailed_comparison['statistical_significance']}")
        
        # 5. Export Results
        print("\n" + "="*60)
        print("📤 EXPORTING COMPREHENSIVE RESULTS")
        print("="*60)
        
        exporter = ResultsExporter()
        
        # Export dual analysis results
        export_data = {
            'full_analysis': {
                'data': data_full,
                'cluster_results': cluster_results_full,
                'classification_results': classification_results_full
            },
            'socio_analysis': {
                'data': data_socio,
                'cluster_results': cluster_results_socio,
                'classification_results': classification_results_socio
            },
            'comparison': detailed_comparison,
            'original_data': combined_data
        }
        
        output_path = exporter.export_dual_analysis(export_data)
        
        # Success Summary
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print(f"📈 Full Analysis: {len(cluster_results_full['profiles'])} clusters identified")
        print(f"🏠 Socio-economic Analysis: {len(cluster_results_socio['profiles'])} clusters identified")
        print(f"🎯 Classification Accuracy: Full={classification_results_full.get('accuracy', 'N/A'):.2%}, Socio={classification_results_socio.get('accuracy', 'N/A'):.2%}")
        print(f"📊 Results exported to: {output_path}")
        print(f"🔍 Total insights generated: {len([insight for insight in detailed_comparison.values() if insight])}")
        
        return {
            'full_results': full_results,
            'socio_results': socio_results,
            'comparison': detailed_comparison,
            'export_path': output_path
        }
        
    except Exception as e:
        print(f"❌ Error in analysis pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Configure matplotlib for better Windows compatibility
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Set console output encoding for Windows
    import sys
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    
    results = main()
    
    if results:
        print("\n🎉 Enhanced ML Analysis Pipeline completed successfully!")
        print("📋 Access detailed insights in the exported Excel file")
    else:
        print("\n❌ Pipeline execution failed. Check error messages above.")
