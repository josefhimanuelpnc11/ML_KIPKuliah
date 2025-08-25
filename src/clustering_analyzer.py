"""
Clustering Analysis Module for KIP Kuliah Analysis
Handles K-Prototypes/K-modes clustering for mixed/categorical data types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """
    Handles clustering analysis with mixed data types
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.cluster_labels = None
        self.optimal_k = None
        
    def find_optimal_clusters(self, data: np.ndarray, categorical_indices: List[int], 
                            max_k: int = None) -> int:
        """
        Find optimal number of clusters using elbow method
        
        Args:
            data: Preprocessed data array
            categorical_indices: Indices of categorical columns
            max_k: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        max_k = max_k or self.config.get('max_k', 10)
        
        logger.info(f"Finding optimal clusters (K=2 to {max_k})...")
        
        costs = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            try:
                # Determine algorithm based on data type
                if len(categorical_indices) == data.shape[1]:
                    # All columns are categorical - use K-modes
                    model = KModes(
                        n_clusters=k,
                        init=self.config.get('init_method', 'Huang'),
                        verbose=0,
                        random_state=self.config.get('random_state', 42),
                        n_init=self.config.get('n_init', 5)
                    )
                    model.fit(data)
                    costs.append(model.cost_)
                elif len(categorical_indices) > 0:
                    # Mixed data - use K-Prototypes
                    model = KPrototypes(
                        n_clusters=k,
                        init=self.config.get('init_method', 'Huang'),
                        verbose=0,
                        random_state=self.config.get('random_state', 42),
                        n_init=self.config.get('n_init', 5)
                    )
                    model.fit(data, categorical=categorical_indices)
                    costs.append(model.cost_)
                else:
                    # All numerical - use K-Means
                    model = KMeans(
                        n_clusters=k,
                        random_state=self.config.get('random_state', 42),
                        n_init=10
                    )
                    model.fit(data)
                    costs.append(model.inertia_)
                
                logger.debug(f"K={k}: Cost={costs[-1]:.2f}")
                
            except Exception as e:
                logger.warning(f"Error with K={k}: {e}")
                costs.append(np.inf)
        
        # Find elbow point using simple method
        optimal_k = self._find_elbow_point(k_range, costs)
        
        # Create visualization
        self._plot_elbow_curve(k_range, costs, optimal_k)
        
        logger.info(f"Optimal number of clusters determined: {optimal_k}")
        self.optimal_k = optimal_k
        
        return optimal_k
    
    def _find_elbow_point(self, k_range: range, costs: List[float]) -> int:
        """Find elbow point in cost curve"""
        
        # Simple elbow detection: find point with maximum distance from line
        valid_costs = [(k, cost) for k, cost in zip(k_range, costs) if not np.isinf(cost)]
        
        if len(valid_costs) < 3:
            return k_range[0]  # Default to minimum K
        
        # Calculate distances from line connecting first and last points
        first_k, first_cost = valid_costs[0]
        last_k, last_cost = valid_costs[-1]
        
        max_distance = 0
        elbow_k = first_k
        
        for k, cost in valid_costs[1:-1]:
            # Distance from point to line
            distance = abs((last_cost - first_cost) * k - (last_k - first_k) * cost + 
                          last_k * first_cost - last_cost * first_k) / \
                      np.sqrt((last_cost - first_cost)**2 + (last_k - first_k)**2)
            
            if distance > max_distance:
                max_distance = distance
                elbow_k = k
        
        return elbow_k
    
    def _find_optimal_k_combined(self, k_range: range, costs: List[float], 
                                silhouette_scores: List[float]) -> int:
        """Find optimal K using combined criteria"""
        
        # Find elbow point
        elbow_k = self._find_elbow_point(k_range, costs)
        
        # Find K with best silhouette score
        if silhouette_scores:
            best_sil_idx = np.argmax(silhouette_scores)
            best_sil_k = list(k_range)[best_sil_idx]
            best_sil_score = silhouette_scores[best_sil_idx]
        else:
            best_sil_k = elbow_k
            best_sil_score = 0
        
        logger.info(f"Elbow method suggests K={elbow_k}")
        logger.info(f"Best silhouette score: K={best_sil_k} (score={best_sil_score:.3f})")
        
        # Prefer silhouette if it's reasonable and different from elbow
        if best_sil_score > 0.3 and abs(best_sil_k - elbow_k) <= 2:
            logger.info(f"Using silhouette-based K={best_sil_k}")
            return best_sil_k
        elif best_sil_score > 0.2:
            logger.info(f"Using silhouette-based K={best_sil_k} (acceptable score)")
            return best_sil_k
        else:
            logger.info(f"Using elbow-based K={elbow_k} (poor silhouette scores)")
            return elbow_k
    
    def _plot_elbow_curve(self, k_range: range, costs: List[float], optimal_k: int):
        """Plot elbow curve for cluster selection"""
        
        plt.figure(figsize=(10, 6))
        valid_indices = [i for i, cost in enumerate(costs) if not np.isinf(cost)]
        valid_k = [list(k_range)[i] for i in valid_indices]
        valid_costs = [costs[i] for i in valid_indices]
        
        plt.plot(valid_k, valid_costs, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                   label=f'Optimal K = {optimal_k}')
        
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Cost/Inertia')
        plt.title('Elbow Method for Optimal Cluster Selection')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.savefig('results/elbow_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_clustering(self, data: np.ndarray, categorical_indices: List[int], 
                          n_clusters: int) -> np.ndarray:
        """
        Perform clustering with specified number of clusters
        
        Args:
            data: Preprocessed data array
            categorical_indices: Indices of categorical columns
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels array
        """
        logger.info(f"Performing clustering with {n_clusters} clusters...")
        
        try:
            # Determine algorithm based on data type
            if len(categorical_indices) == data.shape[1]:
                # All columns are categorical - use K-modes
                self.model = KModes(
                    n_clusters=n_clusters,
                    init=self.config.get('init_method', 'Huang'),
                    verbose=1,
                    random_state=self.config.get('random_state', 42),
                    n_init=self.config.get('n_init', 5)
                )
                self.cluster_labels = self.model.fit_predict(data)
                algorithm_used = 'K-modes'
            elif len(categorical_indices) > 0:
                # Mixed data - use K-Prototypes
                self.model = KPrototypes(
                    n_clusters=n_clusters,
                    init=self.config.get('init_method', 'Huang'),
                    verbose=1,
                    random_state=self.config.get('random_state', 42),
                    n_init=self.config.get('n_init', 5)
                )
                self.cluster_labels = self.model.fit_predict(data, categorical=categorical_indices)
                algorithm_used = 'K-Prototypes'
            else:
                # All numerical - use K-Means
                self.model = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.config.get('random_state', 42),
                    n_init=10
                )
                self.cluster_labels = self.model.fit_predict(data)
                algorithm_used = 'K-Means'
            
            # Log cluster distribution
            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            cluster_dist = dict(zip(unique, counts))
            
            logger.info(f"Clustering completed using {algorithm_used}")
            logger.info(f"Cluster distribution: {cluster_dist}")
            
            # Evaluate clustering quality
            self._evaluate_clustering_quality(data, categorical_indices)
            
            # Check for cluster imbalance
            self._check_cluster_balance(cluster_dist)
            
            return self.cluster_labels
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
    
    def _evaluate_clustering_quality(self, data: np.ndarray, categorical_indices: List[int]):
        """Evaluate clustering quality with multiple metrics"""
        try:
            # For categorical-only data (K-modes), we need special handling
            if len(categorical_indices) == data.shape[1]:
                # All categorical - use matching-based distance
                logger.info("Evaluating K-modes clustering quality...")
                
                # Calculate simple cluster cohesion metric
                cluster_cohesion = self._calculate_categorical_cohesion(data, self.cluster_labels)
                logger.info(f"Cluster Cohesion (categorical): {cluster_cohesion:.4f}")
                
                # Calculate cluster separation
                cluster_separation = self._calculate_categorical_separation(data, self.cluster_labels)
                logger.info(f"Cluster Separation (categorical): {cluster_separation:.4f}")
                
                # Store metrics
                self.quality_metrics = {
                    'cohesion_score': cluster_cohesion,
                    'separation_score': cluster_separation,
                    'algorithm_used': 'K-modes'
                }
                
            else:
                # For mixed data, we need to convert categorical to numerical for sklearn metrics
                data_for_eval = data.copy()
                # Encode categorical variables for evaluation
                for cat_idx in categorical_indices:
                    if cat_idx < data_for_eval.shape[1]:
                        le = LabelEncoder()
                        data_for_eval[:, cat_idx] = le.fit_transform(data_for_eval[:, cat_idx].astype(str))
                
                # Calculate silhouette score
                if len(np.unique(self.cluster_labels)) > 1:
                    sil_score = silhouette_score(data_for_eval, self.cluster_labels)
                    logger.info(f"Silhouette Score: {sil_score:.4f}")
                    
                    # Davies-Bouldin Index (lower is better)
                    db_score = davies_bouldin_score(data_for_eval, self.cluster_labels)
                    logger.info(f"Davies-Bouldin Index: {db_score:.4f}")
                    
                    # Store metrics
                    self.quality_metrics = {
                        'silhouette_score': sil_score,
                        'davies_bouldin_score': db_score,
                        'algorithm_used': 'K-Prototypes' if len(categorical_indices) > 0 else 'K-Means'
                    }
                else:
                    logger.warning("Cannot calculate quality metrics: insufficient clusters")
                    self.quality_metrics = {'algorithm_used': 'K-Prototypes' if len(categorical_indices) > 0 else 'K-Means'}
                    
        except Exception as e:
            logger.warning(f"Error calculating clustering quality metrics: {e}")
            self.quality_metrics = {}
    
    def _calculate_categorical_cohesion(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cohesion for categorical data using matching similarity"""
        total_cohesion = 0.0
        total_pairs = 0
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) <= 1:
                continue
                
            # Calculate pairwise similarity within cluster
            cluster_cohesion = 0.0
            cluster_pairs = 0
            
            for i in range(len(cluster_data)):
                for j in range(i + 1, len(cluster_data)):
                    # Count matching attributes
                    matches = np.sum(cluster_data[i] == cluster_data[j])
                    similarity = matches / len(cluster_data[i])
                    cluster_cohesion += similarity
                    cluster_pairs += 1
            
            if cluster_pairs > 0:
                total_cohesion += cluster_cohesion / cluster_pairs
                total_pairs += 1
        
        return total_cohesion / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_categorical_separation(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate separation for categorical data"""
        cluster_centroids = {}
        
        # Calculate mode for each cluster and feature
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            # Calculate mode for each column
            centroid = []
            for col in range(data.shape[1]):
                values, counts = np.unique(cluster_data[:, col], return_counts=True)
                mode = values[np.argmax(counts)]
                centroid.append(mode)
            cluster_centroids[cluster_id] = np.array(centroid)
        
        # Calculate average dissimilarity between cluster centroids
        total_dissimilarity = 0.0
        total_pairs = 0
        
        cluster_ids = list(cluster_centroids.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                centroid1 = cluster_centroids[cluster_ids[i]]
                centroid2 = cluster_centroids[cluster_ids[j]]
                
                # Count non-matching attributes
                dissimilarity = np.sum(centroid1 != centroid2) / len(centroid1)
                total_dissimilarity += dissimilarity
                total_pairs += 1
        
        return total_dissimilarity / total_pairs if total_pairs > 0 else 0.0
    
    def get_cluster_quality_metrics(self) -> Dict:
        """Return stored clustering quality metrics"""
        return getattr(self, 'quality_metrics', {})
    
    def _check_cluster_balance(self, cluster_dist: Dict[int, int]):
        """Check for cluster imbalance and warn if problematic"""
        total_points = sum(cluster_dist.values())
        percentages = {k: (v / total_points) * 100 for k, v in cluster_dist.items()}
        
        # Check for severe imbalance
        max_percentage = max(percentages.values())
        min_percentage = min(percentages.values())
        
        logger.info("Cluster distribution percentages:")
        for cluster_id, percentage in percentages.items():
            logger.info(f"  Cluster {cluster_id}: {percentage:.1f}%")
        
        # Warning thresholds
        if max_percentage > 80:
            logger.warning(f"⚠️  Severe cluster imbalance detected!")
            logger.warning(f"   Largest cluster: {max_percentage:.1f}% of data")
            logger.warning(f"   This may indicate poor clustering or need for different K")
        
        if min_percentage < 2:
            logger.warning(f"⚠️  Very small clusters detected!")
            logger.warning(f"   Smallest cluster: {min_percentage:.1f}% of data")
            logger.warning(f"   Consider reducing number of clusters")
        
        # Store balance info
        self.cluster_balance = {
            'percentages': percentages,
            'max_percentage': max_percentage,
            'min_percentage': min_percentage,
            'is_balanced': max_percentage < 60 and min_percentage > 5
        }
    
    def get_cluster_profiles(self, df: pd.DataFrame, cluster_col: str = 'cluster') -> pd.DataFrame:
        """
        Create detailed cluster profiles with comprehensive analysis
        
        Args:
            df: Dataframe with cluster assignments
            cluster_col: Name of cluster column
            
        Returns:
            DataFrame with enriched cluster profiles
        """
        logger.info("Creating comprehensive cluster profiles...")
        
        profiles = []
        
        for cluster_id in sorted(df[cluster_col].unique()):
            cluster_data = df[df[cluster_col] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(df)) * 100
            
            profile = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': round(cluster_percentage, 2),
                'interpretation': self._generate_cluster_interpretation(cluster_data, cluster_id),
                'dominant_characteristics': self._identify_dominant_characteristics(cluster_data),
                'unique_patterns': self._identify_unique_patterns(cluster_data, df),
                'socioeconomic_profile': self._analyze_socioeconomic_profile(cluster_data),
                'geographic_concentration': self._analyze_geographic_concentration(cluster_data),
                'temporal_pattern': self._analyze_temporal_pattern(cluster_data)
            }
            
            # Numerical feature statistics
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col in cluster_data.columns and col != cluster_col and not col.startswith('cluster'):
                    if cluster_data[col].notna().any():
                        profile[f'{col}_mean'] = round(cluster_data[col].mean(), 2)
                        profile[f'{col}_std'] = round(cluster_data[col].std(), 2)
                        profile[f'{col}_median'] = round(cluster_data[col].median(), 2)
                        profile[f'{col}_min'] = round(cluster_data[col].min(), 2)
                        profile[f'{col}_max'] = round(cluster_data[col].max(), 2)
                        
                        # Statistical outliers
                        q1 = cluster_data[col].quantile(0.25)
                        q3 = cluster_data[col].quantile(0.75)
                        iqr = q3 - q1
                        outliers = cluster_data[(cluster_data[col] < q1 - 1.5*iqr) | (cluster_data[col] > q3 + 1.5*iqr)]
                        profile[f'{col}_outliers_count'] = len(outliers)
                        profile[f'{col}_outliers_pct'] = round(len(outliers) / len(cluster_data) * 100, 1)
            
            # Categorical feature analysis
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in cluster_data.columns and not col.startswith('cluster'):
                    if cluster_data[col].notna().any():
                        mode_series = cluster_data[col].mode()
                        if len(mode_series) > 0:
                            mode_value = mode_series.iloc[0]
                            mode_freq = (cluster_data[col] == mode_value).sum()
                            mode_pct = (mode_freq / len(cluster_data)) * 100
                            
                            profile[f'{col}_mode'] = mode_value
                            profile[f'{col}_mode_freq'] = mode_freq
                            profile[f'{col}_mode_pct'] = round(mode_pct, 1)
                            profile[f'{col}_diversity'] = cluster_data[col].nunique()
                            profile[f'{col}_diversity_ratio'] = round(cluster_data[col].nunique() / len(cluster_data), 3)
                            
                            # Dominance analysis
                            if mode_pct >= 80:
                                profile[f'{col}_dominance'] = 'Very High'
                            elif mode_pct >= 60:
                                profile[f'{col}_dominance'] = 'High'
                            elif mode_pct >= 40:
                                profile[f'{col}_dominance'] = 'Moderate'
                            else:
                                profile[f'{col}_dominance'] = 'Low'
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        
        # Add cluster comparison metrics
        profiles_df = self._add_cluster_comparison_metrics(profiles_df, df, cluster_col)
        
        # Store interpretations for later use
        cluster_interpretations = {}
        for _, row in profiles_df.iterrows():
            cluster_interpretations[row['cluster_id']] = row['interpretation']
        self.cluster_interpretations = cluster_interpretations
        
        logger.info(f"Created comprehensive profiles for {len(profiles)} clusters")
        
        # Log detailed cluster interpretations
        logger.info("=== COMPREHENSIVE CLUSTER ANALYSIS ===")
        for _, row in profiles_df.iterrows():
            cluster_id = row['cluster_id']
            size = row['size']
            percentage = row['percentage']
            logger.info(f"\nCluster {cluster_id} ({size} records, {percentage:.1f}%):")
            logger.info(f"  Overview: {row['interpretation']}")
            logger.info(f"  Dominant: {row['dominant_characteristics']}")
            logger.info(f"  Unique: {row['unique_patterns']}")
            logger.info(f"  Socioeconomic: {row['socioeconomic_profile']}")
            logger.info(f"  Geographic: {row['geographic_concentration']}")
            logger.info(f"  Temporal: {row['temporal_pattern']}")
        
        return profiles_df
    
    def _identify_dominant_characteristics(self, cluster_data: pd.DataFrame) -> str:
        """Identify the most dominant characteristics of the cluster"""
        
        characteristics = []
        
        # Check for strong patterns in key features
        key_features = ['Jenis Kelamin', 'jalur', 'tahun', 'penghasilan_ortu', 'pekerjaan_ayah', 'pekerjaan_ibu']
        
        for feature in key_features:
            if feature in cluster_data.columns and cluster_data[feature].notna().any():
                value_counts = cluster_data[feature].value_counts()
                if len(value_counts) > 0:
                    dominant_value = value_counts.index[0]
                    percentage = (value_counts.iloc[0] / len(cluster_data)) * 100
                    
                    # Only include if truly dominant (>50%)
                    if percentage > 50:
                        feature_name = feature.replace('_', ' ').replace('Jenis Kelamin', 'Gender').title()
                        characteristics.append(f"{feature_name}: {dominant_value} ({percentage:.0f}%)")
        
        return '; '.join(characteristics[:3]) if characteristics else "Tidak ada karakteristik dominan yang jelas"
    
    def _identify_unique_patterns(self, cluster_data: pd.DataFrame, full_data: pd.DataFrame) -> str:
        """Identify patterns that are unique or distinctive to this cluster"""
        
        unique_patterns = []
        
        # Compare cluster patterns with global patterns
        categorical_cols = cluster_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in cluster_data.columns and col in full_data.columns:
                if cluster_data[col].notna().any() and full_data[col].notna().any():
                    # Calculate cluster frequency vs global frequency
                    cluster_freq = cluster_data[col].value_counts(normalize=True)
                    global_freq = full_data[col].value_counts(normalize=True)
                    
                    for value in cluster_freq.index:
                        if value in global_freq.index:
                            cluster_pct = cluster_freq[value] * 100
                            global_pct = global_freq[value] * 100
                            
                            # Look for over-representation (at least 20% higher than global)
                            if cluster_pct > global_pct + 20 and cluster_pct > 30:
                                feature_name = col.replace('_', ' ').title()
                                over_representation = cluster_pct - global_pct
                                unique_patterns.append(f"{feature_name} '{value}' over-represented (+{over_representation:.0f}% vs global)")
        
        return '; '.join(unique_patterns[:2]) if unique_patterns else "Tidak ada pola unik yang signifikan"
    
    def _analyze_socioeconomic_profile(self, cluster_data: pd.DataFrame) -> str:
        """Analyze socioeconomic characteristics of the cluster"""
        
        socio_characteristics = []
        
        # Analyze parental income
        if 'penghasilan_ortu' in cluster_data.columns and cluster_data['penghasilan_ortu'].notna().any():
            income_dist = cluster_data['penghasilan_ortu'].value_counts()
            dominant_income = income_dist.index[0]
            income_pct = (income_dist.iloc[0] / len(cluster_data)) * 100
            if income_pct > 40:
                socio_characteristics.append(f"Penghasilan ortu: {dominant_income} ({income_pct:.0f}%)")
        
        # Analyze parental occupation
        occupation_features = ['pekerjaan_ayah', 'pekerjaan_ibu']
        for occ_feature in occupation_features:
            if occ_feature in cluster_data.columns and cluster_data[occ_feature].notna().any():
                occ_dist = cluster_data[occ_feature].value_counts()
                if len(occ_dist) > 0:
                    dominant_occ = occ_dist.index[0]
                    occ_pct = (occ_dist.iloc[0] / len(cluster_data)) * 100
                    if occ_pct > 30:
                        parent_type = "Ayah" if "ayah" in occ_feature else "Ibu"
                        socio_characteristics.append(f"{parent_type}: {dominant_occ} ({occ_pct:.0f}%)")
                        break  # Only report one parent to avoid redundancy
        
        return '; '.join(socio_characteristics) if socio_characteristics else "Profil sosio-ekonomi campuran"
    
    def _analyze_geographic_concentration(self, cluster_data: pd.DataFrame) -> str:
        """Analyze geographic concentration patterns"""
        
        geographic_patterns = []
        
        # Check province concentration
        province_features = ['Provinsi Asal', 'provinsi_asal']
        for prov_feature in province_features:
            if prov_feature in cluster_data.columns and cluster_data[prov_feature].notna().any():
                prov_dist = cluster_data[prov_feature].value_counts()
                if len(prov_dist) > 0:
                    top_province = prov_dist.index[0]
                    prov_pct = (prov_dist.iloc[0] / len(cluster_data)) * 100
                    
                    if prov_pct > 50:
                        geographic_patterns.append(f"Konsentrasi tinggi dari {top_province} ({prov_pct:.0f}%)")
                    elif prov_pct > 30:
                        geographic_patterns.append(f"Dominan dari {top_province} ({prov_pct:.0f}%)")
                    else:
                        # Check for geographic diversity
                        num_provinces = prov_dist.nunique() if hasattr(prov_dist, 'nunique') else len(prov_dist)
                        if num_provinces > 10:
                            geographic_patterns.append(f"Tersebar dari {num_provinces} provinsi")
                break
        
        return '; '.join(geographic_patterns) if geographic_patterns else "Distribusi geografis beragam"
    
    def _analyze_temporal_pattern(self, cluster_data: pd.DataFrame) -> str:
        """Analyze temporal patterns in the cluster"""
        
        temporal_patterns = []
        
        if 'tahun' in cluster_data.columns and cluster_data['tahun'].notna().any():
            year_dist = cluster_data['tahun'].value_counts().sort_index()
            
            if len(year_dist) == 1:
                # Single year
                year = year_dist.index[0]
                temporal_patterns.append(f"Eksklusif tahun {year}")
            elif len(year_dist) == 2:
                # Two years
                years = sorted(year_dist.index)
                year1_pct = (year_dist[years[0]] / len(cluster_data)) * 100
                year2_pct = (year_dist[years[1]] / len(cluster_data)) * 100
                
                if year1_pct > 70:
                    temporal_patterns.append(f"Dominan {years[0]} ({year1_pct:.0f}%)")
                elif year2_pct > 70:
                    temporal_patterns.append(f"Dominan {years[1]} ({year2_pct:.0f}%)")
                else:
                    temporal_patterns.append(f"Campuran {years[0]} dan {years[1]}")
            else:
                # Multiple years
                dominant_year = year_dist.index[0]
                dominant_pct = (year_dist.iloc[0] / len(cluster_data)) * 100
                
                if dominant_pct > 50:
                    temporal_patterns.append(f"Dominan {dominant_year} ({dominant_pct:.0f}%)")
                else:
                    temporal_patterns.append(f"Tersebar {year_dist.index.min()}-{year_dist.index.max()}")
        
        return '; '.join(temporal_patterns) if temporal_patterns else "Pola temporal tidak jelas"
    
    def _add_cluster_comparison_metrics(self, profiles_df: pd.DataFrame, full_data: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
        """Add cluster comparison and quality metrics"""
        
        # Calculate cluster similarities and differences
        comparison_metrics = []
        
        for idx, row in profiles_df.iterrows():
            cluster_id = row['cluster_id']
            cluster_data = full_data[full_data[cluster_col] == cluster_id]
            
            # Calculate intra-cluster homogeneity
            homogeneity_scores = []
            categorical_cols = full_data.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if col in cluster_data.columns and not col.startswith('cluster'):
                    if cluster_data[col].notna().any():
                        mode_freq = cluster_data[col].mode()
                        if len(mode_freq) > 0:
                            mode_pct = (cluster_data[col] == mode_freq.iloc[0]).mean()
                            homogeneity_scores.append(mode_pct)
            
            avg_homogeneity = np.mean(homogeneity_scores) if homogeneity_scores else 0
            
            # Cluster quality assessment
            if avg_homogeneity > 0.8:
                quality = "Sangat Homogen"
            elif avg_homogeneity > 0.6:
                quality = "Homogen"
            elif avg_homogeneity > 0.4:
                quality = "Cukup Homogen"
            else:
                quality = "Heterogen"
            
            comparison_metrics.append({
                'cluster_id': cluster_id,
                'homogeneity_score': round(avg_homogeneity, 3),
                'quality_assessment': quality
            })
        
        # Merge with profiles
        comparison_df = pd.DataFrame(comparison_metrics)
        profiles_df = profiles_df.merge(comparison_df, on='cluster_id', how='left')
        
        return profiles_df
    
    def _generate_cluster_interpretation(self, cluster_data: pd.DataFrame, cluster_id: int) -> str:
        """Generate human-readable interpretation of cluster characteristics"""
        
        characteristics = []
        
        # Analyze key socio-economic indicators
        if 'tahun' in cluster_data.columns:
            year_mode = cluster_data['tahun'].mode()
            if len(year_mode) > 0:
                year_dist = cluster_data['tahun'].value_counts(normalize=True)
                dominant_year = year_dist.index[0]
                year_pct = year_dist.iloc[0] * 100
                if year_pct > 50:
                    characteristics.append(f"Dominan {int(dominant_year)} ({year_pct:.0f}%)")
        
        if 'jalur' in cluster_data.columns:
            jalur_mode = cluster_data['jalur'].mode()
            if len(jalur_mode) > 0:
                jalur_dist = cluster_data['jalur'].value_counts(normalize=True)
                dominant_jalur = jalur_dist.index[0]
                jalur_pct = jalur_dist.iloc[0] * 100
                if jalur_pct > 40:
                    characteristics.append(f"Jalur {dominant_jalur} ({jalur_pct:.0f}%)")
        
        if 'Jenis Kelamin' in cluster_data.columns:
            gender_dist = cluster_data['Jenis Kelamin'].value_counts(normalize=True)
            if len(gender_dist) > 0 and gender_dist.iloc[0] > 0.6:
                dominant_gender = gender_dist.index[0]
                gender_pct = gender_dist.iloc[0] * 100
                characteristics.append(f"{dominant_gender} ({gender_pct:.0f}%)")
        
        if 'Kepemilikan Rumah' in cluster_data.columns:
            housing_dist = cluster_data['Kepemilikan Rumah'].value_counts(normalize=True)
            if len(housing_dist) > 0:
                dominant_housing = housing_dist.index[0]
                housing_pct = housing_dist.iloc[0] * 100
                if 'milik' in str(dominant_housing).lower():
                    characteristics.append(f"Rumah milik sendiri ({housing_pct:.0f}%)")
                elif 'sewa' in str(dominant_housing).lower() or 'kontrak' in str(dominant_housing).lower():
                    characteristics.append(f"Rumah sewa/kontrak ({housing_pct:.0f}%)")
        
        if 'Penghasilan Ayah' in cluster_data.columns:
            income_dist = cluster_data['Penghasilan Ayah'].value_counts(normalize=True)
            if len(income_dist) > 0:
                dominant_income = income_dist.index[0]
                income_pct = income_dist.iloc[0] * 100
                if any(term in str(dominant_income).lower() for term in ['rendah', 'kurang', '<', '500']):
                    characteristics.append(f"Penghasilan rendah ({income_pct:.0f}%)")
                elif any(term in str(dominant_income).lower() for term in ['tinggi', '>', '5000', 'juta']):
                    characteristics.append(f"Penghasilan tinggi ({income_pct:.0f}%)")
        
        if 'Status P3KE' in cluster_data.columns:
            p3ke_dist = cluster_data['Status P3KE'].value_counts(normalize=True)
            if len(p3ke_dist) > 0:
                dominant_p3ke = p3ke_dist.index[0]
                p3ke_pct = p3ke_dist.iloc[0] * 100
                if 'ya' in str(dominant_p3ke).lower() or 'iya' in str(dominant_p3ke).lower():
                    characteristics.append(f"P3KE ({p3ke_pct:.0f}%)")
        
        # Generate final interpretation
        if characteristics:
            interpretation = ", ".join(characteristics)
        else:
            interpretation = f"Kelompok campuran {cluster_id + 1}"
        
        return interpretation
    
    def create_cluster_visualizations(self, df: pd.DataFrame, cluster_col: str = 'cluster', 
                                    save_plots: bool = True):
        """Create comprehensive cluster visualizations"""
        
        logger.info("Creating cluster visualizations...")
        
        # 1. Cluster size distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = df[cluster_col].value_counts().sort_index()
        bars = plt.bar(cluster_counts.index, cluster_counts.values, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Records')
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('results/cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Numerical features by cluster
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != cluster_col]
        
        if len(numerical_cols) > 0:
            n_cols = min(4, len(numerical_cols))
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, col in enumerate(numerical_cols[:n_cols]):
                ax = axes[i]
                
                for cluster_id in sorted(df[cluster_col].unique()):
                    cluster_data = df[df[cluster_col] == cluster_id][col]
                    ax.hist(cluster_data, alpha=0.6, label=f'Cluster {cluster_id}', bins=20)
                
                ax.set_title(f'{col} Distribution by Cluster')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('results/numerical_features_by_cluster.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        logger.info("Cluster visualizations completed")
    
    def calculate_cluster_quality_metrics(self, data: np.ndarray) -> Dict:
        """Calculate clustering quality metrics"""
        
        if self.cluster_labels is None:
            logger.warning("No cluster labels available for quality calculation")
            return {}
        
        metrics = {}
        
        try:
            # Silhouette score (works for any clustering algorithm)
            if len(np.unique(self.cluster_labels)) > 1:
                silhouette_avg = silhouette_score(data, self.cluster_labels)
                metrics['silhouette_score'] = silhouette_avg
                
            # Add other metrics if available
            if hasattr(self.model, 'cost_'):
                metrics['cost'] = self.model.cost_
            elif hasattr(self.model, 'inertia_'):
                metrics['inertia'] = self.model.inertia_
                
            logger.info(f"Cluster quality metrics: {metrics}")
            
        except Exception as e:
            logger.warning(f"Could not calculate some quality metrics: {e}")
        
        return metrics
