"""
Modeling Utilities for KIP Kuliah Analysis
Provides helper functions for clustering, classification, and model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class ClusteringAnalyzer:
    """
    Utility class for clustering analysis
    """
    
    def __init__(self, model, model_type='K-Prototypes'):
        self.model = model
        self.model_type = model_type
        self.cluster_labels = None
        
    def fit_predict(self, data, categorical_indices=None):
        """
        Fit clustering model and predict labels
        """
        if self.model_type == 'K-Prototypes' and categorical_indices is not None:
            self.cluster_labels = self.model.fit_predict(data, categorical=categorical_indices)
        else:
            self.cluster_labels = self.model.fit_predict(data)
        
        return self.cluster_labels
    
    def create_cluster_profiles(self, df, numerical_cols, categorical_cols, cluster_col='cluster'):
        """
        Create detailed cluster profiles
        """
        profiles = {}
        
        for cluster_id in sorted(df[cluster_col].unique()):
            cluster_data = df[df[cluster_col] == cluster_id]
            profile = {'size': len(cluster_data)}
            
            # Numerical features statistics
            if numerical_cols:
                available_num_cols = [col for col in numerical_cols if col in cluster_data.columns]
                if available_num_cols:
                    num_stats = cluster_data[available_num_cols].describe()
                    profile['numerical'] = num_stats
            
            # Categorical features mode
            if categorical_cols:
                cat_stats = {}
                for col in categorical_cols:
                    if col in cluster_data.columns:
                        mode_info = cluster_data[col].value_counts()
                        if len(mode_info) > 0:
                            cat_stats[col] = {
                                'most_common': mode_info.index[0],
                                'frequency': mode_info.iloc[0],
                                'percentage': (mode_info.iloc[0] / len(cluster_data) * 100)
                            }
                profile['categorical'] = cat_stats
            
            profiles[f'Cluster_{cluster_id}'] = profile
        
        return profiles
    
    def plot_cluster_distribution(self, cluster_labels):
        """
        Plot cluster size distribution
        """
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(cluster_counts.index, cluster_counts.values)
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Records')
        
        # Add value labels on bars
        for i, v in enumerate(cluster_counts.values):
            plt.text(cluster_counts.index[i], v + 0.1, str(v), ha='center')
        
        plt.show()


class ModelComparator:
    """
    Utility class for comparing multiple classification models
    """
    
    def __init__(self):
        self.trained_models = {}
        self.results = None
        
    def train_models(self, models_config, X_train, y_train, cv=3):
        """
        Train multiple models with hyperparameter tuning
        """
        for model_name, config in models_config.items():
            print(f"Training {model_name}...")
            
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Store results
            self.trained_models[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'grid_search': grid_search
            }
            
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        """
        results = []
        predictions = {}
        
        for model_name, model_info in self.trained_models.items():
            model = model_info['model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Best_CV_Score': model_info['best_score']
            })
        
        self.results = pd.DataFrame(results)
        return self.results, predictions
    
    def plot_model_comparison(self):
        """
        Plot model performance comparison
        """
        if self.results is None:
            print("No results to plot. Run evaluate_models first.")
            return
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(self.results['Model'], self.results[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, feature_names):
        """
        Extract feature importance from trained models
        """
        importance_data = {}
        
        for model_name, model_info in self.trained_models.items():
            model = model_info['model']
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_data[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importance_data[model_name] = np.abs(model.coef_[0])
            else:
                # Default to zeros
                importance_data[model_name] = np.zeros(len(feature_names))
        
        return pd.DataFrame(importance_data, index=feature_names)


class ResultsExporter:
    """
    Utility class for exporting analysis results
    """
    
    def __init__(self, output_folder='../results'):
        self.output_folder = output_folder
        
    def export_to_excel(self, data_dict, filename):
        """
        Export multiple dataframes to Excel with different sheets
        """
        filepath = f'{self.output_folder}/{filename}'
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        
        return filepath
    
    def export_to_csv(self, data_dict, folder_name):
        """
        Export dataframes to separate CSV files
        """
        import os
        csv_folder = f'{self.output_folder}/{folder_name}'
        os.makedirs(csv_folder, exist_ok=True)
        
        exported_files = []
        for name, df in data_dict.items():
            filepath = f'{csv_folder}/{name}.csv'
            df.to_csv(filepath, index=True)
            exported_files.append(filepath)
        
        return exported_files
    
    def create_summary_report(self, analysis_info):
        """
        Create a text summary report
        """
        from datetime import datetime
        
        report = f"""
# LAPORAN ANALISIS KIP KULIAH
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## DATASET OVERVIEW
- Total Records: {analysis_info.get('total_records', 'N/A')}
- Number of Clusters: {analysis_info.get('n_clusters', 'N/A')}
- Features Used: {analysis_info.get('n_features', 'N/A')}

## CLUSTERING RESULTS
"""
        
        if 'cluster_distribution' in analysis_info:
            for cluster, count in analysis_info['cluster_distribution'].items():
                percentage = analysis_info['cluster_percentages'].get(cluster, 0)
                report += f"- Cluster {cluster}: {count} records ({percentage:.1f}%)\n"
        
        if 'best_model' in analysis_info:
            report += f"\n## BEST MODEL\n"
            report += f"- Model: {analysis_info['best_model']['name']}\n"
            report += f"- Accuracy: {analysis_info['best_model']['accuracy']:.4f}\n"
            report += f"- F1-Score: {analysis_info['best_model']['f1']:.4f}\n"
        
        if 'top_features' in analysis_info:
            report += f"\n## TOP FEATURES\n"
            for i, (feature, importance) in enumerate(analysis_info['top_features'], 1):
                report += f"{i}. {feature}: {importance:.4f}\n"
        
        return report


def plot_confusion_matrices(model_predictions, y_test):
    """
    Plot confusion matrices for multiple models
    """
    n_models = len(model_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, y_pred) in enumerate(model_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=axes[i], cbar=True)
        axes[i].set_title(f'{model_name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()


def find_optimal_clusters(data, categorical_indices, max_k=10, algorithm='kprototypes'):
    """
    Find optimal number of clusters using elbow method
    """
    from kmodes.kprototypes import KPrototypes
    from sklearn.cluster import KMeans
    
    costs = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        try:
            if algorithm == 'kprototypes' and len(categorical_indices) > 0:
                model = KPrototypes(n_clusters=k, init='Huang', verbose=0, random_state=42)
                model.fit(data, categorical=categorical_indices)
                costs.append(model.cost_)
            else:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                model.fit(data)
                costs.append(model.inertia_)
        except Exception as e:
            print(f"Error with K={k}: {e}")
            costs.append(np.inf)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, costs, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Cost/Inertia')
    plt.title(f'{algorithm.title()} Elbow Method')
    plt.grid(True)
    plt.show()
    
    return K_range, costs
