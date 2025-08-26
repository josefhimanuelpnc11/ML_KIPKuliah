"""
Machine Learning Models for KIP Kuliah Classification
Author: Data Mining Professional
Date: 2025

This module implements and evaluates multiple classification algorithms:
- Random Forest Classifier
- XGBoost Classifier  
- Support Vector Machine (SVM)

Includes comprehensive evaluation metrics, cross-validation,
and model comparison functionality.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class KIPClassificationModels:
    """
    Comprehensive classification models for KIP Kuliah recipient prediction.
    
    Implements multiple algorithms with hyperparameter tuning,
    cross-validation, and detailed evaluation metrics.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the models manager.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.results = {}
        self.feature_importance = {}
        
    def initialize_models(self) -> dict:
        """
        Initialize all classification models with default parameters.
        
        Returns:
            dict: Dictionary of initialized models
        """
        print("🤖 Initializing classification models...")
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            )
        }
        
        print(f"✅ Initialized {len(self.models)} models")
        return self.models
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           cv_folds: int = 5, n_jobs: int = -1, 
                           fast_svm: bool = True, skip_svm: bool = False) -> dict:
        """
        Perform hyperparameter tuning for all models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of CV folds
            n_jobs (int): Number of parallel jobs
            fast_svm (bool): Use faster SVM parameters
            skip_svm (bool): Skip SVM tuning entirely
            
        Returns:
            dict: Best parameters for each model
        """
        print("🔧 Starting hyperparameter tuning...")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            
            'svm': {
                # Ultra-fast SVM for quick testing
                'C': [1] if fast_svm else [0.1, 1, 10],
                'kernel': ['rbf'],
                'gamma': ['scale'] if fast_svm else ['scale', 0.001]
                # Fast: 1 combination, Normal: 6 combinations
            }
        }
        
        # Remove SVM if skipping
        if skip_svm:
            print("⏭️ Skipping SVM tuning (skip_svm=True)")
            param_grids.pop('svm', None)
            models_to_tune = {k: v for k, v in self.models.items() if k != 'svm'}
        else:
            models_to_tune = self.models
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Tune each model
        for model_name, model in models_to_tune.items():
            print(f"🎯 Tuning {model_name}...")
            
            try:
                # Special handling for SVM to make it faster
                if model_name == 'svm':
                    if fast_svm:
                        print(f"   ⚡ Using fast SVM mode (1 combination)")
                    else:
                        print(f"   📊 SVM Grid: {len(param_grids[model_name]['C']) * len(param_grids[model_name]['kernel']) * len(param_grids[model_name]['gamma'])} combinations")
                    print(f"   ⏱️ Estimated time: ~30 seconds (fast) / 2-3 minutes (normal)")
                    
                    # Use fewer CV folds for SVM to speed up
                    svm_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[model_name],
                        cv=svm_cv,
                        scoring='f1',
                        n_jobs=n_jobs,
                        verbose=1  # Show progress for SVM
                    )
                else:
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[model_name],
                        cv=cv,
                        scoring='f1',  # Focus on F1 score for imbalanced data
                        n_jobs=n_jobs,
                        verbose=0
                    )
                
                grid_search.fit(X_train, y_train)
                
                # Store best parameters and model
                self.best_params[model_name] = grid_search.best_params_
                self.models[model_name] = grid_search.best_estimator_
                
                print(f"✅ {model_name} tuned. Best F1: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                print(f"❌ Error tuning {model_name}: {str(e)}")
                # For SVM timeout, use default parameters
                if model_name == 'svm':
                    print(f"   🔧 Using default SVM parameters due to error")
                    default_svm = SVC(C=1, kernel='rbf', gamma='scale', probability=True, random_state=self.random_state)
                    self.models[model_name] = default_svm
                    self.best_params[model_name] = {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'}
                continue
        
        print("✅ Hyperparameter tuning completed!")
        return self.best_params
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train all models with best parameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            dict: Trained models
        """
        print("🏋️ Training all models...")
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            print(f"📚 Training {model_name}...")
            
            try:
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                print(f"✅ {model_name} trained successfully")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        self.models = trained_models
        print("✅ All models trained successfully!")
        return trained_models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                       X_train: pd.DataFrame = None, y_train: pd.Series = None) -> dict:
        """
        Comprehensive evaluation of all models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            X_train (pd.DataFrame): Training features (for additional metrics)
            y_train (pd.Series): Training target (for additional metrics)
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print("📊 Evaluating all models...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"🔍 Evaluating {model_name}...")
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Basic metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary'),
                    'recall': recall_score(y_test, y_pred, average='binary'),
                    'f1_score': f1_score(y_test, y_pred, average='binary'),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                # ROC AUC if probabilities available
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                # Classification report
                metrics['classification_report'] = classification_report(
                    y_test, y_pred, output_dict=True
                )
                
                # Cross-validation scores if training data provided
                if X_train is not None and y_train is not None:
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=5, scoring='f1'
                    )
                    metrics['cv_f1_mean'] = cv_scores.mean()
                    metrics['cv_f1_std'] = cv_scores.std()
                
                # Feature importance (if available)
                importance_df = None
                
                if hasattr(model, 'feature_importances_'):
                    # For Random Forest and XGBoost
                    if X_test.shape[1] == len(model.feature_importances_):
                        importance_df = pd.DataFrame({
                            'feature': X_test.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                elif hasattr(model, 'coef_') and model.coef_ is not None:
                    # For SVM and other linear models
                    if len(model.coef_.shape) == 2 and model.coef_.shape[0] == 1:
                        # Binary classification
                        coefficients = np.abs(model.coef_[0])
                    else:
                        # Multi-class (use absolute values)
                        coefficients = np.abs(model.coef_.flatten())
                    
                    if X_test.shape[1] == len(coefficients):
                        importance_df = pd.DataFrame({
                            'feature': X_test.columns,
                            'importance': coefficients
                        }).sort_values('importance', ascending=False)
                        
                elif model_name == 'svm':
                    # For SVM with RBF kernel, use permutation importance
                    try:
                        print(f"   🔄 Computing permutation importance for {model_name}...")
                        print(f"   ⚠️ Note: SVM RBF kernel feature importance is less interpretable than tree-based models")
                        
                        # Use full test set for more reliable results
                        perm_importance = permutation_importance(
                            model, X_test, y_test, 
                            n_repeats=10, random_state=self.random_state,
                            scoring='f1', n_jobs=-1
                        )
                        
                        importance_df = pd.DataFrame({
                            'feature': X_test.columns,
                            'importance': perm_importance.importances_mean
                        }).sort_values('importance', ascending=False)
                        
                        # Filter out near-zero importance (< 0.001) for cleaner output
                        importance_df = importance_df[importance_df['importance'] >= 0.001]
                        
                    except Exception as e:
                        print(f"   ⚠️ Could not compute permutation importance for {model_name}: {str(e)}")
                
                if importance_df is not None:
                    self.feature_importance[model_name] = importance_df
                    metrics['top_features'] = importance_df.head(10).to_dict('records')
                
                evaluation_results[model_name] = metrics
                print(f"✅ {model_name} evaluated. F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {str(e)}")
                continue
        
        self.results = evaluation_results
        print("✅ All models evaluated successfully!")
        return evaluation_results
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
        """
        Perform cross-validation for all models.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Cross-validation results
        """
        print(f"🔄 Performing {cv_folds}-fold cross-validation...")
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for model_name, model in self.models.items():
            print(f"🎯 Cross-validating {model_name}...")
            
            try:
                model_cv_results = {}
                
                for metric in scoring_metrics:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                    model_cv_results[f'{metric}_mean'] = scores.mean()
                    model_cv_results[f'{metric}_std'] = scores.std()
                    model_cv_results[f'{metric}_scores'] = scores.tolist()
                
                cv_results[model_name] = model_cv_results
                print(f"✅ {model_name} CV F1: {model_cv_results['f1_mean']:.4f} ± {model_cv_results['f1_std']:.4f}")
                
            except Exception as e:
                print(f"❌ Error in CV for {model_name}: {str(e)}")
                continue
        
        print("✅ Cross-validation completed!")
        return cv_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all models performance.
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        print("📊 Comparing model performance...")
        
        if not self.results:
            print("❌ No evaluation results available. Run evaluate_models() first.")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1 Score': metrics.get('f1_score', 0),
                'ROC AUC': metrics.get('roc_auc', 0),
                'CV F1 Mean': metrics.get('cv_f1_mean', 0),
                'CV F1 Std': metrics.get('cv_f1_std', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        # Sort by F1 score
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        print("✅ Model comparison completed!")
        print("\n📊 Model Performance Summary:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """
        Create visualization comparing model performance.
        
        Args:
            save_path (str): Path to save the plot
        """
        print("📈 Creating model comparison visualization...")
        
        if not self.results:
            print("❌ No evaluation results available.")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            values = [self.results[model].get(metric, 0) for model in models]
            colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(models)]
            
            bars = ax.bar(models, values, color=colors, alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, save_path: str = None) -> None:
        """
        Plot confusion matrices for all models.
        
        Args:
            save_path (str): Path to save the plot
        """
        print("🔲 Creating confusion matrices...")
        
        if not self.results:
            print("❌ No evaluation results available.")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = np.array(metrics['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Recipient', 'Recipient'],
                       yticklabels=['Non-Recipient', 'Recipient'],
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model_name: str = None, top_n: int = 15, 
                              save_path: str = None, show_all_models: bool = True) -> None:
        """
        Plot feature importance for specified model, best model, or all models.
        
        Args:
            model_name (str): Name of model to plot (None for best model)
            top_n (int): Number of top features to show
            save_path (str): Path to save the plot
            show_all_models (bool): Show all models with feature importance
        """
        print(f"📊 Plotting feature importance...")
        
        if not self.feature_importance:
            print("❌ No feature importance data available.")
            return
        
        # Get models with feature importance
        available_models = list(self.feature_importance.keys())
        
        if show_all_models and len(available_models) > 1:
            # Plot all models side by side
            n_models = len(available_models)
            fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8))
            
            if n_models == 1:
                axes = [axes]
            
            for idx, model in enumerate(available_models):
                importance_df = self.feature_importance[model].head(top_n)
                
                bars = axes[idx].barh(range(len(importance_df)), importance_df['importance'], 
                                    color=['steelblue', 'lightcoral', 'lightgreen'][idx % 3], alpha=0.7)
                
                axes[idx].set_yticks(range(len(importance_df)))
                axes[idx].set_yticklabels(importance_df['feature'])
                axes[idx].set_xlabel('Feature Importance')
                axes[idx].set_title(f'Top {top_n} Features - {model.replace("_", " ").title()}',
                                   fontweight='bold')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    axes[idx].text(bar.get_width() + max(importance_df['importance']) * 0.01, 
                                 bar.get_y() + bar.get_height()/2,
                                 f'{importance_df.iloc[i]["importance"]:.3f}',
                                 va='center', ha='left', fontsize=8)
                
                axes[idx].invert_yaxis()
            
            plt.suptitle('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
        else:
            # Single model plot
            if model_name is None:
                # Use best model based on F1 score
                best_model = max(self.results.keys(), 
                               key=lambda x: self.results[x].get('f1_score', 0))
                model_name = best_model
            
            if model_name not in self.feature_importance:
                print(f"❌ Feature importance not available for {model_name}")
                return
            
            # Get feature importance data
            importance_df = self.feature_importance[model_name].head(top_n)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            bars = plt.barh(range(len(importance_df)), importance_df['importance'], 
                           color='steelblue', alpha=0.7)
            
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance - {model_name.replace("_", " ").title()}',
                     fontweight='bold')
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{importance_df.iloc[i]["importance"]:.3f}',
                        va='center', ha='left', fontsize=9)
            
            plt.gca().invert_yaxis()
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def print_feature_importance(self, top_n: int = 10) -> None:
        """
        Print feature importance for all models to terminal.
        
        Args:
            top_n (int): Number of top features to show per model
        """
        print("\n" + "="*70)
        print("🎯 FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        if not self.feature_importance:
            print("❌ No feature importance data available.")
            return
        
        for model_name, importance_df in self.feature_importance.items():
            print(f"\n📊 {model_name.replace('_', ' ').title()} - Top {top_n} Features:")
            print("-" * 60)
            
            for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
                importance = row['importance']
                feature = row['feature']
                bar_length = int(importance * 50)  # Scale to 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                print(f"{i:2d}. {feature:<25} │{bar}│ {importance:.4f}")
            
            print()
        
        print("="*70)
    
    def save_models(self, save_dir: str) -> dict:
        """
        Save trained models to disk.
        
        Args:
            save_dir (str): Directory to save models
            
        Returns:
            dict: Saved model paths
        """
        print("💾 Saving trained models...")
        
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f"{model_name}_{timestamp}.pkl"
            filepath = os.path.join(save_dir, filename)
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                saved_paths[model_name] = filepath
                print(f"✅ {model_name} saved to {filepath}")
            except Exception as e:
                print(f"❌ Error saving {model_name}: {str(e)}")
        
        # Save best parameters and results
        metadata = {
            'best_params': self.best_params,
            'results': self.results,
            'feature_importance': {k: v.to_dict('records') for k, v in self.feature_importance.items()},
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(save_dir, f"model_metadata_{timestamp}.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ All models and metadata saved to {save_dir}")
        return saved_paths
    
    def generate_classification_report(self) -> dict:
        """
        Generate comprehensive classification report.
        
        Returns:
            dict: Complete classification report
        """
        print("📋 Generating comprehensive classification report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(self.models.keys()),
            'best_model': None,
            'model_comparison': None,
            'detailed_results': self.results,
            'feature_importance': {k: v.to_dict('records') if isinstance(v, pd.DataFrame) else v 
                                 for k, v in self.feature_importance.items()},
            'best_parameters': self.best_params
        }
        
        # Identify best model
        if self.results:
            best_model = max(self.results.keys(), 
                           key=lambda x: self.results[x].get('f1_score', 0))
            report['best_model'] = {
                'name': best_model,
                'f1_score': self.results[best_model]['f1_score'],
                'accuracy': self.results[best_model]['accuracy'],
                'precision': self.results[best_model]['precision'],
                'recall': self.results[best_model]['recall']
            }
        
        # Model comparison
        report['model_comparison'] = self.compare_models().to_dict('records')
        
        print("✅ Classification report generated!")
        return report

def main():
    """
    Main function for testing the models module.
    """
    print("🧪 Testing classification models module...")
    
    # This would typically be called from main.py with actual data
    print("ℹ️ Models module ready for use!")

if __name__ == "__main__":
    main()
