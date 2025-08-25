"""
Classification Training Module for KIP Kuliah Analysis
Handles multiple classification algorithms and model comparison
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """
    Handles training and evaluation of multiple classification models
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.trained_models = {}
        self.results = None
        self.feature_importance = None
        
    def train_multiple_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train multiple classification models with hyperparameter tuning
        
        Args:
            X: Features dataframe
            y: Target series
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training multiple classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42),
            stratify=y
        )
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Define models and their hyperparameters
        models_config = self._get_models_config()
        
        # Train each model
        for model_name, config in models_config.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Grid search for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=config['model'],
                    param_grid=config['params'],
                    cv=self.config.get('cv_folds', 3),
                    scoring=self.config.get('scoring', 'accuracy'),
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
                
                logger.info(f"{model_name} - Best CV score: {grid_search.best_score_:.4f}")
                logger.debug(f"{model_name} - Best params: {grid_search.best_params_}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        logger.info(f"Successfully trained {len(self.trained_models)} models")
        return self.trained_models
    
    def _get_models_config(self) -> Dict:
        """Get model configurations"""
        
        models_config = {}
        
        # Random Forest
        if 'RandomForest' in self.config.get('models', {}):
            models_config['Random Forest'] = {
                'model': RandomForestClassifier(random_state=self.config.get('random_state', 42)),
                'params': self.config['models']['RandomForest']
            }
        
        # XGBoost
        if 'XGBoost' in self.config.get('models', {}):
            models_config['XGBoost'] = {
                'model': xgb.XGBClassifier(
                    random_state=self.config.get('random_state', 42),
                    eval_metric='logloss'
                ),
                'params': self.config['models']['XGBoost']
            }
        
        # SVM
        if 'SVM' in self.config.get('models', {}):
            models_config['SVM'] = {
                'model': SVC(random_state=self.config.get('random_state', 42)),
                'params': self.config['models']['SVM']
            }
        
        return models_config
    
    def evaluate_models(self, trained_models: Dict, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict]:
        """
        Evaluate all trained models
        
        Args:
            trained_models: Dictionary of trained models
            X: Features for evaluation
            y: True labels
            
        Returns:
            Tuple of (results_df, predictions_dict)
        """
        logger.info("Evaluating models...")
        
        results = []
        predictions = {}
        
        for model_name, model_info in trained_models.items():
            try:
                model = model_info['model']
                
                # Make predictions
                y_pred = model.predict(X)
                predictions[model_name] = y_pred
                
                # Calculate metrics
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Best_CV_Score': model_info.get('best_score', np.nan)
                })
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        self.results = pd.DataFrame(results)
        
        # Create detailed classification reports
        self._create_classification_reports(predictions, y)
        
        return self.results, predictions
    
    def _create_classification_reports(self, predictions: Dict, y_true: pd.Series):
        """Create detailed classification reports for each model"""
        
        logger.info("Creating detailed classification reports...")
        
        for model_name, y_pred in predictions.items():
            try:
                report = classification_report(y_true, y_pred, output_dict=True)
                logger.info(f"\n{model_name} Classification Report:")
                logger.info(f"Accuracy: {report['accuracy']:.4f}")
                logger.info(f"Macro Avg - Precision: {report['macro avg']['precision']:.4f}")
                logger.info(f"Macro Avg - Recall: {report['macro avg']['recall']:.4f}")
                logger.info(f"Macro Avg - F1-Score: {report['macro avg']['f1-score']:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not create classification report for {model_name}: {e}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from trained models with encoding context
        
        Returns:
            DataFrame with feature importance for each model plus metadata
        """
        logger.info("Extracting feature importance with encoding context...")
        
        if not self.trained_models:
            logger.warning("No trained models available for feature importance")
            return pd.DataFrame()
        
        importance_data = {}
        feature_names = None
        
        for model_name, model_info in self.trained_models.items():
            model = model_info['model']
            
            if feature_names is None and hasattr(self, 'X_test'):
                feature_names = self.X_test.columns.tolist()
            
            try:
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models (Random Forest, XGBoost)
                    importance_data[model_name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models (SVM with linear kernel)
                    if len(model.coef_.shape) > 1:
                        # Multi-class: use mean of absolute coefficients
                        importance_data[model_name] = np.mean(np.abs(model.coef_), axis=0)
                    else:
                        importance_data[model_name] = np.abs(model.coef_[0])
                else:
                    logger.warning(f"No feature importance available for {model_name}")
                    importance_data[model_name] = np.zeros(len(feature_names)) if feature_names else []
                    
            except Exception as e:
                logger.warning(f"Error extracting feature importance for {model_name}: {e}")
                importance_data[model_name] = np.zeros(len(feature_names)) if feature_names else []
        
        if not importance_data or not feature_names:
            return pd.DataFrame()
        
        # Create base feature importance DataFrame
        self.feature_importance = pd.DataFrame(importance_data, index=feature_names)
        
        # Add encoding context and metadata
        self.feature_importance = self._add_feature_context(self.feature_importance)
        
        # Create feature importance visualization
        self._plot_feature_importance()
        
        return self.feature_importance
    
    def _add_feature_context(self, importance_df: pd.DataFrame) -> pd.DataFrame:
        """Add encoding context and metadata to feature importance"""
        
        # Add average importance
        importance_df['Average'] = importance_df.mean(axis=1)
        
        # Add context columns
        context_data = []
        for feature_name in importance_df.index:
            context = self._get_feature_context(feature_name)
            context_data.append(context)
        
        # Create context DataFrame
        context_df = pd.DataFrame(context_data, index=importance_df.index)
        
        # Combine with importance data
        result_df = pd.concat([importance_df, context_df], axis=1)
        
        # Sort by average importance
        result_df = result_df.sort_values('Average', ascending=False)
        
        return result_df
    
    def _get_feature_context(self, feature_name: str) -> dict:
        """Get context information for a feature"""
        
        context = {
            'Display_Name': feature_name,
            'Feature_Type': 'Numerical',
            'Original_Column': feature_name,
            'Encoding_Method': 'None',
            'Description': 'Original numerical feature'
        }
        
        # Handle one-hot encoded categorical features
        if '_' in feature_name and not self._is_numerical_feature(feature_name):
            parts = feature_name.split('_', 1)  # Split only on first underscore
            original_col = parts[0]
            category_value = parts[1] if len(parts) > 1 else ''
            
            # Specific handling for different feature types
            if any(keyword in feature_name.lower() for keyword in ['utbk', 'snbt', 'snbp', 'sbmpn']):
                context.update({
                    'Display_Name': f"Jalur_Pendaftaran: {category_value}",
                    'Feature_Type': 'Categorical (One-Hot)',
                    'Original_Column': original_col,
                    'Encoding_Method': 'One-Hot Encoding',
                    'Description': f'Indikator jalur pendaftaran: {category_value}'
                })
            elif 'jenis kelamin' in feature_name.lower():
                context.update({
                    'Display_Name': f"Jenis_Kelamin: {category_value}",
                    'Feature_Type': 'Categorical (One-Hot)',
                    'Original_Column': 'Jenis Kelamin',
                    'Encoding_Method': 'One-Hot Encoding',
                    'Description': f'Indikator jenis kelamin: {category_value}'
                })
            else:
                context.update({
                    'Display_Name': f"{original_col}: {category_value}",
                    'Feature_Type': 'Categorical (One-Hot)',
                    'Original_Column': original_col,
                    'Encoding_Method': 'One-Hot Encoding',
                    'Description': f'Kategori {original_col}: {category_value}'
                })
        
        # Handle label encoded features
        elif feature_name.lower() in ['status dtks', 'kepemilikan rumah', 'alamat tinggal']:
            context.update({
                'Feature_Type': 'Categorical (Label)',
                'Encoding_Method': 'Label Encoding',
                'Description': f'Kategori terenkode: {feature_name}'
            })
        
        return context
    
    def _is_numerical_feature(self, feature_name: str) -> bool:
        """Check if feature name represents a numerical feature"""
        numerical_keywords = ['luas', 'tanah', 'bangunan', 'tahun', 'umur', 'nilai', 'skor']
        return any(keyword in feature_name.lower() for keyword in numerical_keywords)
    
    def _plot_feature_importance(self):
        """Plot feature importance for all models"""
        
        try:
            if self.feature_importance is None or self.feature_importance.empty:
                logger.warning("No feature importance data available for plotting")
                return
            
            # Filter out metadata columns to get only model columns
            metadata_cols = ['Display_Name', 'Feature_Type', 'Original_Column', 'Encoding_Method', 'Description']
            model_columns = [col for col in self.feature_importance.columns if col not in metadata_cols]
            
            if not model_columns:
                logger.warning("No model importance columns found for plotting")
                return
            
            n_models = len(model_columns)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
            
            if n_models == 1:
                axes = [axes]
            
            for i, model_name in enumerate(model_columns):
                # Convert to numeric to handle any non-numeric values
                importance_values = pd.to_numeric(self.feature_importance[model_name], errors='coerce')
                
                # Get top 10 features
                top_features = importance_values.nlargest(10)
                
                # Use display names if available
                display_names = []
                for idx in top_features.index:
                    if 'Display_Name' in self.feature_importance.columns:
                        display_name = self.feature_importance.loc[idx, 'Display_Name']
                        if pd.notna(display_name) and display_name != '':
                            display_names.append(display_name)
                        else:
                            display_names.append(idx)
                    else:
                        display_names.append(idx)
                
                # Create horizontal bar plot
                axes[i].barh(range(len(top_features)), top_features.values, alpha=0.8)
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(display_names, fontsize=8)
                axes[i].set_xlabel('Importance')
                axes[i].set_title(f'{model_name}\nTop 10 Features')
                axes[i].invert_yaxis()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Ensure results directory exists
            Path('results').mkdir(exist_ok=True)
            
            plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Feature importance plot saved to results/feature_importance.png")
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            # Continue without raising error
    
    def create_model_comparison_plots(self):
        """Create comprehensive model comparison visualizations"""
        
        if self.results is None or self.results.empty:
            logger.warning("No results available for plotting")
            return
        
        logger.info("Creating model comparison plots...")
        
        # 1. Performance metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(self.results['Model'], self.results[metric], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Overall comparison
        plt.figure(figsize=(12, 8))
        x = np.arange(len(self.results))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, self.results[metric], width, label=metric, alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*1.5, self.results['Model'])
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/model_comparison_grouped.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_confusion_matrices(self, predictions: Dict, y_true: pd.Series):
        """Create confusion matrices for all models"""
        
        logger.info("Creating confusion matrices...")
        
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=True)
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_best_model(self) -> Tuple[str, object]:
        """
        Get the best performing model
        
        Returns:
            Tuple of (model_name, model_object)
        """
        if self.results is None or self.results.empty:
            logger.warning("No results available to determine best model")
            return None, None
        
        best_idx = self.results['Accuracy'].idxmax()
        best_model_name = self.results.loc[best_idx, 'Model']
        best_model = self.trained_models[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name} (Accuracy: {self.results.loc[best_idx, 'Accuracy']:.4f})")
        
        return best_model_name, best_model
