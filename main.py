import os
import sys
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_exploration import KIPDataExplorer
from src.preprocessing import KIPDataPreprocessor
from src.models import KIPClassificationModels

def setup_directories():
    """Create necessary directories for the project."""
    directories = ['reports', 'models', 'visualizations']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory '{directory}' ready")

def save_report(data: dict, filename: str, reports_dir: str = 'reports'):
    """
    Save report data to JSON file with timestamp.
    
    Args:
        data (dict): Data to save
        filename (str): Base filename
        reports_dir (str): Reports directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(reports_dir, f"{filename}_{timestamp}.json")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"📄 Report saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Error saving report: {str(e)}")
        return None

def main():
    """
    Main pipeline for KIP Kuliah classification system.
    """
    print("="*70)
    print("🎓 KIP KULIAH CLASSIFICATION SYSTEM")
    print("   Socio-Economic Factor Analysis for Educational Support")
    print("="*70)
    
    # Setup
    setup_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    config = {
        'base_path': r"c:\laragon\www\ML_KIPKuliah\Bahan Laporan KIP Kuliah 2022 s.d 2024",
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'feature_selection': True,
        'hyperparameter_tuning': True
    }
    
    print(f"⚙️ Configuration: {config}")
    
    # ===================================================================
    # PHASE 1: DATA EXPLORATION
    # ===================================================================
    print("\n" + "="*50)
    print("📊 PHASE 1: DATA EXPLORATION")
    print("="*50)
    
    try:
        # Initialize explorer
        explorer = KIPDataExplorer(config['base_path'])
        
        # Discover and load data
        files = explorer.discover_files()
        if not files['pendaftar'] and not files['penerima']:
            raise FileNotFoundError("No CSV files found in the specified directory")
        
        data = explorer.load_all_data()
        if not data or data['pendaftar_combined'].empty:
            raise ValueError("No data could be loaded from CSV files")
        
        # Analyze data structure
        columns_analysis = explorer.analyze_columns()
        socioeconomic_features = explorer.identify_socioeconomic_features()
        missing_analysis = explorer.analyze_missing_values()
        
        # Create unified dataset with target variable
        combined_data = explorer.create_target_variable()
        if combined_data.empty:
            raise ValueError("Could not create unified dataset")
        
        # Generate exploration summary
        exploration_summary = explorer.generate_summary_report()
        
        # Save exploration report
        exploration_report_path = save_report(exploration_summary, "data_exploration")
        
        print(f"✅ Data exploration completed successfully!")
        print(f"📊 Total records: {len(combined_data)}")
        print(f"📊 Features identified: {len(socioeconomic_features)}")
        print(f"📊 Recipient rate: {(combined_data['is_kip_recipient'].mean() * 100):.1f}%")
        
    except Exception as e:
        print(f"❌ Error in data exploration: {str(e)}")
        return False
    
    # ===================================================================
    # PHASE 2: DATA PREPROCESSING
    # ===================================================================
    print("\n" + "="*50)
    print("🔧 PHASE 2: DATA PREPROCESSING")
    print("="*50)
    
    try:
        # Initialize preprocessor
        preprocessor = KIPDataPreprocessor(random_state=config['random_state'])
        
        # Run complete preprocessing pipeline
        processed_data = preprocessor.preprocess_pipeline(
            df=combined_data,
            target_col='is_kip_recipient',
            test_size=config['test_size'],
            feature_selection=config['feature_selection']
        )
        
        # Validate preprocessing results
        if not processed_data or processed_data['X_train'].empty:
            raise ValueError("Preprocessing failed to generate valid training data")
        
        # Prepare preprocessing report
        preprocessing_report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'preprocessing_info': processed_data['preprocessing_info'],
            'feature_names': processed_data['feature_names'],
            'engineered_features': preprocessor.engineered_features,
            'data_quality': {
                'train_samples': len(processed_data['X_train']),
                'test_samples': len(processed_data['X_test']),
                'feature_count': len(processed_data['feature_names']),
                'target_distribution': processed_data['preprocessing_info']['class_distribution']
            }
        }
        
        # Save preprocessing report
        preprocessing_report_path = save_report(preprocessing_report, "preprocessing")
        
        print(f"✅ Data preprocessing completed successfully!")
        print(f"🎯 Training samples: {len(processed_data['X_train'])}")
        print(f"🎯 Test samples: {len(processed_data['X_test'])}")
        print(f"🎯 Final features: {len(processed_data['feature_names'])}")
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {str(e)}")
        return False
    
    # ===================================================================
    # PHASE 3: MODEL TRAINING & EVALUATION
    # ===================================================================
    print("\n" + "="*50)
    print("🤖 PHASE 3: MODEL TRAINING & EVALUATION")
    print("="*50)
    
    try:
        # Initialize models manager
        models_manager = KIPClassificationModels(random_state=config['random_state'])
        
        # Initialize models
        models = models_manager.initialize_models()
        print(f"🎯 Initialized models: {list(models.keys())}")
        
        # Hyperparameter tuning (optional)
        if config['hyperparameter_tuning']:
            print("🔧 Starting hyperparameter tuning...")
            best_params = models_manager.tune_hyperparameters(
                processed_data['X_train'], 
                processed_data['y_train'],
                cv_folds=config['cv_folds'],
                fast_svm=True,  # Use fast SVM mode
                skip_svm=False  # Set to True if you want to skip SVM entirely
            )
            print(f"✅ Hyperparameter tuning completed")
        
        # Train models
        trained_models = models_manager.train_models(
            processed_data['X_train'], 
            processed_data['y_train']
        )
        
        if not trained_models:
            raise ValueError("No models were trained successfully")
        
        # Evaluate models
        evaluation_results = models_manager.evaluate_models(
            processed_data['X_test'], 
            processed_data['y_test'],
            processed_data['X_train'], 
            processed_data['y_train']
        )
        
        # Cross-validation
        cv_results = models_manager.cross_validate_models(
            pd.concat([processed_data['X_train'], processed_data['X_test']]),
            pd.concat([processed_data['y_train'], processed_data['y_test']]),
            cv_folds=config['cv_folds']
        )
        
        # Model comparison
        comparison_df = models_manager.compare_models()
        
        print(f"✅ Model training and evaluation completed!")
        print(f"🏆 Models trained: {len(trained_models)}")
        
        # Display best model
        if not comparison_df.empty:
            best_model_name = comparison_df.iloc[0]['Model']
            best_f1 = comparison_df.iloc[0]['F1 Score']
            print(f"🥇 Best model: {best_model_name} (F1: {best_f1:.4f})")
        
    except Exception as e:
        print(f"❌ Error in model training/evaluation: {str(e)}")
        return False
    
    # ===================================================================
    # PHASE 4: VISUALIZATION & REPORTING
    # ===================================================================
    print("\n" + "="*50)
    print("📈 PHASE 4: VISUALIZATION & REPORTING")
    print("="*50)
    
    try:
        # Create visualizations
        viz_dir = "visualizations"
        
        # Model comparison plot
        comparison_plot_path = os.path.join(viz_dir, f"model_comparison_{timestamp}.png")
        models_manager.plot_model_comparison(save_path=comparison_plot_path)
        
        # Confusion matrices
        confusion_plot_path = os.path.join(viz_dir, f"confusion_matrices_{timestamp}.png")
        models_manager.plot_confusion_matrices(save_path=confusion_plot_path)
        
        # Feature importance (for all models)
        if models_manager.feature_importance:
            feature_plot_path = os.path.join(viz_dir, f"feature_importance_{timestamp}.png")
            models_manager.plot_feature_importance(save_path=feature_plot_path, show_all_models=True)
            
            # Print feature importance to terminal
            models_manager.print_feature_importance(top_n=10)
        
        print(f"✅ Visualizations created in '{viz_dir}' directory")
        
    except Exception as e:
        print(f"⚠️ Warning: Some visualizations could not be created: {str(e)}")
    
    # ===================================================================
    # PHASE 5: SAVE MODELS & FINAL REPORT
    # ===================================================================
    print("\n" + "="*50)
    print("💾 PHASE 5: SAVE MODELS & FINAL REPORT")
    print("="*50)
    
    try:
        # Save trained models
        models_dir = "models"
        saved_model_paths = models_manager.save_models(models_dir)
        print(f"✅ Models saved to '{models_dir}' directory")
        
        # Generate comprehensive final report
        final_report = {
            'project_info': {
                'title': 'KIP Kuliah Classification System',
                'description': 'Socio-Economic Factor Analysis for Educational Support',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            },
            'configuration': config,
            'data_summary': exploration_summary,
            'preprocessing_summary': preprocessing_report,
            'model_results': {
                'evaluation_results': evaluation_results,
                'cross_validation_results': cv_results,
                'model_comparison': comparison_df.to_dict('records'),
                'best_parameters': models_manager.best_params if hasattr(models_manager, 'best_params') else {}
            },
            'feature_analysis': {
                'total_features': len(processed_data['feature_names']),
                'engineered_features': len(preprocessor.engineered_features),
                'selected_features': processed_data['feature_names'],
                'feature_importance': {k: v.to_dict('records') if hasattr(v, 'to_dict') else v 
                                     for k, v in models_manager.feature_importance.items()}
            },
            'file_paths': {
                'saved_models': saved_model_paths,
                'reports': {
                    'exploration': exploration_report_path,
                    'preprocessing': preprocessing_report_path
                },
                'visualizations': {
                    'model_comparison': comparison_plot_path,
                    'confusion_matrices': confusion_plot_path,
                    'feature_importance': feature_plot_path if 'feature_plot_path' in locals() else None
                }
            }
        }
        
        # Save final comprehensive report
        final_report_path = save_report(final_report, "final_classification_report")
        
        print(f"✅ Final report saved: {final_report_path}")
        
    except Exception as e:
        print(f"❌ Error saving final report: {str(e)}")
        return False
    
    # ===================================================================
    # COMPLETION SUMMARY
    # ===================================================================
    print("\n" + "="*70)
    print("🎉 KIP KULIAH CLASSIFICATION SYSTEM - COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 SUMMARY STATISTICS:")
    print(f"   • Total records processed: {len(combined_data):,}")
    print(f"   • Features engineered: {len(preprocessor.engineered_features)}")
    print(f"   • Final features used: {len(processed_data['feature_names'])}")
    print(f"   • Models trained: {len(trained_models)}")
    print(f"   • Best model F1 score: {best_f1:.4f}" if 'best_f1' in locals() else "")
    
    print("\n📁 OUTPUT FILES:")
    print(f"   • Models: {models_dir}/")
    print(f"   • Reports: reports/")
    print(f"   • Visualizations: {viz_dir}/")
    
    print("\n🎯 KEY INSIGHTS:")
    if not comparison_df.empty:
        print(f"   • Best performing model: {best_model_name}")
        print(f"   • Classification accuracy: {comparison_df.iloc[0]['Accuracy']:.1%}")
        print(f"   • Precision: {comparison_df.iloc[0]['Precision']:.1%}")
        print(f"   • Recall: {comparison_df.iloc[0]['Recall']:.1%}")
    
    print("\n✅ System ready for production deployment!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Pipeline execution failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n🚀 Pipeline execution completed successfully!")
