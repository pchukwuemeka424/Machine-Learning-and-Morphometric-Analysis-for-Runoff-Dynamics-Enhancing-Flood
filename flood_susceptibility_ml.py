#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flood Susceptibility Modeling using Morphometric Parameters and Machine Learning
Author: Desmond R. Eteh et al.
Enhanced GitHub-ready Python Script with Comprehensive Features
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create output directory if it doesn't exist
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """
    Load and prepare dataset for modeling.
    First tries to load real data from CSV, else generates synthetic data.
    """
    data_file = 'morphometric_ml_dataset.csv'
    
    if os.path.exists(data_file):
        print(f"Loading data from {data_file}...")
        try:
            df = pd.read_csv(data_file)
            
            # Check if required columns exist
            if 'Flood' in df.columns:
                X = df.drop(['Flood', 'ID'], axis=1, errors='ignore')
                y = df['Flood'].astype(int)
                print(f"Successfully loaded real data: {len(X)} samples, {X.shape[1]} features")
                return X, y
        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Generating synthetic dataset instead...")
    
    print("Generating synthetic morphometric dataset...")
    return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic morphometric dataset for demonstration purposes."""
    n_samples = 1000
    
    X = pd.DataFrame({
        'Dd': np.random.uniform(1.5, 4.0, n_samples),     # Drainage density
        'Rh': np.random.uniform(5, 25, n_samples),       # Relief ratio
        'If': np.random.uniform(10, 50, n_samples),      # Infiltration number
        'Rbm': np.random.uniform(1.5, 4.5, n_samples),   # Ruggedness number
        'Fs': np.random.uniform(3, 12, n_samples),       # Form factor
        'Ff': np.random.uniform(0.5, 1.5, n_samples),    # Circularity ratio
        'Hh': np.random.uniform(100, 500, n_samples),    # Hypsometric integral
        'Slope': np.random.uniform(0, 30, n_samples)     # Average slope
    })
    
    # Complex target function for more realistic flood susceptibility
    flood_prob = (
        0.3 * X['Dd'] +
        0.2 * X['If'] +
        0.2 * X['Rh'] +
        0.15 * X['Slope'] +
        0.15 * X['Hh']
    )
    flood_threshold = np.percentile(flood_prob, 70)
    y = (flood_prob > flood_threshold).astype(int)
    
    print(f"Synthetic data generated: {len(X)} samples, {X.shape[1]} features")
    print(f"Flood-prone samples: {np.sum(y)}, Non-flood-prone: {len(y) - np.sum(y)}")
    
    return X, y

def preprocess_data(X, y):
    """
    Preprocess data: handle missing values, scale features, split into train/test.
    """
    # Check for missing values
    if X.isnull().sum().sum() > 0:
        print("Handling missing values...")
        X = X.fillna(X.mean())
    
    # Split into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    print(f"Data split: {len(X_train)} training, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test

def create_models():
    """
    Create a dictionary of models with optimized hyperparameters.
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
            subsample=0.8,
            random_state=RANDOM_SEED
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma=0.1,
            probability=True,
            random_state=RANDOM_SEED
        )
    }
    
    return models

def train_models(models, X_train, y_train):
    """
    Train all models and return trained instances.
    """
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with scaling for SVM
        if name == 'SVM':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            trained_models[name] = pipeline
        else:
            model.fit(X_train, y_train)
            trained_models[name] = model
    
    return trained_models

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model with comprehensive metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    # Print detailed classification report
    print(f"\n{model_name} Evaluation:")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred, y_prob

def plot_confusion_matrices(trained_models, X_test, y_test):
    """Plot confusion matrices for all trained models."""
    n_models = len(trained_models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axs = axs.flatten()
    
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            ax=axs[i],
            cmap='Blues',
            cbar=False
        )
        
        axs[i].set_title(name)
        axs[i].set_xlabel('Predicted')
        axs[i].set_ylabel('Actual')
    
    # Hide unused axes
    for i in range(len(trained_models), len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrices saved to {OUTPUT_DIR}/confusion_matrices.png")

def plot_roc_curves(trained_models, X_test, y_test):
    """Plot ROC curves for all trained models."""
    plt.figure(figsize=(12, 8))
    
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.plot(
            fpr,
            tpr,
            label=f'{name} (AUC = {auc:.3f})',
            linewidth=2
        )
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {OUTPUT_DIR}/roc_curves.png")

def plot_feature_importance(model, X, model_name, output_dir):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
    else:
        return
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    sorted_features = X.columns[indices]
    sorted_importances = importances[indices]
    
    plt.bar(range(X.shape[1]), sorted_importances, align='center')
    plt.title(f'{model_name} Feature Importance')
    plt.xticks(range(X.shape[1]), sorted_features, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png'), dpi=300)

def perform_shap_analysis(model, X_test, model_name):
    """Perform SHAP analysis for model interpretation."""
    try:
        print(f"\nPerforming SHAP analysis for {model_name}...")
        
        if hasattr(model, 'named_steps'):
            explainer = shap.Explainer(model.named_steps['classifier'])
            shap_values = explainer(model.named_steps['scaler'].transform(X_test))
        else:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
        
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ", "_")}_shap_summary.png'), dpi=300, bbox_inches='tight')
        
        # Beeswarm plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="beeswarm", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ", "_")}_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
        
        print(f"SHAP analysis completed for {model_name}")
    except Exception as e:
        print(f"SHAP analysis failed for {model_name}: {e}")

def create_comparison_table(metrics_dict):
    """Create a comparison table of all model metrics."""
    df = pd.DataFrame(metrics_dict).T
    df = df.round(3)
    df = df.sort_values('ROC-AUC', ascending=False)
    
    print("\n" + "="*50)
    print("Model Performance Comparison")
    print("="*50)
    print(df.to_string())
    
    # Save to CSV
    df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))
    print(f"\nComparison table saved to {OUTPUT_DIR}/model_comparison.csv")

def main():
    """Main pipeline for flood susceptibility modeling."""
    print("="*60)
    print("Flood Susceptibility Modeling Pipeline")
    print("="*60)
    
    # Step 1: Load and prepare data
    X, y = load_data()
    
    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Step 3: Create and train models
    models = create_models()
    trained_models = train_models(models, X_train, y_train)
    
    # Step 4: Evaluate all models
    metrics_dict = {}
    predictions_dict = {}
    probabilities_dict = {}
    
    for name, model in trained_models.items():
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        metrics_dict[name] = metrics
        predictions_dict[name] = y_pred
        probabilities_dict[name] = y_prob
    
    # Step 5: Create comparison table
    create_comparison_table(metrics_dict)
    
    # Step 6: Generate visualizations
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)
    
    for name, model in trained_models.items():
        plot_feature_importance(model, X, name, OUTPUT_DIR)
        perform_shap_analysis(model, X_test, name)
    
    # Step 7: Cross-validation for robustness check
    print("\n" + "="*50)
    print("Cross-Validation Results (5-fold stratified)")
    print("="*50)
    
    cv_scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for name, model in models.items():
        if name == 'SVM':
            pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
            scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        else:
            scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        
        cv_scores[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
        
        print(f"{name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # Save CV results
    cv_df = pd.DataFrame({
        'Model': list(cv_scores.keys()),
        'Mean AUC': [f"{v['mean']:.3f}" for v in cv_scores.values()],
        'Std Dev': [f"{v['std']:.3f}" for v in cv_scores.values()],
        'Scores': [', '.join([f"{s:.3f}" for s in v['scores']]) for v in cv_scores.values()]
    })
    
    cv_df = cv_df.sort_values('Mean AUC', ascending=False)
    cv_df.to_csv(os.path.join(OUTPUT_DIR, 'cross_validation_results.csv'), index=False)
    print(f"\nCross-validation results saved to {OUTPUT_DIR}/cross_validation_results.csv")
    
    # Step 8: Save predictions
    predictions_df = pd.DataFrame(y_test.copy())
    predictions_df = predictions_df.reset_index(drop=True)
    
    for name, y_pred in predictions_dict.items():
        predictions_df[f'{name}_Pred'] = y_pred
        predictions_df[f'{name}_Prob'] = probabilities_dict[name]
    
    predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'model_predictions.csv'), index=False)
    print(f"Predictions saved to {OUTPUT_DIR}/model_predictions.csv")
    
    print("\n" + "="*60)
    print("All analysis completed successfully!")
    print(f"Results saved in '{OUTPUT_DIR}' directory")
    print("="*60)

if __name__ == "__main__":
    main()
