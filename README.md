---

# Flood Susceptibility Mapping Using Machine Learning and Morphometric Analysis

## Overview
This project integrates **Machine Learning (ML)** techniques and **morphometric analysis** to assess flood susceptibility across four major catchments in Bayelsa State, Nigeria. The study leverages **Shuttle Radar Topographic Mission (SRTM)** data to extract hydrological features and applies machine learning classifiers to classify flood-prone zones. The goal is to enhance flood prediction accuracy and granularity beyond traditional morphometric methods.

## Dataset
The dataset used for this analysis includes key morphometric features, such as:
- Drainage Density (Dd)
- Stream Frequency (Fs)
- Relief Ratio (Rh)
- Bifurcation Ratio (Rbm)
- Infiltration Number (If)
- Circularity Ratio (Ff)
- Hypsometric Integral (Hh)
- Average Slope (Slope)

The flood-prone labels are generated from historical flood records, with each catchment marked as either **flood-prone (1)** or **not flood-prone (0)**.

You can access the dataset template for training in `morphometric_ml_dataset.csv`.

## Requirements
Ensure you have the following Python packages installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `lightgbm`
- `shap`

You can install them via `pip`:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap
```

## File Structure

```
├── flood_susceptibility_ml.py       # Enhanced Python script for model training and evaluation
├── morphometric_ml_dataset.csv     # Dataset template for morphometric features
├── results/                         # Directory for generated results and visualizations
│   ├── confusion_matrices.png      # Confusion matrices for all classifiers
│   ├── roc_curves.png              # ROC Curve comparison figure
│   ├── model_comparison.csv        # Performance comparison table
│   ├── cross_validation_results.csv# 5-fold cross-validation results
│   ├── model_predictions.csv       # Predictions on test dataset
│   └── [model-specific files]      # Feature importance and SHAP analysis files
└── README.md                       # Project overview and instructions
```

## Usage

### 1. Prepare the Data
Load your dataset in `morphometric_ml_dataset.csv`, ensuring the dataset includes morphometric features (Dd, Fs, Rh, Rbm, If, Ff, Hh, Slope) and a binary flood-prone label.

### 2. Model Training
Run the Python script `flood_susceptibility_ml.py` to:
- Train five ML models (Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM) on the dataset
- Evaluate the models using accuracy, precision, recall, F1-score, and ROC-AUC metrics
- Generate and save confusion matrices, ROC curves, and feature importance plots
- Perform SHAP analysis for model interpretation
- Run 5-fold cross-validation for robustness check

```bash
python flood_susceptibility_ml.py
```

### 3. Results
The script will generate the following in the `results/` directory:

#### Figures:
- **Confusion Matrices**: Displays confusion matrix for each classifier
- **ROC Curves**: Compares ROC curves for all models with AUC values
- **Feature Importance**: Plots feature importance scores for each tree-based model
- **SHAP Analysis**: Provides SHAP summary and beeswarm plots for model interpretation

#### Data Files:
- **model_comparison.csv**: Performance comparison table of all models
- **cross_validation_results.csv**: 5-fold cross-validation results with mean and standard deviation
- **model_predictions.csv**: Predictions and probabilities for each model on the test dataset

## Enhanced Features
The improved script includes:
1. **Automated Data Loading**: Tries to load real data first, falls back to synthetic data
2. **Comprehensive Preprocessing**: Handles missing values and data scaling
3. **More Models**: Added LightGBM and Gradient Boosting classifiers
4. **Advanced Evaluation**: Classification report, ROC curves, cross-validation
5. **Model Interpretation**: SHAP analysis for feature importance visualization
6. **Structured Output**: All results saved in a dedicated `results/` directory
7. **Robustness Check**: 5-fold stratified cross-validation
8. **Improved Visualizations**: High-quality plots with clear labels and formatting

## Author Contributions


## Acknowledgements
We would like to thank the **Nigerian Hydrological Services Agency (NHSA)** for providing historical flood data, and the **United States Geological Survey (USGS)** for the SRTM data used in this study.

---

Feel free to modify or add any additional details to this README as needed!
