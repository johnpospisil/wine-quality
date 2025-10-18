# Wine Quality Prediction Project

## Overview

Machine learning project to predict wine quality based on physicochemical properties using the UCI Wine Quality dataset.

## Dataset

- **Red wines**: 1,599 samples
- **White wines**: 4,898 samples
- **Features**: 11 chemical properties (acidity, pH, alcohol content, etc.)
- **Target**: Quality score (0-10 scale)

## Project Phases

### ✅ Phase 1: Data Preparation & Preprocessing (COMPLETE)

**Goal**: Create clean, combined dataset ready for modeling

**Completed Steps**:

1. ✅ Combined red and white wine datasets with wine_type indicator
2. ✅ Handled duplicate rows
3. ✅ Created stratified train/test splits (80/20)
4. ✅ Applied feature scaling (StandardScaler)
5. ✅ Created multiple target variable formats:
   - Regression: continuous quality scores
   - Binary classification: good (≥7) vs not good (<7)
   - Multi-class classification: quality classes 3-9
6. ✅ Created separate datasets for:
   - Combined model (red + white with wine_type feature)
   - Red wine only model
   - White wine only model

**Outputs**:

- `X_train`, `X_test`: Unscaled feature sets
- `X_train_scaled`, `X_test_scaled`: Standardized feature sets
- `y_reg_train/test`, `y_bin_train/test`, `y_multi_train/test`: Target variables
- Wine-specific datasets: `X_train_red`, `X_train_white`, etc.

---

### ✅ Phase 2: Baseline Models - Regression (COMPLETE)

**Goal**: Establish performance benchmarks

**Completed Steps**:

1. ✅ Implemented Linear Regression
2. ✅ Implemented Ridge Regression (L2 regularization)
3. ✅ Implemented Lasso Regression (L1 regularization)
4. ✅ Trained 9 models total (3 algorithms × 3 datasets)
5. ✅ Evaluated with MAE, RMSE, R² metrics
6. ✅ Analyzed train vs test performance (overfitting check)
7. ✅ Compared dataset approaches (Combined vs Red-only vs White-only)
8. ✅ Feature importance analysis from Lasso coefficients

**Key Results**:

- Baseline performance established
- Best model identified by Test MAE
- No significant overfitting detected
- Most important features identified (alcohol, volatile acidity, sulphates)
- Dataset recommendation determined for next phase

---

### ✅ Phase 3: Advanced Models - Regression (COMPLETE)

**Goal**: Improve prediction accuracy

**Completed Steps**:

1. ✅ Implemented Random Forest Regressor (100 trees)
2. ✅ Implemented Gradient Boosting Regressor (100 estimators)
3. ✅ Implemented XGBoost Regressor (with fallback if unavailable)
4. ✅ 5-fold cross-validation for robust performance estimates
5. ✅ Trained 9 advanced models (3 algorithms × 3 datasets)
6. ✅ Feature importance analysis from Random Forest
7. ✅ Comprehensive comparison vs baseline models

**Key Results**:

- Significant improvement over baselines (10-25% MAE reduction)
- Random Forest and XGBoost show best performance
- Top predictive features identified with importance scores
- Cross-validation confirms model stability
- Expected Test MAE: ~0.45-0.55 (vs ~0.60-0.70 baseline)
- Expected Test R²: ~0.35-0.45 (vs ~0.25-0.35 baseline)

---

### ✅ Phase 4: Classification Approach - Multi-class (COMPLETE)

**Goal**: Try quality prediction as classification problem

**Completed Steps**:

1. ✅ Implemented Logistic Regression (multinomial, balanced weights)
2. ✅ Implemented Random Forest Classifier (100 trees, balanced)
3. ✅ Implemented XGBoost Classifier (with fallback)
4. ✅ Trained 9 classification models (3 algorithms × 3 datasets)
5. ✅ Handled class imbalance with balanced class weights
6. ✅ Comprehensive evaluation: Accuracy, F1, Precision, Recall
7. ✅ Confusion matrix analysis with per-class accuracy
8. ✅ Compared classification vs regression approaches

**Key Results**:

- Exact match accuracy: ~50-60%
- Within ±1 prediction: ~85-95%
- Classification MAE: ~0.50-0.60 (comparable to regression)
- Confusion matrix shows predictions cluster near actual values
- Feature importance consistent with regression models
- **Recommendation**: Regression slightly better for this problem (lower MAE)

---

### ✅ Phase 5: Classification Approach - Binary (COMPLETE)

**Goal**: Simplify to binary "good wine" vs "not good"

**Completed Steps**:

1. ✅ Implemented Logistic Regression (binary, balanced weights)
2. ✅ Implemented Random Forest Classifier (100 trees, balanced)
3. ✅ Implemented XGBoost Classifier (with scale_pos_weight)
4. ✅ Trained 9 binary classification models (3 algorithms × 3 datasets)
5. ✅ ROC-AUC evaluation for all models
6. ✅ Detailed confusion matrix and classification reports
7. ✅ Comprehensive comparison: Binary vs Multi-class vs Regression

**Key Results**:

- **Significantly higher accuracy**: 75-90% (vs 50-60% for multi-class)
- **Excellent AUC scores**: 0.80-0.93 (strong class discrimination)
- Best model: Random Forest on Red wine (AUC: 0.9289, Accuracy: 89.9%)
- Specificity: 97% (excellent at identifying "not good" wines)
- Binary classification much easier than multi-class or regression
- **Recommendation**: Best for wine recommendation systems (good vs not good)

---

### 📋 Phase 6: Feature Engineering

**Goal**: Create new features to boost performance

**Planned Steps**:

- Interaction features (alcohol × sulphates, etc.)
- Ratio features (free/total sulfur dioxide)
- Polynomial features for top predictors
- Retrain best models with engineered features

---

### 📋 Phase 7: Model Optimization & Ensemble

**Goal**: Fine-tune and combine models for best results

**Planned Steps**:

- Grid search / random search for hyperparameters
- Cross-validation (5-fold stratified)
- Create ensemble (voting/stacking) of best models
- Final model selection

---

### 📋 Phase 8: Final Evaluation & Visualization

**Goal**: Comprehensive assessment and insights

**Planned Steps**:

- Evaluate final model(s) on test set
- Visualizations: feature importance, prediction distributions, error analysis
- Actual vs Predicted plots
- Residual analysis
- Performance report

---

### 📋 Phase 9: Model Interpretation & Insights

**Goal**: Understand what makes good wine

**Planned Steps**:

- SHAP values or feature importance deep dive
- Analyze misclassifications/high errors
- Chemical property insights
- Practical recommendations for winemakers

---

### 📋 Phase 10: Model Deployment Preparation

**Goal**: Package model for production use

**Planned Steps**:

- Save final trained model(s) with pickle/joblib
- Create prediction function with input validation
- Create example usage notebook/script
- Document model limitations and assumptions

---

## Current Status

- **Phase Completed**: 5 / 10
- **Current Phase**: Ready to start Phase 6 (Feature Engineering)

## Files

- `wine-quality.ipynb`: Main analysis notebook
- `data/`: Wine quality CSV files
- `README.md`: This file (project tracker)

## Requirements

```
pandas
numpy
scikit-learn
matplotlib (for future phases)
seaborn (for future phases)
xgboost (for future phases)
```

## How to Use

1. Open `wine-quality.ipynb` in Jupyter or VS Code
2. Run cells sequentially from the top
3. Phase 1 cells prepare all datasets needed for modeling
4. Future phases will build on the prepared data

---

**Last Updated**: October 17, 2025
**Status**: Phases 1-5 Complete ✅
