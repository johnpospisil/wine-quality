# Wine Quality Prediction - Deployment Package

## Overview

Production-ready machine learning models for predicting wine quality based on chemical properties. This package includes trained models, feature engineering pipeline, and easy-to-use API.

**Version**: 1.0.0  
**Created**: 2025-10-17 18:52:46  
**Dataset**: Portuguese Vinho Verde Red Wines

## 📦 Package Contents

```
deployment/
├── models/                          # Trained model files
│   ├── xgb_regression_red.joblib   # XGBoost regression (MAE: 0.45)
│   ├── rf_classification_red.joblib # Random Forest classification (89.9% acc)
│   ├── gb_regression_red.joblib    # Gradient Boosting regression
│   └── xgb_regression_tuned.joblib # Tuned XGBoost
├── scalers/                         # Feature scalers
│   ├── scaler_red_original.joblib  # StandardScaler for red wines
│   └── scaler_combined.joblib      # StandardScaler for combined
├── metadata/                        # Model documentation
│   └── model_metadata.json         # Performance metrics, specs
├── feature_engineering.py          # Feature engineering pipeline
├── wine_predictor_api.py           # Prediction API
├── example_usage.py                # Usage examples
└── README.md                        # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn xgboost joblib

# Verify installation
python example_usage.py
```

### Basic Usage

```python
from wine_predictor_api import WineQualityPredictor

# Initialize predictor
predictor = WineQualityPredictor(
    models_dir='models',
    scalers_dir='scalers'
)

# Define wine sample
wine = {
    'fixed acidity': 8.5,
    'volatile acidity': 0.4,
    'citric acid': 0.35,
    'residual sugar': 2.5,
    'chlorides': 0.08,
    'free sulfur dioxide': 15.0,
    'total sulfur dioxide': 50.0,
    'density': 0.996,
    'pH': 3.3,
    'sulphates': 0.7,
    'alcohol': 11.5
}

# Get comprehensive prediction
result = predictor.predict_comprehensive(wine)

print(f"Quality Score: {result['regression']['quality_score']}")
print(f"Classification: {result['classification']['quality_class']}")
print(f"Recommendation: {result['recommendation']}")
```

## 📊 Model Performance

### Regression Model (XGBoost)
- **MAE**: 0.4503 (±0.45 quality points on average)
- **R²**: 0.4323 (43% variance explained)
- **Within ±1 point**: 91.8% of predictions
- **Use case**: Precise quality score estimation

### Classification Model (Random Forest)
- **Accuracy**: 89.9%
- **AUC-ROC**: 0.9305
- **Precision**: 82.4% (good wines correctly identified)
- **Recall**: 75.7% (good wines detected)
- **Use case**: Binary good/not-good decisions

## 🔧 API Reference

### `WineQualityPredictor`

Main class for wine quality predictions.

#### Methods

**`predict_quality_score(wine_features, wine_type='red')`**
- Predicts continuous quality score (3-8 range)
- Returns: `{'quality_score': float, 'quality_rounded': int, 'confidence': str}`

**`predict_binary_class(wine_features, wine_type='red')`**
- Predicts if wine is good quality (≥7) or not (<7)
- Returns: `{'is_good_quality': bool, 'probability_good': float, 'confidence_level': str}`

**`predict_comprehensive(wine_features, wine_type='red')`**
- Gets both regression and classification predictions
- Returns: Full prediction dictionary with recommendation

## 📝 Input Requirements

All 11 chemical features are **required**:

| Feature | Unit | Typical Range | Description |
|---------|------|---------------|-------------|
| `fixed acidity` | g/L | 4.6 - 15.9 | Tartaric acid (non-volatile) |
| `volatile acidity` | g/L | 0.12 - 1.58 | Acetic acid (vinegar taste) |
| `citric acid` | g/L | 0 - 1 | Freshness additive |
| `residual sugar` | g/L | 0.9 - 15.5 | Unfermented sugar |
| `chlorides` | g/L | 0.01 - 0.61 | Salt content |
| `free sulfur dioxide` | mg/L | 1 - 72 | Free SO₂ (prevents oxidation) |
| `total sulfur dioxide` | mg/L | 6 - 289 | Total SO₂ |
| `density` | g/cm³ | 0.99 - 1.00 | Relative to water |
| `pH` | - | 2.74 - 4.01 | Acidity level |
| `sulphates` | g/L | 0.33 - 2.0 | Potassium sulphate |
| `alcohol` | % vol | 8.4 - 14.9 | Alcohol content |

**Note**: Missing values are not allowed. Handle them before prediction.

## 🎯 Use Cases

### 1. Quality Control
```python
# Check if wine meets quality standards
result = predictor.predict_binary_class(wine_sample)
if result['is_good_quality'] and result['confidence_level'] == 'high':
    print("✅ Approved for premium line")
else:
    print("⚠️  Requires improvement")
```

### 2. Batch Processing
```python
import pandas as pd

# Load multiple wine samples
wines_df = pd.read_csv('wine_samples.csv')

# Predict for all
for idx, wine in wines_df.iterrows():
    result = predictor.predict_quality_score(wine.to_dict())
    wines_df.loc[idx, 'predicted_quality'] = result['quality_score']

wines_df.to_csv('wines_with_predictions.csv')
```

### 3. Production Optimization
```python
# Analyze which features to adjust
result = predictor.predict_quality_score(current_wine)

if result['quality_score'] < 6.5:
    print("Recommendations:")
    print("- Increase alcohol content (target >11%)")
    print("- Reduce volatile acidity (target <0.5 g/L)")
    print("- Optimize sulphates (target 0.7-0.8 g/L)")
```

## ⚠️  Limitations

1. **Wine Type**: Models optimized for **red wines** (specifically Portuguese Vinho Verde)
2. **Quality Range**: Trained on wines rated 3-8 (out of 10)
3. **Geographic**: May not generalize well to wines from other regions
4. **Sample Size**: Trained on 1,599 red wine samples
5. **Features**: Only considers chemical properties, not sensory attributes

## 🔬 Feature Engineering

The pipeline automatically creates 13 engineered features:

**Interactions** (3):
- `alcohol_x_sulphates`: Synergistic quality effect
- `alcohol_x_volatile_acidity`: Balance indicator
- `citric_x_fixed_acid`: Acidity structure

**Ratios** (3):
- `free_to_total_sulfur`: SO₂ availability
- `citric_to_fixed_acid`: Freshness ratio
- `sulphates_to_chlorides`: Preservation balance

**Polynomials** (3):
- `alcohol_squared`: Non-linear alcohol effect
- `volatile_acidity_squared`: VA penalty amplification
- `sulphates_squared`: Optimal sulphate level

**Domain-Specific** (4):
- `total_acidity`: Combined acidity measure
- `acidity_to_alcohol`: Balance metric
- `total_sulfur_dioxide_log`: Normalized SO₂
- `free_sulfur_dioxide_log`: Normalized free SO₂

## 📈 Performance Benchmarks

| Metric | Baseline | Advanced | Final (Tuned) | Improvement |
|--------|----------|----------|---------------|-------------|
| MAE | 0.60 | 0.45 | 0.45 | **25%** |
| R² | 0.25 | 0.43 | 0.43 | **72%** |
| Accuracy | 65% | 89% | 90% | **38%** |
| AUC | 0.75 | 0.93 | 0.93 | **24%** |

## 🛠️ Troubleshooting

**Error: Missing features**
```python
# Use validation before prediction
from feature_engineering import validate_input_features

is_valid, missing = validate_input_features(wine_df)
if not is_valid:
    print(f"Missing features: {missing}")
```

**Error: Model file not found**
```python
# Ensure correct paths
predictor = WineQualityPredictor(
    models_dir='deployment/models',  # Adjust path
    scalers_dir='deployment/scalers'
)
```

## 📚 Additional Resources

- **Model Training Notebook**: `wine-quality.ipynb` (full development process)
- **Metadata**: `metadata/model_metadata.json` (detailed model specs)
- **GitHub**: [github.com/johnpospisil/wine-quality](https://github.com/johnpospisil/wine-quality)

## 📄 License

This deployment package is part of the Wine Quality Prediction project.

## 🤝 Support

For issues or questions:
1. Check `example_usage.py` for common use cases
2. Review `model_metadata.json` for model specifications
3. Refer to the main notebook for training details

---

**Last Updated**: 2025-10-17 18:52:46  
**Version**: 1.0.0
