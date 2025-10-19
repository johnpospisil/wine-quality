"""
Wine Quality Prediction API
============================
Production-ready API for wine quality predictions

Author: Generated from wine-quality project
Date: 2025-10-18 20:21:56
Version: 1.0.0

Models Available:
- Regression: Predict quality score (3-8)
- Binary Classification: Predict good (≥7) vs not good (<7)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from feature_engineering import engineer_features, validate_input_features


class WineQualityPredictor:
    """Wine quality prediction model wrapper"""

    def __init__(self, models_dir='models', scalers_dir='scalers'):
        """
        Initialize predictor with trained models and scalers.

        Args:
            models_dir (str): Path to directory containing model files
            scalers_dir (str): Path to directory containing scaler files
        """
        self.models_dir = Path(models_dir)
        self.scalers_dir = Path(scalers_dir)
        self.models = {}
        self.scalers = {}

    def load_model(self, model_name):
        """Load a specific model"""
        model_path = self.models_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.models[model_name] = joblib.load(model_path)
        return self.models[model_name]

    def load_scaler(self, scaler_name):
        """Load a specific scaler"""
        scaler_path = self.scalers_dir / f"{scaler_name}.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scalers[scaler_name] = joblib.load(scaler_path)
        return self.scalers[scaler_name]

    def predict_quality_score(self, wine_features, wine_type='red'):
        """
        Predict wine quality score (regression).

        Args:
            wine_features (pd.DataFrame or dict): Wine chemical properties
            wine_type (str): 'red', 'white', or 'combined'

        Returns:
            dict: Prediction results with score and confidence
        """
        # Convert dict to DataFrame if needed
        if isinstance(wine_features, dict):
            wine_features = pd.DataFrame([wine_features])

        # Validate input
        is_valid, missing = validate_input_features(wine_features)
        if not is_valid:
            raise ValueError(f"Missing required features: {missing}")

        # Engineer features
        wine_eng = engineer_features(wine_features)

        # Load model and scaler
        if 'xgb_regression_red' not in self.models:
            self.load_model('xgb_regression_red')
        if 'scaler_red_original' not in self.scalers:
            self.load_scaler('scaler_red_original')

        # Scale features
        wine_scaled = self.scalers['scaler_red_original'].transform(wine_eng)

        # Predict
        quality_score = self.models['xgb_regression_red'].predict(wine_scaled)[0]

        # Clip to valid range
        quality_score = np.clip(quality_score, 3, 8)

        return {
            'quality_score': round(quality_score, 2),
            'quality_rounded': round(quality_score),
            'model': 'xgb_regression_red',
            'confidence': 'high' if abs(quality_score - round(quality_score)) < 0.3 else 'medium'
        }

    def predict_binary_class(self, wine_features, wine_type='red'):
        """
        Predict if wine is good quality (≥7) or not (<7).

        Args:
            wine_features (pd.DataFrame or dict): Wine chemical properties
            wine_type (str): 'red', 'white', or 'combined'

        Returns:
            dict: Prediction results with class, probability, and confidence
        """
        # Convert dict to DataFrame if needed
        if isinstance(wine_features, dict):
            wine_features = pd.DataFrame([wine_features])

        # Validate input
        is_valid, missing = validate_input_features(wine_features)
        if not is_valid:
            raise ValueError(f"Missing required features: {missing}")

        # Engineer features
        wine_eng = engineer_features(wine_features)

        # Load model and scaler
        if 'rf_classification_red' not in self.models:
            self.load_model('rf_classification_red')
        if 'scaler_red_original' not in self.scalers:
            self.load_scaler('scaler_red_original')

        # Scale features
        wine_scaled = self.scalers['scaler_red_original'].transform(wine_eng)

        # Predict
        prediction = self.models['rf_classification_red'].predict(wine_scaled)[0]
        probabilities = self.models['rf_classification_red'].predict_proba(wine_scaled)[0]

        return {
            'is_good_quality': bool(prediction),
            'quality_class': 'Good (≥7)' if prediction else 'Not Good (<7)',
            'probability_good': round(probabilities[1], 3),
            'probability_not_good': round(probabilities[0], 3),
            'confidence': round(max(probabilities), 3),
            'confidence_level': 'high' if max(probabilities) > 0.8 else 'medium' if max(probabilities) > 0.6 else 'low',
            'model': 'rf_classification_red'
        }

    def predict_comprehensive(self, wine_features, wine_type='red'):
        """
        Get both regression and classification predictions.

        Args:
            wine_features (pd.DataFrame or dict): Wine chemical properties
            wine_type (str): 'red', 'white', or 'combined'

        Returns:
            dict: Comprehensive prediction results
        """
        regression_result = self.predict_quality_score(wine_features, wine_type)
        classification_result = self.predict_binary_class(wine_features, wine_type)

        return {
            'regression': regression_result,
            'classification': classification_result,
            'recommendation': self._generate_recommendation(regression_result, classification_result)
        }

    def _generate_recommendation(self, reg_result, class_result):
        """Generate actionable recommendation based on predictions"""
        score = reg_result['quality_score']
        is_good = class_result['is_good_quality']
        confidence = class_result['confidence']

        if is_good and confidence > 0.8:
            return f"Excellent wine! Predicted quality: {score:.1f}/10. High confidence classification as premium wine."
        elif is_good:
            return f"Good wine with quality score {score:.1f}/10, though confidence is moderate. Consider minor refinements."
        elif score > 6.0:
            return f"Borderline quality ({score:.1f}/10). Small improvements could push this into premium category."
        else:
            return f"Quality score {score:.1f}/10 suggests significant room for improvement in winemaking process."


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = WineQualityPredictor()

    # Example wine sample
    sample_wine = {
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
    result = predictor.predict_comprehensive(sample_wine, wine_type='red')

    print("Wine Quality Prediction Results:")
    print(f"  Quality Score: {result['regression']['quality_score']}")
    print(f"  Classification: {result['classification']['quality_class']}")
    print(f"  Confidence: {result['classification']['confidence_level']}")
    print(f"\nRecommendation: {result['recommendation']}")
