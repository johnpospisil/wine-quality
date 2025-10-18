"""
Wine Quality Prediction - Example Usage
========================================
Demonstrates how to use the deployment package for real-world predictions

Run this script to see example predictions on sample wines.
"""

import sys
from pathlib import Path

# Add deployment directory to path
sys.path.insert(0, str(Path(__file__).parent))

from wine_predictor_api import WineQualityPredictor


def main():
    """Run example predictions"""

    print("=" * 80)
    print("WINE QUALITY PREDICTION - EXAMPLE USAGE")
    print("=" * 80)

    # Initialize predictor
    predictor = WineQualityPredictor(
        models_dir='models',
        scalers_dir='scalers'
    )

    # Example 1: High-quality red wine
    print("\n" + "-" * 80)
    print("Example 1: Premium Red Wine Sample")
    print("-" * 80)

    premium_wine = {{
        'fixed acidity': 8.8,
        'volatile acidity': 0.35,
        'citric acid': 0.45,
        'residual sugar': 2.2,
        'chlorides': 0.075,
        'free sulfur dioxide': 12.0,
        'total sulfur dioxide': 45.0,
        'density': 0.9965,
        'pH': 3.25,
        'sulphates': 0.85,
        'alcohol': 12.5
    }}

    result1 = predictor.predict_comprehensive(premium_wine)
    print(f"\nQuality Score: {{result1['regression']['quality_score']}}/10")
    print(f"Rounded: {{result1['regression']['quality_rounded']}}")
    print(f"Classification: {{result1['classification']['quality_class']}}")
    print(f"Probability Good: {{result1['classification']['probability_good']*100:.1f}}%")
    print(f"Confidence: {{result1['classification']['confidence_level'].upper()}}")
    print(f"\n💡 {{result1['recommendation']}}")

    # Example 2: Average quality wine
    print("\n" + "-" * 80)
    print("Example 2: Average Quality Wine Sample")
    print("-" * 80)

    average_wine = {{
        'fixed acidity': 7.5,
        'volatile acidity': 0.55,
        'citric acid': 0.25,
        'residual sugar': 2.8,
        'chlorides': 0.095,
        'free sulfur dioxide': 18.0,
        'total sulfur dioxide': 65.0,
        'density': 0.9972,
        'pH': 3.35,
        'sulphates': 0.60,
        'alcohol': 10.2
    }}

    result2 = predictor.predict_comprehensive(average_wine)
    print(f"\nQuality Score: {{result2['regression']['quality_score']}}/10")
    print(f"Rounded: {{result2['regression']['quality_rounded']}}")
    print(f"Classification: {{result2['classification']['quality_class']}}")
    print(f"Probability Good: {{result2['classification']['probability_good']*100:.1f}}%")
    print(f"Confidence: {{result2['classification']['confidence_level'].upper()}}")
    print(f"\n💡 {{result2['recommendation']}}")

    # Example 3: Batch prediction
    print("\n" + "-" * 80)
    print("Example 3: Batch Prediction (Multiple Wines)")
    print("-" * 80)

    import pandas as pd

    batch_wines = pd.DataFrame([premium_wine, average_wine])

    print(f"\nProcessing {{len(batch_wines)}} wine samples...")

    for i, wine in batch_wines.iterrows():
        result = predictor.predict_quality_score(wine.to_dict())
        print(f"  Wine {{i+1}}: Quality {{result['quality_score']:.2f}} ({{result['confidence']}} confidence)")

    # Example 4: Error handling
    print("\n" + "-" * 80)
    print("Example 4: Input Validation")
    print("-" * 80)

    invalid_wine = {{
        'fixed acidity': 8.0,
        'volatile acidity': 0.4,
        # Missing other required features
    }}

    try:
        result = predictor.predict_quality_score(invalid_wine)
    except ValueError as e:
        print(f"\n❌ Validation Error: {{e}}")
        print("✅ Input validation working correctly!")

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
