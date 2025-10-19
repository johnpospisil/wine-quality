"""
Wine Quality Feature Engineering Pipeline
==========================================
Production-ready feature engineering for wine quality prediction

Author: Generated from wine-quality project
Date: 2025-10-18 20:21:56
Version: 1.0.0
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Create engineered features for wine quality prediction.

    Args:
        df (pd.DataFrame): Wine features dataframe with columns:
            - fixed acidity, volatile acidity, citric acid, residual sugar,
              chlorides, free sulfur dioxide, total sulfur dioxide, density,
              pH, sulphates, alcohol

    Returns:
        pd.DataFrame: Original features + 13 engineered features
    """
    df_eng = df.copy()

    # Interaction features
    df_eng['alcohol_x_sulphates'] = df['alcohol'] * df['sulphates']
    df_eng['alcohol_x_volatile_acidity'] = df['alcohol'] * df['volatile acidity']
    df_eng['citric_x_fixed_acid'] = df['citric acid'] * df['fixed acidity']

    # Ratio features
    df_eng['free_to_total_sulfur'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-5)
    df_eng['citric_to_fixed_acid'] = df['citric acid'] / (df['fixed acidity'] + 1e-5)
    df_eng['sulphates_to_chlorides'] = df['sulphates'] / (df['chlorides'] + 1e-5)

    # Polynomial features
    df_eng['alcohol_squared'] = df['alcohol'] ** 2
    df_eng['volatile_acidity_squared'] = df['volatile acidity'] ** 2
    df_eng['sulphates_squared'] = df['sulphates'] ** 2

    # Domain-specific features
    df_eng['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
    df_eng['acidity_to_alcohol'] = (df['fixed acidity'] + df['volatile acidity']) / (df['alcohol'] + 1e-5)
    df_eng['total_sulfur_dioxide_log'] = np.log1p(df['total sulfur dioxide'])
    df_eng['free_sulfur_dioxide_log'] = np.log1p(df['free sulfur dioxide'])

    return df_eng


def validate_input_features(df, required_features=None):
    """
    Validate that input dataframe has all required features.

    Args:
        df (pd.DataFrame): Input wine features
        required_features (list): List of required column names

    Returns:
        tuple: (is_valid, missing_features)
    """
    if required_features is None:
        required_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]

    missing = [col for col in required_features if col not in df.columns]
    return len(missing) == 0, missing


# Example usage
if __name__ == "__main__":
    # Example wine sample
    sample_wine = pd.DataFrame({
        'fixed acidity': [8.5],
        'volatile acidity': [0.4],
        'citric acid': [0.35],
        'residual sugar': [2.5],
        'chlorides': [0.08],
        'free sulfur dioxide': [15.0],
        'total sulfur dioxide': [50.0],
        'density': [0.996],
        'pH': [3.3],
        'sulphates': [0.7],
        'alcohol': [11.5]
    })

    # Validate and engineer features
    is_valid, missing = validate_input_features(sample_wine)
    if is_valid:
        wine_engineered = engineer_features(sample_wine)
        print(f"Original features: {len(sample_wine.columns)}")
        print(f"Engineered features: {len(wine_engineered.columns)}")
        print(f"New features added: {len(wine_engineered.columns) - len(sample_wine.columns)}")
    else:
        print(f"Missing features: {missing}")
