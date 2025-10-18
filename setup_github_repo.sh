#!/bin/bash

# Script to set up wine-quality GitHub repository
# Run this from the wine-quality directory

echo "==================================================================="
echo "Setting up wine-quality GitHub repository"
echo "==================================================================="

# Check if we're in the right directory
if [ ! -f "wine-quality.ipynb" ]; then
    echo "Error: wine-quality.ipynb not found. Please run this from the project directory."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo ""
    echo "Step 1: Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Add all files
echo ""
echo "Step 2: Adding files to git..."
git add .
echo "✓ Files staged"

# Create initial commit
echo ""
echo "Step 3: Creating initial commit..."
git commit -m "Initial commit: Wine Quality Prediction project

- Phase 1: Data preparation and preprocessing
- Phase 2: Baseline regression models (Linear, Ridge, Lasso)
- Phase 3: Advanced regression models (Random Forest, Gradient Boosting, XGBoost)
- Dataset: UCI Wine Quality (red and white wines)
- Comprehensive analysis and model comparison"

echo "✓ Initial commit created"

# Instructions for creating GitHub repo
echo ""
echo "==================================================================="
echo "Next steps to push to GitHub:"
echo "==================================================================="
echo ""
echo "1. Go to https://github.com/new"
echo "2. Repository name: wine-quality"
echo "3. Description: Machine learning project to predict wine quality"
echo "4. Choose Public or Private"
echo "5. DO NOT initialize with README (we already have one)"
echo "6. Click 'Create repository'"
echo ""
echo "7. Then run these commands:"
echo ""
echo "   git branch -M main"
echo "   git remote add origin https://github.com/YOUR_USERNAME/wine-quality.git"
echo "   git push -u origin main"
echo ""
echo "==================================================================="
echo ""
echo "Or if you want to use SSH:"
echo ""
echo "   git branch -M main"
echo "   git remote add origin git@github.com:YOUR_USERNAME/wine-quality.git"
echo "   git push -u origin main"
echo ""
echo "==================================================================="
