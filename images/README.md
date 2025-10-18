# Wine Quality Prediction - Visualizations

This folder contains all visualizations generated during the wine quality prediction project.

## Phase 9: Model Interpretation & Feature Analysis

### phase9_feature_importance_interactions.jpg

**Created:** Phase 9 - Model Interpretation & Insights  
**Size:** 16x12 inches (4 panels)  
**DPI:** 300

**Description:**  
Comprehensive feature analysis visualization with four key components:

1. **Top 10 Feature Importances (Top-Left)**

   - Shows Random Forest feature importance rankings
   - Alcohol content is the strongest predictor (~20% importance)
   - Top features: alcohol, volatile acidity, sulphates, citric acid, total sulfur dioxide

2. **Feature Interaction: Alcohol × Sulphates (Top-Right)**

   - Scatter plot showing synergistic effect
   - High alcohol (>11%) + high sulphates (0.6-0.8) = higher quality wines
   - Color-coded by actual wine quality (3-8 scale)
   - Demonstrates ~+0.41 quality improvement when both features are elevated

3. **Volatile Acidity Impact on Quality (Bottom-Left)**

   - Box plot comparing volatile acidity across quality levels
   - Shows clear inverse relationship: lower VA = higher quality
   - Quality 7-8 wines have VA < 0.5 g/L
   - Quality 3-4 wines have VA > 0.6 g/L

4. **Decision Boundary Confidence Grid (Bottom-Right)**
   - Heatmap showing model confidence across alcohol-sulphates space
   - 95.9% accuracy in high-confidence regions
   - Helps identify where model predictions are most reliable
   - Darker regions = higher classification confidence

**Key Insights:**

- Alcohol content is the dominant quality predictor
- Volatile acidity shows strong inverse correlation with quality
- Feature interactions (especially alcohol × sulphates) provide significant predictive power
- Model confidence varies across feature space, with highest confidence in extreme regions

**Usage:**
This visualization is used in:

- Project documentation (README.md, PROJECT_SUMMARY.md)
- Model interpretation analysis
- Winemaking insight generation
- Presentation materials

---

## Visualization Standards

All visualizations in this project follow these standards:

- **Format:** PNG or JPG
- **DPI:** 300 (high-resolution for presentations/publications)
- **Size:** Optimized for readability (typically 12-16 inches wide)
- **Style:** Seaborn whitegrid theme with clear labels and legends
- **Naming:** `phase<N>_<description>.png/jpg`

## Regenerating Visualizations

To regenerate any visualization:

1. Open `wine-quality.ipynb` in Jupyter
2. Navigate to the "Export All Visualizations" section
3. Run the corresponding export cell
4. Visualizations will be saved to this `images/` folder

Alternatively, run individual phase cells that create the original visualizations, then save with:

```python
plt.savefig('images/<filename>.png', dpi=300, bbox_inches='tight')
```
