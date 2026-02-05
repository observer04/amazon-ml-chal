# Amazon ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** SkyNet Corp

**Team Members:**
1. Omm Prakash Tripathy (Team Leader)
2. Paresh Chandra Pothal
3. Prem Chandra Prasad
4. Archit Mishra

**Submission Date:** October 13, 2025

---

## 1. Executive Summary

This solution combines engineered tabular features with CLIP text embeddings to predict product prices. After testing multiple approaches (image embeddings, various transforms, different algorithms), we settled on a LightGBM model that achieves **27.86% SMAPE** on 5-fold cross-validation.

The key insight: semantic information from product descriptions (via PCA-reduced text embeddings) complements structured features like quantity, quality keywords, and brand. Text features account for 81% of the model's predictive power.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

Initial EDA revealed several challenges:
- **Wide price range** ($0.13 to $2,796) requiring careful transform selection
- **IPQ data** (Item-Pack-Quantity) available for only 85% of products
- **Unstructured text** containing critical pricing signals (quality tier, brand positioning)
- **Test set outliers** with corrupted value fields (70+ billion!)

Key decisions:
- Use **SQRT transform** instead of LOG (better for all price segments)
- **Skip image embeddings** entirely after testing showed R²=-0.01
- Apply **PCA to text embeddings** (512→20 dims) to remove noise

### 2.2 Solution Strategy

**Approach:** Single LightGBM model with 42 features

**Core innovation:** Most solutions either use pure tabular features OR deep learning embeddings. We found that PCA-reduced text embeddings + domain-engineered features work better than either alone. The semantic understanding from CLIP text complements the structured IPQ/quality/brand signals.

---

## 3. Model Architecture

### 3.1 Feature Pipeline

```
Raw Data → Feature Extraction → Text Embeddings → Model Training
```

**Stage 1: Tabular Feature Engineering (22 features)**
- **IPQ extraction:** Parse "Value: 72.0" and "Ounce" from catalog_content
- **Quality flags:** Binary indicators for premium/organic/gourmet/luxury keywords
- **Brand extraction:** Parse brand name from item_name
- **Interactions:** value×premium, pack×quality, brand×quality (9 combinations)

**Stage 2: Text Embedding Generation**
- **Model:** OpenCLIP ViT-B/32 (laion2b_s34b_b79k weights)
- **Input:** Concatenated item_name + description
- **Output:** 512-dimensional embeddings
- **PCA reduction:** 512 → 20 components (keeps 60% variance, removes noise)

**Stage 3: Feature Combination**
- Concatenate: 22 tabular + 20 text PCA = 42 features
- Encode categoricals: LabelEncoder for brand/unit
- Handle missing: -999 for numerical, "missing" for categorical

### 3.2 Model Configuration

**Algorithm:** LightGBM Gradient Boosting

**Key Parameters:**
```python
num_leaves: 63        # Higher complexity than default
learning_rate: 0.03   # Slow learning prevents overfitting
num_rounds: 2000      # With early stopping (50 rounds)
L1/L2 reg: 0.5        # Regularization for 42 features
```

**Target Transform:** `y = sqrt(price)` then inverse with `price = y²`

**Why SQRT?** Tested LOG vs SQRT:
- LOG: Compresses luxury prices too much (46% → 52% SMAPE for >$100)
- SQRT: Balances all segments better (overall 30.7% → 27.86%)

---

## 4. Model Performance

### 4.1 Validation Results

**Cross-Validation:** 5-fold stratified, random_state=42

| Metric | Value |
|--------|-------|
| **Overall SMAPE** | **27.86%** |
| Std across folds | 0.25% (very stable) |
| Naive mean baseline | 39.53% |
| Improvement | 11.67% better |

**Fold breakdown:**
```
Fold 1: 28.35%
Fold 2: 27.81%
Fold 3: 27.77%
Fold 4: 27.66%
Fold 5: 27.72%
```

### 4.2 Performance by Price Segment

| Segment | Price | SMAPE | Coverage | Assessment |
|---------|-------|-------|----------|------------|
| Mid-Range | $10-$50 | **19.01%** | 51% | Excellent |
| Budget | <$10 | 37.44% | 38% | Decent |
| Premium | $50-$100 | 32.32% | 8% | Good |
| Luxury | >$100 | 46.14% | 3% | Weak (small data) |

Model performs best on mid-range products (majority of training data).

### 4.3 Feature Importance

**Top 10 features:**
1. `value` (6,074) - IPQ quantity signal
2-10. `text_pca_*` (5,121-5,426) - All semantic text features

**Category breakdown:**
- Text PCA: 81.1% of model importance
- Tabular: 18.9% of model importance

This validates our hypothesis: semantic understanding from product descriptions is the primary pricing signal, with structured features providing the foundation.

---

## 5. What Didn't Work

### Failed Experiments

**1. Image CLIP Embeddings**
- **Test:** Ridge regression on 512-dim image embeddings
- **Result:** R²=-0.01, SMAPE=73.49%
- **Why failed:** CLIP trained on natural images, not product photography. Package design doesn't correlate with price in our dataset.

**2. Raw Text Embeddings**
- **Test:** Direct use of 512-dim text embeddings
- **Result:** SMAPE=60.26% (worse than tabular baseline)
- **Fix:** PCA reduction to 20 components improved to 27.86%

**3. Description Length Features**
- **Test:** Added text_length, word_count, bullet_count
- **Result:** Made model worse (31.2% → 30.7%)
- **Why failed:** Noisy signal - cheap products can have long descriptions

**4. LOG Transform**
- **Test:** `y = log(price)` instead of SQRT
- **Result:** Luxury segment SMAPE increased from 46% to 52%
- **Why failed:** Over-compresses high prices

---

## 6. Reproducibility

### Environment
- Python 3.12
- LightGBM 4.5.0
- scikit-learn 1.5.2
- OpenCLIP (torch)

### To Reproduce
```bash
python KAGGLE_submission_enhanced.py
```

**Runtime:** ~30-45 minutes on Kaggle

### Seeds
- K-Fold: random_state=42
- LightGBM: seed=42
- PCA: random_state=42

All operations deterministic (no random augmentation).

---

## 7. Conclusion

Final model is LightGBM trained on 22 tabular features + 20 text PCA components. Key learnings:

1. **Domain features matter:** IPQ extraction more valuable than raw embeddings
2. **Text > Images:** For pricing, description semantics beat visual appearance
3. **PCA essential:** Raw embeddings too noisy, dimensionality reduction crucial
4. **Transform choice matters:** SQRT balanced all price segments better than LOG
5. **Validation rigor pays off:** Caught data leakage (price_per_unit) and test outliers

Expected leaderboard: **28-31% SMAPE** (allowing 1-2% validation/LB gap)

---

## Appendix A: Files Submitted

```
submission_package/
├── Documentation.pdf              # This document
├── submission_enhanced.csv        # Final predictions (75,000 rows)
├── KAGGLE_submission_enhanced.py  # Training script
├── baseline_enhanced.py           # Validation script
├── model_comparison.csv           # All experiments
└── notes.md                       # Quick experiment notes
```

## Appendix B: Additional Experiments

**Ensemble (multiple seeds):** Trained 5 models with different seeds and averaged. Expected improvement: 0.3-0.5% SMAPE. Not included in final submission due to time constraints.

**Segment-specific models:** Separate models for Budget/Mid/Premium/Luxury with tuned hyperparameters. Expected improvement: 0.8-1.2% SMAPE. Future work.

---

**Contact:** observer04  
**Repository:** github.com/observer04/amazon-ml-chal
