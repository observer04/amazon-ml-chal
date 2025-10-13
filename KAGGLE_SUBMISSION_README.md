# 🎯 KAGGLE SUBMISSION - Enhanced Model

## 📊 Performance Summary

| Model | Validation SMAPE | Description |
|-------|------------------|-------------|
| **Baseline** | 30.70% | Tabular features only (LightGBM) |
| **Enhanced** | **27.86%** | Tabular + Text PCA (20 components) |
| **Improvement** | **+2.84%** | 9.2% relative improvement |

**Expected Leaderboard:** 27-30% SMAPE

---

## 🏗️ Model Architecture

### **Features (42 total):**

**Tabular Features (22):**
- IPQ: `value`, `unit`, `pack_size`
- Quality: `has_premium`, `has_organic`, `has_gourmet`, `has_natural`, `has_artisan`, `has_luxury`
- Size: `is_travel_size`, `is_bulk`
- Brand: `brand`, `brand_exists`
- Interactions: 
  - Value×Quality: `value_x_premium`, `value_x_luxury`, `value_x_organic`
  - Pack×Quality: `pack_x_premium`, `pack_x_value`
  - Brand×Quality: `brand_x_premium`, `brand_x_organic`
  - Size×Value: `travel_x_value`, `bulk_x_value`

**Text PCA Features (20):**
- Top 20 PCA components from CLIP text embeddings (ViT-B/32)
- Explains ~60% of embedding variance
- Captures semantic price signals

### **Algorithm:**
- **Model:** LightGBM (GBDT)
- **Transform:** Square root on target
- **Training:** Full training data (75,000 samples)
- **Hyperparameters:**
  - `num_leaves`: 63
  - `learning_rate`: 0.03
  - `num_boost_round`: 2000
  - `lambda_l1/l2`: 0.5

---

## 📈 Performance by Segment

| Segment | Validation SMAPE | % of Data |
|---------|------------------|-----------|
| **Mid-Range** ($10-$50) | **19.01%** ✅✅ | 50.9% |
| **Premium** ($50-$100) | **32.32%** ✅ | 8.3% |
| **Budget** (<$10) | **37.44%** ⚠️ | 38.4% |
| **Luxury** (>$100) | **46.14%** ⚠️ | 2.5% |

**Strong on mid-range, struggles with budget/luxury extremes**

---

## 🎯 Key Insights

### **1. Text Embeddings ARE Useful (When Used Correctly)**

**Wrong approach (Direct regression):**
```
Text embeddings only → 60.26% SMAPE ❌
```

**Right approach (PCA + Tabular):**
```
Text PCA + Tabular → 27.86% SMAPE ✅
```

### **2. Feature Importance:**

**Top 10 Features:**
1. 📊 `value` (6,074 importance)
2. 📝 `text_pca_11` (5,426)
3. 📝 `text_pca_10` (5,372)
4. 📝 `text_pca_15` (5,326)
5. 📝 `text_pca_1` (5,234)
6. 📝 `text_pca_16` (5,233)
7. 📝 `text_pca_2` (5,177)
8. 📝 `text_pca_12` (5,145)
9. 📝 `text_pca_13` (5,125)
10. 📝 `text_pca_0` (5,121)

**9/10 top features are text PCA** - embeddings dominate!

**Importance by category:**
- Text PCA: 81.1%
- Tabular: 18.9%

### **3. Image Embeddings: NOT Used**

**Analysis showed:**
- Image CLIP embeddings: R² = -0.0085 (worse than baseline)
- CLIP not trained for product price prediction from images
- Would need fine-tuning (8+ hours, high risk)
- **Decision:** Skip images, focus on text + tabular

---

## 📁 Files for Kaggle

### **Upload to Kaggle:**
1. `KAGGLE_submission_enhanced.py` - Main script
2. `train_text_embeddings_clip.npy` - Pre-generated (outputs/)
3. `test_text_embeddings_clip.npy` - Pre-generated (outputs/)
4. `train_with_features.csv` - Engineered features (dataset/)
5. `test_with_features.csv` - Engineered features (dataset/)

### **Run on Kaggle:**
```python
python /kaggle/working/amazon-ml-chal/KAGGLE_submission_enhanced.py
```

**Output:** `submission_enhanced.csv`

---

## 🔧 How to Generate Test Features

If `test_with_features.csv` doesn't exist, run:

```python
python generate_features.py
```

This will extract the same 22 tabular features from test set.

---

## 📊 Validation Results

**5-Fold Cross-Validation:**
- Fold 1: 28.35%
- Fold 2: 27.81%
- Fold 3: 27.77%
- Fold 4: 27.66%
- Fold 5: 27.72%

**Mean:** 27.86%  
**Std:** 0.25% (very stable!)

---

## 💡 Why This Works

### **1. Engineered Features Capture Structure:**
- `value × pack_size` → quantity pricing
- `has_premium × value` → premium larger packs
- `brand` → manufacturer positioning

### **2. Text PCA Captures Semantics:**
- Product descriptions → quality signals
- "organic", "gourmet", "luxury" keywords
- Category information (coffee vs water)

### **3. SQRT Transform Balances Ranges:**
- Less aggressive than log for cheap items
- Better for wide price range ($0.13 - $2,796)
- Improves mid-range prediction (50% of data)

---

## 🚀 Next Steps (Optional Improvements)

### **Ensemble (Expected: 26-28% SMAPE):**
```python
# Train multiple models
- LightGBM (current)
- CatBoost  
- XGBoost

# Average predictions
final = 0.4*lgb + 0.3*cat + 0.3*xgb
```

### **More PCA Components (Test 30-50):**
```python
# Current: 20 components (60% variance)
# Try: 30-50 components (70-80% variance)
# Risk: May overfit
```

### **Log Transform (Alternative):**
```python
# Current: SQRT
# Try: log1p (might help luxury segment)
```

---

## 📝 Submission Checklist

- [x] Trained on full training data (75,000 samples)
- [x] PCA fitted on train, transformed test
- [x] Label encoders handle unseen categories
- [x] Predictions clipped to positive values
- [x] Inverse transform applied (sqrt² )
- [x] Submission format: id, price
- [x] Sanity checks: reasonable distribution

---

## 🎯 Expected Results

**Validation:** 27.86% SMAPE  
**Expected LB:** 27-30% SMAPE (±2% variance)

**If LB is:**
- **<28%:** ✅✅ Excellent! Model generalizes well
- **28-30%:** ✅ Good! As expected
- **30-32%:** ⚠️ Some overfitting, but acceptable
- **>32%:** ❌ Investigate: test distribution mismatch?

---

## 🔬 What Didn't Work

1. **Image CLIP Embeddings:** R² = -0.01 (completely useless)
2. **Direct Text Regression:** 60% SMAPE (too weak alone)
3. **Random Forest:** Overfits to leaked features
4. **Log Transform:** Worse than SQRT for this data

---

## 👏 Credits

**Approach:** Incremental feature validation  
**Baseline:** 30.70% SMAPE (tabular only)  
**Enhancement:** +Text PCA → 27.86% SMAPE  

**Key Learning:** Embeddings work best as supplementary features, not standalone!
