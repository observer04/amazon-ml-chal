# Amazon ML Challenge - Complete Roadmap (Post-Image Extraction)

## CONTEXT SUMMARY

### Current Situation
- **Current SMAPE**: 57.75% (rank ~3000)
- **Target SMAPE**: 40-45% (leader at 40%)
- **Gap**: ~18% improvement needed
- **Hypothesis**: Images contain price-predictive information not captured by CLIP
- **Status**: Stuck on Marqo-ecommerce-L loading (meta tensor error)

### Previous Attempts
- CLIP ViT-B/32: max_corr = 0.0089 (FAILED - essentially random)
- Text CLIP: max_corr = 0.1089 (12x better than image CLIP)
- Statistical features (resolution, aspect ratio): R² ≈ 0 (FAILED)
- PCA on CLIP embeddings: All R² ≈ 0 (FAILED)

### Available Models (E-Commerce Specific, <8B params)

#### Option 1: Marqo/marqo-ecommerce-embeddings-L (CURRENT - BROKEN)
- **Status**: Meta tensor loading error
- **Params**: 652M
- **Embeddings**: 1024-dim
- **Training**: 3M Amazon + 1M Google Shopping products
- **Performance**: 38.9% better than Amazon-Titan
- **Issue**: `NotImplementedError: Cannot copy out of meta tensor`

#### Option 2: Marqo/marqo-fashionSigLIP (PLAN B v2 - RECOMMENDED) ⭐
- **Status**: Ready to test
- **Params**: 400M (well under limit)
- **Embeddings**: 768-dim
- **Training**: 1M+ fashion/e-commerce products
- **Advantage**: SigLIP-based, no meta tensor issues, production-proven (leboncoin)
- **Expected**: 3-8x better correlation than CLIP (0.03-0.08 range)
- **File**: `kaggle_plan_b_fashion_siglip.py`

#### Option 3: google/siglip-so400m-patch14-384
- **Params**: 400M
- **Embeddings**: 1152-dim
- **Training**: WebLI dataset (web-scale image-text)
- **Advantage**: Base model that Marqo fine-tunes from, very reliable

#### Option 4: Marqo/marqo-fashionCLIP
- **Params**: 150M (lightest)
- **Embeddings**: 512-dim
- **Training**: 1M fashion products

### Dataset Info
- Train: 75,000 samples with image URLs
- Test: 75,000 samples with image URLs
- All images accessible
- Features: PRODUCT_TITLE, DESCRIPTION, BULLET_POINTS, PRODUCT_TYPE_ID, PRODUCT_LENGTH (target)

---

## PHASE 1: IMAGE EMBEDDING EXTRACTION [IMMEDIATE - 1 hour]

### Step 1.1: Test Marqo-ecommerce Fix (5 minutes)
**Location**: On Kaggle

```bash
cd /kaggle/working/amazon-ml-chal
git pull origin main
python kaggle_extract_marqo_embeddings.py
```

**Expected Outcomes**:
- ✅ **SUCCESS**: Model loads → Continue to Phase 2
- ❌ **FAILURE**: Meta tensor error again → Go to Step 1.2

### Step 1.2: Switch to Marqo-FashionSigLIP (50 minutes)
**Location**: Local machine first, then Kaggle

**Local Steps**:
```bash
cd /home/observer/projects/amazonchal
git add kaggle_plan_b_fashion_siglip.py
git commit -m "Add Plan B v2: Marqo-FashionSigLIP (400M, e-commerce specific)"
git push origin main
```

**Kaggle Steps**:
```bash
cd /kaggle/working/amazon-ml-chal
git pull origin main
python kaggle_plan_b_fashion_siglip.py
```

**Expected Runtime**: 30-45 minutes for 150K images
**Expected Output**:
- `outputs/train_fashion_siglip_embeddings.npy` (768-dim × 75K = ~230MB)
- `outputs/test_fashion_siglip_embeddings.npy` (768-dim × 75K = ~230MB)
- Correlation analysis printed to console

### Step 1.3: Analyze Correlation Results (5 minutes)

**Decision Criteria**:

| Max Correlation | Verdict | Action |
|----------------|---------|--------|
| > 0.05 | ✅ STRONG | Go to Phase 2 (Training script) |
| 0.02 - 0.05 | ⚠️ MODERATE | Go to Phase 3 (Ensemble test) |
| < 0.02 | ❌ WEAK | Go to Phase 4 (Alternative approaches) |

**Comparison Baselines**:
- CLIP: 0.0089
- Text CLIP: 0.1089
- Need: > 0.02 minimum to justify using

---

## PHASE 2: TRAINING WITH IMAGE FEATURES [STRONG SIGNAL - 3 hours]

### Prerequisites
- Max correlation > 0.05 from Phase 1
- Image embeddings saved as .npy files

### Step 2.1: Download Embeddings from Kaggle (10 minutes)

**Option A: Git (if embeddings < 100MB)**:
```bash
cd /kaggle/working/amazon-ml-chal
git add outputs/*.npy
git commit -m "Add FashionSigLIP embeddings"
git push origin main
```

**Option B: Manual Download** (if larger):
1. Click on `outputs/train_fashion_siglip_embeddings.npy` → Download
2. Click on `outputs/test_fashion_siglip_embeddings.npy` → Download
3. Move to local `dataset/` folder

**Option C: Copy-paste approach**:
- Too large for this, use Option A or B

### Step 2.2: Create Training Script (30 minutes)

**Create**: `train_with_image_features.py`

**Key Components**:
```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

# Load embeddings
train_img_emb = np.load('dataset/train_fashion_siglip_embeddings.npy')
test_img_emb = np.load('dataset/test_fashion_siglip_embeddings.npy')

# Load existing features
train_df = pd.read_csv('dataset/train_with_features.csv')
test_df = pd.read_csv('dataset/test_with_features.csv')

# Combine features
# 1. Select top K image dimensions (highest correlation)
# 2. Add to existing feature set
# 3. Feature names: img_emb_0, img_emb_1, ..., img_emb_K

# Model configuration
params = {
    'objective': 'regression',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1
}

# 5-fold cross-validation
# Track feature importance
# Generate test predictions
```

**Feature Selection Strategy**:
1. **Top K correlated**: Select K dimensions with highest |correlation| (K=50-100)
2. **PCA**: Reduce 768→50 dims while keeping 95% variance
3. **All dimensions**: Use all 768 (if not overfitting)

### Step 2.3: Run Training Locally (1 hour)

```bash
cd /home/observer/projects/amazonchal
python train_with_image_features.py
```

**Expected Output**:
- Fold 1-5 SMAPE scores
- Average CV SMAPE
- Feature importance (top 20)
- Test predictions saved to `test_out.csv`

**Success Criteria**:
- CV SMAPE < 53% (5% improvement)
- Image features in top 50 by importance
- Consistent across folds (std < 0.5%)

### Step 2.4: Validate Submission (5 minutes)

```bash
python validate_submission.py test_out.csv
```

**Checks**:
- 75,000 rows
- No missing values
- Reasonable price range (5-200)
- No duplicates

### Step 2.5: Submit to Kaggle (5 minutes)

1. Go to competition page
2. Upload `test_out.csv`
3. Add description: "LightGBM + FashionSigLIP image embeddings (768-dim)"
4. Submit

**Expected Result**:
- **Target**: 52-53% SMAPE (5% improvement)
- **Minimum**: 55% SMAPE (2% improvement)
- **Risk**: 57-58% SMAPE (no improvement → images don't help)

### Step 2.6: Iterate Based on Results (1 hour)

**If improvement < 2%**:
- Try PCA on embeddings
- Use only top 10-20 correlated dimensions
- Ensemble with baseline model

**If improvement 2-5%**:
- Try larger K (more dimensions)
- Test Marqo-fashionCLIP (512-dim) as alternative
- Ensemble multiple image models

**If improvement > 5%**:
- Add text-image multimodal features
- Try other e-commerce models
- Fine-tune on competition data (advanced)

---

## PHASE 3: ENSEMBLE APPROACH [MODERATE SIGNAL - 2 hours]

### Prerequisites
- Max correlation 0.02-0.05 from Phase 1
- OR Phase 2 gave < 2% improvement

### Step 3.1: Create Ensemble Script

**Strategy**: Stack image features with baseline model predictions

```python
# Create meta-features from images
# 1. Raw embeddings (top 20 correlated dims)
# 2. Image embedding predictions (separate LightGBM)
# 3. Combine with baseline predictions

# Layer 1: Base models
model_baseline = lgb.train(...)  # Existing features only
model_images = lgb.train(...)    # Image features only

# Layer 2: Meta-model
predictions_baseline = model_baseline.predict(X_val)
predictions_images = model_images.predict(X_img_val)
meta_features = np.column_stack([predictions_baseline, predictions_images, X_val])

model_meta = lgb.train(..., meta_features, y_val)
```

### Step 3.2: Test Multiple Weights

```python
# Simple weighted average
for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
    pred_ensemble = (1-w) * pred_baseline + w * pred_images
    smape = calculate_smape(y_true, pred_ensemble)
```

**Expected**:
- 2-4% improvement if optimal weight found
- Image weight likely 0.1-0.3 (modest contribution)

---

## PHASE 4: ALTERNATIVE APPROACHES [WEAK SIGNAL - 4 hours]

### Prerequisites
- Max correlation < 0.02 from Phase 1
- OR Phase 2 + 3 showed no improvement

### Option A: Try Other E-Commerce Models

**Models to test**:
1. `google/siglip-so400m-patch14-384` (1152-dim)
2. `Marqo/marqo-fashionCLIP` (512-dim, lighter)
3. Self-supervised: `facebook/dinov2-base` (768-dim)

**Process**: Repeat Phase 1 with each model

### Option B: Image Metadata Features

**Extract from images**:
```python
from PIL import Image

def extract_metadata(img):
    return {
        'width': img.width,
        'height': img.height,
        'aspect_ratio': img.width / img.height,
        'file_size': len(img.tobytes()),
        'mode': img.mode,  # RGB, RGBA, L, etc.
        'is_square': abs(img.width - img.height) < 10,
        'is_landscape': img.width > img.height,
        'is_portrait': img.height > img.width,
        'total_pixels': img.width * img.height
    }
```

**Hypothesis**: 
- Professional product photos have consistent dimensions
- Expensive products may have higher-res images
- Image quality correlates with product quality/price

### Option C: Aesthetic Quality Scoring

**Use**: LAION Aesthetics Predictor

```python
from transformers import pipeline

aesthetic_scorer = pipeline("image-classification", 
                           model="cafeai/cafe_aesthetic")

def get_aesthetic_score(image):
    result = aesthetic_scorer(image)
    return result[0]['score']
```

**Hypothesis**: More expensive products have better photography

### Option D: Color Analysis

**Extract dominant colors and statistics**:
```python
from sklearn.cluster import KMeans

def extract_color_features(img):
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)
    
    # K-means to find dominant colors
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(pixels)
    
    return {
        'dominant_colors': kmeans.cluster_centers_,
        'mean_brightness': pixels.mean(),
        'std_brightness': pixels.std(),
        'mean_r': pixels[:, 0].mean(),
        'mean_g': pixels[:, 1].mean(),
        'mean_b': pixels[:, 2].mean(),
    }
```

### Option E: Focus on Text Improvements

**If images truly don't help**:

1. **Better text embeddings**:
   - Try `sentence-transformers/all-MiniLM-L6-v2`
   - Try `BAAI/bge-base-en-v1.5`
   - Try product-specific BERT models

2. **NLP feature engineering**:
   - Named entity recognition (brands, materials)
   - Sentiment analysis of descriptions
   - Keyword extraction (luxury terms, quality indicators)
   - Text complexity metrics (long descriptions = detailed = expensive?)

3. **Interaction features**:
   - title_length × product_type
   - description_sentiment × category
   - brand × material

---

## PHASE 5: ADVANCED OPTIMIZATION [If at 50-52% - 6 hours]

### Step 5.1: Hyperparameter Tuning

**Use Optuna for LightGBM**:
```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    # ... train and return CV score
```

**Target**: 1-2% improvement (51-50% SMAPE)

### Step 5.2: Feature Engineering Deep Dive

**Create interaction features**:
```python
# Polynomial features from top predictors
from sklearn.preprocessing import PolynomialFeatures

top_features = ['PRODUCT_LENGTH_std', 'title_length', 'desc_length']
poly = PolynomialFeatures(degree=2, interaction_only=True)
poly_features = poly.fit_transform(df[top_features])
```

**Category-specific models**:
```python
# Train separate models for each product category
for category in df['PRODUCT_TYPE_ID'].unique():
    mask = df['PRODUCT_TYPE_ID'] == category
    model_category = lgb.train(..., df[mask])
```

### Step 5.3: Ensemble Multiple Models

**Combine**:
- LightGBM (main)
- XGBoost (diversity)
- CatBoost (categorical features)
- Linear regression (on top features)

**Weighted average**:
```python
pred_final = (0.5 * pred_lgb + 
              0.25 * pred_xgb + 
              0.15 * pred_cat + 
              0.1 * pred_linear)
```

### Step 5.4: Pseudo-Labeling (If public LB good)

**If public LB < 48%**:
1. Use model predictions on test set
2. Select high-confidence predictions (low uncertainty)
3. Add to training set
4. Retrain model
5. Repeat

**Risk**: Overfitting to public LB

---

## PHASE 6: FINAL PUSH [If at 45-48% - 4 hours]

### Competition-Specific Optimizations

1. **Leak detection**: Check for test samples in train (image hash, text similarity)
2. **External data**: Search for Amazon product datasets on Kaggle/GitHub
3. **Competition forum**: Read discussions for hints about data quirks
4. **Ensemble with others**: If collaboration allowed, blend with other competitors

### Last-Minute Checks

1. **Cross-validation matches LB**: Ensure CV score correlates with LB
2. **No data leakage**: Validate feature generation doesn't use test info
3. **Reproducibility**: Test script runs end-to-end without errors
4. **Multiple submissions**: Use all daily submission slots (2-5 per day)

---

## CRITICAL FILES REFERENCE

### Current Scripts

1. **kaggle_extract_marqo_embeddings.py**
   - Status: Has meta tensor bug fix (3-strategy loading)
   - Model: Marqo/marqo-ecommerce-embeddings-L (652M)
   - Output: 1024-dim embeddings

2. **kaggle_plan_b_fashion_siglip.py** ⭐ **USE THIS**
   - Status: Ready to run
   - Model: Marqo/marqo-fashionSigLIP (400M)
   - Output: 768-dim embeddings
   - Expected: 0.03-0.08 correlation

3. **baseline_model.py**
   - Current best: 57.75% SMAPE
   - Features: Text length, word counts, basic stats
   - No images used

4. **generate_features.py**
   - Creates: train_with_features.csv, test_with_features.csv
   - Features: Text stats, CLIP embeddings (failed), engineered features

5. **validate_submission.py**
   - Checks submission format
   - Validates predictions

### Data Files

1. **dataset/train.csv** (75K rows)
   - Columns: image_link, PRODUCT_TITLE, DESCRIPTION, BULLET_POINTS, PRODUCT_TYPE_ID, PRODUCT_LENGTH

2. **dataset/test.csv** (75K rows)
   - Same columns except PRODUCT_LENGTH

3. **dataset/train_with_features.csv**
   - Engineered features added
   - CLIP embeddings (512 dims, non-predictive)

4. **dataset/test_with_features.csv**
   - Same features for test set

### Output Files (To Be Created)

1. **outputs/train_fashion_siglip_embeddings.npy**
   - Shape: (75000, 768)
   - Size: ~230MB
   - From: kaggle_plan_b_fashion_siglip.py

2. **outputs/test_fashion_siglip_embeddings.npy**
   - Shape: (75000, 768)
   - Size: ~230MB
   - From: kaggle_plan_b_fashion_siglip.py

---

## DECISION TREE SUMMARY

```
START: Image Embedding Extraction
├─ Try Marqo-ecommerce fix
│  ├─ ✅ Works → Use 1024-dim embeddings
│  └─ ❌ Fails → Use FashionSigLIP (768-dim)
│
└─ Analyze Correlations
   ├─ > 0.05: STRONG
   │  └─ Phase 2: Full Training (Target: 52% SMAPE)
   │     ├─ Success (53-54%) → Phase 5: Optimize
   │     └─ Failure (56-57%) → Phase 4: Alternatives
   │
   ├─ 0.02-0.05: MODERATE
   │  └─ Phase 3: Ensemble (Target: 54-55% SMAPE)
   │     ├─ Success → Phase 5: Optimize
   │     └─ Failure → Phase 4: Alternatives
   │
   └─ < 0.02: WEAK
      └─ Phase 4: Try Alternatives
         ├─ Other models (SigLIP-base, DINOv2)
         ├─ Image metadata
         ├─ Aesthetic scoring
         └─ Focus on text improvements
```

---

## TIME ESTIMATES

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Image Extraction | 1 hour | 1 hour |
| Phase 2: Training (if strong) | 3 hours | 4 hours |
| Phase 3: Ensemble (if moderate) | 2 hours | 3 hours |
| Phase 4: Alternatives (if weak) | 4 hours | 5 hours |
| Phase 5: Optimization | 6 hours | 10 hours |
| Phase 6: Final push | 4 hours | 14 hours |

**Total time to competitive score (45-50%)**: 10-14 hours
**Total time to leaderboard climb (40-45%)**: 20-30 hours + luck

---

## SUCCESS METRICS

| SMAPE | Status | Action |
|-------|--------|--------|
| 40-45% | 🏆 LEADER | Keep iterating, aim for top 10 |
| 45-50% | 🎯 STRONG | Phase 5+6 optimization |
| 50-55% | ✅ GOOD | Images helped, keep improving |
| 55-57% | ⚠️ MODEST | Marginal gain, try alternatives |
| 57%+ | ❌ NO GAIN | Images don't help this way |

---

## IMPORTANT NOTES

### Context from Previous Work

1. **Validation Bug**: Initially thought validation was 27.86%, but that was wrong methodology. Correct validation: 55.89% (matches LB)

2. **CLIP Failed**: The pre-extracted CLIP embeddings are essentially random (0.0089 correlation). Don't use them.

3. **Text Works Better**: Text CLIP (0.1089) is 12x better than image CLIP (0.0089). This is unusual for a visual product dataset.

4. **Leader Uses Images**: Competition organizers explicitly mention images help. Leader at 40% likely uses them effectively.

5. **E-commerce Specific Models**: Generic ImageNet models (ResNet50) likely won't work. Need models trained on product images (Marqo, SigLIP).

### Kaggle Environment

- **GPU**: T4 (16GB VRAM)
- **RAM**: 13GB
- **Time**: 9-hour limit
- **Internet**: Required for downloading models
- **Storage**: 5GB for outputs

### Git Authentication on Kaggle

If git push asks for credentials:
```bash
# Use Personal Access Token
Username: observer04
Password: <paste PAT instead>
```

Or download files manually from Kaggle output panel.

---

## RECOVERY INSTRUCTIONS (If Context Lost)

### Key Information to Remember

1. **Problem**: Predict PRODUCT_LENGTH (price proxy) from product images + text
2. **Current best**: 57.75% SMAPE
3. **Target**: 40-45% SMAPE
4. **Main hypothesis**: Images contain price signals
5. **Previous CLIP failed**: Need better e-commerce models

### Files to Check First

```bash
cd /home/observer/projects/amazonchal

# See current state
ls -lh dataset/
ls -lh outputs/

# Check latest script
cat kaggle_plan_b_fashion_siglip.py

# Check documentation
cat NEXT_PHASES_ROADMAP.md
cat PROJECT_SUMMARY.md
cat CRITICAL_FINDINGS.md
```

### Quick Start Commands

```bash
# On Kaggle: Extract image embeddings
cd /kaggle/working/amazon-ml-chal
git clone https://github.com/observer04/amazon-ml-chal.git .
python kaggle_plan_b_fashion_siglip.py

# On local: Train model
cd /home/observer/projects/amazonchal
# Create train_with_image_features.py based on correlation results
python train_with_image_features.py

# Submit
python validate_submission.py test_out.csv
# Upload to Kaggle competition
```

---

## CONTACT POINTS FOR CONTINUATION

If conversation ends and you need to continue:

1. **Read this file first**: `NEXT_PHASES_ROADMAP.md`
2. **Check current state**: Look at git log, check outputs/ folder
3. **Review correlation results**: From Phase 1 script output
4. **Follow decision tree**: Based on correlation value
5. **Reference**: `PROJECT_SUMMARY.md` for overall context

**Most likely next step**: Run `kaggle_plan_b_fashion_siglip.py` on Kaggle, then decide based on correlation results (Phase 1 → Phase 2/3/4).

---

## END OF ROADMAP

Good luck! The path to 40% SMAPE is:
1. ✅ Get working image embeddings (FashionSigLIP)
2. 🎯 Find correlation > 0.05 
3. 🚀 Integrate into training
4. 📈 Iterate and optimize

Remember: E-commerce specific models > Generic models. Images should help if you use the right model!
