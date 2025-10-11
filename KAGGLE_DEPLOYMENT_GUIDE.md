# 🚀 Kaggle Deployment Guide - CRITICAL PATH

**Status**: ✅ Features ready, code ready, waiting for GitHub push + Kaggle execution

**DO NOT RUN LOCALLY** - All remaining work must be done on Kaggle P100 GPU

---

## ⏰ IMMEDIATE ACTIONS (Next 30 minutes)

### 1. Create GitHub Repository (5 minutes)

```bash
# Go to: https://github.com/new
# Repository name: ml-challenge-2025-amazon-pricing
# Description: Hybrid ensemble for Amazon product pricing (SMAPE optimization)
# Visibility: Private (recommended)
# DO NOT check "Initialize with README"
# Click "Create repository"

# After creation, run these commands:
cd /home/observer/projects/amazonchal

# Add remote (REPLACE YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/ml-challenge-2025-amazon-pricing.git

# Push code
git branch -M main
git push -u origin main

# Verify
git remote -v
```

**What gets pushed:**
- ✅ `src/` - All 5 module files (feature_extraction, text_embeddings, image_processing, models, ensemble)
- ✅ `config/config.py` - Hyperparameters
- ✅ `kaggle.py` - Main orchestration script
- ✅ `requirements.txt` - Dependencies
- ✅ `.gitignore` - Excludes dataset/, amzml/, outputs/
- ✅ Documentation - README.md, DEPLOYMENT.md, process.md

**What does NOT get pushed** (per .gitignore):
- ❌ `dataset/` - Too large (upload to Kaggle separately)
- ❌ `amzml/` - Virtual environment (not needed)
- ❌ `outputs/` - Generated files

---

### 2. Upload Feature CSVs to Kaggle Dataset (10 minutes)

```bash
# Option A: Via Kaggle Website (RECOMMENDED)
# 1. Go to: https://www.kaggle.com/datasets
# 2. Click "New Dataset"
# 3. Title: "Amazon Pricing Challenge - Engineered Features"
# 4. Upload these files:
#    - dataset/train_with_features.csv (75k rows, 28 columns, ~50MB)
#    - dataset/test_with_features.csv (75k rows, 27 columns, ~48MB)
# 5. Visibility: Private
# 6. Click "Create"
# 7. Copy dataset path (e.g., "your-username/amazon-pricing-features")

# Option B: Via Kaggle API (if configured)
# pip install kaggle
# kaggle datasets init -p dataset/
# Edit dataset-metadata.json
# kaggle datasets create -p dataset/
```

**Dataset contains:**
- `train_with_features.csv`: 24 engineered features (value, unit, brand, quality keywords, etc.)
- `test_with_features.csv`: 23 features (same minus price)

**Note dataset path** - you'll need it for config.py:
```
Example: ../input/amazon-pricing-features/train_with_features.csv
```

---

### 3. Create Kaggle Notebook (10 minutes)

```bash
# 1. Go to: https://www.kaggle.com/code
# 2. Click "New Notebook"
# 3. Settings (right sidebar):
#    - Accelerator: GPU P100 ✅ CRITICAL!
#    - Internet: On ✅ (for model downloads)
#    - Language: Python
# 4. Add Data:
#    - Click "Add Data" → "Your Datasets"
#    - Select "Amazon Pricing Challenge - Engineered Features"
#    - Click "Add"
# 5. Add competition data:
#    - Click "Add Data" → "Competitions"
#    - Search for "ML Challenge 2025" (or correct competition name)
#    - Add train.csv and test.csv (original data for image links)
```

---

## 🔧 Setup Kaggle Notebook

### Cell 1: Install Dependencies

```python
# Install required packages
!pip install sentence-transformers lightgbm -q

# Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

### Cell 2: Clone GitHub Repository

```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/ml-challenge-2025-amazon-pricing.git
%cd ml-challenge-2025-amazon-pricing

# Verify files
!ls -la src/
!ls -la config/
```

### Cell 3: Update Configuration

```python
# Edit config/config.py to point to Kaggle paths
import os

# Update these paths in config/config.py
DATA_PATH = '../input/amazon-pricing-features'  # Your dataset path
ORIGINAL_DATA_PATH = '../input/ml-challenge-2025'  # Competition data (for image links)
OUTPUT_PATH = './'  # Kaggle working directory

# Verify paths exist
print("Feature data:", os.listdir(DATA_PATH))
print("Original data:", os.listdir(ORIGINAL_DATA_PATH))
```

Or manually edit `config/config.py`:
```python
# OLD (local):
DATA_PATH = 'dataset'

# NEW (Kaggle):
DATA_PATH = '../input/amazon-pricing-features'
```

### Cell 4: Run Training Pipeline

```python
# Execute kaggle.py
exec(open('kaggle.py').read())
```

**Expected output:**
```
[1/8] Loading data...
[2/8] Engineering features...  # Will load from CSV now
[3/8] Generating text embeddings...  ⏰ 30-40 minutes
[4/8] Generating image embeddings... ⏰ 45-60 minutes
[5/8] Training models with 5-fold CV... ⏰ 50-70 minutes
[6/8] Optimizing ensemble weights...
[7/8] Generating test predictions...
[8/8] Creating submission file...

✅ Submission saved: submission.csv
OOF SMAPE: X.XX%
```

---

## ⏱️ Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Text embeddings | 30-40 min | sentence-transformers on 150k samples |
| Image embeddings | 45-60 min | ResNet18 + download 150k images (100% success expected) |
| LightGBM training | 15-20 min | 5-fold CV on 24 features |
| Text MLP training | 20-30 min | 5-fold CV on 384-dim embeddings |
| Image MLP training | 20-30 min | 5-fold CV on 512-dim embeddings |
| Ensemble optimization | 2-5 min | scipy SLSQP on OOF predictions |
| Test predictions | 5 min | Apply ensemble to test set |
| **TOTAL** | **2.5-3 hours** | On Kaggle P100 GPU |

---

## 📊 Monitoring & Validation

### During Training:

**Watch for:**
- ✅ Text embedding progress bar (150k samples)
- ✅ Image download success rate (~100% expected)
- ✅ LightGBM OOF SMAPE (target: 12-15%)
- ✅ Text MLP OOF SMAPE (target: 18-22%)
- ✅ Image MLP OOF SMAPE (target: 22-28%)
- ✅ Ensemble OOF SMAPE (target: **10-12%**)

**Red flags:**
- 🔴 Image download success <80% → may need to reduce image MLP weight
- 🔴 LightGBM SMAPE >18% → hyperparameter tuning needed
- 🔴 Ensemble SMAPE >15% → weight optimization may have failed

### After Training:

```python
# Check submission file
import pandas as pd
submission = pd.read_csv('submission.csv')

print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:")
print(submission.head(10))

# Validation checks
print(f"\n✅ All sample_ids present: {len(submission) == 75000}")
print(f"✅ No negative prices: {(submission['price'] > 0).all()}")
print(f"✅ No NaN prices: {submission['price'].notna().all()}")

# Price distribution
print(f"\nPrediction stats:")
print(f"  Mean: ${submission['price'].mean():.2f}")
print(f"  Median: ${submission['price'].median():.2f}")
print(f"  Min: ${submission['price'].min():.2f}")
print(f"  Max: ${submission['price'].max():.2f}")
```

Expected predictions:
- Mean: $23-25
- Median: $14-16
- Min: $0.13 (or clipped to $0.01)
- Max: $2,796 (or clipped to $3,000)

---

## 🎯 Submission

### Download submission.csv

```python
# In Kaggle notebook, the file will be in working directory
# Click "Output" tab → Download submission.csv
```

### Submit to Competition

1. Go to competition page
2. Click "Submit Predictions"
3. Upload `submission.csv`
4. Add description: "Hybrid ensemble: LightGBM(55%) + Text MLP(25%) + Image MLP(20%)"
5. Click "Submit"

### Expected Scores

| Scenario | OOF SMAPE | Public LB SMAPE | Position |
|----------|-----------|-----------------|----------|
| **Conservative** | 13-14% | 13-15% | Top 40-50 |
| **Realistic** | 10-12% | 10-13% | Top 25-35 ✅ TARGET |
| **Optimistic** | 9-10% | 9-11% | Top 15-25 |
| **Perfect** | <9% | <10% | Top 10-20 |

**Top 50 cutoff estimate**: ~15% SMAPE
**Our target**: **10-12% SMAPE** → **Top 50 achievable** ✅

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

```python
# Install missing packages
!pip install <package-name> -q
```

### Issue: CUDA Out of Memory

```python
# In config/config.py, reduce:
BATCH_SIZE = 128  # from 256
```

### Issue: Image downloads timing out

```python
# Already handled in code - uses zero vectors for failures
# But if >20% fail, consider reducing image MLP weight in config:
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.60,
    'text_mlp': 0.30,
    'image_mlp': 0.10,  # Reduced from 0.20
}
```

### Issue: Training too slow

```python
# In config/config.py:
N_SPLITS = 3  # from 5 (faster CV)
EPOCHS = 30   # from 50 (faster MLP training)
```

---

## 📝 Post-Submission Checklist

- [ ] OOF SMAPE logged: ____%
- [ ] Public LB SMAPE: ____%
- [ ] Leaderboard position: ____
- [ ] Top 50 achieved? Yes / No
- [ ] Updated process.md with results
- [ ] Created 1-page documentation (Documentation_template.md)
- [ ] Analyzed errors by price segment

---

## 🎉 Success Criteria

**MINIMUM (Top 50)**: 
- ✅ <15% SMAPE on public leaderboard

**TARGET (Top 30)**:
- ✅ <12% SMAPE on public leaderboard
- ✅ Budget segment SMAPE <20%

**STRETCH (Top 20)**:
- ✅ <10% SMAPE on public leaderboard
- ✅ All components working optimally
- ✅ Budget segment handling effective

---

**NEXT IMMEDIATE ACTION**: Create GitHub repo and push code! ⏰ 5 minutes

**Then**: Upload CSVs to Kaggle dataset ⏰ 10 minutes

**Then**: Run kaggle.py on P100 GPU ⏰ 2-3 hours

**GO GO GO!** 🚀
