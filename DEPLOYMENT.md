# Kaggle Deployment Guide - ML Challenge 2025

**Target Environment**: Kaggle P100 GPU (16GB)  
**Expected Runtime**: 2-3 hours  
**Expected SMAPE**: 8-12%

---

## 📦 Files to Upload to Kaggle

### Essential Files (Required):
```
kaggle.py                    # Main training script
requirements.txt             # Dependencies
```

### Module Files (Required):
```
config/
  └── config.py              # Hyperparameters

src/
  ├── feature_extraction.py  # IPQ, keywords, brand extraction
  ├── text_embeddings.py     # Sentence-transformers wrapper
  ├── image_processing.py    # ResNet18 feature extraction
  ├── models.py              # LightGBM + MLP + CV
  └── ensemble.py            # Weighted ensemble + SMAPE
```

### Optional Files (for reference):
```
README.md                    # Documentation
comprehensive_eda.ipynb      # EDA analysis
process.md                   # Methodology notes
```

---

## 🚀 Deployment Steps

### Step 1: Create Kaggle Notebook

1. Go to Kaggle.com and create new notebook
2. Enable GPU: Settings → Accelerator → GPU (P100)
3. Enable Internet: Settings → Internet → On

### Step 2: Upload Files

**Option A: Upload via Notebook UI**
- Click "Add Data" → "Upload"
- Upload all files from `src/` and `config/`
- Upload `kaggle.py`

**Option B: Create Dataset**
- Package `src/` and `config/` as zip
- Upload as private dataset
- Add dataset to notebook

### Step 3: Install Dependencies

**In first cell:**
```python
!pip install sentence-transformers lightgbm -q
```

**Verify installation:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Step 4: Configure Paths

**Edit `config/config.py`:**
```python
# Change these lines:
DATA_PATH = '../input/ml-challenge-2025'  # Or your Kaggle dataset path
OUTPUT_PATH = './'  # Kaggle working directory
```

### Step 5: Run Training

**Copy-paste `kaggle.py` content into cell and run:**
```python
# Or execute as file:
exec(open('kaggle.py').read())
```

**Monitor progress:**
- Watch for "STEP [X/8]" messages
- Check GPU utilization: `!nvidia-smi`
- Expected timeline:
  - Feature engineering: ~5 min
  - Text embeddings: ~30-40 min
  - Image embeddings: ~45-60 min (many failures expected)
  - Model training: ~40-50 min
  - Ensemble & predictions: ~5 min

### Step 6: Submit Results

1. Download `submission.csv` from notebook output
2. Submit to competition leaderboard
3. Check SMAPE score

---

## ⚙️ Configuration Options

### Memory Optimization (if OOM):
```python
# In config/config.py:
BATCH_SIZE = 128  # Reduce from 256
EPOCHS = 30       # Reduce from 50
N_SPLITS = 3      # Reduce from 5
```

### Speed Optimization:
```python
# Skip image processing (faster but lower accuracy):
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.70,
    'text_mlp': 0.30,
    'image_mlp': 0.00,  # Set to 0
}

# In kaggle.py, comment out image sections:
# # STEP 4: IMAGE EMBEDDINGS (skip)
# train_image_embeddings = np.zeros((len(train_df), 512))
# test_image_embeddings = np.zeros((len(test_df), 512))
```

### Accuracy Optimization:
```python
# More folds (slower but better):
N_SPLITS = 7

# Stronger LightGBM:
LIGHTGBM_PARAMS = {
    ...
    'num_leaves': 63,      # More complex trees
    'learning_rate': 0.03, # Slower learning
}
```

---

## 🐛 Common Issues & Fixes

### Issue 1: Import Errors
**Error**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Fix**:
```python
!pip install sentence-transformers transformers -q
```

### Issue 2: Path Errors
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'dataset/train.csv'`

**Fix**: Update `DATA_PATH` in `config/config.py`:
```python
DATA_PATH = '../input/your-dataset-name'
```

### Issue 3: CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Fix**: Reduce batch size or skip images (see Configuration Options above)

### Issue 4: Image Download Failures
**This is EXPECTED**. Many images will fail to download (throttling, dead links). The model handles this by using zero vectors.

**Stats**: Expect ~20-30% successful downloads. This is factored into the ensemble weights.

### Issue 5: Slow Training
**Solution**: Use fewer folds or skip image processing:
```python
N_SPLITS = 3  # Instead of 5
# Skip images or reduce image batch processing
```

---

## 📊 Expected Results

### OOF (Out-of-Fold) Performance:
```
LightGBM:    12-15% SMAPE
Text MLP:    18-22% SMAPE
Image MLP:   22-28% SMAPE
Ensemble:    8-12% SMAPE ✓ TARGET
```

### Test Predictions Distribution:
```
Mean:   ~$23-25
Median: ~$14-16
Min:    $0.13 (clipped to 0.01)
Max:    ~$2796 (clipped to 3000)
```

---

## 🔍 Validation Checklist

Before submitting:

- [ ] GPU is enabled in Kaggle settings
- [ ] All dependencies installed (`sentence-transformers`, `lightgbm`)
- [ ] `DATA_PATH` points to correct Kaggle input folder
- [ ] Training completes all 8 steps without errors
- [ ] `submission.csv` has 75,000 rows
- [ ] No negative prices in submission
- [ ] OOF SMAPE is between 8-15%

---

## 🎯 Submission Checklist

Required files to submit:

1. **submission.csv** - Test predictions
2. **Documentation (1-page)** - Use `Documentation_template.md`:
   - Methodology: Hybrid ensemble (LightGBM + Text MLP + Image MLP)
   - Features: IPQ extraction, quality keywords, embeddings
   - Model: 5-fold CV with weighted ensemble
   - Performance: X% SMAPE on validation

---

## 💡 Pro Tips

1. **Cache embeddings**: After first run, save embeddings and reload instead of regenerating
2. **Test on sample**: Use `sample_test.csv` first to verify pipeline (~1 min vs 2 hours)
3. **Monitor GPU**: Run `!nvidia-smi` periodically to check utilization
4. **Ensemble weights matter**: The optimization step can give 1-2% SMAPE improvement
5. **Don't worry about image failures**: Model is designed to handle this

---

## 📞 Support

If stuck, check:
- Kaggle kernel logs for error messages
- README.md troubleshooting section
- Test locally with `test_modules.py` first

---

**Ready to deploy? Let's ace that leaderboard! 🚀**
