# Multi-Modal Ensemble Training - Option B (Diagnostic Approach)

## 📋 Overview

This is a **methodical, diagnostic approach** to building a multi-modal ensemble for the Amazon ML Challenge.

**Goal:** Achieve 30-40% SMAPE by combining:
- Image embeddings (CLIP ViT-B/32)
- Text embeddings (CLIP ViT-B/32)
- Tabular features (30 engineered features)

---

## 🎯 Three-Script Pipeline

### **Script 1: Image MLP Diagnostic** (`KAGGLE_script1_image_mlp.py`)
- **Purpose:** Train MLP on CLIP image embeddings, evaluate if visual features predict price
- **Time:** 20-30 minutes
- **Outputs:**
  - `image_mlp_best.pth` (trained model weights)
  - `image_mlp_diagnostics.png` (training curves, SMAPE, prediction distribution)
- **Success Criteria:** Validation SMAPE < 50%

### **Script 2: Text MLP Diagnostic** (`KAGGLE_script2_text_mlp.py`)
- **Purpose:** Train MLP on CLIP text embeddings, compare with Image MLP
- **Time:** 20-30 minutes
- **Outputs:**
  - `text_mlp_best.pth` (trained model weights)
  - `text_mlp_diagnostics.png` (training curves, comparison with Image MLP)
- **Success Criteria:** Validation SMAPE < 55%, correlation with Image < 0.8

### **Script 3: Ensemble & Submission** (`KAGGLE_script3_ensemble.py`)
- **Purpose:** Combine all branches, optimize weights, generate submission
- **Time:** 30 minutes
- **Outputs:**
  - `submission_multimodal_ensemble.csv` (final submission)
  - `ensemble_analysis.png` (model comparison, weights visualization)
- **Success Criteria:** Ensemble SMAPE < best individual model

---

## 🚀 How to Run on Kaggle

### Prerequisites
Make sure you have these files in Kaggle working directory:
```
/kaggle/working/amazon-ml-chal/
├── outputs/
│   ├── train_image_embeddings_clip.npy
│   ├── train_text_embeddings_clip.npy
│   ├── test_image_embeddings_clip.npy
│   └── test_text_embeddings_clip.npy
└── dataset/
    ├── train_with_features.csv
    └── test_with_features.csv
```

### Step-by-Step Execution

#### **Step 1: Run Image MLP Diagnostic**
```python
# In Kaggle notebook cell
!python KAGGLE_script1_image_mlp.py
```

**What to watch:**
- Training loss should decrease smoothly
- Validation SMAPE should be < 50% for good results
- Look for overfitting (large train/val gap)

**Review outputs:**
- Download `image_mlp_diagnostics.png`
- Check if prediction distribution matches actual prices
- Look at scatter plot (predictions vs actual)

---

#### **Step 2: Run Text MLP Diagnostic**
```python
# In Kaggle notebook cell
!python KAGGLE_script2_text_mlp.py
```

**What to watch:**
- Compare Text SMAPE to Image SMAPE (which is better?)
- Check correlation between predictions (< 0.8 is good - means complementary)
- Look for similar convergence pattern as Image MLP

**Review outputs:**
- Download `text_mlp_diagnostics.png`
- Compare with Image MLP plots
- Decide if text adds value or is redundant

---

#### **Step 3: Run Final Ensemble**
```python
# In Kaggle notebook cell
!python KAGGLE_script3_ensemble.py
```

**What to watch:**
- Tabular LightGBM SMAPE (baseline comparison)
- Optimal weights (which modality is strongest?)
- Ensemble improvement over best single model

**Review outputs:**
- Download `submission_multimodal_ensemble.csv`
- Download `ensemble_analysis.png`
- Check if ensemble beats individual models

---

## 📊 Expected Results

### **Optimistic Scenario**
- Image MLP: 35-45% SMAPE
- Text MLP: 40-50% SMAPE
- Tabular LGBM: 45-55% SMAPE
- **Ensemble: 25-35% SMAPE** ✅

### **Realistic Scenario**
- Image MLP: 45-55% SMAPE
- Text MLP: 50-60% SMAPE
- Tabular LGBM: 50-60% SMAPE
- **Ensemble: 35-45% SMAPE**

### **Pessimistic Scenario**
- All models > 60% SMAPE
- Ensemble doesn't help much
- **Need to debug:** Check embeddings quality, try different transforms

---

## 🎯 Decision Points

### After Script 1 (Image MLP):
- **If SMAPE < 45%:** ✅ Excellent! Proceed to Script 2
- **If SMAPE 45-55%:** ⚠️ Okay, continue but may need tuning
- **If SMAPE > 55%:** ❌ Investigate: embeddings quality, architecture, learning rate

### After Script 2 (Text MLP):
- **If correlation < 0.5:** ✅ Great! Modalities capture different signals
- **If correlation 0.5-0.8:** ✅ Good! Some overlap but complementary
- **If correlation > 0.8:** ⚠️ Redundant - ensemble may not help much

### After Script 3 (Ensemble):
- **If ensemble < best single:** ✅ Success! Submit to Kaggle
- **If ensemble ≈ best single:** ⚠️ No synergy, but still submit
- **If ensemble > best single:** ❌ Bug in code or overfitting

---

## 📈 Interpreting Diagnostic Plots

### Training Curves (Loss over epochs):
- **Good:** Smooth decrease, train/val close together
- **Bad:** Jagged, train continues decreasing while val plateaus (overfitting)

### SMAPE over Time:
- **Good:** Steady decrease, converges to low value
- **Bad:** Erratic, or plateaus at high value

### Prediction Distribution:
- **Good:** Orange (predicted) overlaps well with Blue (actual)
- **Bad:** Orange shifted or much narrower/wider than Blue

### Actual vs Predicted Scatter:
- **Good:** Points cluster around red diagonal line
- **Bad:** Wide spread, or systematic bias (above/below line)

---

## 🔧 Troubleshooting

### OOM (Out of Memory) Error:
- Reduce `batch_size` from 256 to 128 or 64
- Reduce `num_workers` from 2 to 0

### Models not improving:
- Check if embeddings loaded correctly (print shapes)
- Try different learning rate (0.001 → 0.0001 or 0.01)
- Increase epochs (100 → 200)

### Ensemble worse than individual models:
- Bug in weight optimization - check constraints
- Predictions not on same scale - check sqrt transform
- One model is garbage - remove it from ensemble

---

## 📁 Files Generated

After running all 3 scripts:

```
image_mlp_best.pth              # Image MLP weights (PyTorch)
text_mlp_best.pth               # Text MLP weights (PyTorch)
image_mlp_diagnostics.png       # Image MLP training analysis
text_mlp_diagnostics.png        # Text MLP training analysis
ensemble_analysis.png           # Final ensemble visualization
submission_multimodal_ensemble.csv  # Final submission for Kaggle
```

---

## 🚀 Next Steps After Submission

1. **Submit to Kaggle** and record public LB score
2. **Compare LB to validation SMAPE:**
   - If close (< 5% difference): ✅ Good generalization!
   - If far (> 10% difference): ⚠️ Distribution mismatch or overfitting
3. **If LB score > 40%:** Consider Option D (CLIP fine-tuning)
4. **If LB score < 30%:** Success! Document results and move to next challenge

---

## 💡 Key Design Decisions

### Why sqrt transform?
- Prices are heavily right-skewed (many cheap items, few expensive)
- sqrt(price) makes distribution more normal
- Helps neural networks train better

### Why separate MLPs for image/text?
- Different modalities may need different architectures
- Easier to debug and tune independently
- Can see which modality contributes most

### Why optimize ensemble weights?
- Equal weights (1/3 each) may be suboptimal
- Some modalities are stronger than others
- Scipy optimization finds best combination

### Why 80/20 split with random_state=42?
- Standard split for validation
- Fixed seed ensures all models use SAME validation set
- Critical for fair comparison

---

## ⏱️ Total Time Estimate

- Script 1: 20-30 min
- Script 2: 20-30 min  
- Script 3: 30 min
- Review & submit: 10 min

**Total: ~1.5 hours for complete Option B pipeline**

---

## 📞 Support

If something breaks:
1. Check error message carefully
2. Verify file paths (especially `/kaggle/working/amazon-ml-chal/`)
3. Check data shapes match expectations
4. Try running in interactive mode to debug

Good luck! 🚀
