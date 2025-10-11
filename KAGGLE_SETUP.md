# 🚀 Kaggle Setup Instructions

## Phase 1 Complete ✅
All validation work is done. Now ready for Phase 2-5 on Kaggle.

---

## 📋 Step-by-Step Kaggle Setup

### 1. **Clone Repository**
```bash
git clone https://github.com/observer04/amazon-ml-chal.git
cd amazon-ml-chal
```

### 2. **Generate Feature Files** (10-15 min)
```bash
# This creates train_with_features.csv and test_with_features.csv
python generate_features.py
```

**Expected Output:**
- `dataset/train_with_features.csv` (~146 MB, 75k rows, 30+ features)
- `dataset/test_with_features.csv` (~144 MB, 75k rows, 30+ features)

### 3. **Create Holdout Split** (2 min)
```bash
# This creates 80/20 train/holdout split
python validation/phase1_complete.py
```

**Expected Output:**
- `dataset/train_80pct.csv` (~113 MB, 60k rows)
- `dataset/holdout_20pct.csv` (~29 MB, 15k rows)

### 4. **Download Images** ⏳ (1.5-2 hours)
```bash
# This downloads 150k images from Amazon CDN
python download_images.py
```

**Expected Output:**
- `images/train/` (~7-10 GB, 75k images)
- `images/test/` (~7-10 GB, 75k images)
- Success rate: >90%
- Failed downloads logged to `failed_downloads.txt`

**Resource Requirements:**
- Disk space: ~20 GB total
- Bandwidth: Good internet required
- Time: 1.5-2 hours on P100 or 2xT4

---

## 📊 Verification Commands

After each step, verify success:

```bash
# Check feature files
ls -lh dataset/train_with_features.csv dataset/test_with_features.csv

# Check splits
ls -lh dataset/train_80pct.csv dataset/holdout_20pct.csv

# Check images
ls images/train/ | wc -l  # Should show ~75000
ls images/test/ | wc -l   # Should show ~75000
```

---

## 🎯 Phase Summary

**Phase 1 Complete (Done locally):**
- ✅ Baseline: 71% SMAPE
- ✅ Adversarial validation: AUC 0.50
- ✅ Feature stability verified
- ✅ All validation scripts ready

**Phase 2 (Run on Kaggle):**
- Generate features with `generate_features.py`
- Create splits with `validation/phase1_complete.py`
- Download images with `download_images.py`

**Phase 3-5 (Coming next):**
- Generate text embeddings (sentence-transformers)
- Generate image embeddings (ResNet18)
- Train multi-modal ensemble
- Submit predictions

---

## 📁 Important Files

| File | Purpose | Size |
|------|---------|------|
| `generate_features.py` | Extract 30 features from catalog_content | - |
| `download_images.py` | Download 150k images (100 workers) | - |
| `validation/phase1_complete.py` | Create 80/20 split | - |
| `PROGRESS.md` | Full project status | - |
| `todo.md` | Strategic decision tree (4 options) | - |
| `ARCHITECTURAL_ANALYSIS.md` | Deep dive into approach | - |

---

## ⚠️ Known Issues

1. **Large CSV files not in git**: Regenerate with `generate_features.py`
2. **Images take 1.5-2 hours**: Normal, be patient
3. **Some images may fail**: Expected ~5% failure rate (dead links)

---

## 🔗 Current Status

- **Current LB**: 62.45% SMAPE
- **Target**: 30-40% SMAPE
- **Baseline**: 71% SMAPE (you beat this!)
- **Train/Test Match**: AUC 0.50 (perfect match)

**Path to Target:**
- 62% → 52-57% (add text embeddings)
- 52-57% → 32-42% (add image embeddings)
- 32-42% → 30-35% (optimize ensemble)

---

## 💡 Tips

1. Use Kaggle's **P100** or **2xT4** GPUs
2. Enable **internet** for image download
3. Monitor disk space during image download
4. Save intermediate files to avoid re-runs

---

## 🆘 Need Help?

Check these files:
- `PROGRESS.md` - Overall project status
- `todo.md` - Strategic options A-D
- `validation/baseline_check.py` - Baseline implementation
- `validation/adversarial_validation.py` - Distribution check

---

**Last Updated**: Phase 1 Complete
**Next**: Run setup steps 1-4 on Kaggle
