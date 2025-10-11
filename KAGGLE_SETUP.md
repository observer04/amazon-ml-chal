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

### 4. **Generate Image Embeddings** ⏳ (1-2 hours) - REVISED APPROACH
```bash
# This streams images and extracts embeddings (no disk storage!)
# Only saves compact 512-dim embeddings (~300MB total)
python generate_image_embeddings.py
```

**Expected Output:**
- `outputs/train_image_embeddings.npy` (~150 MB, 75k x 512 dims)
- `outputs/test_image_embeddings.npy` (~150 MB, 75k x 512 dims)
- Success rate: >90%
- No raw images saved (solved disk space issue!)

**Why This Approach:**
- ✅ Downloads images → extracts embeddings → discards images (all in memory)
- ✅ 300MB output vs 40GB (133x smaller!)
- ✅ Fits within Kaggle's 20GB output limit
- ✅ Uses ResNet18 pretrained on ImageNet
- ✅ Automatically cleans up any existing downloaded images

**Resource Requirements:**
- GPU: P100 or 2xT4 (required)
- Time: 1-2 hours
- Internet: Required for streaming downloads

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
- Generate image embeddings with `generate_image_embeddings.py` (streaming, no disk storage!)

**Phase 3-5 (Coming next):**
- Generate text embeddings (sentence-transformers)
- Train multi-modal ensemble
- Submit predictions

---

## 📁 Important Files

| File | Purpose | Size |
|------|---------|------|
| `generate_features.py` | Extract 30 features from catalog_content | - |
| `generate_image_embeddings.py` | Stream images + extract ResNet18 embeddings | - |
| `generate_text_embeddings.py` | Extract sentence-transformer embeddings | - |
| `download_images.py` | ~~Download 150k images~~ (DEPRECATED - use streaming) | - |
| `validation/phase1_complete.py` | Create 80/20 split | - |
| `PROGRESS.md` | Full project status | - |
| `todo.md` | Strategic decision tree (4 options) | - |
| `ARCHITECTURAL_ANALYSIS.md` | Deep dive into approach | - |

---

## ⚠️ Known Issues

1. **Large CSV files not in git**: Regenerate with `generate_features.py`
2. **Disk space on Kaggle**: Solved! Use streaming approach with `generate_image_embeddings.py`
3. **Old download_images.py hits 20GB limit**: Use new `generate_image_embeddings.py` instead
4. **Some images may fail**: Expected ~5% failure rate (dead links, handled with zero vectors)

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
