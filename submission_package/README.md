# Amazon ML Challenge - Solution Summary

**Team:** SkyNet Corp  
**Members:** Omm Prakash Tripathy, Paresh Chandra Pothal, Prem Chandra Prasad, Archit Mishra

**Goal:** Predict product prices, target 30-40% SMAPE

**Final Result:** 27.86% SMAPE ✓

---

## What Worked

**Features (42 total):**
- IPQ extraction: value, unit, pack_size from catalog text
- Quality keywords: premium, organic, gourmet flags (6 features)
- Brand parsing from item names
- Interaction terms: value×quality, pack×quality (9 features)
- **Text embeddings:** CLIP text → PCA 20 components (this is 81% of model!)

**Model:** LightGBM, SQRT transform, 2000 rounds

**Why it works:** Text PCA captures semantic pricing signals (luxury language, brand reputation, quality tier vocabulary) that complement the structured IPQ features.

---

## What Didn't Work

- **Image embeddings:** R²=-0.01, totally useless (CLIP not trained for product pricing)
- **LOG transform:** Hurt luxury segment (46% → 52% SMAPE)
- **Text length features:** Added noise, removed them

---

## Performance by Segment

- Mid-range ($10-50): 19% SMAPE - excellent
- Budget (<$10): 37% SMAPE - okay
- Premium ($50-100): 32% SMAPE - good  
- Luxury (>$100): 46% SMAPE - weak (only 2.5% of data)

---

## Files

- `KAGGLE_submission_enhanced.py` - main script
- `baseline_enhanced.py` - validation run
- `submission_enhanced.csv` - predictions

**Run:** `python KAGGLE_submission_enhanced.py` (~30 min)

---

**Expected LB:** 28-31% SMAPE
