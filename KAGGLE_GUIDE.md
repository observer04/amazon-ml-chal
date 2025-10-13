# KAGGLE EXECUTION GUIDE

## Phase 1: Extract Marqo Embeddings (THIS FIRST)

### Setup Kaggle Notebook:
1. Create new notebook at kaggle.com/code
2. Enable GPU (T4 or better)
3. Add dataset: `amazon-ml-challenge-2024`
4. Set Internet ON

### Single Command:
```bash
wget https://raw.githubusercontent.com/observer04/amazon-ml-chal/main/kaggle_extract_marqo_embeddings.py && python kaggle_extract_marqo_embeddings.py
```

### Expected Output:
- `train_marqo_embeddings.npy` (~300MB)
- `test_marqo_embeddings.npy` (~300MB)  
- `KAGGLE_RUN_RESULTS.md` (detailed analysis)

### After Execution:
1. Download `KAGGLE_RUN_RESULTS.md` from Kaggle
2. Commit and push to GitHub:
   ```bash
   git add KAGGLE_RUN_RESULTS.md
   git commit -m "Results: Marqo embedding extraction"
   git push origin main
   ```

---

## Phase 2: Train Model with Marqo Features (AFTER PHASE 1)

**Status:** Script not yet created (waiting for Phase 1 results)

Will create after seeing correlation results from Phase 1.

---

## Phase 3: Generate Submission (AFTER PHASE 2)

**Status:** Script not yet created (waiting for Phase 2 validation)

Will create after confirming model improvement.

---

## Quick Reference

### Check what scripts are ready:
- ✅ `kaggle_extract_marqo_embeddings.py` - Extract image embeddings
- ⏳ `kaggle_train_with_marqo.py` - Train model (pending Phase 1)
- ⏳ `kaggle_create_submission.py` - Generate predictions (pending Phase 2)

### Current workflow:
```
Phase 1 (Kaggle) → Push results → Pull results (Local) → 
Phase 2 (Kaggle) → Push results → Pull results (Local) →
Phase 3 (Kaggle) → Submit to leaderboard
```

### Files to sync:
- **From Kaggle → GitHub:** `KAGGLE_RUN_RESULTS.md` after each phase
- **From GitHub → Kaggle:** `.py` scripts before execution
